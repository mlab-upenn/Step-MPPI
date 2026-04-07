from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PolicyBounds:
    # bounds for control outputs u = [delta_v, a]
    delta_v_min: float
    delta_v_max: float
    a_min: float
    a_max: float


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
    act_cls,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(act_cls())   # new activation each layer
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class NeuralControlPolicy(nn.Module):
    """
    Simple MLP neural control policy for DPC.

    Inputs:
      - x_k: (B, nx) current state 
      - r_k:  (B, n_r) reference point

    Output:
      - u_k = [delta_v, a] : (B, 2)

    Notes:
      - By default, outputs are squashed to user-provided bounds via tanh scaling.
      - If bounds=None, outputs are unconstrained (raw).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256,
                num_hidden_layers: int = 3, 
                bounds: Optional[PolicyBounds] = None,
                activation: str = "gelu",
                dropout: float = 0.0):
        
        super().__init__()
        self.in_dim = in_dim
        self.bounds = bounds

        act = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }.get(activation.lower(), None)

        if act is None:
            raise ValueError(f"Unsupported activation='{activation}'")

        layers = []
        d = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, 2)) # raw outputs: [delta_v_raw, a_raw]
        self.net = nn.Sequential(*layers)

        # Optional: small init to avoid huge controls at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _squash_to_bounds(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        """
        Maps unconstrained x to [lo, hi] smoothly using tanh.
        """
        mid = 0.5 * (hi + lo)
        half = 0.5 * (hi - lo)
        return mid + half * torch.tanh(x)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        inp: (B, in_dim)
        returns u: (B, 2) = [delta_v, a]
        """
        u_raw = self.net(inp)

        if self.bounds is None:
            return u_raw

        delta_v = self._squash_to_bounds(
            u_raw[..., 0], self.bounds.delta_v_min, self.bounds.delta_v_max
        )
        a = self._squash_to_bounds(
            u_raw[..., 1], self.bounds.a_min, self.bounds.a_max
        )
        return torch.stack([delta_v, a], dim=-1)

class NeuralDistributionPolicy(nn.Module):
    """
    MLP policy that outputs control mean and a distribution parameterization.

    Parameterization:
      - cov_mode="diag": returns diagonal standard deviation vector
      - cov_mode="full": returns Cholesky factor L (cov = L @ L^T)
    """

    def __init__(
        self,
        in_dim: int,
        control_dim: int = 2,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        bounds: Optional[PolicyBounds] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        cov_mode: str = "diag",
        min_std: float = 1e-3,
        max_std: float = 1e1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.control_dim = control_dim
        self.bounds = bounds
        self.cov_mode = cov_mode
        self.min_std = min_std
        self.max_std = max_std

        if cov_mode not in ("diag", "full"):
            raise ValueError(f"Unsupported cov_mode='{cov_mode}', expected 'diag' or 'full'")

        act = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }.get(activation.lower(), None)
        if act is None:
            raise ValueError(f"Unsupported activation='{activation}'")

        layers = []
        d = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        backbone_out_dim = d

        self.mean_head = nn.Linear(backbone_out_dim, control_dim)
        if cov_mode == "diag":
            self.std_head = nn.Linear(backbone_out_dim, control_dim)
        else:
            self.tril_elems = control_dim*(control_dim + 1) // 2
            self.tril_head = nn.Linear(backbone_out_dim, self.tril_elems)

        # Small initialization for stable start.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        if cov_mode == "diag":
            nn.init.zeros_(self.std_head.weight)
            nn.init.zeros_(self.std_head.bias)
        else:
            nn.init.zeros_(self.tril_head.weight)
            nn.init.zeros_(self.tril_head.bias)

    @staticmethod
    def _squash_to_bounds(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        mid = 0.5*(hi + lo)
        half = 0.5*(hi - lo)
        return mid + half*torch.tanh(x)

    def _apply_bounds(self, mean_raw: torch.Tensor) -> torch.Tensor:
        if self.bounds is None:
            return mean_raw
        if self.control_dim != 2:
            raise ValueError("PolicyBounds currently supports control_dim=2 only.")
        delta_v = self._squash_to_bounds(
            mean_raw[..., 0], self.bounds.delta_v_min, self.bounds.delta_v_max
        )
        a = self._squash_to_bounds(
            mean_raw[..., 1], self.bounds.a_min, self.bounds.a_max
        )
        return torch.stack([delta_v, a], dim=-1)

    def forward(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          inp: (B, in_dim)
        Returns:
          mean: (B, control_dim)
          dist_param:
            - cov_mode="diag": std of shape (B, control_dim)
            - cov_mode="full": L of shape (B, control_dim, control_dim)
        """
        h = self.backbone(inp)
        mean_raw = self.mean_head(h)
        mean = self._apply_bounds(mean_raw)

        if self.cov_mode == "diag":
            # std > 0
            std = torch.nn.functional.softplus(self.std_head(h)) + self.min_std
            std = torch.clamp(std, min=self.min_std, max=self.max_std)
            return mean, std
        elif self.cov_mode == "full":
            # Full covariance parameterized by Cholesky factor L.
            raw = self.tril_head(h)  # (B, d*(d+1)//2)
            B = raw.shape[0]
            d = self.control_dim
            L = torch.zeros(B, d, d, dtype=raw.dtype, device=raw.device)
            tril_idx = torch.tril_indices(row=d, col=d, offset=0, device=raw.device)
            L[:, tril_idx[0], tril_idx[1]] = raw

            diag_idx = torch.arange(d, device=raw.device)
            # Keep diagonal positive and bounded so implied std stays stable.
            diag = torch.nn.functional.softplus(L[:, diag_idx, diag_idx]) + self.min_std
            diag = torch.clamp(diag, min=self.min_std, max=self.max_std)
            L[:, diag_idx, diag_idx] = diag
            # Bound lower-triangular off-diagonal entries to avoid covariance blow-up.
            lower_mask = torch.tril(
                torch.ones(d, d, dtype=torch.bool, device=raw.device),
                diagonal=-1,
            )
            L[:, lower_mask] = torch.clamp(
                L[:, lower_mask], min=-self.max_std, max=self.max_std
            )

            return mean, L

class NeuralMPPIUpdate(nn.Module):
    """
    Neural MPPI update: learn MPPI update from
      - mean control: u_mean (B, nu)
      - covariance:   u_cov  (B, nu) [diag std/var] or (B, nu, nu) [full cov/cholesky]
      - sampled costs: costs (B, K)

    Output:
      - u_star (B, nu): learned optimal control
    """

    def __init__(
        self,
        nu: int,
        K: Optional[int] = None,
        hidden_dim: int = 128,
        num_hidden_layers: Sequence[int] = (2, 2, 2), # [global, cost, head]
        bounds: Optional[PolicyBounds] = None,
    ):
        super().__init__()
        self.nu = nu
        self.K = None if K is None else int(K)
        self.bounds = bounds

        if len(num_hidden_layers) != 3:
            raise ValueError("num_hidden_layers must have 3 ints: [global, cost, head]")

        n_global, n_cost, n_head = map(int, num_hidden_layers)
        self.global_mlp = build_mlp(
            in_dim=2*nu+1,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=n_global,
            act_cls=nn.GELU,
        )
        self.cost_mlp = build_mlp(
            in_dim=1,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=n_cost,
            act_cls=nn.GELU,
        )
        self.cost_pool = build_mlp(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=max(n_cost - 1, 0),
            act_cls=nn.GELU,
        )
        self.head = build_mlp(
            in_dim=2*hidden_dim,
            out_dim=nu,
            hidden_dim=hidden_dim,
            num_hidden_layers=n_head,
            act_cls=nn.GELU,
        )

    @staticmethod
    def _squash_to_bounds(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        mid = 0.5*(hi + lo)
        half = 0.5*(hi - lo)
        return mid + half*torch.tanh(x)

    def _apply_bounds(self, u_star_raw: torch.Tensor) -> torch.Tensor:
        if self.bounds is None:
            return u_star_raw
        if self.nu != 2:
            raise ValueError("PolicyBounds currently supports nu=2 only.")
        delta_v = self._squash_to_bounds(
            u_star_raw[..., 0], self.bounds.delta_v_min, self.bounds.delta_v_max
        )
        a = self._squash_to_bounds(
            u_star_raw[..., 1], self.bounds.a_min, self.bounds.a_max
        )
        return torch.stack([delta_v, a], dim=-1)

    def _cov_features(self, u_cov: torch.Tensor) -> torch.Tensor:
        # Returns (B, nu+1): [diag_like, logdet_like]
        if u_cov.ndim == 2:
            # Assume diag input (std or var-like); keep positive for stability
            diag = F.softplus(u_cov)
            logdet_like = torch.log(diag + 1e-8).sum(dim=-1, keepdim=True)
        elif u_cov.ndim == 3:
            # Full matrix input
            diag = torch.diagonal(u_cov, dim1=-2, dim2=-1)
            diag = F.softplus(diag)
            logdet_like = torch.log(diag + 1e-8).sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"u_cov shape not supported: {tuple(u_cov.shape)}")
        return torch.cat([diag, logdet_like], dim=-1)

    def _encode_costs(self, costs: torch.Tensor) -> torch.Tensor:
        # DeepSets-style encoder: phi(cost_i) pooled over i, then rho(...)
        c = (costs - costs.mean(dim=1, keepdim=True)) / (
            costs.std(dim=1, keepdim=True) + 1e-6
        )
        elem_feat = self.cost_mlp(c.unsqueeze(-1))  # (B, K, H)
        pooled_feat = elem_feat.mean(dim=1)  # (B, H), permutation-invariant over K
        return self.cost_pool(pooled_feat)

    def forward(
        self,
        u_mean: torch.Tensor,  # (B, nu)
        u_cov: torch.Tensor,   # (B, nu) or (B, nu, nu)
        costs: torch.Tensor,   # (B, K)
    ):
        B, nu = u_mean.shape
        if nu != self.nu:
            raise ValueError(f"Expected nu={self.nu}, got {nu}")
        if costs.ndim != 2 or costs.shape[0] != B:
            raise ValueError(f"Expected costs shape (B, K), got {tuple(costs.shape)}")
        if self.K is not None and costs.shape[1] != self.K:
            raise ValueError(
                f"Expected costs shape (B, {self.K}), got {tuple(costs.shape)}"
            )

        # Global distribution features
        cov_feat = self._cov_features(u_cov)  # (B, nu+1)
        g = self.global_mlp(torch.cat([u_mean, cov_feat], dim=-1))  # (B, H)

        # Permutation-invariant summary of sampled costs.
        c_feat = self._encode_costs(costs)  # (B, H)

        ctx = torch.cat([g, c_feat], dim=-1)  # (B, 2H)
        u_star = self._apply_bounds(self.head(ctx))
        return u_star
