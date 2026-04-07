
import torch
import torch.nn as nn
import torch.nn.functional as F


# Class to implement the NN
class L2O_Update_Net(nn.Module):
    """
    Mean-only GRU-style gated update network 
    (Sacks & Boots, ICRA 2022 style).

    Maps: (mu_tilde, costs_samples) -> mu_next
      - mu_tilde: (B, H, nu) or (B, H*nu)
      - costs:    (B, M)
      - mu_next:  (B, H, nu)

    Update rule (elementwise):
        g = sigmoid(...)
        h = ...
        mu_next = (1-g) * mu_tilde + g * h
    """

    def __init__(
        self,
        H: int,
        nu: int,
        M: int,
        hidden: int = 1024,
        dropout_p: float = 0.1,
        per_step_cost_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.H = H  # horizon
        self.nu = nu # control dimension 
        self.M = M # number of cost samples
        self.d_mu = H * nu # total mean parameters over horizon
        self.in_dim = self.d_mu + M # concat(mu_flat, costs)
        self.hidden = hidden # hidden dimension of MLP
        self.dropout_p = dropout_p
        self.per_step_cost_norm = per_step_cost_norm
        self.eps = eps

        # Two-layer MLP (ReLU + Dropout)
        self.fc1 = nn.Linear(self.in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(p=dropout_p)

        # Heads: gate g^mu and proposal h^mu
        self.head_g = nn.Linear(hidden, self.d_mu)  # sigmoid later
        self.head_h = nn.Linear(hidden, self.d_mu)  # proposal

        # Stabilize early training: keep updates close to warm start initially
        nn.init.constant_(self.head_g.bias, -2.0)  # sigmoid(-2) ~ 0.12

    def _normalize_costs(self, costs: torch.Tensor) -> torch.Tensor:
        """
        costs: (B, M)
        If per_step_cost_norm=True: normalize per sample (row-wise).
        Otherwise: leave as-is (do dataset-level normalization outside).
        """
        if not self.per_step_cost_norm:
            return costs
        mean = costs.mean(dim=-1, keepdim=True)
        std = costs.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (costs - mean) / std

    def forward(self, mu_tilde: torch.Tensor, costs: torch.Tensor):
        """
        Args:
            mu_tilde: (B, H, nu) OR (B, H*nu)
            costs:    (B, M)
        Returns:
            mu_next: (B, H, nu)
        """
        # Dimension checking
        if mu_tilde.dim() == 3:
            B = mu_tilde.shape[0]
            mu_flat = mu_tilde.reshape(B, -1)  # (B, H*nu)
        elif mu_tilde.dim() == 2:
            B = mu_tilde.shape[0]
            mu_flat = mu_tilde
        else:
            raise ValueError(f"mu_tilde must be (B,H,nu) or (B,H*nu), got {mu_tilde.shape}")

        if costs.dim() != 2 or costs.shape[0] != B or costs.shape[1] != self.M:
            raise ValueError(f"costs must be (B,{self.M}), got {costs.shape}")
        # normalize costs samples
        costs_norm = self._normalize_costs(costs)
        # Build the network input by concatenation
        x = torch.cat([mu_flat, costs_norm], dim=-1)  # (B, d_mu+M)

        # MLP trunk
        z = F.relu(self.fc1(x))
        z = self.drop(z)
        z = F.relu(self.fc2(z))
        z = self.drop(z)

        # Heads
        g = torch.sigmoid(self.head_g(z))  # (B, d_mu) in [0,1]
        h = self.head_h(z)                 # (B, d_mu)

        # Gated update
        mu_next_flat = (1.0 - g) * mu_flat + g * h
        mu_next = mu_next_flat.reshape(B, self.H, self.nu)
        return mu_next