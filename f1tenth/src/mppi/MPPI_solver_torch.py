from __future__ import annotations

from typing import Callable, Iterable

import torch

from f1tenth_planning.control.config.controller_config import APMPPIConfig
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.mpc_solver import MPCSolver

from dpc.discretizers import rk4_discretization_torch


class ConstMPPISolver_torch(MPCSolver):
    """
    PyTorch implementation of constrained AP-MPPI, mirroring src/mppi/MPPI_solver.py.

    State/reference shapes follow the existing solver convention:
      - x0: (nx,)
      - ref_traj: (nx, N+1)
    Returned trajectories:
      - xk: (nx, N+1)
      - uk: (nu, N)
    """

    def __init__(
        self,
        config: APMPPIConfig,
        model: DynamicsModel,
        discretizer: Callable = rk4_discretization_torch,
        step_function: Callable | None = None,
        reward_function: Callable | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        super().__init__(config, model)
        self.config: APMPPIConfig = self.config
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.discretizer = discretizer

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(seed)

        if step_function is not None:
            self._step = step_function
        if reward_function is not None:
            self._reward = reward_function

        self.p = self._as_tensor(self.model.parameters_vector_from_config(self.model.params))
        self.nu_eye = torch.eye(self.config.nu, dtype=self.dtype, device=self.device)
        self.nu_zeros = torch.zeros((self.config.nu,), dtype=self.dtype, device=self.device)

        self.x_min = self._as_tensor(self.config.x_min)
        self.x_max = self._as_tensor(self.config.x_max)
        self.u_min = self._as_tensor(self.config.u_min)
        self.u_max = self._as_tensor(self.config.u_max)

        self.control_params = self._init_control()
        self.lambdas = self._init_lambdas()
        self.constraints_costs = self._init_constraints_costs()
        self.samples = None

    def _as_tensor(self, value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def _init_control(self):
        a_opt = torch.zeros((self.config.N, self.config.nu), dtype=self.dtype, device=self.device)
        a_cov = (self.config.u_std**2) * self.nu_eye.repeat(self.config.N, 1, 1)
        return (a_opt, a_cov)

    def _init_lambdas(self) -> torch.Tensor:
        if self.config.n_constraints == 0 or self.config.n_lambdas == 0:
            return torch.zeros((self.config.n_constraints, self.config.n_lambdas), dtype=self.dtype, device=self.device)

        low = self._as_tensor(self.config.lambdas_sample_range[:, 0])
        high = self._as_tensor(self.config.lambdas_sample_range[:, 1])
        unit = torch.rand(
            (self.config.n_constraints, self.config.n_lambdas),
            generator=self._rng,
            dtype=self.dtype,
            device=self.device,
        )
        return low[:, None] + (high - low)[:, None] * unit

    def _call_constraint_single(self, constraint: Callable, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        try:
            out = constraint(x, u)
        except TypeError:
            out = constraint(x)

        out = torch.as_tensor(out, dtype=self.dtype, device=self.device)
        if out.ndim == 0:
            out = out.repeat(self.config.N)
        return out

    def _init_constraints_costs(self):
        constraints: Iterable[Callable] = tuple(self.config.constraints)
        N = self.config.N
        C = self.config.n_constraints

        def constraints_costs(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            # x: (N, nx) or (S, N, nx), u: (N, nu) or (S, N, nu)
            if len(constraints) == 0:
                if x.ndim == 3:
                    return torch.zeros((x.shape[0], C, N), dtype=self.dtype, device=self.device)
                return torch.zeros((C, N), dtype=self.dtype, device=self.device)

            if x.ndim == 3:
                S = x.shape[0]
                out_per_constraint = []
                for c in constraints:
                    vals = []
                    for s in range(S):
                        vals.append(self._call_constraint_single(c, x[s], u[s]))
                    out_per_constraint.append(torch.stack(vals, dim=0))  # (S, N)
                return torch.stack(out_per_constraint, dim=1)  # (S, C, N)

            out = [self._call_constraint_single(c, x, u) for c in constraints]
            return torch.stack(out, dim=0)  # (C, N)

        return constraints_costs

    def warm_start(self):
        a_opt_prev, _ = self.control_params
        a_opt = torch.cat([a_opt_prev[1:, :], self.nu_zeros[None, :]], dim=0)

        if self.config.adaptive_covariance:
            a_cov = (self.config.u_std**2) * self.nu_eye.repeat(self.config.N, 1, 1)
        else:
            a_cov = self.control_params[1]

        return (a_opt, a_cov)

    def _sample_da(self, a_opt: torch.Tensor) -> torch.Tensor:
        adjusted_lower = self.u_min[None, :, :] - a_opt[None, :, :]
        adjusted_upper = self.u_max[None, :, :] - a_opt[None, :, :]
        da = torch.randn(
            (self.config.n_samples, self.config.N, self.config.nu),
            generator=self._rng,
            dtype=self.dtype,
            device=self.device,
        )
        da = torch.maximum(torch.minimum(da, adjusted_upper), adjusted_lower)
        return da

    def _returns(self, r: torch.Tensor) -> torch.Tensor:
        # Cumulative return from each timestep: (..., N) -> (..., N)
        return torch.flip(torch.cumsum(torch.flip(r, dims=(-1,)), dim=-1), dims=(-1,))

    def _weights(self, returns: torch.Tensor) -> torch.Tensor:
        # returns: (n_samples, N)
        r_max = returns.max(dim=0, keepdim=True).values
        r_min = returns.min(dim=0, keepdim=True).values
        stdzd = (returns - r_max) / ((r_max - r_min) + self.config.damping)
        w = torch.exp(stdzd / self.config.temperature)
        w = w / (w.sum(dim=0, keepdim=True) + 1e-12)
        return w

    def _step(self, x: torch.Tensor, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        next_x = self.discretizer(self.model.f_torch, x, u, p, self.config.dt)
        return torch.clamp(next_x, min=self.x_min, max=self.x_max)

    def _reward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        x_ref: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        dx = x - x_ref
        state_cost = torch.einsum("bi,ij,bj->b", dx, Q, dx)
        control_cost = torch.einsum("bi,ij,bj->b", u, R, u)
        return -(state_cost + control_cost)

    def _rollout(
        self,
        u: torch.Tensor,
        x0: torch.Tensor,
        xref: torch.Tensor,
        p: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
    ):
        # u: (N, nu) or (B, N, nu), x0: (nx,) or (B, nx), xref: (nx, N+1)
        squeeze = u.ndim == 2
        if squeeze:
            u = u.unsqueeze(0)

        B = u.shape[0]
        if x0.ndim == 1:
            x = x0.unsqueeze(0).expand(B, -1).clone()
        else:
            x = x0

        x_states = []
        rewards = []
        for t in range(self.config.N):
            ut = u[:, t, :]
            x = self._step(x, ut, p)
            xref_t = xref[:, t + 1].unsqueeze(0).expand(B, -1)
            r_t = self._reward(x, ut, xref_t, Q, R)
            x_states.append(x)
            rewards.append(r_t)

        s = torch.stack(x_states, dim=1)  # (B, N, nx)
        r = torch.stack(rewards, dim=1)   # (B, N)

        if squeeze:
            return s[0], r[0]
        return s, r

    def iteration_step(
        self,
        input_,
        env_state: torch.Tensor,
        ref_traj: torch.Tensor,
        p: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
    ):
        a_opt, a_cov = input_

        da = self._sample_da(a_opt)
        a = torch.clamp(a_opt[None, :, :] + da, min=self.u_min[None, :, :], max=self.u_max[None, :, :])

        s, r = self._rollout(a, env_state, ref_traj, p, Q, R)  # (S, N, nx), (S, N)

        c = self.constraints_costs(s, a)  # (S, C, N)
        c_weighted = torch.einsum("scn,cl->sln", c, self.lambdas) if self.config.n_constraints > 0 else torch.zeros(
            (self.config.n_samples, self.config.n_lambdas, self.config.N),
            dtype=self.dtype,
            device=self.device,
        )

        r_modified = r[:, None, :] - c_weighted  # (S, L, N)
        R_modified = self._returns(r_modified)    # (S, L, N)

        R_for_weights = R_modified.permute(1, 0, 2)  # (L, S, N)
        w_all = torch.stack([self._weights(R_l) for R_l in R_for_weights], dim=0)  # (L, S, N)

        da_candidates = torch.einsum("lsn,sni->lni", w_all, da)  # (L, N, nu)
        a_candidates = a_opt[None, :, :] + da_candidates

        s_candidates, r_candidates = self._rollout(a_candidates, env_state, ref_traj, p, Q, R)  # (L,N,nx), (L,N)

        if self.config.n_constraints > 0:
            c_candidates = self.constraints_costs(s_candidates, a_candidates)  # (L,C,N)
            violations = torch.clamp_min(c_candidates, 0.0).sum(dim=(1, 2))
        else:
            violations = torch.zeros((self.config.n_lambdas,), dtype=self.dtype, device=self.device)

        pure_returns = r_candidates.sum(dim=1)

        feasible_mask = violations == 0.0
        has_any_feasible = feasible_mask.any()
        feasible_score = torch.where(feasible_mask, pure_returns, torch.full_like(pure_returns, -1e10))
        infeasible_score = -violations
        combined_score = feasible_score if has_any_feasible else infeasible_score

        best_idx = int(torch.argmax(combined_score).item())
        a_opt_new = a_candidates[best_idx]

        if self.config.adaptive_covariance:
            w_best = w_all[best_idx]  # (S, N)
            mu_da = torch.sum(da * w_best[:, :, None], dim=0)
            da_centered = da - mu_da[None, :, :]
            sigma = torch.einsum("sni,snj->nij", da_centered, da_centered * w_best[:, :, None])
            a_cov_new = sigma + 1e-5 * self.nu_eye[None, :, :]
        else:
            a_cov_new = a_cov

        return (a_opt_new, a_cov_new), (a, s, r)

    def update(self, x0, ref_traj, p=None, Q=None, R=None):
        super().update(x0, ref_traj, p=p, Q=Q, R=R)
        return

    def solve(self, x0, ref_traj, vis: bool = True, p=None, Q=None, R=None):
        super().update(x0, ref_traj, p=p, Q=Q, R=R)

        x0_t = self._as_tensor(x0)
        ref_t = self._as_tensor(ref_traj)

        p_t = self._as_tensor(self.p if p is None else p)
        Q_t = self._as_tensor(self.config.Q if Q is None else Q)
        R_t = self._as_tensor(self.config.R if R is None else R)

        a_opt, a_cov = self.control_params
        for _ in range(self.config.n_iterations):
            (a_opt, a_cov), (a_sampled, s_sampled, r_sampled) = self.iteration_step(
                (a_opt, a_cov), x0_t, ref_t, p_t, Q_t, R_t
            )

        self.control_params = (a_opt, a_cov)
        self.samples = (a_sampled, s_sampled, torch.sum(r_sampled, dim=1))

        self.uk = self.control_params[0]  # (N, nu)

        if vis:
            x_pred, _ = self._rollout(self.uk, x0_t, ref_t, p_t, Q_t, R_t)  # (N, nx)
            x_pred = torch.cat([x0_t[None, :], x_pred], dim=0)  # (N+1, nx)
            self.xk = x_pred.transpose(0, 1)  # (nx, N+1)
        else:
            self.xk = torch.zeros((self.config.nx, self.config.N + 1), dtype=self.dtype, device=self.device)

        self.uk = self.uk.transpose(0, 1)  # (nu, N)

        return self.xk.detach().cpu().numpy(), self.uk.detach().cpu().numpy()
