import numpy as np
import torch
from dpc.DPC_solver import DPC_Solver, wrap_angle
from dpc.discretizers import rk4_discretization_torch
from policy import NeuralDistributionPolicy
from dpc.constraints import *
from .constraints import *

class DMPPI_Solver(DPC_Solver):
    """
    DPC-guided MPPI (1-step sampling):
    - Mean control from trained DPC policy
    - Fixed Gaussian covariance
    - One-step rollout for K samples
    - MPPI weighted update on first action
    """

    def __init__(self,
        config,
        model,
        policy: NeuralDistributionPolicy,
        discretizer=rk4_discretization_torch,
        updater=None,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            policy=policy,
            discretizer=discretizer,
            device=device,
        )

        # this is the MPPI updater (None is vanilla importance sampling update)
        self.updater = updater 
        if self.updater is not None and hasattr(self.updater, "to"):
            self.updater = self.updater.to(device)

        # Sampling params (fallbacks if not in config)
        self.n_samples = int(getattr(self.config, "n_samples", 1024))
        self.m_samples = int(getattr(self.config, "m_samples", 256))

        self.temperature = float(getattr(self.config, "temperature", 1.0))
        self.u_std = getattr(self.config, "u_std", 0.5)  # scalar or (nu,)

        # Fixed covariance diag(u_std^2)
        u_std_t = torch.as_tensor(self.u_std, dtype=torch.float32, device=self.device)
        if u_std_t.ndim == 0:
            self.u_cov = (u_std_t ** 2) * torch.eye(self.config.nu, device=self.device)
        else:
            self.u_cov = torch.diag(u_std_t ** 2)
        
        params = self.model.params
        N = self.config.N

        # Build state constraints from vehicle parameters
        # x = [x, y, delta, v, yaw, yaw_rate, beta]
        x_min_constrained = np.array([
            -np.inf,           # x position: unconstrained
            -np.inf,           # y position: unconstrained  
            params.MIN_STEER,  # steering angle
            params.MIN_SPEED,  # velocity
            -np.inf,           # yaw: unconstrained
            -np.inf,           # yaw_rate: unconstrained
            -np.inf,           # beta: unconstrained
        ])
        x_max_constrained = np.array([
            np.inf,            # x position: unconstrained
            np.inf,            # y position: unconstrained
            params.MAX_STEER,  # steering angle
            params.MAX_SPEED,  # for testing, set max speed to 5 m/s
            np.inf,            # yaw: unconstrained
            np.inf,            # yaw_rate: unconstrained
            np.inf,            # beta: unconstrained
        ])
        
        self.A_st_limit, self.b_st_limit = st_limit_constraint_coeffs_single_step(
            x_min_constrained, x_max_constrained) 

    def load_trained_policy(self, file_path: str, strict: bool = True):
        """
        Load trained policy weights from .pt file and updater weights when present.
        Supports:
        1) full checkpoint dict with 'policy_state_dict'
        2) raw state_dict directly
        """
        model = torch.load(file_path, map_location=self.device, weights_only=True)

        if isinstance(model, dict) and "policy_state_dict" in model:
            policy_state_dict = model["policy_state_dict"]
            updater_state_dict = model.get("updater_state_dict")
        else:
            policy_state_dict = model  # assume raw state_dict
            updater_state_dict = None

        self.policy.load_state_dict(policy_state_dict, strict=strict)
        self.policy.to(self.device)
        print("Loaded trained policy from", file_path)

        if (
            self.updater is not None
            and updater_state_dict is not None
            and hasattr(self.updater, "load_state_dict")
        ):
            self.updater.load_state_dict(updater_state_dict, strict=strict)
            self.updater.to(self.device)
            print("Loaded trained update model from", file_path)

    def single_step_stage_cost(self,
        xk1: torch.Tensor, # (B, nx)
        uk: torch.Tensor, # (B, nu)
        rk: torch.Tensor, # (B, 3) [x_ref, y_ref, yaw_ref]
    ) -> torch.Tensor:
        """
        Single-step tracking stage cost.
        Equation: l_stage = (e_x)^T Q e_x + (u_k)^T R u_k
        Returns: l_stage: (B,)
        """
        Qx = torch.as_tensor(self.config.Q, dtype=xk1.dtype, device=xk1.device)
        Ru = torch.as_tensor(self.config.R, dtype=xk1.dtype, device=xk1.device)

        x_ref = torch.zeros_like(xk1)
        x_ref[..., 0] = rk[..., 0]
        x_ref[..., 1] = rk[..., 1]
        x_ref[..., 4] = rk[..., 2]

        # Avoid in-place writes on autograd-tracked tensors.
        e_raw = xk1 - x_ref
        e_yaw = wrap_angle(e_raw[..., 4:5])
        e = torch.cat([e_raw[..., :4], e_yaw, e_raw[..., 5:]], dim=-1)

        cx = torch.einsum("bi,ij,bj->b", e, Qx, e)
        cu = torch.einsum("bi,ij,bj->b", uk, Ru, uk)
        return cx + cu

    def single_step_constraint_penalty(
        self,
        xk1: torch.Tensor, # (B, nx)
        uk: torch.Tensor, # (B, nu)
        ck: torch.Tensor, # (B, 4) [a, b, c_l, c_r]
    ) -> torch.Tensor:
        """
        Single-step soft constraint penalty
        xk1: (B,nx), uk: (B,nu), ck: (B,4) [a,b,c_left,c_right]
        """
        device = self.device
        dtype = xk1.dtype
        lambdas = torch.as_tensor(self.config.lambdas, dtype=xk1.dtype, device=xk1.device)
        # assume lambdas[0]=state_limit, lambdas[1]=boundary
        lam_state = lambdas[0]
        lam_bdry = lambdas[1]

        v_state = st_limit_constraint_single_step(
            xk1, self.A_st_limit, self.b_st_limit, state_indices=(2, 3)
        )  # (B,)

        A_b, b_b = boundary_constraint_coeffs_single_step(ck)
        v_bdry = boundary_constraint_single_step(xk1, A_b, b_b)  # (B,)

        v_state = torch.clamp(v_state, min=0.0, max=3.0)
        v_bdry = torch.clamp(v_bdry, min=0.0, max=3.0)

        pen = lam_state * (v_state ** 2) + lam_bdry * (v_bdry ** 2)  # (B,)
        return pen

    def _mppi_weighted_update(
        self, sampled_controls: torch.Tensor, costs: torch.Tensor, temperature: float,
    ) -> torch.Tensor:
        c_min = costs.min(dim=1, keepdim=True).values
        w = torch.softmax(-(costs - c_min) / temperature, dim=1)  # (B, K)
        return torch.sum(w.unsqueeze(-1) * sampled_controls, dim=1)  # (B, nu)

    def _select_costs_for_updater(self, costs: torch.Tensor) -> torch.Tensor:
        """
        Adapt rollout costs (B, K_rollout) to updater expected shape (B, K_updater).
        If updater K is None, return full costs.
        Sampling:
          - without replacement when K_updater <= K_rollout
          - with replacement when K_updater > K_rollout
        """
        if self.updater is None:
            return costs
        target_k = getattr(self.updater, "K", None)
        if target_k is None:
            return costs

        B, K_rollout = costs.shape
        K_updater = int(target_k)
        if K_updater <= 0:
            raise ValueError(f"Updater K must be positive, got {K_updater}")

        if K_updater <= K_rollout:
            perm = torch.rand(B, K_rollout, device=costs.device).argsort(dim=1)
            idx = perm[:, :K_updater]
        else:
            idx = torch.randint(
                low=0,
                high=K_rollout,
                size=(B, K_updater),
                device=costs.device,
            )
        return torch.gather(costs, dim=1, index=idx)

    def _updater_supervised_loss(
        self, u_mean: torch.Tensor, u_cov: torch.Tensor, costs: torch.Tensor, u_star_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample random subsets of size M from the K rollout costs, predict a
        control for each subset, and regress those predictions to the full MPPI
        control target.
        - If n_candidates * M <= K, sample without replacement.
        - If n_candidates * M > K, sample with replacement.
        """
        if self.updater is None:
            return costs.new_zeros(())

        B, K = costs.shape
        M = max(1, min(self.m_samples, K))
        default_candidates = max(1, K // M)
        n_candidates = int(getattr(self.config, "updater_n_candidates", default_candidates))
        n_candidates = max(1, n_candidates)
        n_used = n_candidates * M

        if n_used <= K:
            perm = torch.rand(B, K, device=costs.device).argsort(dim=1)
            subset_idx = perm[:, :n_used].reshape(B, n_candidates, M)
        else:
            subset_idx = torch.randint(
                low=0,
                high=K,
                size=(B, n_candidates, M),
                device=costs.device,
            )
        subset_costs = torch.gather(
            costs,
            dim=1,
            index=subset_idx.reshape(B, n_used),
        ).reshape(B, n_candidates, M)

        u_mean_rep = u_mean[:, None, :].expand(B, n_candidates, -1)
        if u_cov.ndim == 2:
            u_cov_rep = u_cov[:, None, :].expand(B, n_candidates, -1)
        elif u_cov.ndim == 3:
            u_cov_rep = u_cov[:, None, :, :].expand(B, n_candidates, -1, -1)
        else:
            raise ValueError(f"u_cov shape not supported: {tuple(u_cov.shape)}")

        u_pred = self.updater(
            u_mean_rep.reshape(B * n_candidates, *u_mean.shape[1:]),
            u_cov_rep.reshape(B * n_candidates, *u_cov.shape[1:]),
            subset_costs.reshape(B * n_candidates, M),
        ).reshape(B, n_candidates, -1)

        u_target = u_star_true[:, None, :].expand_as(u_pred)
        return torch.mean((u_pred - u_target) ** 2)

    def solve(self, x0, ref_traj, con_coeffs):
        """
        Single-step MPPI update for inference/testing.
        Returns numpy arrays:
          xk: (nx, 1), uk: (nu, 1)
        """
        dt = self.config.dt
        K = self.m_samples
        temp = max(1e-6, self.temperature)

        # Construct tensors for current state and pose reference
        x0_t = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        if x0_t.ndim == 1:
            x0_t = x0_t.unsqueeze(0)
        x0_t = x0_t.clone()
        x0_t[..., 4] = wrap_angle(x0_t[..., 4])

        params_full = self._build_rollout_params(ref_traj, con_coeffs, batch_size=x0_t.shape[0])

        if self.fast_inference:
            rk1 = params_full[:, 1, :]  # (B, 3)
            with torch.no_grad():
                u_mean, u_std = self.policy(self.build_policy_input(x0_t, rk1))
            B, nu = u_mean.shape
            np_dim = rk1.shape[-1]
            # print("Sample from mean and std: ", u_mean, u_std)

            # Expand batch for K samples
            x0_s = x0_t[:, None, :].expand(B, K, x0_t.shape[-1]).reshape(B * K, -1)
            rk1_s = rk1[:, None, :].expand(B, K, np_dim).reshape(B * K, np_dim)

            # Sample controls from policy distribution
            eps = torch.randn(B, K, nu, device=self.device, dtype=u_mean.dtype)
            if u_std.ndim == 2:
                # diag mode: u_std is std (B, nu)
                u_std_s = u_std[:, None, :].expand(B, K, nu)
                uk = u_mean[:, None, :] + u_std_s * eps  # (B, K, nu)
            elif u_std.ndim == 3:
                # full mode: u_std is Cholesky L (B, nu, nu)
                L = u_std[:, None, :, :].expand(B, K, nu, nu)
                uk = u_mean[:, None, :] + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1)  # (B, K, nu)
            else:
                raise ValueError(f"Unsupported policy distribution output shape: {tuple(u_std.shape)}")

            uk_s = uk.reshape(B * K, nu)

            # Clamp controls to bounds
            u_min = torch.as_tensor(self.config.u_min, dtype=uk_s.dtype, device=self.device).view(1, -1)
            u_max = torch.as_tensor(self.config.u_max, dtype=uk_s.dtype, device=self.device).view(1, -1)
            uk_s = torch.clamp(uk_s, min=u_min, max=u_max)

            # One-step rollout for all samples
            x1_s = self.step(self.model.f_torch, x0_s, uk_s, self.p, dt)  # (B*K, nx)

            # Sample costs
            c_stage = self.single_step_stage_cost(x1_s, uk_s, rk1_s)  # (B*K,)
            ck_s = params_full[:, 1, 3:][:, None, :].expand(B, K, -1).reshape(B * K, -1)
            c_pen = self.single_step_constraint_penalty(x1_s, uk_s, ck_s)   # (B*K,)
            c_total = (c_stage + c_pen).reshape(B, K)                 # (B,K)

            uk_s_reshaped = uk_s.reshape(B, K, nu)
            if self.updater is None:
                u_star = self._mppi_weighted_update(uk_s_reshaped, c_total, temp)
            else:
                u_star = self.updater(u_mean, u_std, c_total)

            x1_star = self.step(self.model.f_torch, x0_t, u_star, self.p, dt)  # (B,nx)

            # Return first batch item in MPC-compatible 2D arrays
            self.uk = u_star[0].unsqueeze(-1).detach().cpu().numpy()
            self.xk = x1_star[0].unsqueeze(-1).detach().cpu().numpy()
            return self.xk, self.uk
        else: 
            raise NotImplementedError(
                f"Full-horizon rollout inference has not been implemented yet"
            )

    def rollout(self, x0: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        x0:      (B, 7)
        ref_traj: (B, N+1, 7) = [x_ref, y_ref, yaw_ref, a, b, c_l, c_r] for each step in horizon
        params:  passed to model.f_torch 
        returns:
            X: (B, N+1, 7)
            U: (B, N, 2)
            L: (B, N, nu, nu) in full mode, else None
            updater_loss: scalar supervision loss for the updater
        """
        N = self.config.N # control horizon length (number of rollout steps)
        dt = self.config.dt # Discrete integration time step [s]
        K = self.n_samples

        temp = max(1e-6, self.temperature)
        nu = self.config.nu
        x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        x0 = x0.clone()
        x0[..., 4] = wrap_angle(x0[..., 4])
        B, nx = x0.shape
        params = torch.as_tensor(params, dtype=torch.float32, device=self.device)
        params = params.clone()
        params[..., 2] = wrap_angle(params[..., 2])

        X_list = [x0] # State trajectory buffer, initialized with initial state x0
        U_list = [] # Control trajectory buffer, one control per step
        L_list = []
        updater_loss = x0.new_zeros(())

        #  Rollout over the control horizon using the policy and dynamics model
        xk = x0
        for k in range(N):
            rk = params[:, k+1, :] 
            np = rk.shape[1]
            inp = self.build_policy_input(xk, rk)
            # Policy distribution at current state/reference
            u_mean, u_std = self.policy(inp) # (B,nu), (B,nu) or (B,nu,nu)
            # Expand for K sampled controls
            xk_s = xk[:, None, :].expand(B, K, nx).reshape(B * K, nx)
            rk_s = rk[:, None, :].expand(B, K, np).reshape(B * K, np)

            eps = torch.randn(B, K, nu, device=self.device, dtype=u_mean.dtype)
            if self.policy.cov_mode == "diag":
                # diag mode: u_std is std
                u_std_s = u_std[:, None, :].expand(B, K, nu)
                uk = u_mean[:, None, :] + u_std_s * eps  # (B,K,nu)
                L_list.append(torch.diag_embed(u_std))
            elif self.policy.cov_mode == "full":
                # full mode: u_std is Cholesky factor L
                L = u_std[:, None, :, :].expand(B, K, nu, nu)
                uk = u_mean[:, None, :] + torch.matmul(L, eps.unsqueeze(-1)).squeeze(-1)  # (B,K,nu)
                L_list.append(u_std)

            uk_s = uk.reshape(B * K, nu)
            # Clamp sampled controls
            u_min = torch.as_tensor(self.config.u_min, dtype=uk_s.dtype, device=self.device).view(1, -1)
            u_max = torch.as_tensor(self.config.u_max, dtype=uk_s.dtype, device=self.device).view(1, -1)
            uk_s = torch.clamp(uk_s, min=u_min, max=u_max)
            # One-step sampled dynamics
            xk1_s = self.step(self.model.f_torch, xk_s, uk_s, self.p, dt) # (B*K,nx)

            # Sample costs at this step
            c_stage = self.single_step_stage_cost(xk1_s, uk_s, rk_s[:,:3]) # (B*K,)
            c_pen = self.single_step_constraint_penalty(xk1_s, uk_s, rk_s[:,3:]) # (B*K,)
            c_total = (c_stage + c_pen).reshape(B, K) # (B,K)

            uk_s_reshaped = uk_s.reshape(B, K, nu)
            u_star_true = self._mppi_weighted_update(uk_s_reshaped, c_total, temp)

            if self.updater is None:
                u_star = u_star_true
            else:
                updater_loss = updater_loss + self._updater_supervised_loss(
                    u_mean, u_std, c_total, u_star_true
                )
                # Roll out the student updater control while supervising to teacher MPPI.
                c_updater = self._select_costs_for_updater(c_total)
                u_star = self.updater(u_mean, u_std, c_updater)

            # Propagate with selected control
            u_star = u_star_true
            xk1 = self.step(self.model.f_torch, xk, u_star, self.p, dt)   # (B,nx)

            U_list.append(u_star)
            X_list.append(xk1)
            xk = xk1

        X = torch.stack(X_list, dim=1)
        U = torch.stack(U_list, dim=1)
        L = torch.stack(L_list, dim=1)
        updater_loss = updater_loss / max(N, 1)
        return X, U, L, updater_loss
