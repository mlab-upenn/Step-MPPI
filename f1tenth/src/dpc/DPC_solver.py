from typing import Dict
import numpy as np 
import torch
from f1tenth_planning.control.config.controller_config import MPCConfig
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.mpc_solver import MPCSolver
from .discretizers import rk4_discretization_torch
from policy import NeuralControlPolicy

@staticmethod
def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """
    Wrap angle(s) to (-pi, pi] in a differentiable way.
    Works with any shape (B,), (B,N), (B,N,...) etc.
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))

class DPC_Solver(MPCSolver):

    # State: x = [x, y, delta, v, yaw, yaw_rate, beta]
    # Control: u = [delta_v, a]

    def __init__(self,
            config: MPCConfig,
            model: DynamicsModel,
            policy: NeuralControlPolicy, # u = pi(theta)(policy_input)
            discretizer = rk4_discretization_torch,
            device = torch.device("cpu"),
            fast_inference: bool = True
            ) -> None:    

        super().__init__(config, model)
        self.device = device
        self.step = discretizer
        self.policy = policy.to(device)
        # the vehicle parameters
        self.p = self.model.parameters_vector_from_config(self.model.params)
        self.p = torch.as_tensor(self.p, dtype=torch.float32, device=device)
        self.fast_inference = fast_inference
    
    def load_trained_policy(self, file_path: str, strict: bool = True):
        """
        Load trained policy weights from .pt file.
        Supports:
        1) full checkpoint dict with 'policy_state_dict'
        2) raw state_dict directly
        """
        model = torch.load(file_path, map_location=self.device, weights_only=True)

        if isinstance(model, dict) and "policy_state_dict" in model:
            state_dict = model["policy_state_dict"]
        else:
            state_dict = model  # assume raw state_dict

        self.policy.load_state_dict(state_dict, strict=strict)
        self.policy.to(self.device)

    def build_policy_input(self, xk: torch.Tensor, rk: torch.Tensor) -> torch.Tensor:
        """
        No preview: concatenate current state and current reference.
        Args:
            xk: (B, 7)
            rk: (B, 7) = [x_ref, y_ref, yaw_ref, a, b, c_left, c_right]
        Returns:
            inp: (B, 14)
        """
        return torch.cat([xk, rk], dim=-1)

    def rollout(self, x0: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        x0:      (B, 7)
        ref_traj: (B, N+1, 7) = [x_ref, y_ref, yaw_ref, a, b, c_l, c_r] for each step in horizon
        params:  passed to model.f_torch 
        returns:
            X: (B, N+1, 7)
            U: (B, N, 2)
        """
        N = self.config.N # control horizon length (number of rollout steps)
        dt = self.config.dt # Discrete integration time step [s]

        X_list = [x0] # State trajectory buffer, initialized with initial state x0
        U_list = [] # Control trajectory buffer, one control per step

        #  Rollout over the control horizon using the policy and dynamics model
        xk = x0
        for k in range(N):
            rk = params[:, k+1, :] 
            inp = self.build_policy_input(xk, rk)
            uk = self.policy(inp) # (B,2) = [delta_v, a]
            xk1 = self.step(self.model.f_torch, xk, uk, self.p, dt)
            X_list.append(xk1)
            U_list.append(uk)
            xk = xk1

        X = torch.stack(X_list, dim=1)
        U = torch.stack(U_list, dim=1)
        return X, U   

    def _build_rollout_params(self, ref_traj, con_coeffs, batch_size: int) -> torch.Tensor:
        """
        Build rollout reference tensor with both pose and boundary coefficients.
        Returns:
            params_full: (B, N+1, 7) = [x_ref, y_ref, yaw_ref, a, b, c_left, c_right]
        """
        N = self.config.N

        r = torch.as_tensor(ref_traj, dtype=torch.float32, device=self.device)
        if r.ndim == 2:
            if r.shape[1] == N + 1 and r.shape[0] >= 5:
                pose_ref = torch.stack([r[0, :], r[1, :], r[4, :]], dim=-1).unsqueeze(0)
            elif r.shape[0] == N + 1 and r.shape[1] >= 3:
                pose_ref = r[:, :3].unsqueeze(0)
            else:
                raise ValueError(f"Unsupported ref_traj shape {tuple(r.shape)}")
        elif r.ndim == 3:
            if r.shape[1] != N + 1:
                raise ValueError(f"Expected time dim N+1={N+1}, got {tuple(r.shape)}")
            if r.shape[-1] >= 5:
                pose_ref = torch.stack([r[..., 0], r[..., 1], r[..., 4]], dim=-1)
            elif r.shape[-1] >= 3:
                pose_ref = r[..., :3]
            else:
                raise ValueError(f"Unsupported ref_traj shape {tuple(r.shape)}")
        else:
            raise ValueError(f"Unsupported ref_traj shape {tuple(r.shape)}")

        pose_ref = pose_ref.clone()
        pose_ref[..., 2] = wrap_angle(pose_ref[..., 2])
        if pose_ref.shape[0] == 1 and batch_size > 1:
            pose_ref = pose_ref.repeat(batch_size, 1, 1)

        con = torch.as_tensor(con_coeffs, dtype=torch.float32, device=self.device)
        if con.ndim == 2:
            if con.shape[0] == 4 and con.shape[1] == N + 1:
                con = con.transpose(0, 1).unsqueeze(0)  # (1, N+1, 4)
            elif con.shape[0] == N + 1 and con.shape[1] >= 4:
                con = con[:, :4].unsqueeze(0)  # (1, N+1, 4)
            else:
                raise ValueError(f"Unsupported con_coeffs shape {tuple(con.shape)}")
        elif con.ndim == 3:
            if con.shape[1] == N + 1 and con.shape[-1] >= 4:
                con = con[..., :4]  # (B, N+1, 4)
            else:
                raise ValueError(f"Unsupported con_coeffs shape {tuple(con.shape)}")
        else:
            raise ValueError(f"Unsupported con_coeffs shape {tuple(con.shape)}")

        if con.shape[0] == 1 and batch_size > 1:
            con = con.repeat(batch_size, 1, 1)
        if con.shape[0] != pose_ref.shape[0]:
            raise ValueError(
                f"Batch mismatch between pose_ref {tuple(pose_ref.shape)} and con {tuple(con.shape)}"
            )

        return torch.cat([pose_ref, con], dim=-1)
    
    def solve(self, x0, ref_traj, con_coeffs):
        """
        Compute the neural control policy given the current state and reference trajectory.
        WARNING: Returned arrays are on the GPU, use jax.device_get() to get them on the CPU.
        Args:
            x0 (np.ndarray): initial state of shape (nx,)
            xref (np.ndarray): reference trajectory of shape (nx, N+1)
            one_step (bool): if True, run only one policy+dynamics step for fast inference.
                if False, run full-horizon rollout (default).

        Note that self.rollout() takes ref_traj (N+1, 7) = [x_ref, y_ref, yaw_ref, a, b, c_l, c_r]
        While the default solve() takes (nx, N+1) and constraint coeffs separately.
        Returns:
            if one_step=False:
              - np.ndarray: optimal control input of shape (nu, N)
              - np.ndarray: optimal state trajectory of shape (nx, N+1)
            if one_step=True:
              - np.ndarray: control at current step of shape (nu, 1)
              - np.ndarray: next state of shape (nx, 1)
        """        
        dt = self.config.dt

        # Construct tensors for current state and pose reference
        x0_t = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        if x0_t.ndim == 1:
            x0_t = x0_t.unsqueeze(0)
        x0_t = x0_t.clone()
        x0_t[..., 4] = wrap_angle(x0_t[..., 4])

        params_full = self._build_rollout_params(ref_traj, con_coeffs, batch_size=x0_t.shape[0])

        # Inference
        self.policy.eval()
        with torch.no_grad():
            if self.fast_inference:
                rk1 = params_full[:, 1, :]  # (B, 7)
                inp = self.build_policy_input(x0_t, rk1)
                uk = self.policy(inp)  # (B, nu)
                x1 = self.step(self.model.f_torch, x0_t, uk, self.p, dt)  # (B, nx)

                # MPC-compatible 2D arrays (single-step horizon)
                self.uk = uk[0].unsqueeze(-1).detach().cpu().numpy()  # (nu, 1)
                self.xk = x1[0].unsqueeze(-1).detach().cpu().numpy()  # (nx, 1)
                return self.xk, self.uk

            X, U = self.rollout(x0_t, params_full) # X:(B,N+1,7), U:(B,N,2)

        # Note that B = 1
        self.xk = X[0].transpose(0, 1).detach().cpu().numpy() # (nx, N+1)
        self.uk = U[0].transpose(0, 1).detach().cpu().numpy() # (nu, N)

        return self.xk, self.uk

    def update(self, x0, ref_traj, p=None, Q=None, R=None):
        """
        Update the parameters of the MPPI solver for the next solve iteration.
        Optionally, custom dynamics parameters and cost matrices can be provided.
        Args:
            x0 (np.ndarray): initial state of shape (nx,)
            xref (np.ndarray): reference trajectory of shape (nx, N+1)
            p (np.ndarray, optional): custom dynamics parameters vector. If None, uses default.
            Q (np.ndarray, optional): custom state cost matrix. If None, uses default.
            R (np.ndarray, optional): custom control input cost matrix. If None, uses default
        Returns:
            None
        """
        super().update(x0, ref_traj, p=p, Q=Q, R=R)
        return
