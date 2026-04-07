import numpy as np
import torch
from dpc.DPC_solver import DPC_Solver, wrap_angle
from dpc.constraints import *

class DPC_Trainer:
    """
    Minimal DPC trainer
    """

    def __init__(self, solver:DPC_Solver, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.solver = solver

    def set_constraints(self, constraints):
        self.constraints = constraints
        params = self.solver.model.params
        N = self.solver.config.N

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
        
        # Construct the constant constrain coefficients at start-up
        A_state_con, b_state_con = st_limit_constraint_coeffs_torch(x_min_constrained, x_max_constrained, N+1)
        self.state_constraint_coeffs = [A_state_con, b_state_con]

    def generate_samples(
        self,
        pose_ref: np.ndarray,  # (T, 3) = [x_ref, y_ref, yaw_ref]
        cv_ref: np.ndarray,    # (T,) reference speed profile (can be constant if set upstream)
        track_bdr_coeffs: dict,
        N: int, # horizon length (controls), need window of N+1
        sample_size: int,
        device: torch.device,
    ):
        """
        Samples:
        - R windows by wrapping on the loop
        - x0 around the first ref point, with *uniform* perturbations inside the given ranges

        Stores:
        self.samples = {"x0": (B,7), "R": (B,N+1,7)}
        """
        pos_range = (-2.0, 2.0) # meters (applied to x and y offsets)
        yaw_range = (-np.pi/4, np.pi/4) # rad (yaw)
        v_range = (1.0, 10.0) # m/s (absolute v)
        delta_range = (-np.pi/8, np.pi/8) # rad (absolute delta)
        yawrate_range = (-np.pi/2, np.pi/2) # rad/s (absolute yaw_rate)
        beta_range = (-np.pi/16, np.pi/16) # rad (absolute beta)

        # Data conversion
        pose_ref = torch.from_numpy(pose_ref).to(device, dtype=torch.float32)
        T = pose_ref.shape[0]
        con_coeffs = np.column_stack([
            track_bdr_coeffs["a"],
            track_bdr_coeffs["b"],
            track_bdr_coeffs["c_left"],
            track_bdr_coeffs["c_right"],
        ])
        con_coeffs = torch.from_numpy(con_coeffs).to(device, dtype=torch.float32)

        def _uniform(lo, hi, shape):
            return (hi - lo) * torch.rand(shape, device=device) + lo

        # Reference windows via interpolation consistent with test-time planner.
        # This mirrors calc_interpolated_reference_trajectory logic in vectorized form.
        dl = torch.linalg.norm(
            torch.stack([pose_ref[1, 0] - pose_ref[0, 0], pose_ref[1, 1] - pose_ref[0, 1]])
        )
        starts = torch.randint(0, T, (sample_size,), device=device)  # nearest segment index
        t_vals = torch.zeros(sample_size, N + 1, device=device, dtype=torch.float32)
        t_vals[:, 0] = torch.rand(sample_size, device=device, dtype=torch.float32)  # local interp in [0,1)

        cv_ref_t = torch.from_numpy(cv_ref).to(device, dtype=torch.float32)
        next_idx = (starts + 1) % T
        current_speed = (1.0 - t_vals[:, 0]) * cv_ref_t[starts] + t_vals[:, 0] * cv_ref_t[next_idx]
        dt = float(self.solver.config.dt)
        for i in range(1, N + 1):
            t_vals[:, i] = t_vals[:, i - 1] + (current_speed * dt) / dl
            current_speed = (1.0 - t_vals[:, i]) * cv_ref_t[starts] + t_vals[:, i] * cv_ref_t[next_idx]

        idx = (torch.floor(t_vals).to(torch.long) + starts[:, None]) % T
        t_frac = torch.remainder(t_vals, 1.0).unsqueeze(-1)  # (B, N+1, 1)

        pose_prev = pose_ref[idx]
        pose_next = pose_ref[(idx + 1) % T]
        pose_interp = (1.0 - t_frac) * pose_prev + t_frac * pose_next

        con_prev = con_coeffs[idx]
        con_next = con_coeffs[(idx + 1) % T]
        con_interp = (1.0 - t_frac) * con_prev + t_frac * con_next

        R = torch.cat([pose_interp, con_interp], dim=-1)  # (B, N+1, 7)
        R = R.clone()
        R[..., 2] = wrap_angle(R[..., 2])
        # Initial state x0 around the first ref point
        x0 = torch.zeros(sample_size, 7, device=device, dtype=torch.float32)

        # x, y: reference + uniform offset in pos_range
        dx = _uniform(pos_range[0], pos_range[1], (sample_size,))
        dy = _uniform(pos_range[0], pos_range[1], (sample_size,))
        x0[:, 0] = R[:, 0, 0] + dx
        x0[:, 1] = R[:, 0, 1] + dy

        # yaw: reference + uniform offset in yaw_range, then wrap
        dyaw = _uniform(yaw_range[0], yaw_range[1], (sample_size,))
        x0[:, 4] = torch.atan2(torch.sin(R[:, 0, 2] + dyaw), torch.cos(R[:, 0, 2] + dyaw))

        # Other states: sample absolute values uniformly within ranges
        x0[:, 2] = _uniform(delta_range[0], delta_range[1], (sample_size,))       # delta
        x0[:, 3] = _uniform(v_range[0], v_range[1], (sample_size,))               # v
        x0[:, 5] = _uniform(yawrate_range[0], yawrate_range[1], (sample_size,))   # yaw_rate
        x0[:, 6] = _uniform(beta_range[0], beta_range[1], (sample_size,))         # beta

        self.samples = {"x0": x0, "R": R}

    def create_sample_batch(self, batch_size: int):
        x0_all = self.samples["x0"] # (M, 7)
        R_all = self.samples["R"] # (M, N+1, ...)
        M = x0_all.shape[0]

        idx = torch.randint(0, M, (batch_size,), device=x0_all.device)
        x0_b = x0_all[idx].to(self.device)
        R_b = R_all[idx].to(self.device)
        return x0_b, R_b        

    def increase_lambdas(self, rate: float):
        for i, l in enumerate(self.solver.config.lambdas):
            self.solver.config.lambdas[i] = round(float(l) * float(rate), 1)
        # print("Penalty weight is increased to ", self.solver.config.lambdas)

    # constraints + loss
    def constraint_penalty(self, X: torch.Tensor, U: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Soft constraint penalties like the DPC paper: ReLU^2 of violations.
        Returns per-batch penalty: (B,)
        self.config.lamdas will be a dictionary with keys like "u" and "x" for control and state penalties
        The NeuralControlPolicy already handled the input constraint.
        """
        device = self.device
        constraint_fns = self.constraints
        lambdas = torch.as_tensor(self.penalty_weight, dtype=torch.float32, device=device)

        dtype = X.dtype
        B = X.shape[0]

        if len(constraint_fns) != len(lambdas):
            raise ValueError(
                f"len(config.constraints)={len(constraint_fns)} != len(config.lamdas)={len(lambdas)}"
            )
        
        total_pen = torch.zeros(B, dtype=dtype, device=device)

        for fn, lam in zip(constraint_fns, lambdas):
            if fn is st_limit_constraint_batched:
                # State bound constraints first, which have constant A and b
                violation = fn(X, self.state_constraint_coeffs[0], self.state_constraint_coeffs[1])# (B, T)
            elif fn is boundary_constraint_batched:
                # Safety constraint, which have batched A and b
                safety_constraint_coeffs = boundary_constraint_coeffs_torch(R[:,:,3:])
                violation = fn(X, safety_constraint_coeffs[0], safety_constraint_coeffs[1])
            else: # otherwise, unidentified constraint, will return 0
                violation = torch.zeros(B, dtype=dtype, device=device)
            violation = torch.clamp(violation, min=0.0, max=3.0)  # tune max (e.g. 2~5)
            stage_pen = violation**2
            total_pen += lam * stage_pen.mean(dim=-1)  # mean over time, keep batch dim
        
        return total_pen
    
    def tracking_cost(self, X: torch.Tensor, U: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Tracking + effort.
        Returns per-batch cost: (B,)
        """
        # Config matrices (numpy -> torch)
        Qx = torch.as_tensor(self.solver.config.Q, dtype=X.dtype, device=X.device)
        Ru = torch.as_tensor(self.solver.config.R, dtype=X.dtype, device=X.device)

        # State cost over horizon
        X_REF = torch.zeros(X.shape[0], X.shape[1], 7, device=X.device, dtype=X.dtype)
        X_REF[..., 0] = R[..., 0]   # x_ref
        X_REF[..., 1] = R[..., 1]   # y_ref
        X_REF[..., 4] = R[..., 2]   # yaw_ref

        # wrap_angle for yaw error
        e_X_raw = X - X_REF
        e_yaw = wrap_angle(e_X_raw[..., 4:5])  # keep last dim
        e_X = torch.cat([e_X_raw[..., :4], e_yaw, e_X_raw[..., 5:]], dim=-1)

        state_stage = torch.einsum("bni,ij,bnj->bn", e_X, Qx, e_X)
        # Control cost over horizon
        control_stage = torch.einsum("bni,ij,bnj->bn", U, Ru, U)

        cost = state_stage.mean(dim=-1) + control_stage.mean(dim=-1)
        return cost

    def forward(self, x0: torch.Tensor, R: torch.Tensor) -> dict:
        """
        Run forward rollout and return loss.

        Args:
            x0:     (B, 7)         initial state
            R:      (B, N+1, 7)     reference [x_ref, y_ref, yaw_ref]
        Returns:
            dict with:
              - loss: scalar tensor
              - track_mean: scalar tensor
              - pen_mean: scalar tensor
              - X: (B, N+1, 7)
              - U: (B, N, 2)
        """
        # Rollout
        X, U = self.solver.rollout(x0, R)

        # Tracking cost using your Q/R matrices (assumes tracking_cost expects (X,U,R_full))
        track = self.tracking_cost(X, U, R).mean()
        # Constraint penalty
        pen = self.constraint_penalty(X, U, R).mean()

        # Total loss
        loss = (track + pen)

        return {
            "loss": loss,
            "track_mean": track,
            "pen_mean": pen,
            "X": X,
            "U": U
        }

    def train(self, training_params):
        """
        training_params: a dict of training parameters, e.g. number of iterations, optimizer settings, etc.
        """
        num_epochs = training_params.get("num_epochs", 50)
        steps_per_epoch = training_params.get("steps_per_epoch", 200)
        batch_size = training_params.get("batch_size", 64)
        lr = training_params.get("lr", 1e-3)
        weight_decay = training_params.get("weight_decay", 0.0)
        grad_clip = training_params.get("grad_clip", 1.0)
        log_every = training_params.get("log_every", 20)
        save_path = training_params.get("save_path", None)
        save_every = training_params.get("save_every", 10)

        # Plateau scheduler params
        plateau_factor = training_params.get("plateau_factor", 0.5)
        plateau_patience = training_params.get("plateau_patience", 5)
        plateau_min_lr = training_params.get("plateau_min_lr", 1e-6)
        plateau_threshold = training_params.get("plateau_threshold", 1e-4)
        plateau_cooldown = training_params.get("plateau_cooldown", 0)
        eval_batch_size = training_params.get("eval_batch_size", 256)
        early_stop_patience = training_params.get("early_stop_patience", 10)
        early_stop_min_delta = training_params.get("early_stop_min_delta", 1e-4)
        penalty_increase_factor = training_params.get("penalty_increase_factor", 1.5)

        # Make sure policy is on device and in train mode
        self.solver.policy.to(self.device)
        self.solver.policy.train()

        opt = torch.optim.Adam(self.solver.policy.parameters(), lr=lr, weight_decay=weight_decay)

        # ReduceLROnPlateau scheduler (step it once per epoch using eval loss)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            cooldown=plateau_cooldown,
            min_lr=plateau_min_lr,
            verbose=True,
        )

        # Fixed eval set (reduces noise vs using random batches)
        torch.manual_seed(training_params.get("eval_seed", 123))
        eval_x0, eval_R = self.create_sample_batch(eval_batch_size)
        eval_x0 = eval_x0.to(self.device)
        eval_R = eval_R.to(self.device)
        eval_x0 = eval_x0.clone()
        eval_x0[:, 4] = wrap_angle(eval_x0[:, 4])

        # Early stopping state (based on eval loss)
        best_eval_loss = float("inf")
        bad_epochs = 0

        self.penalty_weight = self.solver.config.lambdas

        # Start training!
        for ep in range(1, num_epochs + 1):
            loss_sum = 0.0
            track_sum = 0.0
            pen_sum = 0.0

            # Increase lambdas when training is more stable
            self.increase_lambdas(rate = penalty_increase_factor)
            print("Weights for constraint violation:", self.penalty_weight)
            for it in range(1, steps_per_epoch + 1):
                x0_b, R_b = self.create_sample_batch(batch_size)
                x0_b = x0_b.to(self.device)
                R_b  = R_b.to(self.device)

                # wrap yaw in x0 for stability
                x0_b = x0_b.clone()
                x0_b[:, 4] = wrap_angle(x0_b[:, 4])

                opt.zero_grad(set_to_none=True)
                out = self.forward(x0_b, R_b)
                out["loss"].backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.solver.policy.parameters(), grad_clip)

                opt.step()

                loss_sum += out["loss"].item()
                track_sum += out["track_mean"].item()
                pen_sum += out["pen_mean"].item()

                if it == 1 or (it % log_every) == 0:
                    k = float(it)
                    print(f"[ep {ep:03d} it {it:04d}/{steps_per_epoch}] "
                        f"batch_loss={out['loss'].item():.6e} "
                        f"run_loss={loss_sum/k:.6e} "
                        f"run_track={track_sum/k:.6e} run_pen={pen_sum/k:.3e}")
            
            denom = float(steps_per_epoch)
            train_loss = loss_sum / denom
            train_track = track_sum / denom
            train_pen = pen_sum / denom

            # ---- eval + scheduler step ----
            self.solver.policy.eval()
            with torch.no_grad():
                eval_out = self.forward(eval_x0, eval_R)
                eval_loss = float(eval_out["loss"].item())
                eval_track = float(eval_out["track_mean"].item())
                eval_pen = float(eval_out["pen_mean"].item())
            self.solver.policy.train()

            scheduler.step(eval_loss) 

            cur_lr = opt.param_groups[0]["lr"]
            print(f"[epoch {ep:03d}] "
                f"train_loss={train_loss:.6e} train_track={train_track:.6e} train_pen={train_pen:.6e} | "
                f"eval_loss={eval_loss:.6e} eval_track={eval_track:.6e} eval_pen={eval_pen:.6e} | "
                f"lr={cur_lr:.2e}")

            # ---- early stopping ----
            improved = eval_loss < (best_eval_loss - early_stop_min_delta)
            if improved:
                best_eval_loss = eval_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= early_stop_patience:
                print(
                    f"early stopping at epoch {ep:03d}: "
                    f"no eval_loss improvement > {early_stop_min_delta:.1e} "
                    f"for {bad_epochs} epoch(s)"
                )
                if save_path is not None:
                    ckpt = {
                        "epoch": ep,
                        "policy_state_dict": self.solver.policy.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "training_params": training_params,
                        "eval_loss": eval_loss,
                    }
                    torch.save(ckpt, save_path)
                    print(f"saved checkpoint to {save_path}")
                break

            if save_path is not None and (ep % save_every == 0 or ep == num_epochs):
                ckpt = {
                    "epoch": ep,
                    "policy_state_dict": self.solver.policy.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "training_params": training_params,
                    "eval_loss": eval_loss,
                }
                torch.save(ckpt, save_path)
                print(f"saved checkpoint to {save_path}")
