import torch
from trainer.DPC_trainer import DPC_Trainer
from dpc.DPC_solver import wrap_angle
import math

class DMPPI_Trainer(DPC_Trainer):
    """
    DMPPI trainer that extends DPC trainer utilities.
    Keep inherited methods unless explicitly overridden.
    """

    def __init__(self, solver, device=None):
        super().__init__(solver=solver, device=device)
        self.alpha = solver.config.alpha
        self.beta = getattr(solver.config, "beta", 0.0)

    def expl_loss(self,
        L: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Entropy loss of the policy distribution along the rollout.
        Returns per-batch bonus: (B,)
        """
        config = self.solver.config
        
        nu = config.nu
        const = 0.5 * nu * (1.0 + math.log(2.0 * math.pi))

        diag = torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-8)  # (B, N, nu)
        h = (torch.log(diag).sum(dim=-1) + const) # (B, N)
        return h.mean(dim=1)  # (B,)

    def forward(self, x0: torch.Tensor, R: torch.Tensor) -> dict:
        """
        Run one stochastic rollout and return scalar losses.
        """
        X, U, L, updater_sup = self.solver.rollout(x0, R)
        track = self.tracking_cost(X, U, R).mean()
        pen = self.constraint_penalty(X, U, R).mean()
        expl = self.expl_loss(L).mean()
        loss = track + pen - self.alpha * expl + self.beta * updater_sup

        return {
            "loss": loss,
            "track_mean": track,
            "pen_mean": pen,
            "expl_mean": expl,
            "updater_sup_mean": updater_sup,
            "X": X,
            "U": U,
        }

    def train(self, training_params):
        """
        Train with a single rollout per optimizer step.
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
        trainable_params = list(self.solver.policy.parameters())
        if self.solver.updater is not None and hasattr(self.solver.updater, "parameters"):
            self.solver.updater.to(self.device)
            self.solver.updater.train()
            trainable_params.extend(self.solver.updater.parameters())

        opt = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

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

        # Start training
        for ep in range(1, num_epochs + 1):
            loss_sum = 0.0
            track_sum = 0.0
            pen_sum = 0.0
            expl_sum = 0.0
            updater_sum = 0.0

            self.increase_lambdas(rate=penalty_increase_factor)

            for it in range(1, steps_per_epoch + 1):
                x0_b, R_b = self.create_sample_batch(batch_size)
                x0_b = x0_b.to(self.device)
                R_b = R_b.to(self.device)

                x0_b = x0_b.clone()
                x0_b[:, 4] = wrap_angle(x0_b[:, 4])
                opt.zero_grad(set_to_none=True)

                X, U, L, updater_sup = self.solver.rollout(x0_b, R_b)
                track_m = self.tracking_cost(X, U, R_b).mean()
                pen_m = self.constraint_penalty(X, U, R_b).mean()
                expl_m = self.expl_loss(L).mean()
                loss_m = track_m + pen_m - self.alpha * expl_m + self.beta * updater_sup

                loss_m.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

                opt.step()

                out_loss = float(loss_m.detach().item())
                out_track = float(track_m.detach().item())
                out_pen = float(pen_m.detach().item())
                out_expl = float(expl_m.detach().item())
                out_updater = float(updater_sup.detach().item())

                loss_sum += out_loss
                track_sum += out_track
                pen_sum += out_pen
                expl_sum += out_expl
                updater_sum += out_updater

                if it == 1 or (it % log_every) == 0:
                    k = float(it)
                    print(
                        f"[ep {ep:03d} it {it:04d}/{steps_per_epoch}] "
                        f"run_loss={loss_sum/k:.6e} "
                        f"run_track={track_sum/k:.6e} run_pen={pen_sum/k:.3e} "
                        f"run_expl={expl_sum/k:.3e} run_upd={updater_sum/k:.3e}"
                    )

            denom = float(steps_per_epoch)
            train_loss = loss_sum / denom
            train_track = track_sum / denom
            train_pen = pen_sum / denom
            train_expl = expl_sum / denom
            train_updater = updater_sum / denom

            # Eval keeps forward() behavior (no grad graph retained)
            self.solver.policy.eval()
            if self.solver.updater is not None and hasattr(self.solver.updater, "eval"):
                self.solver.updater.eval()
            with torch.no_grad():
                eval_out = self.forward(eval_x0, eval_R)
                eval_loss = float(eval_out["loss"].item())
                eval_track = float(eval_out["track_mean"].item())
                eval_pen = float(eval_out["pen_mean"].item())
                eval_expl = float(eval_out["expl_mean"].item())
                eval_updater = float(eval_out["updater_sup_mean"].item())
            self.solver.policy.train()
            if self.solver.updater is not None and hasattr(self.solver.updater, "train"):
                self.solver.updater.train()

            scheduler.step(eval_loss)

            cur_lr = opt.param_groups[0]["lr"]
            print(
                f"[epoch {ep:03d}] "
                f"train_loss={train_loss:.6e} train_track={train_track:.6e} train_pen={train_pen:.6e} "
                f"train_expl={train_expl:.6e} train_upd={train_updater:.6e} | "
                f"eval_loss={eval_loss:.6e} eval_track={eval_track:.6e} eval_pen={eval_pen:.6e} "
                f"eval_expl={eval_expl:.6e} eval_upd={eval_updater:.6e} | "
                f"lr={cur_lr:.2e}"
            )

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
                    if self.solver.updater is not None and hasattr(self.solver.updater, "state_dict"):
                        ckpt["updater_state_dict"] = self.solver.updater.state_dict()
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
                if self.solver.updater is not None and hasattr(self.solver.updater, "state_dict"):
                    ckpt["updater_state_dict"] = self.solver.updater.state_dict()
                torch.save(ckpt, save_path)
                print(f"saved checkpoint to {save_path}")
