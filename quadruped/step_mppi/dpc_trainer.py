import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .dpc_solver import DPC


class DPC_Trainer:
    """Training for differentiable predictive control policies."""

    def __init__(self, dpc_solver: DPC):
        self.dpc_solver = dpc_solver
        self._optimizer = None
        self._train_step = None
        self._eval_step = jax.jit(lambda params, batch: self.dpc_solver.loss(params, batch))

    def load_dataset(
        self,
        data_filename,
        num_points: int | None = None,
        random_subset: bool = True,
        seed: int = 0,
    ):
        """Load a packed DPC dataset from disk.

        The saved dataset format stores:
        - current_centroidal_state: (N, 24)
        - reference: (N, 24)
        - contact_sequence: either (N, H, 4) or legacy (N, 4, H)

        Args:
            data_filename: Path to the dataset .npz file.
            num_points: Optional number of samples to load from the beginning
                of the dataset. If None, load the full dataset.
            random_subset: If True, randomly sample num_points entries
                without replacement instead of taking the first samples.
            seed: Random seed used when random_subset is enabled.

        Returns a batch dictionary directly compatible with self.loss(...).
        """
        data_path = pathlib.Path(data_filename)
        with np.load(data_path) as data:
            initial_state = np.asarray(data["current_centroidal_state"], dtype=np.float32)
            reference = np.asarray(data["reference"], dtype=np.float32)
            reference_state_horizon = np.asarray(data["reference_state_horizon"], dtype=np.float32)
            contact_sequence = np.asarray(data["contact_sequence"], dtype=np.float32)

            total_num_points = initial_state.shape[0]
            if num_points is not None:
                if num_points <= 0:
                    raise ValueError(f"num_points must be positive, got {num_points}")
                num_points = min(num_points, total_num_points)
                if random_subset:
                    rng = np.random.default_rng(seed)
                    selection = np.sort(rng.choice(total_num_points, size=num_points, replace=False))
                else:
                    selection = slice(0, num_points)
            else:
                selection = slice(None)

            dataset = {
                "initial_state": jnp.asarray(initial_state[selection]),
                "reference": jnp.asarray(reference[selection]),
                "contact_sequence": jnp.asarray(contact_sequence[selection]),
                "reference_state_horizon": jnp.asarray(reference_state_horizon[selection]),
            }

        self.dataset = dataset
        return dataset

    def create_batch(self, batch_size: int, key=None):
        """Sample a batch of x0, R, and C from a loaded dataset."""
        x0_all = self.dataset["initial_state"]
        R_all = self.dataset["reference_state_horizon"]
        C_all = self.dataset["contact_sequence"]
        num_samples = x0_all.shape[0]

        if key is None:
            idx = np.random.randint(0, num_samples, size=batch_size)
            idx = jnp.asarray(idx)
        else:
            idx = jax.random.randint(key, (batch_size,), 0, num_samples)

        x0_b = x0_all[idx]
        R_b = R_all[idx]
        C_b = C_all[idx]
        return x0_b, R_b, C_b

    def init_params(self, key):
        """Initialize policy parameters through the associated DPC solver."""
        return self.dpc_solver.init_policy_params(key)

    def rollout_single(self, x0_single, R_single, C_single, params):
        def step_fn(state, inputs):
            reference, current_contact, step_idx = inputs

            grfs = self.dpc_solver.predict_first_step_grfs(
                params,
                state,
                reference,
                current_contact,
            )
            grfs_matrix = grfs.reshape(4, 3)

            number_of_legs_in_stance = jnp.sum(current_contact)
            safe_num_stance = jnp.maximum(number_of_legs_in_stance, 1.0)
            reference_force_stance_legs = (self.dpc_solver.model.mass * 9.81) / safe_num_stance

            input_vec = jnp.concatenate(
                (jnp.zeros((12,), dtype=jnp.float32), grfs),
                axis=0,
            )
            next_state = self.dpc_solver.model.integrate_jax(state, input_vec, current_contact, step_idx)

            state_error = next_state - reference[: self.dpc_solver.state_dim]
            track = state_error.T @ self.dpc_solver.Q @ state_error

            input_for_cost = jnp.concatenate(
                (
                    jnp.zeros((12,), dtype=jnp.float32),
                    jnp.stack(
                        (
                            grfs_matrix[0, 0],
                            grfs_matrix[0, 1],
                            grfs_matrix[0, 2] - reference_force_stance_legs,
                            grfs_matrix[1, 0],
                            grfs_matrix[1, 1],
                            grfs_matrix[1, 2] - reference_force_stance_legs,
                            grfs_matrix[2, 0],
                            grfs_matrix[2, 1],
                            grfs_matrix[2, 2] - reference_force_stance_legs,
                            grfs_matrix[3, 0],
                            grfs_matrix[3, 1],
                            grfs_matrix[3, 2] - reference_force_stance_legs,
                        ),
                        axis=0,
                    ),
                ),
                axis=0,
            )
            pen = input_for_cost.T @ self.dpc_solver.R @ input_for_cost

            loss = track + pen

            outputs = {
                "state": next_state,
                "control": grfs,
                "loss": loss,
            }
            return next_state, outputs

        horizon = R_single.shape[0]
        step_indices = jnp.arange(horizon, dtype=jnp.int32)
        _, outputs = jax.lax.scan(step_fn, x0_single, (R_single, C_single, step_indices))

        X = jnp.concatenate((x0_single[None, :], outputs["state"]), axis=0)
        U = outputs["control"]
        loss = jnp.mean(outputs["loss"])
        return loss, X, U

    def forward(self, params, x0, R, C):
        """Run a forward rollout and return loss terms and trajectories.

        Args:
            params: Policy parameters.
            x0: Initial state batch, shape (B, 24).
            R: Reference batch, shape (B, H, 24).
            C: Contact-sequence batch, shape (B, H, 4) or (B, 4, H).
        """
        x0 = jnp.asarray(x0, dtype=jnp.float32)
        R = jnp.asarray(R, dtype=jnp.float32)
        C = jnp.asarray(C, dtype=jnp.float32)
        if C.ndim == 3 and C.shape[1] == 4:
            C = jnp.swapaxes(C, 1, 2)

        batched_forward = jax.vmap(
            lambda x0_single, R_single, C_single: self.rollout_single(x0_single, R_single, C_single, params),
            in_axes=(0, 0, 0),
            out_axes=(0, 0, 0),
        )
        loss, X, U = batched_forward(x0, R, C)

        return {
            "loss": jnp.mean(loss),
            "X": X,
            "U": U,
        }

    def _make_optimizer(self, lr, weight_decay, grad_clip):
        transforms = []
        if grad_clip is not None:
            transforms.append(optax.clip_by_global_norm(grad_clip))
        transforms.append(optax.adamw(learning_rate=lr, weight_decay=weight_decay))
        return optax.chain(*transforms)

    def _build_train_step(self, optimizer):
        def train_step(params, optimizer_state, batch):
            loss_value, grads = jax.value_and_grad(
                lambda current_params: self.dpc_solver.loss(current_params, batch)
            )(params)
            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            return params, optimizer_state, loss_value

        return jax.jit(train_step)

    def _create_loss_batch(self, x0, R, C):
        C = jnp.asarray(C, dtype=jnp.float32)
        if C.ndim == 3 and C.shape[1] != 4:
            C = jnp.swapaxes(C, 1, 2)
        return {
            "initial_state": jnp.asarray(x0, dtype=jnp.float32),
            "reference": jnp.asarray(R[:, 0, :], dtype=jnp.float32),
            "contact_sequence": C,
        }

    def _save_checkpoint(self, save_path, epoch, params, optimizer_state, training_params, eval_loss):
        checkpoint = {
            "epoch": epoch,
            "params": params,
            "optimizer_state": optimizer_state,
            "training_params": training_params,
            "eval_loss": eval_loss,
        }
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as file:
            pickle.dump(checkpoint, file)
        print(f"saved checkpoint to {save_path}")

    def load_trained_model(self, checkpoint_path, training_params=None):
        """Load a previously saved checkpoint for continued training."""
        checkpoint_path = pathlib.Path(checkpoint_path)
        with open(checkpoint_path, "rb") as file:
            checkpoint = pickle.load(file)

        checkpoint_training_params = checkpoint.get("training_params", {})
        merged_training_params = dict(checkpoint_training_params)
        if training_params is not None:
            merged_training_params.update(training_params)

        lr = merged_training_params.get("lr", 1e-3)
        weight_decay = merged_training_params.get("weight_decay", 0.0)
        grad_clip = merged_training_params.get("grad_clip", 1.0)

        self._optimizer = self._make_optimizer(lr, weight_decay, grad_clip)
        self._train_step = self._build_train_step(self._optimizer)

        return {
            "params": checkpoint["params"],
            "optimizer_state": checkpoint.get("optimizer_state"),
            "epoch": checkpoint.get("epoch", 0),
            "training_params": merged_training_params,
            "eval_loss": checkpoint.get("eval_loss"),
        }

    def train(self, training_params, params=None, optimizer_state=None, start_epoch: int = 0):
        """Train the DPC policy with random mini-batches from the loaded dataset."""
        num_epochs = training_params.get("num_epochs", 50)
        steps_per_epoch = training_params.get("steps_per_epoch", 200)
        batch_size = training_params.get("batch_size", 64)
        lr = training_params.get("lr", 1e-3)
        weight_decay = training_params.get("weight_decay", 0.0)
        grad_clip = training_params.get("grad_clip", 1.0)
        log_every = training_params.get("log_every", 20)
        save_path = training_params.get("save_path", None)
        save_every = training_params.get("save_every", 10)

        plateau_factor = training_params.get("plateau_factor", 0.5)
        plateau_patience = training_params.get("plateau_patience", 5)
        plateau_min_lr = training_params.get("plateau_min_lr", 1e-6)
        plateau_threshold = training_params.get("plateau_threshold", 1e-4)
        plateau_cooldown = training_params.get("plateau_cooldown", 0)
        eval_batch_size = training_params.get("eval_batch_size", 256)
        early_stop_patience = training_params.get("early_stop_patience", 10)
        early_stop_min_delta = training_params.get("early_stop_min_delta", 1e-4)
        seed = training_params.get("seed", 0)
        eval_seed = training_params.get("eval_seed", 123)

        current_lr = lr
        optimizer = self._make_optimizer(current_lr, weight_decay, grad_clip)
        self._optimizer = optimizer
        self._train_step = self._build_train_step(optimizer)

        if params is None:
            params = self.init_params(jax.random.PRNGKey(seed))
        if optimizer_state is None:
            optimizer_state = optimizer.init(params)

        eval_key = jax.random.PRNGKey(eval_seed)
        eval_x0, eval_R, eval_C = self.create_batch(eval_batch_size, key=eval_key)
        eval_batch = self._create_loss_batch(eval_x0, eval_R, eval_C)

        best_eval_loss = float("inf")
        bad_epochs = 0
        bad_plateau_epochs = 0
        cooldown_counter = 0
        history = []

        for epoch in range(start_epoch + 1, num_epochs + 1):
            loss_sum = 0.0

            for iteration in range(1, steps_per_epoch + 1):
                batch_key = jax.random.PRNGKey(seed + epoch * steps_per_epoch + iteration)
                x0_b, R_b, C_b = self.create_batch(batch_size, key=batch_key)
                batch = self._create_loss_batch(x0_b, R_b, C_b)

                params, optimizer_state, loss_value = self._train_step(params, optimizer_state, batch)

                batch_loss = float(loss_value)
                loss_sum += batch_loss

                if iteration == 1 or (iteration % log_every) == 0:
                    count = float(iteration)
                    print(
                        f"[ep {epoch:03d} it {iteration:04d}/{steps_per_epoch}] "
                        f"batch_loss={batch_loss:.6e} "
                        # f"run_loss={loss_sum / count:.6e} "
                    )

            denom = float(steps_per_epoch)
            train_loss = loss_sum / denom

            eval_loss = float(self._eval_step(params, eval_batch))

            if cooldown_counter > 0:
                cooldown_counter -= 1

            if eval_loss < (best_eval_loss - plateau_threshold):
                bad_plateau_epochs = 0
            else:
                bad_plateau_epochs += 1

            if (
                bad_plateau_epochs >= plateau_patience
                and cooldown_counter == 0
                and current_lr > plateau_min_lr
            ):
                new_lr = max(current_lr * plateau_factor, plateau_min_lr)
                if new_lr < current_lr:
                    current_lr = new_lr
                    optimizer = self._make_optimizer(current_lr, weight_decay, grad_clip)
                    optimizer_state = optimizer.init(params)
                    self._optimizer = optimizer
                    self._train_step = self._build_train_step(optimizer)
                    cooldown_counter = plateau_cooldown
                    bad_plateau_epochs = 0
                    print(f"reduced learning rate to {current_lr:.2e}")

            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_loss:.6e} | "
                f"eval_loss={eval_loss:.6e} | "
                f"lr={current_lr:.2e}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "lr": current_lr,
                }
            )

            improved = eval_loss < (best_eval_loss - early_stop_min_delta)
            if improved:
                best_eval_loss = eval_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if save_path is not None and (epoch % save_every == 0 or epoch == num_epochs):
                self._save_checkpoint(
                    save_path,
                    epoch,
                    params,
                    optimizer_state,
                    training_params,
                    eval_loss,
                )

            if bad_epochs >= early_stop_patience:
                print(
                    f"early stopping at epoch {epoch:03d}: "
                    f"no eval_loss improvement > {early_stop_min_delta:.1e} "
                    f"for {bad_epochs} epoch(s)"
                )
                if save_path is not None:
                    self._save_checkpoint(
                        save_path,
                        epoch,
                        params,
                        optimizer_state,
                        training_params,
                        eval_loss,
                    )
                break

        return {
            "params": params,
            "optimizer_state": optimizer_state,
            "history": history,
            "best_eval_loss": best_eval_loss,
        }
