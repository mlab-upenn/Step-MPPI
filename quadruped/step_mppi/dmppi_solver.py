import jax
import jax.numpy as jnp

from .dpc_policy_jax import (
    NeuralGRFDistributionPolicy,
)
from .dpc_solver import DPC


class DMPPI(DPC):
    """
    DMPPI solver with a single-step MPPI-style correction stage.
    """

    def __init__(
        self,
        policy: NeuralGRFDistributionPolicy | None = None,
        updater=None,
        device: str | None = None,
        horizon: int | None = None,
        dt: float | None = None,
        num_dmppi_samples: int = 64,
        dmppi_temperature: float = 0.2,
    ):
        if policy is None:
            policy = NeuralGRFDistributionPolicy()
        super().__init__(policy=policy, device=device, horizon=horizon, dt=dt)
        self.num_dmppi_samples = int(num_dmppi_samples)
        self.dmppi_temperature = float(dmppi_temperature)
        # this is the MPPI updater (None is vanilla importance sampling update)
        self.updater = updater
        if self.updater is not None and hasattr(self.updater, "to"):
            self.updater = self.updater.to(device)
        self.master_key = jax.random.PRNGKey(0)
        self._runtime_inference_step = self._build_runtime_inference_step()

    def reset(self):
        """Reset controller-side rollout state and restart the sampling key."""
        super().reset()
        self.master_key = jax.random.PRNGKey(0)

    def _policy_params(self, params):
        """Extract policy params from either a plain or combined param tree."""
        if isinstance(params, dict) and "policy" in params:
            return params["policy"]
        return params

    def _updater_params(self, params):
        """Extract updater params from a combined param tree."""
        if isinstance(params, dict) and "updater" in params:
            return params["updater"]
        return None

    def _build_runtime_inference_step(self):
        """Build cached runtime inference with one-step DMPPI ."""

        def runtime_inference_step(params, state, reference, current_contact, key):
            policy_params = self._policy_params(params)
            updater_params = self._updater_params(params)
            nominal_grfs, nominal_std = self.policy.apply(
                {"params": policy_params},
                state,
                reference,
                current_contact,
            )
            nominal_grfs = self.project_grfs(nominal_grfs, current_contact)
            corrected_grfs = self.single_step_dmppi(
                state,
                reference,
                current_contact,
                nominal_grfs,
                nominal_std,
                key,
                updater_params=updater_params,
            )
            input_vec = jnp.concatenate(
                (jnp.zeros((12,), dtype=jnp.float32), corrected_grfs),
                axis=0,
            )
            next_state = self.model.integrate_jax(state, input_vec, current_contact, 0)
            return corrected_grfs, next_state

        return jax.jit(runtime_inference_step)

    def _default_state_cost_weight(self):
        """Reuse the DPC default state-cost structure."""
        return super()._default_state_cost_weight()

    def _default_control_cost_weight(self):
        """Reuse the DPC default control-cost structure."""
        return super()._default_control_cost_weight()

    def init_policy_params(self, key):
        """Initialize neural policy parameters."""
        return super().init_policy_params(key)

    def project_grfs(self, grfs, current_contact):
        """Project policy outputs into friction-constrained contact forces."""
        return super().project_grfs(grfs, current_contact)

    def sample_force_candidates(self, mean_grfs, std_grfs, current_contact, key):
        """Sample candidate GRFs from the policy's Gaussian proposal."""
        noise = jax.random.normal(
            key,
            shape=(self.num_dmppi_samples, mean_grfs.shape[0]),
            dtype=jnp.float32,
        )
        candidates = mean_grfs[None, :] + noise * std_grfs[None, :]
        nominal = mean_grfs[None, :]
        candidates = jnp.concatenate((nominal, candidates), axis=0)
        return jax.vmap(self.project_grfs, in_axes=(0, None))(candidates, current_contact)

    def evaluate_force_candidates(self, state, reference, current_contact, candidate_grfs):
        """Evaluate one-step costs for candidate corrected forces."""

        def candidate_cost(grfs):
            grfs_matrix = grfs.reshape(4, 3)
            number_of_legs_in_stance = jnp.sum(current_contact)
            safe_num_stance = jnp.maximum(number_of_legs_in_stance, 1.0)
            reference_force_stance_legs = (self.model.mass * 9.81) / safe_num_stance

            input_vec = jnp.concatenate(
                (jnp.zeros((12,), dtype=jnp.float32), grfs),
                axis=0,
            )
            next_state = self.model.integrate_jax(state, input_vec, current_contact, 0)

            state_error = next_state - reference[: self.state_dim]
            state_cost = state_error.T @ self.Q @ state_error

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
            control_cost = input_for_cost.T @ self.R @ input_for_cost
            return state_cost + control_cost

        return jax.vmap(candidate_cost)(candidate_grfs)

    def _mppi_weighted_update(self, uk_s_reshaped, c_total, temp):
        """Vanilla MPPI importance-sampling update on stabilized costs."""
        weights = jax.nn.softmax(-c_total / temp)
        return jnp.sum(uk_s_reshaped * weights[:, None], axis=0)

    def single_step_dmppi(self, state, reference, current_contact, nominal_grfs, nominal_std, key, updater_params=None):
        """Apply a single-step DMPPI  around the neural proposal."""
        candidate_grfs = self.sample_force_candidates(nominal_grfs, nominal_std, current_contact, key)
        candidate_costs = self.evaluate_force_candidates(
            state,
            reference,
            current_contact,
            candidate_grfs,
        )
        stabilized_costs = candidate_costs - jnp.min(candidate_costs)
        if self.updater is None or updater_params is None:
            corrected_grfs = self._mppi_weighted_update(
                candidate_grfs,
                stabilized_costs,
                self.dmppi_temperature,
            )
        else:
            corrected_grfs = self.updater.apply(
                {"params": updater_params},
                nominal_grfs[None, :],
                nominal_std[None, :],
                stabilized_costs[None, :],
                current_contact=current_contact[None, :],
            )
            corrected_grfs = corrected_grfs[0]
        return self.project_grfs(corrected_grfs, current_contact)

    def predict_first_step_grfs(self, params, state, reference, current_contact):
        """Predict NN forces, then refine them with one-step DMPPI."""
        self.master_key, subkey = jax.random.split(self.master_key)
        policy_params = self._policy_params(params)
        updater_params = self._updater_params(params)
        nominal_grfs, nominal_std = self.policy.apply(
            {"params": policy_params},
            state,
            reference,
            current_contact,
        )
        nominal_grfs = self.project_grfs(nominal_grfs, current_contact)
        return self.single_step_dmppi(
            jnp.asarray(state, dtype=jnp.float32),
            jnp.asarray(reference, dtype=jnp.float32),
            jnp.asarray(current_contact, dtype=jnp.float32),
            nominal_grfs,
            jnp.asarray(nominal_std, dtype=jnp.float32),
            subkey,
            updater_params=updater_params,
        )

    def runtime_inference_step(self, params, state, reference, current_contact):
        """Run cached jitted single-step DMPPI inference for online control."""
        self.master_key, subkey = jax.random.split(self.master_key)
        state = jnp.asarray(state, dtype=jnp.float32)
        reference = jnp.asarray(reference, dtype=jnp.float32)
        current_contact = jnp.asarray(current_contact, dtype=jnp.float32)
        return self._runtime_inference_step(params, state, reference, current_contact, subkey)

    def prepare_state_and_reference(self, state_current, reference_state, current_contact, previous_contact=None):
        """Pack SRBD-style dictionaries into flat DMPPI tensors."""
        return super().prepare_state_and_reference(
            state_current,
            reference_state,
            current_contact,
            previous_contact,
        )

    def rollout_cost(self, params, initial_state, reference, contact_sequence):
        """Compute horizon rollout cost under the DMPPI-corrected policy."""
        return super().rollout_cost(params, initial_state, reference, contact_sequence)

    def loss(self, params, batch):
        """Compute mean rollout cost over a batch."""
        return super().loss(params, batch)
