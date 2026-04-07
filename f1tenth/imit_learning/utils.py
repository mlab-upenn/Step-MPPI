import numpy as np
import jax

# Convert state dict to numpy array
def obs_dict_to_array(obs: dict) -> np.ndarray:
    keys = ("pose_x", "pose_y", "delta", "linear_vel_x", "pose_theta", "ang_vel_z", "beta")
    return np.array([obs[k] for k in keys], dtype=np.float32)

def jax_to_numpy(x):
    return np.array(jax.device_get(x))

def mu_to_action(mu_seq: np.ndarray) -> np.ndarray:
    """
    Convert a mean control sequence to a single action (steer_vel, accel).
    Handles either (2, N) or (N, 2).
    Returns shape (2,).
    """
    mu_seq = np.asarray(mu_seq)
    if mu_seq.shape[0] == 2:
        # (2, N)
        return mu_seq[:, 0]
    elif mu_seq.shape[-1] == 2:
        # (N, 2)
        return mu_seq[0, :]
    else:
        raise ValueError(f"Unexpected mu shape for action extraction: {mu_seq.shape}")
