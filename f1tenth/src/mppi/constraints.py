import numpy as np
import jax.numpy as jnp

def boundary_constraint_coeffs(ax: np.ndarray, by: np.ndarray, 
            c_left: np.ndarray, c_right: np.ndarray, d_min = 0.2):
    A_seg = jnp.stack([ax, by], axis=-1)        
    A_hor = jnp.stack([A_seg, -A_seg], axis=1) 
    b_hor = jnp.stack([c_left-d_min, -c_right-d_min], axis=1) 

    return A_hor, b_hor

def boundary_constraint(x, A_hor, b_hor):
    pos = x[:, :2]
    lhs = jnp.einsum("ti,tci->tc", pos, A_hor)
    violation = jnp.maximum(0.0, lhs - b_hor)
    return jnp.max(violation, axis=-1)    
    
def st_limit_constraint_coeffs(x_min: np.ndarray, x_max: np.ndarray, horizon: int):
    """
    """
    finite_mask = np.isfinite(x_min)
    finite_indices_np = np.where(finite_mask)[0].astype(np.int32)

    x_min_filtered_np = x_min[finite_mask].astype(np.float32)
    x_max_filtered_np = x_max[finite_mask].astype(np.float32)
    A_min = -jnp.eye(len(finite_indices_np), dtype=jnp.float32)
    b_min = -jnp.asarray(x_min_filtered_np, dtype=jnp.float32)
    A_max = jnp.eye(len(finite_indices_np), dtype=jnp.float32)
    b_max = jnp.asarray(x_max_filtered_np, dtype=jnp.float32)
    A = jnp.concatenate([A_min, A_max]) 
    b = jnp.concatenate([b_min, b_max]) 
    A_hor = jnp.tile(A[None, :, :], (horizon, 1, 1))
    b_hor = jnp.tile(b[None, :], (horizon, 1))

    return A_hor, b_hor

def st_limit_constraint(x, A_hor, b_hor):
    x_filtered = x[:, 2:4]
    if A_hor.ndim == 3:
        lhs = jnp.einsum('ti,tci->tc', x_filtered, A_hor)
    else:
        lhs = jnp.dot(x_filtered, A_hor.T)
    violation = 0*jnp.maximum(0.0, lhs - b_hor)
    return jnp.linalg.norm(violation, axis=-1)
