import numpy as np
import torch

def boundary_constraint_coeffs_torch(coeffs: torch.Tensor, d_min: float = 0.3):
    """
    Build batched horizon-wise boundary constraint coefficients for Ax <= b.

    Args:
        coeffs: (B, N, 4) with columns [ax, by, c_left, c_right]
        d_min: safety margin subtracted from both sides

    Returns:
        A_hor: (B, N, 2, 2)
        b_hor: (B, N, 2)
    """
    if coeffs.ndim != 3 or coeffs.shape[-1] != 4:
        raise ValueError(
            f"Expected coeffs to have shape (B, N, 4), got {tuple(coeffs.shape)}"
        )

    ax = coeffs[..., 0]
    by = coeffs[..., 1]
    c_left = coeffs[..., 2]
    c_right = coeffs[..., 3]

    A_seg = torch.stack([ax, by], dim=-1)
    A_hor = torch.stack([A_seg, -A_seg], dim=2)
    b_hor = torch.stack([c_left - d_min, -c_right - d_min], dim=-1)

    return A_hor, b_hor

def boundary_constraint_batched(
    x: torch.Tensor,
    A_hor: torch.Tensor,
    b_hor: torch.Tensor,
) -> torch.Tensor:
    """
    Batched horizon-wise boundary constraint.

    Args:
        x: (B, N, 7)
        A_hor: (B, N, 2, 2)
        b_hor: (B, N, 2)

    Returns:
        (B, N): positive = violation at each timestep.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape (B, N, 7), got {tuple(x.shape)}")
    if A_hor.ndim != 4:
        raise ValueError(f"Expected A_hor to have shape (B, N, 2, 2), got {tuple(A_hor.shape)}")
    if b_hor.ndim != 3:
        raise ValueError(f"Expected b_hor to have shape (B, N, 2), got {tuple(b_hor.shape)}")

    pos = x[:, :, :2]
    A_hor = A_hor.to(dtype=x.dtype, device=x.device)
    b_hor = b_hor.to(dtype=x.dtype, device=x.device)

    lhs = torch.einsum("bti,btci->btc", pos, A_hor)
    violation = torch.clamp(lhs - b_hor, min=0.0)
    return torch.linalg.norm(violation, dim=-1)

def st_limit_constraint_coeffs_torch(
    x_min: np.ndarray,
    x_max: np.ndarray,
    horizon: int,
    device=None,
    dtype=torch.float32,
):
    """
    Build horizon-wise linear state constraint coefficients for Ax <= b.
    Only finite entries of x_min/x_max are kept.
    Args:
        x_min: (nx,)
        x_max: (nx,)
        horizon: int

    Returns:
        A_hor: (N, C, 2)
        b_hor: (N, C)
    """
    finite_mask = np.isfinite(x_min)
    finite_indices_np = np.where(finite_mask)[0].astype(np.int64)

    x_min_filtered_np = x_min[finite_mask].astype(np.float32)
    x_max_filtered_np = x_max[finite_mask].astype(np.float32)

    n_finite = len(finite_indices_np)

    A_min = -torch.eye(n_finite, dtype=dtype, device=device)
    b_min = -torch.as_tensor(x_min_filtered_np, dtype=dtype, device=device)

    A_max = torch.eye(n_finite, dtype=dtype, device=device)
    b_max = torch.as_tensor(x_max_filtered_np, dtype=dtype, device=device)

    A = torch.cat([A_min, A_max], dim=0)   # (C, n_finite)
    b = torch.cat([b_min, b_max], dim=0)   # (C,)

    A_hor = A.unsqueeze(0).repeat(horizon, 1, 1)  # (N, C, n_finite)
    b_hor = b.unsqueeze(0).repeat(horizon, 1)     # (N, C)

    return A_hor, b_hor

def st_limit_constraint_batched(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Horizon-wise state constraint. Represent in the form of Ax <= b
    Args:
        x: (N, 7) or (B, N, 7)
        A: (N, C, 2) since only 2 among 7 states are involved in this constraints
        b: (N, C)

    Returns:
        (N,) or (B, N): positive = violation at each timestep.
    """
    A = A.to(dtype=x.dtype, device=x.device)
    b = b.to(dtype=x.dtype, device=x.device)

    if x.ndim == 2:
        x_filtered = x[:, 2:4]
        lhs = torch.einsum("ti,tci->tc", x_filtered, A)
        violation = torch.clamp(lhs - b, min=0.0)
        return torch.linalg.norm(violation, dim=-1)

    if x.ndim == 3:
        x_filtered = x[:, :, 2:4]
        lhs = torch.einsum("bti,tci->btc", x_filtered, A)
        violation = torch.clamp(lhs - b.unsqueeze(0), min=0.0)
        return torch.linalg.norm(violation, dim=-1)

    raise ValueError(
        f"Expected x to have shape (N, 7) or (B, N, 7), got {tuple(x.shape)}"
    )
