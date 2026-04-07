import torch
import numpy as np

def boundary_constraint_coeffs_single_step(
    coeffs: torch.Tensor,  # (B, 4): [a, b, c_left, c_right]
    d_min: float = 0.3,
):
    """
    Build one-step boundary Ax <= b.
    Returns:
        A: (B, 2, 2)
        b: (B, 2)
    """
    if coeffs.ndim != 2 or coeffs.shape[-1] != 4:
        raise ValueError(f"Expected coeffs shape (B,4), got {tuple(coeffs.shape)}")

    a = coeffs[:, 0]
    b0 = coeffs[:, 1]
    c_left = coeffs[:, 2]
    c_right = coeffs[:, 3]

    A_seg = torch.stack([a, b0], dim=-1)          # (B,2)
    A = torch.stack([A_seg, -A_seg], dim=1)       # (B,2,2)
    b = torch.stack([c_left - d_min, -c_right - d_min], dim=-1)  # (B,2)
    return A, b

def boundary_constraint_single_step(
    x: torch.Tensor,      # (B, nx)
    A: torch.Tensor,      # (B, 2, 2)
    b: torch.Tensor,      # (B, 2)
) -> torch.Tensor:
    """
    Returns one-step boundary violation magnitude: (B,)
    """
    pos = x[:, :2]  # (B,2)
    A = A.to(dtype=x.dtype, device=x.device)
    b = b.to(dtype=x.dtype, device=x.device)

    lhs = torch.einsum("bi,bci->bc", pos, A)   # (B,2)
    v = torch.clamp(lhs - b, min=0.0)
    return torch.linalg.norm(v, dim=-1)        # (B,)

def st_limit_constraint_coeffs_single_step(
    x_min: np.ndarray,
    x_max: np.ndarray,
    state_indices=(2, 3),  # delta, v
    device=None,
    dtype=torch.float32,
):
    """
    Build one-step Ax <= b for selected state components.
    Returns:
        A: (C, n_sel)
        b: (C,)
    """
    x_min_sel = np.asarray(x_min)[list(state_indices)]
    x_max_sel = np.asarray(x_max)[list(state_indices)]

    finite_mask = np.isfinite(x_min_sel) & np.isfinite(x_max_sel)
    x_min_f = x_min_sel[finite_mask].astype(np.float32)
    x_max_f = x_max_sel[finite_mask].astype(np.float32)

    n = len(x_min_f)
    A_min = -torch.eye(n, dtype=dtype, device=device)
    b_min = -torch.as_tensor(x_min_f, dtype=dtype, device=device)
    A_max = torch.eye(n, dtype=dtype, device=device)
    b_max = torch.as_tensor(x_max_f, dtype=dtype, device=device)

    A = torch.cat([A_min, A_max], dim=0)  # (2n, n)
    b = torch.cat([b_min, b_max], dim=0)  # (2n,)
    return A, b

def st_limit_constraint_single_step(
    x: torch.Tensor,      # (B, nx)
    A: torch.Tensor,      # (C, n_sel)
    b: torch.Tensor,      # (C,)
    state_indices=(2, 3),
) -> torch.Tensor:
    """
    Returns one-step violation magnitude: (B,)
    """
    x_sel = x[:, list(state_indices)]  # (B, n_sel)
    A = A.to(dtype=x.dtype, device=x.device)
    b = b.to(dtype=x.dtype, device=x.device)

    lhs = torch.einsum("bi,ci->bc", x_sel, A)   # (B, C)
    v = torch.clamp(lhs - b.unsqueeze(0), min=0.0)
    return torch.linalg.norm(v, dim=-1)         # (B,)

