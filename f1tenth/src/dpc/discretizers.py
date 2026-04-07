import torch

# Torch version of RK4 discretization for use in differentiable MPC solver with PyTorch autograd.
def rk4_discretization_torch(func, x, u, p, dt):
    k1 = func(x, u, p)
    k2 = func(x + 0.5 * dt * k1, u, p)
    k3 = func(x + 0.5 * dt * k2, u, p)
    k4 = func(x + dt * k3, u, p)
    return x + (dt * (k1 + 2*k2 + 2*k3 + k4)) / 6.0