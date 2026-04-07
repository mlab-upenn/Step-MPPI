from dataclasses import dataclass, field
import numpy as np
from f1tenth_planning.control.config.controller_config import MPPIConfig

def create_DMPPI_config(cfg) -> MPPIConfig:
    planner_cfg = cfg.get("planner", {})
    N = planner_cfg.get("N", 20)
    dt = planner_cfg.get("dt", 0.05)
    nx = 7
    nu = 2    
    # State tracking cost, for now tracking pose only 
    Q = np.diag([1e1, 1e1, 0.0, 0.0, 1e-1, 0.0, 0.0])
    # Control effort cost 
    R = np.diag([1e-1, 1e-1]) 
    Rd = np.diag([0.0, 0.0]) # Control rate cost
    P = Q # Terminal cost same as state cost
    # MPPI specific parameters
    n_iterations = planner_cfg.get("n_iterations", 5)
    n_samples = planner_cfg.get("n_samples", 1024)
    temperature = planner_cfg.get("temperature", 1.0)
    u_std = planner_cfg.get("u_std", 0.5)

    config = MPPIConfig(nx=nx, nu=nu, N=N, dt=dt,
        Q=Q, R=R, Rd=Rd, P=P,
        n_iterations=n_iterations, n_samples=n_samples,
        temperature = temperature, u_std=u_std,
        adaptive_covariance=True, scan=False,
        )

    config.constraints = [] 
    config.lamdas = [] 
        
    return config
