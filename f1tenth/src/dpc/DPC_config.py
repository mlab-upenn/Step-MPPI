from dataclasses import dataclass, field
import numpy as np
from f1tenth_planning.control.config.controller_config import MPCConfig

def create_DPC_config(
        cfg,
    ) -> MPCConfig:
    N = cfg["planner"]["N"]
    dt = cfg["planner"]["dt"]
    nx = 7
    nu = 2
    # State tracking cost, for now tracking pose only 
    Q = np.diag([1e1, 1e1, 0.0, 0.0, 1e-1, 0.0, 0.0])
    # Control effort cost 
    R = np.diag([1e-1, 1e-1]) 
    Rd = np.diag([0.0, 0.0]) # Control rate cost
    P = Q # Terminal cost same as state cost

    config = MPCConfig(N=N, dt=dt, nx=nx, nu=nu,
        Q=Q, R=R, Rd=Rd, P=P,
    )
    # config.constraints = constraints        
    config.lamdas = [] 
    return config
