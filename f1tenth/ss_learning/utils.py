import sys
from pathlib import Path

import numpy as np
import yaml

from f1tenth_planning.control.config.controller_config import dynamic_ap_mppi_config
from f1tenth_planning.control.config.dynamics_config import DynamicsConfig

# Ensure local src/ packages are importable when scripts are run directly.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dpc.DPC_config import create_DPC_config
from dpc.constraints import (
    make_state_max_constraint_batched,
    make_state_min_constraint_batched,
)
from mppi.constraints import make_state_max_constraint, make_state_min_constraint

def _shared_bounds(params: DynamicsConfig):
    """Create shared state/control bounds from vehicle parameters."""
    x_min = np.array(
        [-np.inf, -np.inf, params.MIN_STEER, params.MIN_SPEED, -np.inf, -np.inf, -np.inf],
        dtype=np.float32,
    )
    x_max = np.array(
        [np.inf, np.inf, params.MAX_STEER, params.MAX_SPEED, np.inf, np.inf, np.inf],
        dtype=np.float32,
    )
    u_min = np.array([params.MIN_DSTEER, params.MIN_ACCEL], dtype=np.float32)
    u_max = np.array([params.MAX_DSTEER, params.MAX_ACCEL], dtype=np.float32)
    return x_min, x_max, u_min, u_max


def load_dpc_config(yaml_path: str, params: DynamicsConfig):
    """
    Build DPC config from yaml and attach runtime bounds/constraints.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dpc_config = create_DPC_config(cfg=cfg)
    x_min, x_max, u_min, u_max = _shared_bounds(params)

    dpc_config.x_min = x_min
    dpc_config.x_max = x_max
    dpc_config.u_min = u_min
    dpc_config.u_max = u_max
    dpc_config.constraints = [
        make_state_min_constraint_batched(x_min),
        make_state_max_constraint_batched(x_max),
    ]
    dpc_config.lambdas = [1e3] * len(dpc_config.constraints)
    return dpc_config


def load_mppi_config(yaml_path: str, params: DynamicsConfig):
    """
    Build AP-MPPI config from yaml and attach runtime bounds/constraints.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    x_min, x_max, u_min, u_max = _shared_bounds(params)

    mppi_constraints = [
        make_state_min_constraint(x_min),
        make_state_max_constraint(x_max),
    ]
    lambdas_sample_range = np.array([[0.0, 1000.0]] * len(mppi_constraints), dtype=np.float32)

    mppi_config = dynamic_ap_mppi_config(
        constraints=mppi_constraints,
        n_lambdas=32,
        lambdas_sample_range=lambdas_sample_range,
    )
    mppi_config.N = int(cfg["planner"]["N"])
    mppi_config.dt = float(cfg["planner"]["dt"])
    mppi_config.n_samples = int(cfg["planner"]["n_samples"])
    mppi_config.n_iterations = int(cfg["planner"]["n_iterations"])
    mppi_config.x_min = x_min
    mppi_config.x_max = x_max
    mppi_config.u_min = u_min
    mppi_config.u_max = u_max
    return mppi_config

