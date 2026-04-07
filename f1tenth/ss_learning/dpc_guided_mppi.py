"""
DPC-guided MPPI.
----
At each control step:
1) Solve DPC once to get a full horizon control sequence u_dpc.
2) Utilize u_dpc as MPPI's warm-start nominal sequence (a_opt).
3) Solve MPPI from that warm-start.
4) Execute the selected action (default: MPPI action).

In order to test whether a learned DPC policy helps MPPI converge faster / better.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
import jax.numpy as jnp
import gymnasium as gym

import f1tenth_gym
from f1tenth_gym.envs import F110Env

from f1tenth_planning.control.config.controller_config import dynamic_ap_mppi_config
from f1tenth_planning.control.config.dynamics_config import f1tenth_params
from f1tenth_planning.control.dynamics_models.dynamic_model import DynamicBicycleModel

# Local package path setup so `from dpc...` and `from mppi...` work when script is run directly.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dpc.DPC_config import create_DPC_config
from dpc.DPC_planner import DPC_Planner
from dpc.DPC_solver import DPC_Solver
from dpc.discretizers import rk4_discretization_torch
from policy import NeuralControlPolicy, PolicyBounds
from dpc.constraints import make_state_min_constraint_batched, make_state_max_constraint_batched

from mppi.Constrained_MPPI import Const_MPPI_Planner
from mppi.MPPI_solver import Const_MPPI_Solver
from mppi.constraints import make_state_min_constraint, make_state_max_constraint
from utils import *

def parse_args():
    """CLI options for guided-MPPI experiments."""
    parser = argparse.ArgumentParser(description="DPC-guided AP-MPPI")
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=10000)

    # Paths to configs/checkpoint.
    parser.add_argument("--dpc-config", type=str, default="dpc_config.yaml")
    parser.add_argument("--mppi-config", type=str, default="mppi_config.yaml")
    return parser.parse_args()


def reset_env_to_start(env, waypoints_track):
    """Reset ego vehicle to first raceline pose for reproducible starts."""
    poses = np.array(
        [[
            waypoints_track.raceline.xs[0],
            waypoints_track.raceline.ys[0],
            waypoints_track.raceline.yaws[0],
        ]]
    )
    obs, info = env.reset(options={"poses": poses})
    env.render()
    return obs, info

def build_shared_bounds(params):
    """Build consistent physical bounds used by both DPC and MPPI configs."""
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

def add_render(env, dpc_planner, mppi_planner):
    """
    Render both predicted trajectories:
    - DPC prediction in green
    - MPPI prediction in red
    """
    env.unwrapped.add_render_callback(dpc_planner.render_waypoints)

    dpc_line = {"obj": None}
    mppi_line = {"obj": None}
    dpc_points = {"obj": None}
    mppi_points = {"obj": None}

    def render_dual_pred(e):
        if getattr(dpc_planner, "x_pred", None) is not None:
            dpc_xy = np.asarray(dpc_planner.x_pred[:2, :].T)
            if dpc_line["obj"] is None:
                dpc_line["obj"] = e.render_closed_lines(dpc_xy, color=(0, 220, 0), size=4)
            else:
                dpc_line["obj"].setData(dpc_xy)

            if dpc_points["obj"] is None:
                dpc_points["obj"] = e.render_points(dpc_xy, color=(0, 255, 0), size=6)
            else:
                dpc_points["obj"].setData(dpc_xy)

        if getattr(mppi_planner, "x_pred", None) is not None:
            mppi_xy = np.asarray(mppi_planner.x_pred[:2, :].T)
            if mppi_line["obj"] is None:
                mppi_line["obj"] = e.render_closed_lines(mppi_xy, color=(220, 40, 40), size=4)
            else:
                mppi_line["obj"].setData(mppi_xy)

            if mppi_points["obj"] is None:
                mppi_points["obj"] = e.render_points(mppi_xy, color=(255, 120, 120), size=6)
            else:
                mppi_points["obj"].setData(mppi_xy)

    env.unwrapped.add_render_callback(render_dual_pred)

def dpc_guided_warm_start(mppi_solver, dpc_planner):
    """
    Replace (or blend) MPPI nominal action sequence with DPC sequence.

    Parameters
    ----------
    mppi_solver : Const_MPPI_Solver
        MPPI solver whose internal control_params = (a_opt, a_cov).
    dpc_planner : DPC_Planner
        Planner that just solved and has `u_pred` shaped (nu, N).
    Returns
    -------
    np.ndarray
        Warm-start sequence actually injected (shape (N, nu)).
    """
    # DPC planner stores controls as (nu, N). MPPI nominal expects (N, nu).
    u_dpc = np.asarray(dpc_planner.u_pred, dtype=np.float32).T

    # Read current MPPI parameters.
    a_opt_prev, a_cov_prev = mppi_solver.control_params

    # Convert DPC seed to the same backend/device type (jax array).
    # jax.numpy is already used inside solver, so this cast keeps types aligned.
    u_warmstart = jnp.array(u_dpc)

    # Respect control bounds before injection.
    u_warmstart = jnp.clip(u_warmstart, mppi_solver.config.u_min, mppi_solver.config.u_max)
    # Keep covariance as-is; only replace nominal controls.
    mppi_solver.control_params = (u_warmstart, a_cov_prev)
    return np.asarray(u_warmstart)

def main():
    args = parse_args()
    do_render = not args.no_render

    current_dir = os.path.dirname(__file__)
    dpc_yaml_path = os.path.join(current_dir, args.dpc_config)
    mppi_yaml_path = os.path.join(current_dir, args.mppi_config)

    dpc_policy_path = os.path.abspath(
        # os.path.join(current_dir, "dpc_policy.pt")
        os.path.join(current_dir, "../saved_model/dpc_policy.pt")
    )

    # Torch device is for DPC inference only.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    env: F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": "Spielberg_blank", # Keep this map 
            "num_agents": 1,
            "control_input": ["accl", "steering_speed"],
            "observation_config": {"type": "dynamic_state"},
            "params": F110Env.f1tenth_vehicle_params(),
        },
        render_mode="human" if do_render else None,
    )
    params = f1tenth_params()
    waypoints_track = env.unwrapped.track
    model = DynamicBicycleModel(params)

    # -------------------------
    # Build DPC planner/solver
    # -------------------------
    dpc_config = load_dpc_config(dpc_yaml_path, params)
    bounds = PolicyBounds(
        delta_v_min=params.MIN_DSTEER,
        delta_v_max=params.MAX_DSTEER,
        a_min=params.MIN_ACCEL,
        a_max=params.MAX_ACCEL,
    )
    policy = NeuralControlPolicy(
        in_dim=10,
        hidden_dim=64,
        num_hidden_layers=5,
        bounds=bounds,
        activation="gelu",
        dropout=0.0,
    )

    dpc_solver = DPC_Solver(
        config=dpc_config,
        model=model,
        policy=policy,
        discretizer=rk4_discretization_torch,
        device=device,
    )

    dpc_solver.load_trained_policy(dpc_policy_path)
    print(f"Loaded trained DPC policy from {dpc_policy_path}")

    dpc_planner = DPC_Planner(
        track=waypoints_track,
        params=params,
        config=dpc_config,
        model=model,
        solver=dpc_solver,
    )

    # --------------------------
    # Build MPPI planner
    # --------------------------
    mppi_config = load_mppi_config(mppi_yaml_path, params)
    # Make sure objective weight are consistent
    mppi_config.Q = np.asarray(dpc_config.Q)
    mppi_config.R = np.asarray(dpc_config.R)
    mppi_config.Rd = np.asarray(dpc_config.Rd)
    mppi_config.P = np.asarray(dpc_config.P)   
    
    mppi_solver = Const_MPPI_Solver(mppi_config, model)
    mppi_planner = Const_MPPI_Planner(
        track=waypoints_track,
        params=params,
        config=mppi_config,
        model=model,
        solver=mppi_solver,
    )

    if do_render:
        add_render(env, dpc_planner, mppi_planner)

    obs, _ = reset_env_to_start(env, waypoints_track)

    print("Initialization is done!")

    done = False
    step_count = 0
    laptime = 0.0

    while step_count < args.steps and not done:
        # DPC solve first (provides guidance sequence).
        u_dpc = np.asarray(dpc_planner.plan(obs["agent_0"]), dtype=np.float32)
        # 2) Inject DPC horizon as MPPI warm start.
        u_warmstart = dpc_guided_warm_start(
            mppi_solver=mppi_solver,
            dpc_planner=dpc_planner,
        )

        # 3) Solve MPPI after warm-start injection.
        u_mppi = np.asarray(mppi_planner.plan(obs["agent_0"]), dtype=np.float32)

        if args.verbose:
            print(
                f"step {step_count:04d} | "
                f"u_dpc=[{u_dpc[0]: .3f}, {u_dpc[1]: .3f}] "
                f"u_mppi=[{u_mppi[0]: .3f}, {u_mppi[1]: .3f}] "
                f"seed_u0=[{u_warmstart[0,0]: .3f}, {u_warmstart[0,1]: .3f}]"
            )

        obs, timestep, terminated, truncated, _ = env.step(
            np.array([[u_mppi[0], u_mppi[1]]], dtype=np.float32)
        )
        done = bool(terminated or truncated)
        laptime += float(timestep)
        if do_render:
            env.render()
        step_count += 1

    print(f"Finished: steps={step_count}, laptime={laptime:.2f}s, done={done}")


if __name__ == "__main__":
    main()
