import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
import f1tenth_gym
from f1tenth_gym.envs import F110Env

from f1tenth_planning.control.config.controller_config import dynamic_ap_mppi_config
from f1tenth_planning.control.config.dynamics_config import f1tenth_params
from f1tenth_planning.control.dynamics_models.dynamic_model import DynamicBicycleModel

# Enable local package imports when running from ss_learning/ directly.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dpc.DPC_config import create_DPC_config
from dpc.discretizers import rk4_discretization_torch
from policy import NeuralControlPolicy, PolicyBounds
from dpc.DPC_planner import DPC_Planner
from dpc.DPC_solver import DPC_Solver
from dpc.constraints import *

from mppi.Constrained_MPPI import Const_MPPI_Planner
from mppi.MPPI_solver import Const_MPPI_Solver
from mppi.constraints import make_state_min_constraint, make_state_max_constraint
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DPC and AP-MPPI solutions under the same setup"
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--no-render", action="store_true", default=False)
    parser.add_argument(
        "--execute",
        choices=["dpc", "mppi"],
        default="dpc",
        help="Which controller action to execute in the environment.",
    )
    parser.add_argument("--steps", type=int, default=10000)
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

def _print_cfg(name, cfg):
    import numpy as np

    keys = ["N", "dt", "nx", "nu", "n_samples", "n_iterations", "n_lambdas", "temperature", "damping", "u_std"]
    print(f"\n=== {name} ===")
    for k in keys:
        if hasattr(cfg, k):
            print(f"{k}: {getattr(cfg, k)}")

    arr_keys = ["Q", "R", "P", "Rd", "x_min", "x_max", "u_min", "u_max"]
    for k in arr_keys:
        if hasattr(cfg, k):
            v = np.asarray(getattr(cfg, k))
            print(f"{k}:\n{v}")


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

def main():
    args = parse_args()
    do_render = not args.no_render

    current_dir = os.path.dirname(__file__)
    trained_policy_path = os.path.abspath(
        # os.path.join(current_dir, "dpc_policy.pt")
        os.path.join(current_dir, "../saved_model/dpc_policy.pt")
    )

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

    waypoints_track = env.unwrapped.track
    params = f1tenth_params()
    model = DynamicBicycleModel(params)

    # =========== Construct DPC ===========
    dpc_config = load_dpc_config(
        os.path.join(current_dir, "dpc_config.yaml"), params
        )

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
    dpc_solver.load_trained_policy(trained_policy_path)
    print(f"Loaded trained DPC policy from {trained_policy_path}")

    dpc_planner = DPC_Planner(
        track=waypoints_track,
        params=params,
        config=dpc_config,
        model=model,
        solver=dpc_solver,
    )

    # =========== Construct MPPI ===========
    mppi_config = load_mppi_config(
        yaml_path=os.path.join(current_dir, "mppi_config.yaml"),
        params=params,
    )
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

    # Make sure the two MPC formulation are the same
    # _print_cfg("dpc_config", dpc_config)
    # _print_cfg("mppi_config", mppi_config)

    if do_render:
        add_render(env, dpc_planner, mppi_planner)

    obs, _ = reset_env_to_start(env, waypoints_track)

    if args.execute == "dpc":
        print("Using trained DPC policy")
    elif args.execute == "mppi":
        print("Using MPPI policy")

    done = False
    step_count = 0
    laptime = 0.0

    while step_count < args.steps and not done:
        u_dpc = np.asarray(dpc_planner.plan(obs["agent_0"]), dtype=np.float32)
        u_mppi = np.asarray(mppi_planner.plan(obs["agent_0"]), dtype=np.float32)

        if args.execute == "dpc":
            u_exec = u_dpc
        elif args.execute == "mppi":
            u_exec = u_mppi

        if args.verbose:
            print(
                f"step {step_count:04d} | "
                f"u_dpc=[{u_dpc[0]: .3f}, {u_dpc[1]: .3f}] "
                f"u_mppi=[{u_mppi[0]: .3f}, {u_mppi[1]: .3f}] "
            )

        obs, timestep, terminated, truncated, _ = env.step(np.array([[u_exec[0], u_exec[1]]]))
        done = bool(terminated or truncated)
        laptime += float(timestep)
        if do_render:
            env.render()
        step_count += 1

if __name__ == "__main__":
    main()
