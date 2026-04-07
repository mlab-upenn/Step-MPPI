import os
import time
import csv
import argparse
import sys
from pathlib import Path
from copy import deepcopy
import gymnasium as gym
import numpy as np
import yaml
import torch
import f1tenth_gym
from f1tenth_gym.envs.track import Track, Boundary
from f1tenth_gym.envs import F110Env
from f1tenth_planning.control.config.controller_config import dynamic_ap_mppi_config
from f1tenth_planning.control.config.dynamics_config import f1tenth_params
from f1tenth_planning.control.dynamics_models.dynamic_model import DynamicBicycleModel

# Enable local package imports when running from ss_learning/ directly.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dpc.constraints import *
from dpc.discretizers import rk4_discretization_torch
from policy import NeuralDistributionPolicy, PolicyBounds
from dpc.DPC_planner import DPC_Planner
from dmppi import create_DMPPI_config, DMPPI_Solver
from policy import NeuralMPPIUpdate

f1tenth_gym_path = os.path.dirname(os.path.dirname(f1tenth_gym.__file__))
REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = REPO_ROOT / "make_plot"

def parse_args():
    parser = argparse.ArgumentParser(description="Test DPC-trained policy in F1TENTH.")
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print real-time action comparisons.",
    )
    parser.add_argument(
        "--save_data",
        dest="save_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save closed-loop trajectory to CSV.",
    )
    parser.add_argument(
        "--no-render",
        dest="no_render",
        action="store_true",
        default=False,
        help="Disable rendering for faster policy evaluation.",
    )
    parser.add_argument(
        "--policy_file",
        dest="policy_file",
        type=str,
        default=None,
        help="Path to policy checkpoint used as primary load path.",
    )
    return parser.parse_args()


def obs_dict_to_array(obs_agent: dict) -> np.ndarray:
    return np.array(
        [
            obs_agent["pose_x"],
            obs_agent["pose_y"],
            obs_agent["delta"],
            obs_agent["linear_vel_x"],
            obs_agent["pose_theta"],
            obs_agent["ang_vel_z"],
            obs_agent["beta"],
        ],
        dtype=np.float32,
    )


def save_trajectory_csv(trajectory, path=None):
    if path is None:
        path = PLOT_DIR / "neural_dmppi_trajectory.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "time",
        "x",
        "y",
        "delta",
        "v",
        "yaw",
        "yaw_rate",
        "beta",
        "steerv",
        "accl",
        "plan_time",
    ]

    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for step in trajectory:
            state = np.asarray(step["state"], dtype=np.float32)
            control = np.asarray(step.get("control", [np.nan, np.nan]), dtype=np.float32)
            writer.writerow(
                {
                    "time": float(step["time"]),
                    "x": float(state[0]),
                    "y": float(state[1]),
                    "delta": float(state[2]),
                    "v": float(state[3]),
                    "yaw": float(state[4]),
                    "yaw_rate": float(state[5]),
                    "beta": float(state[6]),
                    "steerv": float(control[0]),
                    "accl": float(control[1]),
                    "plan_time": float(step.get("plan_time", np.nan)),
                }
            )

    print(f"Saved trajectory CSV with {len(trajectory)} rows to {path}")

def reset_env_to_start(env, waypoints_track):
    poses = np.array([[
        waypoints_track.raceline.xs[0],
        waypoints_track.raceline.ys[0],
        waypoints_track.raceline.yaws[0],
    ]])
    obs, info = env.reset(options={"poses": poses})
    env.render()
    return obs, info

def main():
    """
    Differentiable MPPI testing.
    """    
    args = parse_args()
    verbose = args.verbose
    save_data = args.save_data
    do_render = not args.no_render

    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "neural_dmppi_config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    N = int(cfg["planner"]["N"])
    dt_cfg = float(cfg["planner"]["dt"])
    target_speed = cfg.get("planner", {}).get("reference_speed", None)

    # path to your trained NN (from training_dpc.py)
    primary_path = (
        os.path.abspath(args.policy_file)
        if args.policy_file is not None
        else os.path.abspath(os.path.join(current_dir, "neural_dmppi_constrained_policy.pt"))
    )
    fallback_path = os.path.abspath(os.path.join(current_dir, "../saved_model/neural_dmppi_constrained_policy.pt"))
    trained_policy_path = primary_path if os.path.exists(primary_path) else fallback_path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)    
    # create environment
    env: F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": cfg["map"],
            "num_agents": 1,
            "control_input": ["accl", "steering_speed"],
            "observation_config": {"type": "dynamic_state"},
            "params": F110Env.f1tenth_vehicle_params(),
        },
        render_mode="human" if do_render else None)

    # Load track waypoints
    waypoints_track = env.unwrapped.track

    # Load boundary info 
    boundary_file_path = os.path.join(f1tenth_gym_path, "maps", "Spielberg", "Spielberg_border_coeffs.csv")
    boundary = Boundary(boundary_file_path)

    # Get vehicle parameters
    params = f1tenth_params()

    # List of contraints
    constraints = [st_limit_constraint_batched, boundary_constraint_batched]

    # Add constraints and lambdas to config
    config = create_DMPPI_config(cfg=cfg)
    config.m_samples = cfg["planner"].get("m_samples", 64)
    config.lambdas = [1e3] * len(constraints) # same

    # Initialize the neural control policy
    bounds = PolicyBounds(
        delta_v_min=params.MIN_DSTEER,
        delta_v_max=params.MAX_DSTEER,
        a_min=params.MIN_ACCEL,
        a_max=params.MAX_ACCEL,
    )
    policy = NeuralDistributionPolicy(
        in_dim=14, # 7 state + 3 ref + 4 con coeffs
        hidden_dim=256,
        num_hidden_layers=5,
        bounds=bounds,
        activation="gelu",
        dropout=0.0,
    )
    mppi_updater = NeuralMPPIUpdate(
        nu=2, # control input dim
        K=config.m_samples,
        hidden_dim=128,
        num_hidden_layers=(2, 2, 2), # [global, cost, head]
        bounds=bounds,
    )

    # Initialize the dynamic model
    model = DynamicBicycleModel(params)

    # Initialize the DPC solver
    solver = DMPPI_Solver(
        config=config,
        model=model,
        policy=policy,
        discretizer=rk4_discretization_torch,
        updater=mppi_updater,
        device = device
    )
    solver.load_trained_policy(trained_policy_path)
    # print("Loaded trained policy from", trained_policy_path)

    # Initialize the MPPI planner with learner config (fewer samples and iterations for faster test-time planning)
    planner = DPC_Planner(track=waypoints_track, boundary=boundary,
        params=params, model=model, solver=solver)

    if target_speed is not None:
        planner.waypoints[:, 3] = float(target_speed)
        print(f"Reference speed (from yaml): {float(target_speed):.3f} m/s")
    else:
        print("Reference speed (from yaml): raceline default")

    # Render whichever planner is active in this test run.
    if do_render:
        env.unwrapped.add_render_callback(planner.render_waypoints)
        env.unwrapped.add_render_callback(planner.render_local_plan)
        env.unwrapped.add_render_callback(planner.render_control_solution)

    obs, _ = reset_env_to_start(env, waypoints_track)

    # Test
    steerv, accl = planner.plan(obs["agent_0"])
    print("steerv:", steerv, "accl:", accl)

    done = False
    step_count = 0
    laptime = 0.0
    step_max = int(1e4)
    trajectory = []

    if save_data:
        trajectory.append(
            {
                "state": obs_dict_to_array(obs["agent_0"]),
                "time": laptime,
            }
        )

    while (step_count < step_max) and (not done):
        t0 = time.perf_counter()
        steerv, accl = planner.plan(obs["agent_0"])
        dt = time.perf_counter() - t0

        # Execute action
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steerv, accl]])
        )
        
        # print(obs["agent_0"]["pose_x"], obs["agent_0"]["pose_y"], obs["agent_0"]["pose_theta"])
        done = terminated or truncated
        laptime += timestep
        if do_render:
            env.render()
        if save_data:
            trajectory.append(
                {
                    "state": obs_dict_to_array(obs["agent_0"]),
                    "control": np.array([steerv, accl], dtype=np.float32),
                    "time": laptime,
                    "plan_time": dt,
                }
            )
        # Print status
        if verbose:
            speed = obs["agent_0"]["linear_vel_x"]
            print(f"speed: {speed:.2f} m/s, steer vel: {steerv:.3f}, accl: {accl:.2f}")

        step_count += 1

    print("\n=== Test finished ===")
    if verbose:
        print(f"steps: {step_count}, laptime: {laptime:.2f}s, done={done}")
    if save_data:
        dt_ms = int(round(dt_cfg * 1000.0))
        out_csv = PLOT_DIR / f"neural_dmppi_trajectory_N{N}_dt{dt_ms}ms.csv"
        save_trajectory_csv(trajectory, path=out_csv)

    # print(f"steps: {step_count}, laptime: {laptime:.2f}s, done={done}")

if __name__ == "__main__":
    main()
