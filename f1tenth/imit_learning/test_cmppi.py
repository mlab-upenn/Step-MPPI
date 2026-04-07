import time
import os
import csv
from pathlib import Path
import gymnasium as gym
import numpy as np
import yaml
import argparse
import cv2
from copy import deepcopy
import jax 
import jax.numpy as jnp
import f1tenth_gym
import f1tenth_planning

from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.track import Track, Boundary
from f1tenth_planning.control import NonlinearDynamicAPMPPIPlanner
from f1tenth_planning.control.config.controller_config import dynamic_ap_mppi_config
from f1tenth_planning.control.config.dynamics_config import f1tenth_params
from mppi.constraints import *
from mppi.Constrained_MPPI import Const_MPPI_Planner
from utils import *

f1tenth_gym_path = os.path.dirname(os.path.dirname(f1tenth_gym.__file__))
REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = REPO_ROOT / "make_plot"

def parse_args():
    parser = argparse.ArgumentParser(description="Data collection using MPPI planner.")
    parser.add_argument(
        "--save_data",
        dest="save_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save collected data to file.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose logging for debugging.",
    )
    parser.add_argument(
        "--target_speed",
        dest="target_speed",
        type=float,
        default=None,
        help="Override raceline reference speed with a constant target speed in m/s.",
    )
    parser.add_argument(
        "--save_video",
        dest="save_video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save a video of the rollout to file.",
    )
    parser.add_argument(
        "--video_path",
        dest="video_path",
        type=str,
        default=None,
        help="Output path for the rollout video. Defaults to make_plot/cmppi_rollout_N{N}_dt{dt_ms}ms.mp4.",
    )

    return parser.parse_args()

def save_trajectory_npz(trajectory, path=None):
    if path is None:
        path = PLOT_DIR / "cmppi_trajectory.npz"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    states = np.stack([step["state"] for step in trajectory]).astype(np.float32)
    timestamps = np.asarray([step["time"] for step in trajectory], dtype=np.float32)

    data = {
        "states": states,
        "time": timestamps,
    }

    if any("control" in step for step in trajectory):
        data["controls"] = np.stack(
            [step["control"] for step in trajectory if "control" in step]
        ).astype(np.float32)
    if any("plan_time" in step for step in trajectory):
        data["plan_time"] = np.asarray(
            [step["plan_time"] for step in trajectory if "plan_time" in step],
            dtype=np.float32,
        )

    np.savez_compressed(path, **data)
    print(f"Saved trajectory with {len(trajectory)} states to {path}")

def save_trajectory_csv(trajectory, path=None):
    if path is None:
        path = PLOT_DIR / "cmppi_trajectory.csv"
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


def init_video_writer(path, fps, frame_shape):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer, path


def write_video_frame(writer, frame):
    # OpenCV expects BGR ordering.
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def main():
    """
    AP-MPPI example with state limit constraints and safety constraints.
    """
    args = parse_args()
    save_data = args.save_data
    verbose = args.verbose
    target_speed = args.target_speed
    save_video = args.save_video

    # Load config
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "mppi_config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    # important numbers
    N = int(cfg["planner"]["N"])
    dt_cfg = float(cfg["planner"]["dt"])
    dt_ms = int(round(dt_cfg * 1000.0))
    render_mode = "rgb_array" if save_video else "human"

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
        render_mode=render_mode,
    )

    # Load track waypoints
    waypoints_track = env.unwrapped.track
    boundary_file_path = os.path.join(f1tenth_gym_path, "maps", "Spielberg", "Spielberg_border_coeffs.csv")
    boundary = Boundary(boundary_file_path)
    
    # Get vehicle parameters
    params = f1tenth_params()
    
    constraints = [st_limit_constraint, boundary_constraint]
    # constraints = [st_limit_constraint]

    # Lambda range must be (n_constraints, 2) - one range per constraint
    # Higher lambda values = stronger constraint enforcement
    # Use high values (0-1000) for strict constraint satisfaction
    n_constraints = len(constraints)
    lambdas_sample_range = np.array([[1.0, 1e4]] * n_constraints)
    # lambdas_sample_range = np.array([[0.0, 1000.0], [0.0, 1e5]])
    # Create config with custom N, dt and constraints
    config = dynamic_ap_mppi_config(
        constraints=constraints, # 
        n_lambdas=32,  # More lambda samples for better constraint handling
        lambdas_sample_range=lambdas_sample_range,
    )
    config.dt = dt_cfg
    config.N = N
    config.n_samples = cfg["planner"]["expert"]["n_samples"]  # Number of samples in expert MPPI
    config.n_iterations = cfg["planner"]["expert"]["n_iterations"]  # More iterations to converge
    config.adaptive_covariance = False

    # Set physical state/control bounds on the config (used for MPPI rollout clipping)
    # For velocity, use +inf so constraint violations are visible in rollouts
    config.x_min = np.array([
        -np.inf, -np.inf, params.MIN_STEER, params.MIN_SPEED,
        -np.inf, -np.inf, -np.inf,
    ])
    config.x_max = np.array([
        np.inf, np.inf, params.MAX_STEER, np.inf,  # velocity: +inf so constraints work
        np.inf, np.inf, np.inf,
    ])
    config.u_min = np.array([params.MIN_DSTEER, params.MIN_ACCEL])
    config.u_max = np.array([params.MAX_DSTEER, params.MAX_ACCEL])

    # create planner, or exactly the expert MPPI solver
    planner = Const_MPPI_Planner(
        track=waypoints_track,
        boundary=boundary,
        params=params,
        config=config,
    )

    # Optionally modify reference speed for test-time behavior.
    if target_speed is not None:
        planner.waypoints[:, 3] = float(target_speed)
        speed_mode = f"constant {float(target_speed):.3f} m/s"
    else:
        speed_mode = "raceline default"

    # Show only the track raceline in the renderer.
    env.unwrapped.add_render_callback(waypoints_track.raceline.render_waypoints)

    # Reset environment at start of track
    poses = np.array(
        [
            [
                waypoints_track.raceline.xs[0],
                waypoints_track.raceline.ys[0],
                waypoints_track.raceline.yaws[0],
            ]
        ]
    )
    obs, info = env.reset(options={"poses": poses})
    done = False
    video_writer = None
    video_path = None
    first_frame = env.render()

    if save_video:
        if first_frame is None:
            raise RuntimeError("Expected env.render() to return a frame in rgb_array mode.")
        default_video_path = PLOT_DIR / f"cmppi_rollout_N{N}_dt{dt_ms}ms.mp4"
        video_writer, video_path = init_video_writer(
            args.video_path or default_video_path,
            fps=max(1, int(round(1.0 / dt_cfg))),
            frame_shape=first_frame.shape,
        )
        write_video_frame(video_writer, first_frame)

    # Print constraint info
    n_constraints = planner.solver.config.n_constraints
    print(f"\n=== MPPI Configuration ===")
    print(f"Number of constraints: {n_constraints}")
    print(f"Number of lambda samples: {planner.solver.config.n_lambdas}")
    print(f"Lambda sample range:\n{planner.solver.config.lambdas_sample_range}")
    print(f"Horizon: {planner.solver.config.N}")
    print(f"dt: {planner.solver.config.dt}")
    print(f"Reference speed: {speed_mode}")
    print("=" * 30 + "\n")

    is_first_step = True  # Flag to indicate the first step for dataset recording
    trajectory = []

    laptime = 0.0
    # start = time.time()
    step_count = 0
    step_max = int(1e4)

    # Solve the problem once to warm start
    steerv, accl = planner.plan(obs["agent_0"])

    if save_data:
        trajectory.append(
            {
                "state": obs_dict_to_array(obs["agent_0"]),
                "time": laptime,
            }
        )

    while not done: # step_count < step_max and not done:
        # Plan next control action using the expert planner
        t0 = time.perf_counter()
        # warm start from previous solution
        warmstart_solution = planner.solver.warm_start()
        planner.solver.control_params = deepcopy(warmstart_solution)

        # solve MPPI
        steerv, accl = planner.plan(obs["agent_0"])
        plan_dt = time.perf_counter() - t0
        # print(f"Computation time: {dt:.6f} s")

        # Print the optimal solution
        if not is_first_step:
            pass
        else:
            is_first_step = False  # Unset the flag after the first step

        # Step environment
        obs, timestep, terminated, truncated, infos = env.step(
            np.array([[steerv, accl]])
        )
        done = terminated or truncated
        laptime += timestep
        frame = env.render()
        if save_video and frame is not None:
            write_video_frame(video_writer, frame)

        if save_data:
            trajectory.append(
                {
                    "state": obs_dict_to_array(obs["agent_0"]),
                    "control": np.array([steerv, accl], dtype=np.float32),
                    "time": laptime,
                    "plan_time": plan_dt,
                }
            )

        # Print status
        if verbose:
            speed = obs["agent_0"]["linear_vel_x"]
            print(f"speed: {speed:.2f} m/s, steer vel: {steerv:.3f}, accl: {accl:.2f}")

        step_count += 1

    print(f"\nSim elapsed time: {laptime:.2f}s")
    # print(f"Real elapsed time: {time.time() - start:.2f}s")
    if video_writer is not None:
        video_writer.release()
        print(f"Saved rollout video to {video_path}")
    env.close()
    if save_data:
        # save_trajectory_npz(trajectory)
        out_csv = PLOT_DIR / f"cmppi_trajectory_N{N}_dt{dt_ms}ms.csv"
        save_trajectory_csv(trajectory, path=out_csv)


if __name__ == "__main__":
    main()
