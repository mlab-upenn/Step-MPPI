import time
import os
import gymnasium as gym
import numpy as np
import yaml
import argparse
from copy import deepcopy
import jax 
import jax.numpy as jnp
import f1tenth_gym
import f1tenth_planning

from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control import NonlinearDynamicAPMPPIPlanner
from f1tenth_planning.control.config.controller_config import dynamic_ap_mppi_config
from f1tenth_planning.control.config.dynamics_config import f1tenth_params
from mppi.constraints import make_state_min_constraint, make_state_max_constraint
from mppi.Constrained_MPPI import Const_MPPI_Planner
from utils import *

f1tenth_gym_path = os.path.dirname(os.path.dirname(f1tenth_gym.__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Data collection using MPPI planner.")
    parser.add_argument(
        "--save-data",
        dest="save_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save collected data to file (use --no-save-data to disable).",
    )
    
    return parser.parse_args()

def save_dataset_npz(DATASET, path="dagger_dataset.npz"):
    x0 = np.stack([d["x0"] for d in DATASET]).astype(np.float32)  # (K,7)
    mu_tilde = np.stack([d["mu_tilde"] for d in DATASET]).astype(np.float32)   # (K,20,2)
    r_samples = np.stack([d["r_samples"] for d in DATASET]).astype(np.float32) # (K,256)
    mu_star  = np.stack([d["mu_star"] for d in DATASET]).astype(np.float32)    # (K,20,2)

    np.savez_compressed(path, x0=x0, mu_tilde=mu_tilde, r_samples=r_samples, mu_star=mu_star)
    print(f"Saved {len(DATASET)} samples to {path}")

def main():
    """
    AP-MPPI example with state limit constraints and safety constraints.
    For data collection 
    """
    args = parse_args()
    save_data = args.save_data

    # Load config
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "mppi_config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

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
        render_mode="human",
    )

    # Load track waypoints
    waypoints_track = env.unwrapped.track
    # Get vehicle parameters
    params = f1tenth_params()
    
    # Build state constraints from vehicle parameters
    # x = [x, y, delta, v, yaw, yaw_rate, beta]
    x_min_constrained = np.array([
        -np.inf,           # x position: unconstrained
        -np.inf,           # y position: unconstrained  
        params.MIN_STEER,  # steering angle
        params.MIN_SPEED,  # velocity
        -np.inf,           # yaw: unconstrained
        -np.inf,           # yaw_rate: unconstrained
        -np.inf,           # beta: unconstrained
    ])
    x_max_constrained = np.array([
        np.inf,            # x position: unconstrained
        np.inf,            # y position: unconstrained
        params.MAX_STEER,  # steering angle
        params.MAX_SPEED,  # for testing, set max speed to 5 m/s
        np.inf,            # yaw: unconstrained
        np.inf,            # yaw_rate: unconstrained
        np.inf,            # beta: unconstrained
    ])

    # TODO: Include safety constraints
    constraints = [
        make_state_min_constraint(x_min_constrained),
        make_state_max_constraint(x_max_constrained),
    ]

    # Lambda range must be (n_constraints, 2) - one range per constraint
    # Higher lambda values = stronger constraint enforcement
    # Use high values (0-1000) for strict constraint satisfaction
    n_constraints = len(constraints)
    lambdas_sample_range = np.array([[0.0, 1000.0]] * n_constraints)

    # Create config with custom N, dt and constraints
    config = dynamic_ap_mppi_config(
        constraints=constraints,
        n_lambdas=32,  # More lambda samples for better constraint handling
        lambdas_sample_range=lambdas_sample_range,
    )
    config.dt = cfg["planner"]["dt"]
    config.N = cfg["planner"]["N"]
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
        params=params,
        config=config,
        # Operational speed limit for reference trajectory (separate from physical limits)
        ref_velocity_bounds=(params.MIN_SPEED, params.MAX_SPEED),
    )

    # Construct the learner MPPI
    config_learner = deepcopy(config)
    config_learner.n_samples = cfg["planner"]["learner"]["n_samples"]  # Fewer samples for learner MPPI
    config_learner.n_iterations = cfg["planner"]["learner"]["n_iterations"]  # Fewer iterations for learner MPPI
    
    student = Const_MPPI_Planner(
        track=waypoints_track,
        params=params,
        config=config_learner,
        # Operational speed limit for reference trajectory (separate from physical limits)
        ref_velocity_bounds=(params.MIN_SPEED, params.MAX_SPEED),
    )

    # Add render callbacks
    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_control_solution)

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
    env.render()

    # Print constraint info
    n_constraints = planner.solver.config.n_constraints
    print(f"\n=== MPPI Configuration ===")
    print(f"Number of constraints: {n_constraints}")
    print(f"Number of lambda samples: {planner.solver.config.n_lambdas}")
    print(f"Lambda sample range:\n{planner.solver.config.lambdas_sample_range}")
    print(f"Horizon: {planner.solver.config.N}")
    print(f"dt: {planner.solver.config.dt}")
    print("=" * 30 + "\n")

    # Initialize dataset list 
    DATASET = []
    is_first_step = True  # Flag to indicate the first step for dataset recording

    laptime = 0.0
    # start = time.time()
    step_count = 0
    step_max = int(1e4)
    autosave_every = int(1e3)  # Save dataset every
    data_filename = "dataset.npz"
    
    while not done: # step_count < step_max and not done:
        current_state = obs_dict_to_array(obs["agent_0"])

        # Plan next control action using the expert planner
        t0 = time.perf_counter()
        # warm start from previous solution
        warmstart_solution = planner.solver.warm_start()
        planner.solver.control_params = deepcopy(warmstart_solution)

        # solve MPPI
        steerv, accl = planner.plan(obs["agent_0"])
        expert_solution = planner.solver.control_params  # Get the optimal control sequence
        dt = time.perf_counter() - t0
        # print(f"Computation time: {dt:.6f} s")
        print(steerv, accl)
        # Compute the same using the student planner
        student.solver.control_params = deepcopy(warmstart_solution)
        # solve MPPI
        _ = student.plan(obs["agent_0"])
        # Learner rewards for the sampled trajectories 
        learner_sampled_costs = - student.solver.samples[-1]  # Get the cost (-reward) of the sampled trajectories

        # print("Previous solution:\n", previous_solution[0][:3,:].T)
        # print("Warmstart solution\n:", warmstart_solution[0][:3,:].T)
        # print("Expert solution:\n", expert_solution[0][:3,:].T)
        # print("*"*40)

        # Construct the new data as a dictionary
        if save_data:
            new_data = {
                "x0": current_state,
                "mu_tilde": np.array(jax.device_get(warmstart_solution[0])),
                "r_samples": np.array(jax.device_get(learner_sampled_costs)),
                "mu_star": np.array(jax.device_get(expert_solution[0])),
            }

            # Append the dataset 
            DATASET.append(new_data)

        if save_data and (step_count + 1) % autosave_every == 0:
            print(f"[autosave] Saving dataset at step {step_count + 1}!")
            save_dataset_npz(DATASET, data_filename)

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
        env.render()

        # Print status
        # speed = obs["agent_0"]["linear_vel_x"]
        # print(f"speed: {speed:.2f} m/s, steer vel: {steerv:.3f}, accl: {accl:.2f}")

        step_count += 1

    # print(f"\nSim elapsed time: {laptime:.2f}s")
    # print(f"Real elapsed time: {time.time() - start:.2f}s")
    if save_data:
        print("[final save] Saving dataset!")
        save_dataset_npz(DATASET, data_filename)

if __name__ == "__main__":
    main()
