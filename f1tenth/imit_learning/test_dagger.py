import os
import time
import argparse
from copy import deepcopy
import gymnasium as gym
import numpy as np
import yaml
import jax
import torch
import f1tenth_gym
from f1tenth_gym.envs import F110Env
from f1tenth_planning.control.config.controller_config import dynamic_ap_mppi_config
from f1tenth_planning.control.config.dynamics_config import f1tenth_params

from mppi.constraints import make_state_min_constraint, make_state_max_constraint
from mppi.Constrained_MPPI import Const_MPPI_Planner
from utils import *
from NN import L2O_Update_Net


def parse_args():
    parser = argparse.ArgumentParser(description="Test DAgger-trained policy in F1TENTH.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Use MPPI planner action directly instead of NN-predicted action.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print real-time action comparisons.",
    )
    return parser.parse_args()

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = L2O_Update_Net(
        H=ckpt["H"], nu=ckpt["nu"], M=ckpt["M"], hidden=ckpt["hidden"],
        dropout_p=0.1, per_step_cost_norm=True
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

def mu_any_to_Nx2(mu: np.ndarray) -> np.ndarray:
    """
    Normalize MPPI mean sequence shape to (N, 2).
    Accepts either (2, N) or (N, 2).
    """
    mu = np.asarray(mu)
    if mu.ndim != 2:
        raise ValueError(f"Expected 2D mu, got {mu.shape}")
    if mu.shape[0] == 2:
        return mu.T.astype(np.float32)  # (2, N) -> (N, 2)
    if mu.shape[1] == 2:
        return mu.astype(np.float32)    # already (N, 2)
    raise ValueError(f"Expected mu with one axis size 2, got {mu.shape}")

def first_action_from_mu_Nx2(mu_Nx2: np.ndarray) -> np.ndarray:
    """
    mu_Nx2: (N,2) = [ [steer_vel_0, accel_0], [steer_vel_1, accel_1], ... ]
    """
    return mu_Nx2[0].astype(np.float32)  # (2,)

def setup_env_and_planners(cfg):
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
        render_mode="human")

    # Load track waypoints
    waypoints_track = env.unwrapped.track
    # Get vehicle parameters
    params = f1tenth_params()

    # Build state constraints from vehicle parameters
    # x = [x, y, delta, v, yaw, yaw_rate, beta]
    x_min_constrained = np.array([
        -np.inf, -np.inf, params.MIN_STEER, params.MIN_SPEED,
        -np.inf, -np.inf, -np.inf,
    ])
    x_max_constrained = np.array([
        np.inf, np.inf, params.MAX_STEER, params.MAX_SPEED,
        np.inf, np.inf, np.inf,
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

    config = dynamic_ap_mppi_config(constraints=constraints, n_lambdas=32, lambdas_sample_range=lambdas_sample_range)
    config.dt = cfg["planner"]["dt"]
    config.N = cfg["planner"]["N"]
    config.n_samples = cfg["planner"]["learner"]["n_samples"]
    config.n_iterations = cfg["planner"]["learner"]["n_iterations"]
    config.adaptive_covariance = False

    config.x_min = np.array([-np.inf, -np.inf, params.MIN_STEER, params.MIN_SPEED, -np.inf, -np.inf, -np.inf])
    config.x_max = np.array([ np.inf,  np.inf, params.MAX_STEER,  np.inf,      np.inf,  np.inf,  np.inf])
    config.u_min = np.array([params.MIN_DSTEER, params.MIN_ACCEL])
    config.u_max = np.array([params.MAX_DSTEER, params.MAX_ACCEL])

    # Initialize the MPPI planner with learner config (fewer samples and iterations for faster test-time planning)
    planner = Const_MPPI_Planner(
        track=waypoints_track, params=params, config=config,
        ref_velocity_bounds=(params.MIN_SPEED, params.MAX_SPEED))

    # Render whichever planner is active in this test run.
    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_control_solution)

    return env, waypoints_track, planner

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
    args = parse_args()
    baseline = args.baseline
    verbose = args.verbose

    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    # path to your trained NN (from dagger_training.py)
    ckpt_path = cfg.get("dagger", {}).get("ckpt_path", "trained_NN.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the trained model
    model, ckpt = load_model(ckpt_path, device)
    print(f"Loaded model from {ckpt_path} with H={ckpt['H']}, nu={ckpt['nu']}, M={ckpt['M']}")

    env, waypoints_track, planner = setup_env_and_planners(cfg)
    obs, _ = reset_env_to_start(env, waypoints_track)

    done = False
    step_count = 0
    laptime = 0.0
    step_max = int(1e3)

    while (step_count < step_max) or (not done):
        t0 = time.perf_counter()

        # warm start solutions
        warmstart_solution = planner.solver.warm_start()
        planner.solver.control_params = deepcopy(warmstart_solution)

        # Generate MPPI samples and costs for the current observation
        planner.solver.control_params = deepcopy(warmstart_solution)
        steerv, accl = planner.plan(obs["agent_0"])
        costs_samples = jax_to_numpy(-planner.solver.samples[-1])  # (M,)

        # NN forward
        mu_tilde_raw = jax_to_numpy(warmstart_solution[0])
        mu_tilde_Nx2 = mu_any_to_Nx2(mu_tilde_raw)  # (N,2)
        if mu_tilde_Nx2.shape != (ckpt["H"], ckpt["nu"]):
            raise ValueError(
                f"mu_tilde shape {mu_tilde_Nx2.shape} does not match model (H,nu)=({ckpt['H']},{ckpt['nu']})"
            )

        with torch.no_grad():
            mu_t = torch.from_numpy(mu_tilde_Nx2.astype(np.float32)).unsqueeze(0).to(device)  # (1,N,2)
            c_t = torch.from_numpy(costs_samples.astype(np.float32)).unsqueeze(0).to(device)  # (1,M)
            mu_pred = model(mu_t, c_t).squeeze(0).cpu().numpy()                               # (N,2)

        if baseline:
            act = np.array([steerv, accl], dtype=np.float32)         
        else:
            act = first_action_from_mu_Nx2(mu_pred) 
        if verbose: 
            print("Predictive action:", first_action_from_mu_Nx2(mu_pred))
            print("Baseline action:", np.array([steerv, accl]))
            print("-"*30)   

        # Execute student action
        obs, timestep, terminated, truncated, infos = env.step(act.reshape(1, 2))
        done = terminated or truncated
        laptime += timestep
        env.render()
        dt = time.perf_counter() - t0
        step_count += 1

    print("\n=== Test finished ===")
    # print(f"steps: {step_count}, laptime: {laptime:.2f}s, done={done}")

if __name__ == "__main__":
    main()
