import os
import time
import random
from copy import deepcopy

import gymnasium as gym
import numpy as np
import yaml
import jax

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import f1tenth_gym
from f1tenth_gym.envs import F110Env
from f1tenth_planning.control.config.controller_config import dynamic_ap_mppi_config
from f1tenth_planning.control.config.dynamics_config import f1tenth_params

from mppi.constraints import make_state_min_constraint, make_state_max_constraint
from mppi.Constrained_MPPI import Const_MPPI_Planner
from utils import *

from data_collection import save_dataset_npz
from NN import L2O_Update_Net
from simple_training import model_train, load_dataset_npz, ImitationDataset


# Compute the DAgger mixing coefficient for iteration k.
def beta_schedule(k: int, beta_0: float, decay: float, beta_min: float = 0.0) -> float:
    return max(beta_min, beta_0 * (decay ** k))

# Load a saved checkpoint and rebuild the L2O model on the requested device.
def load_model_checkpoint(ckpt_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = L2O_Update_Net(H=ckpt["H"], nu=ckpt["nu"], M=ckpt["M"], hidden=ckpt["hidden"],
                           dropout_p=0.1, per_step_cost_norm=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

# Setup: env + planners
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

    # Create config with custom N, dt and constraints
    config = dynamic_ap_mppi_config(
        constraints=constraints,
        n_lambdas=32,
        lambdas_sample_range=lambdas_sample_range,
    )
    config.dt = cfg["planner"]["dt"]
    config.N = cfg["planner"]["N"]
    config.n_samples = cfg["planner"]["expert"]["n_samples"]
    config.n_iterations = cfg["planner"]["expert"]["n_iterations"]  # More iterations to converge
    config.adaptive_covariance = False

    # Set physical state/control bounds on the config (used for MPPI rollout clipping)
    # For velocity, use +inf so constraint violations are visible in rollouts    config.x_min = np.array([-np.inf, -np.inf, params.MIN_STEER, params.MIN_SPEED, -np.inf, -np.inf, -np.inf])
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
        track=waypoints_track, params=params, config=config,
        ref_velocity_bounds=(params.MIN_SPEED, params.MAX_SPEED),
    )

    # Construct the learner MPPI
    config_learner = deepcopy(config)
    config_learner.n_samples = cfg["planner"]["learner"]["n_samples"]
    config_learner.n_iterations = cfg["planner"]["learner"]["n_iterations"]  # Fewer iterations for learner MPPI

    student_mppi = Const_MPPI_Planner(
        track=waypoints_track, params=params, config=config_learner,
        ref_velocity_bounds=(params.MIN_SPEED, params.MAX_SPEED),
    )

    # Add render callbacks
    env.unwrapped.add_render_callback(planner.render_waypoints)
    env.unwrapped.add_render_callback(planner.render_local_plan)
    env.unwrapped.add_render_callback(planner.render_control_solution)    

    return env, waypoints_track, params, planner, student_mppi

def reset_env_to_start(env, waypoints_track):
    poses = np.array([[
        waypoints_track.raceline.xs[0],
        waypoints_track.raceline.ys[0],
        waypoints_track.raceline.yaws[0],
    ]])
    obs, info = env.reset(options={"poses": poses})
    env.render()
    return obs, info

# DAGGER rollout collection
def collect_steps_dagger(env, planner, student_mppi, model, beta, max_steps, device):
    """
    Collect a list of dict records: {x0, mu_tilde, r_samples, mu_star}
    Execute expert action with probability = beta, else execute NN action.
    Always label with expert mu_star.
    """
    DATA = []
    obs, _ = reset_env_to_start(env, env.unwrapped.track)
    done = False
    step = 0

    while (not done): # and (step < max_steps)
        x0 = obs_dict_to_array(obs["agent_0"])

        # 1) Warm start from expert solver
        warmstart_solution = planner.solver.warm_start()
        planner.solver.control_params = deepcopy(warmstart_solution)

        # 2) Expert plan (label + expert action)
        steerv_exp, accl_exp = planner.plan(obs["agent_0"])
        expert_solution = planner.solver.control_params # Get the optimal control sequence
        mu_star = jax_to_numpy(expert_solution[0])

        # 3) Learner observable costs from student MPPI (few samples)
        student_mppi.solver.control_params = deepcopy(warmstart_solution)
        _ = student_mppi.plan(obs["agent_0"])
        costs_samples = jax_to_numpy(-student_mppi.solver.samples[-1])

        mu_tilde = jax_to_numpy(warmstart_solution[0])

        # 4) Student action from NN (mean-only)
        # use expert action with probability beta
        use_expert = (random.random() < beta) or (model is None)

        if use_expert:
            act = np.array([steerv_exp, accl_exp], dtype=np.float32)
        else:
            with torch.no_grad():
                mu_t = torch.from_numpy(mu_tilde.astype(np.float32)).unsqueeze(0).to(device)     # (1,*,*)
                c_t = torch.from_numpy(costs_samples.astype(np.float32)).unsqueeze(0).to(device) # (1,M)
                mu_pred = model(mu_t, c_t).squeeze(0).detach().cpu().numpy()
            act = mu_to_action(mu_pred).astype(np.float32)

        # 5) Record (always expert label mu_star)
        DATA.append({
            "x0": x0.astype(np.float32),
            "mu_tilde": mu_tilde.astype(np.float32),
            "r_samples": costs_samples.astype(np.float32),
            "mu_star": mu_star.astype(np.float32),
        })

        # 6) Step env
        obs, timestep, terminated, truncated, infos = env.step(act.reshape(1, 2))
        done = terminated or truncated
        env.render()
        step += 1

    return DATA

"""
Main DAGGER loop
"""
def main():
    # Load config from file
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    # Load DAGGER config parameters
    dagger_cfg = cfg.get("dagger", {})
    K = int(dagger_cfg.get("iters", 1))
    max_steps = int(dagger_cfg.get("max_steps", 5000))
    beta_0 = float(dagger_cfg.get("beta_0", 1.0))
    beta_decay = float(dagger_cfg.get("beta_decay", 0.8))
    beta_min = float(dagger_cfg.get("beta_min", 0.0))

    dataset_path = dagger_cfg.get("dataset_path", "dagger_dataset.npz")
    ckpt_path = dagger_cfg.get("ckpt_path", "trained_NN.pt")

    # Load training parameters (for model_train)
    train_params = dagger_cfg.get("train_params", {
        "batch_size": 256,
        "epochs": 300,
        "lr": 1e-4,
        "val_frac": 0.1,
        "print_after_epoch": 50,
        "seed": 0,
        "plateau_patience": 50,
        "early_stop_patience": 50,
        "early_stop_min_delta": 1e-6,
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup environment and planners
    env, waypoints_track, params, planner, student_mppi = setup_env_and_planners(cfg)

    # aggregated dataset across DAGGER iterations
    AGG = []

    # NN model will be trained, first is None 
    model = None

    # DAGGER main loop
    for k in range(K):
        beta_k = beta_schedule(k, beta_0=beta_0, decay=beta_decay, beta_min=beta_min)
        print(f"\n===== DAGGER iter {k}/{K-1} | beta={beta_k:.3f} =====")

        # load model from previous iter (except k=0 which is expert-only)
        if k > 0:
            model = load_model_checkpoint(ckpt_path, device=device)

        # collect data
        new_data = collect_steps_dagger(
            env=env,
            planner=planner,
            student_mppi=student_mppi,
            model=model,
            beta=beta_k,
            max_steps=max_steps,
            device=device,
        )
        AGG.extend(new_data)
        print(f"[dagger] collected {len(new_data)} new steps, total {len(AGG)}")

        # save aggregated dataset
        save_dataset_npz(AGG, dataset_path)

        # train / fine-tune
        mu_tilde, r_samples, mu_star = load_dataset_npz(dataset_path)
        ds = ImitationDataset(mu_tilde, r_samples, mu_star, per_step_normalize=True)

        # IMPORTANT: H and nu must match your L2O_Update_Net expectations.
        # If your mu is shape (2, N), you likely want H=2 and nu=N, or flatten inside your model.
        # In most cases you want H=N and nu=2 with mu shaped (N,2).
        # Here we infer from mu_tilde shape:
        mu_shape = mu_tilde.shape[1:]  # e.g., (2, N) or (N,2)
        print(f"[dagger] mu_tilde per-sample shape: {mu_shape}")

        # Heuristic: if mu is (2,N), set H=N and nu=2 but transpose inside dataset/model if needed.
        # You can adjust this once based on your actual L2O_Update_Net signature.
        if mu_shape[0] == 2:
            # (2, N) -> treat as (N,2) by transposing in dataset
            # simplest: transpose arrays before dataset creation
            mu_tilde_T = np.transpose(mu_tilde, (0, 2, 1))  # (K,N,2)
            mu_star_T = np.transpose(mu_star, (0, 2, 1))    # (K,N,2)
            ds = ImitationDataset(mu_tilde_T, r_samples, mu_star_T, per_step_normalize=True)
            H_train, nu_train = mu_tilde_T.shape[1], mu_tilde_T.shape[2]
        else:
            H_train, nu_train = mu_tilde.shape[1], mu_tilde.shape[2]

        M_train = r_samples.shape[1]

        print(f"[dagger] training with H={H_train}, nu={nu_train}, M={M_train}")
        _ = model_train(ds, H=H_train, nu=nu_train, M=M_train,
                        training_params=train_params, save_path=ckpt_path)

    print("\n[DAGGER] Done.")


if __name__ == "__main__":
    main()
