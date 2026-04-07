import os
import time
import argparse
import sys
from pathlib import Path
from copy import deepcopy
import gymnasium as gym
import numpy as np
import yaml
import jax
import torch
import f1tenth_gym
from f1tenth_gym.envs.track import Boundary
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
from dmppi import create_DMPPI_config, DMPPI_Solver
from trainer.DMPPI_trainer import DMPPI_Trainer
from policy import NeuralMPPIUpdate

f1tenth_gym_path = os.path.dirname(os.path.dirname(f1tenth_gym.__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Train DMPPI policy.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="Load an existing policy checkpoint before training.",
    )
    return parser.parse_args()

def main():
    """
    Differentiable MPPI training.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    # Load config
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "neural_dmppi_config.yaml"), "r") as f:
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
        render_mode=None,
    )

    # Load track waypoints
    waypoints_track = env.unwrapped.track
    pose_ref = np.vstack(
        [
            waypoints_track.raceline.xs, # x
            waypoints_track.raceline.ys, # y
            waypoints_track.raceline.yaws, # yaw
        ]).T

    # Load boundary info 
    boundary_file_path = os.path.join(f1tenth_gym_path, "maps", "Spielberg", "Spielberg_border_coeffs.csv")
    boundary = Boundary(boundary_file_path)
    track_bdr_coeffs = boundary.get_coefficients()

    # Get vehicle parameters
    params = f1tenth_params()

    # List of contraints
    constraints = [st_limit_constraint_batched, boundary_constraint_batched]

    tp = dict(cfg.get("train_params", {}))
    reference_speed = cfg.get("planner", {}).get("reference_speed", None)
    if reference_speed is not None:
        cv_ref = np.full_like(waypoints_track.raceline.vxs, float(reference_speed), dtype=np.float32)
    else:
        cv_ref = waypoints_track.raceline.vxs

    # Add constraints and lambdas to config
    config = create_DMPPI_config(cfg=cfg)
    config.constraints = constraints
    lam = float(tp.get("constraint_weight", 1e2))
    config.lambdas = [lam] * len(constraints) # penalty weights for constraints
    config.alpha = float(tp.get("exploration_weight", 1e0)) # weight for exploration
    config.beta = float(tp.get("supervised_weight", 1e0)) # weight for updater supervision
    config.m_samples = cfg.get("planner", {}).get("m_samples", 64)
    config.updater_n_candidates = int(cfg.get("planner", {}).get("updater_n_candidates", 16))
    base_save_path = Path(tp.get("save_path", "neural_dmppi_policy.pt"))
    dt_ms = int(round(float(config.dt) * 1000.0))
    save_suffix = f"_N{int(config.N)}_dt{dt_ms}ms"
    save_ext = base_save_path.suffix if base_save_path.suffix else ".pt"
    save_name = f"{base_save_path.stem}{save_suffix}{save_ext}"
    resolved_save_path = str(base_save_path.with_name(save_name))

    print(
        "\033[31;42m"
        "[dmppi config] "
        f"Constraint penalty weight: {lam}, "
        f"Exploration weight: {config.alpha}, "
        f"Supervised weight: {config.beta}"
        "\033[0m"
    )
    print("-"*50)

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
        num_hidden_layers=(2, 2, 2),  # [global, cost, head]
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
    if args.retrain:
        primary_path = os.path.abspath(os.path.join(current_dir, "neural_dmppi_constrained_policy_N100_dt50ms.pt"))
        fallback_path = os.path.abspath(os.path.join(current_dir, "../saved_model/dmppi_constrained_policy.pt"))
        trained_policy_path = primary_path if os.path.exists(primary_path) else fallback_path
        solver.load_trained_policy(trained_policy_path)

    # Use a smaller train-time rollout sample count than planner/test-time to fit GPU memory.
    train_rollout_samples = int(tp.get("rollout_n_samples", min(128, int(solver.n_samples))))
    solver.n_samples = max(1, train_rollout_samples)

    trainer = DMPPI_Trainer(solver=solver, device=device)
    trainer.set_constraints(constraints)

    training_params = {
        "n_samples": int(tp.get("n_samples", 1e6)),
        "num_epochs": int(tp.get("num_epochs", 50)),
        "steps_per_epoch": int(tp.get("steps_per_epoch", 200)),
        "batch_size": int(tp.get("batch_size", 64)),
        "lr": float(tp.get("lr", 1e-3)),
        "weight_decay": float(tp.get("weight_decay", 0.0)),
        "grad_clip": float(tp.get("grad_clip", 1.0)),
        "log_every": int(tp.get("log_every", 20)),
        "save_path": resolved_save_path,
        "save_every": int(tp.get("save_every", 10)),
        "plateau_factor": float(tp.get("plateau_factor", 0.5)),
        "plateau_patience": int(tp.get("plateau_patience", 5)),
        "plateau_min_lr": float(tp.get("plateau_min_lr", 1e-6)),
        "plateau_threshold": float(tp.get("plateau_threshold", 1e-4)),
        "plateau_cooldown": int(tp.get("plateau_cooldown", 0)),
        "eval_batch_size": int(tp.get("eval_batch_size", 256)),
        "early_stop_patience": int(tp.get("early_stop_patience", 10)),
        "early_stop_min_delta": float(tp.get("early_stop_min_delta", 1e-4)),
        "penalty_increase_factor": float(tp.get("penalty_increase_factor", 1.5)),
    }
    print(f"Model checkpoints will be saved to: {resolved_save_path}")
    if reference_speed is not None:
        print(f"Training reference speed: constant {float(reference_speed):.3f} m/s")
    else:
        print("Training reference speed: raceline profile")

    # Keep large offline training samples on CPU to avoid exhausting GPU memory.
    sample_device = torch.device("cpu")
    trainer.generate_samples(
        pose_ref=pose_ref,
        cv_ref=cv_ref,
        track_bdr_coeffs=track_bdr_coeffs,
        N=config.N, sample_size=training_params["n_samples"], 
        device=sample_device)
        
    trainer.train(training_params)    

    # # # Dummy test at the end
    # # sample_x0 = trainer.samples["x0"][0,:]
    # # sample_R = trainer.samples["R"][0,:,:]
    # # X, U, _ = solver.rollout(sample_x0.unsqueeze(0), sample_R.unsqueeze(0))
    # # # Cross-track error (signed) relative to reference heading.
    # # # cte > 0 means vehicle is to the left of the reference heading direction.
    # # x_pred = X[0, :, 0]
    # # y_pred = X[0, :, 1]
    # # x_ref = sample_R[:, 0]
    # # y_ref = sample_R[:, 1]
    # # yaw_ref = sample_R[:, 2]
    # # cte = -torch.sin(yaw_ref) * (x_pred - x_ref) + torch.cos(yaw_ref) * (y_pred - y_ref)

    # # print("Sample rollout cross-track error (signed) per step:", cte)
    # # print(
    # #     "Cross-track error stats: "
    # #     f"mean_abs={cte.abs().mean().item():.6f}, "
    # #     f"max_abs={cte.abs().max().item():.6f}"
    # # )

if __name__ == "__main__":
    main()
