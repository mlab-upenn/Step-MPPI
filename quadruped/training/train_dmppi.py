import argparse
import pathlib
import pickle
import sys

import jax
import yaml

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from step_mppi.dpc_policy_jax import (
    NeuralGRFDistributionPolicy,
    warm_start_distribution_policy_params,
)
from step_mppi.dmppi_solver import DMPPI
from step_mppi.dmppi_trainer import DMPPI_Trainer


def load_config(config_path: pathlib.Path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(path_value: str | None, base_dir: pathlib.Path | None = None):
    if path_value is None:
        return None
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return path
    if base_dir is not None:
        config_relative = base_dir / path
        if config_relative.exists():
            return config_relative
    return REPO_ROOT / path


def normalize_train_params(train_params: dict):
    int_keys = {
        "n_samples",
        "num_epochs",
        "steps_per_epoch",
        "batch_size",
        "log_every",
        "save_every",
        "plateau_patience",
        "plateau_cooldown",
        "eval_batch_size",
        "early_stop_patience",
        "seed",
        "eval_seed",
    }
    float_keys = {
        "lr",
        "weight_decay",
        "grad_clip",
        "plateau_factor",
        "plateau_min_lr",
        "plateau_threshold",
        "early_stop_min_delta",
        "penalty_increase_factor",
    }

    normalized = dict(train_params)
    for key in int_keys:
        if key in normalized and normalized[key] is not None:
            normalized[key] = int(normalized[key])
    for key in float_keys:
        if key in normalized and normalized[key] is not None:
            normalized[key] = float(normalized[key])
    return normalized


def normalize_policy_config(policy_config: dict):
    normalized = dict(policy_config)
    int_keys = {"num_layers", "hidden_dim"}
    float_keys = {"min_std", "max_std", "max_fx", "max_fy", "max_fz"}
    for key in int_keys:
        if key in normalized and normalized[key] is not None:
            normalized[key] = int(normalized[key])
    for key in float_keys:
        if key in normalized and normalized[key] is not None:
            normalized[key] = float(normalized[key])
    if "activation" in normalized and normalized["activation"] is not None:
        normalized["activation"] = str(normalized["activation"])
    return normalized


def normalize_dmppi_config(dmppi_config: dict):
    normalized = dict(dmppi_config)
    if "dmppi_samples" in normalized and normalized["dmppi_samples"] is not None:
        normalized["dmppi_samples"] = int(normalized["dmppi_samples"])
    for key in ("dmppi_temperature", "alpha"):
        if key in normalized and normalized[key] is not None:
            normalized[key] = float(normalized[key])
    return normalized


def main():
    default_config_path = CURRENT_DIR / "dmppi_config.yaml"

    parser = argparse.ArgumentParser(description="Train the DMPPI policy from a YAML config.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=default_config_path,
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--retrain",
        type=pathlib.Path,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--warm-start",
        type=pathlib.Path,
        default=None,
        help="Optional trained DPC checkpoint used to warm-start the DMPPI mean branch.",
    )
    args = parser.parse_args()

    if args.retrain is not None and args.warm_start is not None:
        raise ValueError("Use either --retrain or --warm-start, not both.")

    config_path = args.config.resolve()
    config = load_config(config_path)
    train_params = normalize_train_params(dict(config.get("train_params", {})))
    policy_config = normalize_policy_config(dict(config.get("policy", {})))
    dmppi_config = normalize_dmppi_config(dict(config.get("dmppi", {})))

    dataset_path = resolve_repo_path(config.get("dataset_path"), base_dir=config_path.parent)
    if dataset_path is None:
        raise ValueError("dataset_path is missing from the config file.")
    device = str(config.get("device", "gpu"))

    save_path = resolve_repo_path(train_params.get("save_path"), base_dir=config_path.parent)
    if save_path is not None:
        train_params["save_path"] = str(save_path)

    n_samples = train_params.pop("n_samples", None)

    policy = NeuralGRFDistributionPolicy(
        num_layers=policy_config.get("num_layers", 5),
        hidden_dim=policy_config.get("hidden_dim", 256),
        activation=policy_config.get("activation", "gelu"),
        min_std=policy_config.get("min_std", 1e-3),
        max_std=policy_config.get("max_std", 1e2),
        max_fx=policy_config.get("max_fx", 30.0),
        max_fy=policy_config.get("max_fy", 30.0),
        max_fz=policy_config.get("max_fz", 241.68897),
    )
    dmppi_solver = DMPPI(
        policy=policy,
        device=device,
        num_dmppi_samples=dmppi_config.get("dmppi_samples", 64),
        dmppi_temperature=dmppi_config.get("dmppi_temperature", 0.2),
    )
    trainer = DMPPI_Trainer(
        dmppi_solver,
        alpha=dmppi_config.get("alpha", 1e-3),
    )
    trainer.load_dataset(
        data_filename=dataset_path,
        num_points=n_samples,
        random_subset=True,
        seed=train_params.get("seed", 0),
    )

    print(f"Loaded dataset: {dataset_path}")
    print(
        "Policy architecture: "
        f"NeuralGRFDistributionPolicy(num_layers={policy.num_layers}, "
        f"hidden_dim={policy.hidden_dim}, activation='{policy.activation}', "
        f"min_std={policy.min_std}, max_std={policy.max_std})"
    )
    print(
        "DMPPI settings: "
        f"num_dmppi_samples={dmppi_solver.num_dmppi_samples}, "
        f"dmppi_temperature={dmppi_solver.dmppi_temperature}, "
        f"alpha={trainer.alpha}"
    )

    retrain_path = resolve_repo_path(str(args.retrain), base_dir=config_path.parent) if args.retrain is not None else None
    warm_start_path = resolve_repo_path(str(args.warm_start), base_dir=config_path.parent) if args.warm_start is not None else None
    if retrain_path is not None:
        resume = trainer.load_trained_model(retrain_path, training_params=train_params)
        print(f"Loaded checkpoint for retraining: {retrain_path}")
        result = trainer.train(
            resume["training_params"],
            params=resume["params"],
            optimizer_state=resume["optimizer_state"],
            start_epoch=0,
        )
    else:
        initial_params = None
        if warm_start_path is not None:
            with open(warm_start_path, "rb") as file:
                warm_start_checkpoint = pickle.load(file)

            deterministic_params = warm_start_checkpoint["params"]
            init_key = jax.random.PRNGKey(train_params.get("seed", 0))
            initial_params = trainer.init_params(init_key)
            initial_params = warm_start_distribution_policy_params(
                deterministic_params=deterministic_params,
                distribution_params=initial_params,
            )
            print(f"Warm-started DMPPI mean branch from: {warm_start_path}")

        result = trainer.train(train_params, params=initial_params)

    print("Training finished.")
    print(f"Best eval loss: {result['best_eval_loss']:.6e}")
    print(f"Epochs completed: {len(result['history'])}")


if __name__ == "__main__":
    main()
