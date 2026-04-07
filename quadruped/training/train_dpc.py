import argparse
import pathlib
import sys

import jax
import yaml

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from step_mppi.dpc_policy_jax import NeuralGRFPolicy
from step_mppi.dpc_solver import DPC
from step_mppi.dpc_trainer import DPC_Trainer


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
    if "num_layers" in normalized and normalized["num_layers"] is not None:
        normalized["num_layers"] = int(normalized["num_layers"])
    if "hidden_dim" in normalized and normalized["hidden_dim"] is not None:
        normalized["hidden_dim"] = int(normalized["hidden_dim"])
    if "activation" in normalized and normalized["activation"] is not None:
        normalized["activation"] = str(normalized["activation"])
    return normalized


def main():
    default_config_path = pathlib.Path(__file__).resolve().parent / "dpc_config.yaml"

    parser = argparse.ArgumentParser(description="Train the DPC neural policy from a YAML config.")
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
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    train_params = normalize_train_params(dict(config.get("train_params", {})))
    policy_config = normalize_policy_config(dict(config.get("policy", {})))

    dataset_path = resolve_repo_path(config.get("dataset_path"), base_dir=config_path.parent)
    if dataset_path is None:
        raise ValueError("dataset_path is missing from the config file.")
    device = str(config.get("device", "gpu"))

    save_path = resolve_repo_path(train_params.get("save_path"), base_dir=config_path.parent)
    if save_path is not None:
        train_params["save_path"] = str(save_path)

    n_samples = train_params.pop("n_samples", None)

    policy = NeuralGRFPolicy(
        num_layers=policy_config.get("num_layers", 5),
        hidden_dim=policy_config.get("hidden_dim", 256),
        activation=policy_config.get("activation", "gelu"),
    )
    DPC_solver = DPC(policy=policy, device=device)
    trainer = DPC_Trainer(DPC_solver)
    trainer.load_dataset(
        data_filename=dataset_path,
        num_points=n_samples,
        random_subset=True,
        seed=train_params.get("seed", 0),
    )

    print(f"Loaded dataset: {dataset_path}")
    print(
        "Policy architecture: "
        f"NeuralGRFPolicy(num_layers={policy.num_layers}, "
        f"hidden_dim={policy.hidden_dim}, activation='{policy.activation}')"
    )

    retrain_path = resolve_repo_path(str(args.retrain), base_dir=config_path.parent) if args.retrain is not None else None
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
        result = trainer.train(train_params)

    print("Training finished.")
    print(f"Best eval loss: {result['best_eval_loss']:.6e}")
    print(f"Epochs completed: {len(result['history'])}")


if __name__ == "__main__":
    main()
