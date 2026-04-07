import argparse
import pathlib
import sys

import numpy as np

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
QUADRUPED_PYMPC_PATH = REPO_ROOT / "Quadruped-PyMPC"
if str(QUADRUPED_PYMPC_PATH) not in sys.path:
    sys.path.insert(0, str(QUADRUPED_PYMPC_PATH))

from quadruped_pympc import config as cfg


def sample_initial_base_positions(num_episodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ref_z = cfg.simulation_params["ref_z"]
    initial_base_positions = np.zeros((num_episodes, 3), dtype=np.float32)
    initial_base_positions[:, 0] = rng.uniform(-0.3, 0.3, size=num_episodes)
    initial_base_positions[:, 1] = rng.uniform(-0.3, 0.3, size=num_episodes)
    initial_base_positions[:, 2] = ref_z + rng.uniform(-0.02, 0.02, size=num_episodes)
    return initial_base_positions


def sample_goals(initial_base_positions: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ref_z = cfg.simulation_params["ref_z"]
    goals = np.zeros_like(initial_base_positions, dtype=np.float32)
    goals[:, 0] = initial_base_positions[:, 0] + rng.uniform(-2.0, 2.0, size=initial_base_positions.shape[0])
    goals[:, 1] = initial_base_positions[:, 1] + rng.uniform(-2.0, 2.0, size=initial_base_positions.shape[0])
    goals[:, 2] = ref_z
    return goals


def main():
    parser = argparse.ArgumentParser(description="Generate a fixed seeded set of test goals for controller comparisons.")
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of goals to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=REPO_ROOT / "training" / "test_goals.npz",
        help="Output .npz path.",
    )
    args = parser.parse_args()

    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_base_positions = sample_initial_base_positions(args.num_episodes, args.seed)
    goals = sample_goals(initial_base_positions, args.seed + 1)

    np.savez_compressed(
        output_path,
        seed=np.array(args.seed, dtype=np.int64),
        num_episodes=np.array(args.num_episodes, dtype=np.int64),
        initial_base_positions=initial_base_positions,
        goal_base_positions=goals,
    )

    print(f"Saved {args.num_episodes} test goals to: {output_path}")


if __name__ == "__main__":
    main()
