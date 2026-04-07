# Quadr_Step_MPPI

Code for training and evaluating Step-MPPI on a quadrupedal robot:

- `step_mppi/` for DPC, DMPPI (Step-MPPI), and Neural DMPPI training code
- `training/` for the main train/test scripts

## Environment Setup

The easiest path is to create the same style of environment expected by `Quadruped-PyMPC`, then install both this repo and the bundled dependency in editable mode.

### 1. Clone the repository

Clone the repo for simulator in MuJoCo with controller interface (forked from [Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC)):

```bash
git clone https://github.com/vietanhle0101/Quadruped-PyMPC
cd Quadr_Step_MPPI
```

### 2. Create and activate a Conda or Mamba environment

`Quadruped-PyMPC` ships environment files under:

- `Quadruped-PyMPC/installation/mamba/nvidia_cuda/`
- `Quadruped-PyMPC/installation/mamba/integrated_gpu/`

Pick the one that matches your machine. For example:

```bash
cd Quadruped-PyMPC/installation/mamba/nvidia_cuda
conda env create -f mamba_environment.yml
conda activate quadruped_pympc_env
cd ../../..
```

### 3. Install the bundled `Quadruped-PyMPC` package

From the repo root:

```bash
pip install -e ./Quadruped-PyMPC
pip install -e "./Quadruped-PyMPC[sampling]"
```

The extra `[sampling]` installs JAX/JAXLIB support used by Sampling-MPC, DPC, and Step-MPPI.

### 4. Install Python packages used by the training code

```bash
pip install flax optax pyyaml tqdm
```

### 5. Optional: acados setup

You do not need `acados` for the training scripts in `training/`, but some `Quadruped-PyMPC` features depend on it. If you need the full upstream stack, follow:

- `Quadruped-PyMPC/README_install.md`

### 6. Verify the environment

From the repo root:

```bash
python -c "import jax, mujoco, gym_quadruped, quadruped_pympc; print('ok')"
```

## Default Model Checkpoints

This repo already includes trained checkpoints:

- checkpoints:
  - `training/dpc_constrained_policy.pkl`
  - `training/dmppi_constrained_policy.pkl`
  - `training/neural_dmppi_constrained_policy.pkl`

The default YAML configs in `training/` point at these files, so you can usually run the scripts without extra arguments.

## Training

Run all commands from the repository root unless you have a reason not to.

### Train DPC

```bash
python training/train_dpc.py
```

Use a custom config:

```bash
python training/train_dpc.py --config training/dpc_config.yaml
```

Resume from a checkpoint:

```bash
python training/train_dpc.py --retrain training/dpc_constrained_policy.pkl
```

### Train Step-MPPI

```bash
python training/train_dmppi.py
```

Warm-start the Step-MPPI mean branch from a trained DPC checkpoint:

```bash
python training/train_dmppi.py --warm-start training/dpc_constrained_policy.pkl
```

Resume Step-MPPI training:

```bash
python training/train_dmppi.py --retrain training/dmppi_constrained_policy.pkl
```

### Train Neural Step-MPPI

```bash
python training/train_neural_dmppi.py
```

Warm-start from DPC:

```bash
python training/train_neural_dmppi.py --warm-start training/dpc_constrained_policy.pkl
```

Resume from an existing neural Step-MPPI checkpoint:

```bash
python training/train_neural_dmppi.py --retrain training/neural_dmppi_constrained_policy.pkl
```

## Testing and Evaluation

The test scripts run Mujoco rollouts and save summary JSON files into `statistics/`. Some scripts can also save rollout CSVs with `--save_traj`.

### Generate a fixed goal set

For fair comparisons between controllers:

```bash
python training/generate_test_goals.py --num-episodes 20 --seed 0
```

This writes `training/test_goals.npz` by default.

### Test MPPI (Sampling-MPC)

```bash
python training/test_mppi.py --device gpu
```

### Test DPC

```bash
python training/test_dpc.py --policy_file training/dpc_constrained_policy.pkl --config training/dpc_config.yaml --device gpu
```

Headless evaluation on a fixed goal set:

```bash
python training/test_dpc.py --no-render --goal-file training/test_goals.npz --save_traj
```

### Test Step-MPPI

```bash
python training/test_dmppi.py --policy_file training/dmppi_constrained_policy.pkl --config training/dmppi_config.yaml --device gpu
```

### Test Neural Step-MPPI

```bash
python training/test_neural_dmppi.py --policy_file training/neural_dmppi_constrained_policy.pkl --config training/dmppi_neural_config.yaml --device gpu
```

## Useful Script Arguments

Common evaluation flags:

- `--device {cpu,gpu}`: JAX backend preference
- `--seed <int>`: random seed
- `--goal-file <path>`: use a fixed goal set from `.npz`
- `--random-goals`: sample fresh goals per episode
- `--no-render`: disable Mujoco viewer
- `--save_traj`: save rollout CSV when supported

Training configs live in:

- `training/dpc_config.yaml`
- `training/dmppi_config.yaml`
- `training/dmppi_neural_config.yaml`

## Outputs

The evaluation scripts write artifacts into `statistics/`, including:

- `*_summary_seed_<seed>.json`
- `*_rollout_seed_<seed>.csv`

These files contain success rate, lap times, timeout counts, and compute-time statistics.
