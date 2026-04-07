# F1tenth_Step_MPPI
This repository contains code for Step-MPPI, a framework using learning-to-optimize (L2O) methods to learn optimal sampling distributions for Model Predictive Path Integral (MPPI) control.


## Step_MPPI Environment Setup (GPU + F1TENTH)

This README documents the setup steps that worked for running `f1tenth_gym` and `f1tenth_planning` examples (including JAX GPU MPPI) on Ubuntu with `uv`.

## Tested setup (working)
- **OS**: Ubuntu (Linux)
- **Python**: **3.11** (recommended)
- **Environment manager**: `uv`
- **GPU**: NVIDIA RTX 3050 (laptop)
- **NVIDIA driver**: 570.211.x
- **PyTorch**: CUDA wheels from `cu121` index
- **JAX**: `0.4.33` + `jax[cuda12]==0.4.33` (this version worked reliably)

---

## 1) Create and activate a virtual environment (uv)

From the project root (example: `~/github/F1tenth_Step_MPPI`):

```bash
uv venv --python 3.11 l2o_venv
source l2o_venv/bin/activate
```

> Recommended: avoid having Conda `base` active at the same time (to reduce CUDA/JAX conflicts).
```bash
conda deactivate 2>/dev/null || true
```

Check the active Python:
```bash
which python
python --version
echo $VIRTUAL_ENV
```

---

## 2) Install base Python dependencies

If you have a `requirements.txt`:
```bash
uv pip install -r requirements.txt
```

---

## 3) Install PyTorch with CUDA support (using uv)

Use the PyTorch CUDA index (example below uses CUDA 12.1 wheels):

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Test PyTorch CUDA
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

---

## 4) Install JAX with GPU support (known-good pinned version)

### Important
The following **pinned version worked**:

- `jax==0.4.33`
- `jaxlib==0.4.33`
- `jax[cuda12]==0.4.33`

Install:

```bash
uv pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt 2>/dev/null || true
uv pip install "jax==0.4.33" "jaxlib==0.4.33"
uv pip install "jax[cuda12]==0.4.33"
```

### Verify JAX versions
```bash
python - <<'PY'
import jax, jaxlib
print("jax:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
try:
    import jax_plugins.xla_cuda12 as p
    print("cuda12 plugin:", p.__file__)
except Exception as e:
    print("plugin import error:", e)
PY
```

### Test JAX GPU (strict test)
This test fails if JAX silently falls back to CPU:

```bash
JAX_PLATFORMS=cuda python - <<'PY'
import jax, jax.numpy as jnp

print("jax:", jax.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())

assert jax.default_backend() == "gpu", f"Not on GPU: {jax.default_backend()}"

x = jnp.ones((256, 256), dtype=jnp.float32)
y = x @ x
y.block_until_ready()

print("shape:", y.shape)
print("dtype:", y.dtype)
print("JAX GPU compute OK")
PY
```

---

## 5) Clone and install F1TENTH repositories

### `f1tenth_gym` (my forked repo)
```bash
git clone https://github.com/vietanhle0101/f1tenth_gym
```

### `f1tenth_planning` (my forked repo)
```bash
git clone https://github.com/vietanhle0101/f1tenth_planning
```

Install both in **editable mode** (recommended for development):

```bash
uv pip install -e ./f1tenth_gym
uv pip install -e ./f1tenth_planning
```

---

## 6) Re-pin JAX after installing project packages (important)

Some dependencies may change JAX versions indirectly. Reinstall the known-good JAX version after all installs:

```bash
uv pip install --reinstall "jax==0.4.33" "jaxlib==0.4.33"
uv pip install --reinstall "jax[cuda12]==0.4.33"
```

---

## 7) Run examples

### F1TENTH gym example
```bash
python3 f1tenth_gym/examples/waypoint_follow.py
```

### F1TENTH planning JAX MPPI example
```bash
python3 f1tenth_planning/examples/control/dynamic_mppi.py
```

---

## Reproducibility Tips

### Freeze exact installed packages
```bash
uv pip freeze > requirements-lock.txt
```

### Verify JAX GPU after any reinstall
```bash
JAX_PLATFORMS=cuda python - <<'PY'
import jax, jax.numpy as jnp
assert jax.default_backend() == "gpu", jax.default_backend()
x = jnp.ones((256,256), dtype=jnp.float32)
(x @ x).block_until_ready()
print("JAX GPU OK", jax.__version__, jax.devices())
PY
```

---

## 8) Train and Test Policies (PyTorch)

The project includes several learning and control pipelines:

- DMPPI / Step-MPPI: `dmppi_learning/`
- DPC: `ss_learning/`
- Conventional MPPI baseline: `imit_learning/`

### Train DMPPI / Step-MPPI

From the repository root:

```bash
cd dmppi_learning
python3 train_dmppi.py
```

Notes:

- `train_dmppi.py` reads training parameters from `dmppi_config.yaml`.
- The trainer saves checkpoint dicts containing `policy_state_dict`.
- `train_params.save_path` controls the checkpoint location. The current default is `dmppi_constrained_policy.pt`.

To continue training from an existing checkpoint:

```bash
python3 train_dmppi.py --retrain
```

### Train DPC

```bash
cd ss_learning
python3 train_dpc.py
```

Notes:

- `train_dpc.py` reads training parameters from `dpc_config.yaml`.
- `train_params.save_path` controls the checkpoint location. The current default is `dpc_constrained_policy.pt`.

To continue training from an existing checkpoint:

```bash
python3 train_dpc.py --retrain
```

### Test trained policy

Test the trained DMPPI / Step-MPPI policy:

```bash
cd dmppi_learning
python3 test_dmppi.py
```

`test_dmppi.py` loads the policy checkpoint via `DMPPI_Solver.load_trained_policy(...)`.
By default, it looks for `dmppi_constrained_policy.pt` in `dmppi_learning/`, then falls back to `saved_model/dmppi_constrained_policy.pt`.

Test the trained DPC policy:

```bash
cd ss_learning
python3 test_dpc.py
```

`test_dpc.py` loads the policy checkpoint via `DPC_Solver.load_trained_policy(...)`.
By default, it looks for `dpc_constrained_policy.pt` in `ss_learning/`, then falls back to `saved_model/dpc_constrained_policy.pt`.

Test the conventional MPPI baseline:

```bash
cd imit_learning
python3 test_cmppi.py
```
