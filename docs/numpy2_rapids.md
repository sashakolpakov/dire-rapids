# NumPy 2 and RAPIDS 26.06 Setup

DiRe-RAPIDS supports NumPy 2 in environments where PyTorch and the RAPIDS
packages are installed against a compatible NumPy ABI. The most common failure
mode is mixing a system PyTorch package compiled against NumPy 1.x with a newer
RAPIDS environment that installs NumPy 2.x. In that mixed state, PyTorch imports
but `Tensor.numpy()` fails with `RuntimeError: Numpy is not available`.

Use a clean virtual environment for RAPIDS work, and install PyTorch, NumPy, and
RAPIDS together. Avoid `--system-site-packages` unless you know the system
PyTorch build is compatible with the active NumPy major version.

Choose exactly one CUDA family. CUDA 13 is the recommended RAPIDS 26.06 and
Python 3.14 configuration:

```bash
python3.14 -m venv ~/dire-rapids-env
source ~/dire-rapids-env/bin/activate
python -m pip install --upgrade pip

git clone https://github.com/sashakolpakov/dire-rapids.git
cd dire-rapids
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  -e ".[rapids-cu13,bench,viz,keops]"
```

For CUDA 12, use the CUDA 12 extra and matching PyTorch index instead:

```bash
git clone https://github.com/sashakolpakov/dire-rapids.git
cd dire-rapids
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  -e ".[rapids-cu12,bench,viz,keops]"
```

The legacy `rapids` extra remains a backward-compatible alias of
`rapids-cu12`. Do not combine the CUDA 12 and CUDA 13 extras.

Verify the environment:

```bash
python - <<'PY'
import numpy, torch, cupy, cuml, cuvs, cudf
print("numpy", numpy.__version__)
print("torch", torch.__version__, torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
print("cupy", cupy.__version__)
print("cuml", cuml.__version__)
print("cuvs", cuvs.__version__)
print("cudf", cudf.__version__)
print("torch numpy bridge", torch.arange(4, device="cuda").cpu().numpy())
PY

# Check the framework import orders in independent processes.
python -c "import torch, cuml; print('torch -> cuml OK')"
python -c "import cuml, torch; print('cuml -> torch OK')"
```

Expected package family:

- NumPy 2.x
- PyTorch built against NumPy 2.x
- RAPIDS/cuML/cuVS/cuDF constrained to the 26.06.x release line
- CuPy 14.x for the selected CUDA family

The supplied Docker image additionally preloads the RAPIDS conda environment's
`/opt/conda/lib/libnvJitLink.so.13`. This prevents PyTorch's CUDA 13 wheel from
changing whether subsequent cuML/cuVS imports succeed. If a custom conda
environment exhibits an import-order-dependent `libnvJitLink` symbol error,
preload `$CONDA_PREFIX/lib/libnvJitLink.so.13` and repeat both independent
import checks above.

If the verification fails with `RuntimeError: Numpy is not available`, recreate
the environment without system packages and install a PyTorch build compiled
against the active NumPy major version.
