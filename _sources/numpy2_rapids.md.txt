# NumPy 2 and RAPIDS Setup

DiRe-RAPIDS supports NumPy 2 in environments where PyTorch and the RAPIDS
packages are installed against a compatible NumPy ABI. The most common failure
mode is mixing a system PyTorch package compiled against NumPy 1.x with a newer
RAPIDS environment that installs NumPy 2.x. In that mixed state, PyTorch imports
but `Tensor.numpy()` fails with `RuntimeError: Numpy is not available`.

Use a clean virtual environment for RAPIDS work, and install PyTorch, NumPy, and
RAPIDS together. Avoid `--system-site-packages` unless you know the system
PyTorch build is compatible with the active NumPy major version.

Example for CUDA 12 wheels:

```bash
python3 -m venv ~/dire-rapids-env
source ~/dire-rapids-env/bin/activate
python -m pip install --upgrade pip

python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "dire-rapids[rapids,bench,viz,keops]"
```

For development from a clone:

```bash
git clone https://github.com/sashakolpakov/dire-rapids.git
cd dire-rapids
python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e ".[rapids,bench,viz,keops]"
```

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
```

Expected package family for the `rapids` extra:

- NumPy 2.x
- PyTorch built against NumPy 2.x
- RAPIDS/cuML/cuVS/cuDF 26.2 or later

If the verification fails with `RuntimeError: Numpy is not available`, recreate
the environment without system packages and install a PyTorch build compiled
against the active NumPy major version.
