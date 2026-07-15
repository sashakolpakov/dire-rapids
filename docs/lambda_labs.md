# Lambda Labs GPU Setup

This note records a known-good setup for running DiRe-RAPIDS benchmarks on a
Lambda Labs GPU instance. The same NumPy/PyTorch/RAPIDS compatibility rules in
`docs/numpy2_rapids.md` apply here: use a clean virtual environment and avoid
mixing Lambda image system packages with a fresh NumPy 2/RAPIDS stack.

This recipe selects the CUDA 12 RAPIDS 26.06.x packages explicitly and pairs
them with PyTorch's CUDA 12.8 wheel index.

Verified instance class:

- Lambda Labs `gpu_1x_a10`
- NVIDIA A10, 23 GB GPU memory
- Ubuntu 22.04
- NVIDIA driver 580.105.08
- Python 3.11 (RAPIDS 26.06 supports Python 3.11--3.14)

Setup:

```bash
python3.11 -m venv ~/dire-rapids-env
source ~/dire-rapids-env/bin/activate
python -m pip install --upgrade pip

git clone https://github.com/sashakolpakov/dire-rapids.git
cd dire-rapids
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  -e ".[rapids-cu12,bench,viz,keops]"
```

Verification:

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

If `Tensor.numpy()` fails, recreate the environment without
`--system-site-packages` and install PyTorch from the PyTorch wheel index rather
than using the image's system PyTorch package.
