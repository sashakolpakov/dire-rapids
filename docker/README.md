# Docker image for dire-rapids

The RAPIDS ecosystem (cuML, cuVS, cuGraph, CuPy) is not API-stable across
point releases; a working `pip install` combination today is likely to
break within 3–6 months. The container lets you lock in a known-good
toolchain.

## Prerequisites

- Docker 20.10+ with the NVIDIA Container Toolkit
  (`docker info | grep -i nvidia` should show the runtime).
- A Turing-or-newer NVIDIA GPU with driver 580+ for the default CUDA 13
  userspace libraries (the container does not ship a driver).
- The NVIDIA NGC base image requires `nvcr.io` pulls to work; no login is
  needed for public images.

## Build

From the **repository root** (not from inside `docker/`):

```bash
docker build -t dire-rapids:0.3.2 -f docker/Dockerfile .
```

The default image is RAPIDS 26.06, CUDA 13, Python 3.14, with PyTorch 2.11.0's
CUDA 13.0 wheels. RAPIDS NGC tags use the CUDA major (``cuda13``), whereas the
PyTorch index uses major+minor (``cu130``), so the Dockerfile keeps those
selectors separate.

Override the pins together if you need a different published combination:

```bash
docker build \
    --build-arg RAPIDS_VERSION=26.04 \
    --build-arg CUDA_VERSION=12 \
    --build-arg PYTHON_VERSION=3.12 \
    --build-arg TORCH_CUDA=128 \
    --build-arg TORCH_VERSION=2.11.0 \
    -t dire-rapids:0.3.2-custom \
    -f docker/Dockerfile .
```

Architectures: the base image ships x86_64 and arm64 manifests, so the
Dockerfile builds on either. Cross-building is easiest with Docker buildx:

```bash
docker buildx build --platform linux/amd64 -t dire-rapids:0.3.2 \
    -f docker/Dockerfile --load .
```

## Run

Interactive shell with GPU access and the repo mounted for development:

```bash
docker run --gpus=all -it --rm -v $PWD:/workspace dire-rapids:0.3.2 bash
```

Run the test suite in a fresh container:

```bash
docker run --gpus=all --rm dire-rapids:0.3.2 \
    python -m pytest tests/test_cpu_basic.py tests/test_reducer_runner.py -v
```

Run the topology-Pareto benchmark on a couple of small datasets:

```bash
docker run --gpus=all --rm -v $PWD:/workspace -w /workspace dire-rapids:0.3.2 \
    python benchmarking/bench_topology_pareto.py \
        --datasets mfeat-factors,satimage \
        --n-trials 50 \
        --output topology_pareto_results.json
```

## Troubleshooting

- **`undefined symbol: __nvJitLinkComplete_13_2`** at `import cugraph`:
  PyTorch's wheel-provided `libnvJitLink.so.13` was loaded instead of the
  RAPIDS conda copy. The default image sets
  `LD_PRELOAD=/opt/conda/lib/libnvJitLink.so.13` (the suffix follows
  `CUDA_VERSION` when overridden) and smoke-tests both
  `torch` then `cuml` and `cuml` then `torch` in separate Python processes.
  Preserve that preload when overriding the entrypoint.
- **PyTorch reports the wrong version or CUDA family**: `CUDA_VERSION` controls the
  major-only RAPIDS NGC tag, while `TORCH_CUDA` controls the PyTorch wheel
  index and `TORCH_VERSION` selects the release. Change all three as a
  published, compatible combination.
- **Pulling `nvcr.io/nvidia/rapidsai/base:...` fails**: the specific
  `RAPIDS_VERSION-CUDA-py` combination may not exist yet. Check
  <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/rapidsai/containers/base>
  for available tags.
