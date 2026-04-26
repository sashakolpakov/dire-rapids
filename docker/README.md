# Docker image for dire-rapids

The RAPIDS ecosystem (cuML, cuVS, cuGraph, CuPy) is not API-stable across
point releases; a working `pip install` combination today is likely to
break within 3–6 months. The container lets you lock in a known-good
toolchain.

## Prerequisites

- Docker 20.10+ with the NVIDIA Container Toolkit
  (`docker info | grep -i nvidia` should show the runtime).
- An NVIDIA GPU with CUDA 12.x+ drivers installed on the host (the
  container does not ship a driver, only userspace libraries).
- The NVIDIA NGC base image requires `nvcr.io` pulls to work; no login is
  needed for public images.

## Build

From the **repository root** (not from inside `docker/`):

```bash
docker build -t dire-rapids:0.3.0 -f docker/Dockerfile .
```

Override the RAPIDS / CUDA / Python pins at build time if you need to:

```bash
docker build \
    --build-arg RAPIDS_VERSION=26.04 \
    --build-arg CUDA_VERSION=12.8 \
    --build-arg PYTHON_VERSION=3.12 \
    -t dire-rapids:0.3.0-custom \
    -f docker/Dockerfile .
```

Architectures: the base image ships x86_64 and arm64 manifests, so the
Dockerfile builds on either. Cross-building is easiest with Docker buildx:

```bash
docker buildx build --platform linux/amd64 -t dire-rapids:0.3.0 \
    -f docker/Dockerfile --load .
```

## Run

Interactive shell with GPU access and the repo mounted for development:

```bash
docker run --gpus=all -it --rm -v $PWD:/workspace dire-rapids:0.3.0 bash
```

Run the test suite in a fresh container:

```bash
docker run --gpus=all --rm dire-rapids:0.3.0 \
    python -m pytest tests/test_cpu_basic.py tests/test_reducer_runner.py -v
```

Run the topology-Pareto benchmark on a couple of small datasets:

```bash
docker run --gpus=all --rm -v $PWD:/workspace -w /workspace dire-rapids:0.3.0 \
    python benchmarking/bench_topology_pareto.py \
        --datasets mfeat-factors,satimage \
        --n-trials 50 \
        --output topology_pareto_results.json
```

## Troubleshooting

- **`undefined symbol: __nvJitLinkComplete_13_2`** at `import cugraph`:
  a PyTorch + RAPIDS CUDA-version skew. The Dockerfile picks matching
  wheels, but if you change `CUDA_VERSION` at build time without the
  matching PyTorch cu-index, this reappears. Pin both explicitly.
- **Pulling `nvcr.io/nvidia/rapidsai/base:...` fails**: the specific
  `RAPIDS_VERSION-CUDA-py` combination may not exist yet. Check
  <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/rapidsai/containers/base>
  for available tags.
