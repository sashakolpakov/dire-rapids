# syntax=docker/dockerfile:1.6
#
# dire-rapids container.
#
# Bakes a known-good RAPIDS + PyTorch + dire-rapids environment. The RAPIDS
# ecosystem's wheel-level pinning drifts frequently (CUDA minor versions,
# libnvJitLink, cuML/cuVS ABI), so the canonical "pip install it and it will
# work" story fails reliably across a 6-month window. The container version
# gives us something reproducible.
#
# Base: NVIDIA's official RAPIDS base image (contains cuML, cuVS, cuGraph,
# CuPy, the conda env, and matching CUDA userspace libs). We layer PyTorch
# with a matching CUDA build and install dire-rapids from source.
#
# Built for x86_64 and arm64 separately; the base image ships manifests for
# both. Build on a machine that matches the deploy arch, or use buildx for
# cross-building.
#
# --- Build ---
#   docker build -t dire-rapids:0.3.2 -f docker/Dockerfile .
# (Run from the repo root, not from inside docker/.)
#
# --- Run (interactive, with GPU) ---
#   docker run --gpus=all -it --rm -v $PWD:/workspace dire-rapids:0.3.2 bash
# --- Run the test suite ---
#   docker run --gpus=all --rm dire-rapids:0.3.2 \
#       python -m pytest tests/test_cpu_basic.py tests/test_reducer_runner.py -v

ARG RAPIDS_VERSION=26.06
# RAPIDS NGC tags use the CUDA major (cuda13), while the PyTorch index uses
# major+minor (cu130). Keep these independent so neither tag is synthesized
# incorrectly from the other.
ARG CUDA_VERSION=13
ARG PYTHON_VERSION=3.14
ARG TORCH_CUDA=130
ARG TORCH_VERSION=2.11.0

FROM nvcr.io/nvidia/rapidsai/base:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-py${PYTHON_VERSION}

# Re-declare ARGs after FROM so they're available in the build stage.
ARG RAPIDS_VERSION
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG TORCH_CUDA
ARG TORCH_VERSION

LABEL org.opencontainers.image.title="dire-rapids"
LABEL org.opencontainers.image.description="PyTorch + RAPIDS accelerated dimensionality reduction"
LABEL org.opencontainers.image.source="https://github.com/sashakolpakov/dire-rapids"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL rapids.version="${RAPIDS_VERSION}"

# The rapidsai/base image uses conda; stay in its env so we can co-exist
# with cuML / cuVS / cuGraph without shadowing their shared libs.
USER root
SHELL ["/bin/bash", "-c"]

# Guard against the libnvJitLink shadow-import gotcha: the PyTorch CUDA 13
# wheel also supplies libnvJitLink.so.13, but RAPIDS must load the version in
# its conda environment. Preloading that library makes both import orders
# deterministic.
ENV LD_PRELOAD=/opt/conda/lib/libnvJitLink.so.${CUDA_VERSION}

RUN --mount=type=cache,target=/opt/conda/pkgs \
    echo "Installing torch from https://download.pytorch.org/whl/cu${TORCH_CUDA}" && \
    python -m pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/cu${TORCH_CUDA}"

# Install dire-rapids itself.
WORKDIR /workspace
COPY pyproject.toml /tmp/pyproject.toml
COPY dire_rapids /tmp/dire_rapids
COPY tests /tmp/tests
COPY benchmarking /tmp/benchmarking
COPY README.md /tmp/README.md

# Install from the staged copy so the image doesn't embed the full repo
# (keeps the image small; mount the repo at runtime if you want to edit).
RUN python -m pip install --no-cache-dir /tmp[persistent,bench] && \
    python -c "import dire_rapids, fastdtw, flowio, ripser, optuna; print('dire-rapids', dire_rapids.__version__, 'fastdtw OK', 'flowio OK', 'ripser', ripser.__version__, 'optuna', optuna.__version__)"

# Smoke-test both import orders in separate Python processes. This catches a
# libnvJitLink selection problem that can otherwise depend on which framework
# happens to import first.
RUN python -c "import torch; import cuml; print('torch -> cuml import OK', torch.__version__, cuml.__version__)"
RUN python -c "import cuml; import torch; print('cuml -> torch import OK', cuml.__version__, torch.__version__)"

# Also check the installed dire-rapids surface. GPU availability is expected
# to be false during an ordinary image build, so this test only imports.
RUN python -c "import torch; \
    from dire_rapids import DiRePyTorch, presets; \
    from dire_rapids.betti_curve import compute_betti_curve_ripser; \
    assert not hasattr(presets, 'TOPOLOGY_TUNED'); \
    print('dire-rapids imports OK; torch.cuda.is_available() ==', torch.cuda.is_available())"

WORKDIR /workspace
CMD ["bash"]
