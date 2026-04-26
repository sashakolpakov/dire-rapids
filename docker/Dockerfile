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
#   docker build -t dire-rapids:0.3.0 -f docker/Dockerfile .
# (Run from the repo root, not from inside docker/.)
#
# --- Run (interactive, with GPU) ---
#   docker run --gpus=all -it --rm -v $PWD:/workspace dire-rapids:0.3.0 bash
# --- Run the test suite ---
#   docker run --gpus=all --rm dire-rapids:0.3.0 \
#       python -m pytest tests/test_cpu_basic.py tests/test_reducer_runner.py -v

ARG RAPIDS_VERSION=26.04
ARG CUDA_VERSION=12.8
ARG PYTHON_VERSION=3.12

FROM nvcr.io/nvidia/rapidsai/base:${RAPIDS_VERSION}-cuda${CUDA_VERSION}-py${PYTHON_VERSION}

# Re-declare ARGs after FROM so they're available in the build stage.
ARG RAPIDS_VERSION
ARG CUDA_VERSION
ARG PYTHON_VERSION

LABEL org.opencontainers.image.title="dire-rapids"
LABEL org.opencontainers.image.description="PyTorch + RAPIDS accelerated dimensionality reduction"
LABEL org.opencontainers.image.source="https://github.com/sashakolpakov/dire-rapids"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL rapids.version="${RAPIDS_VERSION}"

# The rapidsai/base image uses conda; stay in its env so we can co-exist
# with cuML / cuVS / cuGraph without shadowing their shared libs.
USER root
SHELL ["/bin/bash", "-c"]

# Guard against the libnvJitLink shadow-import gotcha: torch's pip-installed
# nvidia/cu13/lib/libnvJitLink.so.13 (v13.0) shadows conda's v13.2 once
# loaded, which breaks subsequent cuML imports. We install torch with the
# cu12x wheels (which do not ship libnvJitLink) when RAPIDS is on 12.x, and
# rely on conda's CUDA 13 libs for the 13.x path.
#
# Matching-CUDA wheel index computed from CUDA_VERSION ARG:
#   12.8 -> whl/cu128   12.5 -> whl/cu125   etc.
#
RUN --mount=type=cache,target=/opt/conda/pkgs \
    cuda_mm="${CUDA_VERSION/./}" && \
    echo "Installing torch from https://download.pytorch.org/whl/cu${cuda_mm}" && \
    python -m pip install --no-cache-dir \
        torch --index-url "https://download.pytorch.org/whl/cu${cuda_mm}"

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
    python -c "import dire_rapids, ripser, optuna; print('dire-rapids', dire_rapids.__version__, 'ripser', ripser.__version__, 'optuna', optuna.__version__)"

# Smoke-test the import chain that rapids-26.04 likes to break. If this
# fails at build time, the tag combination is unstable and we want to know
# immediately rather than at runtime.
RUN python -c "import cugraph; import torch; \
    from dire_rapids import DiRePyTorch, TOPOLOGY_TUNED, presets; \
    from dire_rapids.betti_curve import compute_betti_curve_ripser; \
    print('import-order OK; torch.cuda.is_available() ==', torch.cuda.is_available())"

WORKDIR /workspace
CMD ["bash"]
