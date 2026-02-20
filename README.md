<!-- Logo + Project title -->
<p align="center">
  <img src="images/dire_rapids_logo.png" alt="DiRe-RAPIDS logo" width="280" style="margin-bottom:10px;">
</p>
<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
  <a href="https://www.python.org/downloads/">
    <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg">
  </a>
  <a href="https://pypi.org/project/dire-rapids/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/dire-rapids.svg">
  </a>
  <a href="https://pepy.tech/projects/dire-rapids">
    <img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/dire-rapids">
  </a>
</p>
<p align="center">
  <a href="https://github.com/sashakolpakov/dire-rapids/actions/workflows/pylint.yml">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/sashakolpakov/dire-rapids/pylint.yml?branch=main&label=CI&logo=github">
  </a>
  <a href="https://github.com/sashakolpakov/dire-rapids/actions/workflows/deploy_docs.yml">
    <img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/sashakolpakov/dire-rapids/deploy_docs.yml?branch=main&label=Docs&logo=github">
  </a>
  <a href="https://sashakolpakov.github.io/dire-rapids/">
    <img alt="Docs Live" src="https://img.shields.io/website-up-down-green-red/https/sashakolpakov.github.io/dire-rapids?label=API%20Documentation">
  </a>
</p>

# DiRe Rapids

GPU-accelerated implementation of [DiRe](https://github.com/sashakolpakov/dire-jax) using PyTorch and optionally NVIDIA RAPIDS for massive-scale datasets.

## What is DiRe?

DiRe (**Di**mensionality **Re**duction) is a dimensionality reduction algorithm based on force-directed graph layout. Unlike methods that focus solely on local neighborhood preservation, DiRe preserves both local and global structure of the data manifold, with theoretical guarantees for **homological stability** -- the topology (connected components, loops) of the original point cloud is faithfully reflected in the low-dimensional embedding. See the [paper on arXiv](https://arxiv.org/abs/2503.03156) for details.

## Performance

DiRe is **9--42x faster than UMAP** on CPU while delivering competitive or better embedding quality (neighborhood preservation). On GPU it leverages `torch.compile` for kernel fusion, pushing throughput even further.

| Dataset | N | D | DiRe (s) | UMAP (s) | Speedup |
|:---|---:|---:|---:|---:|---:|
| digits | 5,620 | 64 | 1.3 | 11.9 | **9.2x** |
| mnist_784 | 10,000 | 784 | 2.5 | 49.4 | **19.8x** |
| Fashion-MNIST | 10,000 | 784 | 2.3 | 46.6 | **20.3x** |
| har | 10,299 | 561 | 2.4 | 101.0 | **42.1x** |
| covertype | 20,000 | 54 | 3.9 | 43.9 | **11.3x** |

*Benchmarks on OpenML datasets; times are wall-clock on a single CPU core.*

At large scale (500K+ points), DiRe also **beats cuML UMAP on embedding quality** (neighborhood preservation), making it the best choice for both speed and fidelity on big data.

### Topological Preservation

DiRe is designed to preserve the topology of the original data manifold. We measure this by computing [Betti curves](https://en.wikipedia.org/wiki/Betti_number) on the original point cloud and on the 2D embedding, then comparing them via DTW distance (lower = better preservation):

| Dataset | Topology | DiRe DTW β₀ | cuML DTW β₀ | DiRe DTW β₁ | cuML DTW β₁ |
|:---|:---|---:|---:|---:|---:|
| circle (S¹) | β₀=1, β₁=1 | **56** | 76 | **29** | 47 |
| torus (T²) | β₀=1, β₁=2 | **38** | 48 | **36** | 41 |
| linked rings | β₀=2, β₁=2 | 66 | **65** | **17** | 42 |
| 5 blobs (R¹⁰) | β₀=5, β₁=0 | **33** | 37 | 378 | **370** |

DiRe wins 6 out of 8 comparisons, preserving both connected components (β₀) and loops (β₁) significantly better than cuML UMAP -- consistent with DiRe's theoretical guarantees for homological stability.

## Installation

### From PyPI (stable)

```bash
# Basic installation (CPU + PyTorch)
pip install dire-rapids

# With CUDA support
pip install dire-rapids[cuda]
```

### From Repository (development)

```bash
git clone https://github.com/sashakolpakov/dire-rapids.git
cd dire-rapids

pip install -e .          # CPU + PyTorch
pip install -e .[cuda]    # With CUDA support
pip install -e .[dev]     # Development (testing + dev tools)
```

#### With RAPIDS Support (Optional, GPU only)

First, install RAPIDS following the [official instructions](https://docs.rapids.ai/install/).
```bash
pip install -e .[rapids]
```

## Quick Start [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sashakolpakov/dire-rapids/blob/main/benchmarking/dire_rapids_benchmarks.ipynb)

```python
from dire_rapids import DiRePyTorch, DiRePyTorchMemoryEfficient
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=1_000, centers=12, n_features=10, random_state=42)

# Standard PyTorch backend
reducer = DiRePyTorch(n_components=2, n_neighbors=16, verbose=True)
X_embedded = reducer.fit_transform(X)

# Memory-efficient backend (recommended for large datasets)
reducer = DiRePyTorchMemoryEfficient(n_components=2, n_neighbors=16, verbose=True)
X_embedded = reducer.fit_transform(X)
```

![12 blobs with 100k points embedded in dimension 2](images/blobs_layout.png)

### Custom Distance Metrics

DiRe Rapids supports custom distance metrics for k-nearest neighbor computation while keeping layout forces Euclidean:

```python
# L1 (Manhattan) distance for k-NN
reducer = DiRePyTorch(metric='(x - y).abs().sum(-1)', n_neighbors=32)
X_embedded = reducer.fit_transform(X)

# Cosine distance via callable
def cosine_distance(x, y):
    return 1 - (x * y).sum(-1) / (x.norm(dim=-1, keepdim=True) * y.norm(dim=-1, keepdim=True) + 1e-8)

reducer = DiRePyTorch(metric=cosine_distance, n_neighbors=32)
X_embedded = reducer.fit_transform(X)
```

**Supported metric types:** `None` / `'euclidean'` / `'l2'` (default), string tensor expressions, or callable functions taking `(x, y)` tensors.

### Available Backends

- **DiRePyTorch** -- Standard PyTorch implementation with adaptive chunking
- **DiRePyTorchMemoryEfficient** -- FP16 support, point-by-point force computation, PyKeOps lazy tensors for repulsion
- **DiReCuVS** -- RAPIDS cuVS backend for massive-scale datasets

### Auto Backend Selection

```python
from dire_rapids import create_dire

# Auto-select optimal backend
# Priority: cuVS > PyTorchMemoryEfficient > PyTorch > CPU
reducer = create_dire(n_neighbors=32, verbose=True)
X_embedded = reducer.fit_transform(X)

# Force memory-efficient backend with FP16
reducer = create_dire(memory_efficient=True, use_fp16=True)
X_embedded = reducer.fit_transform(X)
```

## Betti Curves / Topology

The `betti_curve` module computes **filtered Betti curves** that track topological features across filtration thresholds. It builds an atlas complex from the kNN graph and computes Betti numbers (beta\_0 for connected components, beta\_1 for loops) via Hodge Laplacian eigenvalues.

```python
from dire_rapids.betti_curve import compute_betti_curve

# Automatic backend selection (GPU if CuPy available, else CPU/SciPy)
result = compute_betti_curve(X, k_neighbors=20, n_steps=50)

print(result['filtration_values'])  # filtration thresholds
print(result['beta_0'])             # connected components at each step
print(result['beta_1'])             # 1-cycles (loops) at each step
```

Both CPU (SciPy sparse + ARPACK) and GPU (CuPy sparse + cuSOLVER) backends are available, with automatic fallback.

## ReducerRunner Framework

General-purpose framework for running and comparing dimensionality reduction algorithms. See [benchmarking/dire_rapids_benchmarks.ipynb](benchmarking/dire_rapids_benchmarks.ipynb) for complete examples.

```python
from dire_rapids.utils import ReducerRunner, ReducerConfig
from dire_rapids import create_dire

config = ReducerConfig(
    name="DiRe",
    reducer_class=create_dire,
    reducer_kwargs={"n_neighbors": 16},
    visualize=True,
    max_points=10000
)

runner = ReducerRunner(config=config)
result = runner.run("sklearn:digits")
result = runner.run("openml:mnist_784")
```

**Data sources:** `sklearn:name`, `openml:name`, `cytof:name`, `dire:name` (geometric datasets), `file:path` (.csv, .npy, .npz, .parquet).

## Metrics Module

Evaluation metrics for dimensionality reduction quality:

```python
from dire_rapids.metrics import evaluate_embedding

results = evaluate_embedding(data, layout, labels, compute_topology=True)
print(f"Stress: {results['local']['stress']:.4f}")
print(f"SVM accuracy: {results['context']['svm'][1]:.4f}")
print(f"DTW beta_0: {results['topology']['metrics']['dtw_beta0']:.6f}")
print(f"DTW beta_1: {results['topology']['metrics']['dtw_beta1']:.6f}")
```

**Metrics:** distortion (stress, neighborhood preservation), context (SVM/kNN accuracy), topology (DTW distances between Betti curves). See [METRICS_README.md](METRICS_README.md) for details.

## Testing

```bash
# CPU tests (CI)
pytest tests/test_cpu_basic.py tests/test_reducer_runner.py -v

# Full test suite
pytest tests/ -v
```

## Citation

If you use this work, please cite:

```bibtex
@misc{kolpakov-rivin-2025dimensionality,
  title={Dimensionality reduction for homological stability and global structure preservation},
  author={Kolpakov, Alexander and Rivin, Igor},
  year={2025},
  eprint={2503.03156},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2503.03156}
}
```

## Requirements

- Python 3.10--3.13
- PyTorch 2.0+
- PyKeOps 2.1+
- NumPy, SciPy, scikit-learn
- (Optional) CUDA 12.x+ for GPU acceleration
- (Optional) RAPIDS 23.08+ for cuVS backend
- (Optional) CuPy for GPU-accelerated Betti curves

<p align="center">
  <a href="https://submitaitools.org/github-com-sashakolpakov-dire-rapids/">
    <img src="https://submitaitools.org/static_submitaitools/images/submitaitools.png"
         alt="DiRe-RAPIDS: Fast Dimensionality Reduction on the GPU" height="60" />
  </a>
</p>
