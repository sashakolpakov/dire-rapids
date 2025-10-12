# DiRe-Rapids Benchmarking Documentation

## Overview

This directory contains comprehensive benchmarking results and performance analysis for the DIRE-JAX implementation, focusing on scalability with high-dimensional data. The benchmarks compare different backend implementations (PyTorch, PyKeOps, cuVS) and document optimization strategies that enable processing of datasets with millions of points in up to 1000 dimensions.

## Key Achievements

**High throughput for large datasets:**
- Optimized PyTorch: 500,000 points in <30 seconds (1000D)
- cuVS Backend: 1,500,000+ points with <500MB GPU memory (1000D)

## Backend Comparison

### PyTorch Backend

**Strengths:**
- Excellent for small to medium datasets (<100K points)
- Exact k-NN computation with PyKeOps
- Efficient tensor operations using GPU tensor cores

**Limitations:**
- Exact k-NN needs $O(N^2)$ memory
- Force layout memory overflow for large datasets

**Performance (1000D, H100 GPU):**
| Points | Time | Throughput | Memory | Status |
|--------|------|------------|--------|--------|
| 50K | 2.3s | 76K pts/s | 11GB | Excellent |
| 100K | 3.5s | 138K pts/s | 41GB | Excellent |
| 250K | 10s | 60K pts/s | 52GB | Good |
| 500K | 28s | 31K pts/s | 53GB | Acceptable |
| 1M+ | >60s | <15K pts/s | >70GB | Impractical |

### cuVS Backend (GPU-Accelerated Approximate k-NN)

**Strengths:**
- Handles millions of points with O(N) memory complexity
- 100x better memory efficiency than exact methods
- Scales to 1.5M+ points in 1000D
- Automatic index type selection based on data scale

**Limitations:**
- ~5% accuracy tradeoff (95% recall vs exact k-NN)
- Index building overhead for small datasets

**Performance (1000D, H100 GPU):**
| Points | Time | Memory | Throughput | Index Type |
|--------|------|--------|------------|------------|
| 250K | 9s | 171MB | 28K pts/s | IVF-Flat |
| 500K | 24s | 300MB | 21K pts/s | IVF-Flat |
| 750K | 43s | 400MB | 17K pts/s | IVF-Flat |
| 1M | 63s | 450MB | 16K pts/s | IVF-Flat |
| 1.5M | 118s | 500MB | 13K pts/s | IVF-Flat |

### Backend Selection Strategy

```python
def select_backend(n_samples, n_dims):
    if n_samples < 100000:
        return 'pytorch'  # Fast exact k-NN for small data
    elif n_samples < 250000 and n_dims < 500:
        return 'pytorch'  # Still manageable with exact methods
    else:
        return 'cuvs'  # Switch to approximate for scale
```

## Optimization Techniques

### 1. Memory-Aware Chunking
- Dynamic chunk size adjustment based on available GPU memory
- Prevents OOM errors by using only 20% of available memory
- Graceful fallback to point-by-point processing when necessary

### 2. FP16 Precision
- 2x memory reduction enables larger datasets
- 2.6-14x speedup on modern GPUs (H100: 2000 TFLOPS FP16 vs 67 TFLOPS FP32)
- 97% k-NN accuracy maintained (sufficient for DIRE)

### 3. Backend-Specific Optimizations

**PyTorch Optimizations:**
- Switch from PyKeOps to PyTorch for dimensions ≥ 200 (10-200x speedup)
- Fixed force computation bug (attraction only between k-NN neighbors)
- Efficient use of tensor cores for matrix operations

**cuVS Index Selection:**
- Flat index for <50K points (exact search)
- IVF-Flat for 50K-500K points or high dimensions (balanced)
- IVF-PQ for 500K-5M points (compressed)
- CAGRA for >5M points (graph-based)

## Practical Recommendations

### Use Case Guidelines

**For datasets <100K points:**
- Use PyTorch backend for exact k-NN
- Excellent performance across all dimensions
- Real-time/interactive applications possible

**For datasets 100K-500K points:**
- PyTorch for dimensions <500
- cuVS for dimensions ≥500 or when memory is limited
- Consider FP16 for additional speedup

**For datasets >500K points:**
- cuVS backend required
- Accept ~5% accuracy tradeoff for massive scalability
- Consider dimension reduction as preprocessing step

### Optimization Strategies for Large Scale

**1. Two-Stage Dimension Reduction:**
```python
# Reduce dimensions first
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X_1000d)

# Then apply DIRE
embedding = dire.fit_transform(X_reduced)
```

**2. Sampling and Projection:**
```python
# Embed representative subset
sample_idx = np.random.choice(n, 100000)
embedding_sample = dire.fit_transform(X[sample_idx])

# Project remaining points using learned mapping
```

**3. Hierarchical Processing:**
- Process data in batches
- Use landmarks for initial embedding
- Refine with local neighborhoods

**4. Custom Distance Metrics:**
```python
# Use custom metrics for domain-specific similarity
# L1 metric often performs better for high-dimensional sparse data
reducer = DiRePyTorch(metric='(x - y).abs().sum(-1)')

# Cosine similarity for normalized features
def cosine_distance(x, y):
    return 1 - (x * y).sum(-1) / (x.norm(dim=-1, keepdim=True) * y.norm(dim=-1, keepdim=True) + 1e-8)
reducer = DiReCuVS(metric=cosine_distance)
```

**Performance Impact of Custom Metrics:**
- String expressions: ~5-10% overhead vs Euclidean
- Callable functions: ~10-15% overhead vs Euclidean
- Layout forces remain Euclidean (optimized) regardless of k-NN metric

## Computational Complexity Analysis

### k-NN Computation
- Exact methods: O(N² × D) time, O(N²) memory
- Approximate methods: O(N × log(N) × D) time, O(N) memory

### Practical Limits (1000D data, 80GB GPU)

**PyTorch (Exact):**
- Maximum points: ~500K
- Bottleneck: Distance matrix memory
- Time complexity dominates beyond 250K points

**cuVS (Approximate):**
- Tested up to: 1.5M points
- Theoretical maximum: 5-10M points
- Bottleneck: Search time (linear with N)

## Using ReducerRunner

The **ReducerRunner** framework provides a unified interface for running and comparing dimensionality reduction algorithms. It replaces the previous `DiReRunner` and supports any sklearn-compatible reducer.

### Basic Usage

```python
from dire_rapids.dire_pytorch import ReducerRunner, ReducerConfig
from dire_rapids import create_dire

# Create configuration
config = ReducerConfig(
    name="DiRe",
    reducer_class=create_dire,
    reducer_kwargs={"n_neighbors": 16, "n_components": 2},
    visualize=True,
    categorical_labels=True
)

# Run on dataset
runner = ReducerRunner(config=config)

# Try different datasets
result = runner.run("sklearn:blobs")
result = runner.run("sklearn:digits")
result = runner.run("dire:sphere_uniform", dataset_kwargs={"n_features": 10, "n_samples": 1000})
result = runner.run("openml:mnist_784")
result = runner.run("cytof:levine13")
```

### Comparing Reducers

```python
from benchmarking.compare_reducers import compare_reducers, print_comparison_summary
from dire_rapids.dire_pytorch import ReducerConfig
from dire_rapids import create_dire

# Compare default reducers (DiRe, cuML UMAP, cuML TSNE)
results = compare_reducers(
    "sklearn:blobs",
    dataset_kwargs={"n_samples": 1000, "n_features": 50},
    metrics=['distortion', 'context', 'topology']
)
print_comparison_summary(results)

# Compare specific configurations
from cuml import UMAP
reducers = [
    ReducerConfig("DiRe-16", create_dire, {"n_neighbors": 16}),
    ReducerConfig("DiRe-32", create_dire, {"n_neighbors": 32}),
    ReducerConfig("UMAP", UMAP, {"n_neighbors": 15})
]
results = compare_reducers("sklearn:digits", reducers=reducers)
```

## Benchmark Scripts

### dire_rapids_benchmarks.ipynb
Comprehensive Jupyter notebook with benchmarking results:
- **ReducerRunner** framework for easy data loading and algorithm execution
- **compare_reducers** for comparing DiRe, UMAP, and t-SNE
- Performance on various datasets (MNIST, Levine-13, etc.)
- Visualization of embeddings with plotly
- Global and local structure preservation analysis
- Comprehensive metrics (distortion, context, topology)

### ReducerRunner (in dire_rapids.dire_pytorch)
General-purpose framework for dimensionality reduction:
- **ReducerRunner** class for running any sklearn-compatible reducer
- **ReducerConfig** dataclass for configuring reducers
- Automatic data loading from multiple sources:
  - sklearn datasets (blobs, digits, iris, wine, moons, swiss_roll, etc.)
  - OpenML datasets (by name or ID)
  - CyTOF datasets (levine13, levine32)
  - DiRe geometric datasets (disk_uniform, sphere_uniform, ellipsoid_uniform)
  - Local files (.csv, .npy, .npz, .parquet)
- Built-in visualization with plotly
- Supports both categorical and continuous labels

### compare_reducers.py
Framework for comparing multiple reducers:
- Run multiple reducers on the same dataset
- Automatic quality metrics computation
- Summary table generation
- Default set of reducers (DiRe, cuML UMAP, cuML TSNE)
- Custom reducer configurations

### benchmark_mnist.py
Tests DIRE performance on MNIST dataset with various configurations:
- Compares backends (PyTorch, PyKeOps, cuVS)
- Tests different precision levels (FP32, FP16)
- Measures memory usage and throughput

### profile_pipeline.py
Profiles the complete DIRE pipeline:
- k-NN computation timing
- Force calculation overhead
- Memory allocation patterns
- GPU utilization metrics

## Summary

The benchmarking results demonstrate that DIRE-JAX can efficiently handle high-dimensional data at scale through:
- Intelligent backend selection (PyTorch + cuVS + memory efficient)
- Memory-aware processing with automatic fallbacks
- FP16 optimization for modern GPUs
- Approximate k-NN for massive datasets

For typical use cases (up to 500K points), the PyTorch backend with optimizations provides excellent performance. For larger datasets or memory-constrained environments, the cuVS backend enables processing of millions of points with acceptable accuracy tradeoffs.
