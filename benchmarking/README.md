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

## Benchmark Scripts

### dire_benchmarks.ipynb
Comprehensive Jupyter notebook with benchmarking results:
- Comparison with UMAP and t-SNE
- Performance on various datasets (MNIST, Levine-13, etc.)
- Visualization of embeddings
- Global and local structure preservation analysis

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
