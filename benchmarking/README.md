# DiRe-Rapids Benchmarking Documentation

## Overview

This directory contains comprehensive benchmarking results and performance analysis for DiRe-Rapids, focusing on scalability with high-dimensional data. The benchmarks compare reducer implementations (PyTorch, memory-efficient PyTorch, RAPIDS cuVS/cuML) and k-NN engines (`pytorch`, `pykeops`, `cuvs`) that enable processing datasets with millions of points in up to 1000 dimensions.

## Withdrawn topology preset historical audit

`bench_topology_historical.py` is the frozen issue #14 resolution harness. It
does not restore or export the withdrawn preset. Instead, it privately pins the
former parameters and compares them with default DiRe at:

- `9117dc45a3e130fa1d636dfd181f3e97960c5b3b`, where the preset was added;
- `293b622cc79fa8ea6fd5b54009e0930e3385b22f`, used by the paired audit;
- `471ac168eb2e6638a84f700fe077f29f20e24488`, the original PR #12 head.

Every revision uses the same materialized six-dataset arrays, layout seeds
42--61, and seed-paired 1,000-row subsets. Materialization and loading both
enforce one predeclared array SHA-256 per dataset, so dependency or source
drift cannot silently create a different comparison. The selected labels and
row order reproduce the retained audit exactly for all six datasets. The
evaluator is loaded from the hash-verified `293b622` `betti_curve.py` file for
every revision. Both direct rank-based kNN Atlas and Ripser H0/H1 Betti-DTW
discrepancies are retained. The legacy comparisons explicitly force exact-flat
index/search. The PR #12 head is also run through its proposed automatic
all-neighbors path.

Run the complete matrix inside the repository's RAPIDS 26.06 container on an
NVIDIA H100. The runner rejects a different GPU class so the historical
comparison cannot be mistaken for the requested H100 reproduction:

```bash
benchmarking/run_topology_historical.sh /workspace/issue14-historical-results
```

The command is resumable. It appends one checksummed JSON record per fit and
refuses to summarize until all 960 records are present and their dataset and
subset identities pair exactly across configurations and revisions. Shared
reference Atlas and Ripser curves are content-hashed, and their hashes must
also pair across every record. The final summary predeclares a material
implementation change as a change in the paired preset-minus-default gap
exceeding 5% of the historical default mean, with a descriptive paired 95%
interval excluding zero. Raw records, reference curves, and manifests must be
retained with the three summary files before drawing or publishing the
historical-regression conclusion.

If the exact frozen arrays have already been archived under
`frozen-datasets/`, the runner verifies and reuses their manifest rather than
regenerating them. This is intentional: transcendental preprocessing can
differ by a few float32 ULPs across NumPy/libm builds, while the audit requires
identical input bytes. A missing or altered array still fails the predeclared
array and file hashes before any fit starts.

The runner imports its required PyTorch runtime before importing each pinned
historical checkout. Commit `293b622` used a cuML-before-PyTorch package import
order that was needed by RAPIDS 26.04, but that historical loader order leaves
cuBLAS uninitialized in the frozen RAPIDS 26.06 environment. Loading PyTorch
first fixes only the shared-library order; the historical source, reducer
parameters, and requested/effective cuVS graph policies remain pinned and are
recorded unchanged.

### Crossed Atlas/Ripser preset search

`bench_topology_preset_search.py` searches for replacement presets without
reusing the six validation datasets. It freezes four disjoint OpenML datasets
(mfeat-factors, satimage, pendigits, and isolet), evaluates the same bounded
Sobol candidates with both the fixed Atlas and Ripser metrics, and selects
separate Atlas, Ripser, and compromise candidates. On every tuning dataset, a
candidate is eligible only when its mean 15-NN accuracy, local-neighbor
retention, and sampled global-distance Spearman correlation are no more than
one percentage point below default DiRe, and local stress is no more than 10%
higher. The search never fits UMAP or t-SNE; those thresholds come from the
retained, hash-pinned baseline fixture.

The search requests the released `cuvs_knn_method="auto"` policy and asserts
that the current guarded policy resolves to `index_search`, recording the
effective index type for every fit. The historical audit separately covers
PR #12 head's proposed automatic all-neighbors behavior; candidate tuning does
not silently opt into that unreleased policy change.

```bash
python benchmarking/bench_topology_preset_search.py prepare \
  --output issue14-preset-search/tuning-datasets
python benchmarking/bench_topology_preset_search.py run \
  --source-root . \
  --evaluator-source /path/to/293b622/dire_rapids/betti_curve.py \
  --tuning-root issue14-preset-search/tuning-datasets \
  --reference-cache issue14-preset-search/reference-cache \
  --output issue14-preset-search/raw/search.jsonl \
  --sobol-count 32
python benchmarking/bench_topology_preset_search.py summarize \
  --input issue14-preset-search/raw/search.jsonl \
  --manifest issue14-preset-search/raw/search.manifest.json \
  --output issue14-preset-search/summary/search-summary.json
python benchmarking/bench_topology_preset_search.py validate \
  --source-root . \
  --evaluator-source /path/to/293b622/dire_rapids/betti_curve.py \
  --frozen-root issue14-historical-results/frozen-datasets \
  --search-summary issue14-preset-search/summary/search-summary.json \
  --reference-cache issue14-preset-search/validation-reference-cache \
  --output issue14-preset-search/raw/validation.jsonl \
  --layout-seeds 42
python benchmarking/bench_topology_preset_search.py summarize-validation \
  --input issue14-preset-search/raw/validation.jsonl \
  --manifest issue14-preset-search/raw/validation.manifest.json \
  --baselines tests/data/topology_umap_tsne_atlas_baselines.json \
  --default-audit tests/data/topology_historical_h100_audit.tar.gz \
  --output issue14-preset-search/summary/validation-summary.json
```

The first validation pass deliberately uses only seed 42, which is the seed
for every archived canonical UMAP/t-SNE embedding. If a Ripser candidate is
competitive in that screen, rerun only that candidate with `--candidate NAME
--layout-seeds 42:62` and fit only the strongest missing Ripser comparator(s)
needed for a repeated claim. Atlas already has retained repeat distributions,
so its validation summary uses overlapping seeds and reports paired 95%
intervals immediately.

Passing `--default-audit` also performs a fully paired Atlas/Ripser comparison
against the retained 20-seed current-default records. Dataset arrays, topology
subsets, reference curves, and exact-flat effective graph policy must all match
before those records are compared.

If the broad Sobol design produces no quality-feasible candidate, a bounded
refinement can be run with `--design local`. It evaluates 18 one-parameter
changes around default (plus the default control), making any safe improvement
interpretable and adding only 152 fits. The quality gates are not relaxed.

After the coarse search identified `spread=0.8` as the only setting that
transferred, `--design atlas-fine` resolves a distinct Atlas preset with seven
nearby candidates, the current default control, and the retained Ripser
incumbent. It adds 72 tuning fits; only its quality-feasible Atlas winner is
eligible for held-out confirmation.

The completed H100 run retained 272 broad-search, 152 local-refinement, 12
seed-42 screening, 120 repeated crossed-validation, 72 Atlas-refinement, and
120 Atlas-confirmation records. No broad candidate passed every safeguard.
The coarse local search selected `spread=1.2` for Atlas and `spread=0.8` for
Ripser, but only `spread=0.8` transferred. The focused refinement then selected
`max_iter_layout=96` for Atlas, distinct from Ripser's 128. On six untouched
datasets and 20 paired seeds, `ATLAS_TUNED` had geometric ratios of 0.908 to
default and 0.884 to the retained cell-wise UMAP/t-SNE envelope; it won 11/12
default cells and 6/12 comparator cells. `RIPSER_TUNED` retained its 0.908
ratio and 11/12 wins against default, plus a 0.891 canonical seed-42 ratio and
6/12 wins against the UMAP/t-SNE envelope. The failed Atlas-selected
`spread=1.2` candidate is not exported.

An immediate identical seed-42 reproducibility check found nonzero Atlas drift
in 2/24 topology metrics, with a 0.348 maximum absolute cell change and only a
0.38% aggregate Atlas change; Ripser metrics matched exactly. The preset claims
therefore use the 20-seed paired distributions, not bitwise GPU equality.

The complete raw records, manifests, tuning reference curves, and summaries
are retained in `tests/data/topology_preset_search_h100_audit.tar.gz`; the
compact decision summary is unpacked under
`tests/data/topology_preset_search_h100/`.

## Key Achievements

**High throughput for large datasets:**
- Optimized PyTorch: 500,000 points in <30 seconds (1000D)
- cuVS k-NN engine: 1,500,000+ points with <500MB GPU memory (1000D)

## Reducer and k-NN Engine Comparison

`backend` selects the reducer implementation created by `create_dire`. `knn_backend` selects the internal k-nearest-neighbor engine. Leave `knn_backend='auto'` to preserve the built-in heuristics, or set it explicitly to force an engine. Explicit requests are strict: unavailable or unsupported engines raise instead of silently falling back.

### PyTorch Implementation with PyTorch k-NN

**Strengths:**
- Excellent for small to medium datasets (<100K points)
- Exact k-NN computation via `knn_backend='pytorch'`
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

### cuVS k-NN Engine (GPU-Accelerated Approximate k-NN)

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

### Selection Strategy

```python
def select_dire_config(n_samples, n_dims):
    if n_samples < 100000:
        return {'backend': 'pytorch', 'knn_backend': 'pytorch'}
    elif n_samples < 250000 and n_dims < 500:
        return {'backend': 'pytorch', 'memory_efficient': True, 'knn_backend': 'pytorch'}
    else:
        return {'backend': 'auto', 'knn_backend': 'cuvs'}
```

**Note:** `create_dire()` automatically selects a reducer implementation based on hardware. The k-NN engine can be left on `knn_backend='auto'` or forced independently, for example `create_dire(backend='pytorch', knn_backend='pytorch')` for exact PyTorch k-NN or `create_dire(knn_backend='cuvs')` for strict RAPIDS cuVS k-NN.

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
- Auto k-NN selection switches from PyKeOps to PyTorch for dimensions ≥ 200 (10-200x speedup)
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
- Use `knn_backend='pytorch'` for exact k-NN
- Excellent performance across all dimensions
- Real-time/interactive applications possible

**For datasets 100K-500K points:**
- PyTorch for dimensions <500
- cuVS k-NN for dimensions ≥500 or when memory is limited
- Consider FP16 for additional speedup

**For datasets >500K points:**
- cuVS k-NN is strongly recommended
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
# Use custom metrics for domain-specific similarity.
# Custom tensor expressions/callables run on the PyTorch/PyKeOps paths.
reducer = DiRePyTorch(metric='(x - y).abs().sum(-1)', knn_backend='pytorch')

# Cosine similarity for normalized features
def cosine_distance(x, y):
    return 1 - (x * y).sum(-1) / (x.norm(dim=-1, keepdim=True) * y.norm(dim=-1, keepdim=True) + 1e-8)
reducer = DiRePyTorch(metric=cosine_distance, knn_backend='pytorch')
```

**Performance Impact of Custom Metrics:**
- String expressions: ~5-10% overhead vs Euclidean
- Callable functions: ~10-15% overhead vs Euclidean
- cuVS supports named native metrics only; forced `knn_backend='cuvs'` raises for custom expressions/callables
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
from dire_rapids.utils import ReducerRunner, ReducerConfig
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
from dire_rapids.utils import ReducerConfig
from dire_rapids import create_dire

# Compare default reducers (DiRe, cuML UMAP, cuML TSNE)
results = compare_reducers(
    "sklearn:blobs",
    dataset_kwargs={"n_samples": 1000, "n_features": 50},
    metrics=['distortion', 'context', 'topology'],
    subsample_threshold=0.1  # Use 10% of data for topological metrics
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
Comprehensive benchmarking notebook:
- **ReducerRunner** for data loading and algorithm execution
- **compare_reducers** for comparing DiRe, UMAP, and t-SNE
- Comprehensive metrics:
  - Distortion: stress, neighbor preservation
  - Context: SVM/kNN classification accuracy
  - Topology: DTW distances between Betti curves

For financial market data, see `../examples/finance_analysis_notebook.ipynb`.

### ReducerRunner (in dire_rapids.utils)
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
Framework for comparing reducers:
- Automatic quality metrics:
  - Distortion: stress, neighbor preservation
  - Context: SVM/kNN classification accuracy
  - Topology: DTW distances between Betti curves (β₀ and β₁)
- Default reducers: DiRe, cuML UMAP, cuML TSNE
- Flexible subsampling for topology metrics

### benchmark_mnist.py
Tests DIRE performance on MNIST dataset with various configurations:
- Compares reducer/k-NN configurations (PyTorch, cuVS)
- Tests different precision levels (FP32, FP16)
- Measures memory usage and throughput

### profile_pipeline.py
Profiles the complete DIRE pipeline:
- k-NN computation timing
- Force calculation overhead
- Memory allocation patterns
- GPU utilization metrics

## Summary

The benchmarking results demonstrate that DiRe-Rapids can efficiently handle high-dimensional data at scale through:
- Separate reducer implementation and k-NN engine selection
- Memory-aware processing with automatic fallbacks
- FP16 optimization for modern GPUs
- Approximate k-NN for massive datasets

For typical use cases (up to 500K points), the PyTorch implementation with exact PyTorch k-NN provides excellent performance. For larger datasets or memory-constrained environments, the cuVS k-NN engine enables processing millions of points with acceptable accuracy tradeoffs.
