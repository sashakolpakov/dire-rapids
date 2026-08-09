# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Density visualization for large embeddings**: `DiRePyTorch.visualize` and `ReducerRunner` now switch large 2D embeddings from per-point WebGL scatter to a binned 2D-histogram density so the figure payload stays bounded regardless of point count (binning is done server-side with `np.histogram2d`; only a fixed `n_bins × n_bins` grid is shipped). Categorical labels render as a per-category density overlay (one filled-contour layer per class); unlabeled data renders a count heatmap and continuous labels a mean-value heatmap. Controlled by `mode` (`'auto'` | `'scatter'` | `'density'`) and `density_threshold`. Exposed as the shared, public `dire_rapids.build_embedding_figure` helper.
- **cuVS all-neighbors graph builder**: `DiReCuVS` can use RAPIDS 26.06's
  purpose-built all-neighbors API, including host-backed partitioned builds,
  configurable NN-descent/brute-force/IVF-PQ local algorithms, and optional
  multi-GPU resources.
- **All-neighbors tests**: Added CPU-safe routing and validation coverage plus
  GPU recall, self-removal, and out-of-core graph-contract tests.
- **Fitted backend diagnostics**: Reducers now expose effective k-NN/cuVS
  policy, graph/initialization/layout timings, and vectorized-force fallback
  counts through fitted attributes and `get_diagnostics()`.

### Changed
- **RAPIDS 26.06 baseline**: Added CUDA 12 and CUDA 13 extras pinned to the
  26.06 release line, Python 3.14 metadata, and a reproducible
  RAPIDS 26.06/CUDA 13/Python 3.14 container configuration.
- **cuVS all-neighbors policy**: The RAPIDS reducer preserves the released
  automatic index-and-search policy. All-neighbors is explicit opt-in until
  frozen downstream quality gates justify changing existing embeddings.
- **H100 all-neighbors validation**: A held-fixed 10x/arXiv A/B run found
  dataset-dependent runtime (about 26% slower on full 10x and 1.91x faster on
  full arXiv), graph overlap of 0.623 and 0.839, and mixed downstream quality.
  It remains a viable explicit option, particularly for arXiv-like workloads,
  but was not promoted to the default; the reproducibility harness stays on
  its separate test branch.
- **Atlas topology default**: `compute_betti_curve` now selects GPU then CPU
  Atlas by default. Ripser remains available with `prefer_ripser=True` as an
  explicit reference backend.
- **GPU graph handoff**: Reuses a DLPack-backed PyTorch view of the cuVS graph
  during layout optimization instead of uploading the host copy again.

### Fixed
- **RAPIDS 26.06 parameter compatibility**: Build and search overrides are now
  merged before constructing cuVS parameter extension objects, which do not
  implement `dict.update()`.
- **CUDA 13 import order**: The container and setup guide preload the conda
  `libnvJitLink.so.13` and check both PyTorch/cuML import orders.
- **Repeated-fit graph cache**: Invalidates and rebuilds the DLPack/PyTorch
  neighbor tensor when a reducer is fitted again, including the
  memory-efficient force path.
- **Metric graph contracts**: Preserves the input origin for cosine and
  inner-product normalization and keeps Euclidean distance scales consistent
  between the all-neighbors and legacy cuVS paths.

### Removed
- **Unsupported topology preset**: Withdrew the public `TOPOLOGY_TUNED` export
  after a paired held-out Atlas audit found worse discrepancy in 10 of 12
  dataset-by-homology comparisons. The original fixed-sample Ripser selection
  did not support its general name.

## [0.3.2] - 2026-06-03

### Fixed
- **High-dimensional k-NN fallback safety**: Avoided unsafe cuVS-to-standard-PyTorch fallback chunks for datasets above the cuVS dimensional limit.
- **Named metric memory use**: Switched PyTorch `cosine`, `inner_product`, and `sqeuclidean` k-NN paths to matrix-multiply implementations instead of broadcast tensor materialization.
- **Chunk sizing**: Accounted for arbitrary custom metric broadcast tensors in automatic k-NN chunk estimation.

### Changed
- **Auto backend selection**: Prefer memory-efficient PyTorch for large/high-dimensional or cuVS-unsupported metric cases.
- **k-NN controls**: Documented package-level version references for the `0.3.2` release.

## [0.3.1] - 2026-05-31

### Added
- **k-NN backend selector**: Added `knn_backend` to separate reducer implementation selection from internal k-NN engine selection. `knn_backend='auto'` preserves existing heuristics, while explicit `pytorch`, `pykeops`, and `cuvs` requests are strict.

### Changed
- **Documentation**: Clarified the distinction between `backend` and `knn_backend` across README, Sphinx docs, benchmark docs, and test docs.

## [0.3.0] - 2026-04-25

### Added
- **Test extras**: Added pandas and plotly to test dependencies for CI visualization tests
- **Presets (subsequently withdrawn)**: Added the former `TOPOLOGY_TUNED`
  fixed-sample Ripser study configuration

### Changed
- **Python version**: Updated minimum to 3.10 (was 3.9)
- **Documentation**: Fixed Python version requirement

## [0.3.0] - 2025-10-28

### Added
- **Topology metrics**: Added comprehensive topological evaluation via persistence homology
  - Betti curve computation for β₀ and β₁ homology groups
  - DTW distance metrics between Betti curves for layout quality assessment
  - Fast k-NN atlas (combinatorial Nyström) for Hodge Laplacian computation
  - Efficient persistent homology computation from Hodge Laplacian
- **ReducerRunner framework**: Moved `ReducerRunner` and `ReducerConfig` to main package (`dire_rapids.utils`)
  - General-purpose framework for running dimensionality reduction algorithms
  - Automatic data loading from multiple sources (sklearn, OpenML, CyTOF, local files)
  - Built-in visualization support with WebGL for large datasets
  - Reducer comparison utilities
- **Enhanced metrics module**: Comprehensive evaluation capabilities
  - Distortion metrics (stress, neighborhood preservation)
  - Context metrics (SVM/kNN classification accuracy preservation)
  - Topological metrics (Betti curves, DTW distances)
- **Documentation improvements**:
  - Enhanced API documentation with Sphinx
  - New examples: `metrics_swiss_roll.py` demonstrating topology metrics
  - Updated `METRICS_README.md` with topology metrics documentation
  - Improved benchmarking notebook with new API
- **Testing infrastructure**:
  - Added `test_comprehensive.py` for full pipeline testing
  - New topology-specific tests: `test_betti_metrics_blobs.py`, `test_betti_metrics_circle.py`
  - Enhanced `TEST_INSTRUCTIONS.md` documentation

### Fixed
- **CAGRA backend**: Fixed multiple bugs in CAGRA API calls
  - Corrected parameter names in CAGRA function calls
  - Enforced stable-only CAGRA calls for reliability
  - Improved error handling in cuVS integration
- **DiReCuVS backend**: Enhanced RAPIDS integration
  - Better parameter validation
  - Improved memory management for large-scale datasets
  - Fixed compatibility issues with cuML/cuVS 23.0+

### Changed
- **Metrics module refactoring**: Significant cleanup and optimization
  - Removed redundant backup files (`metrics_backup.py`)
  - Streamlined topology computation workflow
  - Consolidated atlas-based metrics into main metrics module
  - Improved performance and memory efficiency
- **Code cleanup**:
  - Removed obsolete atlas implementations (`atlas_cpu.py`, `atlas_gpu.py`)
  - Removed redundant metrics files (`metrics_atlas.py`, `metrics_atlas_source.py`)
  - Removed outdated examples (`metrics_evaluation.py`, `metrics_simple_test.py`)
  - Removed obsolete test files (`test_compare_backends.py`)
  - Cleaned up benchmarking scripts
- **Visualization**: Use `scattergl` for large dataset visualization to avoid rendering failures in Jupyter notebooks
- **Dependencies**: Updated requirements for improved compatibility
  - Added `fastdtw>=0.3.0` for DTW distance computation
  - Updated test dependencies for better coverage

### Removed
- Obsolete atlas implementations (moved to integrated approach)
- Redundant metrics backup files
- Outdated example scripts
- Obsolete benchmarking utilities (replaced by ReducerRunner)

## [0.1.0] - 2025-09-04

Initial release with core functionality:
- DiRePyTorch: Standard PyTorch implementation
- DiRePyTorchMemoryEfficient: Memory-optimized implementation with FP16 support
- DiReCuVS: RAPIDS cuVS/cuML backend for massive-scale datasets
- Basic metrics for distortion and context preservation
- Examples and benchmarking utilities

[Unreleased]: https://github.com/sashakolpakov/dire-rapids/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/sashakolpakov/dire-rapids/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/sashakolpakov/dire-rapids/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/sashakolpakov/dire-rapids/compare/v0.2.0...v0.3.0
[0.1.0]: https://github.com/sashakolpakov/dire-rapids/releases/tag/v0.1.0
