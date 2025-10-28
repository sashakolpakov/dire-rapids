# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-28

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

[0.2.0]: https://github.com/sashakolpakov/dire-rapids/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sashakolpakov/dire-rapids/releases/tag/v0.1.0
