# Tests

## Overview

Test suite for DiRe-Rapids covering CPU/GPU backends, memory efficiency, scaling, and topological metrics.

## Test Files

### Core Functionality
- `test_cpu_basic.py` - Basic CPU functionality (used in CI)
- `test_reducer_runner.py` - ReducerRunner framework tests (used in CI)
- `test_create_dire.py` - Backend selection and factory function
- `test_comprehensive.py` - Comprehensive test suite for all components (MNIST 70k)

### Scaling and Performance
- `test_memory_usage.py` - Memory-efficient processing
- `test_scaling_high_dim.py` - High-dimensional scaling tests
- `test_cuvs_backend.py` - cuVS backend functionality
- `test_cuvs_scaling.py` - cuVS scaling tests
- `test_cuvs_1000d.py` - 1000D high-dimensional tests
- `test_atlas_scaling.py` - Atlas approach scaling tests

### Topological Metrics
- `test_atlas_approach.py` - H0/H1 computation using atlas approach (circle tests)
- `test_integrated_atlas.py` - Integrated atlas topology tests
- `test_betti_metrics_circle.py` - Betti curve comparison on circle dataset
- `test_betti_metrics_blobs.py` - Betti curve comparison on blob dataset

### Integration Tests
- `test_cuml_pca.py` - cuML PCA integration
- `test_matrix_structure.py` - Matrix structure and sparsity tests

## Running Tests

### CI Tests (CPU only)
```bash
pytest tests/test_cpu_basic.py tests/test_reducer_runner.py -v
```

### All Tests
```bash
pytest tests/ -v
```

### With Coverage
```bash
pytest tests/test_cpu_basic.py -v --cov=dire_rapids --cov-report=term-missing
```

## CI Usage

GitHub Actions CI runs `test_cpu_basic.py` and `test_reducer_runner.py` on every push/PR to main.

See `.github/workflows/tests.yml` for configuration.
