# Tests

## Overview

Test suite for DiRe-Rapids covering CPU/GPU backends, memory efficiency, and scaling.

## Test Files

- `test_cpu_basic.py` - Basic CPU functionality (used in CI)
- `test_reducer_runner.py` - ReducerRunner framework tests (used in CI)
- `test_create_dire.py` - Backend selection and factory function
- `test_memory_usage.py` - Memory-efficient processing
- `test_scaling_high_dim.py` - High-dimensional scaling tests
- `test_cuml_pca.py` - cuML PCA integration
- `test_cuvs_backend.py` - cuVS backend functionality
- `test_cuvs_scaling.py` - cuVS scaling tests
- `test_cuvs_1000d.py` - 1000D high-dimensional tests

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
