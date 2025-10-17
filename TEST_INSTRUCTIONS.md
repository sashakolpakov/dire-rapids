# Comprehensive Test Suite Instructions

## Overview

The `test_comprehensive.py` script runs a complete test suite for dire-rapids on the MNIST 70k dataset, testing both CPU and GPU implementations.

## Quick Start

### From Command Line

Run from the repository root:

```bash
python test_comprehensive.py
```

### From Jupyter Notebook

```python
%run test_comprehensive.py
```

## What Gets Tested

### 1. **Import Tests** (Critical)
   - All main modules can be imported
   - Dependencies are available

### 2. **DiRe CPU** (Critical)
   - Basic DiRe class on CPU
   - 5,000 sample subset
   - Validates output shape and values

### 3. **DiRe GPU** (Non-critical)
   - DiRe class with CUDA
   - 5,000 sample subset
   - Skipped if GPU unavailable

### 4. **DiReMemoryEfficient CPU** (Critical)
   - Memory-efficient implementation on CPU
   - 5,000 sample subset with chunking
   - Validates batching logic

### 5. **DiReMemoryEfficient GPU** (Non-critical)
   - Memory-efficient implementation with CUDA
   - 5,000 sample subset with chunking
   - Skipped if GPU unavailable

### 6. **Metrics CPU** (Critical)
   - Stress computation
   - Neighbor preservation
   - 2,000 sample subset

### 7. **Metrics GPU** (Non-critical)
   - GPU-accelerated metrics
   - Requires cuML and CuPy
   - Skipped if unavailable

### 8. **Full Metrics Evaluation CPU** (Critical)
   - Local metrics (stress, neighbor)
   - Context metrics (SVM, kNN)
   - 1,000 sample subset

### 9. **Atlas CPU** (Critical)
   - H0/H1 topology computation
   - Tested on synthetic circle
   - Validates Betti numbers

### 10. **Atlas GPU** (Non-critical)
   - GPU-accelerated atlas
   - Tested on synthetic circle
   - Skipped if CuPy unavailable

### 11. **Large Scale - DiRe MNIST 70k** (Critical)
   - Full 70,000 sample MNIST
   - Uses GPU if available, otherwise CPU
   - May take several minutes

### 12. **Large Scale - Memory Efficient MNIST 70k** (Critical)
   - Full 70,000 sample MNIST with chunking
   - Uses GPU if available, otherwise CPU
   - May take several minutes

## Test Behavior

- **Stops on first critical error**: If a critical test fails, execution stops immediately
- **Continues on non-critical errors**: GPU tests are non-critical and will be skipped if hardware/software unavailable
- **Verbose output**: Each test prints detailed progress
- **Summary at end**: Shows all test results and timing

## Output Format

Each test produces:
```
--------------------------------------------------------------------------------
TEST: Test Name
--------------------------------------------------------------------------------
[Detailed output from test]
✓ PASSED (12.34s)
```

Or on failure:
```
✗ FAILED (12.34s)
Error: [error message]
[Full traceback]
```

Final summary:
```
================================================================================
TEST SUMMARY
================================================================================
Total tests: 12
Passed: 11
Failed: 1
Total time: 123.45s

✓ 1. Import Tests                                                      0.12s
✓ 2. DiRe CPU                                                         45.67s
✗ 3. DiRe GPU                                                          1.23s
  Error: CUDA not available
...
================================================================================
```

## Requirements

### Minimum (for CPU tests):
- Python 3.8+
- PyTorch
- scikit-learn
- numpy
- scipy

### Full (for GPU tests):
- All minimum requirements
- CUDA-capable GPU
- CuPy
- cuML (RAPIDS)
- PyTorch with CUDA support

## Notes

- The script automatically downloads MNIST if not cached
- GPU tests gracefully skip if hardware unavailable
- Each test is independent and can be run individually
- Tests use different dataset sizes to balance speed vs thoroughness
- The script is designed to be removed in future versions (temporary testing)

## Troubleshooting

### "ModuleNotFoundError"
Install missing dependencies:
```bash
pip install torch scikit-learn numpy scipy
```

### "CUDA not available"
This is expected if you don't have a CUDA GPU. CPU tests will still run.

### "cuML not available"
This is expected without RAPIDS. GPU metrics tests will skip.

### Tests timeout
Large scale tests (11-12) can take 5-15 minutes each depending on hardware.

## Exit Codes

- `0`: All tests passed
- `1`: One or more critical tests failed
