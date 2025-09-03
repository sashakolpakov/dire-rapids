# dire-rapids

PyTorch and RAPIDS accelerated dimensionality reduction

GPU-accelerated implementation of DiRe (Dimensionality Reduction) using PyTorch and optionally NVIDIA RAPIDS for massive-scale datasets.

## Installation

### From Repository

```bash
# Clone the repository
git clone https://github.com/sashakolpakov/dire-rapids.git
cd dire-rapids

# Basic installation (CPU + PyTorch)
pip install -e .

# With CUDA support
pip install -e .[cuda]

# For development (includes testing and dev tools)
pip install -e .[dev]
```

### With RAPIDS Support (Optional, GPU only)

```bash
# First install RAPIDS (requires CUDA 11.x or 12.x)
conda install -c rapidsai -c conda-forge -c nvidia rapids=24.10 python=3.11 cuda-version=11.8

# Then install dire-rapids with RAPIDS support
pip install -e .[rapids]
```

## Quick Start

```python
from dire_rapids import DiRePyTorch, DiRePyTorchMemoryEfficient
import numpy as np

# Generate sample data
X = np.random.randn(1000, 50).astype(np.float32)

# Standard PyTorch implementation
model = DiRePyTorch(n_components=2, n_neighbors=15)
X_embedded = model.fit_transform(X)

# Memory-efficient version (recommended for large datasets)
# Uses FP16, point-by-point forces, and aggressive memory management
model_efficient = DiRePyTorchMemoryEfficient(
    n_components=2, 
    n_neighbors=15,
    use_fp16=True,  # Use half precision for memory savings
    memory_fraction=0.15  # Conservative GPU memory usage
)
X_embedded_efficient = model_efficient.fit_transform(X)

print(X_embedded.shape)  # (1000, 2)
```

### Available Backends

- **DiRePyTorch**: Standard PyTorch implementation with adaptive chunking
- **DiRePyTorchMemoryEfficient**: Memory-optimized version with:
  - FP16 support for 2x memory savings
  - Point-by-point force computation
  - More aggressive memory management
  - PyKeOps LazyTensors for efficient repulsion (when available)
- **DiReCuVS** (optional): RAPIDS cuVS backend for massive-scale datasets

## Testing

```bash
# Run basic CPU tests
pytest tests/test_cpu_basic.py -v

# Run all tests
pytest tests/ -v
```

## Requirements

- Python 3.8-3.11
- PyTorch 2.0+
- PyKeOps 2.1+
- NumPy, SciPy, scikit-learn
- (Optional) CUDA 11.x+ for GPU acceleration
- (Optional) RAPIDS 23.08+ for cuVS backend