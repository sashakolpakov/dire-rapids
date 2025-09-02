# DiRe-RAPIDS

PyTorch and RAPIDS-accelerated dimensionality reduction.

This package provides GPU-accelerated implementations of the DiRe (Dimensionality Reduction) algorithm using PyTorch and optionally NVIDIA RAPIDS (cuVS/cuML) for massive-scale datasets.

## Features

- **DiRePyTorch**: Memory-efficient PyTorch/PyKeOps backend for datasets >50K points
- **DiReCuVS**: Optional RAPIDS integration for massive datasets (millions of points)
- Automatic GPU memory management with adaptive chunking
- Optimized for CUDA GPUs

## Installation

```bash
# Basic installation (PyTorch backend only)
pip install dire-rapids

# With RAPIDS support
pip install dire-rapids[cuvs]
# or
conda install -c rapidsai -c conda-forge rapids=25.08
```

## Usage

```python
from dire_rapids import DiRePyTorch

# For large datasets on GPU
reducer = DiRePyTorch(dim=2)
embedding = reducer.fit_transform(data)

# With RAPIDS acceleration (if available)
from dire_rapids import DiReCuVS
reducer = DiReCuVS(dim=2)
embedding = reducer.fit_transform(data)
```

## Requirements

- Python >= 3.8
- PyTorch
- PyKeOps
- NumPy
- Optional: RAPIDS 25.08+ for cuVS/cuML acceleration