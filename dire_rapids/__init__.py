# dire-rapids

"""
PyTorch and RAPIDS (cuVS/cuML) accelerated dimensionality reduction.

This package provides high-performance dimensionality reduction using the DiRe algorithm
with multiple backend implementations:

- **DiRePyTorch**: Standard PyTorch implementation for general use
- **DiRePyTorchMemoryEfficient**: Memory-optimized PyTorch implementation for large datasets
- **DiReCuVS**: RAPIDS cuVS/cuML accelerated implementation for massive datasets

The package automatically selects the best available backend based on system capabilities
and dataset characteristics. When cuVS is not available, the memory-efficient PyTorch backend
is automatically selected for better GPU memory management.

**Backend Selection Priority:**
1. RAPIDS cuVS (if available and GPU present)
2. PyTorch Memory-Efficient (if GPU present but cuVS unavailable, or memory_efficient=True)
3. PyTorch Standard (if GPU present and memory_efficient=False)
4. PyTorch CPU (fallback)

Additionally, the package provides comprehensive metrics for evaluating dimensionality
reduction quality through the **metrics** module, which includes:

- **Distortion metrics**: stress, neighborhood preservation
- **Context metrics**: SVM/kNN classification accuracy preservation
- **Topological metrics**: persistence homology, Betti curves, Wasserstein/bottleneck distances

Examples
--------
Basic usage with automatic backend selection::

    from dire_rapids import create_dire

    # Create reducer with optimal backend (auto-selects memory-efficient if cuVS unavailable)
    reducer = create_dire()

    # Fit and transform data
    embedding = reducer.fit_transform(X)

Force a specific backend::

    from dire_rapids import DiRePyTorch, DiRePyTorchMemoryEfficient, DiReCuVS

    # Use standard PyTorch backend
    reducer = DiRePyTorch(n_neighbors=32)

    # Use memory-efficient PyTorch backend
    reducer = DiRePyTorchMemoryEfficient(n_neighbors=32, use_fp16=True)

    # Use RAPIDS backend (requires RAPIDS installation)
    reducer = DiReCuVS(use_cuvs=True)

Evaluate embedding quality::

    from dire_rapids.metrics import evaluate_embedding

    # Comprehensive evaluation
    results = evaluate_embedding(data, embedding, labels)
    print(f"Stress: {results['local']['stress']:.4f}")
    print(f"SVM accuracy: {results['context']['svm'][1]:.4f}")
"""

__version__ = "0.2.0"

# Import PyTorch backends
from .dire_pytorch import DiRePyTorch, create_dire
from .dire_pytorch_memory_efficient import DiRePyTorchMemoryEfficient

# Import utility classes
from .utils import ReducerRunner, ReducerConfig

# Attempt to import cuVS backend
try:
    from .dire_cuvs import DiReCuVS
    HAS_CUVS = True
except ImportError:
    HAS_CUVS = False

# Import submodules for convenient access
from . import metrics
from . import betti_curve

# Build __all__ based on available modules
__all__ = [
    'DiRePyTorch',
    'DiRePyTorchMemoryEfficient',
    'create_dire',
    'ReducerRunner',
    'ReducerConfig',
    'metrics',
    'betti_curve',
]
if HAS_CUVS:
    __all__.append('DiReCuVS')