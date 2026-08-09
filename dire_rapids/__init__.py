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

The factory separates reducer implementation selection from k-NN engine selection:
``backend`` chooses the DiRe implementation, while ``knn_backend`` chooses the
neighbor-search engine (``'auto'``, ``'pytorch'``, ``'pykeops'``, or ``'cuvs'``).
Explicit k-NN backend requests are strict and raise if the engine cannot run.

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

Force a specific k-NN engine::

    from dire_rapids import create_dire

    # CPU implementation with forced PyTorch k-NN
    reducer = create_dire(backend='pytorch_cpu', knn_backend='pytorch')

    # Optional engines raise if unavailable
    reducer = create_dire(knn_backend='pykeops')
    reducer = create_dire(knn_backend='cuvs')

Evaluate embedding quality::

    from dire_rapids.metrics import evaluate_embedding

    # Comprehensive evaluation
    results = evaluate_embedding(data, embedding, labels)
    print(f"Stress: {results['local']['stress']:.4f}")
    print(f"SVM accuracy: {results['context']['svm'][1]:.4f}")
"""

__version__ = "0.3.2"

# Import cuML-using submodule first: cuML (and cuVS) pull shared libraries
# that must be loaded before torch on some rapids-26.04+ setups, because
# torch's pip-installed libnvJitLink can shadow conda's version and break
# later cuML imports with an undefined-symbol error. `metrics` is safe to
# import early — it has no torch dependency at module load time.
from . import metrics

# Import PyTorch backends
from .dire_pytorch import DiRePyTorch, create_dire
from .dire_pytorch_memory_efficient import DiRePyTorchMemoryEfficient

# Import utility classes
from .utils import ReducerRunner, ReducerConfig, build_embedding_figure

# Attempt to import cuVS backend
try:
    from .dire_cuvs import DiReCuVS
    HAS_CUVS = True
except ImportError:
    HAS_CUVS = False

# Remaining submodules for convenient access
from . import betti_curve
from . import presets
from .presets import ATLAS_TUNED, RIPSER_TUNED

# Build __all__ based on available modules
__all__ = [
    'DiRePyTorch',
    'DiRePyTorchMemoryEfficient',
    'create_dire',
    'ReducerRunner',
    'ReducerConfig',
    'build_embedding_figure',
    'metrics',
    'betti_curve',
    'presets',
    'ATLAS_TUNED',
    'RIPSER_TUNED',
]
if HAS_CUVS:
    __all__.append('DiReCuVS')
