# dire-rapids

"""
PyTorch and RAPIDS (cuVS/cuML) accelerated dimensionality reduction.
"""

# Import PyTorch backends
from .dire_pytorch import DiRePyTorch, create_dire
from .dire_pytorch_memory_efficient import DiRePyTorchMemoryEfficient

# Attempt to import cuVS backend
try:
    from .dire_cuvs import DiReCuVS
    HAS_CUVS = True
except ImportError:
    HAS_CUVS = False
    import warnings
    warnings.warn(
        "cuVS backend not available. "
        "Install RAPIDS for GPU acceleration: pip install rapids-25.08",
        UserWarning
    )

# Build __all__ based on available modules
__all__ = ['DiRePyTorch', 'DiRePyTorchMemoryEfficient', 'create_dire']
if HAS_CUVS:
    __all__.append('DiReCuVS')