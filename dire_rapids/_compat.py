"""Compatibility helpers for array/tensor interop."""

from __future__ import annotations

from typing import Any

import numpy as np


def torch_tensor_to_numpy(tensor: Any, *, copy: bool = False) -> np.ndarray:
    """Convert a PyTorch tensor to NumPy with a useful NumPy-bridge error.

    PyTorch wheels compiled against NumPy 1.x cannot expose ``Tensor.numpy()``
    when NumPy 2.x is installed. This commonly happens on GPU cloud images when
    a system PyTorch package is mixed with a fresh RAPIDS/NumPy 2 environment.
    The fix is to install PyTorch from a NumPy-2-compatible wheel in a clean
    virtual environment, not to rebuild DiRe-RAPIDS.
    """
    try:
        array = tensor.detach().cpu().numpy()
    except RuntimeError as exc:
        if "Numpy is not available" in str(exc):
            raise RuntimeError(
                "PyTorch cannot convert tensors to NumPy in this environment. "
                "This usually means PyTorch was compiled against NumPy 1.x but "
                "NumPy 2.x is installed. Use a clean virtual environment and "
                "install a PyTorch build compiled against the active NumPy "
                "major version."
            ) from exc
        raise
    return array.copy() if copy else array


def check_torch_numpy_bridge() -> None:
    """Validate that PyTorch can convert CPU tensors to NumPy arrays."""
    try:
        import torch  # pylint: disable=import-outside-toplevel

        _ = torch.arange(1).cpu().numpy()
    except RuntimeError as exc:
        if "Numpy is not available" in str(exc):
            raise RuntimeError(
                "PyTorch's NumPy bridge is unavailable. Avoid mixing a PyTorch "
                "build compiled against NumPy 1.x with a NumPy 2 environment; "
                "install PyTorch and NumPy together in a clean environment so "
                "their ABI expectations match."
            ) from exc
        raise
