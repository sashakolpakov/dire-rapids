"""Environment checks for PyTorch/NumPy interoperability."""

import numpy as np
import pytest

from dire_rapids._compat import check_torch_numpy_bridge, torch_tensor_to_numpy


def test_torch_numpy_bridge_available():
    pytest.importorskip("torch")

    check_torch_numpy_bridge()


def test_torch_tensor_to_numpy():
    torch = pytest.importorskip("torch")

    tensor = torch.arange(4)
    array = torch_tensor_to_numpy(tensor)

    assert isinstance(array, np.ndarray)
    assert array.tolist() == [0, 1, 2, 3]
