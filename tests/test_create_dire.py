#!/usr/bin/env python3
"""Test script for create_dire function."""

import numpy as np
import torch
from dire_rapids import create_dire

print("Testing create_dire function...\n")

# Generate small test data
X = np.random.randn(100, 10).astype(np.float32)

# Test 1: Auto backend
print("1. Testing auto backend:")
try:
    reducer = create_dire(backend='auto', verbose=False)
    print(f"   Created: {type(reducer).__name__}")
    result = reducer.fit_transform(X)
    print(f"   Transform successful, shape: {result.shape}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: PyTorch backend
print("\n2. Testing pytorch backend:")
try:
    reducer = create_dire(backend='pytorch', verbose=False)
    print(f"   Created: {type(reducer).__name__}")
    print(f"   Device: {reducer.device}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: PyTorch CPU backend
print("\n3. Testing pytorch_cpu backend:")
try:
    reducer = create_dire(backend='pytorch_cpu', verbose=False)
    print(f"   Created: {type(reducer).__name__}")
    print(f"   Device: {reducer.device}")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Memory-efficient mode
print("\n4. Testing memory-efficient mode:")
try:
    reducer = create_dire(memory_efficient=True, verbose=False)
    print(f"   Created: {type(reducer).__name__}")
    result = reducer.fit_transform(X)
    print(f"   Transform successful, shape: {result.shape}")
except Exception as e:
    print(f"   Error: {e}")

# Test 5: PyTorch GPU backend (if available)
print("\n5. Testing pytorch_gpu backend:")
try:
    reducer = create_dire(backend='pytorch_gpu', verbose=False)
    print(f"   Created: {type(reducer).__name__}")
    print(f"   Device: {reducer.device}")
except RuntimeError as e:
    if "CUDA not available" in str(e):
        print(f"   Skipped: CUDA not available")
    else:
        print(f"   Error: {e}")
except Exception as e:
    print(f"   Error: {e}")

# Test 6: cuVS backend (if available)
print("\n6. Testing cuvs backend:")
try:
    reducer = create_dire(backend='cuvs', verbose=False)
    print(f"   Created: {type(reducer).__name__}")
except RuntimeError as e:
    if "RAPIDS not installed" in str(e) or "CUDA" in str(e):
        print(f"   Skipped: {e}")
    else:
        print(f"   Error: {e}")
except Exception as e:
    print(f"   Error: {e}")

# Test 7: Invalid backend
print("\n7. Testing invalid backend:")
try:
    reducer = create_dire(backend='invalid')
    print(f"   Should have raised error")
except ValueError as e:
    print(f"   Correctly raised ValueError: {e}")
except Exception as e:
    print(f"   Unexpected error: {e}")

print("\nAll tests completed!")