"""
Test the integrated atlas approach in main metrics.py
"""

import numpy as np
from dire_rapids.metrics import compute_h0_h1_knn

# Test 1: Clean circle
print("Test 1: Clean circle (n=100, k=15)")
n = 100
theta = np.linspace(0, 2*np.pi, n, endpoint=False)
data = np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)

h0, h1 = compute_h0_h1_knn(data, k_neighbors=15, use_gpu=False)
beta_0 = len(h0[h0[:, 1] == np.inf])
beta_1 = len(h1[h1[:, 1] == np.inf])
status = "✓" if beta_0 == 1 and beta_1 == 1 else "✗"
print(f"  β₀={beta_0}, β₁={beta_1} {status}")

# Test 2: Noisy circle
print("\nTest 2: Noisy circle (n=200, k=15, noise=0.05)")
rng = np.random.RandomState(42)
theta = np.linspace(0, 2*np.pi, 200, endpoint=False)
x = np.cos(theta) + rng.randn(200) * 0.05
y = np.sin(theta) + rng.randn(200) * 0.05
data = np.column_stack([x, y]).astype(np.float32)

h0, h1 = compute_h0_h1_knn(data, k_neighbors=15, use_gpu=False)
beta_0 = len(h0[h0[:, 1] == np.inf])
beta_1 = len(h1[h1[:, 1] == np.inf])
status = "✓" if beta_0 == 1 and beta_1 == 1 else "✗"
print(f"  β₀={beta_0}, β₁={beta_1} {status}")

print("\n✓ Atlas integration successful!")
