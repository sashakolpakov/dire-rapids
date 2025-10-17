"""
Compare original fast backend vs atlas backend vs ripser.
"""

import numpy as np
import time
from dire_rapids.metrics import compute_h0_h1_knn
from dire_rapids.metrics_atlas import compute_h0_h1_atlas

def generate_circle(n_samples, noise=0.05, seed=42):
    rng = np.random.RandomState(seed)
    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    x = np.cos(theta) + rng.randn(n_samples) * noise
    y = np.sin(theta) + rng.randn(n_samples) * noise
    return np.column_stack([x, y]).astype(np.float32)

def generate_torus(n_samples, R=2.0, r=1.0, noise=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n_sqrt = int(np.sqrt(n_samples))
    u = np.linspace(0, 2*np.pi, n_sqrt, endpoint=False)
    v = np.linspace(0, 2*np.pi, n_sqrt, endpoint=False)
    u, v = np.meshgrid(u, v)
    u = u.flatten()[:n_samples]
    v = v.flatten()[:n_samples]

    x = (R + r * np.cos(v)) * np.cos(u) + rng.randn(len(u)) * noise
    y = (R + r * np.cos(v)) * np.sin(u) + rng.randn(len(u)) * noise
    z = r * np.sin(v) + rng.randn(len(u)) * noise

    return np.column_stack([x, y, z]).astype(np.float32)

def test_backend(name, data, expected_beta0, expected_beta1, compute_func, **kwargs):
    """Test a backend and return results."""
    t_start = time.time()
    h0, h1 = compute_func(data, **kwargs)
    elapsed = time.time() - t_start

    beta_0 = len(h0[h0[:, 1] == np.inf])
    beta_1 = len(h1[h1[:, 1] == np.inf])

    correct = (beta_0 == expected_beta0 and beta_1 == expected_beta1)
    status = "✓" if correct else "✗"

    print(f"  {name:20s}: β₀={beta_0} (exp {expected_beta0}), β₁={beta_1} (exp {expected_beta1}) | {elapsed:.3f}s {status}")
    return correct

print("="*80)
print("Backend Comparison: Original Fast vs Atlas")
print("="*80)

results = {}

# Test 1: Clean circle
print("\n[Test 1] Clean Circle (n=100)")
data = generate_circle(100, noise=0.0)
results['clean_circle'] = []
results['clean_circle'].append(test_backend("Fast (k=2)", data, 1, 1, compute_h0_h1_knn, k_neighbors=2, use_gpu=False))
results['clean_circle'].append(test_backend("Fast (k=10)", data, 1, 1, compute_h0_h1_knn, k_neighbors=10, use_gpu=False))
results['clean_circle'].append(test_backend("Atlas (k=10, ρ=0.7)", data, 1, 1, compute_h0_h1_atlas, k_local=10, density_threshold=0.7, use_gpu=False))
results['clean_circle'].append(test_backend("Atlas (k=15, ρ=0.7)", data, 1, 1, compute_h0_h1_atlas, k_local=15, density_threshold=0.7, use_gpu=False))

# Test 2: Noisy circle
print("\n[Test 2] Noisy Circle (n=200, noise=0.05)")
data = generate_circle(200, noise=0.05)
results['noisy_circle'] = []
results['noisy_circle'].append(test_backend("Fast (k=2)", data, 1, 1, compute_h0_h1_knn, k_neighbors=2, use_gpu=False))
results['noisy_circle'].append(test_backend("Fast (k=10)", data, 1, 1, compute_h0_h1_knn, k_neighbors=10, use_gpu=False))
results['noisy_circle'].append(test_backend("Atlas (k=10, ρ=0.7)", data, 1, 1, compute_h0_h1_atlas, k_local=10, density_threshold=0.7, use_gpu=False))
results['noisy_circle'].append(test_backend("Atlas (k=15, ρ=0.7)", data, 1, 1, compute_h0_h1_atlas, k_local=15, density_threshold=0.7, use_gpu=False))
results['noisy_circle'].append(test_backend("Atlas (k=20, ρ=0.7)", data, 1, 1, compute_h0_h1_atlas, k_local=20, density_threshold=0.7, use_gpu=False))

# Test 3: Torus
print("\n[Test 3] Torus (n=400, noise=0.05)")
data = generate_torus(400, noise=0.05)
results['torus'] = []
results['torus'].append(test_backend("Fast (k=5)", data, 1, 2, compute_h0_h1_knn, k_neighbors=5, use_gpu=False))
results['torus'].append(test_backend("Atlas (k=15, ρ=0.7)", data, 1, 2, compute_h0_h1_atlas, k_local=15, density_threshold=0.7, use_gpu=False))
results['torus'].append(test_backend("Atlas (k=20, ρ=0.7)", data, 1, 2, compute_h0_h1_atlas, k_local=20, density_threshold=0.7, use_gpu=False))

# Summary
print("\n" + "="*80)
print("Summary")
print("="*80)
for test_name, test_results in results.items():
    passed = sum(test_results)
    total = len(test_results)
    print(f"{test_name:20s}: {passed}/{total} tests passed")
