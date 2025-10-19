"""
Test the local kNN atlas approach for topology computation.
"""

import sys
import numpy as np

sys.path.insert(0, '/Users/sasha/dire-rapids')

from dire_rapids.metrics import compute_h0_h1_knn

def test_circle(n_samples, noise, k_local, density_threshold, label):
    """Test on circle with given parameters."""
    rng = np.random.RandomState(42)
    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    x = np.cos(theta) + rng.randn(n_samples) * noise
    y = np.sin(theta) + rng.randn(n_samples) * noise
    data = np.column_stack([x, y]).astype(np.float32)

    beta_0, beta_1 = compute_h0_h1_knn(data, k_neighbors=k_local,
                                       density_threshold=density_threshold, use_gpu=False)

    status = "✓" if beta_0 == 1 and beta_1 == 1 else "✗"
    print(f"{label:40s}: β₀={beta_0}, β₁={beta_1} {status}")
    return beta_0 == 1 and beta_1 == 1


print("="*70)
print("Testing Local kNN Atlas Approach")
print("="*70)

results = []

print("\n[Clean Circle Tests]")
results.append(test_circle(100, 0.0, 10, 0.7, "Clean circle (n=100, k=10, ρ=0.7)"))
results.append(test_circle(100, 0.0, 15, 0.7, "Clean circle (n=100, k=15, ρ=0.7)"))
results.append(test_circle(100, 0.0, 20, 0.7, "Clean circle (n=100, k=20, ρ=0.7)"))

print("\n[Noisy Circle Tests]")
results.append(test_circle(200, 0.05, 10, 0.7, "Noisy circle (n=200, noise=0.05, k=10)"))
results.append(test_circle(200, 0.05, 15, 0.7, "Noisy circle (n=200, noise=0.05, k=15)"))
results.append(test_circle(200, 0.05, 20, 0.7, "Noisy circle (n=200, noise=0.05, k=20)"))

print("\n[Different Density Thresholds]")
results.append(test_circle(200, 0.05, 15, 0.5, "Noisy circle (k=15, ρ=0.5)"))
results.append(test_circle(200, 0.05, 15, 0.6, "Noisy circle (k=15, ρ=0.6)"))
results.append(test_circle(200, 0.05, 15, 0.8, "Noisy circle (k=15, ρ=0.8)"))

print("\n" + "="*70)
print(f"Results: {sum(results)}/{len(results)} tests passed")
print("="*70)
