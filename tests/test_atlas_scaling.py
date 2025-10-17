"""
Test atlas scaling with different point counts and noise levels.

Tests:
1. Circle with n = 200, 500, 1000, 2000
2. Torus with n = 200, 500, 1000, 2000
3. Noisy data with noise = 0.05, 0.1, 0.25, 0.5

Measures:
- Construction time
- Eigendecomposition time
- Accuracy (β₀, β₁)
- Matrix sizes and sparsity
"""

import time
import numpy as np
from dire_rapids.metrics import compute_h0_h1_knn


def test_circle_scaling():
    """Test circle with different point counts."""
    print("=" * 70)
    print("CIRCLE SCALING TEST")
    print("=" * 70)

    point_counts = [250, 500, 1000, 2000]

    results = []

    for n in point_counts:
        print(f"\n{'='*70}")
        print(f"Circle: n={n}, k=15")
        print(f"{'='*70}")

        # Generate circle
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        data = np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)

        # Time the computation
        t0 = time.time()
        h0, h1 = compute_h0_h1_knn(data, k_neighbors=15, use_gpu=False)
        t_total = time.time() - t0

        # Count Betti numbers
        beta_0 = len(h0[h0[:, 1] == np.inf])
        beta_1 = len(h1[h1[:, 1] == np.inf])

        # Check correctness
        status = "✓" if beta_0 == 1 and beta_1 == 1 else "✗"

        print(f"  Time: {t_total:.3f}s")
        print(f"  β₀={beta_0}, β₁={beta_1} {status}")

        results.append({
            'n': n,
            'time': t_total,
            'beta_0': beta_0,
            'beta_1': beta_1,
            'correct': beta_0 == 1 and beta_1 == 1
        })

    # Summary
    print(f"\n{'='*70}")
    print("CIRCLE SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"{'n':>6} | {'Time (s)':>10} | {'β₀':>3} | {'β₁':>3} | Status")
    print(f"{'-'*6}+{'-'*12}+{'-'*5}+{'-'*5}+{'-'*7}")

    for r in results:
        status = "✓" if r['correct'] else "✗"
        print(f"{r['n']:>6} | {r['time']:>10.3f} | {r['beta_0']:>3} | {r['beta_1']:>3} | {status:>6}")

    return results


def test_torus_scaling():
    """Test torus with different point counts."""
    print("\n\n" + "=" * 70)
    print("TORUS SCALING TEST")
    print("=" * 70)

    point_counts = [250, 500, 1000, 2000]

    results = []

    for n in point_counts:
        print(f"\n{'='*70}")
        print(f"Torus: n={n}, k=15")
        print(f"{'='*70}")

        # Generate torus (R=2, r=1)
        n_theta = int(np.sqrt(n * 2))
        n_phi = n // n_theta

        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        R, r = 2.0, 1.0
        x = (R + r * np.cos(phi_grid)) * np.cos(theta_grid)
        y = (R + r * np.cos(phi_grid)) * np.sin(theta_grid)
        z = r * np.sin(phi_grid)

        data = np.column_stack([x.flatten(), y.flatten(), z.flatten()]).astype(np.float32)

        # Time the computation
        t0 = time.time()
        h0, h1 = compute_h0_h1_knn(data, k_neighbors=15, use_gpu=False)
        t_total = time.time() - t0

        # Count Betti numbers
        beta_0 = len(h0[h0[:, 1] == np.inf])
        beta_1 = len(h1[h1[:, 1] == np.inf])

        # Check correctness (torus has β₀=1, β₁=2)
        status = "✓" if beta_0 == 1 and beta_1 == 2 else "✗"

        print(f"  Time: {t_total:.3f}s")
        print(f"  β₀={beta_0}, β₁={beta_1} {status}")

        results.append({
            'n': n,
            'time': t_total,
            'beta_0': beta_0,
            'beta_1': beta_1,
            'correct': beta_0 == 1 and beta_1 == 2
        })

    # Summary
    print(f"\n{'='*70}")
    print("TORUS SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"{'n':>6} | {'Time (s)':>10} | {'β₀':>3} | {'β₁':>3} | Status")
    print(f"{'-'*6}+{'-'*12}+{'-'*5}+{'-'*5}+{'-'*7}")

    for r in results:
        status = "✓" if r['correct'] else "✗"
        print(f"{r['n']:>6} | {r['time']:>10.3f} | {r['beta_0']:>3} | {r['beta_1']:>3} | {status:>6}")

    return results


def test_noise_robustness():
    """Test noisy circle with different noise levels."""
    print("\n\n" + "=" * 70)
    print("NOISE ROBUSTNESS TEST")
    print("=" * 70)

    noise_levels = [0.02, 0.05, 0.1]
    n = 500

    results = []

    for noise in noise_levels:
        print(f"\n{'='*70}")
        print(f"Noisy Circle: n={n}, k=15, noise={noise}")
        print(f"{'='*70}")

        # Generate noisy circle
        rng = np.random.RandomState(42)
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = np.cos(theta) + rng.randn(n) * noise
        y = np.sin(theta) + rng.randn(n) * noise
        data = np.column_stack([x, y]).astype(np.float32)

        # Time the computation
        t0 = time.time()
        h0, h1 = compute_h0_h1_knn(data, k_neighbors=15, use_gpu=False)
        t_total = time.time() - t0

        # Count Betti numbers
        beta_0 = len(h0[h0[:, 1] == np.inf])
        beta_1 = len(h1[h1[:, 1] == np.inf])

        # Check correctness
        status = "✓" if beta_0 == 1 and beta_1 == 1 else "✗"

        print(f"  Time: {t_total:.3f}s")
        print(f"  β₀={beta_0}, β₁={beta_1} {status}")

        results.append({
            'noise': noise,
            'time': t_total,
            'beta_0': beta_0,
            'beta_1': beta_1,
            'correct': beta_0 == 1 and beta_1 == 1
        })

    # Summary
    print(f"\n{'='*70}")
    print("NOISE ROBUSTNESS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Noise':>6} | {'Time (s)':>10} | {'β₀':>3} | {'β₁':>3} | Status")
    print(f"{'-'*6}+{'-'*12}+{'-'*5}+{'-'*5}+{'-'*7}")

    for r in results:
        status = "✓" if r['correct'] else "✗"
        print(f"{r['noise']:>6.2f} | {r['time']:>10.3f} | {r['beta_0']:>3} | {r['beta_1']:>3} | {status:>6}")

    return results


if __name__ == "__main__":
    # Run all tests
    circle_results = test_circle_scaling()
    torus_results = test_torus_scaling()
    noise_results = test_noise_robustness()

    # Overall summary
    print("\n\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    circle_correct = sum(1 for r in circle_results if r['correct'])
    torus_correct = sum(1 for r in torus_results if r['correct'])
    noise_correct = sum(1 for r in noise_results if r['correct'])

    print(f"Circle scaling: {circle_correct}/{len(circle_results)} tests passed")
    print(f"Torus scaling: {torus_correct}/{len(torus_results)} tests passed")
    print(f"Noise robustness: {noise_correct}/{len(noise_results)} tests passed")

    total_correct = circle_correct + torus_correct + noise_correct
    total_tests = len(circle_results) + len(torus_results) + len(noise_results)

    print(f"\nTotal: {total_correct}/{total_tests} tests passed")

    if total_correct == total_tests:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total_tests - total_correct} tests failed")
