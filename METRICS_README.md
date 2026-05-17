# Metrics Module

Evaluation metrics for dimensionality reduction quality.

## Quick Start

```python
from dire_rapids.metrics import evaluate_embedding

results = evaluate_embedding(data, layout, labels, compute_topology=True)
print(f"Stress: {results['local']['stress']:.4f}")
print(f"SVM accuracy: {results['context']['svm'][1]:.4f}")
print(f"DTW β₀: {results['topology']['metrics']['dtw_beta0']:.6f}")
print(f"DTW β₁: {results['topology']['metrics']['dtw_beta1']:.6f}")
```

Topology settings can be tuned through the public API while preserving the
default protocol:

```python
results = evaluate_embedding(
    data,
    layout,
    labels,
    compute_topology=True,
    topology_n_steps=50,
    topology_k_neighbors=20,
    topology_density_threshold=0.8,
    topology_overlap_factor=1.5,
    topology_metrics_only=True,
)
print(results["topology"]["protocol"])
```

## Metrics

1. **Distortion**: stress, neighborhood preservation
2. **Context**: SVM/kNN classification accuracy
3. **Topology**: DTW distances between Betti curves (β₀, β₁), plus protocol metadata

## Installation

```bash
pip install numpy scikit-learn fastdtw

# Optional GPU acceleration
pip install cupy cuml
```

## Examples

```bash
python examples/metrics_swiss_roll.py
```

## Documentation

Full API: [https://sashakolpakov.github.io/dire-rapids/](https://sashakolpakov.github.io/dire-rapids/)
