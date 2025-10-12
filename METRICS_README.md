# Metrics Module

Comprehensive evaluation metrics for dimensionality reduction quality with GPU acceleration.

## Quick Start

```python
from dire_rapids.metrics import evaluate_embedding

results = evaluate_embedding(data, layout, labels)
print(f"Stress: {results['local']['stress']:.4f}")
print(f"SVM accuracy: {results['context']['svm'][1]:.4f}")
print(f"Wasserstein: {results['topology']['metrics']['wass'][0]:.6f}")
```

## Metrics Categories

1. **Distortion**: stress, neighborhood preservation
2. **Context**: SVM/kNN classification accuracy
3. **Topology**: persistence diagrams, Betti curves, distances

## Installation

```bash
# Core
pip install numpy cupy cuml scikit-learn

# Persistence backends (choose one or more)
pip install giotto-ph         # Recommended
pip install ripserplusplus    # GPU
pip install ripser            # CPU fallback

# Distance metrics
pip install persim POT fastdtw twed
```

## Persistence Backends

Multiple backends with automatic selection:

```python
from dire_rapids.metrics import set_persistence_backend

set_persistence_backend('giotto-ph')  # or 'ripser++', 'ripser', None (auto)
```

| Backend | Speed | GPU | Multi-threaded |
|---------|-------|-----|----------------|
| giotto-ph | Fastest | No | Yes |
| ripser++ | Fast | Yes | No |
| ripser | Moderate | No | No |

## Examples

```bash
python examples/metrics_simple_test.py
python examples/metrics_evaluation.py
```

## Documentation

See full API documentation at: [https://sashakolpakov.github.io/dire-rapids/](https://sashakolpakov.github.io/dire-rapids/)
