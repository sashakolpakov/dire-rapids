"""Tests for topology parameter passthrough in evaluate_embedding."""

import numpy as np

from dire_rapids import metrics


def test_evaluate_embedding_uses_topology_defaults(monkeypatch):
    """Topology defaults stay compatible with the previous hardcoded values."""
    captured = {}

    def fake_compute_global_metrics(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "metrics": {"dtw_beta0": 0.0, "dtw_beta1": 0.0},
            "backend": "atlas",
            "protocol": kwargs,
        }

    monkeypatch.setattr(metrics, "compute_global_metrics", fake_compute_global_metrics)

    data = np.zeros((8, 3), dtype=np.float32)
    layout = np.zeros((8, 2), dtype=np.float32)

    result = metrics.evaluate_embedding(
        data,
        layout,
        compute_distortion=False,
        compute_context=False,
        compute_topology=True,
        use_gpu=False,
    )

    assert "topology" in result
    assert captured["kwargs"] == {
        "n_steps": 100,
        "k_neighbors": 20,
        "density_threshold": 0.8,
        "overlap_factor": 1.5,
        "use_gpu": False,
        "metrics_only": True,
    }


def test_evaluate_embedding_passes_custom_topology_parameters(monkeypatch):
    """Public topology kwargs are forwarded to compute_global_metrics."""
    captured = {}

    def fake_compute_global_metrics(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "metrics": {"dtw_beta0": 1.0, "dtw_beta1": 2.0},
            "backend": "atlas",
            "protocol": kwargs,
        }

    monkeypatch.setattr(metrics, "compute_global_metrics", fake_compute_global_metrics)

    data = np.zeros((8, 3), dtype=np.float32)
    layout = np.zeros((8, 2), dtype=np.float32)

    result = metrics.evaluate_embedding(
        data,
        layout,
        compute_distortion=False,
        compute_context=False,
        compute_topology=True,
        subsample_threshold=0.25,
        random_state=7,
        use_gpu=False,
        topology_n_steps=25,
        topology_k_neighbors=8,
        topology_density_threshold=0.65,
        topology_overlap_factor=2.0,
        topology_metrics_only=False,
    )

    assert result["topology"]["metrics"] == {"dtw_beta0": 1.0, "dtw_beta1": 2.0}
    assert captured["args"][2] == 0.25
    assert captured["args"][3] == 7
    assert captured["kwargs"] == {
        "n_steps": 25,
        "k_neighbors": 8,
        "density_threshold": 0.65,
        "overlap_factor": 2.0,
        "use_gpu": False,
        "metrics_only": False,
    }


def test_compute_global_metrics_returns_protocol(monkeypatch):
    """Topology results include enough metadata to audit the protocol."""
    from dire_rapids import betti_curve

    def fake_compute_betti_curve(*args, **kwargs):
        return {
            "filtration_values": np.array([0.0, 1.0], dtype=np.float32),
            "beta_0": np.array([2, 1], dtype=np.float32),
            "beta_1": np.array([0, 1], dtype=np.float32),
        }

    monkeypatch.setattr(betti_curve, "compute_betti_curve", fake_compute_betti_curve)

    data = np.zeros((8, 3), dtype=np.float32)
    layout = np.zeros((8, 2), dtype=np.float32)

    result = metrics.compute_global_metrics(
        data,
        layout,
        subsample_threshold=1.0,
        random_state=11,
        n_steps=12,
        k_neighbors=6,
        density_threshold=0.7,
        overlap_factor=1.25,
        use_gpu=False,
        metrics_only=True,
    )

    assert result["backend"] == "atlas"
    assert result["protocol"] == {
        "subsample_threshold": 1.0,
        "random_state": 11,
        "n_steps": 12,
        "k_neighbors": 6,
        "density_threshold": 0.7,
        "overlap_factor": 1.25,
        "use_gpu": False,
        "metrics_only": True,
    }
