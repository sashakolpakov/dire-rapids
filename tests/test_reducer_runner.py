"""
Basic tests for ReducerRunner framework (CPU only).
"""

import pytest
from dire_rapids import ReducerRunner, ReducerConfig, DiRePyTorch


def test_reducer_config():
    """Test ReducerConfig creation."""
    config = ReducerConfig(
        name="TestReducer",
        reducer_class=DiRePyTorch,
        reducer_kwargs={"n_components": 2, "n_neighbors": 8},
        visualize=False
    )
    assert config.name == "TestReducer"
    assert config.reducer_class == DiRePyTorch
    assert config.visualize is False


def test_reducer_runner_sklearn_dataset():
    """Test ReducerRunner with sklearn dataset."""
    config = ReducerConfig(
        name="DiRe",
        reducer_class=DiRePyTorch,
        reducer_kwargs={"n_components": 2, "n_neighbors": 8, "max_iter_layout": 16, "verbose": False},
        visualize=False
    )
    runner = ReducerRunner(config=config)

    result = runner.run("blobs", dataset_kwargs={"n_samples": 100, "n_features": 10, "centers": 3})

    assert "embedding" in result
    assert "labels" in result
    assert "fit_time_sec" in result
    assert "dataset_info" in result
    assert result["embedding"].shape == (100, 2)
    assert result["dataset_info"]["n_samples"] == 100
    assert result["dataset_info"]["n_features"] == 10


def test_reducer_runner_dire_dataset():
    """Test ReducerRunner with DiRe geometric dataset."""
    config = ReducerConfig(
        name="DiRe",
        reducer_class=DiRePyTorch,
        reducer_kwargs={"n_components": 2, "n_neighbors": 8, "max_iter_layout": 16, "verbose": False},
        visualize=False
    )
    runner = ReducerRunner(config=config)

    result = runner.run("dire:sphere_uniform", dataset_kwargs={"n_features": 5, "n_samples": 50})

    assert result["embedding"].shape == (50, 2)
    assert result["dataset_info"]["n_samples"] == 50
    assert result["dataset_info"]["n_features"] == 5


def test_multiple_sklearn_datasets():
    """Test ReducerRunner with multiple sklearn datasets."""
    config = ReducerConfig(
        name="DiRe",
        reducer_class=DiRePyTorch,
        reducer_kwargs={"n_components": 2, "n_neighbors": 8, "max_iter_layout": 16, "verbose": False},
        visualize=False
    )
    runner = ReducerRunner(config=config)

    # Test different datasets
    datasets = ["sklearn:iris", "sklearn:wine", "moons"]
    for dataset in datasets:
        result = runner.run(dataset)
        assert result["embedding"].shape[1] == 2
        assert "fit_time_sec" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
