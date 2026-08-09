"""Focused tests for the cuVS all-neighbors k-NN integration."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

try:
    import dire_rapids.dire_cuvs as dire_cuvs_module
    from dire_rapids.dire_cuvs import DiReCuVS
except ImportError as exc:  # pragma: no cover - depends on optional install
    pytest.skip(f"DiReCuVS cannot be imported: {exc}", allow_module_level=True)


def _cpu_reducer(**kwargs):
    """Construct a reducer without initializing a CUDA backend."""
    return DiReCuVS(
        use_cuvs=False,
        use_cuml=False,
        verbose=False,
        **kwargs,
    )


def _set_all_neighbors_available(monkeypatch, available):
    """Control availability without requiring RAPIDS in CPU-only CI."""
    module = SimpleNamespace() if available else None
    monkeypatch.setattr(
        dire_cuvs_module, "ALL_NEIGHBORS_AVAILABLE", available, raising=False
    )
    monkeypatch.setattr(
        dire_cuvs_module, "CUVS_ALL_NEIGHBORS_AVAILABLE", available, raising=False
    )
    monkeypatch.setattr(dire_cuvs_module, "all_neighbors", module, raising=False)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "n_clusters, expected_overlap",
    [(1, 0), (2, 1), (4, 2)],
)
def test_all_neighbors_defaults_and_overlap_normalization(
    n_clusters, expected_overlap
):
    reducer = _cpu_reducer(all_neighbors_n_clusters=n_clusters)

    assert reducer.cuvs_knn_method == "auto"
    assert reducer.all_neighbors_algo == "nn_descent"
    assert reducer.all_neighbors_n_clusters == n_clusters
    assert reducer.all_neighbors_overlap_factor == expected_overlap
    assert reducer.all_neighbors_device_ids is None
    assert reducer.all_neighbors_algo_params == {}


@pytest.mark.cpu
def test_all_neighbors_public_options_are_retained():
    algo_params = {"graph_degree": 32, "max_iterations": 20}
    reducer = _cpu_reducer(
        cuvs_knn_method="all_neighbors",
        all_neighbors_algo="brute_force",
        all_neighbors_n_clusters=3,
        all_neighbors_overlap_factor=1,
        all_neighbors_device_ids=[0, 2],
        all_neighbors_algo_params=algo_params,
    )

    assert reducer.cuvs_knn_method == "all_neighbors"
    assert reducer.all_neighbors_algo == "brute_force"
    assert reducer.all_neighbors_n_clusters == 3
    assert reducer.all_neighbors_overlap_factor == 1
    assert reducer.all_neighbors_device_ids == [0, 2]
    assert reducer.all_neighbors_algo_params == algo_params


@pytest.mark.cpu
@pytest.mark.parametrize(
    "n_samples, n_neighbors, n_clusters, overlap, message",
    [
        (4096, 1024, 2, 1, "at most 1024 candidates"),
        (20, 10, 4, 1, "insufficient average cluster capacity"),
    ],
)
def test_partitioned_all_neighbors_rejects_impossible_graphs(
    monkeypatch, n_samples, n_neighbors, n_clusters, overlap, message
):
    _set_all_neighbors_available(monkeypatch, available=True)
    reducer = _cpu_reducer(
        n_neighbors=n_neighbors,
        cuvs_knn_method="all_neighbors",
        all_neighbors_n_clusters=n_clusters,
        all_neighbors_overlap_factor=overlap,
    )
    data = np.zeros((n_samples, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        reducer._compute_knn_all_neighbors(data)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "kwargs",
    [
        {"cuvs_knn_method": "unknown"},
        {"all_neighbors_algo": "cagra"},
        {"all_neighbors_n_clusters": True},
        {"all_neighbors_n_clusters": 0},
        {"all_neighbors_n_clusters": 1.5},
        {"all_neighbors_overlap_factor": True},
        {"all_neighbors_overlap_factor": -1},
        {
            "all_neighbors_n_clusters": 1,
            "all_neighbors_overlap_factor": 1,
        },
        {
            "all_neighbors_n_clusters": 4,
            "all_neighbors_overlap_factor": 0,
        },
        {
            "all_neighbors_n_clusters": 4,
            "all_neighbors_overlap_factor": 4,
        },
        {"all_neighbors_device_ids": []},
        {"all_neighbors_device_ids": [0]},
        {
            "all_neighbors_n_clusters": 2,
            "all_neighbors_device_ids": [0, 0],
        },
        {
            "all_neighbors_n_clusters": 2,
            "all_neighbors_device_ids": [True],
        },
        {
            "all_neighbors_n_clusters": 2,
            "all_neighbors_device_ids": [-1],
        },
        {"all_neighbors_algo_params": []},
    ],
)
def test_all_neighbors_options_reject_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        _cpu_reducer(**kwargs)


@pytest.mark.cpu
@pytest.mark.parametrize("method", ["all_neighbors", "index_search"])
def test_explicit_cuvs_knn_method_wins_over_auto_selection(
    monkeypatch, method
):
    _set_all_neighbors_available(monkeypatch, available=False)
    reducer = _cpu_reducer(cuvs_knn_method=method)

    assert reducer._select_cuvs_knn_method() == method


@pytest.mark.cpu
@pytest.mark.parametrize(
    "available, index_type, expected",
    [
        (True, "auto", "index_search"),
        (False, "auto", "index_search"),
        (True, "ivf_flat", "index_search"),
        (True, "ivf_pq", "index_search"),
        (True, "cagra", "index_search"),
        (True, "flat", "index_search"),
    ],
)
def test_auto_method_preserves_released_index_search_policy(
    monkeypatch, available, index_type, expected
):
    """API availability alone must not silently change the automatic graph."""
    _set_all_neighbors_available(monkeypatch, available)
    reducer = _cpu_reducer(
        cuvs_knn_method="auto",
        cuvs_index_type=index_type,
    )

    assert reducer._select_cuvs_knn_method() == expected


@pytest.mark.cpu
@pytest.mark.parametrize(
    "overrides",
    [
        {"cuvs_build_params": {"n_lists": 7}},
        {"cuvs_search_params": {"n_probes": 2}},
    ],
)
def test_auto_method_preserves_legacy_parameter_overrides(monkeypatch, overrides):
    _set_all_neighbors_available(monkeypatch, available=True)
    reducer = _cpu_reducer(**overrides)

    assert reducer._select_cuvs_knn_method() == "index_search"


@pytest.mark.cpu
@pytest.mark.parametrize(
    "n_samples, n_dims, metric, expected",
    [
        (49_999, 32, "sqeuclidean", "flat"),
        (50_000, 32, "sqeuclidean", "ivf_flat"),
        (499_999, 32, "sqeuclidean", "ivf_flat"),
        (500_000, 32, "sqeuclidean", "ivf_pq"),
        (4_999_999, 32, "sqeuclidean", "ivf_pq"),
        (5_000_000, 500, "sqeuclidean", "cagra"),
        (5_000_000, 501, "sqeuclidean", "ivf_flat"),
        (5_000_000, 32, "cosine", "ivf_pq"),
        (500_000, 501, "sqeuclidean", "ivf_flat"),
    ],
)
def test_legacy_auto_index_boundaries(n_samples, n_dims, metric, expected):
    reducer = _cpu_reducer()

    assert reducer._select_cuvs_index_type(n_samples, n_dims, metric) == expected


@pytest.mark.cpu
def test_diagnostics_export_requested_and_effective_policy():
    import json

    data, _ = make_blobs(
        n_samples=24, n_features=4, centers=3, random_state=41
    )
    reducer = _cpu_reducer(
        n_neighbors=3,
        init="random",
        max_iter_layout=0,
        knn_backend="pytorch",
        random_state=41,
    )

    reducer.fit_transform(data)
    diagnostics = reducer.get_diagnostics()

    assert diagnostics["requested_knn_backend"] == "pytorch"
    assert diagnostics["effective_knn_backend"] == "pytorch"
    assert set(diagnostics["stage_timings_seconds"]) == {
        "graph_construction",
        "initialization",
        "layout",
        "total",
    }
    assert all(
        value >= 0 for value in diagnostics["stage_timings_seconds"].values()
    )
    assert diagnostics["force_chunked_fallback_used"] is False
    assert diagnostics["force_chunked_fallback_calls"] == 0
    assert diagnostics["cuvs"] == {
        "requested_knn_method": "auto",
        "effective_knn_method": None,
        "requested_index_type": "auto",
        "effective_index_type": None,
        "effective_all_neighbors_algo": None,
    }
    json.dumps(diagnostics)


@pytest.mark.cpu
def test_custom_legacy_parameter_dicts_do_not_require_update(monkeypatch):
    """cuVS parameter extension types do not implement ``dict.update``."""

    class ParamsWithoutUpdate:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    calls = {}

    def build(params, data):
        calls["build_params"] = params.kwargs
        calls["build_data"] = data
        return "index"

    def search(params, index, queries, k):
        calls["search_params"] = params.kwargs
        calls["search_k"] = k
        shape = (queries.shape[0], k)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.int64)

    fake_ivf_flat = SimpleNamespace(
        IndexParams=ParamsWithoutUpdate,
        SearchParams=ParamsWithoutUpdate,
        build=build,
        search=search,
    )
    monkeypatch.setattr(dire_cuvs_module, "ivf_flat", fake_ivf_flat, raising=False)

    reducer = _cpu_reducer(
        cuvs_build_params={"n_lists": 7, "add_data_on_build": False},
        cuvs_search_params={"n_probes": 3},
    )
    data = np.zeros((100, 8), dtype=np.float32)

    index = reducer._build_cuvs_index(data, "ivf_flat")
    reducer._search_cuvs(index, "ivf_flat", data, k=5)

    assert index == "index"
    assert calls["build_params"]["n_lists"] == 7
    assert calls["build_params"]["add_data_on_build"] is False
    assert calls["search_params"]["n_probes"] == 3
    # Legacy search asks cuVS for k+1 so that self can be removed later.
    assert calls["search_k"] == 6


@pytest.mark.cpu
@pytest.mark.parametrize("index_type", ["flat", "ivf_flat", "ivf_pq"])
def test_legacy_probe_default_is_at_least_one(monkeypatch, index_type):
    class ParamsWithoutUpdate:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    calls = {}

    def build(params, data):
        calls["build"] = params.kwargs
        return "index"

    def search(params, index, queries, k):
        del index
        calls["search"] = params.kwargs
        shape = (queries.shape[0], k)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.int64)

    fake_module = SimpleNamespace(
        IndexParams=ParamsWithoutUpdate,
        SearchParams=ParamsWithoutUpdate,
        build=build,
        search=search,
    )
    monkeypatch.setattr(dire_cuvs_module, "ivf_flat", fake_module, raising=False)
    monkeypatch.setattr(dire_cuvs_module, "ivf_pq", fake_module, raising=False)
    reducer = _cpu_reducer(cuvs_build_params={"n_lists": 1})
    reducer._n_lists = 1
    data = np.zeros((16, 4), dtype=np.float32)

    reducer._search_cuvs("index", index_type, data, k=3)

    assert calls["search"]["n_probes"] == 1


def _require_gpu_all_neighbors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not getattr(dire_cuvs_module, "CUVS_AVAILABLE", False):
        pytest.skip("cuVS is not available")
    if not getattr(
        dire_cuvs_module,
        "ALL_NEIGHBORS_AVAILABLE",
        getattr(dire_cuvs_module, "CUVS_ALL_NEIGHBORS_AVAILABLE", False),
    ):
        pytest.skip("cuVS all-neighbors is not available")


def _exact_neighbors(data, k):
    indices = NearestNeighbors(n_neighbors=k + 1).fit(data).kneighbors(
        data, return_distance=False
    )
    return np.stack(
        [row[row != row_id][:k] for row_id, row in enumerate(indices)]
    )


def _mean_recall(actual, expected):
    return np.mean(
        [
            len(set(actual_row) & set(expected_row)) / expected.shape[1]
            for actual_row, expected_row in zip(actual, expected)
        ]
    )


def _assert_valid_knn_graph(reducer, n_samples, k):
    assert reducer._knn_indices.shape == (n_samples, k)
    assert reducer._knn_distances.shape == (n_samples, k)
    assert np.issubdtype(reducer._knn_indices.dtype, np.integer)
    assert np.all((0 <= reducer._knn_indices) & (reducer._knn_indices < n_samples))
    assert np.all(np.isfinite(reducer._knn_distances))
    assert np.all(reducer._knn_distances >= 0)
    assert not np.any(
        reducer._knn_indices == np.arange(n_samples, dtype=np.int64)[:, None]
    )


@pytest.mark.gpu
def test_all_neighbors_in_core_recall_and_self_removal():
    _require_gpu_all_neighbors()
    # k+1=16 follows the graph degree exercised by cuVS itself.
    n_samples, k = 1024, 15
    data, _ = make_blobs(
        n_samples=n_samples,
        n_features=24,
        centers=12,
        cluster_std=0.8,
        random_state=42,
    )
    data = np.asarray(data, dtype=np.float32, order="C")
    expected = _exact_neighbors(data, k)
    reducer = DiReCuVS(
        n_neighbors=k,
        knn_backend="cuvs",
        use_cuvs=True,
        use_cuml=False,
        cuvs_knn_method="all_neighbors",
        all_neighbors_algo="nn_descent",
        all_neighbors_n_clusters=1,
        verbose=False,
        random_state=42,
    )

    reducer._compute_knn(data)

    _assert_valid_knn_graph(reducer, n_samples, k)
    assert reducer._last_cuvs_knn_method == "all_neighbors"
    assert reducer.effective_cuvs_knn_method_ == "all_neighbors"
    assert reducer.effective_cuvs_index_type_ is None
    assert reducer.effective_all_neighbors_algo_ == "nn_descent"
    assert _mean_recall(reducer._knn_indices, expected) >= 0.85
    assert reducer._knn_indices_torch.is_cuda
    np.testing.assert_array_equal(
        reducer._knn_indices_torch.cpu().numpy(), reducer._knn_indices
    )


@pytest.mark.gpu
def test_all_neighbors_out_of_core_host_path_recall():
    _require_gpu_all_neighbors()
    n_samples, k = 1536, 10
    data, _ = make_blobs(
        n_samples=n_samples,
        n_features=16,
        centers=12,
        cluster_std=0.7,
        random_state=7,
    )
    data = np.asarray(data, dtype=np.float32, order="C")
    expected = _exact_neighbors(data, k)
    reducer = DiReCuVS(
        n_neighbors=k,
        knn_backend="cuvs",
        use_cuvs=True,
        use_cuml=False,
        cuvs_knn_method="all_neighbors",
        all_neighbors_algo="brute_force",
        all_neighbors_n_clusters=4,
        all_neighbors_overlap_factor=2,
        verbose=False,
        random_state=7,
    )

    # cuVS rejects a device array when n_clusters > 1, so success here also
    # guards the out-of-core host-input path.
    reducer._compute_knn(data)

    _assert_valid_knn_graph(reducer, n_samples, k)
    assert reducer._last_cuvs_knn_method == "all_neighbors"
    assert _mean_recall(reducer._knn_indices, expected) >= 0.80


@pytest.mark.gpu
def test_all_neighbors_rejects_invalid_sentinel_rows():
    _require_gpu_all_neighbors()
    cp = dire_cuvs_module.cp
    sentinel = np.iinfo(np.int64).max
    indices = cp.asarray(
        [[0, 1, sentinel], [1, 0, sentinel]], dtype=cp.int64
    )
    distances = cp.asarray(
        [[0.0, 1.0, np.finfo(np.float32).max]] * 2, dtype=cp.float32
    )

    with pytest.raises(RuntimeError, match="invalid or underfilled"):
        dire_cuvs_module._remove_self_from_knn_cupy(
            indices, distances, n_neighbors=2
        )


@pytest.mark.gpu
def test_all_neighbors_end_to_end_after_cupy_cache_release():
    _require_gpu_all_neighbors()
    data, _ = make_blobs(
        n_samples=1024,
        n_features=16,
        centers=10,
        cluster_std=0.8,
        random_state=19,
    )
    reducer = DiReCuVS(
        n_neighbors=10,
        init="random",
        max_iter_layout=2,
        knn_backend="cuvs",
        use_cuvs=True,
        use_cuml=False,
        cuvs_knn_method="all_neighbors",
        verbose=False,
        random_state=19,
    )

    embedding = reducer.fit_transform(data)

    assert embedding.shape == (1024, 2)
    assert np.all(np.isfinite(embedding))
    assert reducer._knn_indices_torch.is_cuda


@pytest.mark.gpu
@pytest.mark.parametrize(
    "index_type, metric",
    [("flat", None), ("cagra", "euclidean")],
)
def test_legacy_index_search_distances_are_euclidean(index_type, metric):
    _require_gpu_all_neighbors()
    n_samples, k = 256, 5
    data = np.random.default_rng(23).standard_normal((n_samples, 12)).astype(
        np.float32
    )
    reducer = DiReCuVS(
        n_neighbors=k,
        knn_backend="cuvs",
        use_cuvs=True,
        use_cuml=False,
        cuvs_knn_method="index_search",
        cuvs_index_type=index_type,
        metric=metric,
        verbose=False,
        random_state=23,
    )

    reducer._compute_knn(data)

    selected = data[reducer._knn_indices]
    assert reducer.effective_cuvs_knn_method_ == "index_search"
    assert reducer.effective_cuvs_index_type_ == index_type
    expected_distances = np.linalg.norm(data[:, None, :] - selected, axis=2)
    np.testing.assert_allclose(
        reducer._knn_distances, expected_distances, rtol=2e-4, atol=2e-5
    )


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["all_neighbors", "index_search"])
def test_cuvs_inner_product_distances_match_base_contract(method):
    _require_gpu_all_neighbors()
    data = np.random.default_rng(31).standard_normal((1024, 12)).astype(
        np.float32
    )
    reducer = DiReCuVS(
        n_neighbors=5,
        metric="inner_product",
        knn_backend="cuvs",
        use_cuvs=True,
        use_cuml=False,
        cuvs_knn_method=method,
        cuvs_index_type="flat" if method == "index_search" else "auto",
        verbose=False,
        random_state=31,
    )

    reducer._compute_knn(data)

    selected = data[reducer._knn_indices]
    expected_distances = -np.sum(data[:, None, :] * selected, axis=2)
    np.testing.assert_allclose(
        reducer._knn_distances, expected_distances, rtol=2e-4, atol=2e-5
    )
