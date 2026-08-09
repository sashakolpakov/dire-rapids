"""CPU-safe contract tests for the issue #14 historical audit harness."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import types

import numpy as np
import pytest


HARNESS = (
    Path(__file__).resolve().parents[1]
    / "benchmarking"
    / "bench_topology_historical.py"
)
SPECIFICATION = importlib.util.spec_from_file_location(
    "bench_topology_historical", HARNESS
)
assert SPECIFICATION is not None and SPECIFICATION.loader is not None
historical = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(historical)


@pytest.mark.cpu
def test_retained_h100_archive_contains_complete_historical_evidence():
    archive = (
        Path(__file__).resolve().parent
        / "data"
        / "topology_historical_h100_audit.tar.gz"
    )
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        "50e548e9b93beebb54e06e023889595a6304b2c3e5ba9746cde574bdc5dfa812"
    )

    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
        reference_names = {
            name
            for name in names
            if name.startswith("reference-cache/") and name.endswith(".json")
        }
        assert len(reference_names) == 6 * 20
        for variant in historical.VARIANTS:
            raw_name = f"raw/{variant}.jsonl"
            manifest_name = f"raw/{variant}.manifest.json"
            assert raw_name in names
            assert manifest_name in names
            raw_stream = bundle.extractfile(raw_name)
            assert raw_stream is not None
            assert len(raw_stream.read().splitlines()) == 6 * 20 * 2

        summary_stream = bundle.extractfile(
            "summary/topology_preset_historical_summary.json"
        )
        assert summary_stream is not None
        summary = json.loads(summary_stream.read())
        assert summary["material_post_9117_implementation_change"] is False
        assert not summary["material_post_9117_implementation_change_cells"]


def make_complete_records(repeats=2):
    records = []
    dataset_hashes = {
        dataset: f"dataset-{dataset}" for dataset in historical.DATASETS
    }
    for variant, specification in historical.VARIANTS.items():
        for dataset_index, dataset in enumerate(historical.DATASETS):
            for seed in historical.LAYOUT_SEEDS[:repeats]:
                for configuration in historical.CONFIGURATIONS:
                    base = 1.0 + 0.1 * dataset_index + 0.001 * seed
                    preset_gap = 0.1
                    if variant == "471ac16_auto_all_neighbors":
                        preset_gap = 0.2
                    value = base + (
                        preset_gap
                        if configuration == "former_topology_preset"
                        else 0.0
                    )
                    records.append(
                        {
                            "schema_version": historical.SCHEMA_VERSION,
                            "variant": variant,
                            "revision": specification["revision"],
                            "requested_policy": specification["policy"],
                            "environment_sha256": "same-frozen-environment",
                            "effective_policy": {
                                "method": (
                                    "all_neighbors"
                                    if specification["policy"]
                                    == "pr12_auto_all_neighbors"
                                    else "index_search"
                                ),
                                "index_type": (
                                    None
                                    if specification["policy"]
                                    == "pr12_auto_all_neighbors"
                                    else "flat"
                                ),
                            },
                            "dataset": dataset,
                            "dataset_array_sha256": dataset_hashes[dataset],
                            "configuration": configuration,
                            "parameters": historical.reducer_kwargs(
                                specification["revision"],
                                specification["policy"],
                                configuration,
                                seed,
                            ),
                            "layout_seed": seed,
                            "subset_seed": seed + 1_000,
                            "subset_indices_sha256": f"subset-{dataset}-{seed}",
                            "subset_size": 1_000,
                            "reference_curve_sha256": {
                                "atlas": f"{'a' * 56}{dataset_index:08x}",
                                "ripser": f"{'b' * 56}{dataset_index:08x}",
                            },
                            "metrics": {
                                metric: value for metric in historical.METRICS
                            },
                        }
                    )
    return records


@pytest.mark.cpu
def test_historical_summary_requires_and_retains_full_paired_matrix(tmp_path):
    records = make_complete_records()
    summary = historical.summarize_records(records, tmp_path, repeats=2)

    assert summary["material_post_9117_implementation_change"] is False
    assert (
        summary["atlas_preset_effect_by_variant"]
        ["9117dc4_index_search_flat"]["cells"]
        == 12
    )
    assert (
        summary["atlas_preset_effect_by_variant"]
        ["471ac16_auto_all_neighbors"]["preset_numerically_worse"]
        == 12
    )
    assert (tmp_path / "topology_preset_historical_paired_effects.csv").is_file()
    assert (tmp_path / "topology_preset_historical_differences.csv").is_file()
    stored = json.loads(
        (tmp_path / "topology_preset_historical_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored == summary


@pytest.mark.cpu
def test_historical_summary_rejects_missing_record(tmp_path):
    records = make_complete_records()
    records.pop()

    with pytest.raises(RuntimeError, match="matrix is incomplete"):
        historical.summarize_records(records, tmp_path, repeats=2)


@pytest.mark.cpu
def test_historical_summary_rejects_cross_revision_subset_mismatch(tmp_path):
    records = make_complete_records()
    for record in records:
        if (
            record["variant"] == "293b622_index_search_flat"
            and record["dataset"] == "blobs"
            and record["layout_seed"] == 42
        ):
            record["subset_indices_sha256"] = "different-subset"

    with pytest.raises(RuntimeError, match="cross-revision .* mismatch"):
        historical.summarize_records(records, tmp_path, repeats=2)


@pytest.mark.cpu
def test_historical_reducer_kwargs_keep_removed_preset_private_and_pinned():
    old = historical.reducer_kwargs(
        historical.REVISIONS["preset_introduction"],
        "index_search_flat",
        "former_topology_preset",
        42,
    )
    pr_auto = historical.reducer_kwargs(
        historical.REVISIONS["pr12_head"],
        "pr12_auto_all_neighbors",
        "default",
        42,
    )

    assert old["backend"] == "cuvs"
    assert old["cuvs_index_type"] == "flat"
    assert "knn_backend" not in old
    assert old["spread"] == 3.6
    assert old["max_iter_layout"] == 150
    default = historical.reducer_kwargs(
        historical.REVISIONS["preset_introduction"],
        "index_search_flat",
        "default",
        42,
    )
    assert {
        name: default[name]
        for name in ("spread", "min_dist", "cutoff", "neg_ratio")
    } == {
        "spread": 1.0,
        "min_dist": 1e-2,
        "cutoff": 42.0,
        "neg_ratio": 8,
    }
    assert pr_auto["knn_backend"] == "cuvs"
    assert pr_auto["cuvs_knn_method"] == "auto"
    assert pr_auto["cuvs_index_type"] == "auto"


@pytest.mark.cpu
def test_dataset_manifest_rejects_self_consistent_noncanonical_input(tmp_path):
    data = np.asarray([[1.0, 2.0]], dtype=np.float32)
    path = tmp_path / "blobs.npy"
    historical.save_array(path, data)
    actual_array_sha256 = historical.array_sha256(data)
    historical.write_json(
        tmp_path / "manifest.json",
        {
            "datasets": {
                "blobs": {
                    "path": path.name,
                    "file_sha256": historical.sha256_file(path),
                    "array_sha256": actual_array_sha256,
                }
            }
        },
    )
    original_datasets = historical.DATASETS
    historical.DATASETS = ("blobs",)
    try:
        with pytest.raises(RuntimeError, match="frozen issue #14 input contract"):
            historical.load_dataset_manifest(tmp_path)
    finally:
        historical.DATASETS = original_datasets


@pytest.mark.cpu
def test_run_variant_writes_resumable_records_and_shared_reference_cache(
    tmp_path, monkeypatch
):
    data = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    dataset_hash = historical.array_sha256(data)
    dataset_manifest = {
        "datasets": {"tiny": {"array_sha256": dataset_hash}}
    }

    class FakeReducer:
        cuvs_index_type = "flat"

        def fit_transform(self, values):
            return np.asarray(values[:, :2], dtype=np.float32)

    class FakeDire:
        @staticmethod
        def create_dire(**_kwargs):
            return FakeReducer()

    class FakeEvaluator:
        calls = 0

        @classmethod
        def curve(cls, _values, **_kwargs):
            cls.calls += 1
            return {
                "filtration_values": np.asarray([1.0, 0.0]),
                "beta_0": np.asarray([1, 4]),
                "beta_1": np.asarray([0, 0]),
                "n_edges_active": np.asarray([4, 0]),
                "n_triangles_active": np.asarray([0, 0]),
            }

        compute_betti_curve_gpu = curve
        compute_betti_curve_ripser = curve

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            runtimeGetVersion=lambda: 13000,
            driverGetVersion=lambda: 13000,
        )
    )
    fake_cupy.get_default_memory_pool = lambda: types.SimpleNamespace(
        free_all_blocks=lambda: None
    )
    fake_cuvs = types.ModuleType("cuvs")
    fake_neighbors = types.ModuleType("cuvs.neighbors")
    fake_neighbors.brute_force = object()
    fake_cuvs.neighbors = fake_neighbors

    import torch

    monkeypatch.setattr(historical, "DATASETS", ("tiny",))
    monkeypatch.setattr(historical, "LAYOUT_SEEDS", (42,))
    monkeypatch.setattr(historical, "TOPOLOGY_SUBSET_SIZE", 3)
    monkeypatch.setattr(
        historical,
        "import_target_dire",
        lambda _root, _revision: FakeDire,
    )
    monkeypatch.setattr(
        historical,
        "load_fixed_evaluator",
        lambda _source: FakeEvaluator,
    )
    monkeypatch.setattr(
        historical,
        "load_dataset_manifest",
        lambda _root: (dataset_manifest, {"tiny": data}),
    )
    monkeypatch.setattr(historical, "sha256_file", lambda _path: "manifest-hash")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda _index: "Fake H100"
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cuvs", fake_cuvs)
    monkeypatch.setitem(sys.modules, "cuvs.neighbors", fake_neighbors)

    output = tmp_path / "raw" / "variant.jsonl"
    cache = tmp_path / "reference-cache"
    arguments = (
        "9117dc4_index_search_flat",
        tmp_path / "source",
        tmp_path / "evaluator.py",
        tmp_path / "datasets",
        output,
        cache,
    )
    historical.run_variant(*arguments)
    first_lines = output.read_text(encoding="utf-8").splitlines()
    calls_after_first_run = FakeEvaluator.calls
    historical.run_variant(*arguments)

    assert len(first_lines) == 2
    assert output.read_text(encoding="utf-8").splitlines() == first_lines
    assert FakeEvaluator.calls == calls_after_first_run
    assert (cache / "tiny" / "seed-42.json").is_file()
    records = [json.loads(line) for line in first_lines]
    assert {record["configuration"] for record in records} == set(
        historical.CONFIGURATIONS
    )
    assert all(
        record["effective_policy"]
        == {"method": "index_search", "index_type": "flat"}
        for record in records
    )
    assert all(
        set(record["reference_curve_sha256"]) == {"atlas", "ripser"}
        for record in records
    )
