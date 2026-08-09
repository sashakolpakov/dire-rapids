"""CPU-safe contract tests for the issue #14 historical audit harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
    assert pr_auto["knn_backend"] == "cuvs"
    assert pr_auto["cuvs_knn_method"] == "auto"
    assert pr_auto["cuvs_index_type"] == "auto"
