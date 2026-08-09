"""CPU-safe tests for the crossed Atlas/Ripser preset search."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


HARNESS = (
    Path(__file__).resolve().parents[1]
    / "benchmarking"
    / "bench_topology_preset_search.py"
)
SPECIFICATION = importlib.util.spec_from_file_location(
    "bench_topology_preset_search", HARNESS
)
assert SPECIFICATION is not None and SPECIFICATION.loader is not None
search = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(search)


@pytest.mark.cpu
def test_retained_h100_search_archive_contains_complete_evidence():
    archive = (
        Path(__file__).resolve().parent
        / "data"
        / "topology_preset_search_h100_audit.tar.gz"
    )
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        "feb553d81aa5cc01adc957eecf02b85c6a9e2ae7003c4b4580d8900f62e68c45"
    )
    expected_records = {
        "search-production-auto": 272,
        "search-local-auto": 152,
        "search-atlas-fine-auto": 72,
        "validation-seed42": 12,
        "validation-spread08-seeds42-61": 120,
        "validation-atlas-maxiter96-seeds42-61": 120,
    }
    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
        references = {
            name
            for name in names
            if name.startswith("reference-cache/") and name.endswith(".json")
        }
        assert len(references) == 4 * 2
        for stem, count in expected_records.items():
            raw_name = f"raw/{stem}.jsonl"
            assert raw_name in names
            assert f"raw/{stem}.manifest.json" in names
            stream = bundle.extractfile(raw_name)
            assert stream is not None
            assert len(stream.read().splitlines()) == count
        reproducibility_stream = bundle.extractfile(
            "reprocheck/validation-atlas-maxiter96-seed42-repeat.jsonl"
        )
        assert reproducibility_stream is not None
        assert len(reproducibility_stream.read().splitlines()) == 6
        summary_stream = bundle.extractfile(
            "summary/topology_preset_search_summary.json"
        )
        assert summary_stream is not None
        summary = json.loads(summary_stream.read())
        assert summary["decision"]["ripser_tuned_export"] == "RIPSER_TUNED"
        assert summary["decision"]["atlas_tuned_export"] == "ATLAS_TUNED"
        assert summary["decision"]["canonical_evaluator_exports"] == [
            "ATLAS_TUNED",
            "RIPSER_TUNED",
        ]
        assert summary["decision"]["separate_evaluator_names"] is True
        assert (
            summary["decision"]["canonical_parameter_sets_currently_distinct"]
            is True
        )
        assert summary["decision"]["atlas_tuned_parameters"] != summary[
            "decision"
        ]["ripser_tuned_parameters"]
        confirmation = summary["held_out_confirmation"]
        assert confirmation["layout_seeds"] == 20
        assert confirmation["ripser_vs_default"]["mean_wins"] == 11
        atlas = summary["held_out_atlas_confirmation"]
        assert atlas["layout_seeds"] == 20
        assert atlas["atlas_vs_default"]["mean_wins"] == 11
        assert atlas["atlas_vs_default"]["paired_interval_wins"] == 8
        assert (
            atlas["atlas_vs_best_cellwise_umap_tsne"]["geometric_ratio"]
            < 0.9
        )
        reproducibility = summary["seed42_reproducibility_check"]
        assert reproducibility["topology_metrics_with_nonzero_drift"] == 2
        assert reproducibility["atlas_geometric_repeat_to_first_ratio"] < 1.01
        assert reproducibility["ripser_geometric_repeat_to_first_ratio"] == 1.0


@pytest.mark.cpu
def test_frozen_tuning_manifest_is_disjoint_and_complete():
    manifest = json.loads(
        (
            Path(__file__).resolve().parent
            / "data"
            / "topology_preset_tuning_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["suite"] == "issue-14-disjoint-openml-tuning"
    assert set(manifest["datasets"]) == set(search.TUNING_DATASETS)
    assert set(manifest["datasets"]).isdisjoint(
        {"blobs", "disk", "moons", "mnist", "levine13", "levine32"}
    )
    for name, source in search.TUNING_DATASETS.items():
        record = manifest["datasets"][name]
        assert record["openml_id"] == source["openml_id"]
        assert len(record["data_array_sha256"]) == 64
        assert len(record["labels_array_sha256"]) == 64


@pytest.mark.cpu
def test_candidate_design_is_deterministic_bounded_and_separate_by_role():
    first = search.candidate_parameters(8)
    second = search.candidate_parameters(8)

    assert first == second
    assert set(first) == {"default", "withdrawn_9117"} | {
        f"sobol_{index:03d}" for index in range(8)
    }
    assert first["default"]["role"] == "control"
    assert first["withdrawn_9117"]["role"] == "withdrawn_control"
    for name, record in first.items():
        if not name.startswith("sobol_"):
            continue
        params = record["parameters"]
        assert params["init"] in ("pca", "spectral", "random")
        assert 8 <= params["n_neighbors"] <= 48
        assert 2.0 <= params["cutoff"] <= 42.0
        assert 0.5 <= params["spread"] <= 4.0
        assert 1e-4 <= params["min_dist"] <= 1e-1
        assert 2 <= params["neg_ratio"] <= 32
        assert 64 <= params["max_iter_layout"] <= 256


@pytest.mark.cpu
def test_candidate_design_requires_power_of_two_budget():
    with pytest.raises(ValueError, match="power of two"):
        search.candidate_parameters(7)


@pytest.mark.cpu
def test_local_refinement_changes_exactly_one_default_parameter():
    candidates = search.local_candidate_parameters()

    assert candidates["default"]["parameters"] == search.DEFAULT_PARAMETERS
    assert len(candidates) == 1 + 18
    for name, record in candidates.items():
        if name == "default":
            continue
        changed = {
            key
            for key, value in record["parameters"].items()
            if value != search.DEFAULT_PARAMETERS[key]
        }
        assert len(changed) == 1


@pytest.mark.cpu
def test_atlas_refinement_is_small_and_distinct_from_ripser_tuned():
    candidates = search.atlas_candidate_parameters()
    ripser = candidates["ripser_tuned"]

    assert candidates["default"]["parameters"] == search.DEFAULT_PARAMETERS
    assert ripser["role"] == "incumbent_control"
    assert ripser["parameters"] == {
        **search.DEFAULT_PARAMETERS,
        "spread": 0.8,
    }
    assert len(candidates) == 2 + 7
    for name, record in candidates.items():
        if record["role"] != "candidate":
            continue
        assert name.startswith("atlas_")
        assert record["parameters"] != ripser["parameters"]
        changed = {
            key
            for key, value in record["parameters"].items()
            if value != ripser["parameters"][key]
        }
        assert len(changed) == 1


@pytest.mark.cpu
def test_validation_can_retain_only_a_search_shortlist_subset():
    summary = {
        "selections": {
            "atlas": "atlas_candidate",
            "ripser": "ripser_candidate",
            "compromise": "ripser_candidate",
        }
    }

    assert search.validation_candidate_names(summary, None) == (
        "atlas_candidate",
        "ripser_candidate",
    )
    assert search.validation_candidate_names(summary, ("ripser_candidate",)) == (
        "ripser_candidate",
    )
    with pytest.raises(ValueError, match="not shortlisted"):
        search.validation_candidate_names(summary, ("unsafe_candidate",))


@pytest.mark.cpu
def test_summary_can_select_distinct_atlas_and_ripser_candidates(monkeypatch):
    monkeypatch.setattr(search, "TUNING_DATASETS", {"tiny": {"openml_id": 1}})
    candidates = {
        "default": {"role": "control", "parameters": {}},
        "atlas_candidate": {"role": "candidate", "parameters": {"spread": 1}},
        "ripser_candidate": {"role": "candidate", "parameters": {"spread": 2}},
    }
    records = []
    for candidate in candidates:
        for seed in search.SEARCH_SEEDS:
            atlas = 1.0
            ripser = 1.0
            if candidate == "atlas_candidate":
                atlas, ripser = 0.5, 1.2
            elif candidate == "ripser_candidate":
                atlas, ripser = 1.1, 0.4
            records.append(
                {
                    "candidate": candidate,
                    "dataset": "tiny",
                    "layout_seed": seed,
                    "knn_accuracy": 0.9,
                    "local": {"neighbor_mean": 0.8, "stress": 1.0},
                    "global_distance_spearman": 0.8,
                    "metrics": {
                        backend: {metric: value for metric in search.TOPOLOGY_METRICS}
                        for backend, value in (("atlas", atlas), ("ripser", ripser))
                    },
                }
            )

    summary = search.summarize_records(records, candidates)

    assert summary["selections"]["atlas"] == "atlas_candidate"
    assert summary["selections"]["ripser"] == "ripser_candidate"
    with pytest.raises(RuntimeError, match="duplicate"):
        search.summarize_records([*records, records[0]], candidates)


@pytest.mark.cpu
def test_summary_retains_diagnostics_when_no_candidate_passes_quality_guard(
    monkeypatch,
):
    monkeypatch.setattr(search, "TUNING_DATASETS", {"tiny": {"openml_id": 1}})
    candidates = {
        "default": {"role": "control", "parameters": {}},
        "unsafe_candidate": {"role": "candidate", "parameters": {"spread": 2}},
    }
    records = []
    for candidate in candidates:
        for seed in search.SEARCH_SEEDS:
            unsafe = candidate == "unsafe_candidate"
            records.append(
                {
                    "candidate": candidate,
                    "dataset": "tiny",
                    "layout_seed": seed,
                    "knn_accuracy": 0.7 if unsafe else 0.9,
                    "local": {"neighbor_mean": 0.6 if unsafe else 0.8, "stress": 1.0},
                    "global_distance_spearman": 0.6 if unsafe else 0.8,
                    "metrics": {
                        backend: {
                            metric: 0.5 if unsafe else 1.0
                            for metric in search.TOPOLOGY_METRICS
                        }
                        for backend in ("atlas", "ripser")
                    },
                }
            )

    summary = search.summarize_records(records, candidates)

    assert summary["eligible_candidate_count"] == 0
    assert summary["selections"] == {}
    assert summary["unconstrained_diagnostic_winners"] == {
        backend: "unsafe_candidate" for backend in ("atlas", "ripser", "compromise")
    }
    assert "diagnostic only" in summary["selection_status"]


@pytest.mark.cpu
def test_validation_summary_uses_paired_atlas_and_seed42_ripser(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(search.historical, "DATASETS", ("tiny",))
    monkeypatch.setattr(search.historical, "LAYOUT_SEEDS", (42, 43))
    candidates = {"atlas_candidate": {"spread": 1}, "ripser_candidate": {"spread": 2}}
    environment = {}
    manifest = {
        "schema_version": search.SCHEMA_VERSION,
        "source_revision": "revision",
        "layout_seeds": [42, 43],
        "candidates": candidates,
        "candidate_sha256": search.json_sha256(candidates),
        "dataset_array_sha256": {"tiny": "dataset-hash"},
        "environment": environment,
        "environment_sha256": search.json_sha256(environment),
        "policy": {"expected_effective_method": search.EXPECTED_AUTO_METHOD},
    }
    baselines = {
        "datasets": {
            "tiny": {
                "thresholds": {
                    metric: {"best_method": "umap", "best_mean": 1.0}
                    for metric in search.TOPOLOGY_METRICS
                },
                "methods": {
                    "umap": {
                        "seeds": [42, 43],
                        "atlas": {
                            metric: {"values": [1.0, 1.0]}
                            for metric in search.TOPOLOGY_METRICS
                        },
                    }
                },
            }
        },
        "ripser_canonical": {
            "datasets": {
                "tiny": {
                    "thresholds": {
                        metric: {"best_method": "opentsne", "best_value": 1.0}
                        for metric in search.TOPOLOGY_METRICS
                    }
                }
            }
        },
    }
    records = []
    for candidate in candidates:
        for seed in (42, 43):
            atlas_value = 0.5 if candidate == "atlas_candidate" else 1.1
            ripser_value = 1.2 if candidate == "atlas_candidate" else 0.4
            records.append(
                {
                    "schema_version": search.SCHEMA_VERSION,
                    "source_revision": "revision",
                    "environment_sha256": search.json_sha256(environment),
                    "candidate": candidate,
                    "dataset": "tiny",
                    "layout_seed": seed,
                    "parameters": candidates[candidate],
                    "effective_policy": {
                        "method": "index_search",
                        "index_type": "flat",
                    },
                    "dataset_array_sha256": "dataset-hash",
                    "subset_indices_sha256": f"subset-{seed}",
                    "reference_curve_sha256": {
                        "atlas": f"atlas-{seed}",
                        "ripser": f"ripser-{seed}",
                    },
                    "metrics": {
                        "atlas": {
                            metric: atlas_value for metric in search.TOPOLOGY_METRICS
                        },
                        "ripser": {
                            metric: ripser_value for metric in search.TOPOLOGY_METRICS
                        },
                    },
                }
            )
    manifest_path = tmp_path / "validation.manifest.json"
    baseline_path = tmp_path / "baselines.json"
    input_path = tmp_path / "validation.jsonl"
    output_path = tmp_path / "summary.json"
    default_audit_path = tmp_path / "default-audit.tar.gz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    baseline_path.write_text(json.dumps(baselines), encoding="utf-8")
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    default_records = []
    for seed in (42, 43):
        default_records.append(
            {
                "configuration": "default",
                "variant": search.HISTORICAL_DEFAULT_VARIANT,
                "dataset": "tiny",
                "layout_seed": seed,
                "dataset_array_sha256": "dataset-hash",
                "subset_indices_sha256": f"subset-{seed}",
                "reference_curve_sha256": {
                    "atlas": f"atlas-{seed}",
                    "ripser": f"ripser-{seed}",
                },
                "effective_policy": {
                    "method": "index_search",
                    "index_type": "flat",
                },
                "metrics": {
                    f"{backend}_{metric}": 1.0
                    for backend in ("atlas", "ripser")
                    for metric in search.TOPOLOGY_METRICS
                },
            }
        )
    payload = "".join(json.dumps(record) + "\n" for record in default_records).encode()
    with tarfile.open(default_audit_path, "w:gz") as bundle:
        info = tarfile.TarInfo(
            f"raw/{search.HISTORICAL_DEFAULT_VARIANT}.jsonl"
        )
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))

    summary = search.summarize_validation(
        input_path,
        manifest_path,
        baseline_path,
        output_path,
        default_audit_path,
    )

    assert summary["selections"]["atlas"] == "atlas_candidate"
    assert summary["selections"]["ripser_seed42_screen"] == "ripser_candidate"
    assert summary["scores"]["atlas_candidate"]["atlas_interval_wins"] == 2
    assert summary["scores"]["ripser_candidate"]["ripser_seed42_wins"] == 2
    default_comparison = summary["scores"]["atlas_candidate"][
        "historical_default_comparison"
    ]
    assert default_comparison["atlas_geometric_ratio_to_default"] == pytest.approx(
        0.5
    )
    assert default_comparison["atlas_interval_wins"] == 2
