#!/usr/bin/env python3
"""Frozen historical-commit audit for the withdrawn topology preset.

This is an issue-resolution harness, not a public preset.  It compares the
exact parameters formerly exported as ``TOPOLOGY_TUNED`` with default DiRe at
the preset-introduction commit, the Revision 3 audit commit, and the original
PR #12 head.  Layout seeds, input arrays, row-index topology subsets, and the
topology evaluator are paired across every comparison.

The runner deliberately loads DiRe from an explicitly checked-out source tree
while loading the topology evaluator from the pinned ``293b622`` source file.
That separation prevents the evaluator from changing with the implementation
under test.  Raw records are append-only JSON Lines so an interrupted GPU run
can resume without silently accepting an incomplete result matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
DATASET_SEED = 42
LAYOUT_SEEDS = tuple(range(42, 62))
TOPOLOGY_SUBSET_SEED_OFFSET = 1_000
TOPOLOGY_SUBSET_SIZE = 1_000
PRACTICAL_BAND = 0.05

REVISIONS = {
    "preset_introduction": "9117dc45a3e130fa1d636dfd181f3e97960c5b3b",
    "audited_main": "293b622cc79fa8ea6fd5b54009e0930e3385b22f",
    "pr12_head": "471ac168eb2e6638a84f700fe077f29f20e24488",
}

VARIANTS = {
    "9117dc4_index_search_flat": {
        "revision": REVISIONS["preset_introduction"],
        "policy": "index_search_flat",
    },
    "293b622_index_search_flat": {
        "revision": REVISIONS["audited_main"],
        "policy": "index_search_flat",
    },
    "471ac16_index_search_flat": {
        "revision": REVISIONS["pr12_head"],
        "policy": "index_search_flat",
    },
    "471ac16_auto_all_neighbors": {
        "revision": REVISIONS["pr12_head"],
        "policy": "pr12_auto_all_neighbors",
    },
}

CONFIGURATIONS = {
    "default": {
        "init": "pca",
        "n_neighbors": 16,
        "max_iter_layout": 128,
    },
    "former_topology_preset": {
        "init": "spectral",
        "n_neighbors": 15,
        "spread": 3.6,
        "min_dist": 1e-2,
        "cutoff": 42.0,
        "neg_ratio": 8,
        "max_iter_layout": 150,
    },
}

DATASETS = ("blobs", "disk", "moons", "mnist", "levine13", "levine32")
METRICS = (
    "atlas_dtw_beta0",
    "atlas_dtw_beta1",
    "ripser_dtw_beta0",
    "ripser_dtw_beta1",
)
ATLAS_METRICS = frozenset({"atlas_dtw_beta0", "atlas_dtw_beta1"})

# The evaluator is loaded directly from this exact file at audited main.
EVALUATOR_SHA256 = "553cc327cb0397c3fc1b058109f720d0db7b1fbe02d064924b07e5f774479ad9"

CYTOF_SOURCES = {
    "levine13": {
        "url": (
            "https://raw.githubusercontent.com/lmweber/"
            "benchmark-data-Levine-13-dim/"
            "23b414060396f3f0d9274035e5b7acff1de8634d/"
            "data/Levine_13dim.fcs"
        ),
        "sha256": "9a7cbfd8e258d06d8ac309c3868c67c44fee661753f4dfa2adcc2cab68849089",
    },
    "levine32": {
        "url": (
            "https://raw.githubusercontent.com/lmweber/"
            "benchmark-data-Levine-32-dim/"
            "9323796d9b3d5dc5446497d1ef5474fb9ff7497c/"
            "data/Levine_32dim.fcs"
        ),
        "sha256": "0519ce964b801026c4dafcb3672c05c6e0f6582835ca3e4181074698d17e12e0",
    },
}


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(array, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == expected_sha256:
        return destination
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with urllib.request.urlopen(url, timeout=180) as response:  # nosec B310
            with temporary.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"download hash mismatch for {url}: {actual} != {expected_sha256}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _take_fixed_rows(data: np.ndarray, count: int = 10_000) -> np.ndarray:
    if len(data) <= count:
        return np.asarray(data, dtype=np.float32)
    rng = np.random.default_rng(DATASET_SEED)
    indices = rng.choice(len(data), size=count, replace=False)
    return np.asarray(data[indices], dtype=np.float32)


def _load_cytof(path: Path) -> np.ndarray:
    try:
        import flowio
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "CyTOF preparation requires flowio and pandas; install the bench extra"
        ) from exc

    fcs = flowio.FlowData(str(path))
    labels = fcs.pnn_labels or [f"Ch{index}" for index in range(fcs.channel_count)]
    frame = pd.DataFrame(fcs.as_array(), columns=labels)
    if "label" in frame.columns:
        frame = frame[frame["label"].notna()].copy()
    numeric = frame.drop(
        columns=[name for name in ("label", "individual") if name in frame],
        errors="ignore",
    ).select_dtypes(include=[np.number])
    data = numeric.to_numpy(dtype=np.float32, copy=False)
    return np.arcsinh(data / 5.0).astype(np.float32)


def materialize_datasets(output: Path, requested: Iterable[str] = DATASETS) -> dict:
    """Create one checksummed input array per requested dataset."""
    from sklearn.datasets import fetch_openml, make_blobs, make_moons

    requested = tuple(requested)
    unknown = set(requested) - set(DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")

    output.mkdir(parents=True, exist_ok=True)
    source_root = output / "sources"
    records = {}
    for name in requested:
        if name == "blobs":
            data, _labels = make_blobs(
                n_samples=10_000,
                centers=12,
                n_features=1_000,
                random_state=DATASET_SEED,
            )
            source = "sklearn.make_blobs(seed=42,centers=12,features=1000)"
        elif name == "disk":
            rng = np.random.default_rng(DATASET_SEED)
            points = rng.standard_normal((10_000, 2))
            radii = np.sqrt(rng.random(10_000)).reshape(-1, 1)
            data = points * radii / np.linalg.norm(points, axis=1).reshape(-1, 1)
            source = "frozen disk_uniform implementation(seed=42,features=2)"
        elif name == "moons":
            data, _labels = make_moons(
                n_samples=10_000, noise=0.05, random_state=DATASET_SEED
            )
            source = "sklearn.make_moons(seed=42,noise=0.05)"
        elif name == "mnist":
            dataset = fetch_openml(data_id=554, as_frame=False, parser="auto")
            data = np.asarray(dataset.data, dtype=np.float32) / 255.0
            data = _take_fixed_rows(data)
            source = "OpenML data_id=554; scaled [0,1]; fixed seed-42 10000 rows"
        else:
            source_record = CYTOF_SOURCES[name]
            fcs_path = download_verified(
                source_record["url"],
                source_root / f"{name}.fcs",
                source_record["sha256"],
            )
            data = _take_fixed_rows(_load_cytof(fcs_path))
            source = (
                f"{source_record['url']}; sha256={source_record['sha256']}; "
                "arcsinh cofactor 5; fixed seed-42 10000 rows"
            )

        data = np.asarray(data, dtype=np.float32, order="C")
        if data.shape[0] != 10_000:
            raise RuntimeError(f"{name} has {data.shape[0]} rows, expected 10000")
        path = output / f"{name}.npy"
        save_array(path, data)
        records[name] = {
            "path": path.name,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "array_sha256": array_sha256(data),
            "file_sha256": sha256_file(path),
            "source": source,
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_seed": DATASET_SEED,
        "datasets": records,
        "versions": package_versions(("numpy", "scikit-learn", "flowio", "pandas")),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_dataset_manifest(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = {}
    for name in DATASETS:
        record = manifest.get("datasets", {}).get(name)
        if record is None:
            raise RuntimeError(f"dataset manifest is missing {name}")
        path = root / record["path"]
        if sha256_file(path) != record["file_sha256"]:
            raise RuntimeError(f"dataset file hash mismatch: {path}")
        data = np.load(path, allow_pickle=False)
        if array_sha256(data) != record["array_sha256"]:
            raise RuntimeError(f"dataset array hash mismatch: {path}")
        arrays[name] = np.asarray(data, dtype=np.float32)
    return manifest, arrays


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def load_fixed_evaluator(source: Path):
    actual = sha256_file(source)
    if actual != EVALUATOR_SHA256:
        raise RuntimeError(
            f"fixed evaluator hash mismatch: {actual} != {EVALUATOR_SHA256}"
        )
    specification = importlib.util.spec_from_file_location(
        "_dire_issue14_fixed_betti_293b622", source
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load fixed evaluator: {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def import_target_dire(source_root: Path, revision: str):
    if git_commit(source_root) != revision:
        raise RuntimeError(
            f"source checkout {source_root} is not the requested revision {revision}"
        )
    sys.path.insert(0, str(source_root))
    try:
        import dire_rapids
    finally:
        sys.path.pop(0)
    loaded = Path(dire_rapids.__file__).resolve()
    if source_root.resolve() not in loaded.parents:
        raise RuntimeError(f"loaded DiRe from {loaded}, not {source_root}")
    return dire_rapids


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def topology_subset_indices(count: int, layout_seed: int) -> np.ndarray:
    subset_seed = layout_seed + TOPOLOGY_SUBSET_SEED_OFFSET
    rng = np.random.default_rng(subset_seed)
    return np.sort(
        rng.choice(count, size=min(TOPOLOGY_SUBSET_SIZE, count), replace=False)
    )


def subset_sha256(indices: np.ndarray) -> str:
    return sha256_bytes(np.asarray(indices, dtype=np.int64).tobytes(order="C"))


def curve_distances(
    reference: dict, embedding: dict, sample_size: int
) -> dict[str, float]:
    from fastdtw import fastdtw

    output = {}
    for dimension in (0, 1):
        reference_curve = np.column_stack(
            (reference["filtration_values"], reference[f"beta_{dimension}"])
        )
        embedding_curve = np.column_stack(
            (embedding["filtration_values"], embedding[f"beta_{dimension}"])
        )
        distance, _path = fastdtw(reference_curve, embedding_curve, dist=2)
        output[f"dtw_beta{dimension}"] = float(distance / sample_size)
    return output


def reducer_kwargs(revision: str, policy: str, configuration: str, seed: int) -> dict:
    kwargs = {
        "backend": "cuvs",
        "n_components": 2,
        "random_state": seed,
        "verbose": False,
        **CONFIGURATIONS[configuration],
    }
    if revision != REVISIONS["preset_introduction"]:
        kwargs["knn_backend"] = "cuvs"
    if policy == "index_search_flat":
        kwargs["cuvs_index_type"] = "flat"
        if revision == REVISIONS["pr12_head"]:
            kwargs["cuvs_knn_method"] = "index_search"
    elif policy == "pr12_auto_all_neighbors":
        if revision != REVISIONS["pr12_head"]:
            raise ValueError("the PR #12 automatic policy exists only at 471ac16")
        kwargs["cuvs_index_type"] = "auto"
        kwargs["cuvs_knn_method"] = "auto"
    else:
        raise ValueError(f"unknown policy: {policy}")
    return kwargs


def effective_policy(reducer, revision: str, requested_policy: str) -> dict:
    method = getattr(reducer, "effective_cuvs_knn_method_", None)
    if method is None:
        method = getattr(reducer, "_last_cuvs_knn_method", None)
    if method is None and revision != REVISIONS["pr12_head"]:
        method = "index_search"
    index_type = getattr(reducer, "effective_cuvs_index_type_", None)
    if index_type is None and method == "index_search":
        index_type = getattr(reducer, "cuvs_index_type", None)

    if requested_policy == "index_search_flat":
        if method != "index_search" or index_type != "flat":
            raise RuntimeError(
                f"expected exact flat index_search, got method={method}, index={index_type}"
            )
    elif method != "all_neighbors":
        raise RuntimeError(f"expected PR #12 automatic all_neighbors, got {method}")
    return {"method": method, "index_type": index_type}


def existing_record_keys(
    path: Path, variant: str
) -> set[tuple[str, int, str]]:
    keys = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(f"unsupported record schema at {path}:{line_number}")
            specification = VARIANTS[variant]
            if (
                record.get("variant") != variant
                or record.get("revision") != specification["revision"]
                or record.get("requested_policy") != specification["policy"]
            ):
                raise RuntimeError(
                    f"stale or foreign record at {path}:{line_number}"
                )
            key = (record["dataset"], int(record["layout_seed"]), record["configuration"])
            if key in keys:
                raise RuntimeError(f"duplicate record {key} at {path}:{line_number}")
            keys.add(key)
    return keys


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_ready(record), sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_or_compute_reference_curves(
    evaluator,
    data: np.ndarray,
    indices: np.ndarray,
    dataset_name: str,
    dataset_sha256: str,
    layout_seed: int,
    cache_root: Path,
) -> tuple[dict, dict]:
    """Load or atomically retain the fixed input curves shared by variants."""
    cache_path = cache_root / dataset_name / f"seed-{layout_seed}.json"
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "dataset_array_sha256": dataset_sha256,
        "layout_seed": layout_seed,
        "subset_seed": layout_seed + TOPOLOGY_SUBSET_SEED_OFFSET,
        "subset_indices_sha256": subset_sha256(indices),
        "subset_size": len(indices),
        "evaluator_sha256": EVALUATOR_SHA256,
    }
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if any(cached.get(key) != value for key, value in expected_identity.items()):
            raise RuntimeError(f"reference-curve cache identity mismatch: {cache_path}")
        return cached["atlas"], cached["ripser"]

    subset = data[indices]
    atlas = evaluator.compute_betti_curve_gpu(
        subset,
        k_neighbors=20,
        density_threshold=0.8,
        overlap_factor=1.5,
        n_steps=100,
    )
    ripser = evaluator.compute_betti_curve_ripser(
        subset, n_steps=100, maxdim=1
    )
    write_json(
        cache_path,
        {
            **expected_identity,
            "atlas": atlas,
            "ripser": ripser,
        },
    )
    return atlas, ripser


def run_variant(
    variant: str,
    source_root: Path,
    evaluator_source: Path,
    datasets_root: Path,
    output: Path,
    reference_cache: Path,
) -> None:
    """Run or resume one revision/policy variant on a CUDA host."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    specification = VARIANTS[variant]
    revision = specification["revision"]
    policy = specification["policy"]
    dire_rapids = import_target_dire(source_root, revision)
    evaluator = load_fixed_evaluator(evaluator_source)

    import cupy as cp
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("historical audit requires a CUDA GPU")
    try:
        from cuvs.neighbors import brute_force as _brute_force  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("historical audit requires cuVS brute_force") from exc

    dataset_manifest, datasets = load_dataset_manifest(datasets_root)
    output = output.resolve()
    completed = existing_record_keys(output, variant)
    manifest_path = output.with_suffix(".manifest.json")
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        "versions": package_versions(
            (
                "numpy",
                "torch",
                "cupy-cuda12x",
                "cupy-cuda13x",
                "cuvs-cu12",
                "cuvs-cu13",
                "cuml-cu12",
                "cuml-cu13",
                "scikit-learn",
                "fastdtw",
                "ripser",
            )
        ),
    }
    environment_sha256 = sha256_bytes(
        json.dumps(environment, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "revision": revision,
        "policy": policy,
        "source_root": str(source_root.resolve()),
        "evaluator_sha256": EVALUATOR_SHA256,
        "dataset_manifest_sha256": sha256_file(datasets_root / "manifest.json"),
        "dataset_array_sha256": {
            name: record["array_sha256"]
            for name, record in dataset_manifest["datasets"].items()
        },
        "layout_seeds": list(LAYOUT_SEEDS),
        "topology_subset_seed_offset": TOPOLOGY_SUBSET_SEED_OFFSET,
        "topology_subset_size": TOPOLOGY_SUBSET_SIZE,
        "configurations": CONFIGURATIONS,
        "environment": environment,
        "environment_sha256": environment_sha256,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        stable_keys = (
            "schema_version",
            "variant",
            "revision",
            "policy",
            "evaluator_sha256",
            "dataset_manifest_sha256",
            "dataset_array_sha256",
            "layout_seeds",
            "topology_subset_seed_offset",
            "topology_subset_size",
            "configurations",
            "environment",
            "environment_sha256",
        )
        if any(previous.get(key) != run_manifest.get(key) for key in stable_keys):
            raise RuntimeError(f"resume manifest does not match current run: {manifest_path}")
    else:
        write_json(manifest_path, run_manifest)

    for dataset_name in DATASETS:
        data = datasets[dataset_name]
        for seed in LAYOUT_SEEDS:
            pending = [
                configuration
                for configuration in CONFIGURATIONS
                if (dataset_name, seed, configuration) not in completed
            ]
            if not pending:
                continue
            indices = topology_subset_indices(len(data), seed)
            reference_atlas, reference_ripser = load_or_compute_reference_curves(
                evaluator,
                data,
                indices,
                dataset_name,
                dataset_manifest["datasets"][dataset_name]["array_sha256"],
                seed,
                reference_cache,
            )
            for configuration in pending:
                key = (dataset_name, seed, configuration)
                if key in completed:
                    continue
                kwargs = reducer_kwargs(revision, policy, configuration, seed)
                reducer = dire_rapids.create_dire(**kwargs)
                started = time.perf_counter()
                embedding = reducer.fit_transform(data)
                torch.cuda.synchronize()
                fit_seconds = time.perf_counter() - started
                if hasattr(embedding, "get"):
                    embedding = embedding.get()
                if torch.is_tensor(embedding):
                    embedding = embedding.detach().cpu().numpy()
                embedding = np.asarray(embedding, dtype=np.float32)
                if embedding.shape != (len(data), 2) or not np.all(np.isfinite(embedding)):
                    raise RuntimeError(
                        f"invalid embedding for {variant}/{dataset_name}/{seed}/{configuration}"
                    )
                effective = effective_policy(reducer, revision, policy)

                embedded_subset = embedding[indices]
                embedding_atlas = evaluator.compute_betti_curve_gpu(
                    embedded_subset,
                    k_neighbors=20,
                    density_threshold=0.8,
                    overlap_factor=1.5,
                    n_steps=100,
                )
                embedding_ripser = evaluator.compute_betti_curve_ripser(
                    embedded_subset, n_steps=100, maxdim=1
                )
                atlas = curve_distances(
                    reference_atlas, embedding_atlas, len(indices)
                )
                ripser = curve_distances(
                    reference_ripser, embedding_ripser, len(indices)
                )
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "variant": variant,
                    "revision": revision,
                    "requested_policy": policy,
                    "effective_policy": effective,
                    "environment_sha256": environment_sha256,
                    "dataset": dataset_name,
                    "dataset_array_sha256": dataset_manifest["datasets"][dataset_name][
                        "array_sha256"
                    ],
                    "configuration": configuration,
                    "parameters": kwargs,
                    "layout_seed": seed,
                    "subset_seed": seed + TOPOLOGY_SUBSET_SEED_OFFSET,
                    "subset_indices_sha256": subset_sha256(indices),
                    "subset_size": len(indices),
                    "fit_seconds": fit_seconds,
                    "metrics": {
                        "atlas_dtw_beta0": atlas["dtw_beta0"],
                        "atlas_dtw_beta1": atlas["dtw_beta1"],
                        "ripser_dtw_beta0": ripser["dtw_beta0"],
                        "ripser_dtw_beta1": ripser["dtw_beta1"],
                    },
                }
                append_record(output, record)
                completed.add(key)
                del reducer, embedding
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()
                print(
                    f"completed {variant} {dataset_name} seed={seed} {configuration}",
                    flush=True,
                )


def read_records(paths: Iterable[Path]) -> list[dict]:
    records = []
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (
                    record.get("variant"),
                    record.get("dataset"),
                    record.get("layout_seed"),
                    record.get("configuration"),
                )
                if key in seen:
                    raise RuntimeError(f"duplicate result key {key} at {path}:{line_number}")
                seen.add(key)
                records.append(record)
    return records


def mean_interval(values: Iterable[float]) -> tuple[float, float, float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        raise ValueError("cannot summarize an empty sample")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return mean, std, mean - half_width, mean + half_width


def validate_complete_records(records: list[dict], repeats: int = 20) -> dict:
    by_key = {}
    expected_seeds = set(LAYOUT_SEEDS[:repeats])
    for record in records:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("unsupported historical audit record schema")
        variant = record["variant"]
        if variant not in VARIANTS:
            raise RuntimeError(f"unknown result variant: {variant}")
        if record["revision"] != VARIANTS[variant]["revision"]:
            raise RuntimeError(f"wrong revision for {variant}")
        if record["requested_policy"] != VARIANTS[variant]["policy"]:
            raise RuntimeError(f"wrong policy for {variant}")
        key = (
            variant,
            record["dataset"],
            int(record["layout_seed"]),
            record["configuration"],
        )
        if key in by_key:
            raise RuntimeError(f"duplicate result key: {key}")
        if set(record["metrics"]) != set(METRICS):
            raise RuntimeError(f"wrong metric set for {key}")
        expected_method = (
            "all_neighbors"
            if VARIANTS[variant]["policy"] == "pr12_auto_all_neighbors"
            else "index_search"
        )
        expected_index = None if expected_method == "all_neighbors" else "flat"
        if record.get("effective_policy") != {
            "method": expected_method,
            "index_type": expected_index,
        }:
            raise RuntimeError(f"wrong effective policy for {key}")
        if int(record.get("subset_seed", -1)) != int(record["layout_seed"]) + 1_000:
            raise RuntimeError(f"wrong subset seed for {key}")
        if int(record.get("subset_size", -1)) != TOPOLOGY_SUBSET_SIZE:
            raise RuntimeError(f"wrong subset size for {key}")
        parameters = record.get("parameters", {})
        for name, value in CONFIGURATIONS[record["configuration"]].items():
            if parameters.get(name) != value:
                raise RuntimeError(f"wrong configuration parameter {name} for {key}")
        if (
            parameters.get("backend") != "cuvs"
            or parameters.get("n_components") != 2
            or parameters.get("random_state") != record["layout_seed"]
        ):
            raise RuntimeError(f"wrong common reducer parameters for {key}")
        for metric, value in record["metrics"].items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise RuntimeError(f"invalid {metric} for {key}: {value}")
        by_key[key] = record

    expected = {
        (variant, dataset, seed, configuration)
        for variant in VARIANTS
        for dataset in DATASETS
        for seed in expected_seeds
        for configuration in CONFIGURATIONS
    }
    actual = set(by_key)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"historical audit matrix is incomplete: missing={missing[:5]} "
            f"({len(missing)} total), extra={extra[:5]} ({len(extra)} total)"
        )

    environment_hashes = {record.get("environment_sha256") for record in records}
    if len(environment_hashes) != 1 or None in environment_hashes:
        raise RuntimeError(
            "historical revisions were not run in one identical recorded environment"
        )

    for variant in VARIANTS:
        for dataset in DATASETS:
            for seed in expected_seeds:
                left = by_key[(variant, dataset, seed, "default")]
                right = by_key[(variant, dataset, seed, "former_topology_preset")]
                for field in (
                    "dataset_array_sha256",
                    "subset_seed",
                    "subset_indices_sha256",
                    "subset_size",
                ):
                    if left[field] != right[field]:
                        raise RuntimeError(
                            f"unpaired {field} for {variant}/{dataset}/seed={seed}"
                        )

    for dataset in DATASETS:
        for seed in expected_seeds:
            reference = by_key[
                ("9117dc4_index_search_flat", dataset, seed, "default")
            ]
            for variant in VARIANTS:
                candidate = by_key[(variant, dataset, seed, "default")]
                for field in (
                    "dataset_array_sha256",
                    "subset_seed",
                    "subset_indices_sha256",
                    "subset_size",
                ):
                    if reference[field] != candidate[field]:
                        raise RuntimeError(
                            f"cross-revision {field} mismatch for {dataset}/seed={seed}"
                        )
    return by_key


def _classification(gap: float, relative_gap: float, low: float, high: float) -> str:
    if relative_gap > PRACTICAL_BAND and low > 0:
        return "preset worse"
    if relative_gap < -PRACTICAL_BAND and high < 0:
        return "preset better"
    if gap > 0:
        return "practically/statistically inconclusive; preset numerically worse"
    if gap < 0:
        return "practically/statistically inconclusive; preset numerically better"
    return "equal"


def paired_effect_rows(by_key: dict, repeats: int = 20) -> list[dict]:
    rows = []
    seeds = LAYOUT_SEEDS[:repeats]
    for variant in VARIANTS:
        for dataset in DATASETS:
            for metric in METRICS:
                defaults = np.asarray(
                    [by_key[(variant, dataset, seed, "default")]["metrics"][metric] for seed in seeds]
                )
                presets = np.asarray(
                    [
                        by_key[(variant, dataset, seed, "former_topology_preset")][
                            "metrics"
                        ][metric]
                        for seed in seeds
                    ]
                )
                gaps = presets - defaults
                gap, gap_std, low, high = mean_interval(gaps)
                default_mean = float(np.mean(defaults))
                preset_mean = float(np.mean(presets))
                relative_gap = gap / abs(default_mean) if default_mean else math.nan
                rows.append(
                    {
                        "variant": variant,
                        "revision": VARIANTS[variant]["revision"],
                        "policy": VARIANTS[variant]["policy"],
                        "dataset": dataset,
                        "metric": metric,
                        "paired_count": repeats,
                        "default_mean": default_mean,
                        "preset_mean": preset_mean,
                        "paired_gap_preset_minus_default_mean": gap,
                        "paired_gap_std": gap_std,
                        "paired_mean_95pct_low": low,
                        "paired_mean_95pct_high": high,
                        "relative_gap": relative_gap,
                        "classification": _classification(gap, relative_gap, low, high),
                    }
                )
    return rows


def difference_rows(by_key: dict, repeats: int = 20) -> list[dict]:
    comparisons = (
        ("293b622_index_search_flat", "9117dc4_index_search_flat", "implementation"),
        ("471ac16_index_search_flat", "9117dc4_index_search_flat", "implementation"),
        ("471ac16_auto_all_neighbors", "471ac16_index_search_flat", "graph_policy"),
    )
    rows = []
    seeds = LAYOUT_SEEDS[:repeats]
    for candidate, reference, comparison_kind in comparisons:
        for dataset in DATASETS:
            for metric in METRICS:
                changes = []
                reference_defaults = []
                for seed in seeds:
                    reference_default = by_key[(reference, dataset, seed, "default")][
                        "metrics"
                    ][metric]
                    reference_preset = by_key[
                        (reference, dataset, seed, "former_topology_preset")
                    ]["metrics"][metric]
                    candidate_default = by_key[(candidate, dataset, seed, "default")][
                        "metrics"
                    ][metric]
                    candidate_preset = by_key[
                        (candidate, dataset, seed, "former_topology_preset")
                    ]["metrics"][metric]
                    changes.append(
                        (candidate_preset - candidate_default)
                        - (reference_preset - reference_default)
                    )
                    reference_defaults.append(reference_default)
                change, change_std, low, high = mean_interval(changes)
                scale = abs(float(np.mean(reference_defaults)))
                normalized = change / scale if scale else math.nan
                material = (
                    math.isfinite(normalized)
                    and abs(normalized) > PRACTICAL_BAND
                    and (low > 0 or high < 0)
                )
                rows.append(
                    {
                        "comparison_kind": comparison_kind,
                        "candidate_variant": candidate,
                        "reference_variant": reference,
                        "dataset": dataset,
                        "metric": metric,
                        "paired_count": repeats,
                        "difference_in_preset_gap_mean": change,
                        "difference_in_preset_gap_std": change_std,
                        "paired_mean_95pct_low": low,
                        "paired_mean_95pct_high": high,
                        "reference_default_mean_scale": scale,
                        "normalized_difference": normalized,
                        "material_change": material,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize_records(records: list[dict], output: Path, repeats: int = 20) -> dict:
    by_key = validate_complete_records(records, repeats=repeats)
    effects = paired_effect_rows(by_key, repeats=repeats)
    differences = difference_rows(by_key, repeats=repeats)

    atlas_effects = [row for row in effects if row["metric"] in ATLAS_METRICS]
    atlas_differences = [
        row
        for row in differences
        if row["metric"] in ATLAS_METRICS
        and row["comparison_kind"] == "implementation"
    ]
    material_implementation_changes = [
        row for row in atlas_differences if row["material_change"]
    ]
    by_variant = defaultdict(dict)
    for variant in VARIANTS:
        cells = [row for row in atlas_effects if row["variant"] == variant]
        by_variant[variant] = {
            "cells": len(cells),
            "preset_numerically_worse": sum(
                row["paired_gap_preset_minus_default_mean"] > 0 for row in cells
            ),
            "preset_worse_over_5pct": sum(row["relative_gap"] > PRACTICAL_BAND for row in cells),
            "preset_worse_interval_excludes_zero": sum(
                row["paired_mean_95pct_low"] > 0 for row in cells
            ),
            "preset_materially_better": sum(
                row["relative_gap"] < -PRACTICAL_BAND
                and row["paired_mean_95pct_high"] < 0
                for row in cells
            ),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "paired_repeats": repeats,
        "practical_band": PRACTICAL_BAND,
        "materiality_rule": (
            "absolute difference-in-paired-preset-gap exceeds 5% of the "
            "reference default mean and the descriptive paired 95% interval excludes zero"
        ),
        "atlas_preset_effect_by_variant": by_variant,
        "material_post_9117_implementation_change": bool(
            material_implementation_changes
        ),
        "material_post_9117_implementation_change_cells": material_implementation_changes,
        "conclusion": (
            "At least one Atlas cell shows a material post-9117 implementation change; "
            "inspect the retained difference rows before attributing preset failure."
            if material_implementation_changes
            else "No Atlas cell shows a material post-9117 implementation change under "
            "the predeclared paired rule; the preset failure is consistent with objective "
            "mismatch/out-of-sample over-specialization rather than an implementation regression."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "topology_preset_historical_paired_effects.csv", effects)
    write_csv(output / "topology_preset_historical_differences.csv", differences)
    write_json(output / "topology_preset_historical_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="materialize frozen input arrays")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))

    run = subparsers.add_parser("run", help="run or resume one GPU variant")
    run.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--evaluator-source", type=Path, required=True)
    run.add_argument("--datasets-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--reference-cache", type=Path, required=True)

    summarize = subparsers.add_parser("summarize", help="validate and summarize raw records")
    summarize.add_argument("--inputs", nargs="+", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        manifest = materialize_datasets(args.output, args.datasets)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "run":
        run_variant(
            args.variant,
            args.source_root,
            args.evaluator_source,
            args.datasets_root,
            args.output,
            args.reference_cache,
        )
    else:
        records = read_records(args.inputs)
        summary = summarize_records(records, args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
