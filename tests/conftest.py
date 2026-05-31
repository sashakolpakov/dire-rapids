import pytest


EXPERIMENTAL_FILES = {
    "test_atlas_approach.py",
    "test_comprehensive.py",
    "test_integrated_atlas.py",
}

BENCHMARK_FILES = {
    "test_atlas_scaling.py",
    "test_cuml_pca.py",
    "test_cuvs_1000d.py",
    "test_cuvs_scaling.py",
    "test_memory_usage.py",
    "test_scaling_high_dim.py",
}

CI_GATED_FILES = {
    "test_cpu_basic.py",
    "test_reducer_runner.py",
}


def pytest_collection_modifyitems(items):
    for item in items:
        name = item.fspath.basename
        if name in EXPERIMENTAL_FILES:
            item.add_marker(pytest.mark.experimental)
        if name in BENCHMARK_FILES:
            item.add_marker(pytest.mark.benchmark)
        if name in CI_GATED_FILES:
            item.add_marker(pytest.mark.ci)
