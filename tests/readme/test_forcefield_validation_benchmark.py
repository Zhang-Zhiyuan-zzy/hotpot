"""Smoke tests for the reproducible README force-field comparison."""

from tests.readme.benchmark_forcefield_validation import run_benchmark


def test_forcefield_validation_benchmark_smoke() -> None:
    results = run_benchmark(("CCO",), repeats=1)

    assert tuple(result.backend for result in results) == (
        "RDKit native",
        "Open Babel native",
        "Hotpot",
    )
    assert all(result.molecule_count == 1 for result in results)
    assert all(result.successful_runs == 1 for result in results)
    assert all(result.quality_passes == 1 for result in results)
