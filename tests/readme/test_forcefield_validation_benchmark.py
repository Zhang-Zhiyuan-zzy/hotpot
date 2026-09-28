"""Smoke tests for the reproducible README force-field comparison."""

import json
from pathlib import Path

from tests.readme.benchmark_forcefield_validation import run_benchmark


ROOT = Path(__file__).resolve().parents[2]


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


def test_extractant_validation_evidence_matches_dataset() -> None:
    evidence = json.loads(
        (ROOT / "assets/readme/extractant_validation_20260928.json").read_text(
            encoding="utf-8"
        )
    )
    inputs = [
        line
        for line in (ROOT / evidence["input"]).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    statuses = evidence["results"]["status_counts"]

    assert len(inputs) == evidence["protocol"]["input_count"] == 187
    assert sum(statuses.values()) == 187
    assert evidence["integrity"] == {
        "report_count": 187,
        "missing_report_indices": [],
        "missing_archive_after_cbond_indices": [],
        "missing_optimized_output_indices": [],
        "readback_issues": [],
    }
