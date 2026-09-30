"""Regression tests for force-field evidence published in README.md."""

import csv
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


def test_coordination_backend_comparison_evidence_matches_readme() -> None:
    asset_root = ROOT / "assets" / "readme"
    json_path = asset_root / "coordination_complex_backend_comparison.json"
    csv_path = asset_root / "coordination_complex_backend_comparison.csv"
    case_csv_path = (
        asset_root / "coordination_complex_backend_comparison_cases.csv"
    )
    png_path = asset_root / "coordination_complex_backend_comparison.png"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    evidence = json.loads(json_path.read_text(encoding="utf-8"))
    results = {row["backend"]: row for row in evidence["results"]}
    expected = {
        "hotpot": (178, 178, 177, 178),
        "rdkit": (174, 174, 27, 0),
        "openbabel": (177, 177, 162, 177),
    }

    assert evidence["protocol"]["sample_count"] == 178
    assert set(results) == set(expected)
    assert results["hotpot"]["convergence_reported_count"] == 178
    assert results["hotpot"]["converged_count"] == 16
    assert results["rdkit"]["convergence_reported_count"] == 174
    assert results["rdkit"]["converged_count"] == 165
    for backend, (
        build_count,
        optimization_count,
        quality_count,
        fully_parameterized_count,
    ) in expected.items():
        result = results[backend]
        assert result["sample_count"] == 178
        assert result["build_success_count"] == build_count
        assert result["optimization_success_count"] == optimization_count
        assert result["quality_pass_count"] == quality_count
        assert (
            result["forcefield_fully_parameterized_count"]
            == fully_parameterized_count
        )

    with csv_path.open(encoding="utf-8", newline="") as stream:
        aggregate_rows = {row["backend"]: row for row in csv.DictReader(stream)}
    assert set(aggregate_rows) == set(expected)
    for backend, counts in expected.items():
        assert int(aggregate_rows[backend]["sample_count"]) == 178
        assert int(aggregate_rows[backend]["build_success_count"]) == counts[0]
        assert (
            int(aggregate_rows[backend]["optimization_success_count"])
            == counts[1]
        )
        assert int(aggregate_rows[backend]["quality_pass_count"]) == counts[2]

    with case_csv_path.open(encoding="utf-8", newline="") as stream:
        case_rows = tuple(csv.DictReader(stream))
    assert len(case_rows) == 3 * 178
    assert {
        backend: sum(row["backend"] == backend for row in case_rows)
        for backend in expected
    } == {backend: 178 for backend in expected}

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png_path.stat().st_size > 10_000
    assert "coordination_complex_backend_comparison.png" in readme
    assert "| Hotpot | 178/178 (100.0%)" in readme
    assert "| RDKit | 174/178 (97.8%)" in readme
    assert "| Open Babel | 177/178 (99.4%)" in readme
    assert "27/178 (15.2%)" in readme
    assert "162/178 (91.0%)" in readme
    assert "partial UFF" in readme
