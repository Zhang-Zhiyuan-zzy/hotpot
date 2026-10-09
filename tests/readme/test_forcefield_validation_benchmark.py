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
    results = {
        (row["workflow"], row["target"]): row for row in evidence["results"]
    }
    expected_passes = {
        ("rdkit", "ligand"): 185,
        ("rdkit", "complex"): 31,
        ("openbabel", "ligand"): 182,
        ("openbabel", "complex"): 165,
        ("obwrappers", "ligand"): 184,
        ("obwrappers", "complex"): 165,
        ("hotpot_optimize_complex", "ligand"): 182,
        ("hotpot_optimize_complex", "complex"): 180,
        ("hotpot_auto", "ligand"): 182,
        ("hotpot_auto", "complex"): 180,
    }
    labels = {
        "rdkit": "RDKit",
        "openbabel": "Open Babel",
        "obwrappers": "Hotpot `obWrappers` (`FAST`)",
        "hotpot_optimize_complex": "Hotpot `optimize_complex` workflow (`FAST`)",
        "hotpot_auto": (
            "Hotpot automatic workflow (`FAST` first, complex fallback)"
        ),
    }

    assert evidence["schema_version"] == 1
    assert evidence["protocol"]["input_sample_count"] == 187
    assert evidence["protocol"]["complex_sample_count"] == 181
    assert set(results) == set(expected_passes)
    for key, pass_count in expected_passes.items():
        result = results[key]
        denominator = 187 if key[1] == "ligand" else 181
        assert result["sample_count"] == denominator
        assert result["quality_pass_count"] == pass_count
        assert result["quality_pass_rate"] == pass_count / denominator
        assert 0 < result["timed_sample_count"] <= denominator
        assert result["median_compute_seconds"] > 0.0

    with csv_path.open(encoding="utf-8", newline="") as stream:
        aggregate_rows = {
            (row["workflow"], row["target"]): row
            for row in csv.DictReader(stream)
        }
    assert set(aggregate_rows) == set(expected_passes)
    for key, result in results.items():
        assert int(aggregate_rows[key]["quality_pass_count"]) == result[
            "quality_pass_count"
        ]
        assert float(aggregate_rows[key]["median_compute_seconds"]) == result[
            "median_compute_seconds"
        ]

    with case_csv_path.open(encoding="utf-8", newline="") as stream:
        case_rows = tuple(csv.DictReader(stream))
    assert len(case_rows) == 5 * 187 * 2
    for workflow in labels:
        workflow_rows = [row for row in case_rows if row["workflow"] == workflow]
        assert len(workflow_rows) == 187 * 2
        assert sum(row["eligible"] == "True" for row in workflow_rows) == 187 + 181
        assert {
            int(row["index"])
            for row in workflow_rows
            if row["status"] == "not_eligible"
        } == {134, 139, 182, 185, 186, 187}

        ligand = results[(workflow, "ligand")]
        complex_result = results[(workflow, "complex")]
        expected_row = (
            f"| {labels[workflow]} | "
            f"{ligand['quality_pass_count']}/187 "
            f"({100.0 * ligand['quality_pass_rate']:.1f}%) | "
            f"{ligand['median_compute_seconds']:.3f} s | "
            f"{complex_result['quality_pass_count']}/181 "
            f"({100.0 * complex_result['quality_pass_rate']:.1f}%) | "
            f"{complex_result['median_compute_seconds']:.3f} s |"
        )
        assert expected_row in readme

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png_path.stat().st_size > 10_000
    assert "coordination_complex_backend_comparison.png" in readme
    benchmark_section = readme.split("## Validation evidence", maxsplit=1)[1].split(
        "## Scientific boundaries", maxsplit=1
    )[0]
    assert "3D build" not in benchmark_section
    assert "partial UFF" not in benchmark_section
    assert "Case 61" not in benchmark_section
