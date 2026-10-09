"""Regression tests for force-field evidence published in README.md."""

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

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
        ("rdkit", "ligand"): 186,
        ("rdkit", "complex"): 30,
        ("openbabel", "ligand"): 181,
        ("openbabel", "complex"): 162,
        ("obwrappers", "ligand"): 183,
        ("obwrappers", "complex"): 164,
        ("hotpot_auto", "ligand"): 182,
        ("hotpot_auto", "complex"): 182,
    }
    labels = {
        "rdkit": "RDKit",
        "openbabel": "Open Babel",
        "obwrappers": "Hotpot `obWrappers` (`FAST`)",
        "hotpot_auto": "Hotpot `optimize_complex`",
    }

    assert evidence["schema_version"] == 1
    assert evidence["protocol"]["input_sample_count"] == 187
    assert evidence["protocol"]["complex_sample_count"] == 182
    assert set(results) == set(expected_passes)
    for key, pass_count in expected_passes.items():
        result = results[key]
        denominator = 187 if key[1] == "ligand" else 182
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
    assert len(case_rows) == 4 * 187 * 2
    for workflow in labels:
        workflow_rows = [row for row in case_rows if row["workflow"] == workflow]
        assert len(workflow_rows) == 187 * 2
        assert sum(row["eligible"] == "True" for row in workflow_rows) == 187 + 182
        assert {
            int(row["index"])
            for row in workflow_rows
            if row["status"] == "not_eligible"
        } == {139, 182, 185, 186, 187}

        ligand = results[(workflow, "ligand")]
        complex_result = results[(workflow, "complex")]
        expected_row = (
            f"| {labels[workflow]} | "
            f"{ligand['quality_pass_count']}/187 "
            f"({100.0 * ligand['quality_pass_rate']:.1f}%) | "
            f"{ligand['median_compute_seconds']:.3f} s | "
            f"{complex_result['quality_pass_count']}/182 "
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


def test_am_gallery_evidence_matches_readme() -> None:
    asset_root = ROOT / "assets" / "readme"
    evidence_path = asset_root / "am_extractant_gallery_evidence.json"
    image_paths = {
        "cbond": asset_root / "am_extractant_cbond_complexes.png",
        "failed_cbond": asset_root / "am_extractant_no_cbond_ligands.png",
    }
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    evidence_text = evidence_path.read_text(encoding="utf-8")
    evidence = json.loads(evidence_text)
    cases = evidence["cases"]
    groups = {
        name: set(group["case_indices"])
        for name, group in evidence["groups"].items()
    }

    assert evidence["schema_version"] == 1
    assert evidence["sample_count"] == 187
    assert len(cases) == 187
    assert groups["cbond"].isdisjoint(groups["failed_cbond"])
    assert groups["cbond"] | groups["failed_cbond"] == set(range(1, 188))
    assert evidence["groups"]["cbond"]["count"] == 182
    assert evidence["groups"]["failed_cbond"]["count"] == 5
    assert groups["failed_cbond"] == {139, 182, 185, 186, 187}
    assert evidence["quality_passed_count"] == 182
    assert evidence["rendered_count"] == 187
    assert evidence["placeholder_count"] == 0

    cbond_cases = [item for item in cases if item["group"] == "cbond"]
    no_cbond_cases = [item for item in cases if item["group"] == "failed_cbond"]
    assert all(item["benchmark_status"] == "passed" for item in cbond_cases)
    assert all(
        item["benchmark_status"] == "failed_cbond" for item in no_cbond_cases
    )
    assert all(item["render_status"] == "rendered" for item in cases)
    assert all(item["explicit_hydrogen_count"] > 0 for item in cases)
    assert all(item["americium_count"] == 1 for item in cbond_cases)
    assert all(item["americium_count"] == 0 for item in no_cbond_cases)
    assert all(
        item["structure_origin"]
        == "optimized_or_last_finite_benchmark_frame"
        for item in cbond_cases
    )
    assert all(
        item["structure_origin"]
        == "visualization_only_explicit_hydrogen_3d_ligand"
        for item in no_cbond_cases
    )

    for group, image_path in image_paths.items():
        image_bytes = image_path.read_bytes()
        image_evidence = evidence["contact_sheets"][group]
        assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(image_bytes) > 10_000
        assert hashlib.sha256(image_bytes).hexdigest() == image_evidence["sha256"]
        with Image.open(image_path) as image:
            assert list(image.size) == [
                image_evidence["width"],
                image_evidence["height"],
            ]

    assert "**182/187** inputs" in readme
    assert "**5/187** inputs" in readme
    assert "**182/182** optimized outputs" in readme
    assert "current generic geometry gate" in readme
    assert "Materials Studio-inspired" in readme
    assert "maximum principal moment axis" in readme
    assert "visualization-only 3D" in readme
    assert "not Am complexes" in readme
    assert "assets/readme/am_extractant_cbond_complexes.png" in readme
    assert "assets/readme/am_extractant_no_cbond_ligands.png" in readme
    assert "/home/" not in evidence_text
    assert "zhangzhiyuan" not in evidence_text.lower()
