"""Regression tests for xTB evidence published in README.md."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_xtb_coordination_evidence_matches_readme_and_report() -> None:
    evidence_path = ROOT / "assets" / "readme" / "xtb_coordination_benchmark.json"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    report = (
        ROOT
        / "plan"
        / "12_xtb_workflow"
        / "xtb_workflow_validation_report.md"
    ).read_text(encoding="utf-8")
    evidence_text = evidence_path.read_text(encoding="utf-8")
    evidence = json.loads(evidence_text)

    assert evidence["schema_version"] == 1
    assert evidence["backend"] == {
        "name": "xTB",
        "version": "6.7.1",
        "revision": "edcfbbe",
    }
    assert evidence["official_validation"] == {
        "direct_parity_passed": 5,
        "controlled_pipeline_passed": 3,
    }
    assert evidence["protocol"]["input_sample_count"] == 187
    assert evidence["protocol"]["complex_sample_count"] == 182
    assert (
        evidence["protocol"]["case_workers"]
        * evidence["protocol"]["threads_per_case"]
        == evidence["protocol"]["requested_cores"]
    )
    assert evidence["protocol"]["requested_cores"] == 64
    assert evidence["protocol"]["manifest_sha256"] in report
    assert evidence["protocol"]["summary_sha256"] in report
    assert f"{evidence['protocol']['wall_seconds']:.2f} s wall time" in report
    assert "xTB 6.7.1, revision `edcfbbe`" in report

    results = {
        (row["target"], row["route"]): row for row in evidence["results"]
    }
    assert set(results) == {
        ("ligand", "direct-gfn2"),
        ("ligand", "gfnff-gfn2"),
        ("complex", "direct-gfn2"),
        ("complex", "gfnff-gfn2"),
    }
    for result in results.values():
        assert (
            result["quality_pass_count"]
            + result["quality_failure_count"]
            + result["execution_failure_count"]
            == result["sample_count"]
        )
        assert result["quality_pass_rate"] == (
            result["quality_pass_count"] / result["sample_count"]
        )
        assert result["median_seconds"] > 0.0
    for route, label in (
        ("direct-gfn2", "Direct GFN2"),
        ("gfnff-gfn2", "GFN-FF → GFN2"),
    ):
        ligand = results[("ligand", route)]
        complex_result = results[("complex", route)]
        expected_row = (
            f"| {label} | "
            f"{ligand['quality_pass_count']}/{ligand['sample_count']} "
            f"({100.0 * ligand['quality_pass_rate']:.1f}%) | "
            f"{ligand['median_seconds']:.3f} s | "
            f"{complex_result['quality_pass_count']}/"
            f"{complex_result['sample_count']} "
            f"({100.0 * complex_result['quality_pass_rate']:.1f}%) | "
            f"{complex_result['median_seconds']:.3f} s |"
        )
        assert expected_row in readme

        report_label = (
            "Ligand, direct GFN2"
            if route == "direct-gfn2"
            else "Ligand, GFN-FF -> GFN2"
        )
        complex_report_label = (
            "Eu complex, direct GFN2"
            if route == "direct-gfn2"
            else "Eu complex, GFN-FF -> GFN2"
        )
        assert (
            f"| {report_label} | "
            f"{ligand['quality_pass_count']}/{ligand['sample_count']} | "
            f"{ligand['quality_failure_count']} | "
            f"{ligand['execution_failure_count']} | "
            f"{100.0 * ligand['quality_pass_rate']:.3f}% | "
            f"{ligand['median_seconds']:.3f} s |"
        ) in report
        assert (
            f"| {complex_report_label} | "
            f"{complex_result['quality_pass_count']}/"
            f"{complex_result['sample_count']} | "
            f"{complex_result['quality_failure_count']} | "
            f"{complex_result['execution_failure_count']} | "
            f"{100.0 * complex_result['quality_pass_rate']:.3f}% | "
            f"{complex_result['median_seconds']:.3f} s |"
        ) in report

    assert "five direct-parity checks" in normalized_readme
    assert "three controlled coordination pipelines" in normalized_readme.lower()
    assert "xtb_coordination_benchmark.json" in readme
    assert "/home/" not in evidence_text
    assert "zhangzhiyuan" not in evidence_text.lower()
