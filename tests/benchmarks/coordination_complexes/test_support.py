"""Fast tests for benchmark contracts; no force-field benchmark is run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .cli import build_parser
from .configuration import (
    BUILTIN_SUITES,
    SMOKE_SETTINGS,
    STANDARD_SETTINGS,
    RunProfile,
)
from .io import load_smiles
from .rendering import render_experiment
from .reporting import aggregate_run
from .runner import _write_or_check_manifest


def test_builtin_extractant_suite_has_187_records() -> None:
    suite = BUILTIN_SUITES["extractants-eu-187"]
    records = load_smiles(suite.input_path)

    assert len(records) == suite.expected_count == 187
    assert records[0][0] == 1
    assert records[-1][0] == 187


def test_smoke_profile_is_explicitly_smaller_than_standard() -> None:
    assert SMOKE_SETTINGS.epochs < STANDARD_SETTINGS.epochs
    assert SMOKE_SETTINGS.steps_per_epoch < STANDARD_SETTINGS.steps_per_epoch
    assert SMOKE_SETTINGS.max_attempts < STANDARD_SETTINGS.max_attempts


def test_cli_parses_case_selection_without_running_chemistry() -> None:
    arguments = build_parser().parse_args(
        ("--cases", "54,61,109", "--render", "required")
    )

    assert arguments.cases == (54, 61, 109)
    assert arguments.render == "required"


def test_aggregate_writes_reports_and_integrity(tmp_path: Path) -> None:
    suite = BUILTIN_SUITES["extractants-eu-187"]
    case_dir = tmp_path / "cases" / "0001"
    trajectory_dir = case_dir / "trajectory"
    trajectory_dir.mkdir(parents=True)
    (case_dir / "optimized.sdf").write_text("synthetic\n", encoding="utf-8")
    (case_dir / "optimized.mol2").write_text("synthetic\n", encoding="utf-8")
    (trajectory_dir / "archive.json").write_text("{}\n", encoding="utf-8")
    record = {
        "index": 1,
        "smiles": "N",
        "status": "passed",
        "phase": "complete",
        "cbond": {"donor_count": 1},
        "forcefield": {"effective_forcefield": "UFF"},
        "validation": {"passed": True, "failures": [], "warnings": []},
        "optimization": {
            "converged": True,
            "termination_reason": "converged",
            "final_energy": 1.0,
            "best_energy": 1.0,
        },
        "trajectory": {
            "main_frame_count": 2,
            "ligand_build_attempt_count": 1,
        },
        "output_frame_index": 1,
        "output_frame_role": "selected_success_frame",
        "visualization_topology_lossy": False,
        "cbond_seconds": 0.1,
        "ligand_build_seconds": 0.03,
        "coordination_restoration_seconds": 0.04,
        "complex_optimization_seconds": 0.12,
        "forcefield_seconds": 0.2,
        "total_seconds": 0.3,
    }
    (case_dir / "report.json").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    summary = aggregate_run(
        tmp_path,
        [record],
        suite,
        STANDARD_SETTINGS,
        RunProfile.STANDARD,
        (1,),
        wall_seconds=0.4,
        wall_seconds_scope="unit_test",
    )

    assert summary["overall_success_rate"] == 1.0
    assert summary["aggregate_ligand_build_seconds"] == 0.03
    assert summary["aggregate_coordination_restoration_seconds"] == 0.04
    assert summary["aggregate_complex_optimization_seconds"] == 0.12
    assert (tmp_path / "results.csv").is_file()
    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as stream:
        result_row = next(csv.DictReader(stream))
    assert result_row["ligand_build_seconds"] == "0.03"
    assert result_row["coordination_restoration_seconds"] == "0.04"
    assert result_row["complex_optimization_seconds"] == "0.12"
    report_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Aggregate Stage 1 ligand-build time: 0.030 s" in report_text
    assert (
        "Aggregate Stage 2 coordination-restoration time: 0.040 s"
        in report_text
    )
    assert (
        "Aggregate Stage 3 complex-optimization time: 0.120 s"
        in report_text
    )
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "integrity.json").is_file()
    assert (tmp_path / "report.md").is_file()
    integrity = json.loads((tmp_path / "integrity.json").read_text())
    assert integrity["trajectory_archive_count"] == 1


def test_render_off_is_recorded_without_pymol_execution(tmp_path: Path) -> None:
    (tmp_path / "cases").mkdir()

    rendered = render_experiment(
        tmp_path,
        title="Synthetic",
        workers=1,
        mode="off",
    )

    assert rendered == []
    report = json.loads((tmp_path / "render_report.json").read_text())
    assert report["status"] == "disabled"


def test_existing_output_requires_explicit_resume(tmp_path: Path) -> None:
    configuration = {"suite": "synthetic"}
    _write_or_check_manifest(
        tmp_path,
        configuration,
        workers=1,
        resume=False,
    )

    try:
        _write_or_check_manifest(
            tmp_path,
            configuration,
            workers=1,
            resume=False,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing run must require --resume")
