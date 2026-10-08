"""Fast contract tests for the optional convergence benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

from .runner import (
    LEVELS,
    _add_strict_pairing,
    _base_row,
    build_parser,
    summarize,
    write_report,
    write_results_csv,
)


def _completed_row(
    case_index: int,
    target: str,
    level: str,
    wall_seconds: float,
    *,
    geometry_passed: bool = True,
) -> dict[str, object]:
    row = _base_row(
        case_index,
        "N",
        target,
        level,
        LEVELS.index(level),
        f"start-{case_index}-{target}",
    )
    row.update(
        status="completed",
        wall_seconds=wall_seconds,
        process_seconds=wall_seconds * 0.9,
        final_energy_kj_mol=100.0 + wall_seconds,
        best_energy_kj_mol=100.0 + wall_seconds,
        converged=True,
        geometry_passed=geometry_passed,
    )
    return row


def test_cli_exposes_all_reproducibility_inputs() -> None:
    arguments = build_parser().parse_args(
        (
            "--source",
            "source",
            "--cohort",
            "cohort.json",
            "--input",
            "input.smi",
            "--output",
            "output",
            "--workers",
            "8",
            "--cases",
            "1,62",
            "--targets",
            "complex",
        )
    )

    assert arguments.source == Path("source")
    assert arguments.cohort == Path("cohort.json")
    assert arguments.input_path == Path("input.smi")
    assert arguments.output == Path("output")
    assert arguments.workers == 8
    assert arguments.cases == (1, 62)
    assert arguments.targets == ["complex"]


def test_paired_savings_use_the_same_target_strict_run() -> None:
    rows = [
        _completed_row(1, "ligand", "OPENBABEL", 2.0),
        _completed_row(1, "ligand", "FAST", 4.0),
        _completed_row(1, "ligand", "BALANCED", 8.0),
        _completed_row(1, "ligand", "STRICT", 10.0),
    ]

    _add_strict_pairing(rows)

    assert rows[0]["paired_wall_savings_seconds"] == 8.0
    assert rows[0]["paired_wall_savings_percent"] == 80.0
    assert rows[0]["best_energy_difference_vs_strict_kj_mol"] == -8.0
    assert rows[3]["paired_wall_savings_seconds"] == 0.0
    assert rows[3]["paired_wall_savings_percent"] == 0.0


def test_nonfinite_start_remains_in_each_level_denominator() -> None:
    rows = []
    for level in LEVELS:
        rows.append(_completed_row(1, "complex", level, 1.0))
        nonfinite = _base_row(
            62,
            "P",
            "complex",
            level,
            LEVELS.index(level),
            "nonfinite-start",
        )
        nonfinite["status"] = "nonfinite_start"
        rows.append(nonfinite)

    summary = summarize(rows)

    for level in LEVELS:
        values = summary["targets"]["complex"][level]
        assert values["denominator"] == 2
        assert values["completed_count"] == 1
        assert values["status_counts"] == {
            "completed": 1,
            "nonfinite_start": 1,
        }
        assert values["geometry_pass_rate_over_denominator"] == 0.5


def test_energy_difference_aggregation_is_one_value_per_completed_row() -> None:
    first = _completed_row(1, "ligand", "FAST", 1.0)
    second = _completed_row(2, "ligand", "FAST", 2.0)
    first["best_energy_difference_vs_strict_kj_mol"] = -2.0
    second["best_energy_difference_vs_strict_kj_mol"] = 6.0

    values = summarize((first, second))["targets"]["ligand"]["FAST"]

    assert values[
        "median_absolute_best_energy_difference_vs_strict_kj_mol"
    ] == 4.0
    assert values[
        "maximum_absolute_best_energy_difference_vs_strict_kj_mol"
    ] == 6.0


def test_summary_reports_paired_geometry_regressions() -> None:
    rows = [
        _completed_row(1, "ligand", "FAST", 1.0, geometry_passed=False),
        _completed_row(1, "ligand", "STRICT", 2.0),
    ]

    values = summarize(rows)["targets"]["ligand"]["FAST"]

    assert values["quality_vs_strict"] == {
        "both_passed": 0,
        "both_failed": 0,
        "regressed_vs_strict": 1,
        "improved_vs_strict": 0,
        "unavailable": 0,
    }


def test_writers_preserve_level_order_and_report_pairing(tmp_path: Path) -> None:
    rows = [
        _completed_row(2, "ligand", level, float(index + 1))
        for index, level in enumerate(reversed(LEVELS))
    ]
    _add_strict_pairing(rows)
    summary = summarize(rows)

    write_results_csv(tmp_path / "results.csv", rows)
    write_report(tmp_path / "report.md", summary)

    with (tmp_path / "results.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert [record["level"] for record in records] == list(LEVELS)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Ligand" in report
    assert "Median saving vs STRICT" in report
