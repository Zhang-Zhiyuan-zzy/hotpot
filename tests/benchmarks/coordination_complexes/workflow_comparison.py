"""Aggregate five independent coordination-structure benchmark workflows.

This module performs no molecular construction or optimization.  It accepts
the completed RDKit, Open Babel, Hotpot ``obWrappers`` and Hotpot
``optimize_complex`` and FAST-first ``auto_optimize`` result roots, validates
their shared corpus identities, and publishes the compact evidence used by
the project README.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Optional

from .io import sha256_file, write_json


SCHEMA_VERSION = 1
INPUT_SAMPLE_COUNT = 187
COMPLEX_SAMPLE_COUNT = 181
WORKFLOWS = (
    "rdkit",
    "openbabel",
    "obwrappers",
    "hotpot_optimize_complex",
    "hotpot_auto",
)
FAST_WORKFLOWS = frozenset(
    {"obwrappers", "hotpot_optimize_complex", "hotpot_auto"}
)
TARGETS = ("ligand", "complex")
DISPLAY_NAMES = {
    "rdkit": "RDKit",
    "openbabel": "Open Babel",
    "obwrappers": "Hotpot obWrappers FAST",
    "hotpot_optimize_complex": "Hotpot optimize_complex FAST",
    "hotpot_auto": "Hotpot auto FAST-first",
}
TARGET_NAMES = {
    "ligand": "Ligand",
    "complex": "Metal–ligand complex",
}
SUMMARY_FIELDS = (
    "workflow",
    "display_name",
    "target",
    "target_name",
    "sample_count",
    "quality_pass_count",
    "quality_pass_rate",
    "timed_sample_count",
    "median_compute_seconds",
    "aggregate_compute_seconds",
)
CASE_FIELDS = (
    "index",
    "smiles",
    "workflow",
    "display_name",
    "target",
    "eligible",
    "status",
    "quality_passed",
    "compute_seconds",
    "failed_checks",
    "error_type",
    "error_message",
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return value


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return value


def _number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def _sha256(value: object, location: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{location} must be one SHA-256 hexadecimal digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(
            f"{location} must be one SHA-256 hexadecimal digest"
        ) from error
    return value.lower()


def _assert_close(actual: float, expected: float, location: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(f"{location} is {actual}, expected {expected}")


def _failed_checks(validation: object) -> str:
    if validation is None:
        return ""
    report = _mapping(validation, "case target validation")
    failures = report.get("failures") or ()
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes)):
        raise ValueError("case target validation.failures must be an array")
    names = []
    for failure in failures:
        check = _mapping(failure, "case target validation failure")
        names.append(str(check["name"]))
    return ";".join(names)


def _validate_target_summary(
    summary: Mapping[str, object],
    workflow: str,
    target: str,
) -> dict[str, object]:
    denominator = INPUT_SAMPLE_COUNT if target == "ligand" else COMPLEX_SAMPLE_COUNT
    location = f"{workflow} summary.targets.{target}"
    sample_count = _integer(summary["denominator"], f"{location}.denominator")
    report_count = _integer(summary["report_count"], f"{location}.report_count")
    pass_count = _integer(summary["pass_count"], f"{location}.pass_count")
    pass_rate = _number(summary["pass_rate"], f"{location}.pass_rate")
    timed_count = _integer(summary["timed_count"], f"{location}.timed_count")
    median_seconds = _number(
        summary["median_compute_seconds"],
        f"{location}.median_compute_seconds",
    )
    aggregate_seconds = _number(
        summary["aggregate_compute_seconds"],
        f"{location}.aggregate_compute_seconds",
    )
    if sample_count != denominator or report_count != denominator:
        raise ValueError(
            f"{location} must report all {denominator} benchmark samples"
        )
    if not 0 <= pass_count <= denominator:
        raise ValueError(f"{location}.pass_count is outside its denominator")
    if not 0 < timed_count <= denominator:
        raise ValueError(f"{location}.timed_count is outside its denominator")
    _assert_close(pass_rate, pass_count / denominator, f"{location}.pass_rate")
    if median_seconds < 0.0 or aggregate_seconds < 0.0:
        raise ValueError(f"{location} contains a negative compute duration")
    return {
        "workflow": workflow,
        "display_name": DISPLAY_NAMES[workflow],
        "target": target,
        "target_name": TARGET_NAMES[target],
        "sample_count": denominator,
        "quality_pass_count": pass_count,
        "quality_pass_rate": pass_rate,
        "timed_sample_count": timed_count,
        "median_compute_seconds": median_seconds,
        "aggregate_compute_seconds": aggregate_seconds,
    }


def _validate_run(
    root: Path,
    workflow: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
]:
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    manifest = _read_json(manifest_path)
    summary = _read_json(summary_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{manifest_path} has an unsupported schema version")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{summary_path} has an unsupported schema version")
    if manifest.get("workflow") != workflow or summary.get("workflow") != workflow:
        raise ValueError(f"{root} does not contain the {workflow} workflow")
    if workflow in FAST_WORKFLOWS:
        settings = _mapping(manifest["settings"], f"{workflow} manifest.settings")
        if settings.get("convergence_level") != "FAST":
            raise ValueError(f"{workflow} must use FAST convergence")

    input_identity = _mapping(manifest["input"], f"{workflow} manifest.input")
    cohort_identity = _mapping(manifest["cohort"], f"{workflow} manifest.cohort")
    if _integer(
        input_identity["sample_count"],
        f"{workflow} manifest.input.sample_count",
    ) != INPUT_SAMPLE_COUNT:
        raise ValueError(
            f"{workflow} input denominator must be {INPUT_SAMPLE_COUNT}"
        )
    if _integer(
        cohort_identity["sample_count"],
        f"{workflow} manifest.cohort.sample_count",
    ) != COMPLEX_SAMPLE_COUNT:
        raise ValueError(
            f"{workflow} complex denominator must be {COMPLEX_SAMPLE_COUNT}"
        )
    _sha256(input_identity["sha256"], f"{workflow} manifest.input.sha256")
    _sha256(cohort_identity["sha256"], f"{workflow} manifest.cohort.sha256")
    cohort_indices = cohort_identity["indices"]
    if not isinstance(cohort_indices, list):
        raise ValueError(f"{workflow} manifest.cohort.indices must be an array")
    normalized_indices = [
        _integer(index, f"{workflow} manifest.cohort.indices")
        for index in cohort_indices
    ]
    if (
        len(normalized_indices) != COMPLEX_SAMPLE_COUNT
        or len(set(normalized_indices)) != COMPLEX_SAMPLE_COUNT
        or not set(normalized_indices).issubset(
            range(1, INPUT_SAMPLE_COUNT + 1)
        )
    ):
        raise ValueError(
            f"{workflow} cohort must contain {COMPLEX_SAMPLE_COUNT} unique "
            "input indices"
        )
    if (
        _integer(summary["input_count"], f"{workflow} summary.input_count")
        != INPUT_SAMPLE_COUNT
    ):
        raise ValueError(
            f"{workflow} summary input denominator must be {INPUT_SAMPLE_COUNT}"
        )
    if _integer(
        summary["complex_eligible_count"],
        f"{workflow} summary.complex_eligible_count",
    ) != COMPLEX_SAMPLE_COUNT:
        raise ValueError(
            f"{workflow} summary complex denominator must be "
            f"{COMPLEX_SAMPLE_COUNT}"
        )
    if summary["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError(f"{workflow} summary does not identify its manifest")

    target_summaries = _mapping(summary["targets"], f"{workflow} summary.targets")
    rows = [
        _validate_target_summary(
            _mapping(target_summaries[target], f"{workflow} summary.targets.{target}"),
            workflow,
            target,
        )
        for target in TARGETS
    ]
    return manifest, summary, rows


def _case_rows(
    root: Path,
    workflow: str,
    cohort_indices: set[int],
) -> list[dict[str, object]]:
    paths = sorted((root / "cases").glob("*/report.json"))
    if len(paths) != INPUT_SAMPLE_COUNT:
        raise ValueError(
            f"{workflow} contains {len(paths)} case reports, expected 187"
        )
    reports = [_read_json(path) for path in paths]
    indices = [
        _integer(report["index"], f"{workflow} case index")
        for report in reports
    ]
    if len(set(indices)) != INPUT_SAMPLE_COUNT:
        raise ValueError(f"{workflow} case reports contain duplicate indices")
    if set(indices) != set(range(1, INPUT_SAMPLE_COUNT + 1)):
        raise ValueError(f"{workflow} case reports do not cover the input corpus")

    rows = []
    for report in reports:
        index = int(report["index"])
        if report.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{workflow} case {index} has an unsupported schema")
        if report.get("workflow") != workflow:
            raise ValueError(f"{workflow} case {index} identifies another workflow")
        targets = _mapping(report["targets"], f"{workflow} case {index}.targets")
        for target in TARGETS:
            target_record = _mapping(
                targets[target],
                f"{workflow} case {index}.targets.{target}",
            )
            eligible = target == "ligand" or index in cohort_indices
            status = str(target_record["status"])
            if eligible == (status == "not_eligible"):
                raise ValueError(
                    f"{workflow} case {index} has an inconsistent {target} eligibility"
                )
            quality_passed = target_record.get("quality_passed")
            compute_seconds = target_record.get("compute_seconds")
            if eligible:
                if not isinstance(quality_passed, bool):
                    raise ValueError(
                        f"{workflow} case {index} {target} quality flag is missing"
                    )
                if compute_seconds is not None:
                    compute_seconds = _number(
                        compute_seconds,
                        f"{workflow} case {index} {target} compute_seconds",
                    )
                    if compute_seconds < 0.0:
                        raise ValueError(
                            f"{workflow} case {index} {target} time is negative"
                        )
            else:
                quality_passed = None
                compute_seconds = None
            rows.append(
                {
                    "index": index,
                    "smiles": str(report["smiles"]),
                    "workflow": workflow,
                    "display_name": DISPLAY_NAMES[workflow],
                    "target": target,
                    "eligible": eligible,
                    "status": status,
                    "quality_passed": quality_passed,
                    "compute_seconds": compute_seconds,
                    "failed_checks": (
                        _failed_checks(target_record["validation"])
                        if eligible
                        else ""
                    ),
                    "error_type": target_record.get("error_type") or "",
                    "error_message": target_record.get("error_message") or "",
                }
            )
    return rows


def _validate_case_aggregate(
    rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
) -> None:
    summaries_by_target = {str(row["target"]): row for row in summaries}
    for target in TARGETS:
        eligible = [
            row
            for row in rows
            if row["target"] == target and row["eligible"] is True
        ]
        durations = [
            float(row["compute_seconds"])
            for row in eligible
            if row["status"] in ("passed", "failed_quality")
            and row["compute_seconds"] is not None
        ]
        passed = sum(row["quality_passed"] is True for row in eligible)
        summary = summaries_by_target[target]
        if len(eligible) != int(summary["sample_count"]):
            raise ValueError(f"{target} case denominator differs from its summary")
        if passed != int(summary["quality_pass_count"]):
            raise ValueError(f"{target} case pass count differs from its summary")
        if len(durations) != int(summary["timed_sample_count"]):
            raise ValueError(f"{target} timed case count differs from its summary")
        _assert_close(
            median(durations),
            float(summary["median_compute_seconds"]),
            f"{target} case median compute time",
        )
        _assert_close(
            sum(durations),
            float(summary["aggregate_compute_seconds"]),
            f"{target} case aggregate compute time",
        )


def _write_summary_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in SUMMARY_FIELDS} for row in rows)


def _write_cases_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in CASE_FIELDS} for row in rows)


def write_comparison_plot(
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    """Plot only geometry-gate performance and median compute time."""
    import matplotlib.pyplot as plt

    by_key = {
        (str(row["workflow"]), str(row["target"])): row
        for row in rows
    }
    x = list(range(len(WORKFLOWS)))
    width = 0.36
    colors = {"ligand": "#4C78A8", "complex": "#F58518"}
    figure, (gate_axis, time_axis) = plt.subplots(1, 2, figsize=(14.0, 4.8))
    for position, target in enumerate(TARGETS):
        offset = (position - 0.5) * width
        pass_rates = [
            100.0 * float(by_key[(workflow, target)]["quality_pass_rate"])
            for workflow in WORKFLOWS
        ]
        median_times = [
            float(by_key[(workflow, target)]["median_compute_seconds"])
            for workflow in WORKFLOWS
        ]
        gate_bars = gate_axis.bar(
            [value + offset for value in x],
            pass_rates,
            width,
            label=TARGET_NAMES[target],
            color=colors[target],
        )
        time_bars = time_axis.bar(
            [value + offset for value in x],
            median_times,
            width,
            color=colors[target],
        )
        gate_axis.bar_label(gate_bars, fmt="%.1f%%", padding=2, fontsize=8)
        time_axis.bar_label(time_bars, fmt="%.3g s", padding=2, fontsize=8)

    labels = [DISPLAY_NAMES[workflow].replace(" ", "\n", 1) for workflow in WORKFLOWS]
    for axis in (gate_axis, time_axis):
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
    gate_axis.set_title("Standard geometry gate")
    gate_axis.set_ylabel("Pass rate (%)")
    gate_axis.set_ylim(0.0, 110.0)
    time_axis.set_title("Build + optimize efficiency")
    time_axis.set_ylabel("Median compute time per molecule (s)")
    time_axis.set_ylim(bottom=0.0)
    figure.legend(loc="lower center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    figure.savefig(path, dpi=200)
    plt.close(figure)


def aggregate_workflow_comparison(
    rdkit_root: Path,
    openbabel_root: Path,
    obwrappers_root: Path,
    hotpot_optimize_complex_root: Path,
    hotpot_auto_root: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Validate five independent runs and publish compact README evidence."""
    roots = {
        "rdkit": rdkit_root,
        "openbabel": openbabel_root,
        "obwrappers": obwrappers_root,
        "hotpot_optimize_complex": hotpot_optimize_complex_root,
        "hotpot_auto": hotpot_auto_root,
    }
    manifests: dict[str, dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []

    for workflow in WORKFLOWS:
        manifest, _summary, rows = _validate_run(roots[workflow], workflow)
        manifests[workflow] = manifest
        summary_rows.extend(rows)
        cohort = _mapping(manifest["cohort"], f"{workflow} manifest.cohort")
        workflow_case_rows = _case_rows(
            roots[workflow],
            workflow,
            {int(index) for index in cohort["indices"]},
        )
        _validate_case_aggregate(workflow_case_rows, rows)
        case_rows.extend(workflow_case_rows)

    input_hashes = {
        _sha256(
            _mapping(manifest["input"], f"{workflow} manifest.input")["sha256"],
            f"{workflow} manifest.input.sha256",
        )
        for workflow, manifest in manifests.items()
    }
    cohort_hashes = {
        _sha256(
            _mapping(manifest["cohort"], f"{workflow} manifest.cohort")["sha256"],
            f"{workflow} manifest.cohort.sha256",
        )
        for workflow, manifest in manifests.items()
    }
    cohort_index_sets = {
        tuple(
            sorted(
                int(index)
                for index in _mapping(
                    manifest["cohort"], f"{workflow} manifest.cohort"
                )["indices"]
            )
        )
        for workflow, manifest in manifests.items()
    }
    if len(input_hashes) != 1:
        raise ValueError("workflow input SHA-256 identities differ")
    if len(cohort_hashes) != 1 or len(cohort_index_sets) != 1:
        raise ValueError("workflow complex-cohort identities differ")
    reference_smiles = {
        int(row["index"]): str(row["smiles"])
        for row in case_rows
        if row["workflow"] == WORKFLOWS[0] and row["target"] == "ligand"
    }
    for row in case_rows:
        if str(row["smiles"]) != reference_smiles[int(row["index"])]:
            raise ValueError(
                f"case {int(row['index'])} ligand SMILES differ among workflows"
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "input_sample_count": INPUT_SAMPLE_COUNT,
            "complex_sample_count": COMPLEX_SAMPLE_COUNT,
            "targets": list(TARGETS),
            "geometry_gate": "Hotpot standard geometry acceptance policy",
            "efficiency": (
                "median build + optimize compute time among completed workflows"
            ),
        },
        "provenance": {
            "input_sha256": next(iter(input_hashes)),
            "cohort_sha256": next(iter(cohort_hashes)),
            "runs": {
                workflow: {
                    "manifest_sha256": sha256_file(roots[workflow] / "manifest.json"),
                    "summary_sha256": sha256_file(roots[workflow] / "summary.json"),
                }
                for workflow in WORKFLOWS
            },
        },
        "results": summary_rows,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "coordination_complex_backend_comparison"
    write_json(stem.with_suffix(".json"), payload)
    _write_summary_csv(summary_rows, stem.with_suffix(".csv"))
    cases_path = output_directory / "coordination_complex_backend_comparison_cases.csv"
    case_rows.sort(
        key=lambda row: (
            WORKFLOWS.index(str(row["workflow"])),
            int(row["index"]),
            TARGETS.index(str(row["target"])),
        )
    )
    _write_cases_csv(case_rows, cases_path)
    write_comparison_plot(summary_rows, stem.with_suffix(".png"))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdkit", type=Path, required=True)
    parser.add_argument("--openbabel", type=Path, required=True)
    parser.add_argument("--obwrappers", type=Path, required=True)
    parser.add_argument("--hotpot-optimize-complex", type=Path, required=True)
    parser.add_argument("--hotpot-auto", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = build_parser().parse_args(argv)
    aggregate_workflow_comparison(
        arguments.rdkit.resolve(),
        arguments.openbabel.resolve(),
        arguments.obwrappers.resolve(),
        arguments.hotpot_optimize_complex.resolve(),
        arguments.hotpot_auto.resolve(),
        arguments.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
