"""Aggregate case evidence into machine- and human-readable reports."""

from __future__ import annotations

import csv
import json
import platform
import statistics
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .configuration import (
    BenchmarkBackend,
    BenchmarkSettings,
    BenchmarkSuite,
    RunProfile,
)
from .io import write_json
from .pipeline import flatten_record


def _write_structure_collections(
    output_root: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    ordered_records = sorted(records, key=lambda item: int(item["index"]))
    all_blocks = []
    passed_blocks = []
    for record in ordered_records:
        optimized_path = (
            output_root / "cases" / f"{int(record['index']):04d}" / "optimized.sdf"
        )
        if not optimized_path.is_file():
            continue
        block = optimized_path.read_text(encoding="utf-8")
        block = block if block.endswith("\n") else block + "\n"
        all_blocks.append(block)
        if record["status"] == "passed":
            passed_blocks.append(block)
    (output_root / "optimized_all.sdf").write_text(
        "".join(all_blocks),
        encoding="utf-8",
    )
    (output_root / "optimized_passed.sdf").write_text(
        "".join(passed_blocks),
        encoding="utf-8",
    )

    for status in sorted({str(record["status"]) for record in ordered_records}):
        lines = [
            f"{record['smiles']} case_{int(record['index']):04d}"
            for record in ordered_records
            if record["status"] == status
        ]
        (output_root / f"{status}.smi").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )


def _integrity_payload(
    output_root: Path,
    records: Sequence[Mapping[str, object]],
    expected_indices: Sequence[int],
) -> dict[str, object]:
    expected = set(expected_indices)
    record_indices = {int(record["index"]) for record in records}
    optimized_indices = {
        int(record["index"])
        for record in records
        if (
            output_root / "cases" / f"{int(record['index']):04d}" / "optimized.mol2"
        ).is_file()
    }
    archive_indices = {
        int(record["index"])
        for record in records
        if (
            output_root
            / "cases"
            / f"{int(record['index']):04d}"
            / "trajectory"
            / "archive.json"
        ).is_file()
    }
    cbond_success_indices = {
        int(record["index"]) for record in records if record.get("cbond")
    }
    validation_indices = {
        int(record["index"])
        for record in records
        if record.get("validation") is not None
    }
    return {
        "expected_case_count": len(expected),
        "report_count": len(record_indices),
        "missing_report_indices": sorted(expected - record_indices),
        "unexpected_report_indices": sorted(record_indices - expected),
        "cbond_success_count": len(cbond_success_indices),
        "validation_report_count": len(validation_indices),
        "trajectory_archive_count": len(archive_indices),
        "missing_archive_after_cbond_indices": sorted(
            cbond_success_indices - archive_indices
        ),
        "optimized_mol2_count": len(optimized_indices),
        "missing_optimized_for_output_indices": sorted(
            int(record["index"])
            for record in records
            if isinstance(record.get("output_frame_index"), int)
            and int(record["index"]) not in optimized_indices
        ),
        "status_count_matches_reports": (
            sum(Counter(str(record["status"]) for record in records).values())
            == len(record_indices)
        ),
    }


def _runtime_payload() -> dict[str, object]:
    import onnxruntime
    from openbabel import openbabel

    return {
        "python": platform.python_version(),
        "openbabel": openbabel.OBReleaseVersion(),
        "onnxruntime": onnxruntime.__version__,
        "onnx_execution_provider": "CPUExecutionProvider",
    }


def _validation_check_counts(
    records: Sequence[Mapping[str, object]],
) -> tuple[Counter[str], Counter[str]]:
    failures: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    for record in records:
        validation = record.get("validation") or {}
        failures.update(str(check["name"]) for check in validation.get("failures", ()))
        warnings.update(str(check["name"]) for check in validation.get("warnings", ()))
    return failures, warnings


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _failure_case_payload(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    failures = []
    for record in records:
        if record["status"] == "passed":
            continue
        validation = record.get("validation") or {}
        failures.append(
            {
                "index": int(record["index"]),
                "status": str(record["status"]),
                "phase": str(record["phase"]),
                "error_type": record.get("error_type"),
                "error_message": record.get("error_message"),
                "failed_checks": [
                    {
                        "name": check.get("name"),
                        "measured": check.get("measured"),
                        "threshold": check.get("threshold"),
                        "atom_indices": check.get("atom_indices"),
                        "bond_indices": check.get("bond_indices"),
                        "message": check.get("message"),
                    }
                    for check in validation.get("failures", ())
                ],
            }
        )
    return failures


def _summary_payload(
    records: Sequence[Mapping[str, object]],
    suite: BenchmarkSuite,
    settings: BenchmarkSettings,
    profile: RunProfile,
    backend: BenchmarkBackend,
    wall_seconds: Optional[float],
    wall_seconds_scope: str,
) -> dict[str, object]:
    status_counts = Counter(str(record["status"]) for record in records)
    validation_failures, validation_warnings = _validation_check_counts(records)
    elapsed_values = [float(record["total_seconds"]) for record in records]
    forcefield_values = [
        float(record["forcefield_seconds"])
        for record in records
        if record.get("forcefield_seconds") is not None
    ]
    cbond_values = [
        float(record["cbond_seconds"])
        for record in records
        if record.get("cbond_seconds") is not None
    ]
    ligand_build_values = [
        float(record["ligand_build_seconds"])
        for record in records
        if record.get("ligand_build_seconds") is not None
    ]
    coordination_restoration_values = [
        float(record["coordination_restoration_seconds"])
        for record in records
        if record.get("coordination_restoration_seconds") is not None
    ]
    complex_optimization_values = [
        float(record["complex_optimization_seconds"])
        for record in records
        if record.get("complex_optimization_seconds") is not None
    ]
    main_frame_counts = [
        int(record["trajectory"]["main_frame_count"])
        for record in records
        if record.get("trajectory")
    ]
    cbond_completed = sum(bool(record.get("cbond")) for record in records)
    selected_routes = Counter(
        str(record["routing_report"]["selected_route"])
        for record in records
        if record.get("routing_report") is not None
    )
    route_attempt_statuses = Counter(
        f"{attempt['route']}:{attempt['status']}"
        for record in records
        for attempt in (record.get("routing_report") or {}).get("attempts", ())
    )
    preliminary_attempt_counts = [
        int(record["trajectory"].get("preliminary_attempt_count", 0))
        for record in records
        if record.get("trajectory")
    ]
    cbond_success_cases = [
        {
            "index": int(record["index"]),
            "smiles": str(record["smiles"]),
        }
        for record in records
        if record.get("cbond")
    ]
    passed = status_counts["passed"]
    return {
        "suite": suite.to_manifest(),
        "backend": backend.to_manifest(),
        "profile": profile.value,
        "sample_count": len(records),
        "settings": settings.to_manifest(),
        "workflow": backend.workflow,
        "status_counts": dict(status_counts),
        "overall_success_rate": passed / len(records) if records else None,
        "success_rate_after_cbond": (
            passed / cbond_completed if cbond_completed else None
        ),
        "failure_phases": dict(
            Counter(
                str(record["phase"])
                for record in records
                if record["status"] != "passed"
            )
        ),
        "error_types": dict(
            Counter(
                str(record["error_type"])
                for record in records
                if record.get("error_type")
            )
        ),
        "cbond_completed": cbond_completed,
        "cbond_success_count": cbond_completed,
        "cbond_success_cases": cbond_success_cases,
        "forcefield_attempted": cbond_completed,
        "validation_completed": sum(
            record.get("validation") is not None for record in records
        ),
        "quality_passed": passed,
        "forcefield_converged": sum(
            bool((record.get("optimization") or {}).get("converged"))
            for record in records
        ),
        "selected_route_counts": dict(selected_routes),
        "route_attempt_status_counts": dict(route_attempt_statuses),
        "preliminary_attempt_total": sum(preliminary_attempt_counts),
        "validation_failure_checks": dict(validation_failures),
        "validation_warning_checks": dict(validation_warnings),
        "trajectory_archive_count": len(main_frame_counts),
        "trajectory_main_frame_total": sum(main_frame_counts),
        "trajectory_main_frame_median": (
            statistics.median(main_frame_counts) if main_frame_counts else None
        ),
        "trajectory_main_frame_maximum": (
            max(main_frame_counts) if main_frame_counts else None
        ),
        "aggregate_case_seconds": sum(elapsed_values),
        "aggregate_cbond_seconds": sum(cbond_values),
        "aggregate_ligand_build_seconds": sum(ligand_build_values),
        "aggregate_coordination_restoration_seconds": sum(
            coordination_restoration_values
        ),
        "aggregate_complex_optimization_seconds": sum(
            complex_optimization_values
        ),
        "aggregate_forcefield_seconds": sum(forcefield_values),
        "wall_seconds": wall_seconds,
        "wall_seconds_scope": wall_seconds_scope,
        "median_case_seconds": (
            statistics.median(elapsed_values) if elapsed_values else None
        ),
        "p95_case_seconds": _percentile(elapsed_values, 0.95),
        "maximum_case_seconds": max(elapsed_values) if elapsed_values else None,
        "throughput_cases_per_wall_second": (
            len(records) / wall_seconds if wall_seconds else None
        ),
        "failure_cases": _failure_case_payload(records),
        "slowest_cases": [
            {
                "index": int(record["index"]),
                "status": record["status"],
                "seconds": float(record["total_seconds"]),
            }
            for record in sorted(
                records,
                key=lambda item: float(item["total_seconds"]),
                reverse=True,
            )[:10]
        ],
        "runtime": _runtime_payload(),
    }


def _format_rate(value: Optional[float]) -> str:
    return "unavailable" if value is None else f"{100.0 * value:.2f}%"


def _markdown_table(counter: Mapping[str, object]) -> str:
    if not counter:
        return "None."
    rows = ["| Item | Count |", "|---|---:|"]
    rows.extend(
        f"| `{name}` | {count} |"
        for name, count in sorted(counter.items(), key=lambda item: str(item[0]))
    )
    return "\n".join(rows)


def _write_markdown_report(
    output_root: Path,
    summary: Mapping[str, object],
    integrity: Mapping[str, object],
) -> None:
    slowest = summary["slowest_cases"]
    slowest_rows = ["| Case | Status | Seconds |", "|---:|---|---:|"]
    slowest_rows.extend(
        f"| {item['index']:04d} | `{item['status']}` | {item['seconds']:.3f} |"
        for item in slowest
    )
    failure_rows = [
        "| Case | Status | Phase | Reason |",
        "|---:|---|---|---|",
    ]
    for item in summary["failure_cases"]:
        check_names = ", ".join(str(check["name"]) for check in item["failed_checks"])
        reason = check_names or item["error_message"] or item["error_type"] or "unknown"
        escaped_reason = str(reason).replace("|", "\\|")
        failure_rows.append(
            f"| {item['index']:04d} | `{item['status']}` | `{item['phase']}` | "
            f"{escaped_reason} |"
        )
    if len(failure_rows) == 2:
        failure_rows.append("| - | - | - | None |")
    cbond_rows = ["| Case | Ligand SMILES |", "|---:|---|"]
    for item in summary["cbond_success_cases"]:
        escaped_smiles = item["smiles"].replace("|", "\\|")
        cbond_rows.append(f"| {item['index']:04d} | `{escaped_smiles}` |")
    if len(cbond_rows) == 2:
        cbond_rows.append("| - | None |")
    routing_section = ""
    if summary["backend"]["name"] == "hotpot-auto":
        routing_section = f"""
## Automatic routing

Selected routes:

{_markdown_table(summary["selected_route_counts"])}

Route attempt outcomes:

{_markdown_table(summary["route_attempt_status_counts"])}

Rejected preliminary trajectories retained: {summary["preliminary_attempt_total"]}
"""
    text = f"""# Coordination-complex benchmark report

## Scope

- Suite: `{summary["suite"]["name"]}`
- Profile: `{summary["profile"]}`
- Workflow: `{summary["workflow"]}`
- Backend: `{summary["backend"]["name"]}` ({summary["backend"]["description"]})
- First CBond threshold: `{summary["settings"]["first_cbond_threshold"]}`
- Subsequent CBond threshold: `{summary["settings"]["subsequent_cbond_threshold"]}`
- Cases: {summary["sample_count"]}
- Overall success: {_format_rate(summary["overall_success_rate"])}
- Success after CBond: {_format_rate(summary["success_rate_after_cbond"])}
- Wall time: {summary["wall_seconds"]} s ({summary["wall_seconds_scope"]})

The force-field validation report is the geometry and numerical quality gate.
Every CBond-successful case is expected to retain its complete immutable
trajectory archive. A failed optimization exports its last finite frame for
inspection without reclassifying that frame as successful.

## Outcomes

{_markdown_table(summary["status_counts"])}

{routing_section}

## CBond-successful cases

Count: {summary["cbond_completed"]}

{chr(10).join(cbond_rows)}

## Validation failures

{_markdown_table(summary["validation_failure_checks"])}

## Validation warnings

{_markdown_table(summary["validation_warning_checks"])}

## Failure cases

{chr(10).join(failure_rows)}

## Performance

- Aggregate case time: {summary["aggregate_case_seconds"]:.3f} s
- Aggregate CBond time: {summary["aggregate_cbond_seconds"]:.3f} s
- Aggregate Stage 1 ligand-build time: {summary["aggregate_ligand_build_seconds"]:.3f} s
- Aggregate Stage 2 coordination-restoration time: {summary["aggregate_coordination_restoration_seconds"]:.3f} s
- Aggregate Stage 3 complex-optimization time: {summary["aggregate_complex_optimization_seconds"]:.3f} s
- Aggregate force-field time: {summary["aggregate_forcefield_seconds"]:.3f} s
- Median case time: {summary["median_case_seconds"]} s
- P95 case time: {summary["p95_case_seconds"]} s
- Maximum case time: {summary["maximum_case_seconds"]} s
- Throughput: {summary["throughput_cases_per_wall_second"]} cases/s

{chr(10).join(slowest_rows)}

## Integrity

```json
{json.dumps(integrity, indent=2)}
```

## Artifacts

- `manifest.json`: immutable input identity and settings
- `results.csv`: one row per case
- `summary.json`: aggregate scientific and timing metrics
- `integrity.json`: missing-artifact checks
- `cases/NNNN/trajectory/`: lossless full trajectory
- `cases/NNNN/optimized.mol2` and `.sdf`: selected or last finite output frame
- `cases/NNNN/final.png`: optional PyMOL rendering
- `final.png`: optional all-case contact sheet
"""
    (output_root / "report.md").write_text(text, encoding="utf-8")


def aggregate_run(
    output_root: Path,
    records: Sequence[Mapping[str, object]],
    suite: BenchmarkSuite,
    settings: BenchmarkSettings,
    profile: RunProfile,
    expected_indices: Sequence[int],
    *,
    backend: BenchmarkBackend,
    wall_seconds: Optional[float] = None,
    wall_seconds_scope: str = "unavailable",
) -> dict[str, object]:
    """Write all aggregate artifacts and return the summary payload."""
    output_root.mkdir(parents=True, exist_ok=True)
    ordered_records = sorted(records, key=lambda item: int(item["index"]))
    observed_identities = {
        (str(record["backend"]), str(record["workflow"]))
        for record in ordered_records
    }
    expected_identity = {(backend.name, backend.workflow)}
    if observed_identities != expected_identity:
        raise ValueError(
            "case reports do not match the configured backend and workflow: "
            f"expected={sorted(expected_identity)}, "
            f"observed={sorted(observed_identities)}"
        )
    _write_structure_collections(output_root, ordered_records)
    rows = [flatten_record(record) for record in ordered_records]
    if rows:
        with (output_root / "results.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = _summary_payload(
        ordered_records,
        suite,
        settings,
        profile,
        backend,
        wall_seconds,
        wall_seconds_scope,
    )
    integrity = _integrity_payload(output_root, ordered_records, expected_indices)
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "integrity.json", integrity)
    _write_markdown_report(output_root, summary, integrity)
    return summary
