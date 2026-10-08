"""Publish one four-workflow view of completed coordination benchmarks.

This module performs no chemistry.  It validates and combines one completed
shared-start Hotpot optimizer comparison with one completed native-backend
comparison, then writes the README evidence assets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Mapping, Optional, Sequence

import numpy as np

from .io import sha256_file, write_json
from .backend_comparison import _topology_sha256, _trajectory_topology


WORKFLOWS = (
    "hotpot_optimize_complex",
    "hotpot_optimize",
    "rdkit",
    "openbabel",
)
DISPLAY_NAMES = {
    "hotpot_optimize_complex": "Hotpot optimize_complex",
    "hotpot_optimize": "Hotpot optimize",
    "rdkit": "RDKit",
    "openbabel": "Open Babel",
}
CASE_FIELDS = (
    "index",
    "smiles",
    "workflow",
    "cbond_succeeded",
    "status",
    "build_succeeded",
    "forcefield_supported",
    "forcefield_fully_parameterized",
    "forcefield_parameterization",
    "optimization_succeeded",
    "converged",
    "finite_final_coordinates",
    "topology_preserved",
    "quality_passed",
    "build_seconds",
    "optimization_seconds",
    "workflow_seconds",
    "trajectory_frame_count",
    "trajectory_sdf_written",
    "failed_checks",
    "error_type",
    "error_message",
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _optional_bool(value: object) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _failure_names(validation: Mapping[str, object]) -> str:
    failures = validation.get("failures") or ()
    return ";".join(str(check["name"]) for check in failures)


def _check_passed(
    validation: Mapping[str, object],
    name: str,
) -> Optional[bool]:
    return next(
        (
            bool(check["passed"])
            for check in validation.get("checks") or ()
            if check.get("name") == name
        ),
        None,
    )


def _sha256_paths(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _hotpot_rows(
    optimizer_root: Path,
    optimizer_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    arm_workflows = {
        "complex_optimizer": "hotpot_optimize_complex",
        "ordinary_optimizer": "hotpot_optimize",
    }
    for record in optimizer_records:
        cbond_succeeded = bool(record.get("cbond"))
        for arm_name, workflow in arm_workflows.items():
            arm = (record.get("arms") or {}).get(arm_name) or {}
            validation = arm.get("validation") or {}
            build_seconds = _optional_float(record.get("shared_build_seconds"))
            optimization_seconds = _optional_float(arm.get("elapsed_seconds"))
            workflow_seconds = (
                build_seconds + optimization_seconds
                if build_seconds is not None and optimization_seconds is not None
                else None
            )
            outcome = str(arm.get("outcome") or "not_attempted_cbond")
            finite_coordinates = _check_passed(validation, "finite_coordinates")
            topology_preserved = _check_passed(validation, "topology")
            forcefield_supported = _check_passed(validation, "forcefield_setup")
            optimization_succeeded = (
                outcome in {"passed", "failed_quality"}
                and finite_coordinates is True
            )
            trajectory_payload = arm.get("trajectory") or {}
            trajectory_sdf_path = (
                optimizer_root
                / "cases"
                / f"{int(record['index']):04d}"
                / arm_name
                / "trajectory"
                / "main"
                / "trajectory.sdf"
            )
            rows.append(
                {
                    "index": int(record["index"]),
                    "smiles": str(record["smiles"]),
                    "workflow": workflow,
                    "cbond_succeeded": cbond_succeeded,
                    "status": outcome,
                    "build_succeeded": (
                        bool(record.get("shared_build"))
                        if cbond_succeeded
                        else None
                    ),
                    "forcefield_supported": (
                        forcefield_supported if cbond_succeeded else None
                    ),
                    "forcefield_fully_parameterized": (
                        forcefield_supported if cbond_succeeded else None
                    ),
                    "forcefield_parameterization": (
                        "full" if forcefield_supported is True else ""
                    ),
                    "optimization_succeeded": (
                        optimization_succeeded if cbond_succeeded else None
                    ),
                    "converged": _optional_bool(arm.get("converged")),
                    "finite_final_coordinates": finite_coordinates,
                    "topology_preserved": topology_preserved,
                    "quality_passed": _optional_bool(arm.get("quality_passed")),
                    "build_seconds": build_seconds,
                    "optimization_seconds": optimization_seconds,
                    "workflow_seconds": workflow_seconds,
                    "trajectory_frame_count": trajectory_payload.get(
                        "main_frame_count"
                    ),
                    "trajectory_sdf_written": (
                        trajectory_sdf_path.is_file()
                        if trajectory_payload
                        else None
                    ),
                    "failed_checks": _failure_names(validation),
                    "error_type": arm.get("error_type") or record.get("error_type") or "",
                    "error_message": (
                        arm.get("error_message") or record.get("error_message") or ""
                    ),
                }
            )
    return rows


def _native_rows(
    native_records: Sequence[Mapping[str, str]],
    optimizer_records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    records_by_key = {
        (str(record["backend"]), int(record["index"])): record
        for record in native_records
        if record["backend"] in {"rdkit", "openbabel"}
    }
    rows: list[dict[str, object]] = []
    for workflow in ("rdkit", "openbabel"):
        for optimizer_record in optimizer_records:
            index = int(optimizer_record["index"])
            source = records_by_key.get((workflow, index))
            if source is None:
                rows.append(
                    {
                        "index": index,
                        "smiles": str(optimizer_record["smiles"]),
                        "workflow": workflow,
                        "cbond_succeeded": False,
                        "status": "not_attempted_cbond",
                        "build_succeeded": None,
                        "forcefield_supported": None,
                        "forcefield_fully_parameterized": None,
                        "forcefield_parameterization": "",
                        "optimization_succeeded": None,
                        "converged": None,
                        "finite_final_coordinates": None,
                        "topology_preserved": None,
                        "quality_passed": None,
                        "build_seconds": None,
                        "optimization_seconds": None,
                        "workflow_seconds": None,
                        "trajectory_frame_count": None,
                        "trajectory_sdf_written": None,
                        "failed_checks": "",
                        "error_type": str(optimizer_record.get("error_type") or ""),
                        "error_message": str(
                            optimizer_record.get("error_message") or ""
                        ),
                    }
                )
                continue
            rows.append(
                {
                    "index": index,
                    "smiles": str(optimizer_record["smiles"]),
                    "workflow": workflow,
                    "cbond_succeeded": True,
                    "status": source["status"],
                    "build_succeeded": _optional_bool(source["build_succeeded"]),
                    "forcefield_supported": _optional_bool(
                        source["forcefield_supported"]
                    ),
                    "forcefield_fully_parameterized": _optional_bool(
                        source["forcefield_fully_parameterized"]
                    ),
                    "forcefield_parameterization": source[
                        "forcefield_parameterization"
                    ],
                    "optimization_succeeded": _optional_bool(
                        source["optimization_succeeded"]
                    ),
                    "converged": _optional_bool(source["converged"]),
                    "finite_final_coordinates": _optional_bool(
                        source["finite_final_coordinates"]
                    ),
                    "topology_preserved": _optional_bool(
                        source["topology_preserved"]
                    ),
                    "quality_passed": _optional_bool(source["quality_passed"]),
                    "build_seconds": _optional_float(source["build_seconds"]),
                    "optimization_seconds": _optional_float(
                        source["optimization_seconds"]
                    ),
                    "workflow_seconds": _optional_float(source["total_seconds"]),
                    "trajectory_frame_count": (
                        int(source["trajectory_frame_count"])
                        if source["trajectory_frame_count"]
                        else None
                    ),
                    "trajectory_sdf_written": _optional_bool(
                        source["trajectory_sdf_written"]
                    ),
                    "failed_checks": source["failed_checks"],
                    "error_type": source["error_type"],
                    "error_message": source["error_message"],
                }
            )
    return rows


def _summary_row(
    workflow: str,
    records: Sequence[Mapping[str, object]],
    input_count: int,
) -> dict[str, object]:
    eligible = [record for record in records if record["cbond_succeeded"]]
    durations = [
        float(record["workflow_seconds"])
        for record in eligible
        if record["workflow_seconds"] is not None
    ]
    convergence = [
        bool(record["converged"])
        for record in eligible
        if record["converged"] is not None
    ]
    eligible_count = len(eligible)
    quality_pass_count = sum(
        record["quality_passed"] is True for record in eligible
    )
    return {
        "workflow": workflow,
        "display_name": DISPLAY_NAMES[workflow],
        "input_count": input_count,
        "cbond_eligible_count": eligible_count,
        "not_attempted_count": input_count - eligible_count,
        "build_success_count": sum(
            record["build_succeeded"] is True for record in eligible
        ),
        "optimization_success_count": sum(
            record["optimization_succeeded"] is True for record in eligible
        ),
        "finite_coordinate_count": sum(
            record["finite_final_coordinates"] is True for record in eligible
        ),
        "topology_preserved_count": sum(
            record["topology_preserved"] is True for record in eligible
        ),
        "quality_pass_count": quality_pass_count,
        "quality_pass_rate": quality_pass_count / eligible_count,
        "end_to_end_quality_pass_rate": quality_pass_count / input_count,
        "forcefield_fully_parameterized_count": sum(
            record["forcefield_fully_parameterized"] is True for record in eligible
        ),
        "convergence_reported_count": len(convergence),
        "converged_count": sum(convergence),
        "trajectory_archive_count": sum(
            record["trajectory_frame_count"] is not None
            for record in eligible
        ),
        "trajectory_frame_count": sum(
            int(record["trajectory_frame_count"])
            for record in eligible
            if record["trajectory_frame_count"] is not None
        ),
        "trajectory_sdf_count": sum(
            record["trajectory_sdf_written"] is True
            for record in eligible
        ),
        "median_workflow_seconds": median(durations),
        "aggregate_workflow_seconds": sum(durations),
        "eligible_status_counts": dict(
            Counter(str(record["status"]) for record in eligible)
        ),
    }


def write_comparison_plot(
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    """Plot outcomes over the CBond-eligible cohort."""
    import matplotlib.pyplot as plt

    labels = [str(row["display_name"]) for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.24
    denominator = int(rows[0]["cbond_eligible_count"])
    figure, axis = plt.subplots(figsize=(11.2, 5.2))
    series = (
        ("3D build", "build_success_count", "#4C78A8"),
        ("Finite optimized output", "finite_coordinate_count", "#F58518"),
        ("Standard geometry gate", "quality_pass_count", "#54A24B"),
    )
    for offset, (label, field, color) in zip((-width, 0.0, width), series):
        values = [100.0 * int(row[field]) / denominator for row in rows]
        bars = axis.bar(x + offset, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylabel(f"Success rate across {denominator} CBond complexes (%)")
    axis.set_ylim(0.0, 110.0)
    axis.set_title("Eu-complex construction and optimization outcomes")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
        ncol=3,
    )
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    figure.savefig(path, dpi=200)
    plt.close(figure)


def aggregate_four_way_comparison(
    optimizer_root: Path,
    native_root: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Validate completed runs and publish four-workflow README evidence."""
    optimizer_summary_path = optimizer_root / "summary.json"
    optimizer_manifest_path = optimizer_root / "manifest.json"
    native_summary_path = native_root / "comparison.json"
    native_cases_path = native_root / "case_results.csv"
    canonical_cases_path = native_root / "canonical_cases.json"
    optimizer_summary = _read_json(optimizer_summary_path)
    optimizer_manifest = _read_json(optimizer_manifest_path)
    native_summary = _read_json(native_summary_path)
    canonical_payload = _read_json(canonical_cases_path)
    optimizer_records = [
        _read_json(path)
        for path in sorted((optimizer_root / "cases").glob("*/report.json"))
    ]
    native_records = _read_csv(native_cases_path)

    input_count = int(optimizer_summary["sample_count"])
    eligible_count = int(optimizer_summary["cbond_success_count"])
    optimizer_hash = optimizer_manifest["scientific_configuration"]["input_sha256"]
    native_hash = native_summary["provenance"]["input_sha256"]
    if len(optimizer_records) != input_count:
        raise ValueError(
            f"optimizer results contain {len(optimizer_records)} of {input_count} cases"
        )
    optimizer_indices = [int(record["index"]) for record in optimizer_records]
    if len(set(optimizer_indices)) != len(optimizer_indices):
        raise ValueError("optimizer results contain duplicate case indices")
    if int(native_summary["protocol"]["sample_count"]) != eligible_count:
        raise ValueError("native and Hotpot CBond cohorts differ")
    if optimizer_hash != native_hash:
        raise ValueError("optimizer and native inputs have different SHA-256 values")

    eligible_indices = {
        int(record["index"])
        for record in optimizer_records
        if record.get("cbond")
    }
    native_keys = [
        (record["backend"], int(record["index"]))
        for record in native_records
        if record["backend"] in {"rdkit", "openbabel"}
    ]
    if len(set(native_keys)) != len(native_keys):
        raise ValueError("native results contain duplicate backend/case records")
    for backend in ("rdkit", "openbabel"):
        backend_indices = {
            int(record["index"])
            for record in native_records
            if record["backend"] == backend
        }
        if backend_indices != eligible_indices:
            missing = sorted(eligible_indices - backend_indices)
            unexpected = sorted(backend_indices - eligible_indices)
            raise ValueError(
                f"{backend} cohort differs from the Hotpot CBond cohort; "
                f"missing={missing}, unexpected={unexpected}"
            )

    optimizer_by_index = {
        int(record["index"]): record for record in optimizer_records
    }
    canonical_cases = canonical_payload["cases"]
    canonical_indices = {int(case["index"]) for case in canonical_cases}
    if canonical_indices != eligible_indices:
        raise ValueError("canonical manifest differs from the Hotpot CBond cohort")
    for canonical_case in canonical_cases:
        index = int(canonical_case["index"])
        optimizer_record = optimizer_by_index[index]
        if str(canonical_case["smiles"]) != str(optimizer_record["smiles"]):
            raise ValueError(f"case {index} ligand SMILES differ between runs")
        if tuple(canonical_case["donor_indices"]) != tuple(
            (optimizer_record["cbond"] or {})["donor_indices"]
        ):
            raise ValueError(f"case {index} CBond donor indices differ between runs")
        trajectory_path = (
            optimizer_root
            / "cases"
            / f"{index:04d}"
            / "shared_build"
            / "trajectory"
            / "main"
            / "trajectory.json"
        )
        trajectory = _read_json(trajectory_path)
        topology = _trajectory_topology(trajectory)
        if len(topology["atoms"]) != int(canonical_case["atom_count"]):
            raise ValueError(f"case {index} atom counts differ between runs")
        if len(topology["bonds"]) != int(canonical_case["bond_count"]):
            raise ValueError(f"case {index} bond counts differ between runs")
        if _topology_sha256(topology) != str(canonical_case["topology_sha256"]):
            raise ValueError(f"case {index} topology hashes differ between runs")

    case_rows = _hotpot_rows(optimizer_root, optimizer_records)
    case_rows.extend(_native_rows(native_records, optimizer_records))
    case_rows.sort(key=lambda row: (WORKFLOWS.index(row["workflow"]), row["index"]))
    workflow_rows = {
        workflow: [row for row in case_rows if row["workflow"] == workflow]
        for workflow in WORKFLOWS
    }
    for workflow, rows in workflow_rows.items():
        if len(rows) != input_count:
            raise ValueError(
                f"{workflow} contains {len(rows)} of {input_count} input records"
            )
        if sum(row["cbond_succeeded"] is True for row in rows) != eligible_count:
            raise ValueError(f"{workflow} uses a different CBond-eligible cohort")
    results = [
        _summary_row(workflow, workflow_rows[workflow], input_count)
        for workflow in WORKFLOWS
    ]
    settings = optimizer_summary["settings"]
    payload = {
        "schema_version": 2,
        "protocol": {
            "input_count": input_count,
            "cbond_eligible_count": eligible_count,
            "first_cbond_threshold": settings["first_cbond_threshold"],
            "subsequent_cbond_threshold": settings[
                "subsequent_cbond_threshold"
            ],
            "input": (
                f"one {input_count}-ligand corpus; all workflows use the same "
                f"{eligible_count} explicit-H atom tables and Eu-CBond "
                "connectivities; native backends start coordinate-free while "
                "the two Hotpot arms share identical built coordinates"
            ),
            "validation": "Hotpot standard structure-acceptance policy",
            "optimization_success_definition": (
                "the optimizer returned finite coordinates; this does not imply "
                "convergence to a backend stopping criterion"
            ),
            "timing": {
                "hotpot": (
                    "shared Hotpot build_complex3d time plus the named optimizer "
                    "arm, including in-call trajectory persistence; CBond "
                    "inference excluded"
                ),
                "native": (
                    "backend-native molecule reconstruction through validation "
                    "and structure/trajectory persistence; CBond inference excluded"
                ),
            },
        },
        "provenance": {
            "input_sha256": optimizer_hash,
            "optimizer_summary_sha256": sha256_file(optimizer_summary_path),
            "optimizer_manifest_sha256": sha256_file(optimizer_manifest_path),
            "optimizer_case_reports_sha256": _sha256_paths(
                tuple((optimizer_root / "cases").glob("*/report.json")),
                optimizer_root,
            ),
            "native_comparison_sha256": sha256_file(native_summary_path),
            "native_case_results_sha256": sha256_file(native_cases_path),
            "canonical_cases_sha256": sha256_file(canonical_cases_path),
            "git": optimizer_manifest.get("git"),
            "versions": {
                name: native_summary["provenance"][name]
                for name in ("python", "hotpot-zzy", "rdkit", "openbabel")
            },
        },
        "results": results,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / "coordination_complex_backend_comparison"
    write_json(stem.with_suffix(".json"), payload)
    flat_fields = tuple(key for key in results[0] if key != "eligible_status_counts")
    with stem.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=flat_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in flat_fields}
            for row in results
        )
    with (output_directory / "coordination_complex_backend_comparison_cases.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=CASE_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in CASE_FIELDS}
            for row in case_rows
        )
    write_comparison_plot(results, stem.with_suffix(".png"))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-results", type=Path, required=True)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = build_parser().parse_args(argv)
    aggregate_four_way_comparison(
        arguments.optimizer_results.resolve(),
        arguments.native_results.resolve(),
        arguments.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
