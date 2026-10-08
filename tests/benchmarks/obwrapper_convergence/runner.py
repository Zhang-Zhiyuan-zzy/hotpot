"""Run four obWrappers convergence policies from identical 3D structures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, process_time
from typing import Optional

import numpy as np


SCHEMA_VERSION = 1
LEVELS = ("OPENBABEL", "FAST", "BALANCED", "STRICT")
LEVEL_POLICIES = {
    "OPENBABEL": {
        "backend_stop_required": True,
        "independent_requirement": "usable numerical state",
    },
    "FAST": {
        "backend_stop_required": True,
        "maximum_rms_gradient_kj_mol_angstrom": 3.0,
        "maximum_gradient_kj_mol_angstrom": 10.0,
    },
    "BALANCED": {
        "backend_stop_required": True,
        "maximum_rms_gradient_kj_mol_angstrom": 1.0,
        "maximum_gradient_kj_mol_angstrom": 5.0,
    },
    "STRICT": {
        "backend_stop_required": True,
        "maximum_gradient_backend_units_per_angstrom": 0.1,
        "note": "converted to kJ/mol/angstrom using the backend energy unit",
    },
}
TARGETS = ("ligand", "complex")
DEFAULT_FORCEFIELD = "UFF"
DEFAULT_EPOCHS = 100
DEFAULT_STEPS_PER_EPOCH = 100
RESULT_COLUMNS = (
    "case_index",
    "smiles",
    "target",
    "level",
    "level_value",
    "execution_order",
    "status",
    "start_sha256",
    "final_coordinates_sha256",
    "wall_seconds",
    "process_seconds",
    "epochs_completed",
    "steps_submitted",
    "initialization_steps",
    "termination_reason",
    "converged",
    "terminal_converged",
    "final_energy_kj_mol",
    "best_energy_kj_mol",
    "rms_gradient_kj_mol_angstrom",
    "max_gradient_kj_mol_angstrom",
    "exploded",
    "geometry_passed",
    "geometry_failure_names",
    "strict_best_energy_kj_mol",
    "best_energy_difference_vs_strict_kj_mol",
    "strict_wall_seconds",
    "paired_wall_savings_seconds",
    "paired_wall_savings_percent",
    "strict_process_seconds",
    "paired_process_savings_seconds",
    "paired_process_savings_percent",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class BenchmarkConfiguration:
    """Immutable scientific and execution settings for one benchmark run."""

    source: Path
    cohort: Path
    input_path: Path
    output: Path
    workers: int
    forcefield: str
    epochs: int
    steps_per_epoch: int
    targets: tuple[str, ...]
    cases: Optional[tuple[int, ...]]
    resume: bool


@dataclass(frozen=True)
class StartState:
    """One immutable build-complete topology and coordinate state."""

    atoms: tuple[Mapping[str, object], ...]
    bonds: tuple[Mapping[str, object], ...]
    coordinates: np.ndarray
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinates_sha256(coordinates: np.ndarray) -> str:
    array = np.ascontiguousarray(coordinates, dtype="<f8")
    return hashlib.sha256(memoryview(array)).hexdigest()


def _implementation_sha256(repository_root: Path) -> str:
    root = repository_root / "hotpot/cheminfo/obWrappers"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".pyi", ".cpp", ".hpp"}
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repository_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(repository_root):
        return str(resolved.relative_to(repository_root))
    return resolved.name


def _runtime_identity() -> dict[str, object]:
    """Describe the exact native runtime used by the benchmark."""
    import hotpot
    from openbabel import openbabel as ob

    from hotpot.cheminfo.obWrappers.native import _native_module

    native_path = Path(_native_module().__file__).resolve()
    return {
        "hotpot": hotpot.version(),
        "python": sys.version,
        "numpy": np.__version__,
        "openbabel": ob.OBReleaseVersion(),
        "platform": platform.platform(),
        "native_extension": {
            "filename": native_path.name,
            "sha256": _sha256_file(native_path),
        },
    }


def _git_identity(repository_root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _parse_cases(value: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(item) for item in value.split(",") if item))


def _load_smiles(path: Path) -> dict[int, str]:
    return {
        index: line.split(maxsplit=1)[0]
        for index, line in enumerate(
            (
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ),
            start=1,
        )
    }


def _load_complex_indices(path: Path) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    return tuple(sorted(int(case["index"]) for case in cases))


def _verify_source_smiles(source: Path, case_index: int, smiles: str) -> None:
    report_path = source / "cases" / f"{case_index:04d}" / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("smiles") != smiles:
        raise ValueError(
            f"case {case_index:04d} source SMILES differs from the input corpus"
        )


def _trajectory_directory(source: Path, case_index: int, target: str) -> Path:
    directory = source / "cases" / f"{case_index:04d}" / target / "trajectory"
    main = directory / "main"
    return main if (main / "trajectory.json").is_file() else directory


def _load_start_state(source: Path, case_index: int, target: str) -> StartState:
    """Load the unique ``build_complete`` frame from a source trajectory."""
    from hotpot.cheminfo.forcefields import (
        ForceFieldTrajectory,
        TrajectoryEvent,
    )

    trajectory = ForceFieldTrajectory.read(
        _trajectory_directory(source, case_index, target)
    )
    frames = tuple(
        frame
        for frame in trajectory.frames
        if frame.event is TrajectoryEvent.BUILD_COMPLETE
    )
    if len(frames) != 1:
        raise ValueError(
            f"case {case_index:04d} {target} has {len(frames)} "
            "build_complete frames; exactly one is required"
        )
    frame = frames[0]
    atoms = tuple(asdict(atom) for atom in trajectory.atoms)
    bonds = tuple(asdict(bond) for bond in trajectory.topology(frame.index).bonds)
    coordinates = trajectory.coordinates(frame.index)
    fingerprint_payload = json.dumps(
        {"atoms": atoms, "bonds": bonds},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(fingerprint_payload)
    digest.update(np.ascontiguousarray(coordinates, dtype="<f8").tobytes())
    return StartState(atoms, bonds, coordinates, digest.hexdigest())


def _materialize_start(state: StartState):
    """Construct an independent Hotpot molecule from an immutable start."""
    from hotpot import Molecule

    mol = Molecule()
    for atom, coordinate in zip(state.atoms, state.coordinates):
        mol.create_atom(
            atomic_number=int(atom["atomic_number"]),
            formal_charge=int(atom["formal_charge"]),
            id=int(atom["atom_id"]),
            coordinates=coordinate,
        )
    for bond in state.bonds:
        mol.add_bond(
            *tuple(int(index) for index in bond["atom_indices"]),
            float(bond["bond_order"]),
            bond_kind=str(bond["bond_kind"]),
        )
    return mol


def _geometry_failure_names(report: object) -> str:
    return ";".join(str(check.name) for check in report.failures)


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def _base_row(
    case_index: int,
    smiles: str,
    target: str,
    level: str,
    execution_order: int,
    start_sha256: str,
) -> dict[str, object]:
    return {
        column: None
        for column in RESULT_COLUMNS
    } | {
        "case_index": case_index,
        "smiles": smiles,
        "target": target,
        "level": level,
        "level_value": LEVELS.index(level),
        "execution_order": execution_order,
        "status": "running",
        "start_sha256": start_sha256,
    }


def _run_level(
    case_index: int,
    smiles: str,
    target: str,
    state: StartState,
    level_name: str,
    execution_order: int,
    forcefield: str,
    epochs: int,
    steps_per_epoch: int,
) -> dict[str, object]:
    """Run one level while timing optimization only."""
    from hotpot.cheminfo import forcefields as ff
    from hotpot.cheminfo import obWrappers
    from hotpot.cheminfo.obWrappers.native import _native_module

    row = _base_row(
        case_index,
        smiles,
        target,
        level_name,
        execution_order,
        state.sha256,
    )
    mol = _materialize_start(state)
    topology_reference = ff.capture_topology(mol, allow_added_hydrogens=False)
    _native_module()
    wall_started = perf_counter()
    process_started = process_time()
    try:
        report = obWrappers.optimize(
            mol,
            forcefield,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            retain_frames=False,
            retain_epoch_history=False,
            convergence_level=obWrappers.ConvergenceLevel[level_name],
        )
    except Exception as error:  # noqa: BLE001 - benchmark retains all failures.
        process_seconds = process_time() - process_started
        wall_seconds = perf_counter() - wall_started
        row.update(
            status="failed_execution",
            wall_seconds=wall_seconds,
            process_seconds=process_seconds,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        return row
    process_seconds = process_time() - process_started
    wall_seconds = perf_counter() - wall_started

    # Geometry is intentionally evaluated after both optimization clocks stop.
    row.update(
        status="completed",
        final_coordinates_sha256=_coordinates_sha256(mol.coordinates),
        wall_seconds=wall_seconds,
        process_seconds=process_seconds,
        epochs_completed=report.epochs_completed,
        steps_submitted=report.steps_submitted,
        initialization_steps=report.initialization_steps,
        termination_reason=report.termination_reason,
        converged=report.converged,
        terminal_converged=report.terminal_converged,
        final_energy_kj_mol=_finite_or_none(report.final_energy),
        best_energy_kj_mol=_finite_or_none(report.best_energy),
        rms_gradient_kj_mol_angstrom=_finite_or_none(report.rms_gradient),
        max_gradient_kj_mol_angstrom=_finite_or_none(report.max_gradient),
        exploded=report.exploded,
    )
    try:
        geometry_report = ff.evaluate_structure_acceptance(
            mol,
            level="standard",
            topology_reference=topology_reference,
        )
    except Exception as error:  # noqa: BLE001 - retain optimizer evidence.
        row.update(
            status="failed_geometry_evaluation",
            error_type=type(error).__name__,
            error_message=str(error),
        )
    else:
        row.update(
            geometry_passed=geometry_report.passed,
            geometry_failure_names=_geometry_failure_names(geometry_report),
        )
    return row


def _add_strict_pairing(rows: list[dict[str, object]]) -> None:
    strict = next(row for row in rows if row["level"] == "STRICT")
    completed_statuses = {"completed", "failed_geometry_evaluation"}
    if strict["status"] not in completed_statuses:
        return
    strict_wall = float(strict["wall_seconds"])
    strict_process = float(strict["process_seconds"])
    if strict["best_energy_kj_mol"] is None:
        return
    strict_energy = float(strict["best_energy_kj_mol"])
    for row in rows:
        if row["status"] not in completed_statuses:
            continue
        wall = float(row["wall_seconds"])
        process = float(row["process_seconds"])
        if row["best_energy_kj_mol"] is None:
            continue
        energy = float(row["best_energy_kj_mol"])
        wall_savings = strict_wall - wall
        process_savings = strict_process - process
        row.update(
            strict_best_energy_kj_mol=strict_energy,
            best_energy_difference_vs_strict_kj_mol=energy - strict_energy,
            strict_wall_seconds=strict_wall,
            paired_wall_savings_seconds=wall_savings,
            paired_wall_savings_percent=(
                100.0 * wall_savings / strict_wall if strict_wall else None
            ),
            strict_process_seconds=strict_process,
            paired_process_savings_seconds=process_savings,
            paired_process_savings_percent=(
                100.0 * process_savings / strict_process
                if strict_process
                else None
            ),
        )


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_target(
    source_text: str,
    output_text: str,
    case_index: int,
    smiles: str,
    target: str,
    forcefield: str,
    epochs: int,
    steps_per_epoch: int,
    resume: bool,
) -> list[dict[str, object]]:
    """Run a paired four-level target in one spawn-safe worker."""
    source = Path(source_text)
    report_path = (
        Path(output_text) / "cases" / f"{case_index:04d}" / f"{target}.json"
    )
    if resume and report_path.is_file():
        rows = json.loads(report_path.read_text(encoding="utf-8"))["rows"]
        current_state = _load_start_state(source, case_index, target)
        recorded_hashes = {
            str(row["start_sha256"])
            for row in rows
            if row["start_sha256"]
        }
        if recorded_hashes != {current_state.sha256}:
            raise ValueError(
                f"case {case_index:04d} {target} source changed since its "
                "saved benchmark result"
            )
        return rows
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _verify_source_smiles(source, case_index, smiles)
        state = _load_start_state(source, case_index, target)
    except Exception as error:  # noqa: BLE001 - benchmark retains all failures.
        rows = []
        for order, level in enumerate(LEVELS):
            row = _base_row(case_index, smiles, target, level, order, "")
            row.update(
                status="failed_source",
                error_type=type(error).__name__,
                error_message=str(error),
            )
            rows.append(row)
    else:
        if not np.all(np.isfinite(state.coordinates)):
            rows = []
            for order, level in enumerate(LEVELS):
                row = _base_row(
                    case_index,
                    smiles,
                    target,
                    level,
                    order,
                    state.sha256,
                )
                row.update(
                    status="nonfinite_start",
                    error_type="NonFiniteStartCoordinates",
                    error_message=(
                        "build_complete coordinates contain NaN or infinity"
                    ),
                )
                rows.append(row)
        else:
            offset = (case_index + TARGETS.index(target)) % len(LEVELS)
            execution_levels = LEVELS[offset:] + LEVELS[:offset]
            rows = [
                _run_level(
                    case_index,
                    smiles,
                    target,
                    state,
                    level,
                    order,
                    forcefield,
                    epochs,
                    steps_per_epoch,
                )
                for order, level in enumerate(execution_levels)
            ]
            _add_strict_pairing(rows)

    _write_json(
        report_path,
        {
            "schema_version": SCHEMA_VERSION,
            "case_index": case_index,
            "smiles": smiles,
            "target": target,
            "rows": rows,
        },
    )
    return rows


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _level_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    started = [row for row in rows if row["wall_seconds"] is not None]
    completed = [
        row
        for row in rows
        if row["status"] in {"completed", "failed_geometry_evaluation"}
    ]
    walls = [float(row["wall_seconds"]) for row in started]
    processes = [float(row["process_seconds"]) for row in started]
    completed_walls = [float(row["wall_seconds"]) for row in completed]
    completed_processes = [float(row["process_seconds"]) for row in completed]
    savings = [
        float(row["paired_wall_savings_seconds"])
        for row in completed
        if row["paired_wall_savings_seconds"] is not None
    ]
    savings_percent = [
        float(row["paired_wall_savings_percent"])
        for row in completed
        if row["paired_wall_savings_percent"] is not None
    ]
    process_savings = [
        float(row["paired_process_savings_seconds"])
        for row in completed
        if row["paired_process_savings_seconds"] is not None
    ]
    strict_walls = [
        float(row["strict_wall_seconds"])
        for row in completed
        if row["strict_wall_seconds"] is not None
    ]
    strict_processes = [
        float(row["strict_process_seconds"])
        for row in completed
        if row["strict_process_seconds"] is not None
    ]
    absolute_energy_differences = [
        abs(float(row["best_energy_difference_vs_strict_kj_mol"]))
        for row in completed
        if row["best_energy_difference_vs_strict_kj_mol"] is not None
    ]
    return {
        "denominator": len(rows),
        "started_count": len(started),
        "completed_count": len(completed),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "converged_count": sum(bool(row["converged"]) for row in completed),
        "geometry_pass_count": sum(
            bool(row["geometry_passed"]) for row in completed
        ),
        "geometry_pass_rate_over_denominator": (
            sum(bool(row["geometry_passed"]) for row in completed) / len(rows)
            if rows
            else None
        ),
        "aggregate_wall_seconds": sum(walls),
        "median_wall_seconds": _median(walls),
        "aggregate_process_seconds": sum(processes),
        "median_process_seconds": _median(processes),
        "completed_aggregate_wall_seconds": sum(completed_walls),
        "completed_median_wall_seconds": _median(completed_walls),
        "completed_aggregate_process_seconds": sum(completed_processes),
        "completed_median_process_seconds": _median(completed_processes),
        "paired_with_strict_count": len(savings),
        "aggregate_wall_savings_vs_strict_seconds": sum(savings),
        "aggregate_wall_savings_vs_strict_percent": (
            100.0 * sum(savings) / sum(strict_walls)
            if strict_walls and sum(strict_walls)
            else None
        ),
        "median_wall_savings_vs_strict_seconds": _median(savings),
        "median_wall_savings_vs_strict_percent": _median(savings_percent),
        "aggregate_process_savings_vs_strict_seconds": sum(process_savings),
        "aggregate_process_savings_vs_strict_percent": (
            100.0 * sum(process_savings) / sum(strict_processes)
            if strict_processes and sum(strict_processes)
            else None
        ),
        "median_absolute_best_energy_difference_vs_strict_kj_mol": _median(
            absolute_energy_differences
        ),
        "maximum_absolute_best_energy_difference_vs_strict_kj_mol": (
            max(absolute_energy_differences)
            if absolute_energy_differences
            else None
        ),
    }


def _quality_pair_summary(
    rows: Sequence[Mapping[str, object]],
    level: str,
) -> dict[str, int]:
    by_case = {
        (int(row["case_index"]), str(row["target"]), str(row["level"])): row
        for row in rows
    }
    counts = Counter()
    for row in rows:
        if row["level"] != level:
            continue
        strict = by_case.get(
            (int(row["case_index"]), str(row["target"]), "STRICT")
        )
        current_passed = row["geometry_passed"]
        strict_passed = None if strict is None else strict["geometry_passed"]
        if current_passed is None or strict_passed is None:
            counts["unavailable"] += 1
        elif bool(strict_passed) and not bool(current_passed):
            counts["regressed_vs_strict"] += 1
        elif not bool(strict_passed) and bool(current_passed):
            counts["improved_vs_strict"] += 1
        elif bool(current_passed):
            counts["both_passed"] += 1
        else:
            counts["both_failed"] += 1
    return {
        name: counts[name]
        for name in (
            "both_passed",
            "both_failed",
            "regressed_vs_strict",
            "improved_vs_strict",
            "unavailable",
        )
    }


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate denominators, quality, timing, and paired STRICT savings."""
    targets: dict[str, object] = {}
    for target in TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        if not target_rows:
            continue
        targets[target] = {}
        for level in LEVELS:
            level_summary = _level_summary(
                [row for row in target_rows if row["level"] == level]
            )
            level_summary["quality_vs_strict"] = _quality_pair_summary(
                target_rows, level
            )
            targets[target][level] = level_summary
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "target_count": len({(row["case_index"], row["target"]) for row in rows}),
        "targets": targets,
        "levels": {
            level: _level_summary([row for row in rows if row["level"] == level])
            for level in LEVELS
        },
    }


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_results_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write stable, machine-readable per-case results."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                int(item["case_index"]),
                TARGETS.index(str(item["target"])),
                int(item["level_value"]),
            ),
        ):
            writer.writerow(
                {column: _csv_value(row.get(column)) for column in RESULT_COLUMNS}
            )
    temporary.replace(path)


def _format_number(value: object, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _format_percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.1f}%"


def write_report(path: Path, summary: Mapping[str, object]) -> None:
    """Write the human-readable paired benchmark report."""
    lines = [
        "# obWrappers convergence-level benchmark",
        "",
        "All four policies start from the exact same `build_complete` frame for "
        "each case and target. Optimization clocks exclude source I/O, geometry "
        "validation, and artifact writing.",
        "",
    ]
    for target, level_summaries in summary["targets"].items():
        lines.extend(
            (
                f"## {str(target).title()}",
                "",
                "| Level | Denominator | Completed | Converged | Geometry pass | "
                "Median wall (s) | Aggregate wall (s) | Median saving vs STRICT | "
                "Median absolute selected-energy delta (kJ/mol) | "
                "Geometry regressions vs STRICT |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for level in LEVELS:
            values = level_summaries[level]
            lines.append(
                "| "
                + " | ".join(
                    (
                        level,
                        str(values["denominator"]),
                        str(values["completed_count"]),
                        str(values["converged_count"]),
                        str(values["geometry_pass_count"]),
                        _format_number(values["median_wall_seconds"]),
                        _format_number(values["aggregate_wall_seconds"]),
                        _format_percent(
                            values["median_wall_savings_vs_strict_percent"]
                        ),
                        _format_number(
                            values[
                                "median_absolute_best_energy_difference_vs_"
                                "strict_kj_mol"
                            ]
                        ),
                        str(
                            values["quality_vs_strict"][
                                "regressed_vs_strict"
                            ]
                        ),
                    )
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest_payload(configuration: BenchmarkConfiguration) -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[3]
    source_manifest = configuration.source / "manifest.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "obwrappers-convergence-levels-paired",
        "git": _git_identity(repository_root),
        "implementation_sha256": _implementation_sha256(repository_root),
        "runtime": _runtime_identity(),
        "source": {
            "path": _manifest_path(configuration.source, repository_root),
            "manifest_sha256": (
                _sha256_file(source_manifest) if source_manifest.is_file() else None
            ),
            "frame": "unique build_complete",
        },
        "cohort": {
            "path": _manifest_path(configuration.cohort, repository_root),
            "sha256": _sha256_file(configuration.cohort),
        },
        "input": {
            "path": _manifest_path(configuration.input_path, repository_root),
            "sha256": _sha256_file(configuration.input_path),
        },
        "settings": {
            "forcefield": configuration.forcefield,
            "epochs": configuration.epochs,
            "steps_per_epoch": configuration.steps_per_epoch,
            "levels": list(LEVELS),
            "level_policies": LEVEL_POLICIES,
            "targets": list(configuration.targets),
            "cases": list(configuration.cases) if configuration.cases else None,
            "workers": configuration.workers,
            "retain_frames": False,
            "retain_epoch_history": False,
            "geometry_gate": "standard",
            "timing_scope": (
                "complete public obWrappers.optimize call, including Python/C++ "
                "packing and result materialization; excludes native-module "
                "loading, source I/O, geometry validation, and artifact writing"
            ),
        },
    }


def _write_or_check_manifest(configuration: BenchmarkConfiguration) -> None:
    path = configuration.output / "manifest.json"
    payload = _manifest_payload(configuration)
    if path.is_file():
        if not configuration.resume:
            raise FileExistsError(
                f"{path} exists; pass --resume or choose another output directory"
            )
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("--resume configuration differs from manifest.json")
        return
    _write_json(path, payload)


def _task_specs(
    configuration: BenchmarkConfiguration,
) -> tuple[tuple[object, ...], ...]:
    smiles_by_index = _load_smiles(configuration.input_path)
    complex_indices = set(_load_complex_indices(configuration.cohort))
    selected_indices = (
        tuple(smiles_by_index)
        if configuration.cases is None
        else configuration.cases
    )
    unknown = tuple(index for index in selected_indices if index not in smiles_by_index)
    if unknown:
        raise ValueError(f"case indices absent from input: {unknown}")
    tasks = []
    for index in selected_indices:
        for target in configuration.targets:
            if target == "complex" and index not in complex_indices:
                continue
            tasks.append(
                (
                    str(configuration.source),
                    str(configuration.output),
                    index,
                    smiles_by_index[index],
                    target,
                    configuration.forcefield,
                    configuration.epochs,
                    configuration.steps_per_epoch,
                    configuration.resume,
                )
            )
    return tuple(tasks)


def run_benchmark(configuration: BenchmarkConfiguration) -> dict[str, object]:
    """Run and persist the complete paired convergence benchmark."""
    configuration.output.mkdir(parents=True, exist_ok=True)
    _write_or_check_manifest(configuration)
    tasks = _task_specs(configuration)
    rows: list[dict[str, object]] = []
    wall_started = perf_counter()
    if configuration.workers == 1:
        completions: Iterable[list[dict[str, object]]] = (
            run_target(*task) for task in tasks
        )
        for completed, target_rows in enumerate(completions, start=1):
            rows.extend(target_rows)
            print(
                f"[{completed}/{len(tasks)}] case={target_rows[0]['case_index']:04d} "
                f"target={target_rows[0]['target']}",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=configuration.workers,
            mp_context=context,
        ) as pool:
            futures = {pool.submit(run_target, *task): task[2:5] for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                target_rows = future.result()
                rows.extend(target_rows)
                print(
                    f"[{completed}/{len(tasks)}] "
                    f"case={target_rows[0]['case_index']:04d} "
                    f"target={target_rows[0]['target']}",
                    flush=True,
                )
    wall_seconds = perf_counter() - wall_started
    summary = summarize(rows)
    summary["benchmark_wall_seconds"] = wall_seconds
    summary["manifest_sha256"] = _sha256_file(configuration.output / "manifest.json")
    write_results_csv(configuration.output / "results.csv", rows)
    _write_json(configuration.output / "summary.json", summary)
    write_report(configuration.output / "report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Return the standalone benchmark CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare all four obWrappers convergence levels from paired "
            "build-complete ligand and complex structures."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--forcefield", default=DEFAULT_FORCEFIELD)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=DEFAULT_STEPS_PER_EPOCH,
    )
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    parser.add_argument("--cases", type=_parse_cases)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    """Parse command-line settings and execute the benchmark."""
    arguments = build_parser().parse_args(argv)
    configuration = BenchmarkConfiguration(
        source=arguments.source.resolve(),
        cohort=arguments.cohort.resolve(),
        input_path=arguments.input_path.resolve(),
        output=arguments.output.resolve(),
        workers=arguments.workers,
        forcefield=arguments.forcefield,
        epochs=arguments.epochs,
        steps_per_epoch=arguments.steps_per_epoch,
        targets=tuple(arguments.targets),
        cases=arguments.cases,
        resume=arguments.resume,
    )
    return run_benchmark(configuration)


__all__ = (
    "BenchmarkConfiguration",
    "LEVELS",
    "LEVEL_POLICIES",
    "RESULT_COLUMNS",
    "build_parser",
    "main",
    "run_benchmark",
    "run_target",
    "summarize",
    "write_report",
    "write_results_csv",
)
