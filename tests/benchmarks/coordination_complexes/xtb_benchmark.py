"""Opt-in xTB refinement benchmark for the coordination corpus.

This benchmark consumes structures produced by an existing force-field
workflow.  It deliberately does not repeat CBond inference or UFF building.
Each selected structure is refined through two independent routes that start
from identical coordinates:

``direct-gfn2``
    UFF -> GFN2-xTB

``gfnff-gfn2``
    UFF -> GFN-FF -> GFN2-xTB
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import statistics
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Optional, TYPE_CHECKING

import numpy as np

from hotpot.cheminfo.calculator.electronic_state import (
    ChargeInferenceResult,
    ElectronicState,
)
from hotpot.cheminfo.calculator.electronic_state.resolver import (
    resolve_electronic_state,
)
from hotpot.cheminfo.calculator.formal_charges import infer_charge
from hotpot.plugins._harness import ProcessTimeoutError
from hotpot.plugins.xtb import (
    GFNXTBMethod,
    XTBApplicabilityError,
    XTBError,
    XTBExecutionError,
    XTBMethod,
    XTBResultError,
    XTBRunReport,
    XTBTask,
    probe_xtb_backend,
    run_gfn_xtb,
    run_gfnff,
)

from .backend_comparison import CanonicalCase, _rebuild_complex
from .cohort import BenchmarkCohort, LigandCase, resolve_cohort
from .io import json_value, sha256_file, write_json

if TYPE_CHECKING:
    from hotpot.cheminfo.core import Molecule
    from hotpot.plugins.xtb import XTBBackendInfo


__all__ = (
    "BenchmarkTarget",
    "XTBBenchmarkConfig",
    "XTBRoute",
    "build_parser",
    "main",
    "parse_case_selection",
    "run_benchmark",
    "run_case",
)


SCHEMA_VERSION = 1
WORKFLOW_NAME = "xtb_coordination_refinement"


# Public benchmark contracts.


class BenchmarkTarget(str, Enum):
    """Prepared molecular targets consumed from the force-field benchmark."""

    LIGAND = "ligand"
    COMPLEX = "complex"


class XTBRoute(str, Enum):
    """Independent xTB refinement routes compared from one UFF structure."""

    DIRECT_GFN2 = "direct-gfn2"
    GFNFF_GFN2 = "gfnff-gfn2"


@dataclass(frozen=True)
class XTBBenchmarkConfig:
    """Immutable scientific and execution settings recorded in the manifest."""

    routes: tuple[XTBRoute, ...] = (
        XTBRoute.DIRECT_GFN2,
        XTBRoute.GFNFF_GFN2,
    )
    targets: tuple[BenchmarkTarget, ...] = (
        BenchmarkTarget.LIGAND,
        BenchmarkTarget.COMPLEX,
    )
    case_indices: tuple[int, ...] = ()
    workers: int = 16
    threads: int = 4
    timeout_seconds: float = 1000.0
    quality_level: str = "standard"
    executable: Optional[Path] = None


class _RecordedStageFailure(RuntimeError):
    """A stage failure whose diagnostics have already been persisted."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = dict(payload)
        super().__init__(str(payload["error_message"]))


# Input and selection helpers.


def parse_case_selection(value: str) -> tuple[int, ...]:
    """Parse one-based comma-separated indices and inclusive ranges."""

    selected: set[int] = set()
    for token in value.split(","):
        item = token.strip()
        if not item:
            continue
        if "-" in item:
            lower_text, upper_text = item.split("-", maxsplit=1)
            lower = int(lower_text)
            upper = int(upper_text)
            if upper < lower:
                raise argparse.ArgumentTypeError(
                    f"case range {item!r} is descending"
                )
            selected.update(range(lower, upper + 1))
        else:
            selected.add(int(item))
    if not selected or min(selected) < 1:
        raise argparse.ArgumentTypeError(
            "case selection must contain positive one-based indices"
        )
    return tuple(sorted(selected))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _route_selection(value: str) -> tuple[XTBRoute, ...]:
    if value == "both":
        return (XTBRoute.DIRECT_GFN2, XTBRoute.GFNFF_GFN2)
    return (XTBRoute(value),)


def _target_selection(value: str) -> tuple[BenchmarkTarget, ...]:
    if value == "both":
        return (BenchmarkTarget.LIGAND, BenchmarkTarget.COMPLEX)
    return (BenchmarkTarget(value),)


def _thread_environment(threads: int) -> dict[str, str]:
    thread_count = str(threads)
    return {
        **os.environ,
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
    }


def _read_single_molecule(path: Path) -> "Molecule":
    from hotpot import read_mol
    from hotpot.cheminfo.core import Molecule

    loaded = read_mol(path)
    if isinstance(loaded, Molecule):
        return loaded
    molecules = tuple(loaded)
    if len(molecules) != 1:
        raise ValueError(f"{path} must contain exactly one molecule")
    return molecules[0]


def _source_structure_path(
    source_root: Path,
    case_index: int,
    target: BenchmarkTarget,
) -> Path:
    return (
        source_root
        / "cases"
        / f"{case_index:04d}"
        / target.value
        / "optimized.sdf"
    )


def _load_source_mol(
    path: Path,
    target: BenchmarkTarget,
    complex_case: Optional[CanonicalCase],
) -> "Molecule":
    source_mol = _read_single_molecule(path)
    if target is BenchmarkTarget.LIGAND:
        return source_mol
    if complex_case is None:
        raise ValueError("an eligible complex target requires a canonical case")

    canonical_mol = _rebuild_complex(complex_case)
    source_atomic_numbers = tuple(atom.atomic_number for atom in source_mol.atoms)
    canonical_atomic_numbers = tuple(
        atom.atomic_number for atom in canonical_mol.atoms
    )
    if source_atomic_numbers != canonical_atomic_numbers:
        raise ValueError(
            "source complex and canonical cohort have different atom order"
        )
    canonical_mol.coordinates = np.asarray(source_mol.coordinates, dtype=float)
    return canonical_mol


def _quality_payload(report: object) -> dict[str, object]:
    return {
        "level": str(getattr(report, "level")),
        "passed": bool(getattr(report, "passed")),
        "checks": json_value(getattr(report, "checks")),
        "failures": json_value(getattr(report, "failures")),
        "warnings": json_value(getattr(report, "warnings")),
        "metrics": json_value(getattr(report, "metrics")),
    }


def _evaluate_quality(
    mol: "Molecule",
    quality_level: str,
    topology_reference: object,
) -> Mapping[str, object]:
    from hotpot.cheminfo import forcefields as ff

    return _quality_payload(
        ff.evaluate_structure_acceptance(
            mol,
            level=quality_level,
            topology_reference=topology_reference,
        )
    )


# xTB stage execution and evidence persistence.


def _execute_xtb_stage(
    mol: "Molecule",
    method: XTBMethod,
    charge_state: ChargeInferenceResult,
    electronic_state: ElectronicState,
    config: XTBBenchmarkConfig,
    work_directory: Path,
) -> XTBRunReport:
    common_options = {
        "task": XTBTask.OPTIMIZE,
        "executable": config.executable,
        "environment": _thread_environment(config.threads),
        "timeout_seconds": config.timeout_seconds,
        "work_directory": work_directory,
        "keep_work_directory": True,
    }
    if method is XTBMethod.GFNFF:
        return run_gfnff(
            mol,
            charge_state=charge_state,
            **common_options,
        )
    return run_gfn_xtb(
        mol,
        method=GFNXTBMethod.GFN2_XTB,
        state=electronic_state,
        **common_options,
    )


def _artifact_payload(
    report: XTBRunReport,
    stage_directory: Path,
) -> tuple[dict[str, object], ...]:
    artifacts = []
    stage_root = stage_directory.resolve()
    for artifact in report.artifacts.values():
        artifact_path = artifact.path.resolve()
        try:
            relative_path = artifact_path.relative_to(stage_root).as_posix()
        except ValueError:
            relative_path = str(artifact_path)
        artifacts.append(
            {
                "name": artifact.name,
                "path": relative_path,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
        )
    return tuple(artifacts)


def _run_report_payload(
    report: XTBRunReport,
    stage_directory: Path,
) -> dict[str, object]:
    return {
        "backend": {
            "executable": str(report.backend_info.executable),
            "version": report.backend_info.version,
            "revision": report.backend_info.revision,
            "executable_sha256": report.backend_info.executable_sha256,
        },
        "requested_method": report.requested_method.value,
        "effective_method": report.effective_method.value,
        "task": report.task.value,
        "charge": report.charge,
        "unpaired_electrons": report.unpaired_electrons,
        "charge_source": (
            None if report.charge_source is None else report.charge_source.value
        ),
        "spin_source": (
            None if report.spin_source is None else report.spin_source.value
        ),
        "state_assumptions": report.state_assumptions,
        "fragment_charges": report.fragment_charges,
        "argv": report.argv,
        "return_code": report.return_code,
        "native_elapsed_seconds": report.elapsed_seconds,
        "process_succeeded": report.process_succeeded,
        "converged": report.converged,
        "energy_hartree": report.energy_hartree,
        "gradient_norm": report.gradient_norm,
        "atom_order_verified": report.atom_order_verified,
        "coordinates_committed": report.coordinates_committed,
        "workspace_retained": report.workspace_retained,
        "artifacts": _artifact_payload(report, stage_directory),
    }


def _write_native_log(path: Path, report: XTBRunReport) -> None:
    path.write_text(
        "=== stdout ===\n"
        + report.stdout.rstrip()
        + "\n=== stderr ===\n"
        + report.stderr.rstrip()
        + "\n",
        encoding="utf-8",
    )


def _failure_payload(
    method: XTBMethod,
    error: BaseException,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "status": "failed_execution",
        "method": method.value,
        "elapsed_seconds": elapsed_seconds,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "applicability_rejected": isinstance(error, XTBApplicabilityError),
    }


def _run_stage(
    mol: "Molecule",
    method: XTBMethod,
    charge_state: ChargeInferenceResult,
    electronic_state: ElectronicState,
    config: XTBBenchmarkConfig,
    stage_directory: Path,
) -> dict[str, object]:
    stage_directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    try:
        report = _execute_xtb_stage(
            mol,
            method,
            charge_state,
            electronic_state,
            config,
            stage_directory / "native",
        )
    except (XTBExecutionError, XTBResultError) as error:
        elapsed_seconds = perf_counter() - started
        _write_native_log(stage_directory / "native.log", error.report)
        payload = {
            **_failure_payload(method, error, elapsed_seconds),
            "report": _run_report_payload(error.report, stage_directory),
        }
        write_json(stage_directory / "report.json", payload)
        raise _RecordedStageFailure(payload) from error
    except (XTBError, ProcessTimeoutError) as error:
        payload = _failure_payload(
            method,
            error,
            perf_counter() - started,
        )
        write_json(stage_directory / "report.json", payload)
        raise _RecordedStageFailure(payload) from error

    elapsed_seconds = perf_counter() - started
    mol.write(
        stage_directory / "structure.sdf",
        overwrite=True,
        write_single=True,
    )
    payload = {
        "status": "succeeded",
        "method": method.value,
        "elapsed_seconds": elapsed_seconds,
        "report": _run_report_payload(report, stage_directory),
    }
    _write_native_log(stage_directory / "native.log", report)
    write_json(stage_directory / "report.json", payload)
    return payload


def _route_methods(route: XTBRoute) -> tuple[XTBMethod, ...]:
    if route is XTBRoute.DIRECT_GFN2:
        return (XTBMethod.GFN2_XTB,)
    return (XTBMethod.GFNFF, XTBMethod.GFN2_XTB)


def _run_route(
    source_mol: "Molecule",
    route: XTBRoute,
    charge_state: ChargeInferenceResult,
    electronic_state: ElectronicState,
    config: XTBBenchmarkConfig,
    route_directory: Path,
) -> dict[str, object]:
    from hotpot.cheminfo import forcefields as ff

    route_directory.mkdir(parents=True, exist_ok=True)
    working_mol = source_mol.copy()
    topology_reference = ff.capture_topology(
        working_mol,
        allow_added_hydrogens=False,
    )
    stages = []
    started = perf_counter()
    for stage_index, method in enumerate(_route_methods(route)):
        stage_directory = route_directory / f"{stage_index:02d}-{method.value}"
        try:
            stages.append(
                _run_stage(
                    working_mol,
                    method,
                    charge_state,
                    electronic_state,
                    config,
                    stage_directory,
                )
            )
        except _RecordedStageFailure as error:
            stages.append(error.payload)
            if np.all(np.isfinite(working_mol.coordinates)):
                working_mol.write(
                    route_directory / "last_finite.sdf",
                    overwrite=True,
                    write_single=True,
                )
            payload = {
                "route": route.value,
                "status": "failed_execution",
                "total_seconds": perf_counter() - started,
                "stages": stages,
                "quality": None,
                "error_type": error.payload["error_type"],
                "error_message": error.payload["error_message"],
                "applicability_rejected": error.payload[
                    "applicability_rejected"
                ],
            }
            write_json(route_directory / "report.json", payload)
            return payload

    quality = dict(
        _evaluate_quality(
            working_mol,
            config.quality_level,
            topology_reference,
        )
    )
    working_mol.write(
        route_directory / "final.sdf",
        overwrite=True,
        write_single=True,
    )
    payload = {
        "route": route.value,
        "status": "passed" if quality["passed"] else "failed_quality",
        "total_seconds": perf_counter() - started,
        "stages": stages,
        "quality": quality,
        "error_type": None,
        "error_message": None,
        "applicability_rejected": False,
    }
    write_json(route_directory / "report.json", payload)
    return payload


# Per-case and aggregate orchestration.


def _target_record(
    ligand_case: LigandCase,
    complex_case: Optional[CanonicalCase],
    target: BenchmarkTarget,
    source_root: Path,
    case_directory: Path,
    config: XTBBenchmarkConfig,
) -> dict[str, object]:
    target_directory = case_directory / target.value
    if target not in config.targets:
        return {"status": "not_selected", "routes": {}}
    if target is BenchmarkTarget.COMPLEX and complex_case is None:
        return {"status": "not_eligible", "routes": {}}

    target_directory.mkdir(parents=True, exist_ok=True)
    source_path = _source_structure_path(
        source_root,
        ligand_case.index,
        target,
    )
    if not source_path.is_file():
        payload = {
            "status": "source_unavailable",
            "source": str(source_path),
            "routes": {},
            "error_type": "FileNotFoundError",
            "error_message": f"prepared source structure is absent: {source_path}",
        }
        write_json(target_directory / "report.json", payload)
        return payload

    try:
        source_mol = _load_source_mol(source_path, target, complex_case)
        source_mol.write(
            target_directory / "input.sdf",
            overwrite=True,
            write_single=True,
        )
        charge_state = infer_charge(source_mol)
        electronic_state = resolve_electronic_state(source_mol)
    except (OSError, ValueError) as error:
        payload = {
            "status": "source_invalid",
            "source": str(source_path),
            "routes": {},
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        write_json(target_directory / "report.json", payload)
        return payload

    routes = {
        route.value: _run_route(
            source_mol,
            route,
            charge_state,
            electronic_state,
            config,
            target_directory / route.value,
        )
        for route in config.routes
    }
    payload = {
        "status": "completed",
        "source": str(source_path),
        "atom_count": len(source_mol.atoms),
        "charge": electronic_state.charge,
        "unpaired_electrons": electronic_state.unpaired_electrons,
        "state_assumptions": electronic_state.assumptions,
        "routes": routes,
    }
    write_json(target_directory / "report.json", payload)
    return payload


def run_case(
    ligand_case_payload: Mapping[str, object],
    complex_case_payload: Optional[Mapping[str, object]],
    source_root_text: str,
    output_root_text: str,
    config: XTBBenchmarkConfig,
    resume: bool,
) -> dict[str, object]:
    """Run both requested targets and routes for one corpus case."""

    ligand_case = LigandCase(
        index=int(ligand_case_payload["index"]),
        smiles=str(ligand_case_payload["smiles"]),
        seed=int(ligand_case_payload["seed"]),
    )
    complex_case = (
        None
        if complex_case_payload is None
        else CanonicalCase.from_dict(complex_case_payload)
    )
    source_root = Path(source_root_text)
    output_root = Path(output_root_text)
    case_directory = output_root / "cases" / f"{ligand_case.index:04d}"
    report_path = case_directory / "report.json"
    if resume and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    if case_directory.exists():
        shutil.rmtree(case_directory)
    case_directory.mkdir(parents=True)

    started = perf_counter()
    try:
        targets = {
            target.value: _target_record(
                ligand_case,
                complex_case,
                target,
                source_root,
                case_directory,
                config,
            )
            for target in BenchmarkTarget
        }
        record = {
            "schema_version": SCHEMA_VERSION,
            "index": ligand_case.index,
            "smiles": ligand_case.smiles,
            "workflow": WORKFLOW_NAME,
            "targets": targets,
            "total_seconds": perf_counter() - started,
        }
    except Exception as error:  # Benchmark workers persist unknown failures.
        record = {
            "schema_version": SCHEMA_VERSION,
            "index": ligand_case.index,
            "smiles": ligand_case.smiles,
            "workflow": WORKFLOW_NAME,
            "targets": {},
            "total_seconds": perf_counter() - started,
            "worker_error_type": type(error).__name__,
            "worker_error_message": str(error),
            "traceback": traceback.format_exc(),
        }
    write_json(report_path, record)
    return record


def _source_manifest(source_root: Path) -> tuple[Path, Mapping[str, object]]:
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"force-field source root lacks manifest.json: {source_root}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source manifest must contain one JSON object")
    return manifest_path, payload


def _validate_source_identity(
    source_manifest: Mapping[str, object],
    cohort: BenchmarkCohort,
) -> None:
    input_identity = source_manifest.get("input")
    cohort_identity = source_manifest.get("cohort")
    if not isinstance(input_identity, Mapping) or not isinstance(
        cohort_identity,
        Mapping,
    ):
        raise ValueError(
            "xTB source must be an independent workflow benchmark root with "
            "input and cohort identities"
        )
    if input_identity.get("sha256") != cohort.input_sha256:
        raise ValueError("source workflow and ligand corpus SHA-256 differ")
    if cohort_identity.get("sha256") != cohort.cohort_sha256:
        raise ValueError("source workflow and canonical cohort SHA-256 differ")


def _validate_eu_cohort(
    cohort: BenchmarkCohort,
    targets: Sequence[BenchmarkTarget],
) -> None:
    if BenchmarkTarget.COMPLEX not in targets:
        return
    metals = {case.metal for case in cohort.complex_cases.values()}
    if metals != {"Eu"}:
        raise ValueError(
            "stable xTB 6.7.1 cannot run the existing Am cohort; the complex "
            "benchmark requires a separately generated Eu canonical cohort"
        )


def _backend_payload(backend_info: "XTBBackendInfo") -> dict[str, object]:
    return {
        "executable": str(backend_info.executable),
        "version": backend_info.version,
        "revision": backend_info.revision,
        "executable_sha256": backend_info.executable_sha256,
        "gfn_xtb_max_atomic_number": backend_info.gfn_xtb_max_atomic_number,
        "gfnff_max_atomic_number": backend_info.gfnff_max_atomic_number,
    }


def _manifest_payload(
    cohort: BenchmarkCohort,
    source_root: Path,
    source_manifest_path: Path,
    config: XTBBenchmarkConfig,
    backend_info: "XTBBackendInfo",
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(cohort.input_path),
            "sha256": cohort.input_sha256,
            "sample_count": len(cohort.ligand_cases),
        },
        "cohort": {
            "path": str(cohort.cohort_path),
            "sha256": cohort.cohort_sha256,
            "sample_count": len(cohort.complex_cases),
            "indices": list(cohort.complex_indices),
        },
        "source": {
            "root": str(source_root),
            "manifest_sha256": sha256_file(source_manifest_path),
            "contract": "cases/NNNN/{ligand,complex}/optimized.sdf",
        },
        "backend": _backend_payload(backend_info),
        "settings": {
            "routes": [route.value for route in config.routes],
            "targets": [target.value for target in config.targets],
            "case_indices": list(config.case_indices),
            "workers": config.workers,
            "threads_per_worker": config.threads,
            "timeout_seconds": config.timeout_seconds,
            "quality_level": config.quality_level,
            "task": XTBTask.OPTIMIZE.value,
        },
    }


def _manifest_comparison(payload: Mapping[str, object]) -> dict[str, object]:
    comparable = dict(payload)
    comparable.pop("created_at", None)
    return comparable


def _write_or_check_manifest(
    output_root: Path,
    payload: Mapping[str, object],
    *,
    resume: bool,
) -> None:
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(
                f"{manifest_path} exists; use --resume or choose another output"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_comparison(existing) != _manifest_comparison(payload):
            raise ValueError("--resume configuration differs from manifest.json")
        return
    write_json(manifest_path, payload)


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(values: Sequence[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "median_seconds": statistics.median(values) if values else None,
        "p90_seconds": _percentile(values, 0.90),
        "p95_seconds": _percentile(values, 0.95),
        "maximum_seconds": max(values) if values else None,
        "total_seconds": sum(values),
    }


def _route_summary(
    records: Sequence[Mapping[str, object]],
    target: BenchmarkTarget,
    route: XTBRoute,
) -> dict[str, object]:
    target_records = [
        record.get("targets", {}).get(target.value, {})
        for record in records
    ]
    eligible_records = [
        target_record
        for target_record in target_records
        if target_record.get("status") not in {"not_eligible", "not_selected"}
    ]
    route_records = [
        target_record.get("routes", {}).get(route.value)
        for target_record in eligible_records
    ]
    completed_routes = [
        route_record
        for route_record in route_records
        if isinstance(route_record, Mapping)
    ]
    statuses = Counter(
        str(route_record["status"]) for route_record in completed_routes
    )
    route_times = [
        float(route_record["total_seconds"])
        for route_record in completed_routes
        if route_record.get("total_seconds") is not None
    ]
    stage_times: dict[str, list[float]] = {}
    for route_record in completed_routes:
        for stage in route_record.get("stages", ()):
            method = str(stage["method"])
            stage_times.setdefault(method, []).append(float(stage["elapsed_seconds"]))
    return {
        "denominator": len(eligible_records),
        "source_available_count": len(completed_routes),
        "status_counts": dict(statuses),
        "quality_pass_count": statuses["passed"],
        "quality_pass_rate": (
            statuses["passed"] / len(eligible_records)
            if eligible_records
            else None
        ),
        "applicability_rejected_count": sum(
            bool(route_record.get("applicability_rejected"))
            for route_record in completed_routes
        ),
        "timing": _timing_summary(route_times),
        "stage_timing": {
            method: _timing_summary(values)
            for method, values in sorted(stage_times.items())
        },
    }


def _write_summary(
    output_root: Path,
    records: Sequence[Mapping[str, object]],
    config: XTBBenchmarkConfig,
    wall_seconds: float,
) -> dict[str, object]:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW_NAME,
        "manifest_sha256": sha256_file(output_root / "manifest.json"),
        "sample_count": len(records),
        "wall_seconds": wall_seconds,
        "results": {
            target.value: {
                route.value: _route_summary(records, target, route)
                for route in config.routes
            }
            for target in config.targets
        },
    }
    write_json(output_root / "summary.json", summary)
    return summary


def _probe_backend(config: XTBBenchmarkConfig) -> "XTBBackendInfo":
    return probe_xtb_backend(
        config.executable,
        environment=_thread_environment(config.threads),
        timeout_seconds=config.timeout_seconds,
    )


def run_benchmark(
    cohort: BenchmarkCohort,
    source_root: Path,
    output_root: Path,
    config: XTBBenchmarkConfig,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Run the selected xTB routes over prepared ligand and Eu structures."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    _validate_eu_cohort(cohort, config.targets)
    source_manifest_path, source_manifest = _source_manifest(source_root)
    _validate_source_identity(source_manifest, cohort)
    backend_info = _probe_backend(config)
    resolved_config = XTBBenchmarkConfig(
        routes=config.routes,
        targets=config.targets,
        case_indices=config.case_indices,
        workers=config.workers,
        threads=config.threads,
        timeout_seconds=config.timeout_seconds,
        quality_level=config.quality_level,
        executable=backend_info.executable,
    )

    available_indices = {case.index for case in cohort.ligand_cases}
    selected_indices = (
        set(resolved_config.case_indices)
        if resolved_config.case_indices
        else available_indices
    )
    missing_indices = sorted(selected_indices - available_indices)
    if missing_indices:
        raise ValueError(f"corpus indices do not exist: {missing_indices}")
    ligand_cases = tuple(
        case for case in cohort.ligand_cases if case.index in selected_indices
    )
    if not ligand_cases:
        raise ValueError("benchmark selection contains no cases")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_payload(
        cohort,
        source_root,
        source_manifest_path,
        resolved_config,
        backend_info,
    )
    _write_or_check_manifest(output_root, manifest, resume=resume)
    tasks = tuple(
        (
            case.to_dict(),
            (
                None
                if case.index not in cohort.complex_cases
                else asdict(cohort.complex_cases[case.index])
            ),
            str(source_root),
            str(output_root),
            resolved_config,
            resume,
        )
        for case in ligand_cases
    )

    started = perf_counter()
    records = []
    if resolved_config.workers == 1:
        for completed, task in enumerate(tasks, start=1):
            record = run_case(*task)
            records.append(record)
            print(
                f"[xtb {completed}/{len(tasks)}] case={int(record['index']):04d}",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=resolved_config.workers,
            mp_context=context,
        ) as pool:
            futures = {pool.submit(run_case, *task): task[0]["index"] for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                print(
                    f"[xtb {completed}/{len(tasks)}] "
                    f"case={int(record['index']):04d}",
                    flush=True,
                )
    records.sort(key=lambda record: int(record["index"]))
    return _write_summary(
        output_root,
        records,
        resolved_config,
        perf_counter() - started,
    )


# CLI.


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit opt-in command for the scientific benchmark."""

    parser = argparse.ArgumentParser(
        description=(
            "Refine prepared 187-ligand and Eu-complex UFF structures through "
            "direct GFN2 and GFN-FF -> GFN2 routes."
        )
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="acknowledge that the command launches the official xTB backend",
    )
    parser.add_argument("--input", type=Path, required=True)
    cohort_source = parser.add_mutually_exclusive_group(required=True)
    cohort_source.add_argument("--reference", type=Path)
    cohort_source.add_argument("--cohort", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--routes",
        choices=("both", *(route.value for route in XTBRoute)),
        default="both",
    )
    parser.add_argument(
        "--targets",
        choices=("both", *(target.value for target in BenchmarkTarget)),
        default="both",
    )
    parser.add_argument(
        "--cases",
        type=parse_case_selection,
        help="one-based indices/ranges, for example 1-10,54,61,109",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only the first selected case without changing scientific settings",
    )
    parser.add_argument("--workers", type=_positive_int, default=16)
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--timeout", type=_positive_float, default=1000.0)
    parser.add_argument(
        "--quality",
        choices=("basic", "standard", "strict"),
        default="standard",
    )
    parser.add_argument("--xtb-executable", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Resolve a canonical cohort and launch the explicit official benchmark."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.official:
        parser.error("--official is required to launch the real xTB benchmark")

    cohort = resolve_cohort(
        arguments.input,
        arguments.output,
        reference_root=arguments.reference,
        cohort_path=arguments.cohort,
    )
    selected_indices = arguments.cases or ()
    if arguments.smoke:
        selected_indices = (selected_indices[0],) if selected_indices else (1,)
    config = XTBBenchmarkConfig(
        routes=_route_selection(arguments.routes),
        targets=_target_selection(arguments.targets),
        case_indices=selected_indices,
        workers=arguments.workers,
        threads=arguments.threads,
        timeout_seconds=arguments.timeout,
        quality_level=arguments.quality,
        executable=arguments.xtb_executable,
    )
    summary = run_benchmark(
        cohort,
        arguments.source,
        arguments.output,
        config,
        resume=arguments.resume,
    )
    print(
        f"completed={summary['sample_count']} wall_seconds={summary['wall_seconds']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
