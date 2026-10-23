"""Resumable execution for the owner-approved Plan-17 validation matrix.

The semantic cohort is the only graph input accepted by route workers.  CBond
is invoked while building that cohort and is never invoked by ``run_case`` or
``run_matrix``.  Result records describe automated A01--A10 acceptance only;
human review remains explicitly pending until the separate gallery workflow
produces a completed review artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import re
import resource
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .configuration import REPOSITORY_ROOT, BenchmarkSettings, STANDARD_SETTINGS
from .validation_acceptance import GATE_IDS
from .validation_cohort import (
    CANONICAL_SOURCE_PATH,
    build_validation_cohort,
    load_validation_cohort,
    semantic_case_molecule,
    validate_validation_cohort,
    write_validation_cohort,
)
from .validation_evidence import evaluate_route, write_selected_structure
from .validation_routes import (
    ValidationRoute,
    execute_provider_route,
    execute_whole_complex_route,
)


RESULT_SCHEMA = "hotpot-plan17-automated-route-result/v1"
RUN_MANIFEST_SCHEMA = "hotpot-plan17-validation-run/v1"
SUMMARY_SCHEMA = "hotpot-plan17-validation-summary/v1"
PINNED_RDKIT_VERSION = "2026.09.1"
ROUTES: tuple[ValidationRoute, ...] = ("O0", "F1", "F2", "C1", "C2")
STAGE_TIMING_FIELDS = (
    "semantic_reconstruction_seconds",
    "provider_or_generation_seconds",
    "sampling_seconds",
    "ligand_build_seconds",
    "placement_seconds",
    "coordination_restoration_seconds",
    "optimization_seconds",
    "acceptance_seconds",
    "selected_structure_write_seconds",
)
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


CaseExecutor = Callable[[object, ValidationRoute, BenchmarkSettings, int], object]
EvidenceEvaluator = Callable[[Mapping[str, object], object], tuple[object, object]]
StructureWriter = Callable[[object, object, Path], Mapping[str, object]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: object) -> object:
    """Return strict JSON data while retaining non-finite diagnostics."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        token = "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
        return {"nonfinite_float": token}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported validation JSON value: {type(value).__name__}")


def _atomic_write_json(path: Path, value: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _json_value(value),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_git_identity() -> dict[str, object]:
    """Return the exact repository revision and worktree state."""

    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def collect_runtime_identity() -> dict[str, object]:
    """Read installed Hotpot, native RDKit, Open Babel, and Python identities."""

    import hotpot
    from hotpot.cheminfo import rdWrappers

    native = rdWrappers.runtime_info()
    try:
        from openbabel import openbabel

        openbabel_version: Optional[str] = str(openbabel.OBReleaseVersion())
    except (ImportError, AttributeError):
        openbabel_version = None
    return {
        "hotpot_version": str(native.hotpot_version),
        "imported_hotpot_version": str(hotpot.version()),
        "rdkit_version": str(native.rdkit_version),
        "native": bool(native.native),
        "openbabel_version": openbabel_version,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def collect_process_resource_identity() -> dict[str, object]:
    """Record the host/process/thread controls without estimating usage."""

    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    )
    return {
        "hostname": platform.node(),
        "process_id": os.getpid(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "thread_environment": {
            name: os.environ.get(name) for name in thread_variables
        },
    }


def _resource_usage_snapshot() -> object:
    return resource.getrusage(resource.RUSAGE_SELF)


def _resource_usage_delta(before: object, after: object) -> dict[str, object]:
    return {
        "user_cpu_seconds": after.ru_utime - before.ru_utime,
        "system_cpu_seconds": after.ru_stime - before.ru_stime,
        "minor_page_faults": after.ru_minflt - before.ru_minflt,
        "major_page_faults": after.ru_majflt - before.ru_majflt,
        "voluntary_context_switches": after.ru_nvcsw - before.ru_nvcsw,
        "involuntary_context_switches": after.ru_nivcsw - before.ru_nivcsw,
        "process_max_rss_kib_at_case_end": after.ru_maxrss,
        "max_rss_scope": "process_high_watermark_not_case_delta",
    }


def validate_formal_start(
    runtime: Mapping[str, object],
    git: Mapping[str, object],
) -> None:
    """Reject a formal run unless the pinned native runtime is reproducible."""

    if runtime.get("rdkit_version") != PINNED_RDKIT_VERSION:
        raise RuntimeError(
            "formal validation requires native RDKit "
            f"{PINNED_RDKIT_VERSION}; found {runtime.get('rdkit_version')!r}"
        )
    if runtime.get("native") is not True:
        raise RuntimeError("formal validation requires the native RDKit backend")
    if runtime.get("hotpot_version") != runtime.get("imported_hotpot_version"):
        raise RuntimeError("installed Hotpot and native wrapper versions differ")
    if not git.get("commit"):
        raise RuntimeError("formal validation requires a Git commit identity")
    if git.get("dirty") is not False:
        raise RuntimeError("formal validation requires a clean Git worktree")


def build_frozen_cohort(
    output_path: Path,
    *,
    source_path: Path = CANONICAL_SOURCE_PATH,
    reuse: bool = False,
) -> dict[str, object]:
    """Build CBond semantics once, then atomically freeze the 187/182 cohort."""

    output_path = output_path.resolve()
    if output_path.is_file():
        if not reuse:
            raise FileExistsError(
                f"{output_path} already exists; pass reuse=True to validate it"
            )
        return load_validation_cohort(output_path)

    cohort = build_validation_cohort(source_path=source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.building")
    try:
        write_validation_cohort(temporary, cohort)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return cohort


def _safe_segment(value: str, field: str) -> str:
    if not _SAFE_SEGMENT.fullmatch(value):
        raise ValueError(
            f"{field} must contain only letters, digits, '.', '_', and '-'"
        )
    return value


def case_result_path(
    output_root: Path,
    *,
    route: ValidationRoute,
    run_id: str,
    case_index: int,
) -> Path:
    safe_run = _safe_segment(run_id, "run_id")
    if route not in ROUTES:
        raise ValueError(f"unsupported validation route: {route}")
    return (
        output_root.resolve()
        / "runs"
        / safe_run
        / route
        / "cases"
        / f"{case_index:04d}"
        / "result.json"
    )


def _execute_route(
    molecule: object,
    route: ValidationRoute,
    settings: BenchmarkSettings,
    case_index: int,
    *,
    proposal_artifact_path: Optional[Path] = None,
) -> object:
    if route in {"O0", "F1", "F2"}:
        return execute_provider_route(
            molecule,
            route,
            settings,
            case_index,
            proposal_artifact_path=proposal_artifact_path,
        )
    return execute_whole_complex_route(
        molecule,
        route,
        settings,
        case_index,
        proposal_artifact_path=proposal_artifact_path,
    )


def _source_identity(cohort: Mapping[str, object]) -> dict[str, object]:
    source = cohort["source"]
    return {
        "path": str(CANONICAL_SOURCE_PATH.resolve()),
        "sha256": source["sha256"],
        "physical_line_count": int(source["physical_line_count"]),
    }


def _cohort_identity(
    cohort: Mapping[str, object],
    cohort_path: Optional[Path],
) -> dict[str, object]:
    return {
        "path": None if cohort_path is None else str(cohort_path.resolve()),
        "canonical_sha256": cohort["canonical_sha256"],
        "schema_version": int(cohort["schema_version"]),
        "cbond_count": int(cohort["partition"]["cbond_count"]),
        "no_cbond_count": int(cohort["partition"]["no_cbond_count"]),
    }


def _base_identity(
    cohort: Mapping[str, object],
    *,
    cohort_path: Optional[Path],
    settings: BenchmarkSettings,
    runtime: Mapping[str, object],
    git: Mapping[str, object],
) -> dict[str, object]:
    return {
        "source": _source_identity(cohort),
        "cohort": _cohort_identity(cohort, cohort_path),
        "git": dict(git),
        "runtime": dict(runtime),
        "settings": dict(settings.to_manifest()),
    }


def _positive_case(
    cohort: Mapping[str, object],
    case_index: int,
) -> Mapping[str, object]:
    case = next(
        (item for item in cohort["cases"] if int(item["index"]) == case_index),
        None,
    )
    if case is None:
        raise ValueError(f"case index {case_index} is absent from the cohort")
    if case["status"] != "cbond":
        raise ValueError(f"case index {case_index} is not CBond-positive")
    return case


def _record_value(value: object) -> dict[str, object]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"validation record has no JSON representation: {type(value)}")


def _proposal_artifact_identity(path: Optional[Path]) -> Optional[dict[str, object]]:
    if path is None or not path.is_file():
        return None
    identity: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {
            **identity,
            "valid_json": False,
            "diagnostic": f"{type(error).__name__}: {error}",
        }
    return {
        **identity,
        "valid_json": True,
        **{
            field: document[field]
            for field in (
                "canonical_sha256",
                "proposal_count",
                "atom_count",
                "coordinate_sha256",
            )
            if field in document
        },
    }


def _verify_proposal_artifact(
    execution: object,
    path: Path,
    route: ValidationRoute,
) -> dict[str, object]:
    archived = _proposal_artifact_identity(path)
    if archived is None or archived.get("valid_json") is not True:
        raise RuntimeError(f"{route} pre-forcefield proposal artifact is unavailable")
    declared = getattr(execution, "proposal_artifact", None)
    if not isinstance(declared, Mapping):
        raise RuntimeError(f"{route} execution omitted proposal artifact identity")
    route_fields = (
        ("proposal_count",)
        if route == "F2"
        else ("atom_count", "coordinate_sha256")
    )
    for field in ("sha256", "canonical_sha256", *route_fields):
        if declared.get(field) != archived.get(field):
            raise RuntimeError(
                f"{route} proposal artifact {field} differs after replay"
            )
    if Path(str(declared.get("path"))).resolve() != path.resolve():
        raise RuntimeError(f"{route} proposal artifact path differs after replay")
    return archived


def _proposal_identity(
    execution: Optional[object],
    artifact_path: Optional[Path],
) -> Optional[dict[str, object]]:
    archived = _proposal_artifact_identity(artifact_path)
    if execution is None and archived is None:
        return None
    proposal = getattr(execution, "proposal", None)
    manifest = getattr(execution, "proposal_manifest", ()) if execution else ()
    return {
        "record": None if proposal is None else _json_value(proposal),
        "frozen_manifest": _json_value(tuple(manifest)),
        "declared_artifact": (
            None
            if execution is None
            else _json_value(getattr(execution, "proposal_artifact", None))
        ),
        "archived_artifact": archived,
    }


def _selected_optimization(
    evidence: Optional[Mapping[str, object]],
) -> dict[str, object]:
    selected = None
    if evidence is not None:
        selected = next(
            (
                frame
                for frame in evidence.get("frames", ())
                if frame.get("role") == "selected"
            ),
            None,
        )
    if selected is None:
        return {
            "best_energy": None,
            "rms_gradient": None,
            "max_gradient": None,
        }
    return {
        "best_energy": selected["energy"].get("value"),
        "rms_gradient": selected["rms_gradient"].get("value"),
        "max_gradient": selected["max_gradient"].get("value"),
    }


def _stage_timings(
    execution: Optional[object],
    *,
    reconstruction_seconds: float,
    acceptance_seconds: Optional[float],
    selected_structure_seconds: Optional[float],
) -> dict[str, object]:
    if execution is None:
        return {
            "semantic_reconstruction_seconds": reconstruction_seconds,
            "provider_or_generation_seconds": None,
            "sampling_seconds": None,
            "sampling_timing_scope": "unavailable",
            "ligand_build_seconds": None,
            "placement_seconds": None,
            "coordination_restoration_seconds": None,
            "optimization_seconds": None,
            "acceptance_seconds": acceptance_seconds,
            "selected_structure_write_seconds": selected_structure_seconds,
        }

    report = execution.report
    build = getattr(report, "build", None)
    optimization = getattr(report, "optimization", report)
    restoration = getattr(build, "coordination_restoration", None)
    route = str(execution.route)
    proposal_seconds = float(execution.proposal_seconds)
    return {
        "semantic_reconstruction_seconds": reconstruction_seconds,
        "provider_or_generation_seconds": proposal_seconds,
        "sampling_seconds": proposal_seconds if route in {"F2", "C1", "C2"} else None,
        "sampling_timing_scope": (
            "pre_forcefield_generation_including_adapter_overhead"
            if route in {"F2", "C1", "C2"}
            else "included_in_ligand_build_not_separately_instrumented"
        ),
        "ligand_build_seconds": getattr(
            build,
            "ligand_build_elapsed_seconds",
            None,
        ),
        # The production report has no independent placement clock.  Do not
        # infer it by subtracting overlapping stage totals.
        "placement_seconds": None,
        "coordination_restoration_seconds": getattr(
            restoration,
            "elapsed_seconds",
            None,
        ),
        "optimization_seconds": getattr(
            optimization,
            "elapsed_seconds",
            None,
        ),
        "acceptance_seconds": acceptance_seconds,
        "selected_structure_write_seconds": selected_structure_seconds,
    }


def _unevaluated_acceptance(
    case_index: int,
    route: ValidationRoute,
    phase: str,
) -> dict[str, object]:
    diagnostic = f"not_evaluated:{phase}"
    return {
        "case_id": f"extractant-{case_index:03d}",
        "route": route,
        "passed": False,
        "gates": [
            {
                "gate_id": gate_id,
                "passed": False,
                "diagnostics": [diagnostic],
            }
            for gate_id in GATE_IDS
        ],
        "diagnostics": [diagnostic],
    }


def _selected_structure(
    execution: object,
    acceptance: object,
    result_path: Path,
    writer: StructureWriter,
) -> dict[str, object]:
    target = result_path.parent / "selected.sdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.writing{target.suffix}")
    try:
        metadata = dict(writer(execution, acceptance, temporary))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    metadata.update(
        {
            "path": str(target.resolve()),
            "sha256": _sha256(target),
            "format": "sdf",
        }
    )
    return metadata


def _tree_identity(path: Path) -> dict[str, object]:
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    digest = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        content_digest = bytes.fromhex(_sha256(file_path))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(content_digest)
        total_bytes += file_path.stat().st_size
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "includes_sdf": False,
    }


def _execution_trajectory(execution: object) -> Optional[object]:
    report = execution.report
    optimization = getattr(report, "optimization", report)
    archive = getattr(optimization, "trajectory", None)
    if archive is None:
        archive = getattr(report, "trajectory", None)
    return archive


def _write_trajectory_artifact(
    archive: Optional[object],
    result_path: Path,
) -> Optional[dict[str, object]]:
    if archive is None:
        return None
    target = result_path.parent / "trajectory"
    archive.write(target, include_sdf=False)
    return _tree_identity(target)


def _resume_record(
    result_path: Path,
    *,
    route: ValidationRoute,
    run_id: str,
    case_index: int,
    identity: Mapping[str, object],
) -> dict[str, object]:
    record = json.loads(result_path.read_text(encoding="utf-8"))
    expected = {
        "route": route,
        "run_id": run_id,
        "case_index": case_index,
        "identity": _json_value(identity),
    }
    actual = {key: record.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"resume identity differs from existing result {result_path}"
        )
    return record


def run_case(
    cohort: Mapping[str, object],
    *,
    case_index: int,
    route: ValidationRoute,
    run_id: str,
    output_root: Path,
    cohort_path: Optional[Path] = None,
    settings: BenchmarkSettings = STANDARD_SETTINGS,
    formal: bool = True,
    resume: bool = False,
    runtime: Optional[Mapping[str, object]] = None,
    git: Optional[Mapping[str, object]] = None,
    executor: Optional[CaseExecutor] = None,
    evaluator: Optional[EvidenceEvaluator] = None,
    structure_writer: Optional[StructureWriter] = None,
    cohort_validator: Optional[Callable[[Mapping[str, object]], None]] = None,
    molecule_builder: Optional[Callable[[Mapping[str, object]], object]] = None,
) -> dict[str, object]:
    """Execute one route-case from frozen semantics and atomically archive it."""

    selected_validator = cohort_validator or validate_validation_cohort
    selected_validator(cohort)
    if formal and settings != STANDARD_SETTINGS:
        raise ValueError("formal validation requires STANDARD_SETTINGS")
    runtime_identity = dict(runtime or collect_runtime_identity())
    git_identity = dict(git or collect_git_identity())
    if formal:
        validate_formal_start(runtime_identity, git_identity)
    identity = _base_identity(
        cohort,
        cohort_path=cohort_path,
        settings=settings,
        runtime=runtime_identity,
        git=git_identity,
    )
    result_path = case_result_path(
        output_root,
        route=route,
        run_id=run_id,
        case_index=case_index,
    )
    proposal_artifact_path = (
        result_path.parent / "frozen-proposals.json"
        if route == "F2"
        else (
            result_path.parent / "whole-complex-proposal.json"
            if route in {"C1", "C2"}
            else None
        )
    )
    if result_path.is_file():
        if not resume:
            raise FileExistsError(
                f"{result_path} already exists; pass resume=True to reuse it"
            )
        return _resume_record(
            result_path,
            route=route,
            run_id=run_id,
            case_index=case_index,
            identity=identity,
        )

    case = _positive_case(cohort, case_index)
    selected_builder = molecule_builder or semantic_case_molecule
    selected_executor = executor or _execute_route
    selected_evaluator = evaluator or evaluate_route
    selected_writer = structure_writer or write_selected_structure
    started = perf_counter()
    resource_started = _resource_usage_snapshot()
    phase = "semantic_reconstruction"
    execution: Optional[object] = None
    evidence_record: Optional[dict[str, object]] = None
    acceptance_record: Optional[dict[str, object]] = None
    selected_structure: Optional[dict[str, object]] = None
    error_record: Optional[dict[str, object]] = None
    trajectory_artifact: Optional[dict[str, object]] = None
    acceptance_seconds: Optional[float] = None
    selected_structure_seconds: Optional[float] = None

    try:
        reconstruction_started = perf_counter()
        molecule = selected_builder(case)
        reconstruction_seconds = perf_counter() - reconstruction_started
        phase = "route_execution"
        execution = (
            _execute_route(
                molecule,
                route,
                settings,
                case_index,
                proposal_artifact_path=proposal_artifact_path,
            )
            if executor is None
            else selected_executor(molecule, route, settings, case_index)
        )
        if proposal_artifact_path is not None:
            phase = "proposal_artifact_verification"
            _verify_proposal_artifact(
                execution,
                proposal_artifact_path,
                route,
            )
        phase = "trajectory_archive_write"
        trajectory_artifact = _write_trajectory_artifact(
            _execution_trajectory(execution),
            result_path,
        )
        phase = "acceptance_evaluation"
        acceptance_started = perf_counter()
        evidence, acceptance = selected_evaluator(case, execution)
        acceptance_seconds = perf_counter() - acceptance_started
        evidence_record = _record_value(evidence)
        acceptance_record = _record_value(acceptance)
        automated_passed = bool(acceptance_record["passed"])
        if automated_passed:
            phase = "selected_structure_write"
            structure_started = perf_counter()
            selected_structure = _selected_structure(
                execution,
                acceptance,
                result_path,
                selected_writer,
            )
            selected_structure_seconds = perf_counter() - structure_started
        status = "automated_passed" if automated_passed else "automated_failed"
    except Exception as error:  # Preserve every route failure as evidence.
        reconstruction_seconds = (
            perf_counter() - reconstruction_started
            if phase == "semantic_reconstruction"
            else reconstruction_seconds
        )
        automated_passed = False
        status = "execution_failed"
        error_record = {
            "phase": phase,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        if trajectory_artifact is None:
            exception_archive = getattr(error, "trajectory", None)
            try:
                trajectory_artifact = _write_trajectory_artifact(
                    exception_archive,
                    result_path,
                )
            except Exception as archive_error:
                error_record["trajectory_archive_error"] = (
                    f"{type(archive_error).__name__}: {archive_error}"
                )
        if acceptance_record is None:
            acceptance_record = _unevaluated_acceptance(
                case_index,
                route,
                phase,
            )

    completed = perf_counter()
    resource_completed = _resource_usage_snapshot()
    proposal_seconds = (
        None if execution is None else float(execution.proposal_seconds)
    )
    forcefield_seconds = (
        None if execution is None else float(execution.forcefield_seconds)
    )
    route_seconds = None if execution is None else float(execution.elapsed_seconds)
    record = {
        "schema": RESULT_SCHEMA,
        "case_index": case_index,
        "source_smiles": case["smiles"],
        "semantic_graph_sha256": case["semantic_graph_sha256"],
        "route": route,
        "run_id": run_id,
        "seed": int(case["seed"]),
        "status": status,
        "passed": automated_passed,
        "identity": identity,
        "proposal": _proposal_identity(execution, proposal_artifact_path),
        "evidence": evidence_record,
        "acceptance": acceptance_record,
        "optimization": _selected_optimization(evidence_record),
        "timing": {
            "proposal_seconds": proposal_seconds,
            "forcefield_seconds": forcefield_seconds,
            "route_elapsed_seconds": route_seconds,
            "runner_elapsed_seconds": completed - started,
            "stages": _stage_timings(
                execution,
                reconstruction_seconds=reconstruction_seconds,
                acceptance_seconds=acceptance_seconds,
                selected_structure_seconds=selected_structure_seconds,
            ),
        },
        "resources": {
            "identity": collect_process_resource_identity(),
            "usage": _resource_usage_delta(
                resource_started,
                resource_completed,
            ),
        },
        "trajectory_artifact": trajectory_artifact,
        "selected_structure": selected_structure,
        "manual_review": {
            "status": "pending_human_review",
            "reviewer": None,
            "completed_at": None,
            "flags": [],
        },
        "error": error_record,
        "recorded_at": _utc_now(),
    }
    _atomic_write_json(result_path, record)
    return record


def _run_case_from_file(task: Mapping[str, object]) -> dict[str, object]:
    cohort_path = Path(str(task["cohort_path"]))
    cohort = load_validation_cohort(cohort_path)
    return run_case(
        cohort,
        case_index=int(task["case_index"]),
        route=str(task["route"]),
        run_id=str(task["run_id"]),
        output_root=Path(str(task["output_root"])),
        cohort_path=cohort_path,
        formal=bool(task["formal"]),
        resume=bool(task["resume"]),
        runtime=task["runtime"],
        git=task["git"],
    )


def validation_run_specs(
    *,
    run_prefix: str,
    routes: Sequence[ValidationRoute] = ROUTES,
    f1_repeat: bool = True,
) -> tuple[dict[str, object], ...]:
    """Return stable primary run IDs plus the independent full F1 repeat."""

    _safe_segment(run_prefix, "run_prefix")
    if len(set(routes)) != len(routes) or any(route not in ROUTES for route in routes):
        raise ValueError("routes must be unique members of O0/F1/F2/C1/C2")
    specs = [
        {
            "route": route,
            "run_id": f"{run_prefix}-{route.lower()}",
            "repeat_of": None,
        }
        for route in routes
    ]
    if f1_repeat:
        if "F1" not in routes:
            raise ValueError("an F1 repeat requires F1 in the primary routes")
        primary = next(spec["run_id"] for spec in specs if spec["route"] == "F1")
        specs.append(
            {
                "route": "F1",
                "run_id": f"{run_prefix}-f1-repeat",
                "repeat_of": primary,
            }
        )
    return tuple(specs)


def _run_manifest_configuration(
    cohort: Mapping[str, object],
    *,
    cohort_path: Path,
    run_specs: Sequence[Mapping[str, object]],
    case_indices: Sequence[int],
    settings: BenchmarkSettings,
    runtime: Mapping[str, object],
    git: Mapping[str, object],
    formal: bool,
) -> dict[str, object]:
    return {
        "formal": formal,
        "identity": _base_identity(
            cohort,
            cohort_path=cohort_path,
            settings=settings,
            runtime=runtime,
            git=git,
        ),
        "run_specs": [dict(spec) for spec in run_specs],
        "case_indices": list(case_indices),
    }


def _validate_formal_matrix_contract(
    run_specs: Sequence[Mapping[str, object]],
    selected_indices: Sequence[int],
    available_indices: Sequence[int],
) -> None:
    if tuple(selected_indices) != tuple(available_indices):
        raise ValueError("formal matrix requires all 182 CBond-positive cases")
    primary = tuple(spec for spec in run_specs if spec.get("repeat_of") is None)
    repeats = tuple(spec for spec in run_specs if spec.get("repeat_of") is not None)
    if tuple(spec.get("route") for spec in primary) != ROUTES:
        raise ValueError("formal matrix requires primary O0/F1/F2/C1/C2 runs")
    if len(repeats) != 1 or repeats[0].get("route") != "F1":
        raise ValueError("formal matrix requires exactly one complete F1 repeat")
    primary_f1 = next(spec for spec in primary if spec.get("route") == "F1")
    if repeats[0].get("repeat_of") != primary_f1.get("run_id"):
        raise ValueError("formal F1 repeat must identify the primary F1 run")


def _write_or_check_manifest(
    output_root: Path,
    configuration: Mapping[str, object],
    *,
    workers: int,
    resume: bool,
) -> Path:
    manifest_path = output_root.resolve() / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(
                f"{manifest_path} already exists; pass resume=True to reuse it"
            )
        if existing.get("scientific_configuration") != _json_value(configuration):
            raise ValueError("resume scientific configuration differs from manifest")
        return manifest_path
    _atomic_write_json(
        manifest_path,
        {
            "schema": RUN_MANIFEST_SCHEMA,
            "created_at": _utc_now(),
            "scientific_configuration": configuration,
            "execution": {
                "workers": workers,
                "multiprocessing_start_method": "spawn" if workers > 1 else None,
            },
            "human_review": {
                "status": "not_performed",
                "default_replacement_authorized": False,
            },
        },
    )
    return manifest_path


def run_matrix(
    cohort_path: Path,
    output_root: Path,
    *,
    run_specs: Sequence[Mapping[str, object]],
    case_indices: Optional[Sequence[int]] = None,
    workers: int = 1,
    formal: bool = True,
    resume: bool = False,
    runtime: Optional[Mapping[str, object]] = None,
    git: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    """Run a bounded, restartable route-case matrix from one frozen cohort."""

    if workers < 1:
        raise ValueError("workers must be positive")
    cohort_path = cohort_path.resolve()
    cohort = load_validation_cohort(cohort_path)
    runtime_identity = dict(runtime or collect_runtime_identity())
    git_identity = dict(git or collect_git_identity())
    if formal:
        validate_formal_start(runtime_identity, git_identity)
    available = tuple(int(index) for index in cohort["partition"]["cbond_indices"])
    selected_indices = available if case_indices is None else tuple(case_indices)
    if not selected_indices or len(set(selected_indices)) != len(selected_indices):
        raise ValueError("case_indices must be a non-empty unique sequence")
    missing = sorted(set(selected_indices) - set(available))
    if missing:
        raise ValueError(f"case_indices are not CBond-positive: {missing}")
    normalized_specs = tuple(dict(spec) for spec in run_specs)
    if not normalized_specs:
        raise ValueError("run_specs must not be empty")
    run_ids = [str(spec["run_id"]) for spec in normalized_specs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run_specs require distinct run IDs")
    for spec in normalized_specs:
        route = str(spec["route"])
        if route not in ROUTES:
            raise ValueError(f"unsupported route in run_specs: {route}")
        _safe_segment(str(spec["run_id"]), "run_id")
    if formal:
        _validate_formal_matrix_contract(
            normalized_specs,
            selected_indices,
            available,
        )

    configuration = _run_manifest_configuration(
        cohort,
        cohort_path=cohort_path,
        run_specs=normalized_specs,
        case_indices=selected_indices,
        settings=STANDARD_SETTINGS,
        runtime=runtime_identity,
        git=git_identity,
        formal=formal,
    )
    _write_or_check_manifest(
        output_root,
        configuration,
        workers=workers,
        resume=resume,
    )
    tasks = tuple(
        {
            "cohort_path": str(cohort_path),
            "output_root": str(output_root.resolve()),
            "case_index": case_index,
            "route": spec["route"],
            "run_id": spec["run_id"],
            "formal": formal,
            "resume": resume,
            "runtime": runtime_identity,
            "git": git_identity,
        }
        for spec in normalized_specs
        for case_index in selected_indices
    )
    if workers == 1:
        records = [_run_case_from_file(task) for task in tasks]
    else:
        records = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_run_case_from_file, task) for task in tasks]
            for future in as_completed(futures):
                records.append(future.result())
    summary = summarize_results(output_root)
    return {
        "planned_case_count": len(tasks),
        "completed_case_count": len(records),
        "summary": summary,
    }


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_values_summary(
    values: Sequence[float],
    *,
    expected_count: int,
) -> dict[str, object]:
    return {
        "expected_count": expected_count,
        "observed_count": len(values),
        "p50_seconds": _percentile(values, 0.50),
        "p95_seconds": _percentile(values, 0.95),
        "max_seconds": max(values) if values else None,
        "sum_seconds": sum(values),
    }


def _timing_summary(
    records: Sequence[Mapping[str, object]],
    *,
    expected_count: int,
) -> dict[str, object]:
    values = [
        float(record["timing"]["runner_elapsed_seconds"])
        for record in records
        if record.get("timing", {}).get("runner_elapsed_seconds") is not None
    ]
    return _timing_values_summary(values, expected_count=expected_count)


def _stage_timing_summary(
    records: Sequence[Mapping[str, object]],
    *,
    expected_count: int,
) -> dict[str, object]:
    result = {}
    for field in STAGE_TIMING_FIELDS:
        values = [
            float(value)
            for record in records
            if (
                value := record.get("timing", {})
                .get("stages", {})
                .get(field)
            )
            is not None
        ]
        result[field] = _timing_values_summary(
            values,
            expected_count=expected_count,
        )
    return result


def _resource_summary(
    records: Sequence[Mapping[str, object]],
    *,
    expected_count: int,
) -> dict[str, object]:
    usage = [
        record["resources"]["usage"]
        for record in records
        if record.get("resources", {}).get("usage") is not None
    ]
    rss = [float(item["process_max_rss_kib_at_case_end"]) for item in usage]
    return {
        "expected_count": expected_count,
        "observed_count": len(usage),
        "user_cpu_seconds_sum": sum(float(item["user_cpu_seconds"]) for item in usage),
        "system_cpu_seconds_sum": sum(
            float(item["system_cpu_seconds"]) for item in usage
        ),
        "maximum_observed_process_max_rss_kib": max(rss) if rss else None,
        "max_rss_scope": "process_high_watermark_observations_not_case_deltas",
    }


def _gate_signature(record: Mapping[str, object]) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (str(gate["gate_id"]), bool(gate["passed"]))
        for gate in record["acceptance"]["gates"]
    )


def _f1_repeat_summary(
    records_by_run: Mapping[str, Mapping[int, Mapping[str, object]]],
    run_specs: Sequence[Mapping[str, object]],
    case_indices: Sequence[int],
) -> dict[str, object]:
    repeat = next(
        (spec for spec in run_specs if spec.get("repeat_of") is not None),
        None,
    )
    if repeat is None:
        return {"status": "not_requested"}
    primary_id = str(repeat["repeat_of"])
    repeat_id = str(repeat["run_id"])
    primary = records_by_run.get(primary_id, {})
    repeated = records_by_run.get(repeat_id, {})
    comparable = sorted(set(primary) & set(repeated) & set(case_indices))
    pass_mismatches = [
        index
        for index in comparable
        if bool(primary[index]["passed"]) != bool(repeated[index]["passed"])
    ]
    gate_mismatches = [
        index
        for index in comparable
        if _gate_signature(primary[index]) != _gate_signature(repeated[index])
    ]
    missing = sorted(set(case_indices) - set(comparable))
    return {
        "status": "complete" if not missing else "incomplete",
        "primary_run_id": primary_id,
        "repeat_run_id": repeat_id,
        "expected_count": len(case_indices),
        "comparable_count": len(comparable),
        "missing_case_indices": missing,
        "acceptance_classification_consistent": not missing and not pass_mismatches,
        "acceptance_mismatch_case_indices": pass_mismatches,
        "gate_classification_consistent": not missing and not gate_mismatches,
        "gate_mismatch_case_indices": gate_mismatches,
    }


def load_result_records(output_root: Path) -> tuple[dict[str, object], ...]:
    """Load only canonical per-case result paths beneath a run root."""

    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_root.resolve().glob("runs/*/*/cases/*/result.json"))
    )


def summarize_results(output_root: Path) -> dict[str, object]:
    """Aggregate exact planned denominators, gates, timing, and F1 repeat parity."""

    output_root = output_root.resolve()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    configuration = manifest["scientific_configuration"]
    run_specs = configuration["run_specs"]
    case_indices = tuple(int(index) for index in configuration["case_indices"])
    records = load_result_records(output_root)
    records_by_run: dict[str, dict[int, Mapping[str, object]]] = {}
    duplicates: list[dict[str, object]] = []
    for record in records:
        run_id = str(record["run_id"])
        case_index = int(record["case_index"])
        by_case = records_by_run.setdefault(run_id, {})
        if case_index in by_case:
            duplicates.append({"run_id": run_id, "case_index": case_index})
        by_case[case_index] = record

    run_summaries = []
    missing_results = []
    expected_identities = {
        (str(spec["run_id"]), str(spec["route"]), case_index)
        for spec in run_specs
        for case_index in case_indices
    }
    actual_identities = {
        (str(record["run_id"]), str(record["route"]), int(record["case_index"]))
        for record in records
    }
    unexpected_results = [
        {"run_id": run_id, "route": route, "case_index": case_index}
        for run_id, route, case_index in sorted(
            actual_identities - expected_identities
        )
    ]
    for spec in run_specs:
        run_id = str(spec["run_id"])
        route = str(spec["route"])
        by_case = records_by_run.get(run_id, {})
        selected = [
            by_case[index]
            for index in case_indices
            if index in by_case and by_case[index].get("route") == route
        ]
        missing = [
            index
            for index in case_indices
            if index not in by_case or by_case[index].get("route") != route
        ]
        missing_results.extend(
            {"run_id": run_id, "route": route, "case_index": index}
            for index in missing
        )
        gate_counts = {
            gate_id: {
                "passed": sum(
                    any(
                        gate["gate_id"] == gate_id and gate["passed"]
                        for gate in record["acceptance"]["gates"]
                    )
                    for record in selected
                ),
                "denominator": len(case_indices),
            }
            for gate_id in GATE_IDS
        }
        accepted = sum(bool(record["passed"]) for record in selected)
        run_summaries.append(
            {
                "route": route,
                "run_id": run_id,
                "repeat_of": spec.get("repeat_of"),
                "expected_count": len(case_indices),
                "observed_count": len(selected),
                "missing_case_indices": missing,
                "automated_accepted_count": accepted,
                "automated_rejected_count": len(selected) - accepted,
                "automated_acceptance_rate": accepted / len(case_indices),
                "status_counts": {
                    status: sum(record["status"] == status for record in selected)
                    for status in sorted({str(record["status"]) for record in selected})
                },
                "gates": gate_counts,
                "timing": _timing_summary(
                    selected,
                    expected_count=len(case_indices),
                ),
                "stage_timing": _stage_timing_summary(
                    selected,
                    expected_count=len(case_indices),
                ),
                "resources": _resource_summary(
                    selected,
                    expected_count=len(case_indices),
                ),
            }
        )

    expected_total = len(run_specs) * len(case_indices)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": _utc_now(),
        "manifest_sha256": _sha256(output_root / "manifest.json"),
        "expected_result_count": expected_total,
        "observed_result_count": sum(item["observed_count"] for item in run_summaries),
        "complete": not missing_results and not duplicates and not unexpected_results,
        "missing_results": missing_results,
        "unexpected_results": unexpected_results,
        "duplicate_results": duplicates,
        "runs": run_summaries,
        "overall_timing": _timing_summary(
            records,
            expected_count=expected_total,
        ),
        "overall_stage_timing": _stage_timing_summary(
            records,
            expected_count=expected_total,
        ),
        "overall_resources": _resource_summary(
            records,
            expected_count=expected_total,
        ),
        "f1_repeat": _f1_repeat_summary(records_by_run, run_specs, case_indices),
        "human_review": {
            "status": "not_performed",
            "default_replacement_authorized": False,
        },
    }
    _atomic_write_json(output_root / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build-cohort")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--source", type=Path, default=CANONICAL_SOURCE_PATH)
    build.add_argument("--reuse", action="store_true")

    run_case_parser = subcommands.add_parser("run-case")
    run_case_parser.add_argument("--cohort", type=Path, required=True)
    run_case_parser.add_argument("--output-root", type=Path, required=True)
    run_case_parser.add_argument("--case-index", type=int, required=True)
    run_case_parser.add_argument("--route", choices=ROUTES, required=True)
    run_case_parser.add_argument("--run-id", required=True)
    run_case_parser.add_argument("--resume", action="store_true")
    run_case_parser.add_argument("--non-formal", action="store_true")

    run = subcommands.add_parser("run")
    run.add_argument("--cohort", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--run-prefix", required=True)
    run.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    run.add_argument("--indices", nargs="+", type=int)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--no-f1-repeat", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--non-formal", action="store_true")

    summarize = subcommands.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build-cohort":
        cohort = build_frozen_cohort(
            arguments.output,
            source_path=arguments.source,
            reuse=arguments.reuse,
        )
        result: object = {
            "path": str(arguments.output.resolve()),
            "canonical_sha256": cohort["canonical_sha256"],
            "cbond_count": cohort["partition"]["cbond_count"],
        }
    elif arguments.command == "run-case":
        cohort = load_validation_cohort(arguments.cohort)
        result = run_case(
            cohort,
            case_index=arguments.case_index,
            route=arguments.route,
            run_id=arguments.run_id,
            output_root=arguments.output_root,
            cohort_path=arguments.cohort,
            formal=not arguments.non_formal,
            resume=arguments.resume,
        )
    elif arguments.command == "run":
        specs = validation_run_specs(
            run_prefix=arguments.run_prefix,
            routes=arguments.routes,
            f1_repeat=not arguments.no_f1_repeat,
        )
        result = run_matrix(
            arguments.cohort,
            arguments.output_root,
            run_specs=specs,
            case_indices=arguments.indices,
            workers=arguments.workers,
            formal=not arguments.non_formal,
            resume=arguments.resume,
        )
    else:
        result = summarize_results(arguments.output_root)
    print(json.dumps(_json_value(result), allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = (
    "PINNED_RDKIT_VERSION",
    "RESULT_SCHEMA",
    "ROUTES",
    "build_frozen_cohort",
    "case_result_path",
    "collect_git_identity",
    "collect_runtime_identity",
    "load_result_records",
    "main",
    "run_case",
    "run_matrix",
    "summarize_results",
    "validate_formal_start",
    "validation_run_specs",
)
