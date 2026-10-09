"""One-case Hotpot CBond and force-field benchmark pipeline."""

from __future__ import annotations

import json
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional, TYPE_CHECKING

import numpy as np

from .configuration import BUILTIN_BACKENDS, BenchmarkBackend, BenchmarkSettings
from .io import json_value, write_json

if TYPE_CHECKING:
    from hotpot import Molecule
    from hotpot.cheminfo.AImodels.cbond.apply import CBondPathResult


def _infer_cbond(
    ligand: "Molecule",
    metal: str,
    settings: BenchmarkSettings,
) -> "CBondPathResult":
    """Apply the benchmark's explicitly recorded two-threshold policy."""
    from hotpot.cheminfo.AImodels.cbond.apply import (
        auto_build_cbond,
        get_cbond_runtime,
    )

    return auto_build_cbond(
        ligand,
        metal,
        threshold=settings.subsequent_cbond_threshold,
        first_threshold=settings.first_cbond_threshold,
        runtime=get_cbond_runtime("cpu"),
        return_details=True,
    )


def _quality_payload(report: object) -> dict[str, object]:
    return {
        "level": report.level,
        "passed": report.passed,
        "checks": json_value(report.checks),
        "failures": json_value(report.failures),
        "warnings": json_value(report.warnings),
        "metrics": json_value(report.metrics),
    }


def _trajectory_payload(archive: object) -> Optional[dict[str, object]]:
    if archive is None:
        return None
    main = archive.main
    return {
        "main_frame_count": len(main),
        "main_coordinate_revision_count": main.coordinate_revision_count,
        "main_topology_revision_count": main.topology_revision_count,
        "selected_frame_index": main.selected_index,
        "start": main.start.value,
        "stage_counts": dict(Counter(frame.stage.value for frame in main.frames)),
        "event_counts": dict(Counter(frame.event.value for frame in main.frames)),
        "ligand_build_attempt_count": len(archive.ligand_build_attempts),
        "ligand_build_frame_counts": [
            len(trajectory) for trajectory in archive.ligand_build_attempts
        ],
        "preliminary_attempt_count": len(archive.preliminary_attempts),
        "preliminary_attempt_frame_counts": [
            len(trajectory) for trajectory in archive.preliminary_attempts
        ],
    }


def _cbond_payload(result: object) -> dict[str, object]:
    return {
        "donor_indices": result.donor_indices,
        "donor_count": len(result.donor_indices),
        "path_probability": result.path_probability,
        "steps": json_value(result.steps),
        "complex_smiles": result.molecule.smiles,
    }


@dataclass(frozen=True)
class _ForceFieldOutcome:
    """Normalized evidence returned by either Hotpot force-field workflow."""

    optimization: object
    quality_report: object
    trajectory: object
    requested_forcefield: Optional[str]
    effective_forcefield: str
    build_diagnostics: Optional[object] = None
    routing_report: Optional[object] = None


def _forcefield_options(
    ff: object,
    settings: BenchmarkSettings,
    index: int,
) -> dict[str, object]:
    """Return the settings shared by complete and automatic workflows."""
    return {
        "epochs": settings.epochs,
        "steps_per_epoch": settings.steps_per_epoch,
        "max_attempts": settings.max_attempts,
        "candidate_warmup_steps": settings.candidate_warmup_steps,
        "candidate_score_steps": settings.candidate_score_steps,
        "best_candidate_refine_steps": settings.best_candidate_refine_steps,
        "ligand_untangling_attempts": settings.ligand_untangling_attempts,
        "coordination_restoration_attempts": (
            settings.coordination_restoration_attempts
        ),
        "coordination_relaxation_steps": settings.coordination_relaxation_steps,
        "complex_untangling_attempts": settings.complex_untangling_attempts,
        "timeout": settings.timeout,
        "quality_level": settings.quality_level,
        "seed": settings.seed + index,
        "perturb_sigma": settings.perturb_sigma,
        "convergence_level": ff.ConvergenceLevel.FAST,
        "save_movie": True,
        "trajectory_start": ff.TrajectoryStart(settings.trajectory_start),
        "trajectory_path": None,
    }


def _run_forcefield_backend(
    complex_mol: "Molecule",
    backend: BenchmarkBackend,
    settings: BenchmarkSettings,
    index: int,
) -> _ForceFieldOutcome:
    """Run one configured backend without an external duplicate quality gate."""
    from hotpot.cheminfo import forcefields as ff

    options = _forcefield_options(ff, settings, index)
    if backend.name == "hotpot":
        workflow_report = ff.complexes_build(complex_mol, "UFF", **options)
        optimization_report = workflow_report.optimization
        if optimization_report is None or workflow_report.trajectory is None:
            raise RuntimeError("The complex workflow omitted its final evidence")
        return _ForceFieldOutcome(
            optimization=optimization_report,
            quality_report=workflow_report.quality_report,
            trajectory=workflow_report.trajectory,
            requested_forcefield=workflow_report.requested_forcefield,
            effective_forcefield=workflow_report.effective_forcefield,
            build_diagnostics=workflow_report.build,
        )
    if backend.name == "hotpot-auto":
        optimization_report = ff.auto_optimize(complex_mol, "UFF", **options)
        if (
            optimization_report.quality_report is None
            or optimization_report.trajectory is None
            or optimization_report.routing_report is None
        ):
            raise RuntimeError("The automatic workflow omitted its final evidence")
        return _ForceFieldOutcome(
            optimization=optimization_report,
            quality_report=optimization_report.quality_report,
            trajectory=optimization_report.trajectory,
            requested_forcefield=optimization_report.requested_forcefield,
            effective_forcefield=optimization_report.effective_forcefield,
            routing_report=optimization_report.routing_report,
        )
    raise ValueError(f"unsupported benchmark backend {backend.name!r}")


def _complex_build_diagnostics_from_routing(
    routing_report: object,
) -> Optional[object]:
    """Return fallback build diagnostics when the automatic route used them."""
    if routing_report is None:
        return None
    for attempt in routing_report.attempts:
        if attempt.route.value == "complex_workflow":
            return attempt.build_diagnostics
    return None


def _last_finite_main_frame_index(archive: object) -> Optional[int]:
    for frame in reversed(archive.main.frames):
        if np.all(np.isfinite(archive.main.coordinates(frame.index))):
            return int(frame.index)
    return None


def _selected_or_last_finite_main_frame_index(archive: object) -> Optional[int]:
    selected_index = archive.main.selected_index
    if np.all(np.isfinite(archive.main.coordinates(selected_index))):
        return int(selected_index)
    return _last_finite_main_frame_index(archive)


def _materialize_trajectory_frame(
    trajectory: object,
    frame_index: int,
) -> tuple[object, bool]:
    """Build a serializable molecule from an immutable trajectory frame."""
    from hotpot import Molecule

    output_mol = Molecule()
    for atom, coordinate in zip(
        trajectory.atoms,
        trajectory.coordinates(frame_index),
    ):
        output_mol.create_atom(
            atomic_number=atom.atomic_number,
            formal_charge=atom.formal_charge,
            id=atom.atom_id,
            coordinates=coordinate,
        )

    visualization_topology_lossy = False
    for bond in trajectory.topology(frame_index).bonds:
        bond_kind = bond.bond_kind
        bond_order = bond.bond_order
        if bond_kind == "dative":
            bond_kind = "single"
            bond_order = 1.0
            visualization_topology_lossy = True
        output_mol.add_bond(*bond.atom_indices, bond_order, bond_kind=bond_kind)
    return output_mol, visualization_topology_lossy


def _write_frame_outputs(
    case_dir: Path,
    archive: object,
    frame_index: int,
    *,
    output_frame_role: str,
) -> dict[str, object]:
    return _write_trajectory_frame_outputs(
        case_dir,
        archive.main,
        frame_index,
        output_frame_role=output_frame_role,
    )


def _write_trajectory_frame_outputs(
    case_dir: Path,
    trajectory: object,
    frame_index: int,
    *,
    output_frame_role: str,
) -> dict[str, object]:
    output_mol, visualization_topology_lossy = _materialize_trajectory_frame(
        trajectory,
        frame_index,
    )
    output_mol.write(case_dir / "optimized.mol2", overwrite=True, write_single=True)
    output_mol.write(case_dir / "optimized.sdf", overwrite=True, write_single=True)
    return {
        "optimized_atom_count": len(output_mol.atoms),
        "output_frame_index": frame_index,
        "output_frame_role": output_frame_role,
        "visualization_topology_lossy": visualization_topology_lossy,
    }


def _write_last_finite_failure_frame(
    case_dir: Path,
    archive: object,
) -> dict[str, object]:
    frame_index = _last_finite_main_frame_index(archive)
    if frame_index is not None:
        return _write_frame_outputs(
            case_dir,
            archive,
            frame_index,
            output_frame_role="last_finite_failure_frame",
        )
    for attempt_index in range(len(archive.preliminary_attempts) - 1, -1, -1):
        trajectory = archive.preliminary_attempts[attempt_index]
        for frame in reversed(trajectory.frames):
            if np.all(np.isfinite(trajectory.coordinates(frame.index))):
                return _write_trajectory_frame_outputs(
                    case_dir,
                    trajectory,
                    int(frame.index),
                    output_frame_role=(
                        f"last_finite_preliminary_attempt_{attempt_index}"
                    ),
                )
    return {
        "output_structure_unavailable_reason": (
            "trajectory archive contains no finite coordinate frame"
        )
    }


def _write_quality_failure_frame(
    case_dir: Path,
    archive: object,
) -> dict[str, object]:
    frame_index = _selected_or_last_finite_main_frame_index(archive)
    if frame_index is None:
        return {
            "output_structure_unavailable_reason": (
                "trajectory main branch contains no finite coordinate frame"
            )
        }
    selected_index = int(archive.main.selected_index)
    role = (
        "selected_quality_failure_frame"
        if frame_index == selected_index
        else "last_finite_quality_failure_frame"
    )
    return _write_frame_outputs(
        case_dir,
        archive,
        frame_index,
        output_frame_role=role,
    )


def _record_stage_timings(
    record: dict[str, object],
    diagnostics: Optional[object],
) -> None:
    if diagnostics is None:
        return
    record["ligand_build_seconds"] = diagnostics.ligand_build_elapsed_seconds
    restoration = diagnostics.coordination_restoration
    if restoration is not None:
        record["coordination_restoration_seconds"] = restoration.elapsed_seconds


def _run_case(
    index: int,
    smiles: str,
    output_root_text: str,
    metal: str,
    settings: BenchmarkSettings,
    resume: bool,
    backend_name: str,
) -> dict[str, object]:
    """Run one end-to-end case in a spawn-safe worker process."""
    from hotpot import read_mol
    from hotpot.cheminfo import forcefields as ff

    backend = BUILTIN_BACKENDS[backend_name]
    case_dir = Path(output_root_text) / "cases" / f"{index:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    report_path = case_dir / "report.json"
    if resume and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    record: dict[str, object] = {
        "index": index,
        "smiles": smiles,
        "backend": backend.name,
        "workflow": backend.workflow,
        "status": "running",
        "phase": "read",
        "settings": settings.to_manifest(),
        "routing_report": None,
    }
    (case_dir / "input.smi").write_text(smiles + "\n", encoding="utf-8")
    started = perf_counter()
    phase_started = started

    try:
        ligand = read_mol(smiles, fmt="smi")
        record["input_atom_count"] = len(ligand.atoms)
        record["read_seconds"] = perf_counter() - phase_started

        record["phase"] = "cbond"
        phase_started = perf_counter()
        try:
            cbond_result = _infer_cbond(ligand, metal, settings)
        except ValueError as error:
            record.update(
                status="failed_cbond",
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback.format_exc(),
                cbond_seconds=perf_counter() - phase_started,
            )
        else:
            complex_mol = cbond_result.molecule
            record["cbond_seconds"] = perf_counter() - phase_started
            record["cbond"] = _cbond_payload(cbond_result)
            (case_dir / "cbond.smi").write_text(
                complex_mol.smiles + "\n",
                encoding="utf-8",
            )

            record["phase"] = "forcefield"
            phase_started = perf_counter()
            emitted_messages: list[str] = []
            try:
                with warnings.catch_warnings(record=True) as emitted_warnings:
                    warnings.simplefilter("always")
                    forcefield_outcome = _run_forcefield_backend(
                        complex_mol,
                        backend,
                        settings,
                        index,
                    )
                emitted_messages = [str(item.message) for item in emitted_warnings]
            except ff.ForceFieldError as error:
                emitted_messages.extend(
                    str(item.message) for item in emitted_warnings
                )
                forcefield_seconds = perf_counter() - phase_started
                record.update(
                    status="failed_forcefield",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    traceback=traceback.format_exc(),
                    warnings=emitted_messages,
                    forcefield_seconds=forcefield_seconds,
                    trajectory=_trajectory_payload(error.trajectory),
                )
                diagnostics = getattr(error, "diagnostics", None)
                if diagnostics is not None:
                    record["error_diagnostics"] = json_value(diagnostics)
                if isinstance(diagnostics, ff.ComplexBuildDiagnostics):
                    _record_stage_timings(record, diagnostics)
                error_report = getattr(error, "report", None)
                if isinstance(error_report, ff.ForceFieldValidationReport):
                    validation = _quality_payload(error_report)
                    record["validation"] = validation
                    record["error_quality"] = validation
                elif error_report is not None:
                    record["error_report"] = json_value(error_report)
                    if isinstance(error_report, ff.ForceFieldRunReport):
                        record["complex_optimization_seconds"] = (
                            error_report.elapsed_seconds
                        )
                        routing_report = error_report.routing_report
                        if routing_report is not None:
                            record["routing_report"] = json_value(routing_report)
                if error.trajectory is None:
                    record["output_structure_unavailable_reason"] = (
                        "force-field failure did not preserve a trajectory archive"
                    )
                else:
                    error.trajectory.write(case_dir / "trajectory")
                    try:
                        record.update(
                            _write_last_finite_failure_frame(
                                case_dir,
                                error.trajectory,
                            )
                        )
                    except Exception as output_error:
                        record["output_structure_unavailable_reason"] = (
                            f"{type(output_error).__name__}: {output_error}"
                        )
            else:
                forcefield_seconds = perf_counter() - phase_started
                forcefield_outcome.trajectory.write(case_dir / "trajectory")
                validation = _quality_payload(forcefield_outcome.quality_report)
                routing_report = forcefield_outcome.routing_report
                build_diagnostics = forcefield_outcome.build_diagnostics
                if build_diagnostics is None:
                    build_diagnostics = _complex_build_diagnostics_from_routing(
                        routing_report
                    )
                _record_stage_timings(record, build_diagnostics)
                record.update(
                    forcefield_seconds=forcefield_seconds,
                    complex_optimization_seconds=(
                        forcefield_outcome.optimization.elapsed_seconds
                    ),
                    warnings=emitted_messages,
                    optimization=json_value(forcefield_outcome.optimization),
                    routing_report=json_value(routing_report),
                    forcefield={
                        "requested_forcefield": (
                            forcefield_outcome.requested_forcefield
                        ),
                        "effective_forcefield": (
                            forcefield_outcome.effective_forcefield
                        ),
                        "build": json_value(build_diagnostics),
                        "quality_report": validation,
                    },
                    validation=validation,
                    trajectory=_trajectory_payload(forcefield_outcome.trajectory),
                )
                record["status"] = (
                    "passed"
                    if forcefield_outcome.quality_report.passed
                    else "failed_quality"
                )
                if record["status"] == "passed":
                    complex_mol.write(
                        case_dir / "optimized.mol2",
                        overwrite=True,
                        write_single=True,
                    )
                    complex_mol.write(
                        case_dir / "optimized.sdf",
                        overwrite=True,
                        write_single=True,
                    )
                    record.update(
                        optimized_atom_count=len(complex_mol.atoms),
                        output_frame_index=(
                            forcefield_outcome.trajectory.main.selected_index
                        ),
                        output_frame_role="selected_success_frame",
                        visualization_topology_lossy=False,
                    )
                else:
                    record.update(
                        _write_quality_failure_frame(
                            case_dir,
                            forcefield_outcome.trajectory,
                        )
                    )
                record["phase"] = "complete"
    except Exception as error:
        record.update(
            status="failed_internal",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )

    record["total_seconds"] = perf_counter() - started
    write_json(report_path, record)
    return record


def run_hotpot_case(
    index: int,
    smiles: str,
    output_root_text: str,
    metal: str,
    settings: BenchmarkSettings,
    resume: bool,
) -> dict[str, object]:
    """Run one complete three-stage Hotpot benchmark case."""
    return _run_case(
        index,
        smiles,
        output_root_text,
        metal,
        settings,
        resume,
        "hotpot",
    )


def run_hotpot_auto_case(
    index: int,
    smiles: str,
    output_root_text: str,
    metal: str,
    settings: BenchmarkSettings,
    resume: bool,
) -> dict[str, object]:
    """Run native FAST first and the complete complex fallback when needed."""
    return _run_case(
        index,
        smiles,
        output_root_text,
        metal,
        settings,
        resume,
        "hotpot-auto",
    )


def flatten_record(record: Mapping[str, object]) -> dict[str, object]:
    """Flatten one case into the stable tabular result schema."""
    cbond = record.get("cbond") or {}
    forcefield = record.get("forcefield") or {}
    optimization = record.get("optimization") or {}
    validation = record.get("validation") or {}
    trajectory = record.get("trajectory") or {}
    routing_report = record.get("routing_report") or {}
    return {
        "index": record["index"],
        "backend": record["backend"],
        "workflow": record["workflow"],
        "status": record["status"],
        "phase": record["phase"],
        "smiles": record["smiles"],
        "donor_count": cbond.get("donor_count"),
        "donor_indices": cbond.get("donor_indices"),
        "cbond_path_probability": cbond.get("path_probability"),
        "effective_forcefield": forcefield.get("effective_forcefield"),
        "quality_passed": validation.get("passed"),
        "converged": optimization.get("converged"),
        "termination_reason": optimization.get("termination_reason"),
        "final_energy_kj_mol": optimization.get("final_energy"),
        "best_energy_kj_mol": optimization.get("best_energy"),
        "output_frame_index": record.get("output_frame_index"),
        "output_frame_role": record.get("output_frame_role"),
        "visualization_topology_lossy": record.get("visualization_topology_lossy"),
        "main_frame_count": trajectory.get("main_frame_count"),
        "ligand_build_attempt_count": trajectory.get("ligand_build_attempt_count"),
        "preliminary_attempt_count": trajectory.get("preliminary_attempt_count"),
        "selected_route": routing_report.get("selected_route"),
        "cbond_seconds": record.get("cbond_seconds"),
        "ligand_build_seconds": record.get("ligand_build_seconds"),
        "coordination_restoration_seconds": record.get(
            "coordination_restoration_seconds"
        ),
        "complex_optimization_seconds": record.get(
            "complex_optimization_seconds"
        ),
        "forcefield_seconds": record.get("forcefield_seconds"),
        "total_seconds": record.get("total_seconds"),
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
    }


CASE_RUNNERS = {
    "hotpot": run_hotpot_case,
    "hotpot-auto": run_hotpot_auto_case,
}
