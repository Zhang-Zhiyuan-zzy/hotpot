"""One-case Hotpot CBond and force-field benchmark pipeline."""

from __future__ import annotations

import json
import traceback
import warnings
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional

import numpy as np

from .configuration import BenchmarkSettings
from .io import json_value, write_json


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
    }


def _cbond_payload(result: object) -> dict[str, object]:
    return {
        "donor_indices": result.donor_indices,
        "donor_count": len(result.donor_indices),
        "path_probability": result.path_probability,
        "steps": json_value(result.steps),
        "complex_smiles": result.molecule.smiles,
    }


def _last_finite_main_frame_index(archive: object) -> Optional[int]:
    for frame in reversed(archive.main.frames):
        if np.all(np.isfinite(archive.main.coordinates(frame.index))):
            return int(frame.index)
    return None


def _materialize_main_frame(
    archive: object,
    frame_index: int,
) -> tuple[object, bool]:
    """Build a serializable molecule from an immutable trajectory frame."""
    from hotpot import Molecule

    trajectory = archive.main
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
    output_mol, visualization_topology_lossy = _materialize_main_frame(
        archive,
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
    if frame_index is None:
        return {
            "output_structure_unavailable_reason": (
                "trajectory main branch contains no finite coordinate frame"
            )
        }
    return _write_frame_outputs(
        case_dir,
        archive,
        frame_index,
        output_frame_role="last_finite_failure_frame",
    )


def run_hotpot_case(
    index: int,
    smiles: str,
    output_root_text: str,
    metal: str,
    settings: BenchmarkSettings,
    resume: bool,
) -> dict[str, object]:
    """Run one end-to-end case in a spawn-safe worker process."""
    from hotpot import read_mol
    from hotpot.cheminfo import forcefields as ff
    from hotpot.cheminfo.AImodels.cbond.apply import (
        auto_build_cbond,
        get_cbond_runtime,
    )

    case_dir = Path(output_root_text) / "cases" / f"{index:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    report_path = case_dir / "report.json"
    if resume and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))

    record: dict[str, object] = {
        "index": index,
        "smiles": smiles,
        "backend": "hotpot",
        "status": "running",
        "phase": "read",
        "settings": settings.to_manifest(),
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
            cbond_result = auto_build_cbond(
                ligand,
                metal,
                threshold=settings.cbond_threshold,
                runtime=get_cbond_runtime("cpu"),
                return_details=True,
            )
        except Exception as error:
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
                    forcefield_report = ff.complexes_build(
                        complex_mol,
                        epochs=settings.epochs,
                        steps_per_epoch=settings.steps_per_epoch,
                        max_attempts=settings.max_attempts,
                        candidate_warmup_steps=settings.candidate_warmup_steps,
                        candidate_score_steps=settings.candidate_score_steps,
                        best_candidate_refine_steps=(
                            settings.best_candidate_refine_steps
                        ),
                        ligand_untangling_attempts=(
                            settings.ligand_untangling_attempts
                        ),
                        coordination_restoration_attempts=(
                            settings.coordination_restoration_attempts
                        ),
                        coordination_relaxation_steps=(
                            settings.coordination_relaxation_steps
                        ),
                        complex_untangling_attempts=(
                            settings.complex_untangling_attempts
                        ),
                        timeout=settings.timeout,
                        quality_level=settings.quality_level,
                        seed=settings.seed + index,
                        perturb_sigma=settings.perturb_sigma,
                        save_movie=True,
                        trajectory_start=ff.TrajectoryStart(settings.trajectory_start),
                        trajectory_path=case_dir / "trajectory",
                    )
                emitted_messages = [str(item.message) for item in emitted_warnings]
            except ff.ForceFieldError as error:
                emitted_messages.extend(
                    str(item.message) for item in locals().get("emitted_warnings", ())
                )
                record.update(
                    status="failed_forcefield",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    traceback=traceback.format_exc(),
                    warnings=emitted_messages,
                    forcefield_seconds=perf_counter() - phase_started,
                    trajectory=_trajectory_payload(error.trajectory),
                )
                diagnostics = getattr(error, "diagnostics", None)
                if diagnostics is not None:
                    record["error_diagnostics"] = json_value(diagnostics)
                if isinstance(diagnostics, ff.ComplexBuildDiagnostics):
                    record["ligand_build_seconds"] = (
                        diagnostics.ligand_build_elapsed_seconds
                    )
                    restoration = diagnostics.coordination_restoration
                    if restoration is not None:
                        record["coordination_restoration_seconds"] = (
                            restoration.elapsed_seconds
                        )
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
                if error.trajectory is None:
                    record["output_structure_unavailable_reason"] = (
                        "force-field failure did not preserve a trajectory archive"
                    )
                else:
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
                validation = _quality_payload(forcefield_report.quality_report)
                build_report = forcefield_report.build
                optimization_report = forcefield_report.optimization
                record.update(
                    forcefield_seconds=perf_counter() - phase_started,
                    ligand_build_seconds=(
                        build_report.ligand_build_elapsed_seconds
                    ),
                    coordination_restoration_seconds=(
                        build_report.coordination_restoration.elapsed_seconds
                        if build_report.coordination_restoration is not None
                        else None
                    ),
                    complex_optimization_seconds=(
                        optimization_report.elapsed_seconds
                        if optimization_report is not None
                        else None
                    ),
                    warnings=emitted_messages,
                    optimization=json_value(optimization_report),
                    forcefield={
                        "requested_forcefield": (
                            forcefield_report.requested_forcefield
                        ),
                        "effective_forcefield": (
                            forcefield_report.effective_forcefield
                        ),
                        "build": json_value(forcefield_report.build),
                        "quality_report": validation,
                    },
                    validation=validation,
                    trajectory=_trajectory_payload(forcefield_report.trajectory),
                )
                record["status"] = (
                    "passed"
                    if forcefield_report.quality_report.passed
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
                            forcefield_report.trajectory.main.selected_index
                        ),
                        output_frame_role="selected_success_frame",
                        visualization_topology_lossy=False,
                    )
                else:
                    record.update(
                        _write_frame_outputs(
                            case_dir,
                            forcefield_report.trajectory,
                            forcefield_report.trajectory.main.selected_index,
                            output_frame_role="selected_quality_failure_frame",
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


def flatten_record(record: Mapping[str, object]) -> dict[str, object]:
    """Flatten one case into the stable tabular result schema."""
    cbond = record.get("cbond") or {}
    forcefield = record.get("forcefield") or {}
    optimization = record.get("optimization") or {}
    validation = record.get("validation") or {}
    trajectory = record.get("trajectory") or {}
    return {
        "index": record["index"],
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
}
