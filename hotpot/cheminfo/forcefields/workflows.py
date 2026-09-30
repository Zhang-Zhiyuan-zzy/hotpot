"""Force-field construction and optimization workflow orchestration."""

from __future__ import annotations

import multiprocessing as mp
import warnings
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple, TYPE_CHECKING, cast

from ..obWrappers.native import _native_module
from .acceptance import (
    _format_geometry_checks,
    evaluate_structure_acceptance,
    evaluate_structure_acceptance_at_native_checkpoint,
)
from .backend import (
    _ob_build,
    _resolve_complex_forcefield,
    _resolve_organic_forcefield,
)
from .contracts import (
    AcceptanceLevel,
    Build3DReport,
    BuildAndOptimizeReport,
    ComplexBuildDiagnostics,
    ComplexBuildReport,
    ComplexBuildWarning,
    ForceFieldError,
    ForceFieldAcceptanceEvidence,
    ForceFieldRunReport,
    ForceFieldSetupError,
    ForceFieldSetupReport,
    ForceFieldWorkflowStage,
    ForceFieldValidationReport,
    ForceFieldWorkflowReport,
    GeometryQualityError,
    GeometryQualityWarning,
    OptimizationAlgorithm,
    OptimizationStoppingCriteria,
    StructureAcceptanceThresholds,
    TrajectoryPath,
)
from .coordination import (
    _require_explicit_complex,
    prepare_coordination_geometry,
)
from .native import (
    ComplexOptimizationOptions,
    CoordinationStageOptions,
    FrameDetail,
    OptimizationStoppingOptions,
    create_coordination_session,
    create_optimization_session,
    optimize_complex as _native_optimize_complex,
    restore_coordination as _native_restore_coordination,
    run_complex_workflow_from_input as _native_run_complex_workflow_from_input,
)
from .native_adapters import (
    apply_native_selected_structure,
    coordination_restoration_report,
    forcefield_run_report,
    ingest_native_trajectory,
    native_coordination_offsets,
    native_optimization_offsets,
    native_perturbation_streams,
    native_warning_messages,
)
from .native_packing import ComplexSessionInput, pack_complex_session_input
from .native_reports import (
    ComplexOptimizationResult,
    CoordinationStageResult,
    _coordination_stage_result,
)
from .optimizer import _optimize_working_mol
from .topology import TopologyReference, capture_topology
from .trajectory import (
    ForceFieldTrajectory,
    ForceFieldTrajectoryArchive,
    TrajectoryStart,
)
from .working_copy import _commit_working_copy, _hydrogenated_working_copy, _make_worker_mol
from .workers import (
    _build_ligand_proxies_worker,
    _receive_worker_result,
    _seeded_ob_build_coordinates,
    _validated_worker_coordinates,
)


if TYPE_CHECKING:
    from ..core import Molecule
    from ..obWrappers import _ob_native


__all__ = (
    "build3d",
    "optimize",
    "build_complex3d",
    "optimize_complex",
    "complexes_build",
    "build_and_optimize",
    "auto_optimize",
)


@dataclass(frozen=True)
class _PreparedComplex:
    mol: "Molecule"
    diagnostics: ComplexBuildDiagnostics
    trajectory: ForceFieldTrajectory
    ligand_build_attempts: Tuple[ForceFieldTrajectory, ...] = ()


def _forcefield_acceptance_evidence(
    report: ForceFieldRunReport,
) -> ForceFieldAcceptanceEvidence:
    """Translate the selected numerical frame into acceptance evidence."""
    return {
        "setup_succeeded": report.setup_succeeded,
        "converged": report.converged,
        "epochs_completed": report.epochs_completed,
        "segment_epochs_completed": report.selected_segment_epochs_completed,
        "final_energy": report.best_energy,
        "energy_unit": report.energy_unit,
        "rms_gradient": report.rms_gradient,
        "max_gradient": report.max_gradient,
        "exploded": report.exploded,
        "energy_changes": report.energy_changes,
        "max_displacements": report.max_displacements,
    }


def _warn_failed_acceptance(
    report: ForceFieldValidationReport,
    *,
    prefix: str,
) -> None:
    """Warn while retaining a finite diagnostic structure."""
    if report.passed:
        return
    warnings.warn(
        _format_geometry_checks(prefix, tuple(report.failures)),
        GeometryQualityWarning,
        stacklevel=3,
    )


# Non-committing workflow stages.


def _finalize_trajectory(
    mol: "Molecule",
    trajectory: ForceFieldTrajectory,
    *,
    ligand_build_attempts: Sequence[ForceFieldTrajectory] = (),
    save_movie: bool,
    trajectory_path: Optional[TrajectoryPath],
) -> ForceFieldTrajectoryArchive:
    """Persist and expose one completed trajectory without choosing its frame."""
    archive = ForceFieldTrajectoryArchive(
        main=trajectory,
        ligand_build_attempts=tuple(ligand_build_attempts),
    )
    if trajectory_path is not None:
        archive.write(trajectory_path)
    trajectory.materialize(mol, keep_all=save_movie)
    return archive


def _preserve_failed_trajectory(
    error: ForceFieldError,
    trajectory: ForceFieldTrajectory,
    *,
    ligand_build_attempts: Sequence[ForceFieldTrajectory] = (),
    trajectory_path: Optional[TrajectoryPath],
) -> None:
    """Attach and optionally persist the facts recorded before a failure."""
    archive = ForceFieldTrajectoryArchive(
        main=trajectory,
        ligand_build_attempts=tuple(ligand_build_attempts),
    )
    error.trajectory = archive
    if trajectory_path is not None:
        archive.write(trajectory_path)


def _prepare_complex_working_mol(
    mol: "Molecule",
    *,
    effective_forcefield: str,
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    ligand_untangling_attempts: int = 20,
    timeout: float,
    add_hydrogens: bool,
    seed: Optional[int],
    perturb_sigma: float = 0.5,
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    coordination_geometry: Optional[str],
) -> _PreparedComplex:
    """Run Stage 1 and return built ligand coordinates without restoring bonds."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if ligand_untangling_attempts < 1:
        raise ValueError("ligand_untangling_attempts must be at least 1")
    if min(
        candidate_warmup_steps,
        candidate_score_steps,
        best_candidate_refine_steps,
    ) < 1:
        raise ValueError("all candidate optimization step counts must be at least 1")
    if perturb_sigma < 0.0:
        raise ValueError("perturb_sigma must be non-negative")
    if timeout <= 0.0:
        raise ValueError("timeout must be positive")
    working_mol = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        working_mol,
        start=trajectory_start,
    )
    worker_mol = _make_worker_mol(working_mol)
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_build_ligand_proxies_worker,
        args=(
            worker_mol,
            send_connection,
            max_attempts,
            candidate_warmup_steps,
            candidate_score_steps,
            best_candidate_refine_steps,
            effective_forcefield,
            seed,
            ligand_untangling_attempts,
            perturb_sigma,
            trajectory_start is TrajectoryStart.LIGAND_BUILD,
        ),
    )
    ligand_build_attempts: Tuple[ForceFieldTrajectory, ...] = ()
    try:
        result = _receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=timeout,
            seed=seed,
        )
        ligand_build_attempts = result.ligand_build_attempts
        working_mol.coordinates = _validated_worker_coordinates(
            result,
            expected_atom_count=len(working_mol.atoms),
        )
    except ForceFieldError as error:
        _preserve_failed_trajectory(
            error,
            trajectory,
            ligand_build_attempts=(
                ligand_build_attempts or error.ligand_build_attempts
            ),
            trajectory_path=trajectory_path,
        )
        raise
    diagnostics = cast(ComplexBuildDiagnostics, result.diagnostics)
    for message in diagnostics.warning_messages:
        warnings.warn(message, ComplexBuildWarning, stacklevel=3)
    if coordination_geometry is not None:
        prepare_coordination_geometry(
            working_mol, strategy=coordination_geometry, seed=seed
        )
    return _PreparedComplex(
        mol=working_mol,
        diagnostics=diagnostics,
        trajectory=trajectory,
        ligand_build_attempts=ligand_build_attempts,
    )


def _native_frame_detail(save_movie: bool) -> FrameDetail:
    """Select native diagnostics without changing mandatory stage frames."""
    return FrameDetail.ALL_ATTEMPTS if save_movie else FrameDetail.NONE


def _native_stopping_options(
    criteria: Optional[OptimizationStoppingCriteria],
) -> Optional[OptimizationStoppingOptions]:
    """Translate the public stopping contract to the native stage contract."""
    if criteria is None:
        return None
    return OptimizationStoppingOptions(
        window=criteria.window,
        maximum_energy_change_kj_mol=criteria.maximum_energy_change_kj_mol,
        maximum_atom_displacement_angstrom=(
            criteria.maximum_atom_displacement_angstrom
        ),
        maximum_rms_gradient_kj_mol_angstrom=(
            criteria.maximum_rms_gradient_kj_mol_angstrom
        ),
        maximum_gradient_kj_mol_angstrom=(
            criteria.maximum_gradient_kj_mol_angstrom
        ),
    )


def _coordination_options(
    *,
    effective_forcefield: str,
    attempt_limit: int,
    relaxation_steps: int,
    perturb_sigma: float,
    trajectory_start: TrajectoryStart,
    save_movie: bool,
) -> CoordinationStageOptions:
    return CoordinationStageOptions(
        forcefield=effective_forcefield,
        attempt_limit=attempt_limit,
        relaxation_steps=relaxation_steps,
        perturb_sigma=perturb_sigma,
        trajectory_start=trajectory_start,
        frame_detail=_native_frame_detail(save_movie),
    )


def _optimization_options(
    *,
    effective_forcefield: str,
    algorithm: OptimizationAlgorithm,
    epochs: int,
    steps_per_epoch: int,
    untangling_attempt_limit: int,
    perturb_interval: Optional[int],
    perturb_sigma: float,
    trajectory_start: TrajectoryStart,
    save_movie: bool,
    stopping_criteria: Optional[OptimizationStoppingCriteria],
    increasing_vdw: bool,
    vdw_cutoff_start: float,
    vdw_cutoff_end: float,
) -> ComplexOptimizationOptions:
    return ComplexOptimizationOptions(
        forcefield=effective_forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        untangling_attempt_limit=untangling_attempt_limit,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        trajectory_start=trajectory_start,
        frame_detail=_native_frame_detail(save_movie),
        retain_epoch_history=save_movie,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        stopping=_native_stopping_options(stopping_criteria),
    )


def _native_setup_error(
    error: "_ob_native.ForceFieldSetupError",
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
    diagnostics: Optional[ComplexBuildDiagnostics] = None,
) -> ForceFieldSetupError:
    workflow_stage = error.workflow_stage
    return ForceFieldSetupError(
        str(error),
        ForceFieldSetupReport(
            requested_forcefield=requested_forcefield,
            effective_forcefield=effective_forcefield,
            stage=error.stage,
            workflow_stage=cast(
                Optional[ForceFieldWorkflowStage],
                workflow_stage,
            ),
        ),
        diagnostics=diagnostics,
    )


def _completed_coordination_result(
    error: "_ob_native.ForceFieldSetupError",
) -> Optional[CoordinationStageResult]:
    completed = error.completed_coordination
    return None if completed is None else _coordination_stage_result(completed)


def _prepared_with_coordination_report(
    prepared: _PreparedComplex,
    result: CoordinationStageResult,
) -> _PreparedComplex:
    restoration = coordination_restoration_report(result)
    diagnostics = replace(
        prepared.diagnostics,
        elapsed_seconds=prepared.diagnostics.elapsed_seconds + result.elapsed_seconds,
        warning_messages=(
            prepared.diagnostics.warning_messages + restoration.warning_messages
        ),
        coordination_restoration=restoration,
    )
    for message in restoration.warning_messages:
        warnings.warn(message, ComplexBuildWarning, stacklevel=3)
    return replace(prepared, diagnostics=diagnostics)


def _apply_coordination_result(
    prepared: _PreparedComplex,
    session_input: ComplexSessionInput,
    result: CoordinationStageResult,
) -> _PreparedComplex:
    ingest_native_trajectory(prepared.trajectory, result.trajectory, session_input)
    apply_native_selected_structure(prepared.mol, session_input, result)
    return _prepared_with_coordination_report(prepared, result)


def _native_optimization_report(
    working_mol: "Molecule",
    result: ComplexOptimizationResult,
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
    quality_level: AcceptanceLevel,
    topology_reference: TopologyReference,
    quality_thresholds: Optional[StructureAcceptanceThresholds],
) -> ForceFieldRunReport:
    report = forcefield_run_report(
        result,
        requested_forcefield=requested_forcefield,
        effective_forcefield=effective_forcefield,
    )
    for message in native_warning_messages(result.warning_codes):
        warnings.warn(message, GeometryQualityWarning, stacklevel=3)
    quality_report = evaluate_structure_acceptance_at_native_checkpoint(
        working_mol,
        bond_ring_report=result.final_checkpoint,
        level=quality_level,
        topology_reference=topology_reference,
        forcefield_report=_forcefield_acceptance_evidence(report),
        forcefield_stage="final",
        thresholds=quality_thresholds,
    )
    _warn_failed_acceptance(
        quality_report,
        prefix=(
            "The selected complex optimization frame failed terminal "
            "structure acceptance; retaining it for inspection"
        ),
    )
    return replace(report, quality_report=quality_report)


def _complete_native_optimization(
    working_mol: "Molecule",
    session_input: ComplexSessionInput,
    result: ComplexOptimizationResult,
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
    quality_level: AcceptanceLevel,
    topology_reference: TopologyReference,
    quality_thresholds: Optional[StructureAcceptanceThresholds],
    trajectory: ForceFieldTrajectory,
) -> ForceFieldRunReport:
    ingest_native_trajectory(trajectory, result.trajectory, session_input)
    apply_native_selected_structure(working_mol, session_input, result)
    return _native_optimization_report(
        working_mol,
        result,
        requested_forcefield=requested_forcefield,
        effective_forcefield=effective_forcefield,
        quality_level=quality_level,
        topology_reference=topology_reference,
        quality_thresholds=quality_thresholds,
    )


def _complexes_build_workflow(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    complex_untangling_attempts: int = 30,
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    stopping_criteria: Optional[OptimizationStoppingCriteria] = None,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    """Build, optimize, validate, and atomically commit a complete complex."""
    _require_explicit_complex(mol)
    topology_reference = capture_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    effective_forcefield = _resolve_complex_forcefield(forcefield)
    prepared = _prepare_complex_working_mol(
        mol,
        effective_forcefield=effective_forcefield,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        ligand_untangling_attempts=ligand_untangling_attempts,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        seed=seed,
        perturb_sigma=perturb_sigma,
        trajectory_start=trajectory_start,
        trajectory_path=trajectory_path,
        coordination_geometry=coordination_geometry,
    )
    session_input = pack_complex_session_input(prepared.mol)
    coordination_options = _coordination_options(
        effective_forcefield=effective_forcefield,
        attempt_limit=coordination_restoration_attempts,
        relaxation_steps=coordination_relaxation_steps,
        perturb_sigma=perturb_sigma,
        trajectory_start=trajectory_start,
        save_movie=save_movie,
    )
    optimization_options = _optimization_options(
        effective_forcefield=effective_forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        untangling_attempt_limit=complex_untangling_attempts,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        trajectory_start=trajectory_start,
        save_movie=save_movie,
        stopping_criteria=stopping_criteria,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
    streams = native_perturbation_streams(
        len(prepared.mol.atoms),
        seed,
        coordination_options=coordination_options,
        optimization_options=optimization_options,
    )
    try:
        native_result = _native_run_complex_workflow_from_input(
            session_input,
            streams.coordination,
            streams.untangling,
            streams.optimization,
            coordination_options=coordination_options,
            optimization_options=optimization_options,
        )
        ingest_native_trajectory(
            prepared.trajectory,
            native_result.trajectory,
            session_input,
        )
        apply_native_selected_structure(prepared.mol, session_input, native_result)
        prepared = _prepared_with_coordination_report(
            prepared,
            native_result.coordination,
        )
        optimization_report = _native_optimization_report(
            prepared.mol,
            native_result.optimization,
            requested_forcefield=forcefield,
            effective_forcefield=effective_forcefield,
            quality_level=quality_level,
            topology_reference=topology_reference,
            quality_thresholds=quality_thresholds,
        )
    except _native_module().ForceFieldSetupError as native_error:
        completed_coordination = _completed_coordination_result(native_error)
        if completed_coordination is not None:
            ingest_native_trajectory(
                prepared.trajectory,
                completed_coordination.trajectory,
                session_input,
            )
            prepared = _prepared_with_coordination_report(
                prepared,
                completed_coordination,
            )
        error = _native_setup_error(
            native_error,
            requested_forcefield=forcefield,
            effective_forcefield=effective_forcefield,
            diagnostics=prepared.diagnostics,
        )
        _preserve_failed_trajectory(
            error,
            prepared.trajectory,
            ligand_build_attempts=prepared.ligand_build_attempts,
            trajectory_path=trajectory_path,
        )
        raise error from native_error
    except ForceFieldError as error:
        _preserve_failed_trajectory(
            error,
            prepared.trajectory,
            ligand_build_attempts=prepared.ligand_build_attempts,
            trajectory_path=trajectory_path,
        )
        raise
    trajectory_archive = _finalize_trajectory(
        prepared.mol,
        prepared.trajectory,
        ligand_build_attempts=prepared.ligand_build_attempts,
        save_movie=save_movie,
        trajectory_path=trajectory_path,
    )
    optimization_report = replace(
        optimization_report,
        trajectory=trajectory_archive,
    )
    report = ComplexBuildReport(
        requested_forcefield=forcefield,
        effective_forcefield=effective_forcefield,
        build=prepared.diagnostics,
        optimization=optimization_report,
        quality_report=optimization_report.quality_report,
        trajectory=trajectory_archive,
    )
    _commit_working_copy(mol, prepared.mol)
    return report


# Public force-field and coordination interfaces, ordered from primitives to workflows.



def _build3d_workflow(
    mol: "Molecule",
    *,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    timeout: float = 1000.0,
) -> Build3DReport:
    """Generate initial 3D coordinates with OBBuilder, without optimization."""
    topology_reference = capture_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    initial_hydrogens = len(mol.hydrogens)
    working_mol = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    if seed is None:
        _ob_build(working_mol)
    else:
        working_mol.coordinates = _seeded_ob_build_coordinates(
            working_mol,
            seed,
            timeout=timeout,
        )
    quality_report = evaluate_structure_acceptance(
        working_mol,
        level="off",
        topology_reference=topology_reference,
    )
    if not quality_report.passed:
        raise GeometryQualityError(quality_report)
    report = Build3DReport(
        atom_count=len(working_mol.atoms),
        added_hydrogen_count=len(working_mol.hydrogens) - initial_hydrogens,
        quality_report=quality_report,
    )
    _commit_working_copy(mol, working_mol)
    return report


def build3d(
    mol: "Molecule",
    *,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    timeout: float = 1000.0,
) -> Build3DReport:
    """Generate initial 3D coordinates with OBBuilder, without optimization."""
    return _build3d_workflow(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
        timeout=timeout,
    )


def optimize(
    mol: "Molecule",
    forcefield: Optional[str] = "UFF",
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 1,
    steps_per_epoch: int = 100,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    stopping_criteria: Optional[OptimizationStoppingCriteria] = None,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.FINAL_OPTIMIZATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
) -> ForceFieldRunReport:
    """Run the ordinary Open Babel optimizer, including on explicit complexes."""
    topology_reference = capture_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    working_mol = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        working_mol,
        start=trajectory_start,
    )
    effective_forcefield = _resolve_organic_forcefield(forcefield)
    try:
        report = _optimize_working_mol(
            working_mol,
            requested_forcefield=forcefield,
            effective_forcefield=effective_forcefield,
            algorithm=algorithm,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            seed=seed,
            perturb_interval=perturb_interval,
            perturb_sigma=perturb_sigma,
            retain_epoch_history=save_movie,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
            trajectory=trajectory,
            stopping_criteria=stopping_criteria,
        )
        quality_report = evaluate_structure_acceptance(
            working_mol,
            level=quality_level,
            topology_reference=topology_reference,
            forcefield_report=_forcefield_acceptance_evidence(report),
            forcefield_stage="final",
            thresholds=quality_thresholds,
        )
        _warn_failed_acceptance(
            quality_report,
            prefix=(
                "The selected optimization frame failed terminal structure "
                "acceptance; retaining it for inspection"
            ),
        )
        report = replace(report, quality_report=quality_report)
    except ForceFieldError as error:
        _preserve_failed_trajectory(
            error,
            trajectory,
            trajectory_path=trajectory_path,
        )
        raise
    trajectory_archive = _finalize_trajectory(
        working_mol,
        trajectory,
        save_movie=save_movie,
        trajectory_path=trajectory_path,
    )
    report = replace(report, trajectory=trajectory_archive)
    _commit_working_copy(mol, working_mol)
    return report


def _build_complex3d_workflow(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    """Build ligand proxies and restore the complete complex topology."""
    _require_explicit_complex(mol)
    topology_reference = capture_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    effective_forcefield = _resolve_complex_forcefield(forcefield)
    prepared = _prepare_complex_working_mol(
        mol,
        effective_forcefield=effective_forcefield,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        ligand_untangling_attempts=ligand_untangling_attempts,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        seed=seed,
        perturb_sigma=perturb_sigma,
        trajectory_start=trajectory_start,
        trajectory_path=trajectory_path,
        coordination_geometry=coordination_geometry,
    )
    session_input = pack_complex_session_input(prepared.mol)
    session = create_coordination_session(session_input)
    options = _coordination_options(
        effective_forcefield=effective_forcefield,
        attempt_limit=coordination_restoration_attempts,
        relaxation_steps=coordination_relaxation_steps,
        perturb_sigma=perturb_sigma,
        trajectory_start=trajectory_start,
        save_movie=save_movie,
    )
    offsets = native_coordination_offsets(
        len(prepared.mol.atoms),
        seed,
        options=options,
    )
    try:
        native_result = _native_restore_coordination(
            session,
            offsets,
            options=options,
        )
        prepared = _apply_coordination_result(
            prepared,
            session_input,
            native_result,
        )
    except _native_module().ForceFieldSetupError as native_error:
        error = _native_setup_error(
            native_error,
            requested_forcefield=forcefield,
            effective_forcefield=effective_forcefield,
            diagnostics=prepared.diagnostics,
        )
        _preserve_failed_trajectory(
            error,
            prepared.trajectory,
            ligand_build_attempts=prepared.ligand_build_attempts,
            trajectory_path=trajectory_path,
        )
        raise error from native_error
    except ForceFieldError as error:
        _preserve_failed_trajectory(
            error,
            prepared.trajectory,
            ligand_build_attempts=prepared.ligand_build_attempts,
            trajectory_path=trajectory_path,
        )
        raise
    quality_report = evaluate_structure_acceptance(
        prepared.mol,
        level="off",
        topology_reference=topology_reference,
    )
    if not quality_report.passed:
        error = GeometryQualityError(quality_report)
        _preserve_failed_trajectory(
            error,
            prepared.trajectory,
            ligand_build_attempts=prepared.ligand_build_attempts,
            trajectory_path=trajectory_path,
        )
        raise error
    trajectory_archive = _finalize_trajectory(
        prepared.mol,
        prepared.trajectory,
        ligand_build_attempts=prepared.ligand_build_attempts,
        save_movie=save_movie,
        trajectory_path=trajectory_path,
    )
    report = ComplexBuildReport(
        requested_forcefield=forcefield,
        effective_forcefield=effective_forcefield,
        build=prepared.diagnostics,
        optimization=None,
        quality_report=quality_report,
        trajectory=trajectory_archive,
    )
    _commit_working_copy(mol, prepared.mol)
    return report


def build_complex3d(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    candidate_count: Optional[int] = None,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    """Build ligand proxies and restore the complete complex topology.

    ``candidate_count`` is reserved and currently has no effect.  The current
    workflow builds one ligand starting geometry.  A future multi-conformer
    implementation will generate independent starting conformers, optimize
    and gate them uniformly, deduplicate or cluster them by geometry, rank
    them using topology, geometry, and energy evidence, and refine the
    selected conformer.
    """
    _ = candidate_count
    return _build_complex3d_workflow(
        mol,
        forcefield,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        ligand_untangling_attempts=ligand_untangling_attempts,
        coordination_restoration_attempts=coordination_restoration_attempts,
        coordination_relaxation_steps=coordination_relaxation_steps,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        seed=seed,
        perturb_sigma=perturb_sigma,
        save_movie=save_movie,
        trajectory_start=trajectory_start,
        trajectory_path=trajectory_path,
        coordination_geometry=coordination_geometry,
    )


def optimize_complex(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    complex_untangling_attempts: int = 30,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    stopping_criteria: Optional[OptimizationStoppingCriteria] = None,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.COMPLEX_UNTANGLING,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
) -> ForceFieldRunReport:
    """Optimize existing complex coordinates with the complex force-field policy."""
    _require_explicit_complex(mol)
    topology_reference = capture_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    working_mol = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        working_mol,
        start=trajectory_start,
    )
    effective_forcefield = _resolve_complex_forcefield(forcefield)
    session_input = pack_complex_session_input(working_mol)
    session = create_optimization_session(session_input)
    options = _optimization_options(
        effective_forcefield=effective_forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        untangling_attempt_limit=complex_untangling_attempts,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        trajectory_start=trajectory_start,
        save_movie=save_movie,
        stopping_criteria=stopping_criteria,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
    offsets = native_optimization_offsets(
        len(working_mol.atoms),
        seed,
        options=options,
    )
    try:
        native_result = _native_optimize_complex(
            session,
            offsets.untangling,
            offsets.optimization,
            options=options,
        )
        report = _complete_native_optimization(
            working_mol,
            session_input,
            native_result,
            requested_forcefield=forcefield,
            effective_forcefield=effective_forcefield,
            quality_level=quality_level,
            topology_reference=topology_reference,
            quality_thresholds=quality_thresholds,
            trajectory=trajectory,
        )
    except _native_module().ForceFieldSetupError as native_error:
        error = _native_setup_error(
            native_error,
            requested_forcefield=forcefield,
            effective_forcefield=effective_forcefield,
        )
        _preserve_failed_trajectory(
            error,
            trajectory,
            trajectory_path=trajectory_path,
        )
        raise error from native_error
    except ForceFieldError as error:
        _preserve_failed_trajectory(
            error,
            trajectory,
            trajectory_path=trajectory_path,
        )
        raise
    trajectory_archive = _finalize_trajectory(
        working_mol,
        trajectory,
        save_movie=save_movie,
        trajectory_path=trajectory_path,
    )
    report = replace(report, trajectory=trajectory_archive)
    _commit_working_copy(mol, working_mol)
    return report


def _build_and_optimize_workflow(
    mol: "Molecule",
    forcefield: Optional[str] = "UFF",
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    timeout: float = 1000.0,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    stopping_criteria: Optional[OptimizationStoppingCriteria] = None,
    save_movie: bool = False,
    trajectory_start: Optional[TrajectoryStart] = None,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    complex_untangling_attempts: int = 30,
    coordination_geometry: Optional[str] = None,
) -> ForceFieldWorkflowReport:
    """Build and optimize through the organic or complex workflow."""
    if mol.has_metal:
        return _complexes_build_workflow(
            mol,
            forcefield,
            algorithm=algorithm,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            max_attempts=max_attempts,
            candidate_warmup_steps=candidate_warmup_steps,
            candidate_score_steps=candidate_score_steps,
            best_candidate_refine_steps=best_candidate_refine_steps,
            ligand_untangling_attempts=ligand_untangling_attempts,
            coordination_restoration_attempts=coordination_restoration_attempts,
            coordination_relaxation_steps=coordination_relaxation_steps,
            complex_untangling_attempts=complex_untangling_attempts,
            timeout=timeout,
            add_hydrogens=add_hydrogens,
            quality_level=quality_level,
            quality_thresholds=quality_thresholds,
            seed=seed,
            perturb_interval=perturb_interval,
            perturb_sigma=perturb_sigma,
            stopping_criteria=stopping_criteria,
            save_movie=save_movie,
            trajectory_start=(
                trajectory_start
                if trajectory_start is not None
                else TrajectoryStart.COORDINATION_RESTORATION
            ),
            trajectory_path=trajectory_path,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
            coordination_geometry=coordination_geometry,
        )

    working_mol = _hydrogenated_working_copy(mol, add_hydrogens=False)
    build_report = _build3d_workflow(
        working_mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
        timeout=timeout,
    )
    optimization_report = optimize(
        working_mol,
        forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        add_hydrogens=False,
        quality_level=quality_level,
        quality_thresholds=quality_thresholds,
        seed=seed,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        stopping_criteria=stopping_criteria,
        save_movie=save_movie,
        trajectory_start=(
            trajectory_start
            if trajectory_start is not None
            else TrajectoryStart.FINAL_OPTIMIZATION
        ),
        trajectory_path=trajectory_path,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
    _commit_working_copy(mol, working_mol)
    return BuildAndOptimizeReport(
        requested_forcefield=optimization_report.requested_forcefield,
        effective_forcefield=optimization_report.effective_forcefield,
        build=build_report,
        optimization=optimization_report,
        quality_report=optimization_report.quality_report,
        trajectory=optimization_report.trajectory,
    )


def complexes_build(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    candidate_count: Optional[int] = None,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    complex_untangling_attempts: int = 30,
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    stopping_criteria: Optional[OptimizationStoppingCriteria] = None,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    """Build, optimize, validate, and atomically commit a complete complex.

    ``candidate_count`` is reserved and currently has no effect.  The current
    workflow builds one ligand starting geometry.  A future multi-conformer
    implementation will generate independent starting conformers, optimize
    and gate them uniformly, deduplicate or cluster them by geometry, rank
    them using topology, geometry, and energy evidence, and refine the
    selected conformer.
    """
    _ = candidate_count
    return _complexes_build_workflow(
        mol,
        forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        ligand_untangling_attempts=ligand_untangling_attempts,
        coordination_restoration_attempts=coordination_restoration_attempts,
        coordination_relaxation_steps=coordination_relaxation_steps,
        complex_untangling_attempts=complex_untangling_attempts,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        quality_level=quality_level,
        quality_thresholds=quality_thresholds,
        seed=seed,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        stopping_criteria=stopping_criteria,
        save_movie=save_movie,
        trajectory_start=trajectory_start,
        trajectory_path=trajectory_path,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        coordination_geometry=coordination_geometry,
    )


def build_and_optimize(
    mol: "Molecule",
    forcefield: Optional[str] = "UFF",
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    timeout: float = 1000.0,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    stopping_criteria: Optional[OptimizationStoppingCriteria] = None,
    save_movie: bool = False,
    trajectory_start: Optional[TrajectoryStart] = None,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    candidate_count: Optional[int] = None,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    complex_untangling_attempts: int = 30,
    coordination_geometry: Optional[str] = None,
) -> ForceFieldWorkflowReport:
    """Build and optimize through the organic or complex workflow.

    ``candidate_count`` is reserved and currently has no effect.  The current
    complex workflow builds one ligand starting geometry.  A future
    multi-conformer implementation will generate independent starting
    conformers, optimize and gate them uniformly, deduplicate or cluster them
    by geometry, rank them using topology, geometry, and energy evidence, and
    refine the selected conformer.
    """
    _ = candidate_count
    return _build_and_optimize_workflow(
        mol,
        forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        add_hydrogens=add_hydrogens,
        quality_level=quality_level,
        quality_thresholds=quality_thresholds,
        seed=seed,
        timeout=timeout,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        stopping_criteria=stopping_criteria,
        save_movie=save_movie,
        trajectory_start=trajectory_start,
        trajectory_path=trajectory_path,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        ligand_untangling_attempts=ligand_untangling_attempts,
        coordination_restoration_attempts=coordination_restoration_attempts,
        coordination_relaxation_steps=coordination_relaxation_steps,
        complex_untangling_attempts=complex_untangling_attempts,
        coordination_geometry=coordination_geometry,
    )


def auto_optimize(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    complex_untangling_attempts: int = 30,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    stopping_criteria: Optional[OptimizationStoppingCriteria] = None,
    save_movie: bool = False,
    trajectory_start: Optional[TrajectoryStart] = None,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
) -> ForceFieldRunReport:
    """Optimize existing coordinates through the appropriate workflow."""
    if mol.has_metal:
        return optimize_complex(
            mol,
            forcefield,
            algorithm=algorithm,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            complex_untangling_attempts=complex_untangling_attempts,
            add_hydrogens=add_hydrogens,
            quality_level=quality_level,
            quality_thresholds=quality_thresholds,
            seed=seed,
            perturb_interval=perturb_interval,
            perturb_sigma=perturb_sigma,
            stopping_criteria=stopping_criteria,
            save_movie=save_movie,
            trajectory_start=(
                trajectory_start
                if trajectory_start is not None
                else TrajectoryStart.COMPLEX_UNTANGLING
            ),
            trajectory_path=trajectory_path,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
        )
    return optimize(
        mol,
        forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        add_hydrogens=add_hydrogens,
        quality_level=quality_level,
        quality_thresholds=quality_thresholds,
        seed=seed,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        stopping_criteria=stopping_criteria,
        save_movie=save_movie,
        trajectory_start=(
            trajectory_start
            if trajectory_start is not None
            else TrajectoryStart.FINAL_OPTIMIZATION
        ),
        trajectory_path=trajectory_path,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
