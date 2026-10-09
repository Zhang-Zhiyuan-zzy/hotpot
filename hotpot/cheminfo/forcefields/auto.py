"""Automatic force-field route selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Optional, TYPE_CHECKING

import numpy as np

from ..obWrappers import build as _native_build
from .attempts import _run_ordinary_optimization_attempt
from .backend import _resolve_complex_forcefield
from .contracts import (
    AcceptanceLevel,
    ComplexBuildDiagnostics,
    ConvergenceLevel,
    DEFAULT_CONVERGENCE_LEVEL,
    ForceFieldError,
    ForceFieldRunReport,
    OptimizationAlgorithm,
    OptimizationAttemptReport,
    OptimizationAttemptStatus,
    OptimizationRoute,
    OptimizationRoutingReport,
    OptimizationStoppingCriteria,
    StructureAcceptanceThresholds,
    TrajectoryPath,
)
from .coordination import _require_explicit_complex
from .topology import capture_topology
from .trajectory import (
    ForceFieldTrajectory,
    ForceFieldTrajectoryArchive,
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)
from .working_copy import _commit_working_copy, _hydrogenated_working_copy
from .workflows import _complexes_build_workflow, _finalize_trajectory, optimize


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("auto_optimize",)


_NUMERICAL_FAILURE_REASONS = frozenset({
    "nonfinite_coordinates",
    "nonfinite_energy",
    "nonfinite_gradients",
    "explosion_detected",
})


@dataclass(frozen=True)
class _NativeFastBuildAttempt:
    """Uncommitted native build-and-optimize result."""

    mol: "Molecule"
    trajectory: ForceFieldTrajectory
    report: Optional[ForceFieldRunReport]
    provenance: OptimizationAttemptReport


def _attempt_status(report: ForceFieldRunReport) -> OptimizationAttemptStatus:
    if report.termination_reason in _NUMERICAL_FAILURE_REASONS:
        return OptimizationAttemptStatus.NUMERICAL_FAILURE
    quality_report = report.quality_report
    if quality_report is None:
        raise RuntimeError("The optimizer omitted its quality report")
    if quality_report.passed:
        return OptimizationAttemptStatus.ACCEPTED
    return OptimizationAttemptStatus.QUALITY_REJECTED


def _optimization_attempt_report(
    report: ForceFieldRunReport,
    route: OptimizationRoute,
    *,
    build_succeeded: Optional[bool] = None,
    build_diagnostics: Optional[ComplexBuildDiagnostics] = None,
    elapsed_seconds: float = 0.0,
) -> OptimizationAttemptReport:
    return OptimizationAttemptReport(
        route=route,
        status=_attempt_status(report),
        quality_report=report.quality_report,
        termination_reason=report.termination_reason,
        build_succeeded=build_succeeded,
        build_diagnostics=build_diagnostics,
        elapsed_seconds=elapsed_seconds,
    )


def _routing_report(
    *attempts: OptimizationAttemptReport,
    selected_route: OptimizationRoute,
) -> OptimizationRoutingReport:
    return OptimizationRoutingReport(
        attempts=tuple(attempts),
        selected_route=selected_route,
    )


def _record_quick_build_frame(
    trajectory: ForceFieldTrajectory,
    mol: "Molecule",
    event: TrajectoryEvent,
) -> None:
    if trajectory.records(TrajectoryStage.LIGAND_BUILD):
        trajectory.record_molecule(
            mol,
            stage=TrajectoryStage.LIGAND_BUILD,
            event=event,
        )


def _run_native_fast_build_attempt(
    mol: "Molecule",
    forcefield: Optional[str],
    *,
    algorithm: OptimizationAlgorithm,
    epochs: int,
    steps_per_epoch: int,
    add_hydrogens: bool,
    quality_level: AcceptanceLevel,
    quality_thresholds: Optional[StructureAcceptanceThresholds],
    seed: Optional[int],
    perturb_interval: Optional[int],
    perturb_sigma: float,
    stopping_criteria: Optional[OptimizationStoppingCriteria],
    save_movie: bool,
    trajectory_start: TrajectoryStart,
    increasing_vdw: bool,
    vdw_cutoff_start: float,
    vdw_cutoff_end: float,
) -> _NativeFastBuildAttempt:
    """Run direct native build and FAST optimization without committing."""
    started = perf_counter()
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
    _record_quick_build_frame(trajectory, working_mol, TrajectoryEvent.INITIAL)
    build_report = _native_build(working_mol)
    if not build_report.succeeded:
        return _NativeFastBuildAttempt(
            mol=working_mol,
            trajectory=trajectory,
            report=None,
            provenance=OptimizationAttemptReport(
                route=OptimizationRoute.NATIVE_FAST,
                status=OptimizationAttemptStatus.BUILD_FAILED,
                build_succeeded=False,
                message="obWrappers.build returned an unsuccessful report",
                elapsed_seconds=perf_counter() - started,
            ),
        )
    _record_quick_build_frame(
        trajectory,
        working_mol,
        TrajectoryEvent.BUILD_COMPLETE,
    )
    if not np.all(np.isfinite(working_mol.coordinates)):
        return _NativeFastBuildAttempt(
            mol=working_mol,
            trajectory=trajectory,
            report=None,
            provenance=OptimizationAttemptReport(
                route=OptimizationRoute.NATIVE_FAST,
                status=OptimizationAttemptStatus.NUMERICAL_FAILURE,
                build_succeeded=True,
                termination_reason="nonfinite_coordinates",
                message="obWrappers.build returned non-finite coordinates",
                elapsed_seconds=perf_counter() - started,
            ),
        )
    report = _run_ordinary_optimization_attempt(
        working_mol,
        requested_forcefield=forcefield,
        effective_forcefield=_resolve_complex_forcefield(forcefield),
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
        quality_level=quality_level,
        topology_reference=topology_reference,
        quality_thresholds=quality_thresholds,
        stopping_criteria=stopping_criteria,
        convergence_level=ConvergenceLevel.FAST,
    )
    return _NativeFastBuildAttempt(
        mol=working_mol,
        trajectory=trajectory,
        report=report,
        provenance=_optimization_attempt_report(
            report,
            OptimizationRoute.NATIVE_FAST,
            build_succeeded=True,
            elapsed_seconds=perf_counter() - started,
        ),
    )


def _accept_native_fast_attempt(
    mol: "Molecule",
    attempt: _NativeFastBuildAttempt,
    *,
    save_movie: bool,
    trajectory_path: Optional[TrajectoryPath],
) -> ForceFieldRunReport:
    report = attempt.report
    if report is None:
        raise RuntimeError("An accepted native attempt omitted its optimization report")
    archive = _finalize_trajectory(
        attempt.mol,
        attempt.trajectory,
        save_movie=save_movie,
        trajectory_path=trajectory_path,
    )
    _commit_working_copy(mol, attempt.mol)
    return replace(
        report,
        trajectory=archive,
        routing_report=_routing_report(
            attempt.provenance,
            selected_route=OptimizationRoute.NATIVE_FAST,
        ),
    )


def _archive_with_preliminary_attempt(
    archive: ForceFieldTrajectoryArchive,
    preliminary: ForceFieldTrajectory,
) -> ForceFieldTrajectoryArchive:
    return ForceFieldTrajectoryArchive(
        main=archive.main,
        ligand_build_attempts=archive.ligand_build_attempts,
        preliminary_attempts=(preliminary,) + archive.preliminary_attempts,
    )


def _fallback_to_complex_workflow(
    mol: "Molecule",
    preliminary: _NativeFastBuildAttempt,
    forcefield: Optional[str],
    *,
    algorithm: OptimizationAlgorithm,
    epochs: int,
    steps_per_epoch: int,
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    ligand_untangling_attempts: int,
    coordination_restoration_attempts: int,
    coordination_relaxation_steps: int,
    complex_untangling_attempts: int,
    timeout: float,
    add_hydrogens: bool,
    quality_level: AcceptanceLevel,
    quality_thresholds: Optional[StructureAcceptanceThresholds],
    seed: Optional[int],
    perturb_interval: Optional[int],
    perturb_sigma: float,
    stopping_criteria: Optional[OptimizationStoppingCriteria],
    convergence_level: ConvergenceLevel,
    save_movie: bool,
    trajectory_start: TrajectoryStart,
    trajectory_path: Optional[TrajectoryPath],
    increasing_vdw: bool,
    vdw_cutoff_start: float,
    vdw_cutoff_end: float,
    coordination_geometry: Optional[str],
) -> ForceFieldRunReport:
    """Discard the quick candidate and run all three complex stages."""
    fallback_mol = _hydrogenated_working_copy(
        mol,
        add_hydrogens=False,
        seed=seed,
    )
    try:
        workflow_report = _complexes_build_workflow(
            fallback_mol,
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
            convergence_level=convergence_level,
            save_movie=save_movie,
            trajectory_start=trajectory_start,
            trajectory_path=None,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
            coordination_geometry=coordination_geometry,
        )
    except ForceFieldError as error:
        if error.trajectory is not None:
            error.trajectory = _archive_with_preliminary_attempt(
                error.trajectory,
                preliminary.trajectory,
            )
            if trajectory_path is not None:
                error.trajectory.write(trajectory_path)
        raise
    optimization_report = workflow_report.optimization
    if optimization_report is None or workflow_report.trajectory is None:
        raise RuntimeError("The complex workflow omitted its optimization result")
    archive = _archive_with_preliminary_attempt(
        workflow_report.trajectory,
        preliminary.trajectory,
    )
    if trajectory_path is not None:
        archive.write(trajectory_path)
    _commit_working_copy(mol, fallback_mol)
    fallback_attempt = _optimization_attempt_report(
        optimization_report,
        OptimizationRoute.COMPLEX_WORKFLOW,
        build_succeeded=True,
        build_diagnostics=workflow_report.build,
        elapsed_seconds=(
            workflow_report.build.elapsed_seconds
            + optimization_report.elapsed_seconds
        ),
    )
    return replace(
        optimization_report,
        trajectory=archive,
        routing_report=_routing_report(
            preliminary.provenance,
            fallback_attempt,
            selected_route=OptimizationRoute.COMPLEX_WORKFLOW,
        ),
    )


def auto_optimize(
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
    convergence_level: ConvergenceLevel = DEFAULT_CONVERGENCE_LEVEL,
    save_movie: bool = False,
    trajectory_start: Optional[TrajectoryStart] = None,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    coordination_geometry: Optional[str] = None,
) -> ForceFieldRunReport:
    """Use native FAST first, then the complete complex workflow if needed."""
    if not mol.has_metal:
        report = optimize(
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
            convergence_level=convergence_level,
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
        attempt = _optimization_attempt_report(
            report,
            OptimizationRoute.ORDINARY,
        )
        return replace(
            report,
            routing_report=_routing_report(
                attempt,
                selected_route=OptimizationRoute.ORDINARY,
            ),
        )

    _require_explicit_complex(mol)
    selected_trajectory_start = (
        trajectory_start
        if trajectory_start is not None
        else TrajectoryStart.LIGAND_BUILD
    )
    preliminary = _run_native_fast_build_attempt(
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
        trajectory_start=selected_trajectory_start,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
    if preliminary.provenance.status is OptimizationAttemptStatus.ACCEPTED:
        return _accept_native_fast_attempt(
            mol,
            preliminary,
            save_movie=save_movie,
            trajectory_path=trajectory_path,
        )
    return _fallback_to_complex_workflow(
        mol,
        preliminary,
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
        convergence_level=convergence_level,
        save_movie=save_movie,
        trajectory_start=selected_trajectory_start,
        trajectory_path=trajectory_path,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        coordination_geometry=coordination_geometry,
    )
