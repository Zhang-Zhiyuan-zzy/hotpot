"""Automatic selection between ordinary and complex force-field workflows."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, TYPE_CHECKING

from .attempts import _run_ordinary_optimization_attempt
from .backend import _resolve_complex_forcefield
from .contracts import (
    AcceptanceLevel,
    ConvergenceLevel,
    DEFAULT_CONVERGENCE_LEVEL,
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
    TrajectoryStart,
)
from .working_copy import _commit_working_copy, _hydrogenated_working_copy
from .workflows import _finalize_trajectory, optimize, optimize_complex


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("auto_optimize",)


_FAST_FIRST_QUALITY_LEVELS = frozenset({"standard", "strict"})
_NUMERICAL_FAILURE_REASONS = frozenset({
    "nonfinite_coordinates",
    "nonfinite_energy",
    "nonfinite_gradients",
    "explosion_detected",
})


def _attempt_report(
    report: ForceFieldRunReport,
    route: OptimizationRoute,
) -> OptimizationAttemptReport:
    quality_report = report.quality_report
    if quality_report is None:
        raise RuntimeError("The optimizer omitted its quality report")
    if report.termination_reason in _NUMERICAL_FAILURE_REASONS:
        status = OptimizationAttemptStatus.NUMERICAL_FAILURE
    elif quality_report.passed:
        status = OptimizationAttemptStatus.ACCEPTED
    else:
        status = OptimizationAttemptStatus.QUALITY_REJECTED
    return OptimizationAttemptReport(
        route=route,
        status=status,
        quality_report=quality_report,
        termination_reason=report.termination_reason,
    )


def _routing_report(
    *attempts: OptimizationAttemptReport,
    selected_route: OptimizationRoute,
) -> OptimizationRoutingReport:
    return OptimizationRoutingReport(
        attempts=tuple(attempts),
        selected_route=selected_route,
    )


def _direct_auto_result(
    report: ForceFieldRunReport,
    route: OptimizationRoute,
) -> ForceFieldRunReport:
    attempt = _attempt_report(report, route)
    return replace(
        report,
        routing_report=_routing_report(attempt, selected_route=route),
    )


def _run_fast_complex_attempt(
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
) -> tuple["Molecule", ForceFieldTrajectory, ForceFieldRunReport]:
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
    return working_mol, trajectory, report


def _accept_fast_result(
    mol: "Molecule",
    working_mol: "Molecule",
    trajectory: ForceFieldTrajectory,
    report: ForceFieldRunReport,
    attempt: OptimizationAttemptReport,
    *,
    save_movie: bool,
    trajectory_path: Optional[TrajectoryPath],
) -> ForceFieldRunReport:
    archive = _finalize_trajectory(
        working_mol,
        trajectory,
        save_movie=save_movie,
        trajectory_path=trajectory_path,
    )
    _commit_working_copy(mol, working_mol)
    return replace(
        report,
        trajectory=archive,
        routing_report=_routing_report(
            attempt,
            selected_route=OptimizationRoute.ORDINARY_FAST,
        ),
    )


def _fallback_to_complex(
    mol: "Molecule",
    preliminary_trajectory: ForceFieldTrajectory,
    preliminary_attempt: OptimizationAttemptReport,
    forcefield: Optional[str],
    *,
    algorithm: OptimizationAlgorithm,
    epochs: int,
    steps_per_epoch: int,
    complex_untangling_attempts: int,
    add_hydrogens: bool,
    quality_level: AcceptanceLevel,
    quality_thresholds: Optional[StructureAcceptanceThresholds],
    seed: Optional[int],
    perturb_interval: Optional[int],
    perturb_sigma: float,
    stopping_criteria: Optional[OptimizationStoppingCriteria],
    save_movie: bool,
    trajectory_start: TrajectoryStart,
    trajectory_path: Optional[TrajectoryPath],
    increasing_vdw: bool,
    vdw_cutoff_start: float,
    vdw_cutoff_end: float,
) -> ForceFieldRunReport:
    fallback_mol = _hydrogenated_working_copy(
        mol,
        add_hydrogens=False,
        seed=seed,
    )
    fallback_report = optimize_complex(
        fallback_mol,
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
        convergence_level=ConvergenceLevel.FAST,
        save_movie=save_movie,
        trajectory_start=trajectory_start,
        trajectory_path=None,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
    fallback_archive = fallback_report.trajectory
    if fallback_archive is None:
        raise RuntimeError("The complex optimizer omitted its trajectory")
    archive = ForceFieldTrajectoryArchive(
        main=fallback_archive.main,
        ligand_build_attempts=fallback_archive.ligand_build_attempts,
        preliminary_optimization_attempts=(preliminary_trajectory,),
    )
    if trajectory_path is not None:
        archive.write(trajectory_path)
    _commit_working_copy(mol, fallback_mol)
    fallback_attempt = _attempt_report(
        fallback_report,
        OptimizationRoute.COMPLEX,
    )
    return replace(
        fallback_report,
        trajectory=archive,
        routing_report=_routing_report(
            preliminary_attempt,
            fallback_attempt,
            selected_route=OptimizationRoute.COMPLEX,
        ),
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
    convergence_level: ConvergenceLevel = DEFAULT_CONVERGENCE_LEVEL,
    save_movie: bool = False,
    trajectory_start: Optional[TrajectoryStart] = None,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
) -> ForceFieldRunReport:
    """Optimize existing coordinates with a gated fast complex route."""
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
        return _direct_auto_result(report, OptimizationRoute.ORDINARY)

    _require_explicit_complex(mol)
    complex_trajectory_start = (
        trajectory_start
        if trajectory_start is not None
        else TrajectoryStart.COMPLEX_UNTANGLING
    )
    fast_first = (
        convergence_level is ConvergenceLevel.FAST
        and quality_level in _FAST_FIRST_QUALITY_LEVELS
    )
    if not fast_first:
        report = optimize_complex(
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
            convergence_level=convergence_level,
            save_movie=save_movie,
            trajectory_start=complex_trajectory_start,
            trajectory_path=trajectory_path,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
        )
        return _direct_auto_result(report, OptimizationRoute.COMPLEX)

    working_mol, preliminary_trajectory, report = _run_fast_complex_attempt(
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
        trajectory_start=TrajectoryStart.FINAL_OPTIMIZATION,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
    preliminary_attempt = _attempt_report(
        report,
        OptimizationRoute.ORDINARY_FAST,
    )
    if preliminary_attempt.status is OptimizationAttemptStatus.ACCEPTED:
        return _accept_fast_result(
            mol,
            working_mol,
            preliminary_trajectory,
            report,
            preliminary_attempt,
            save_movie=save_movie,
            trajectory_path=trajectory_path,
        )
    return _fallback_to_complex(
        mol,
        preliminary_trajectory,
        preliminary_attempt,
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
        trajectory_start=complex_trajectory_start,
        trajectory_path=trajectory_path,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
