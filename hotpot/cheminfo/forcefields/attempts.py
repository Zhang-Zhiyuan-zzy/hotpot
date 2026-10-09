"""Non-committing force-field optimization attempts."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, TYPE_CHECKING

from .acceptance import evaluate_structure_acceptance
from .contracts import (
    AcceptanceLevel,
    ConvergenceLevel,
    DEFAULT_CONVERGENCE_LEVEL,
    ForceFieldAcceptanceEvidence,
    ForceFieldRunReport,
    OptimizationAlgorithm,
    OptimizationStoppingCriteria,
    StructureAcceptanceThresholds,
)
from .optimizer import _optimize_working_mol
from .topology import TopologyReference
from .trajectory import ForceFieldTrajectory, TrajectoryStage


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ()


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


def _run_ordinary_optimization_attempt(
    working_mol: "Molecule",
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
    algorithm: OptimizationAlgorithm,
    epochs: int,
    steps_per_epoch: int,
    seed: Optional[int],
    perturb_interval: Optional[int],
    perturb_sigma: float,
    retain_epoch_history: bool,
    increasing_vdw: bool,
    vdw_cutoff_start: float,
    vdw_cutoff_end: float,
    trajectory: ForceFieldTrajectory,
    quality_level: AcceptanceLevel,
    topology_reference: TopologyReference,
    quality_thresholds: Optional[StructureAcceptanceThresholds],
    stopping_criteria: Optional[OptimizationStoppingCriteria] = None,
    convergence_level: ConvergenceLevel = DEFAULT_CONVERGENCE_LEVEL,
    trajectory_stage: TrajectoryStage = TrajectoryStage.FINAL_OPTIMIZATION,
) -> ForceFieldRunReport:
    """Optimize and assess one working molecule without committing it."""
    report = _optimize_working_mol(
        working_mol,
        requested_forcefield=requested_forcefield,
        effective_forcefield=effective_forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        retain_epoch_history=retain_epoch_history,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        trajectory=trajectory,
        trajectory_stage=trajectory_stage,
        stopping_criteria=stopping_criteria,
        convergence_level=convergence_level,
    )
    quality_report = evaluate_structure_acceptance(
        working_mol,
        level=quality_level,
        topology_reference=topology_reference,
        forcefield_report=_forcefield_acceptance_evidence(report),
        forcefield_stage="final",
        thresholds=quality_thresholds,
    )
    return replace(report, quality_report=quality_report)
