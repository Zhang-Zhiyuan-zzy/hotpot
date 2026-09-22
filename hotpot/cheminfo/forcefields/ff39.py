"""Force-field public façade for Python 3.9 and Open Babel 3.1."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from . import utils as _utils
from . import utils39 as _utils39


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = _utils.__all__

# Shared public contracts have one defining module across both façades.
OptimizationAlgorithm = _utils.OptimizationAlgorithm
TerminationReason = _utils.TerminationReason
ForceFieldDiagnosticValue = _utils.ForceFieldDiagnosticValue
ForceFieldRunReport = _utils.ForceFieldRunReport
Build3DReport = _utils.Build3DReport
CandidateRejection = _utils.CandidateRejection
RingUntanglingReport = _utils.RingUntanglingReport
CoordinationBondRestorationReport = _utils.CoordinationBondRestorationReport
ComplexBuildDiagnostics = _utils.ComplexBuildDiagnostics
BuildWorkerResult = _utils.BuildWorkerResult
ForceFieldWorkflowReport = _utils.ForceFieldWorkflowReport
BuildAndOptimizeReport = _utils.BuildAndOptimizeReport
ComplexBuildReport = _utils.ComplexBuildReport
ForceFieldSetupReport = _utils.ForceFieldSetupReport
AcceptanceCheck = _utils.AcceptanceCheck
StructureAcceptanceThresholds = _utils.StructureAcceptanceThresholds
ForceFieldAcceptanceEvidence = _utils.ForceFieldAcceptanceEvidence
AtomTopologySignature = _utils.AtomTopologySignature
BondTopologySignature = _utils.BondTopologySignature
TopologyReference = _utils.TopologyReference
ForceFieldValidationReport = _utils.ForceFieldValidationReport
CoordinationEnvironment = _utils.CoordinationEnvironment
CoordinationGeometryCandidate = _utils.CoordinationGeometryCandidate
CoordinationGeometryResult = _utils.CoordinationGeometryResult
ForceFieldError = _utils.ForceFieldError
ForceFieldSetupError = _utils.ForceFieldSetupError
BuildWorkerError = _utils.BuildWorkerError
BuildTimeoutError = _utils.BuildTimeoutError
ComplexBuildError = _utils.ComplexBuildError
ComplexBuildWarning = _utils.ComplexBuildWarning
ComplexBuildWorkerError = _utils.ComplexBuildWorkerError
ComplexBuildTimeoutError = _utils.ComplexBuildTimeoutError
GeometryQualityError = _utils.GeometryQualityError
GeometryQualityWarning = _utils.GeometryQualityWarning

AcceptanceLevel = _utils.AcceptanceLevel
ForceFieldStage = _utils.ForceFieldStage

# These workflows do not select an Open Babel build worker and therefore use
# the exact same function objects on every supported Python version.
capture_topology = _utils.capture_topology
evaluate_structure_acceptance = _utils.evaluate_structure_acceptance
is_structure_accepted = _utils.is_structure_accepted
perturb = _utils.perturb
collect_coordination_environments = _utils.collect_coordination_environments
prepare_coordination_geometry = _utils.prepare_coordination_geometry
optimize = _utils.optimize
optimize_complex = _utils.optimize_complex
auto_optimize = _utils.auto_optimize


def build3d(
    mol: "Molecule",
    *,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    timeout: float = 1000.0,
) -> Build3DReport:
    return _utils._build3d_workflow(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
        timeout=timeout,
        worker_target=_utils39._seeded_ob_build_worker,
    )


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
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    """Build one ligand start and restore the complex coordination topology.

    ``candidate_count`` is a reserved API parameter and currently has no
    effect. Future multi-conformer support will generate genuinely independent
    starts, optimize and gate each one, deduplicate or cluster them by geometry,
    rank them by topology, geometry, and energy evidence, and refine the
    selected conformer.
    """
    _ = candidate_count
    return _utils._build_complex3d_workflow(
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
        coordination_geometry=coordination_geometry,
        worker_target=_utils39._build_ligand_proxies_worker,
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
    save_movie: bool = False,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    """Build, optimize, validate, and commit one complete complex structure.

    ``candidate_count`` is a reserved API parameter and currently has no
    effect. Future multi-conformer support will generate genuinely independent
    starts, optimize and gate each one, deduplicate or cluster them by geometry,
    rank them by topology, geometry, and energy evidence, and refine the
    selected conformer.
    """
    _ = candidate_count
    return _utils._complexes_build_workflow(
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
        save_movie=save_movie,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        coordination_geometry=coordination_geometry,
        worker_target=_utils39._build_ligand_proxies_worker,
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
    save_movie: bool = False,
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
    """Dispatch 3D building and optimization by molecular system type.

    ``candidate_count`` is a reserved API parameter and currently has no
    effect. Future complex multi-conformer support will generate genuinely
    independent starts, optimize and gate each one, deduplicate or cluster them
    by geometry, rank them by topology, geometry, and energy evidence, and
    refine the selected conformer.
    """
    _ = candidate_count
    return _utils._build_and_optimize_workflow(
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
        save_movie=save_movie,
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
        seeded_build_worker=_utils39._seeded_ob_build_worker,
        complex_build_worker=_utils39._build_ligand_proxies_worker,
    )
