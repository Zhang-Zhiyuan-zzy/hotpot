"""Force-field public façade for Python 3.10+ and current Open Babel."""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from . import utils as _utils
from .utils import (
    AcceptanceCheck,
    AtomTopologySignature,
    BondTopologySignature,
    Build3DReport,
    BuildAndOptimizeReport,
    BuildTimeoutError,
    BuildWorkerError,
    BuildWorkerResult,
    CandidateRejection,
    ComplexBuildDiagnostics,
    ComplexBuildError,
    ComplexBuildReport,
    ComplexBuildTimeoutError,
    ComplexBuildWarning,
    ComplexBuildWorkerError,
    CoordinationEnvironment,
    CoordinationGeometryCandidate,
    CoordinationGeometryResult,
    ForceFieldAcceptanceEvidence,
    ForceFieldDiagnosticValue,
    ForceFieldError,
    ForceFieldRunReport,
    ForceFieldSetupError,
    ForceFieldSetupReport,
    ForceFieldValidationReport,
    ForceFieldWorkflowReport,
    GeometryQualityError,
    GeometryQualityWarning,
    OptimizationAlgorithm,
    StructureAcceptanceThresholds,
    TerminationReason,
    TopologyReference,
)


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = _utils.__all__

AcceptanceLevel = _utils.AcceptanceLevel
ForceFieldStage = _utils.ForceFieldStage


def capture_topology(
    mol: "Molecule",
    *,
    allow_added_hydrogens: bool = True,
) -> TopologyReference:
    return _utils.capture_topology(
        mol,
        allow_added_hydrogens=allow_added_hydrogens,
    )


def evaluate_structure_acceptance(
    mol: "Molecule",
    *,
    level: AcceptanceLevel = "standard",
    topology_reference: Optional[TopologyReference] = None,
    forcefield_report: Optional[ForceFieldAcceptanceEvidence] = None,
    forcefield_stage: ForceFieldStage = "final",
    thresholds: Optional[StructureAcceptanceThresholds] = None,
) -> ForceFieldValidationReport:
    return _utils.evaluate_structure_acceptance(
        mol,
        level=level,
        topology_reference=topology_reference,
        forcefield_report=forcefield_report,
        forcefield_stage=forcefield_stage,
        thresholds=thresholds,
    )


def is_structure_accepted(
    mol: "Molecule",
    *,
    level: AcceptanceLevel = "standard",
    topology_reference: Optional[TopologyReference] = None,
    forcefield_report: Optional[ForceFieldAcceptanceEvidence] = None,
    forcefield_stage: ForceFieldStage = "final",
    thresholds: Optional[StructureAcceptanceThresholds] = None,
) -> bool:
    return _utils.is_structure_accepted(
        mol,
        level=level,
        topology_reference=topology_reference,
        forcefield_report=forcefield_report,
        forcefield_stage=forcefield_stage,
        thresholds=thresholds,
    )


def perturb(
    mol: "Molecule",
    *,
    sigma: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    return _utils.perturb(mol, sigma=sigma, seed=seed)


def collect_coordination_environments(
    mol: "Molecule",
) -> Tuple[CoordinationEnvironment, ...]:
    return _utils.collect_coordination_environments(mol)


def prepare_coordination_geometry(
    mol: "Molecule",
    *,
    environments: Optional[Tuple[CoordinationEnvironment, ...]] = None,
    strategy: Optional[str] = None,
    seed: Optional[int] = None,
) -> CoordinationGeometryResult:
    return _utils.prepare_coordination_geometry(
        mol,
        environments=environments,
        strategy=strategy,
        seed=seed,
    )


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
        worker_target=_utils._seeded_ob_build_worker,
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
    save_movie: bool = False,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
) -> ForceFieldRunReport:
    return _utils.optimize(
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
        save_movie=save_movie,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
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
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    return _utils._build_complex3d_workflow(
        mol,
        forcefield,
        candidate_count=candidate_count,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        seed=seed,
        coordination_geometry=coordination_geometry,
        worker_target=_utils._build_ligand_proxies_worker,
    )


def optimize_complex(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
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
) -> ForceFieldRunReport:
    return _utils.optimize_complex(
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
        save_movie=save_movie,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
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
    return _utils._complexes_build_workflow(
        mol,
        forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        candidate_count=candidate_count,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
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
        worker_target=_utils._build_ligand_proxies_worker,
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
    coordination_geometry: Optional[str] = None,
) -> ForceFieldWorkflowReport:
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
        candidate_count=candidate_count,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        coordination_geometry=coordination_geometry,
        seeded_build_worker=_utils._seeded_ob_build_worker,
        complex_build_worker=_utils._build_ligand_proxies_worker,
    )


def auto_optimize(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
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
) -> ForceFieldRunReport:
    return _utils.auto_optimize(
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
        save_movie=save_movie,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
