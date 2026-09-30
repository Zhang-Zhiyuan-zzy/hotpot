"""Map native stage and trajectory values into immutable Python contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from ..geometry import PiercingState, SegmentCycleIndeterminacy
from .trajectory import (
    CoordinationFrameEvidence,
    FrameEvidence,
    OptimizationFrameEvidence,
    RingFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)

if TYPE_CHECKING:
    from ..obWrappers import _ob_native


__all__ = (
    "ComplexOptimizationResult",
    "ComplexWorkflowResult",
    "CoordinationStageResult",
    "NativeBondRingFinding",
    "NativeRingCheckpointReport",
    "NativeRingGraphScope",
    "NativeStageStatus",
    "NativeTopologyRevision",
    "NativeTrajectoryBatch",
)


class NativeStageStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class NativeRingGraphScope(str, Enum):
    LIGAND_SKELETON = "ligand_skeleton"
    FULL_GRAPH = "full_graph"


@dataclass(frozen=True)
class NativeBondRingFinding:
    ring_index: int
    ring_atom_indices: Tuple[int, ...]
    bond_key: Tuple[int, int]
    state: PiercingState
    indeterminacy_causes: Tuple[SegmentCycleIndeterminacy, ...]
    aabb_separated: bool
    surface_complete: bool


@dataclass(frozen=True)
class NativeRingCheckpointReport:
    state: PiercingState
    scope: NativeRingGraphScope
    maximum_actionable_ring_size: int
    maximum_relevant_cycle_count: int
    relevant_cycle_count: int
    selected_ring_count: int
    excluded_ring_count: int
    active_bond_count: int
    candidate_pair_count: int
    aabb_separated_pair_count: int
    exact_pair_count: int
    piercing_pair_count: int
    does_not_pierce_pair_count: int
    undetermined_pair_count: int
    scan_complete: bool
    actionable_findings: Tuple[NativeBondRingFinding, ...]


@dataclass(frozen=True)
class NativeTopologyRevision:
    active_ligand_bond_mask: NDArray[np.uint8]
    active_coordination_bond_mask: NDArray[np.uint8]


@dataclass(frozen=True)
class NativeTrajectoryBatch:
    coordinates: NDArray[np.float64]
    start: TrajectoryStart
    stages: Tuple[TrajectoryStage, ...]
    events: Tuple[TrajectoryEvent, ...]
    component_indices: Tuple[Optional[int], ...]
    attempts: Tuple[Optional[int], ...]
    steps: Tuple[Optional[int], ...]
    energies_kj_mol: NDArray[np.float64]
    evidence: Tuple[Optional[FrameEvidence], ...]
    topology_revisions: Tuple[NativeTopologyRevision, ...]
    frame_topology_revisions: NDArray[np.int32]
    selected_frame_index: Optional[int]
    terminal_frame_index: Optional[int]

    @property
    def frame_count(self) -> int:
        return len(self.stages)


@dataclass(frozen=True)
class CoordinationStageResult:
    status: NativeStageStatus
    selected_coordinates: NDArray[np.float64]
    terminal_coordinates: NDArray[np.float64]
    final_active_coordination_mask: NDArray[np.uint8]
    attempt_limit: int
    attempts_completed: int
    metal_relocation_attempt_count: int
    relocated_metal_indices: Tuple[int, ...]
    infeasible_metal_indices: Tuple[int, ...]
    forced_bond_keys: Tuple[Tuple[int, int], ...]
    rejected_piercing_trial_count: int
    undetermined_trial_count: int
    excluded_ring_observation_count: int
    warning_codes: Tuple[str, ...]
    trajectory: NativeTrajectoryBatch
    bond_count: int
    placement_report: "_ob_native.MetalPlacementReport"
    elapsed_seconds: float


@dataclass(frozen=True)
class ComplexOptimizationResult:
    status: NativeStageStatus
    selected_coordinates: NDArray[np.float64]
    terminal_coordinates: NDArray[np.float64]
    final_active_coordination_mask: NDArray[np.uint8]
    untangling_attempt_limit: int
    untangling_attempts_completed: int
    initial_piercing_count: int
    final_piercing_count: int
    minimum_piercing_count: int
    untangling_resolved: bool
    selected_frame_index: int
    best_epoch: int
    final_energy_kj_mol: float
    best_energy_kj_mol: float
    rms_gradient_kj_mol_angstrom: float
    max_gradient_kj_mol_angstrom: float
    energy_changes: Tuple[float, ...]
    max_displacements: Tuple[float, ...]
    epoch_energies: Tuple[float, ...]
    exploded: bool
    converged: bool
    terminal_converged: bool
    epochs_completed: int
    steps_submitted: int
    initialization_steps: int
    selected_segment_epochs_completed: int
    backend_energy_unit: str
    termination_reason: str
    warning_codes: Tuple[str, ...]
    trajectory: NativeTrajectoryBatch
    final_checkpoint: NativeRingCheckpointReport
    elapsed_seconds: float


@dataclass(frozen=True)
class ComplexWorkflowResult:
    coordination: CoordinationStageResult
    optimization: ComplexOptimizationResult
    selected_coordinates: NDArray[np.float64]
    terminal_coordinates: NDArray[np.float64]
    final_active_coordination_mask: NDArray[np.uint8]
    warning_codes: Tuple[str, ...]
    trajectory: NativeTrajectoryBatch


def _readonly(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    copied = np.array(array, dtype=dtype, order="C", copy=True)
    copied.setflags(write=False)
    return copied


def _native_frame_evidence(evidence: object) -> Optional[FrameEvidence]:
    if evidence is None:
        return None
    evidence_name = evidence.__class__.__name__
    if evidence_name == "NativeRingFrameEvidence":
        ring = cast("_ob_native.NativeRingFrameEvidence", evidence)
        return RingFrameEvidence(
            confirmed_piercing_count=ring.confirmed_piercing_count,
            uncertain_relation_count=ring.uncertain_relation_count,
            ring_scope=ring.ring_scope,
            max_ring_size=ring.max_ring_size,
            selected_ring_count=ring.selected_ring_count,
            excluded_ring_count=ring.excluded_ring_count,
            candidate_pair_count=ring.candidate_pair_count,
            aabb_separated_pair_count=ring.aabb_separated_pair_count,
            exact_pair_count=ring.exact_pair_count,
            does_not_pierce_pair_count=ring.does_not_pierce_pair_count,
            scan_complete=ring.scan_complete,
        )
    if evidence_name == "NativeCoordinationFrameEvidence":
        coordination = cast(
            "_ob_native.NativeCoordinationFrameEvidence",
            evidence,
        )
        bond_indices = coordination.bond_atom_indices
        return CoordinationFrameEvidence(
            bond_atom_indices=(
                None if bond_indices is None else tuple(bond_indices)
            ),
            accepted=coordination.accepted,
            pending_bond_count=coordination.pending_bond_count,
            forced=coordination.forced,
            piercing_relation_count=coordination.piercing_relation_count,
            undetermined_relation_count=(
                coordination.undetermined_relation_count
            ),
            excluded_ring_count=coordination.excluded_ring_count,
            metal_atom_index=coordination.metal_atom_index,
            relocation_status=coordination.relocation_status,
            relocation_candidates_evaluated=(
                coordination.relocation_candidates_evaluated
            ),
            safe_donor_atom_indices=tuple(
                coordination.safe_donor_atom_indices
            ),
            minimum_normalized_clearance=(
                coordination.minimum_normalized_clearance
            ),
            coordination_distance_deviation=(
                coordination.coordination_distance_deviation
            ),
        )
    if evidence_name == "NativeOptimizationFrameEvidence":
        optimization = cast(
            "_ob_native.NativeOptimizationFrameEvidence",
            evidence,
        )
        return OptimizationFrameEvidence(
            converged=optimization.converged,
            exploded=optimization.exploded,
            finite_coordinates=optimization.finite_coordinates,
            finite_energy=optimization.finite_energy,
            finite_gradients=optimization.finite_gradients,
            rms_gradient_kj_mol_angstrom=(
                optimization.rms_gradient_kj_mol_angstrom
            ),
            max_gradient_kj_mol_angstrom=(
                optimization.max_gradient_kj_mol_angstrom
            ),
            energy_change_kj_mol=optimization.energy_change_kj_mol,
            max_displacement_angstrom=(
                optimization.max_displacement_angstrom
            ),
        )
    raise TypeError(f"unsupported native frame evidence: {evidence_name}")


def _native_trajectory_batch(
    native_batch: "_ob_native.NativeTrajectoryBatch",
) -> NativeTrajectoryBatch:
    frames = tuple(native_batch.frames)
    atom_count = native_batch.atom_count
    coordinates = (
        np.stack(tuple(frame.coordinates for frame in frames))
        if frames
        else np.empty((0, atom_count, 3), dtype=np.float64)
    )
    revisions = tuple(
        NativeTopologyRevision(
            active_ligand_bond_mask=_readonly(
                revision.active_ligand_bond_mask,
                np.dtype(np.uint8),
            ),
            active_coordination_bond_mask=_readonly(
                revision.active_coordination_bond_mask,
                np.dtype(np.uint8),
            ),
        )
        for revision in native_batch.topology_revisions
    )
    return NativeTrajectoryBatch(
        coordinates=_readonly(coordinates, np.dtype(np.float64)),
        start=TrajectoryStart[native_batch.start.name],
        stages=tuple(TrajectoryStage[frame.stage.name] for frame in frames),
        events=tuple(TrajectoryEvent[frame.event.name] for frame in frames),
        component_indices=tuple(frame.component_index for frame in frames),
        attempts=tuple(frame.attempt for frame in frames),
        steps=tuple(frame.step for frame in frames),
        energies_kj_mol=_readonly(
            np.asarray(
                [
                    np.nan if frame.energy_kj_mol is None else frame.energy_kj_mol
                    for frame in frames
                ]
            ),
            np.dtype(np.float64),
        ),
        evidence=tuple(_native_frame_evidence(frame.evidence) for frame in frames),
        topology_revisions=revisions,
        frame_topology_revisions=_readonly(
            np.asarray(
                [frame.topology_revision for frame in frames],
                dtype=np.int32,
            ),
            np.dtype(np.int32),
        ),
        selected_frame_index=native_batch.selected_frame_index,
        terminal_frame_index=native_batch.terminal_frame_index,
    )


def _coordination_stage_result(
    result: "_ob_native.CoordinationStageResult",
) -> CoordinationStageResult:
    return CoordinationStageResult(
        status=NativeStageStatus[result.status.name],
        selected_coordinates=_readonly(
            result.selected_coordinates,
            np.dtype(np.float64),
        ),
        terminal_coordinates=_readonly(
            result.terminal_coordinates,
            np.dtype(np.float64),
        ),
        final_active_coordination_mask=_readonly(
            result.final_active_coordination_mask,
            np.dtype(np.uint8),
        ),
        attempt_limit=result.attempt_limit,
        attempts_completed=result.attempts_completed,
        metal_relocation_attempt_count=result.metal_relocation_attempt_count,
        relocated_metal_indices=tuple(result.relocated_metal_indices),
        infeasible_metal_indices=tuple(result.infeasible_metal_indices),
        forced_bond_keys=tuple(
            tuple(indices) for indices in result.forced_bond_keys
        ),
        rejected_piercing_trial_count=result.rejected_piercing_trial_count,
        undetermined_trial_count=result.undetermined_trial_count,
        excluded_ring_observation_count=(
            result.excluded_ring_observation_count
        ),
        warning_codes=tuple(result.warning_codes),
        trajectory=_native_trajectory_batch(result.trajectory),
        bond_count=result.bond_count,
        placement_report=result.placement_report,
        elapsed_seconds=result.elapsed_seconds,
    )


def _native_bond_ring_finding(
    finding: "_ob_native.NativeBondRingFinding",
) -> NativeBondRingFinding:
    return NativeBondRingFinding(
        ring_index=finding.ring_index,
        ring_atom_indices=tuple(finding.ring_atom_indices),
        bond_key=tuple(finding.bond_key),
        state=PiercingState[finding.state.name],
        indeterminacy_causes=tuple(
            SegmentCycleIndeterminacy[cause.name]
            for cause in finding.indeterminacy_causes
        ),
        aabb_separated=finding.aabb_separated,
        surface_complete=finding.surface_complete,
    )


def _native_ring_checkpoint_report(
    report: "_ob_native.NativeRingCheckpointReport",
) -> NativeRingCheckpointReport:
    return NativeRingCheckpointReport(
        state=PiercingState[report.state.name],
        scope=NativeRingGraphScope[report.scope.name],
        maximum_actionable_ring_size=report.maximum_actionable_ring_size,
        maximum_relevant_cycle_count=report.maximum_relevant_cycle_count,
        relevant_cycle_count=report.relevant_cycle_count,
        selected_ring_count=report.selected_ring_count,
        excluded_ring_count=report.excluded_ring_count,
        active_bond_count=report.active_bond_count,
        candidate_pair_count=report.candidate_pair_count,
        aabb_separated_pair_count=report.aabb_separated_pair_count,
        exact_pair_count=report.exact_pair_count,
        piercing_pair_count=report.piercing_pair_count,
        does_not_pierce_pair_count=report.does_not_pierce_pair_count,
        undetermined_pair_count=report.undetermined_pair_count,
        scan_complete=report.scan_complete,
        actionable_findings=tuple(
            _native_bond_ring_finding(finding)
            for finding in report.actionable_findings
        ),
    )


def _complex_optimization_result(
    result: "_ob_native.ComplexOptimizationResult",
) -> ComplexOptimizationResult:
    return ComplexOptimizationResult(
        status=NativeStageStatus[result.status.name],
        selected_coordinates=_readonly(
            result.selected_coordinates,
            np.dtype(np.float64),
        ),
        terminal_coordinates=_readonly(
            result.terminal_coordinates,
            np.dtype(np.float64),
        ),
        final_active_coordination_mask=_readonly(
            result.final_active_coordination_mask,
            np.dtype(np.uint8),
        ),
        untangling_attempt_limit=result.untangling_attempt_limit,
        untangling_attempts_completed=result.untangling_attempts_completed,
        initial_piercing_count=result.initial_piercing_count,
        final_piercing_count=result.final_piercing_count,
        minimum_piercing_count=result.minimum_piercing_count,
        untangling_resolved=result.untangling_resolved,
        selected_frame_index=result.selected_frame_index,
        best_epoch=result.best_epoch,
        final_energy_kj_mol=result.final_energy_kj_mol,
        best_energy_kj_mol=result.best_energy_kj_mol,
        rms_gradient_kj_mol_angstrom=(
            result.rms_gradient_kj_mol_angstrom
        ),
        max_gradient_kj_mol_angstrom=(
            result.max_gradient_kj_mol_angstrom
        ),
        energy_changes=tuple(result.energy_changes),
        max_displacements=tuple(result.max_displacements),
        epoch_energies=tuple(result.epoch_energies),
        exploded=result.exploded,
        converged=result.converged,
        terminal_converged=result.terminal_converged,
        epochs_completed=result.epochs_completed,
        steps_submitted=result.steps_submitted,
        initialization_steps=result.initialization_steps,
        selected_segment_epochs_completed=(
            result.selected_segment_epochs_completed
        ),
        backend_energy_unit=result.backend_energy_unit,
        termination_reason=result.termination_reason,
        warning_codes=tuple(result.warning_codes),
        trajectory=_native_trajectory_batch(result.trajectory),
        final_checkpoint=_native_ring_checkpoint_report(
            result.final_checkpoint
        ),
        elapsed_seconds=result.elapsed_seconds,
    )


def _complex_workflow_result(
    result: "_ob_native.ComplexWorkflowResult",
) -> ComplexWorkflowResult:
    return ComplexWorkflowResult(
        coordination=_coordination_stage_result(result.coordination),
        optimization=_complex_optimization_result(result.optimization),
        selected_coordinates=_readonly(
            result.selected_coordinates,
            np.dtype(np.float64),
        ),
        terminal_coordinates=_readonly(
            result.terminal_coordinates,
            np.dtype(np.float64),
        ),
        final_active_coordination_mask=_readonly(
            result.final_active_coordination_mask,
            np.dtype(np.uint8),
        ),
        warning_codes=tuple(result.warning_codes),
        trajectory=_native_trajectory_batch(result.trajectory),
    )
