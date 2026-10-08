from enum import Enum
from typing import Dict, List, Optional, overload, Tuple, Union

import numpy as np
import numpy.typing as npt


Coordinate = Tuple[float, float, float]


def _default_torsion_settings_snapshot() -> Dict[str, float]: ...


class RuleStage(Enum):
    PRE_BUILD: RuleStage
    PRE_FORCEFIELD_SETUP: RuleStage


class BondKind(Enum):
    SINGLE: BondKind
    DOUBLE: BondKind
    TRIPLE: BondKind
    AROMATIC: BondKind
    ZERO: BondKind
    DATIVE: BondKind
    UNKNOWN: BondKind


class RuleDescriptor:
    rule_id: str
    version: str
    stage: RuleStage
    priority: int


class HybridizationChange:
    atom_index: int
    before: int
    after: int


class CoordinateChange:
    atom_index: int
    before: Coordinate
    after: Coordinate


class RuleApplication:
    rule_id: str
    version: str
    stage: RuleStage
    priority: int
    atom_indices: List[int]
    metric_before: Optional[float]
    hybridization_changes: List[HybridizationChange]
    coordinate_changes: List[CoordinateChange]


class RulePlan:
    stage: RuleStage
    applications: List[RuleApplication]


class ForceFieldSetupError(RuntimeError):
    forcefield: str
    stage: str
    workflow_stage: str
    completed_coordination: Optional["CoordinationStageResult"]


class ForceFieldEnergyUnitError(RuntimeError):
    forcefield: str
    unit: str


class OptimizationFrameError(RuntimeError):
    forcefield: str


class MoleculeData:
    def __init__(
        self,
        schema_version: int,
        atomic_numbers: npt.NDArray[np.int32],
        formal_charges: npt.NDArray[np.int32],
        partial_charges: npt.NDArray[np.float64],
        coordinates: npt.NDArray[np.float64],
        atom_aromatic: npt.NDArray[np.uint8],
        bond_indices: npt.NDArray[np.int32],
        bond_orders: npt.NDArray[np.float64],
        bond_kinds: npt.NDArray[np.uint8],
        bond_aromatic: npt.NDArray[np.uint8],
        unit_cell: Optional[npt.NDArray[np.float64]] = ...,
    ) -> None: ...

    atom_count: int
    bond_count: int


class BuildResult:
    succeeded: bool
    coordinates: npt.NDArray[np.float64]
    rules: RulePlan


class SingleOptimizationResult:
    coordinates: npt.NDArray[np.float64]
    energy_kj_mol: float
    backend_energy_unit: str
    exploded: bool
    rules: RulePlan


class GradientMetrics:
    rms_kj_mol_angstrom: float
    maximum_kj_mol_angstrom: float


class OptimizationMeasurements:
    energy_kj_mol: float
    gradients: GradientMetrics
    energy_change_kj_mol: Optional[float]
    maximum_displacement_angstrom: Optional[float]
    finite_coordinates: bool
    exploded: bool


class OptimizationFailure(Enum):
    NONE: OptimizationFailure
    NONFINITE_COORDINATES: OptimizationFailure
    NONFINITE_ENERGY: OptimizationFailure
    NONFINITE_GRADIENTS: OptimizationFailure
    EXPLOSION_DETECTED: OptimizationFailure


class OptimizationCheckResult:
    evaluated_coordinates: npt.NDArray[np.float64]
    measurements: OptimizationMeasurements
    failure: OptimizationFailure
    backend_energy_unit: str
    rules: RulePlan


class OptimizationFrame:
    coordinates: npt.NDArray[np.float64]
    energy: float
    rms_gradient: float
    max_gradient: float
    exploded: bool
    converged: bool
    epoch_index: int
    segment_epochs_completed: int
    segment_index: int
    energy_change: Optional[float]
    max_displacement: Optional[float]


class OptimizationResult:
    coordinates: npt.NDArray[np.float64]
    terminal_coordinates: npt.NDArray[np.float64]
    frames: List[OptimizationFrame]
    selected_frame_index: int
    best_epoch: int
    final_energy: float
    best_energy: float
    rms_gradient: float
    max_gradient: float
    exploded: bool
    converged: bool
    epochs_completed: int
    steps_submitted: int
    initialization_steps: int
    selected_segment_epochs_completed: int
    backend_energy_unit: str
    termination_reason: str
    terminal_converged: bool
    energy_changes: List[float]
    max_displacements: List[float]
    epoch_energies: List[float]
    rules: RulePlan


class RuntimeInfo:
    compiled_openbabel_version: str
    runtime_openbabel_version: str
    cxx11_abi: int
    openbabel_library_path: str
    babel_libdir: str
    babel_datadir: str


class FrameDetail(Enum):
    NONE: FrameDetail
    OPTIMIZATION: FrameDetail
    ALL_ATTEMPTS: FrameDetail


class NativeStageStatus(Enum):
    COMPLETED: NativeStageStatus
    PARTIAL: NativeStageStatus
    FAILED: NativeStageStatus


class NativeRingGraphScope(Enum):
    LIGAND_SKELETON: NativeRingGraphScope
    FULL_GRAPH: NativeRingGraphScope


class NativePiercingState(Enum):
    PIERCES: NativePiercingState
    DOES_NOT_PIERCE: NativePiercingState
    UNDETERMINED: NativePiercingState


class NativeSegmentCycleIndeterminacy(Enum):
    NONFINITE_INPUT: NativeSegmentCycleIndeterminacy
    NUMERIC_BAND: NativeSegmentCycleIndeterminacy
    TOLERANCE_DOMAIN: NativeSegmentCycleIndeterminacy
    DEGENERATE_CYCLE: NativeSegmentCycleIndeterminacy
    DEGENERATE_SEGMENT: NativeSegmentCycleIndeterminacy
    DEGENERATE_TRIANGLE: NativeSegmentCycleIndeterminacy
    SELF_INTERSECTION: NativeSegmentCycleIndeterminacy
    SURFACE_DISAGREEMENT: NativeSegmentCycleIndeterminacy
    INCOMPLETE_SURFACE_FAMILY: NativeSegmentCycleIndeterminacy
    SURFACE_CONSTRUCTION: NativeSegmentCycleIndeterminacy


class NativeTrajectoryStart(Enum):
    LIGAND_BUILD: NativeTrajectoryStart
    COORDINATION_RESTORATION: NativeTrajectoryStart
    COMPLEX_UNTANGLING: NativeTrajectoryStart
    FINAL_OPTIMIZATION: NativeTrajectoryStart


class NativeTrajectoryStage(Enum):
    LIGAND_BUILD: NativeTrajectoryStage
    COORDINATION_RESTORATION: NativeTrajectoryStage
    COMPLEX_UNTANGLING: NativeTrajectoryStage
    FINAL_OPTIMIZATION: NativeTrajectoryStage


class NativeTrajectoryEvent(Enum):
    INITIAL: NativeTrajectoryEvent
    BUILD_COMPLETE: NativeTrajectoryEvent
    WARMUP_COMPLETE: NativeTrajectoryEvent
    COORDINATION_READY: NativeTrajectoryEvent
    BOND_TRIAL: NativeTrajectoryEvent
    BOND_ACCEPTED: NativeTrajectoryEvent
    BOND_REJECTED: NativeTrajectoryEvent
    BOND_ROLLBACK: NativeTrajectoryEvent
    BOND_FORCED: NativeTrajectoryEvent
    METAL_RELOCATION_TRIAL: NativeTrajectoryEvent
    METAL_RELOCATED: NativeTrajectoryEvent
    METAL_RELOCATION_FAILED: NativeTrajectoryEvent
    TOPOLOGY_CHECKPOINT: NativeTrajectoryEvent
    RING_OPENED: NativeTrajectoryEvent
    PERTURBED: NativeTrajectoryEvent
    OPTIMIZED: NativeTrajectoryEvent
    RING_CLOSED: NativeTrajectoryEvent
    SETTLED: NativeTrajectoryEvent
    ROLLED_BACK: NativeTrajectoryEvent
    EPOCH_COMPLETE: NativeTrajectoryEvent
    TERMINAL: NativeTrajectoryEvent


class ComplexSessionInput:
    def __init__(
        self,
        schema_version: int,
        atomic_numbers: npt.NDArray[np.int32],
        formal_charges: npt.NDArray[np.int32],
        partial_charges: npt.NDArray[np.float64],
        coordinates: npt.NDArray[np.float64],
        atom_aromatic: npt.NDArray[np.uint8],
        ligand_bond_indices: npt.NDArray[np.int32],
        ligand_bond_orders: npt.NDArray[np.float64],
        ligand_bond_kinds: npt.NDArray[np.uint8],
        ligand_bond_aromatic: npt.NDArray[np.uint8],
        metal_indices: npt.NDArray[np.int32],
        intended_coordination_bonds: npt.NDArray[np.int32],
        intended_coordination_orders: npt.NDArray[np.float64],
        intended_coordination_kinds: npt.NDArray[np.uint8],
        unit_cell: Optional[npt.NDArray[np.float64]] = ...,
    ) -> None: ...

    atom_count: int
    ligand_bond_count: int
    intended_coordination_bond_count: int


class PerturbationOffsetBatch:
    def __init__(self, offsets: npt.NDArray[np.float64]) -> None: ...

    atom_count: int
    frame_count: int
    offsets: npt.NDArray[np.float64]


class StructureSession:
    atom_count: int
    ligand_bond_count: int
    intended_coordination_bond_count: int


class StructureSnapshot:
    coordinates: npt.NDArray[np.float64]
    active_ligand_bond_mask: npt.NDArray[np.uint8]
    active_coordination_mask: npt.NDArray[np.uint8]
    component_ids: npt.NDArray[np.int32]
    ligand_bond_count: int
    active_bond_count: int
    coordinate_revision: int
    topology_revision: int


class RadiusSource(Enum):
    OPENBABEL_COVALENT: RadiusSource
    DEFAULT_COVALENT: RadiusSource


class AtomicRadius:
    angstrom: float
    source: RadiusSource


class PlacementStatus(Enum):
    FULLY_FEASIBLE: PlacementStatus
    PARTIAL: PlacementStatus
    INFEASIBLE: PlacementStatus


class DonorPathStatus(Enum):
    SAFE: DonorPathStatus
    UNDETERMINED: DonorPathStatus
    OUT_OF_RANGE: DonorPathStatus
    ATOM_OBSTRUCTION: DonorPathStatus
    BOND_OBSTRUCTION: DonorPathStatus
    RING_PIERCING: DonorPathStatus


class PlacementProposalKind(Enum):
    CURRENT: PlacementProposalKind
    TARGET_SPHERE: PlacementProposalKind
    SPHERE_INTERSECTION: PlacementProposalKind
    LEAST_SQUARES: PlacementProposalKind
    FIBONACCI_FALLBACK: PlacementProposalKind


class MetalPlacementOptions:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
        self,
        maximum_candidate_count: int,
        fibonacci_direction_count: int,
        sphere_intersection_count: int,
        least_squares_iteration_count: int,
        maximum_actionable_ring_size: int,
        coordination_distance_scale: float,
        coordination_distance_ratio_minimum: float,
        coordination_distance_ratio_maximum: float,
        absolute_center_clearance_angstrom: float,
        center_covalent_radius_scale: float,
        minimum_path_atom_clearance: float,
        minimum_path_bond_clearance: float,
        broad_phase_skin_angstrom: float,
        duplicate_tolerance_angstrom: float,
        retain_candidate_evidence: bool,
        geometry_absolute_length: float,
        geometry_relative_length: float,
        geometry_parameter: float,
        geometry_machine_epsilon_factor: float,
        geometry_predicate_guard_factor: float,
        geometry_planarity_factor: float,
        geometry_winding_residual: float,
        geometry_intersection_merge_factor: float,
        geometry_aabb_padding_factor: float,
        surface_maximum_cycle_vertices: int,
        surface_maximum_surface_count: int,
        surface_maximum_segment_triangle_tests: int,
        surface_maximum_triangle_pair_tests: int,
    ) -> None: ...

    maximum_candidate_count: int
    fibonacci_direction_count: int
    sphere_intersection_count: int
    least_squares_iteration_count: int
    maximum_actionable_ring_size: int
    coordination_distance_scale: float
    coordination_distance_ratio_minimum: float
    coordination_distance_ratio_maximum: float
    absolute_center_clearance_angstrom: float
    center_covalent_radius_scale: float
    minimum_path_atom_clearance: float
    minimum_path_bond_clearance: float
    broad_phase_skin_angstrom: float
    duplicate_tolerance_angstrom: float
    retain_candidate_evidence: bool
    geometry_absolute_length: float
    geometry_relative_length: float
    geometry_parameter: float
    geometry_machine_epsilon_factor: float
    geometry_predicate_guard_factor: float
    geometry_planarity_factor: float
    geometry_winding_residual: float
    geometry_intersection_merge_factor: float
    geometry_aabb_padding_factor: float
    surface_maximum_cycle_vertices: int
    surface_maximum_surface_count: int
    surface_maximum_segment_triangle_tests: int
    surface_maximum_triangle_pair_tests: int


class DonorApproachEvidence:
    neighbour_index: int
    metal_donor_neighbour_angle_degrees: Optional[float]


class DonorPathEvidence:
    donor_index: int
    group_index: int
    status: DonorPathStatus
    distance_reachable: bool
    atom_obstructed: bool
    bond_obstructed: bool
    target_distance_angstrom: float
    distance_angstrom: float
    distance_ratio: float
    normalized_atom_clearance: float
    normalized_bond_clearance: float
    definite_piercing_count: int
    undetermined_relation_count: int
    atom_pair_count: int
    atom_aabb_rejected_pair_count: int
    bond_pair_count: int
    bond_aabb_rejected_pair_count: int
    cycle_pair_count: int
    cycle_aabb_rejected_pair_count: int
    approach_angles: List[DonorApproachEvidence]


class DonorPairEvidence:
    first_donor_index: int
    second_donor_index: int
    donor_separation_angstrom: float
    target_distance_sum_angstrom: float
    target_distance_difference_angstrom: float
    target_shells_intersect: bool
    donor_metal_donor_angle_degrees: Optional[float]


class PlacementCandidateEvidence:
    coordinates: Coordinate
    proposal_kind: PlacementProposalKind
    status: PlacementStatus
    excluded_large_cycle_count: int
    covered_group_count: int
    safe_donor_count: int
    out_of_range_donor_count: int
    atom_obstruction_count: int
    bond_obstruction_count: int
    definite_piercing_count: int
    hard_obstruction_count: int
    minimum_normalized_clearance: float
    worst_distance_deviation: float
    rms_distance_deviation: float
    undetermined_relation_count: int
    atom_pair_count: int
    atom_aabb_rejected_pair_count: int
    bond_pair_count: int
    bond_aabb_rejected_pair_count: int
    cycle_pair_count: int
    cycle_aabb_rejected_pair_count: int
    displacement_angstrom: float
    proposal_ordinal: int
    donor_paths: List[DonorPathEvidence]
    donor_pairs: List[DonorPairEvidence]


class MetalPlacementResult:
    metal_index: int
    status: PlacementStatus
    original_coordinates: Coordinate
    selected_coordinates: Coordinate
    moved: bool
    candidates_evaluated: int
    selected_evidence: PlacementCandidateEvidence
    retained_candidates: List[PlacementCandidateEvidence]
    excluded_large_cycle_count: int
    warning_codes: List[str]


class MetalPlacementReport:
    metals: List[MetalPlacementResult]
    selected_coordinates: npt.NDArray[np.float64]
    warning_codes: List[str]


class OptimizationStoppingOptions:
    def __init__(
        self,
        window: int,
        maximum_energy_change_kj_mol: float,
        maximum_atom_displacement_angstrom: float,
        maximum_rms_gradient_kj_mol_angstrom: float,
        maximum_gradient_kj_mol_angstrom: float,
    ) -> None: ...

    window: int
    maximum_energy_change_kj_mol: float
    maximum_atom_displacement_angstrom: float
    maximum_rms_gradient_kj_mol_angstrom: float
    maximum_gradient_kj_mol_angstrom: float


class RingScreeningOptions:
    def __init__(
        self,
        maximum_actionable_ring_size: int,
        maximum_relevant_cycle_count: int,
        geometry_absolute_length: float,
        geometry_relative_length: float,
        geometry_parameter: float,
        geometry_machine_epsilon_factor: float,
        geometry_predicate_guard_factor: float,
        geometry_planarity_factor: float,
        geometry_winding_residual: float,
        geometry_intersection_merge_factor: float,
        geometry_aabb_padding_factor: float,
        surface_maximum_cycle_vertices: int,
        surface_maximum_surface_count: int,
        surface_maximum_segment_triangle_tests: int,
        surface_maximum_triangle_pair_tests: int,
    ) -> None: ...

    maximum_actionable_ring_size: int
    maximum_relevant_cycle_count: int
    geometry_absolute_length: float
    geometry_relative_length: float
    geometry_parameter: float
    geometry_machine_epsilon_factor: float
    geometry_predicate_guard_factor: float
    geometry_planarity_factor: float
    geometry_winding_residual: float
    geometry_intersection_merge_factor: float
    geometry_aabb_padding_factor: float
    surface_maximum_cycle_vertices: int
    surface_maximum_surface_count: int
    surface_maximum_segment_triangle_tests: int
    surface_maximum_triangle_pair_tests: int


class CoordinationStageOptions:
    def __init__(
        self,
        forcefield: str,
        attempt_limit: int,
        relaxation_steps: int,
        perturb_sigma: float,
        trajectory_start: NativeTrajectoryStart,
        frame_detail: FrameDetail,
        torsion_singularity_threshold: float,
        torsion_repair_angle_radians: float,
        placement: MetalPlacementOptions = ...,
    ) -> None: ...

    forcefield: str
    attempt_limit: int
    relaxation_steps: int
    perturb_sigma: float
    trajectory_start: NativeTrajectoryStart
    frame_detail: FrameDetail
    torsion_singularity_threshold: float
    torsion_repair_angle_radians: float
    placement: MetalPlacementOptions


class ComplexOptimizationOptions:
    def __init__(
        self,
        forcefield: str,
        algorithm: str,
        epochs: int,
        steps_per_epoch: int,
        untangling_attempt_limit: int,
        perturb_interval: Optional[int],
        perturb_sigma: float,
        trajectory_start: NativeTrajectoryStart,
        frame_detail: FrameDetail,
        retain_epoch_history: bool,
        increasing_vdw: bool,
        vdw_cutoff_start: float,
        vdw_cutoff_end: float,
        energy_tolerance: float,
        stopping: Optional[OptimizationStoppingOptions],
        torsion_singularity_threshold: float,
        torsion_repair_angle_radians: float,
        ring_screening: RingScreeningOptions,
    ) -> None: ...

    forcefield: str
    algorithm: str
    epochs: int
    steps_per_epoch: int
    untangling_attempt_limit: int
    perturb_interval: Optional[int]
    perturb_sigma: float
    trajectory_start: NativeTrajectoryStart
    frame_detail: FrameDetail
    retain_epoch_history: bool
    increasing_vdw: bool
    vdw_cutoff_start: float
    vdw_cutoff_end: float
    energy_tolerance: float
    stopping: Optional[OptimizationStoppingOptions]
    torsion_singularity_threshold: float
    torsion_repair_angle_radians: float
    ring_screening: RingScreeningOptions


_DEFAULT_METAL_PLACEMENT_OPTIONS: MetalPlacementOptions
_DEFAULT_OPTIMIZATION_STOPPING_OPTIONS: OptimizationStoppingOptions
_DEFAULT_RING_SCREENING_OPTIONS: RingScreeningOptions
_DEFAULT_COORDINATION_STAGE_OPTIONS: CoordinationStageOptions
_DEFAULT_COMPLEX_OPTIMIZATION_OPTIONS: ComplexOptimizationOptions


class NativeRingFrameEvidence:
    def __init__(
        self,
        confirmed_piercing_count: int,
        uncertain_relation_count: Optional[int],
        ring_scope: Optional[str],
        max_ring_size: Optional[int],
        selected_ring_count: Optional[int],
        excluded_ring_count: Optional[int],
        candidate_pair_count: Optional[int],
        aabb_separated_pair_count: Optional[int],
        exact_pair_count: Optional[int],
        does_not_pierce_pair_count: Optional[int],
        scan_complete: Optional[bool],
    ) -> None: ...

    confirmed_piercing_count: int
    uncertain_relation_count: Optional[int]
    ring_scope: Optional[str]
    max_ring_size: Optional[int]
    selected_ring_count: Optional[int]
    excluded_ring_count: Optional[int]
    candidate_pair_count: Optional[int]
    aabb_separated_pair_count: Optional[int]
    exact_pair_count: Optional[int]
    does_not_pierce_pair_count: Optional[int]
    scan_complete: Optional[bool]


class NativeCoordinationFrameEvidence:
    def __init__(
        self,
        bond_atom_indices: Optional[Tuple[int, int]],
        accepted: Optional[bool],
        pending_bond_count: int,
        forced: bool,
        piercing_relation_count: int,
        undetermined_relation_count: int,
        excluded_ring_count: int,
        metal_atom_index: Optional[int],
        relocation_status: Optional[str],
        relocation_candidates_evaluated: int,
        safe_donor_atom_indices: List[int],
        minimum_normalized_clearance: Optional[float],
        coordination_distance_deviation: Optional[float],
    ) -> None: ...

    bond_atom_indices: Optional[Tuple[int, int]]
    accepted: Optional[bool]
    pending_bond_count: int
    forced: bool
    piercing_relation_count: int
    undetermined_relation_count: int
    excluded_ring_count: int
    metal_atom_index: Optional[int]
    relocation_status: Optional[str]
    relocation_candidates_evaluated: int
    safe_donor_atom_indices: List[int]
    minimum_normalized_clearance: Optional[float]
    coordination_distance_deviation: Optional[float]


class NativeOptimizationFrameEvidence:
    def __init__(
        self,
        converged: bool,
        exploded: bool,
        finite_coordinates: bool,
        finite_energy: bool,
        finite_gradients: bool,
        rms_gradient_kj_mol_angstrom: Optional[float],
        max_gradient_kj_mol_angstrom: Optional[float],
        energy_change_kj_mol: Optional[float],
        max_displacement_angstrom: Optional[float],
    ) -> None: ...

    converged: bool
    exploded: bool
    finite_coordinates: bool
    finite_energy: bool
    finite_gradients: bool
    rms_gradient_kj_mol_angstrom: Optional[float]
    max_gradient_kj_mol_angstrom: Optional[float]
    energy_change_kj_mol: Optional[float]
    max_displacement_angstrom: Optional[float]


class NativeTopologyRevision:
    def __init__(
        self,
        active_ligand_bond_mask: npt.NDArray[np.uint8],
        active_coordination_bond_mask: npt.NDArray[np.uint8],
    ) -> None: ...

    active_ligand_bond_mask: npt.NDArray[np.uint8]
    active_coordination_bond_mask: npt.NDArray[np.uint8]


NativeFrameEvidence = Optional[
    Union[
        NativeRingFrameEvidence,
        NativeCoordinationFrameEvidence,
        NativeOptimizationFrameEvidence,
    ]
]


class NativeTrajectoryFrame:
    def __init__(
        self,
        coordinates: npt.NDArray[np.float64],
        stage: NativeTrajectoryStage,
        event: NativeTrajectoryEvent,
        component_index: Optional[int],
        attempt: Optional[int],
        step: Optional[int],
        energy_kj_mol: Optional[float],
        evidence: NativeFrameEvidence,
        topology_revision: int,
    ) -> None: ...

    coordinates: npt.NDArray[np.float64]
    stage: NativeTrajectoryStage
    event: NativeTrajectoryEvent
    component_index: Optional[int]
    attempt: Optional[int]
    step: Optional[int]
    energy_kj_mol: Optional[float]
    evidence: NativeFrameEvidence
    topology_revision: int


class NativeTrajectoryBatch:
    def __init__(
        self,
        atom_count: int,
        ligand_bond_count: int,
        intended_coordination_bond_count: int,
        start: NativeTrajectoryStart,
        topology_revisions: List[NativeTopologyRevision],
        frames: List[NativeTrajectoryFrame],
        selected_frame_index: Optional[int],
        terminal_frame_index: Optional[int],
    ) -> None: ...

    atom_count: int
    ligand_bond_count: int
    intended_coordination_bond_count: int
    start: NativeTrajectoryStart
    topology_revisions: List[NativeTopologyRevision]
    frame_count: int
    frames: List[NativeTrajectoryFrame]
    selected_frame_index: Optional[int]
    terminal_frame_index: Optional[int]


class CoordinationStageResult:
    def __init__(
        self,
        status: NativeStageStatus,
        selected_coordinates: npt.NDArray[np.float64],
        terminal_coordinates: npt.NDArray[np.float64],
        final_active_coordination_mask: npt.NDArray[np.uint8],
        attempt_limit: int,
        attempts_completed: int,
        metal_relocation_attempt_count: int,
        relocated_metal_indices: List[int],
        infeasible_metal_indices: List[int],
        forced_bond_keys: List[Tuple[int, int]],
        rejected_piercing_trial_count: int,
        undetermined_trial_count: int,
        excluded_ring_observation_count: int,
        warning_codes: List[str],
        trajectory: NativeTrajectoryBatch,
        elapsed_seconds: float = 0.0,
    ) -> None: ...

    status: NativeStageStatus
    selected_coordinates: npt.NDArray[np.float64]
    terminal_coordinates: npt.NDArray[np.float64]
    final_active_coordination_mask: npt.NDArray[np.uint8]
    attempt_limit: int
    attempts_completed: int
    metal_relocation_attempt_count: int
    relocated_metal_indices: List[int]
    infeasible_metal_indices: List[int]
    forced_bond_keys: npt.NDArray[np.int32]
    rejected_piercing_trial_count: int
    undetermined_trial_count: int
    excluded_ring_observation_count: int
    warning_codes: List[str]
    trajectory: NativeTrajectoryBatch
    bond_count: int
    placement_report: MetalPlacementReport
    elapsed_seconds: float


class NativeBondRingFinding:
    def __init__(
        self,
        ring_index: int,
        ring_atom_indices: List[int],
        bond_key: Tuple[int, int],
        state: NativePiercingState,
        indeterminacy_causes: List[NativeSegmentCycleIndeterminacy],
        aabb_separated: bool,
        surface_complete: bool,
    ) -> None: ...

    ring_index: int
    ring_atom_indices: List[int]
    bond_key: Tuple[int, int]
    state: NativePiercingState
    indeterminacy_causes: List[NativeSegmentCycleIndeterminacy]
    aabb_separated: bool
    surface_complete: bool


class NativeRingCheckpointReport:
    def __init__(
        self,
        state: NativePiercingState,
        scope: NativeRingGraphScope,
        maximum_actionable_ring_size: int,
        maximum_relevant_cycle_count: int,
        relevant_cycle_count: int,
        selected_ring_count: int,
        excluded_ring_count: int,
        active_bond_count: int,
        candidate_pair_count: int,
        aabb_separated_pair_count: int,
        exact_pair_count: int,
        piercing_pair_count: int,
        does_not_pierce_pair_count: int,
        undetermined_pair_count: int,
        scan_complete: bool,
        actionable_findings: List[NativeBondRingFinding],
    ) -> None: ...

    state: NativePiercingState
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
    actionable_findings: List[NativeBondRingFinding]


class ComplexOptimizationResult:
    def __init__(
        self,
        status: NativeStageStatus,
        selected_coordinates: npt.NDArray[np.float64],
        terminal_coordinates: npt.NDArray[np.float64],
        final_active_coordination_mask: npt.NDArray[np.uint8],
        untangling_attempt_limit: int,
        untangling_attempts_completed: int,
        initial_piercing_count: int,
        final_piercing_count: int,
        minimum_piercing_count: int,
        untangling_resolved: bool,
        selected_frame_index: int,
        best_epoch: int,
        final_energy_kj_mol: float,
        best_energy_kj_mol: float,
        rms_gradient_kj_mol_angstrom: float,
        max_gradient_kj_mol_angstrom: float,
        energy_changes: List[float],
        max_displacements: List[float],
        epoch_energies: List[float],
        exploded: bool,
        converged: bool,
        terminal_converged: bool,
        epochs_completed: int,
        steps_submitted: int,
        initialization_steps: int,
        selected_segment_epochs_completed: int,
        backend_energy_unit: str,
        termination_reason: str,
        warning_codes: List[str],
        trajectory: NativeTrajectoryBatch,
        final_checkpoint: NativeRingCheckpointReport,
        elapsed_seconds: float = 0.0,
    ) -> None: ...

    status: NativeStageStatus
    selected_coordinates: npt.NDArray[np.float64]
    terminal_coordinates: npt.NDArray[np.float64]
    final_active_coordination_mask: npt.NDArray[np.uint8]
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
    energy_changes: List[float]
    max_displacements: List[float]
    epoch_energies: List[float]
    exploded: bool
    converged: bool
    terminal_converged: bool
    epochs_completed: int
    steps_submitted: int
    initialization_steps: int
    selected_segment_epochs_completed: int
    backend_energy_unit: str
    termination_reason: str
    warning_codes: List[str]
    trajectory: NativeTrajectoryBatch
    final_checkpoint: NativeRingCheckpointReport
    elapsed_seconds: float


class ComplexWorkflowResult:
    def __init__(
        self,
        coordination: CoordinationStageResult,
        optimization: ComplexOptimizationResult,
        selected_coordinates: npt.NDArray[np.float64],
        terminal_coordinates: npt.NDArray[np.float64],
        final_active_coordination_mask: npt.NDArray[np.uint8],
        warning_codes: List[str],
        trajectory: NativeTrajectoryBatch,
    ) -> None: ...

    coordination: CoordinationStageResult
    optimization: ComplexOptimizationResult
    selected_coordinates: npt.NDArray[np.float64]
    terminal_coordinates: npt.NDArray[np.float64]
    final_active_coordination_mask: npt.NDArray[np.uint8]
    warning_codes: List[str]
    trajectory: NativeTrajectoryBatch


def create_coordination_session(
    session_input: ComplexSessionInput,
) -> StructureSession: ...

def restore_coordination(
    session: StructureSession,
    options: CoordinationStageOptions,
    perturbation_offsets: PerturbationOffsetBatch,
) -> CoordinationStageResult: ...

def optimize_complex(
    session: StructureSession,
    options: ComplexOptimizationOptions,
    untangling_offsets: PerturbationOffsetBatch,
    optimization_offsets: PerturbationOffsetBatch,
) -> ComplexOptimizationResult: ...

def run_complex_workflow(
    session: StructureSession,
    coordination_options: CoordinationStageOptions,
    optimization_options: ComplexOptimizationOptions,
    coordination_offsets: PerturbationOffsetBatch,
    untangling_offsets: PerturbationOffsetBatch,
    optimization_offsets: PerturbationOffsetBatch,
) -> ComplexWorkflowResult: ...

def run_complex_workflow_from_input(
    session_input: ComplexSessionInput,
    coordination_options: CoordinationStageOptions,
    optimization_options: ComplexOptimizationOptions,
    coordination_offsets: PerturbationOffsetBatch,
    untangling_offsets: PerturbationOffsetBatch,
    optimization_offsets: PerturbationOffsetBatch,
) -> ComplexWorkflowResult: ...

def create_optimization_session(
    session_input: ComplexSessionInput,
) -> StructureSession: ...

def snapshot_structure(session: StructureSession) -> StructureSnapshot: ...

def update_structure_coordinates(
    session: StructureSession,
    coordinates: npt.NDArray[np.float64],
) -> None: ...

def set_ligand_bond_active_mask(
    session: StructureSession,
    active_mask: npt.NDArray[np.uint8],
) -> None: ...

def set_coordination_active_mask(
    session: StructureSession,
    active_mask: npt.NDArray[np.uint8],
) -> None: ...

def covalent_radius(atomic_number: int) -> AtomicRadius: ...

def assess_metal_position(
    session: StructureSession,
    metal_index: int,
    candidate: Coordinate,
    options: MetalPlacementOptions = ...,
) -> PlacementCandidateEvidence: ...

def place_metal(
    session: StructureSession,
    metal_index: int,
    options: MetalPlacementOptions = ...,
) -> MetalPlacementResult: ...

def place_metals(
    session: StructureSession,
    options: MetalPlacementOptions = ...,
) -> MetalPlacementReport: ...

def runtime_info() -> RuntimeInfo: ...

def seed_random(seed: int) -> None: ...

def inspect_rules(
    molecule: MoleculeData,
    stage: RuleStage,
    singularity_threshold: float = ...,
    repair_angle_radians: float = ...,
) -> RulePlan: ...


def available_rules(stage: Optional[RuleStage] = ...) -> List[RuleDescriptor]: ...

def build(
    molecule: MoleculeData,
    stereo_warnings: Optional[bool] = ...,
) -> BuildResult: ...

def single_optimize(
    molecule: MoleculeData,
    forcefield: str,
    steps: int,
    singularity_threshold: float = ...,
    repair_angle_radians: float = ...,
) -> SingleOptimizationResult: ...

def check_optimization_state(
    molecule: MoleculeData,
    forcefield: str,
    previous_coordinates: Optional[npt.NDArray[np.float64]] = ...,
    previous_energy_kj_mol: Optional[float] = ...,
    singularity_threshold: float = ...,
    repair_angle_radians: float = ...,
) -> OptimizationCheckResult: ...

def optimize(
    molecule: MoleculeData,
    forcefield: str,
    algorithm: str,
    epochs: int,
    steps_per_epoch: int,
    perturb_interval: Optional[int] = ...,
    perturbation_offsets: Optional[npt.NDArray[np.float64]] = ...,
    retain_frames: bool = ...,
    retain_epoch_history: bool = ...,
    increasing_vdw: bool = ...,
    vdw_cutoff_start: float = ...,
    vdw_cutoff_end: float = ...,
    energy_tolerance: float = ...,
    stopping_window: Optional[int] = ...,
    maximum_energy_change_kj_mol: float = ...,
    maximum_atom_displacement_angstrom: float = ...,
    maximum_rms_gradient_kj_mol_angstrom: float = ...,
    maximum_gradient_kj_mol_angstrom: float = ...,
    singularity_threshold: float = ...,
    repair_angle_radians: float = ...,
) -> OptimizationResult: ...
