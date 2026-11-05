# 原生声明目录

这些是源码随附的 `.pyi`，用于核对 native callable 的参数/返回类型。私有函数、返回值内部属性和结构方法均照声明记录；不将其自动提升为公开 API。

## `hotpot/cheminfo/adapters/native/substructure/_search_native.pyi`

```python
class MappingCursor(Iterator[tuple[int, ...]]):
    __iter__(self) -> MappingCursor
    __next__(self) -> tuple[int, ...]
class NativeTarget():
    __init__(self, target_spec: dict[str, Any]) -> None
class NativeProblem():
    __init__(self, query_spec: dict[str, Any], target: NativeTarget) -> None
    has_match(self) -> bool
    cursor(self) -> MappingCursor
    query_keys(self) -> tuple[int, ...]
class _NativeAutomorphism():
    __init__(self, target: NativeTarget, node_labels: tuple[int, ...], edge_labels: tuple[int, ...]) -> None
    equivalent(self, first: int, second: int) -> bool
```

## `hotpot/cheminfo/obWrappers/_ob_native.pyi`

```python
_default_torsion_settings_snapshot() -> Dict[str, float]
class RuleStage(Enum):
    PRE_BUILD: RuleStage  # read-only
    PRE_FORCEFIELD_SETUP: RuleStage  # read-only
class BondKind(Enum):
    SINGLE: BondKind  # read-only
    DOUBLE: BondKind  # read-only
    TRIPLE: BondKind  # read-only
    AROMATIC: BondKind  # read-only
    ZERO: BondKind  # read-only
    DATIVE: BondKind  # read-only
    UNKNOWN: BondKind  # read-only
class ConvergenceLevel(Enum):
    OPENBABEL: ConvergenceLevel  # read-only
    FAST: ConvergenceLevel  # read-only
    BALANCED: ConvergenceLevel  # read-only
    STRICT: ConvergenceLevel  # read-only
class RuleDescriptor():
    rule_id: str  # read-only
    version: str  # read-only
    stage: RuleStage  # read-only
    priority: int  # read-only
class HybridizationChange():
    atom_index: int  # read-only
    before: int  # read-only
    after: int  # read-only
class CoordinateChange():
    atom_index: int  # read-only
    before: Coordinate  # read-only
    after: Coordinate  # read-only
class RuleApplication():
    rule_id: str  # read-only
    version: str  # read-only
    stage: RuleStage  # read-only
    priority: int  # read-only
    atom_indices: List[int]  # read-only
    metric_before: Optional[float]  # read-only
    hybridization_changes: List[HybridizationChange]  # read-only
    coordinate_changes: List[CoordinateChange]  # read-only
class RulePlan():
    stage: RuleStage  # read-only
    applications: List[RuleApplication]  # read-only
class ForceFieldSetupError(RuntimeError):
    forcefield: str  # not specified by stub
    stage: str  # not specified by stub
    workflow_stage: str  # not specified by stub
    completed_coordination: Optional['CoordinationStageResult']  # not specified by stub
class ForceFieldEnergyUnitError(RuntimeError):
    forcefield: str  # not specified by stub
    unit: str  # not specified by stub
class OptimizationFrameError(RuntimeError):
    forcefield: str  # not specified by stub
class MoleculeData():
    __init__(self, schema_version: int, atomic_numbers: npt.NDArray[np.int32], formal_charges: npt.NDArray[np.int32], partial_charges: npt.NDArray[np.float64], coordinates: npt.NDArray[np.float64], atom_aromatic: npt.NDArray[np.uint8], bond_indices: npt.NDArray[np.int32], bond_orders: npt.NDArray[np.float64], bond_kinds: npt.NDArray[np.uint8], bond_aromatic: npt.NDArray[np.uint8], unit_cell: Optional[npt.NDArray[np.float64]]=..., stereo_configs: Tuple[Tuple[str, str, bool, Tuple[int, ...], Tuple[Optional[int], ...], Tuple[int, ...]], ...]=...) -> None
    atom_count: int  # read-only
    bond_count: int  # read-only
class BuildResult():
    succeeded: bool  # read-only
    coordinates: npt.NDArray[np.float64]  # read-only
    rules: RulePlan  # read-only
class SingleOptimizationResult():
    coordinates: npt.NDArray[np.float64]  # read-only
    energy_kj_mol: float  # read-only
    backend_energy_unit: str  # read-only
    exploded: bool  # read-only
    rules: RulePlan  # read-only
class GradientMetrics():
    rms_kj_mol_angstrom: float  # read-only
    maximum_kj_mol_angstrom: float  # read-only
class OptimizationMeasurements():
    energy_kj_mol: float  # read-only
    gradients: GradientMetrics  # read-only
    energy_change_kj_mol: Optional[float]  # read-only
    maximum_displacement_angstrom: Optional[float]  # read-only
    finite_coordinates: bool  # read-only
    exploded: bool  # read-only
class OptimizationFailure(Enum):
    NONE: OptimizationFailure  # read-only
    NONFINITE_COORDINATES: OptimizationFailure  # read-only
    NONFINITE_ENERGY: OptimizationFailure  # read-only
    NONFINITE_GRADIENTS: OptimizationFailure  # read-only
    EXPLOSION_DETECTED: OptimizationFailure  # read-only
class OptimizationCheckResult():
    evaluated_coordinates: npt.NDArray[np.float64]  # read-only
    measurements: OptimizationMeasurements  # read-only
    failure: OptimizationFailure  # read-only
    backend_energy_unit: str  # read-only
    rules: RulePlan  # read-only
class OptimizationFrame():
    coordinates: npt.NDArray[np.float64]  # read-only
    energy: float  # read-only
    rms_gradient: float  # read-only
    max_gradient: float  # read-only
    exploded: bool  # read-only
    converged: bool  # read-only
    epoch_index: int  # read-only
    segment_epochs_completed: int  # read-only
    segment_index: int  # read-only
    energy_change: Optional[float]  # read-only
    max_displacement: Optional[float]  # read-only
class OptimizationResult():
    coordinates: npt.NDArray[np.float64]  # read-only
    terminal_coordinates: npt.NDArray[np.float64]  # read-only
    frames: List[OptimizationFrame]  # read-only
    selected_frame_index: int  # read-only
    best_epoch: int  # read-only
    final_energy: float  # read-only
    best_energy: float  # read-only
    rms_gradient: float  # read-only
    max_gradient: float  # read-only
    exploded: bool  # read-only
    converged: bool  # read-only
    epochs_completed: int  # read-only
    steps_submitted: int  # read-only
    initialization_steps: int  # read-only
    selected_segment_epochs_completed: int  # read-only
    backend_energy_unit: str  # read-only
    termination_reason: str  # read-only
    terminal_converged: bool  # read-only
    convergence_level: ConvergenceLevel  # read-only
    energy_changes: List[float]  # read-only
    max_displacements: List[float]  # read-only
    epoch_energies: List[float]  # read-only
    rules: RulePlan  # read-only
class RuntimeInfo():
    compiled_openbabel_version: str  # read-only
    runtime_openbabel_version: str  # read-only
    cxx11_abi: int  # read-only
    openbabel_library_path: str  # read-only
    babel_libdir: str  # read-only
    babel_datadir: str  # read-only
class FrameDetail(Enum):
    NONE: FrameDetail  # read-only
    OPTIMIZATION: FrameDetail  # read-only
    ALL_ATTEMPTS: FrameDetail  # read-only
class NativeStageStatus(Enum):
    COMPLETED: NativeStageStatus  # read-only
    PARTIAL: NativeStageStatus  # read-only
    FAILED: NativeStageStatus  # read-only
class NativeRingGraphScope(Enum):
    LIGAND_SKELETON: NativeRingGraphScope  # read-only
    FULL_GRAPH: NativeRingGraphScope  # read-only
class NativePiercingState(Enum):
    PIERCES: NativePiercingState  # read-only
    DOES_NOT_PIERCE: NativePiercingState  # read-only
    UNDETERMINED: NativePiercingState  # read-only
class NativeSegmentCycleIndeterminacy(Enum):
    NONFINITE_INPUT: NativeSegmentCycleIndeterminacy  # read-only
    NUMERIC_BAND: NativeSegmentCycleIndeterminacy  # read-only
    TOLERANCE_DOMAIN: NativeSegmentCycleIndeterminacy  # read-only
    DEGENERATE_CYCLE: NativeSegmentCycleIndeterminacy  # read-only
    DEGENERATE_SEGMENT: NativeSegmentCycleIndeterminacy  # read-only
    DEGENERATE_TRIANGLE: NativeSegmentCycleIndeterminacy  # read-only
    SELF_INTERSECTION: NativeSegmentCycleIndeterminacy  # read-only
    SURFACE_DISAGREEMENT: NativeSegmentCycleIndeterminacy  # read-only
    INCOMPLETE_SURFACE_FAMILY: NativeSegmentCycleIndeterminacy  # read-only
    SURFACE_CONSTRUCTION: NativeSegmentCycleIndeterminacy  # read-only
class NativeTrajectoryStart(Enum):
    LIGAND_BUILD: NativeTrajectoryStart  # read-only
    COORDINATION_RESTORATION: NativeTrajectoryStart  # read-only
    COMPLEX_UNTANGLING: NativeTrajectoryStart  # read-only
    FINAL_OPTIMIZATION: NativeTrajectoryStart  # read-only
class NativeTrajectoryStage(Enum):
    LIGAND_BUILD: NativeTrajectoryStage  # read-only
    COORDINATION_RESTORATION: NativeTrajectoryStage  # read-only
    COMPLEX_UNTANGLING: NativeTrajectoryStage  # read-only
    FINAL_OPTIMIZATION: NativeTrajectoryStage  # read-only
class NativeTrajectoryEvent(Enum):
    INITIAL: NativeTrajectoryEvent  # read-only
    BUILD_COMPLETE: NativeTrajectoryEvent  # read-only
    WARMUP_COMPLETE: NativeTrajectoryEvent  # read-only
    COORDINATION_READY: NativeTrajectoryEvent  # read-only
    BOND_TRIAL: NativeTrajectoryEvent  # read-only
    BOND_ACCEPTED: NativeTrajectoryEvent  # read-only
    BOND_REJECTED: NativeTrajectoryEvent  # read-only
    BOND_ROLLBACK: NativeTrajectoryEvent  # read-only
    BOND_FORCED: NativeTrajectoryEvent  # read-only
    METAL_RELOCATION_TRIAL: NativeTrajectoryEvent  # read-only
    METAL_RELOCATED: NativeTrajectoryEvent  # read-only
    METAL_RELOCATION_FAILED: NativeTrajectoryEvent  # read-only
    TOPOLOGY_CHECKPOINT: NativeTrajectoryEvent  # read-only
    RING_OPENED: NativeTrajectoryEvent  # read-only
    PERTURBED: NativeTrajectoryEvent  # read-only
    OPTIMIZED: NativeTrajectoryEvent  # read-only
    RING_CLOSED: NativeTrajectoryEvent  # read-only
    SETTLED: NativeTrajectoryEvent  # read-only
    ROLLED_BACK: NativeTrajectoryEvent  # read-only
    EPOCH_COMPLETE: NativeTrajectoryEvent  # read-only
    TERMINAL: NativeTrajectoryEvent  # read-only
class ComplexSessionInput():
    __init__(self, schema_version: int, atomic_numbers: npt.NDArray[np.int32], formal_charges: npt.NDArray[np.int32], partial_charges: npt.NDArray[np.float64], coordinates: npt.NDArray[np.float64], atom_aromatic: npt.NDArray[np.uint8], ligand_bond_indices: npt.NDArray[np.int32], ligand_bond_orders: npt.NDArray[np.float64], ligand_bond_kinds: npt.NDArray[np.uint8], ligand_bond_aromatic: npt.NDArray[np.uint8], metal_indices: npt.NDArray[np.int32], intended_coordination_bonds: npt.NDArray[np.int32], intended_coordination_orders: npt.NDArray[np.float64], intended_coordination_kinds: npt.NDArray[np.uint8], unit_cell: Optional[npt.NDArray[np.float64]]=...) -> None
    atom_count: int  # read-only
    ligand_bond_count: int  # read-only
    intended_coordination_bond_count: int  # read-only
class PerturbationOffsetBatch():
    __init__(self, offsets: npt.NDArray[np.float64]) -> None
    atom_count: int  # read-only
    frame_count: int  # read-only
    offsets: npt.NDArray[np.float64]  # read-only
class StructureSession():
    atom_count: int  # read-only
    ligand_bond_count: int  # read-only
    intended_coordination_bond_count: int  # read-only
class StructureSnapshot():
    coordinates: npt.NDArray[np.float64]  # read-only
    active_ligand_bond_mask: npt.NDArray[np.uint8]  # read-only
    active_coordination_mask: npt.NDArray[np.uint8]  # read-only
    component_ids: npt.NDArray[np.int32]  # read-only
    ligand_bond_count: int  # read-only
    active_bond_count: int  # read-only
    coordinate_revision: int  # read-only
    topology_revision: int  # read-only
class RadiusSource(Enum):
    OPENBABEL_COVALENT: RadiusSource  # read-only
    DEFAULT_COVALENT: RadiusSource  # read-only
class AtomicRadius():
    angstrom: float  # read-only
    source: RadiusSource  # read-only
class PlacementStatus(Enum):
    FULLY_FEASIBLE: PlacementStatus  # read-only
    PARTIAL: PlacementStatus  # read-only
    INFEASIBLE: PlacementStatus  # read-only
class DonorPathStatus(Enum):
    SAFE: DonorPathStatus  # read-only
    UNDETERMINED: DonorPathStatus  # read-only
    OUT_OF_RANGE: DonorPathStatus  # read-only
    ATOM_OBSTRUCTION: DonorPathStatus  # read-only
    BOND_OBSTRUCTION: DonorPathStatus  # read-only
    RING_PIERCING: DonorPathStatus  # read-only
class PlacementProposalKind(Enum):
    CURRENT: PlacementProposalKind  # read-only
    TARGET_SPHERE: PlacementProposalKind  # read-only
    SPHERE_INTERSECTION: PlacementProposalKind  # read-only
    LEAST_SQUARES: PlacementProposalKind  # read-only
    FIBONACCI_FALLBACK: PlacementProposalKind  # read-only
class MetalPlacementOptions():
    __init__(self) -> None
    __init__(self, maximum_candidate_count: int, fibonacci_direction_count: int, sphere_intersection_count: int, least_squares_iteration_count: int, maximum_actionable_ring_size: int, coordination_distance_scale: float, coordination_distance_ratio_minimum: float, coordination_distance_ratio_maximum: float, absolute_center_clearance_angstrom: float, center_covalent_radius_scale: float, minimum_path_atom_clearance: float, minimum_path_bond_clearance: float, broad_phase_skin_angstrom: float, duplicate_tolerance_angstrom: float, retain_candidate_evidence: bool, geometry_absolute_length: float, geometry_relative_length: float, geometry_parameter: float, geometry_machine_epsilon_factor: float, geometry_predicate_guard_factor: float, geometry_planarity_factor: float, geometry_winding_residual: float, geometry_intersection_merge_factor: float, geometry_aabb_padding_factor: float, surface_maximum_cycle_vertices: int, surface_maximum_surface_count: int, surface_maximum_segment_triangle_tests: int, surface_maximum_triangle_pair_tests: int) -> None
    maximum_candidate_count: int  # read-only
    fibonacci_direction_count: int  # read-only
    sphere_intersection_count: int  # read-only
    least_squares_iteration_count: int  # read-only
    maximum_actionable_ring_size: int  # read-only
    coordination_distance_scale: float  # read-only
    coordination_distance_ratio_minimum: float  # read-only
    coordination_distance_ratio_maximum: float  # read-only
    absolute_center_clearance_angstrom: float  # read-only
    center_covalent_radius_scale: float  # read-only
    minimum_path_atom_clearance: float  # read-only
    minimum_path_bond_clearance: float  # read-only
    broad_phase_skin_angstrom: float  # read-only
    duplicate_tolerance_angstrom: float  # read-only
    retain_candidate_evidence: bool  # read-only
    geometry_absolute_length: float  # read-only
    geometry_relative_length: float  # read-only
    geometry_parameter: float  # read-only
    geometry_machine_epsilon_factor: float  # read-only
    geometry_predicate_guard_factor: float  # read-only
    geometry_planarity_factor: float  # read-only
    geometry_winding_residual: float  # read-only
    geometry_intersection_merge_factor: float  # read-only
    geometry_aabb_padding_factor: float  # read-only
    surface_maximum_cycle_vertices: int  # read-only
    surface_maximum_surface_count: int  # read-only
    surface_maximum_segment_triangle_tests: int  # read-only
    surface_maximum_triangle_pair_tests: int  # read-only
class DonorApproachEvidence():
    neighbour_index: int  # read-only
    metal_donor_neighbour_angle_degrees: Optional[float]  # read-only
class DonorPathEvidence():
    donor_index: int  # read-only
    group_index: int  # read-only
    status: DonorPathStatus  # read-only
    distance_reachable: bool  # read-only
    atom_obstructed: bool  # read-only
    bond_obstructed: bool  # read-only
    target_distance_angstrom: float  # read-only
    distance_angstrom: float  # read-only
    distance_ratio: float  # read-only
    normalized_atom_clearance: float  # read-only
    normalized_bond_clearance: float  # read-only
    definite_piercing_count: int  # read-only
    undetermined_relation_count: int  # read-only
    atom_pair_count: int  # read-only
    atom_aabb_rejected_pair_count: int  # read-only
    bond_pair_count: int  # read-only
    bond_aabb_rejected_pair_count: int  # read-only
    cycle_pair_count: int  # read-only
    cycle_aabb_rejected_pair_count: int  # read-only
    approach_angles: List[DonorApproachEvidence]  # read-only
class DonorPairEvidence():
    first_donor_index: int  # read-only
    second_donor_index: int  # read-only
    donor_separation_angstrom: float  # read-only
    target_distance_sum_angstrom: float  # read-only
    target_distance_difference_angstrom: float  # read-only
    target_shells_intersect: bool  # read-only
    donor_metal_donor_angle_degrees: Optional[float]  # read-only
class PlacementCandidateEvidence():
    coordinates: Coordinate  # read-only
    proposal_kind: PlacementProposalKind  # read-only
    status: PlacementStatus  # read-only
    excluded_large_cycle_count: int  # read-only
    covered_group_count: int  # read-only
    safe_donor_count: int  # read-only
    out_of_range_donor_count: int  # read-only
    atom_obstruction_count: int  # read-only
    bond_obstruction_count: int  # read-only
    definite_piercing_count: int  # read-only
    hard_obstruction_count: int  # read-only
    minimum_normalized_clearance: float  # read-only
    worst_distance_deviation: float  # read-only
    rms_distance_deviation: float  # read-only
    undetermined_relation_count: int  # read-only
    atom_pair_count: int  # read-only
    atom_aabb_rejected_pair_count: int  # read-only
    bond_pair_count: int  # read-only
    bond_aabb_rejected_pair_count: int  # read-only
    cycle_pair_count: int  # read-only
    cycle_aabb_rejected_pair_count: int  # read-only
    displacement_angstrom: float  # read-only
    proposal_ordinal: int  # read-only
    donor_paths: List[DonorPathEvidence]  # read-only
    donor_pairs: List[DonorPairEvidence]  # read-only
class MetalPlacementResult():
    metal_index: int  # read-only
    status: PlacementStatus  # read-only
    original_coordinates: Coordinate  # read-only
    selected_coordinates: Coordinate  # read-only
    moved: bool  # read-only
    candidates_evaluated: int  # read-only
    selected_evidence: PlacementCandidateEvidence  # read-only
    retained_candidates: List[PlacementCandidateEvidence]  # read-only
    excluded_large_cycle_count: int  # read-only
    warning_codes: List[str]  # read-only
class MetalPlacementReport():
    metals: List[MetalPlacementResult]  # read-only
    selected_coordinates: npt.NDArray[np.float64]  # read-only
    warning_codes: List[str]  # read-only
class OptimizationStoppingOptions():
    __init__(self, window: int, maximum_energy_change_kj_mol: float, maximum_atom_displacement_angstrom: float, maximum_rms_gradient_kj_mol_angstrom: float, maximum_gradient_kj_mol_angstrom: float) -> None
    window: int  # read-only
    maximum_energy_change_kj_mol: float  # read-only
    maximum_atom_displacement_angstrom: float  # read-only
    maximum_rms_gradient_kj_mol_angstrom: float  # read-only
    maximum_gradient_kj_mol_angstrom: float  # read-only
class RingScreeningOptions():
    __init__(self, maximum_actionable_ring_size: int, maximum_relevant_cycle_count: int, geometry_absolute_length: float, geometry_relative_length: float, geometry_parameter: float, geometry_machine_epsilon_factor: float, geometry_predicate_guard_factor: float, geometry_planarity_factor: float, geometry_winding_residual: float, geometry_intersection_merge_factor: float, geometry_aabb_padding_factor: float, surface_maximum_cycle_vertices: int, surface_maximum_surface_count: int, surface_maximum_segment_triangle_tests: int, surface_maximum_triangle_pair_tests: int) -> None
    maximum_actionable_ring_size: int  # read-only
    maximum_relevant_cycle_count: int  # read-only
    geometry_absolute_length: float  # read-only
    geometry_relative_length: float  # read-only
    geometry_parameter: float  # read-only
    geometry_machine_epsilon_factor: float  # read-only
    geometry_predicate_guard_factor: float  # read-only
    geometry_planarity_factor: float  # read-only
    geometry_winding_residual: float  # read-only
    geometry_intersection_merge_factor: float  # read-only
    geometry_aabb_padding_factor: float  # read-only
    surface_maximum_cycle_vertices: int  # read-only
    surface_maximum_surface_count: int  # read-only
    surface_maximum_segment_triangle_tests: int  # read-only
    surface_maximum_triangle_pair_tests: int  # read-only
class CoordinationStageOptions():
    __init__(self, forcefield: str, attempt_limit: int, relaxation_steps: int, perturb_sigma: float, trajectory_start: NativeTrajectoryStart, frame_detail: FrameDetail, torsion_singularity_threshold: float, torsion_repair_angle_radians: float, placement: MetalPlacementOptions=...) -> None
    forcefield: str  # read-only
    attempt_limit: int  # read-only
    relaxation_steps: int  # read-only
    perturb_sigma: float  # read-only
    trajectory_start: NativeTrajectoryStart  # read-only
    frame_detail: FrameDetail  # read-only
    torsion_singularity_threshold: float  # read-only
    torsion_repair_angle_radians: float  # read-only
    placement: MetalPlacementOptions  # read-only
class ComplexOptimizationOptions():
    __init__(self, forcefield: str, algorithm: str, epochs: int, steps_per_epoch: int, untangling_attempt_limit: int, perturb_interval: Optional[int], perturb_sigma: float, trajectory_start: NativeTrajectoryStart, frame_detail: FrameDetail, retain_epoch_history: bool, increasing_vdw: bool, vdw_cutoff_start: float, vdw_cutoff_end: float, energy_tolerance: float, stopping: Optional[OptimizationStoppingOptions], torsion_singularity_threshold: float, torsion_repair_angle_radians: float, ring_screening: RingScreeningOptions, convergence_level: ConvergenceLevel=...) -> None
    forcefield: str  # read-only
    algorithm: str  # read-only
    epochs: int  # read-only
    steps_per_epoch: int  # read-only
    untangling_attempt_limit: int  # read-only
    perturb_interval: Optional[int]  # read-only
    perturb_sigma: float  # read-only
    trajectory_start: NativeTrajectoryStart  # read-only
    frame_detail: FrameDetail  # read-only
    retain_epoch_history: bool  # read-only
    increasing_vdw: bool  # read-only
    vdw_cutoff_start: float  # read-only
    vdw_cutoff_end: float  # read-only
    energy_tolerance: float  # read-only
    stopping: Optional[OptimizationStoppingOptions]  # read-only
    torsion_singularity_threshold: float  # read-only
    torsion_repair_angle_radians: float  # read-only
    ring_screening: RingScreeningOptions  # read-only
    convergence_level: ConvergenceLevel  # read-only
class NativeRingFrameEvidence():
    __init__(self, confirmed_piercing_count: int, uncertain_relation_count: Optional[int], ring_scope: Optional[str], max_ring_size: Optional[int], selected_ring_count: Optional[int], excluded_ring_count: Optional[int], candidate_pair_count: Optional[int], aabb_separated_pair_count: Optional[int], exact_pair_count: Optional[int], does_not_pierce_pair_count: Optional[int], scan_complete: Optional[bool]) -> None
    confirmed_piercing_count: int  # read-only
    uncertain_relation_count: Optional[int]  # read-only
    ring_scope: Optional[str]  # read-only
    max_ring_size: Optional[int]  # read-only
    selected_ring_count: Optional[int]  # read-only
    excluded_ring_count: Optional[int]  # read-only
    candidate_pair_count: Optional[int]  # read-only
    aabb_separated_pair_count: Optional[int]  # read-only
    exact_pair_count: Optional[int]  # read-only
    does_not_pierce_pair_count: Optional[int]  # read-only
    scan_complete: Optional[bool]  # read-only
class NativeCoordinationFrameEvidence():
    __init__(self, bond_atom_indices: Optional[Tuple[int, int]], accepted: Optional[bool], pending_bond_count: int, forced: bool, piercing_relation_count: int, undetermined_relation_count: int, excluded_ring_count: int, metal_atom_index: Optional[int], relocation_status: Optional[str], relocation_candidates_evaluated: int, safe_donor_atom_indices: List[int], minimum_normalized_clearance: Optional[float], coordination_distance_deviation: Optional[float]) -> None
    bond_atom_indices: Optional[Tuple[int, int]]  # read-only
    accepted: Optional[bool]  # read-only
    pending_bond_count: int  # read-only
    forced: bool  # read-only
    piercing_relation_count: int  # read-only
    undetermined_relation_count: int  # read-only
    excluded_ring_count: int  # read-only
    metal_atom_index: Optional[int]  # read-only
    relocation_status: Optional[str]  # read-only
    relocation_candidates_evaluated: int  # read-only
    safe_donor_atom_indices: List[int]  # read-only
    minimum_normalized_clearance: Optional[float]  # read-only
    coordination_distance_deviation: Optional[float]  # read-only
class NativeOptimizationFrameEvidence():
    __init__(self, converged: bool, exploded: bool, finite_coordinates: bool, finite_energy: bool, finite_gradients: bool, rms_gradient_kj_mol_angstrom: Optional[float], max_gradient_kj_mol_angstrom: Optional[float], energy_change_kj_mol: Optional[float], max_displacement_angstrom: Optional[float]) -> None
    converged: bool  # read-only
    exploded: bool  # read-only
    finite_coordinates: bool  # read-only
    finite_energy: bool  # read-only
    finite_gradients: bool  # read-only
    rms_gradient_kj_mol_angstrom: Optional[float]  # read-only
    max_gradient_kj_mol_angstrom: Optional[float]  # read-only
    energy_change_kj_mol: Optional[float]  # read-only
    max_displacement_angstrom: Optional[float]  # read-only
class NativeTopologyRevision():
    __init__(self, active_ligand_bond_mask: npt.NDArray[np.uint8], active_coordination_bond_mask: npt.NDArray[np.uint8]) -> None
    active_ligand_bond_mask: npt.NDArray[np.uint8]  # read-only
    active_coordination_bond_mask: npt.NDArray[np.uint8]  # read-only
class NativeTrajectoryFrame():
    __init__(self, coordinates: npt.NDArray[np.float64], stage: NativeTrajectoryStage, event: NativeTrajectoryEvent, component_index: Optional[int], attempt: Optional[int], step: Optional[int], energy_kj_mol: Optional[float], evidence: NativeFrameEvidence, topology_revision: int) -> None
    coordinates: npt.NDArray[np.float64]  # read-only
    stage: NativeTrajectoryStage  # read-only
    event: NativeTrajectoryEvent  # read-only
    component_index: Optional[int]  # read-only
    attempt: Optional[int]  # read-only
    step: Optional[int]  # read-only
    energy_kj_mol: Optional[float]  # read-only
    evidence: NativeFrameEvidence  # read-only
    topology_revision: int  # read-only
class NativeTrajectoryBatch():
    __init__(self, atom_count: int, ligand_bond_count: int, intended_coordination_bond_count: int, start: NativeTrajectoryStart, topology_revisions: List[NativeTopologyRevision], frames: List[NativeTrajectoryFrame], selected_frame_index: Optional[int], terminal_frame_index: Optional[int]) -> None
    atom_count: int  # read-only
    ligand_bond_count: int  # read-only
    intended_coordination_bond_count: int  # read-only
    start: NativeTrajectoryStart  # read-only
    topology_revisions: List[NativeTopologyRevision]  # read-only
    frame_count: int  # read-only
    frames: List[NativeTrajectoryFrame]  # read-only
    selected_frame_index: Optional[int]  # read-only
    terminal_frame_index: Optional[int]  # read-only
class CoordinationStageResult():
    __init__(self, status: NativeStageStatus, selected_coordinates: npt.NDArray[np.float64], terminal_coordinates: npt.NDArray[np.float64], final_active_coordination_mask: npt.NDArray[np.uint8], attempt_limit: int, attempts_completed: int, metal_relocation_attempt_count: int, relocated_metal_indices: List[int], infeasible_metal_indices: List[int], forced_bond_keys: List[Tuple[int, int]], rejected_piercing_trial_count: int, undetermined_trial_count: int, excluded_ring_observation_count: int, warning_codes: List[str], trajectory: NativeTrajectoryBatch, elapsed_seconds: float=0.0) -> None
    status: NativeStageStatus  # read-only
    selected_coordinates: npt.NDArray[np.float64]  # read-only
    terminal_coordinates: npt.NDArray[np.float64]  # read-only
    final_active_coordination_mask: npt.NDArray[np.uint8]  # read-only
    attempt_limit: int  # read-only
    attempts_completed: int  # read-only
    metal_relocation_attempt_count: int  # read-only
    relocated_metal_indices: List[int]  # read-only
    infeasible_metal_indices: List[int]  # read-only
    forced_bond_keys: npt.NDArray[np.int32]  # read-only
    rejected_piercing_trial_count: int  # read-only
    undetermined_trial_count: int  # read-only
    excluded_ring_observation_count: int  # read-only
    warning_codes: List[str]  # read-only
    trajectory: NativeTrajectoryBatch  # read-only
    bond_count: int  # read-only
    placement_report: MetalPlacementReport  # read-only
    elapsed_seconds: float  # read-only
class NativeBondRingFinding():
    __init__(self, ring_index: int, ring_atom_indices: List[int], bond_key: Tuple[int, int], state: NativePiercingState, indeterminacy_causes: List[NativeSegmentCycleIndeterminacy], aabb_separated: bool, surface_complete: bool) -> None
    ring_index: int  # read-only
    ring_atom_indices: List[int]  # read-only
    bond_key: Tuple[int, int]  # read-only
    state: NativePiercingState  # read-only
    indeterminacy_causes: List[NativeSegmentCycleIndeterminacy]  # read-only
    aabb_separated: bool  # read-only
    surface_complete: bool  # read-only
class NativeRingCheckpointReport():
    __init__(self, state: NativePiercingState, scope: NativeRingGraphScope, maximum_actionable_ring_size: int, maximum_relevant_cycle_count: int, relevant_cycle_count: int, selected_ring_count: int, excluded_ring_count: int, active_bond_count: int, candidate_pair_count: int, aabb_separated_pair_count: int, exact_pair_count: int, piercing_pair_count: int, does_not_pierce_pair_count: int, undetermined_pair_count: int, scan_complete: bool, actionable_findings: List[NativeBondRingFinding]) -> None
    state: NativePiercingState  # read-only
    scope: NativeRingGraphScope  # read-only
    maximum_actionable_ring_size: int  # read-only
    maximum_relevant_cycle_count: int  # read-only
    relevant_cycle_count: int  # read-only
    selected_ring_count: int  # read-only
    excluded_ring_count: int  # read-only
    active_bond_count: int  # read-only
    candidate_pair_count: int  # read-only
    aabb_separated_pair_count: int  # read-only
    exact_pair_count: int  # read-only
    piercing_pair_count: int  # read-only
    does_not_pierce_pair_count: int  # read-only
    undetermined_pair_count: int  # read-only
    scan_complete: bool  # read-only
    actionable_findings: List[NativeBondRingFinding]  # read-only
class ComplexOptimizationResult():
    __init__(self, status: NativeStageStatus, selected_coordinates: npt.NDArray[np.float64], terminal_coordinates: npt.NDArray[np.float64], final_active_coordination_mask: npt.NDArray[np.uint8], untangling_attempt_limit: int, untangling_attempts_completed: int, initial_piercing_count: int, final_piercing_count: int, minimum_piercing_count: int, untangling_resolved: bool, selected_frame_index: int, best_epoch: int, final_energy_kj_mol: float, best_energy_kj_mol: float, rms_gradient_kj_mol_angstrom: float, max_gradient_kj_mol_angstrom: float, energy_changes: List[float], max_displacements: List[float], epoch_energies: List[float], exploded: bool, converged: bool, terminal_converged: bool, epochs_completed: int, steps_submitted: int, initialization_steps: int, selected_segment_epochs_completed: int, backend_energy_unit: str, termination_reason: str, warning_codes: List[str], trajectory: NativeTrajectoryBatch, final_checkpoint: NativeRingCheckpointReport, elapsed_seconds: float=0.0) -> None
    status: NativeStageStatus  # read-only
    selected_coordinates: npt.NDArray[np.float64]  # read-only
    terminal_coordinates: npt.NDArray[np.float64]  # read-only
    final_active_coordination_mask: npt.NDArray[np.uint8]  # read-only
    untangling_attempt_limit: int  # read-only
    untangling_attempts_completed: int  # read-only
    initial_piercing_count: int  # read-only
    final_piercing_count: int  # read-only
    minimum_piercing_count: int  # read-only
    untangling_resolved: bool  # read-only
    selected_frame_index: int  # read-only
    best_epoch: int  # read-only
    final_energy_kj_mol: float  # read-only
    best_energy_kj_mol: float  # read-only
    rms_gradient_kj_mol_angstrom: float  # read-only
    max_gradient_kj_mol_angstrom: float  # read-only
    energy_changes: List[float]  # read-only
    max_displacements: List[float]  # read-only
    epoch_energies: List[float]  # read-only
    exploded: bool  # read-only
    converged: bool  # read-only
    terminal_converged: bool  # read-only
    epochs_completed: int  # read-only
    steps_submitted: int  # read-only
    initialization_steps: int  # read-only
    selected_segment_epochs_completed: int  # read-only
    backend_energy_unit: str  # read-only
    termination_reason: str  # read-only
    warning_codes: List[str]  # read-only
    trajectory: NativeTrajectoryBatch  # read-only
    final_checkpoint: NativeRingCheckpointReport  # read-only
    elapsed_seconds: float  # read-only
class ComplexWorkflowResult():
    __init__(self, coordination: CoordinationStageResult, optimization: ComplexOptimizationResult, selected_coordinates: npt.NDArray[np.float64], terminal_coordinates: npt.NDArray[np.float64], final_active_coordination_mask: npt.NDArray[np.uint8], warning_codes: List[str], trajectory: NativeTrajectoryBatch) -> None
    coordination: CoordinationStageResult  # read-only
    optimization: ComplexOptimizationResult  # read-only
    selected_coordinates: npt.NDArray[np.float64]  # read-only
    terminal_coordinates: npt.NDArray[np.float64]  # read-only
    final_active_coordination_mask: npt.NDArray[np.uint8]  # read-only
    warning_codes: List[str]  # read-only
    trajectory: NativeTrajectoryBatch  # read-only
create_coordination_session(session_input: ComplexSessionInput) -> StructureSession
restore_coordination(session: StructureSession, options: CoordinationStageOptions, perturbation_offsets: PerturbationOffsetBatch) -> CoordinationStageResult
optimize_complex(session: StructureSession, options: ComplexOptimizationOptions, untangling_offsets: PerturbationOffsetBatch, optimization_offsets: PerturbationOffsetBatch) -> ComplexOptimizationResult
run_complex_workflow(session: StructureSession, coordination_options: CoordinationStageOptions, optimization_options: ComplexOptimizationOptions, coordination_offsets: PerturbationOffsetBatch, untangling_offsets: PerturbationOffsetBatch, optimization_offsets: PerturbationOffsetBatch) -> ComplexWorkflowResult
run_complex_workflow_from_input(session_input: ComplexSessionInput, coordination_options: CoordinationStageOptions, optimization_options: ComplexOptimizationOptions, coordination_offsets: PerturbationOffsetBatch, untangling_offsets: PerturbationOffsetBatch, optimization_offsets: PerturbationOffsetBatch) -> ComplexWorkflowResult
create_optimization_session(session_input: ComplexSessionInput) -> StructureSession
snapshot_structure(session: StructureSession) -> StructureSnapshot
update_structure_coordinates(session: StructureSession, coordinates: npt.NDArray[np.float64]) -> None
set_ligand_bond_active_mask(session: StructureSession, active_mask: npt.NDArray[np.uint8]) -> None
set_coordination_active_mask(session: StructureSession, active_mask: npt.NDArray[np.uint8]) -> None
covalent_radius(atomic_number: int) -> AtomicRadius
assess_metal_position(session: StructureSession, metal_index: int, candidate: Coordinate, options: MetalPlacementOptions=...) -> PlacementCandidateEvidence
place_metal(session: StructureSession, metal_index: int, options: MetalPlacementOptions=...) -> MetalPlacementResult
place_metals(session: StructureSession, options: MetalPlacementOptions=...) -> MetalPlacementReport
runtime_info() -> RuntimeInfo
seed_random(seed: int) -> None
inspect_rules(molecule: MoleculeData, stage: RuleStage, singularity_threshold: float=..., repair_angle_radians: float=...) -> RulePlan
available_rules(stage: Optional[RuleStage]=...) -> List[RuleDescriptor]
build(molecule: MoleculeData, stereo_warnings: Optional[bool]=...) -> BuildResult
single_optimize(molecule: MoleculeData, forcefield: str, steps: int, singularity_threshold: float=..., repair_angle_radians: float=...) -> SingleOptimizationResult
check_optimization_state(molecule: MoleculeData, forcefield: str, previous_coordinates: Optional[npt.NDArray[np.float64]]=..., previous_energy_kj_mol: Optional[float]=..., singularity_threshold: float=..., repair_angle_radians: float=...) -> OptimizationCheckResult
optimize(molecule: MoleculeData, forcefield: str, algorithm: str, epochs: int, steps_per_epoch: int, perturb_interval: Optional[int]=..., perturbation_offsets: Optional[npt.NDArray[np.float64]]=..., retain_frames: bool=..., retain_epoch_history: bool=..., increasing_vdw: bool=..., vdw_cutoff_start: float=..., vdw_cutoff_end: float=..., energy_tolerance: float=..., stopping_window: Optional[int]=..., maximum_energy_change_kj_mol: float=..., maximum_atom_displacement_angstrom: float=..., maximum_rms_gradient_kj_mol_angstrom: float=..., maximum_gradient_kj_mol_angstrom: float=..., singularity_threshold: float=..., repair_angle_radians: float=..., convergence_level: ConvergenceLevel=...) -> OptimizationResult
```

## `hotpot/cheminfo/rdWrappers/_rd_native.pyi`

```python
generate(molecule_buffers: NativeMoleculeBuffers, options: ETKDGOptions, constraints: CoordinationEmbeddingConstraints | None=...) -> ConformerBatch
runtime_info() -> RuntimeInfo
```

## `hotpot/cheminfo/rules/mathematics/geometry/_geometry_native.pyi`

```python
_default_settings_snapshot() -> Dict[str, Dict[str, Union[float, int]]]
class LineRelationKind():
    INTERSECTING: 'LineRelationKind'  # not specified by stub
    PARALLEL: 'LineRelationKind'  # not specified by stub
    COINCIDENT: 'LineRelationKind'  # not specified by stub
    SKEW: 'LineRelationKind'  # not specified by stub
    DEGENERATE: 'LineRelationKind'  # not specified by stub
    UNDETERMINED: 'LineRelationKind'  # not specified by stub
class PlanarityKind():
    PLANAR: 'PlanarityKind'  # not specified by stub
    NONPLANAR: 'PlanarityKind'  # not specified by stub
    DEGENERATE: 'PlanarityKind'  # not specified by stub
    UNDETERMINED: 'PlanarityKind'  # not specified by stub
class PolygonSimplicity():
    SIMPLE: 'PolygonSimplicity'  # not specified by stub
    SELF_INTERSECTING: 'PolygonSimplicity'  # not specified by stub
    UNDETERMINED: 'PolygonSimplicity'  # not specified by stub
class PointCycleLocation():
    INTERIOR: 'PointCycleLocation'  # not specified by stub
    BOUNDARY: 'PointCycleLocation'  # not specified by stub
    EXTERIOR: 'PointCycleLocation'  # not specified by stub
    UNDETERMINED: 'PointCycleLocation'  # not specified by stub
class SurfaceEmbeddingState():
    EMBEDDED: 'SurfaceEmbeddingState'  # not specified by stub
    PROVEN_NON_EMBEDDED: 'SurfaceEmbeddingState'  # not specified by stub
    CONSTRUCTION_UNDETERMINED: 'SurfaceEmbeddingState'  # not specified by stub
class NonplanarSurfaceCause():
    INCOMPLETE_SURFACE_FAMILY: 'NonplanarSurfaceCause'  # not specified by stub
    SURFACE_CONSTRUCTION: 'NonplanarSurfaceCause'  # not specified by stub
class PiercingState():
    PIERCES: 'PiercingState'  # not specified by stub
    DOES_NOT_PIERCE: 'PiercingState'  # not specified by stub
    UNDETERMINED: 'PiercingState'  # not specified by stub
class DetailLevel():
    STATE_ONLY: 'DetailLevel'  # not specified by stub
    ACTIONABLE: 'DetailLevel'  # not specified by stub
    FULL: 'DetailLevel'  # not specified by stub
class SegmentCyclePair():
    __init__(self, segment_index: int, cycle_index: int) -> None
    segment_index: int  # read-only
    cycle_index: int  # read-only
class SegmentCycleFeature():
    TRANSVERSE_INTERIOR: 'SegmentCycleFeature'  # not specified by stub
    LINE_EXTENSION_INTERIOR: 'SegmentCycleFeature'  # not specified by stub
    CYCLE_EDGE_CONTACT: 'SegmentCycleFeature'  # not specified by stub
    CYCLE_VERTEX_CONTACT: 'SegmentCycleFeature'  # not specified by stub
    SEGMENT_ENDPOINT_CONTACT: 'SegmentCycleFeature'  # not specified by stub
    COPLANAR_CONTACT: 'SegmentCycleFeature'  # not specified by stub
class SegmentCycleIndeterminacy():
    NONFINITE_INPUT: 'SegmentCycleIndeterminacy'  # not specified by stub
    NUMERIC_BAND: 'SegmentCycleIndeterminacy'  # not specified by stub
    TOLERANCE_DOMAIN: 'SegmentCycleIndeterminacy'  # not specified by stub
    DEGENERATE_CYCLE: 'SegmentCycleIndeterminacy'  # not specified by stub
    DEGENERATE_SEGMENT: 'SegmentCycleIndeterminacy'  # not specified by stub
    DEGENERATE_TRIANGLE: 'SegmentCycleIndeterminacy'  # not specified by stub
    SELF_INTERSECTION: 'SegmentCycleIndeterminacy'  # not specified by stub
    SURFACE_DISAGREEMENT: 'SegmentCycleIndeterminacy'  # not specified by stub
    INCOMPLETE_SURFACE_FAMILY: 'SegmentCycleIndeterminacy'  # not specified by stub
    SURFACE_CONSTRUCTION: 'SegmentCycleIndeterminacy'  # not specified by stub
class CycleSurfaceModel():
    PLANAR_POLYGON: 'CycleSurfaceModel'  # not specified by stub
    VERTEX_TRIANGULATION_FAMILY: 'CycleSurfaceModel'  # not specified by stub
class NumericTolerances():
    __init__(self, absolute_length: float, relative_length: float, parameter: float, machine_epsilon_factor: float, predicate_guard_factor: float, planarity_factor: float, winding_residual: float, intersection_merge_factor: float, aabb_padding_factor: float) -> None
    absolute_length: float  # read-only
    relative_length: float  # read-only
    parameter: float  # read-only
    machine_epsilon_factor: float  # read-only
    predicate_guard_factor: float  # read-only
    planarity_factor: float  # read-only
    winding_residual: float  # read-only
    intersection_merge_factor: float  # read-only
    aabb_padding_factor: float  # read-only
class LineRelation():
    kind: LineRelationKind  # read-only
    distance: Optional[float]  # read-only
    parallel_measure: float  # read-only
class PointSegmentMeasurement():
    distance: float  # read-only
    closest_point: Tuple[float, float, float]  # read-only
    parameter: float  # read-only
    segment_degenerate: bool  # read-only
class SegmentSegmentMeasurement():
    distance: float  # read-only
    first_closest_point: Tuple[float, float, float]  # read-only
    second_closest_point: Tuple[float, float, float]  # read-only
    first_parameter: float  # read-only
    second_parameter: float  # read-only
    first_segment_degenerate: bool  # read-only
    second_segment_degenerate: bool  # read-only
class PlanarityMeasurement():
    kind: PlanarityKind  # read-only
    centroid: Tuple[float, float, float]  # read-only
    normal: Optional[Tuple[float, float, float]]  # read-only
    singular_values: Tuple[float, float, float]  # read-only
    maximum_deviation: float  # read-only
    rms_deviation: float  # read-only
    length_scale: float  # read-only
    length_tolerance: float  # read-only
class PreparedPlanarCycle():
    coordinates(self) -> List[Tuple[float, float, float]]
    planarity(self) -> PlanarityMeasurement
    tolerances(self) -> NumericTolerances
    projection(self) -> List[Tuple[float, float]]
    simplicity(self) -> PolygonSimplicity
    has_planar_surface(self) -> bool
    has_simple_planar_surface(self) -> bool
class SurfaceEnumerationLimits():
    __init__(self, maximum_cycle_vertices: int, maximum_surface_count: int, maximum_segment_triangle_tests: int, maximum_triangle_pair_tests: int) -> None
    maximum_cycle_vertices: int  # read-only
    maximum_surface_count: int  # read-only
    maximum_segment_triangle_tests: int  # read-only
    maximum_triangle_pair_tests: int  # read-only
class PreparedNonplanarSurfaceFamily():
    coordinates(self) -> List[Tuple[float, float, float]]
    tolerances(self) -> NumericTolerances
    limits(self) -> SurfaceEnumerationLimits
    enumeration_complete(self) -> bool
    enumerated_surface_count(self) -> int
    embedded_surface_count(self) -> int
    proven_non_embedded_surface_count(self) -> int
    construction_undetermined_count(self) -> int
    triangle_pair_tests_used(self) -> int
    causes(self) -> FrozenSet[NonplanarSurfaceCause]
    surface_states(self) -> List[SurfaceEmbeddingState]
class PreparedCycle():
    coordinates(self) -> List[Tuple[float, float, float]]
    bounds(self) -> np.ndarray
    planarity(self) -> PlanarityMeasurement
    tolerances(self) -> NumericTolerances
    limits(self) -> SurfaceEnumerationLimits
    uses_nonplanar_surface_family(self) -> bool
class PreparedCycleBatch():
    cycle(self, index: int) -> PreparedCycle
    coordinate_count(self) -> int
    cycle_count(self) -> int
    coordinates(self) -> List[Tuple[float, float, float]]
    cycle_indices(self) -> List[int]
    cycle_offsets(self) -> List[int]
    cycle_bounds(self) -> np.ndarray
class ClosestCycleEdge():
    edge_index: int  # read-only
    distance: float  # read-only
class SurfaceFamilyEvidence():
    enumeration_complete: bool  # read-only
    enumerated_surface_count: int  # read-only
    embedded_surface_count: int  # read-only
    proven_non_embedded_surface_count: int  # read-only
    construction_undetermined_count: int  # read-only
    intersecting_surface_count: int  # read-only
    non_piercing_surface_count: int  # read-only
    evaluation_undetermined_count: int  # read-only
    segment_triangle_tests_used: int  # read-only
    triangle_pair_tests_used: int  # read-only
class SegmentCycleRelation():
    state: PiercingState  # read-only
    features: List[SegmentCycleFeature]  # read-only
    indeterminacy_causes: List[SegmentCycleIndeterminacy]  # read-only
    surface_model: Optional[CycleSurfaceModel]  # read-only
    intersection_points: List[Tuple[float, float, float]]  # read-only
    closest_boundary_edge: Optional[ClosestCycleEdge]  # read-only
    surface_evidence: SurfaceFamilyEvidence  # read-only
class SegmentCycleScreening():
    state: PiercingState  # read-only
    relation: Optional[SegmentCycleRelation]  # read-only
    aabb_separated: bool  # read-only
    surface_complete: bool  # read-only
class SegmentCycleBatch():
    detail(self) -> DetailLevel
    requested_pair_count(self) -> int
    evaluated_pair_count(self) -> int
    aabb_separated_pair_count(self) -> int
    exact_pair_count(self) -> int
    piercing_pair_count(self) -> int
    does_not_pierce_pair_count(self) -> int
    undetermined_pair_count(self) -> int
    scan_complete(self) -> bool
    states(self) -> List[PiercingState]
    aabb_separated(self) -> np.ndarray
    surface_complete(self) -> np.ndarray
    relation_positions(self) -> List[int]
    relations(self) -> List[SegmentCycleRelation]
measure_planarity(cycle: np.ndarray, tolerances: NumericTolerances) -> PlanarityMeasurement
prepare_planar_cycle(cycle: np.ndarray, tolerances: NumericTolerances) -> PreparedPlanarCycle
prepare_nonplanar_surface_family(cycle: np.ndarray, tolerances: NumericTolerances, limits: SurfaceEnumerationLimits) -> PreparedNonplanarSurfaceFamily
prepare_cycle(cycle: np.ndarray, tolerances: NumericTolerances, limits: SurfaceEnumerationLimits) -> PreparedCycle
prepare_cycles(coordinates: np.ndarray, cycle_indices: np.ndarray, cycle_offsets: np.ndarray, tolerances: NumericTolerances, limits: SurfaceEnumerationLimits) -> PreparedCycleBatch
determine_segment_cycle_relations(cycles: PreparedCycleBatch, segments: np.ndarray, candidate_pairs: np.ndarray) -> List[SegmentCycleRelation]
screen_segments(cycles: PreparedCycleBatch, segments: np.ndarray, candidate_pairs: np.ndarray, detail: DetailLevel, stop_after_confirmed: bool=False) -> SegmentCycleBatch
locate_point_in_planar_cycle(point: np.ndarray, cycle: PreparedPlanarCycle, plane_origin: np.ndarray, plane_normal: np.ndarray) -> PointCycleLocation
closest_cycle_edge(cycle: Union[PreparedPlanarCycle, PreparedCycle], segment_start: np.ndarray, segment_end: np.ndarray) -> Optional[ClosestCycleEdge]
determine_planar_segment_cycle_relation(segment_start: np.ndarray, segment_end: np.ndarray, cycle: PreparedPlanarCycle) -> SegmentCycleRelation
planar_segment_cycle_relations(segments: np.ndarray, cycle: PreparedPlanarCycle) -> List[SegmentCycleRelation]
planar_segment_cycle_screenings(segments: np.ndarray, cycle: PreparedPlanarCycle) -> List[SegmentCycleScreening]
determine_nonplanar_segment_cycle_relation(segment_start: np.ndarray, segment_end: np.ndarray, family: PreparedNonplanarSurfaceFamily) -> SegmentCycleRelation
nonplanar_segment_cycle_relations(segments: np.ndarray, family: PreparedNonplanarSurfaceFamily) -> List[SegmentCycleRelation]
nonplanar_segment_cycle_screenings(segments: np.ndarray, family: PreparedNonplanarSurfaceFamily) -> List[SegmentCycleScreening]
determine_segment_cycle_relation(segment_start: np.ndarray, segment_end: np.ndarray, cycle: PreparedCycle) -> SegmentCycleRelation
segment_cycle_relations(segments: np.ndarray, cycle: PreparedCycle) -> List[SegmentCycleRelation]
segment_cycle_screenings(segments: np.ndarray, cycle: PreparedCycle) -> List[SegmentCycleScreening]
determine_line_relation(first_origin: np.ndarray, first_direction: np.ndarray, second_origin: np.ndarray, second_direction: np.ndarray, tolerances: NumericTolerances) -> LineRelation
line_distance(first_origin: np.ndarray, first_direction: np.ndarray, second_origin: np.ndarray, second_direction: np.ndarray, tolerances: NumericTolerances) -> float
point_segment_measurement(point: np.ndarray, start: np.ndarray, end: np.ndarray, tolerances: NumericTolerances) -> PointSegmentMeasurement
point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray, tolerances: NumericTolerances) -> float
segment_segment_measurement(first_start: np.ndarray, first_end: np.ndarray, second_start: np.ndarray, second_end: np.ndarray, tolerances: NumericTolerances) -> SegmentSegmentMeasurement
segment_segment_distance(first_start: np.ndarray, first_end: np.ndarray, second_start: np.ndarray, second_end: np.ndarray, tolerances: NumericTolerances) -> float
point_pair_distances(coordinates: np.ndarray, pair_indices: Optional[np.ndarray]=...) -> Tuple[np.ndarray, np.ndarray]
find_point_pairs_below_distance(coordinates: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]
aabb_bounds(coordinates: np.ndarray) -> np.ndarray
aabb_separation_mask(first_bounds: np.ndarray, second_bounds: np.ndarray, paddings: np.ndarray) -> np.ndarray
segment_aabb_separation_mask(segments: np.ndarray, target_bounds: np.ndarray, paddings: np.ndarray) -> np.ndarray
aabb_candidate_pairs(first_bounds: np.ndarray, second_bounds: np.ndarray, padding: float) -> np.ndarray
```

## `hotpot/cheminfo/rules/mathematics/graph/_graph_native.pyi`

```python
class RelevantCycleLimitExceeded(RuntimeError):
relevant_cycles(edges: Sequence[Tuple[int, int]], max_size: Optional[int]=..., max_cycles: Optional[int]=...) -> List[List[int]]
cycle_basis(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> List[List[int]]
all_cycle_basis(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> List[List[List[int]]]
```

## `hotpot/cheminfo/structure/_fact_native.pyi`

```python
_configure(declarations: dict[str, Any]) -> None
_install_atom_dispatch(atom_type: type, eager_setter: Any, sparse_commit: Any, foreign_error: type, detached_error: type) -> None
_commit_atom_scalar(atom: Any, field_index: int, value: Any, *, producer: int, present: bool=True) -> None
_commit_atom_facts(mol: Any, update: _AtomFactUpdate) -> None
_commit_atom_fields(atom: Any, columns: dict[str, Any], *, producer: int, present: bool=True) -> None
_commit_atom_columns(mol: Any, atoms: tuple[Any, ...] | None, columns: dict[str, Any]) -> _AtomFactUpdate
_write_prepared(table: Any, selection: Any, prepared: Any, transaction: Any=None) -> None
_restore_write(state: Any) -> None
_capture_status(table: Any, rows: Iterable[int], groups: Iterable[int]) -> Any
_update_dependency_seeds(mol: Any, groups: tuple[int, ...], uids: tuple[int, ...], present: bool) -> None
_clear_dependency_seeds(mol: Any, group: int, rows: Iterable[int]) -> None
_scalar_value(value: Any) -> float
_normalize_atom_scalar(source: Any, row: int, atom_uid: int, field: str, value: Any, producer: int, present: bool) -> _AtomFactUpdate
_normalize_atom_facts(source: Any, rows: Iterable[int], uids: Iterable[int], columns: dict[str, Any]) -> _AtomFactUpdate
_normalize_atom_fields(source: Any, rows: Iterable[int], uids: Iterable[int], columns: dict[str, Any], producer: int, present: bool) -> _AtomFactUpdate
```
