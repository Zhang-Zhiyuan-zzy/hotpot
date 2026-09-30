"""Python boundary for workflow-scoped native force-field sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..geometry.settings import DEFAULT_GEOMETRY_SETTINGS, GeometrySettings
from ..obWrappers.native import _native_module
from ..obWrappers.settings import (
    TORSION_REPAIR_ANGLE_RADIANS,
    TORSION_SINGULARITY_THRESHOLD,
)
from .native_packing import ComplexSessionInput, pack_complex_session_input
from .native_reports import (
    CoordinationStageResult as NativeCoordinationStageResult,
    _coordination_stage_result,
)
from .trajectory import TrajectoryStart

if TYPE_CHECKING:
    from ..core import Molecule
    from ..obWrappers import _ob_native


__all__ = (
    "ComplexOptimizationOptions",
    "CoordinationStageOptions",
    "FrameDetail",
    "MetalPlacementOptions",
    "OptimizationStoppingOptions",
    "StructureSnapshot",
    "assess_metal_position",
    "create_coordination_session",
    "create_optimization_session",
    "place_metal",
    "place_metals",
    "restore_coordination",
    "set_coordination_active_mask",
    "set_ligand_bond_active_mask",
    "snapshot_structure",
    "update_structure_coordinates",
)


class FrameDetail(str, Enum):
    """Additional native diagnostic frames requested by a stage.

    ``NONE`` suppresses optional diagnostics only; stage-boundary, selected,
    terminal, and other required factual frames remain part of the contract.
    """

    NONE = "none"
    OPTIMIZATION = "optimization"
    ALL_ATTEMPTS = "all_attempts"


@dataclass(frozen=True)
class OptimizationStoppingOptions:
    window: int = 5
    maximum_energy_change_kj_mol: float = 1.0e-4
    maximum_atom_displacement_angstrom: float = 1.0e-4
    maximum_rms_gradient_kj_mol_angstrom: float = 1.0
    maximum_gradient_kj_mol_angstrom: float = 5.0


@dataclass(frozen=True)
class MetalPlacementOptions:
    """Scientific policy for deterministic native metal placement."""

    maximum_candidate_count: int = 96
    fibonacci_direction_count: int = 32
    sphere_intersection_count: int = 24
    least_squares_iteration_count: int = 16
    maximum_actionable_ring_size: int = 16
    coordination_distance_scale: float = 1.0
    coordination_distance_ratio_minimum: float = 0.70
    coordination_distance_ratio_maximum: float = 1.50
    absolute_center_clearance_angstrom: float = 0.50
    center_covalent_radius_scale: float = 0.55
    minimum_path_atom_clearance: float = 0.35
    minimum_path_bond_clearance: float = 0.50
    broad_phase_skin_angstrom: float = 0.25
    duplicate_tolerance_angstrom: float = 1.0e-8
    retain_candidate_evidence: bool = False
    geometry_settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS


_DEFAULT_METAL_PLACEMENT_OPTIONS = MetalPlacementOptions()


@dataclass(frozen=True)
class CoordinationStageOptions:
    forcefield: str = "UFF"
    attempt_limit: int = 20
    relaxation_steps: int = 100
    perturb_sigma: float = 0.5
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION
    frame_detail: FrameDetail = FrameDetail.NONE
    torsion_singularity_threshold: float = TORSION_SINGULARITY_THRESHOLD
    torsion_repair_angle_radians: float = TORSION_REPAIR_ANGLE_RADIANS
    placement: MetalPlacementOptions = field(
        default_factory=MetalPlacementOptions
    )


@dataclass(frozen=True)
class ComplexOptimizationOptions:
    forcefield: str = "UFF"
    algorithm: str = "conjugate"
    epochs: int = 100
    steps_per_epoch: int = 100
    untangling_attempt_limit: int = 30
    perturb_interval: Optional[int] = None
    perturb_sigma: float = 0.5
    trajectory_start: TrajectoryStart = TrajectoryStart.COMPLEX_UNTANGLING
    frame_detail: FrameDetail = FrameDetail.NONE
    retain_epoch_history: bool = False
    increasing_vdw: bool = False
    vdw_cutoff_start: float = 0.0
    vdw_cutoff_end: float = 12.5
    energy_tolerance: float = 1.0e-6
    stopping: Optional[OptimizationStoppingOptions] = None


@dataclass(frozen=True)
class StructureSnapshot:
    coordinates: NDArray[np.float64]
    active_ligand_bond_mask: NDArray[np.uint8]
    active_coordination_mask: NDArray[np.uint8]
    component_ids: NDArray[np.int32]
    ligand_bond_count: int
    active_bond_count: int
    coordinate_revision: int
    topology_revision: int


def _readonly_copy(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    copied = np.array(array, dtype=dtype, order="C", copy=True)
    copied.setflags(write=False)
    return copied


def _native_session_input(
    source: Union["Molecule", ComplexSessionInput],
) -> "_ob_native.ComplexSessionInput":
    buffers = (
        source
        if isinstance(source, ComplexSessionInput)
        else pack_complex_session_input(source)
    )
    native = _native_module()
    return native.ComplexSessionInput(
        schema_version=buffers.schema_version,
        atomic_numbers=buffers.atomic_numbers,
        formal_charges=buffers.formal_charges,
        partial_charges=buffers.partial_charges,
        coordinates=buffers.coordinates,
        atom_aromatic=buffers.atom_aromatic,
        ligand_bond_indices=buffers.ligand_bond_indices,
        ligand_bond_orders=buffers.ligand_bond_orders,
        ligand_bond_kinds=buffers.ligand_bond_kinds,
        ligand_bond_aromatic=buffers.ligand_bond_aromatic,
        metal_indices=buffers.metal_indices,
        intended_coordination_bonds=buffers.intended_coordination_bonds,
        intended_coordination_orders=buffers.intended_coordination_orders,
        intended_coordination_kinds=buffers.intended_coordination_kinds,
        unit_cell=buffers.unit_cell,
    )


def create_coordination_session(
    source: Union["Molecule", ComplexSessionInput],
) -> "_ob_native.StructureSession":
    """Create a session with every intended coordination bond inactive."""
    native = _native_module()
    return native.create_coordination_session(_native_session_input(source))


def create_optimization_session(
    source: Union["Molecule", ComplexSessionInput],
) -> "_ob_native.StructureSession":
    """Create a session with every intended coordination bond active."""
    native = _native_module()
    return native.create_optimization_session(_native_session_input(source))


def snapshot_structure(
    session: "_ob_native.StructureSession",
) -> StructureSnapshot:
    """Return a detached snapshot of current session coordinates and topology."""
    snapshot = _native_module().snapshot_structure(session)
    return StructureSnapshot(
        coordinates=_readonly_copy(snapshot.coordinates, np.dtype(np.float64)),
        active_ligand_bond_mask=_readonly_copy(
            snapshot.active_ligand_bond_mask,
            np.dtype(np.uint8),
        ),
        active_coordination_mask=_readonly_copy(
            snapshot.active_coordination_mask,
            np.dtype(np.uint8),
        ),
        component_ids=_readonly_copy(
            snapshot.component_ids,
            np.dtype(np.int32),
        ),
        ligand_bond_count=snapshot.ligand_bond_count,
        active_bond_count=snapshot.active_bond_count,
        coordinate_revision=snapshot.coordinate_revision,
        topology_revision=snapshot.topology_revision,
    )


def update_structure_coordinates(
    session: "_ob_native.StructureSession",
    coordinates: NDArray[np.float64],
) -> None:
    """Replace session coordinates without exposing its mutable native storage."""
    _native_module().update_structure_coordinates(
        session,
        np.ascontiguousarray(coordinates, dtype=np.float64),
    )


def set_coordination_active_mask(
    session: "_ob_native.StructureSession",
    active_mask: NDArray[np.uint8],
) -> None:
    """Apply one complete intended-coordination topology state."""
    _native_module().set_coordination_active_mask(
        session,
        np.ascontiguousarray(active_mask, dtype=np.uint8),
    )


def set_ligand_bond_active_mask(
    session: "_ob_native.StructureSession",
    active_mask: NDArray[np.uint8],
) -> None:
    """Apply one complete ligand-covalent topology state."""
    _native_module().set_ligand_bond_active_mask(
        session,
        np.ascontiguousarray(active_mask, dtype=np.uint8),
    )


def _native_metal_placement_options(
    options: MetalPlacementOptions,
) -> "_ob_native.MetalPlacementOptions":
    return _native_module().MetalPlacementOptions(
        maximum_candidate_count=options.maximum_candidate_count,
        fibonacci_direction_count=options.fibonacci_direction_count,
        sphere_intersection_count=options.sphere_intersection_count,
        least_squares_iteration_count=options.least_squares_iteration_count,
        maximum_actionable_ring_size=options.maximum_actionable_ring_size,
        coordination_distance_scale=options.coordination_distance_scale,
        coordination_distance_ratio_minimum=(
            options.coordination_distance_ratio_minimum
        ),
        coordination_distance_ratio_maximum=(
            options.coordination_distance_ratio_maximum
        ),
        absolute_center_clearance_angstrom=(
            options.absolute_center_clearance_angstrom
        ),
        center_covalent_radius_scale=options.center_covalent_radius_scale,
        minimum_path_atom_clearance=options.minimum_path_atom_clearance,
        minimum_path_bond_clearance=options.minimum_path_bond_clearance,
        broad_phase_skin_angstrom=options.broad_phase_skin_angstrom,
        duplicate_tolerance_angstrom=options.duplicate_tolerance_angstrom,
        retain_candidate_evidence=options.retain_candidate_evidence,
        geometry_absolute_length=(
            options.geometry_settings.tolerance.absolute_length
        ),
        geometry_relative_length=(
            options.geometry_settings.tolerance.relative_length
        ),
        geometry_parameter=options.geometry_settings.tolerance.parameter,
        geometry_machine_epsilon_factor=(
            options.geometry_settings.tolerance.machine_epsilon_factor
        ),
        geometry_predicate_guard_factor=(
            options.geometry_settings.tolerance.predicate_guard_factor
        ),
        geometry_planarity_factor=(
            options.geometry_settings.tolerance.planarity_factor
        ),
        geometry_winding_residual=(
            options.geometry_settings.tolerance.winding_residual
        ),
        geometry_intersection_merge_factor=(
            options.geometry_settings.tolerance.intersection_merge_factor
        ),
        geometry_aabb_padding_factor=(
            options.geometry_settings.tolerance.aabb_padding_factor
        ),
        surface_maximum_cycle_vertices=(
            options.geometry_settings.surface.maximum_cycle_vertices
        ),
        surface_maximum_surface_count=(
            options.geometry_settings.surface.maximum_surface_count
        ),
        surface_maximum_segment_triangle_tests=(
            options.geometry_settings.surface.maximum_segment_triangle_tests
        ),
        surface_maximum_triangle_pair_tests=(
            options.geometry_settings.surface.maximum_triangle_pair_tests
        ),
    )


def assess_metal_position(
    session: "_ob_native.StructureSession",
    metal_index: int,
    coordinates: tuple[float, float, float],
    *,
    options: MetalPlacementOptions = _DEFAULT_METAL_PLACEMENT_OPTIONS,
) -> "_ob_native.PlacementCandidateEvidence":
    """Evaluate one metal coordinate without mutating the native session."""
    return _native_module().assess_metal_position(
        session,
        metal_index,
        coordinates,
        _native_metal_placement_options(options),
    )


def place_metal(
    session: "_ob_native.StructureSession",
    metal_index: int,
    *,
    options: MetalPlacementOptions = _DEFAULT_METAL_PLACEMENT_OPTIONS,
) -> "_ob_native.MetalPlacementResult":
    """Select and commit one metal coordinate before coordination bonds form."""
    return _native_module().place_metal(
        session,
        metal_index,
        _native_metal_placement_options(options),
    )


def place_metals(
    session: "_ob_native.StructureSession",
    *,
    options: MetalPlacementOptions = _DEFAULT_METAL_PLACEMENT_OPTIONS,
) -> "_ob_native.MetalPlacementReport":
    """Place every declared metal with intended donors in stable index order."""
    return _native_module().place_metals(
        session,
        _native_metal_placement_options(options),
    )


def _native_stopping_options(
    options: OptimizationStoppingOptions,
) -> "_ob_native.OptimizationStoppingOptions":
    return _native_module().OptimizationStoppingOptions(
        window=options.window,
        maximum_energy_change_kj_mol=options.maximum_energy_change_kj_mol,
        maximum_atom_displacement_angstrom=(
            options.maximum_atom_displacement_angstrom
        ),
        maximum_rms_gradient_kj_mol_angstrom=(
            options.maximum_rms_gradient_kj_mol_angstrom
        ),
        maximum_gradient_kj_mol_angstrom=(
            options.maximum_gradient_kj_mol_angstrom
        ),
    )


def _native_coordination_stage_options(
    options: CoordinationStageOptions,
) -> "_ob_native.CoordinationStageOptions":
    native = _native_module()
    return native.CoordinationStageOptions(
        forcefield=options.forcefield,
        attempt_limit=options.attempt_limit,
        relaxation_steps=options.relaxation_steps,
        perturb_sigma=options.perturb_sigma,
        trajectory_start=getattr(
            native.NativeTrajectoryStart,
            options.trajectory_start.name,
        ),
        frame_detail=getattr(native.FrameDetail, options.frame_detail.name),
        torsion_singularity_threshold=options.torsion_singularity_threshold,
        torsion_repair_angle_radians=options.torsion_repair_angle_radians,
        placement=_native_metal_placement_options(options.placement),
    )


def _native_complex_optimization_options(
    options: ComplexOptimizationOptions,
) -> "_ob_native.ComplexOptimizationOptions":
    native = _native_module()
    stopping = (
        None
        if options.stopping is None
        else _native_stopping_options(options.stopping)
    )
    return native.ComplexOptimizationOptions(
        forcefield=options.forcefield,
        algorithm=options.algorithm,
        epochs=options.epochs,
        steps_per_epoch=options.steps_per_epoch,
        untangling_attempt_limit=options.untangling_attempt_limit,
        perturb_interval=options.perturb_interval,
        perturb_sigma=options.perturb_sigma,
        trajectory_start=getattr(
            native.NativeTrajectoryStart,
            options.trajectory_start.name,
        ),
        frame_detail=getattr(native.FrameDetail, options.frame_detail.name),
        retain_epoch_history=options.retain_epoch_history,
        increasing_vdw=options.increasing_vdw,
        vdw_cutoff_start=options.vdw_cutoff_start,
        vdw_cutoff_end=options.vdw_cutoff_end,
        energy_tolerance=options.energy_tolerance,
        stopping=stopping,
    )


def restore_coordination(
    session: "_ob_native.StructureSession",
    perturbation_offsets: NDArray[np.float64],
    *,
    options: CoordinationStageOptions = CoordinationStageOptions(),
) -> NativeCoordinationStageResult:
    """Run native Stage 2 with an explicit, deterministic offset schedule."""
    native = _native_module()
    offsets = native.PerturbationOffsetBatch(
        np.ascontiguousarray(perturbation_offsets, dtype=np.float64)
    )
    result = native.restore_coordination(
        session,
        _native_coordination_stage_options(options),
        offsets,
    )
    return _coordination_stage_result(result)
