"""Python boundary for workflow-scoped native force-field sessions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

from ..obWrappers.native import _native_module
from .native_packing import ComplexSessionInput, pack_complex_session_input
from .trajectory import TrajectoryStart


if TYPE_CHECKING:
    from ..core import Molecule
    from ..obWrappers import _ob_native


__all__ = (
    "ComplexOptimizationOptions",
    "CoordinationStageOptions",
    "FrameDetail",
    "OptimizationStoppingOptions",
    "StructureSnapshot",
    "create_coordination_session",
    "create_optimization_session",
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
class CoordinationStageOptions:
    forcefield: str = "UFF"
    attempt_limit: int = 20
    relaxation_steps: int = 100
    perturb_sigma: float = 0.5
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION
    frame_detail: FrameDetail = FrameDetail.NONE


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
