"""Hotpot-molecule facade for native Open Babel force-field operations."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from .contracts import (
    OptimizationFrame,
    OptimizationReport,
)
from .native import _native_module, _native_molecule_data
from .operation import single_optimize
from .reports import _execution_report
from .settings import (
    TORSION_REPAIR_ANGLE_RADIANS,
    TORSION_SINGULARITY_THRESHOLD,
)


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("optimize", "single_optimize")


def optimize(
    mol: "Molecule",
    forcefield: str,
    *,
    algorithm: str = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 5,
    perturb_interval: Optional[int] = None,
    perturbation_offsets: Optional[np.ndarray] = None,
    retain_frames: bool = False,
    retain_epoch_history: bool = False,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 1.0,
    vdw_cutoff_end: float = 10.0,
    energy_tolerance: float = 1.0e-6,
    stopping_window: Optional[int] = None,
    maximum_energy_change_kj_mol: float = 1.0e-4,
    maximum_atom_displacement_angstrom: float = 1.0e-4,
    maximum_rms_gradient_kj_mol_angstrom: float = 1.0,
    maximum_gradient_kj_mol_angstrom: float = 5.0,
    singularity_threshold: float = TORSION_SINGULARITY_THRESHOLD,
    repair_angle_radians: float = TORSION_REPAIR_ANGLE_RADIANS,
) -> OptimizationReport:
    """Optimize ``mol`` using the direct native Open Babel backend."""
    offsets = None
    if perturbation_offsets is not None:
        offsets = np.ascontiguousarray(
            perturbation_offsets,
            dtype=np.float64,
        )
    result = _native_module().optimize(
        _native_molecule_data(mol),
        forcefield,
        algorithm,
        epochs,
        steps_per_epoch,
        perturb_interval,
        offsets,
        retain_frames,
        retain_epoch_history,
        increasing_vdw,
        vdw_cutoff_start,
        vdw_cutoff_end,
        energy_tolerance,
        stopping_window,
        maximum_energy_change_kj_mol,
        maximum_atom_displacement_angstrom,
        maximum_rms_gradient_kj_mol_angstrom,
        maximum_gradient_kj_mol_angstrom,
        singularity_threshold,
        repair_angle_radians,
    )
    coordinates = np.asarray(result.coordinates, dtype=np.float64)
    terminal_coordinates = np.asarray(
        result.terminal_coordinates,
        dtype=np.float64,
    )
    mol.coordinates = coordinates
    frames = tuple(
        OptimizationFrame(
            coordinates=np.asarray(frame.coordinates, dtype=np.float64),
            energy=frame.energy,
            rms_gradient=frame.rms_gradient,
            max_gradient=frame.max_gradient,
            exploded=frame.exploded,
            converged=frame.converged,
            epoch_index=frame.epoch_index,
            segment_epochs_completed=frame.segment_epochs_completed,
            segment_index=frame.segment_index,
            energy_change=frame.energy_change,
            max_displacement=frame.max_displacement,
        )
        for frame in result.frames
    )
    return OptimizationReport(
        coordinates=coordinates,
        terminal_coordinates=terminal_coordinates,
        frames=frames,
        selected_frame_index=result.selected_frame_index,
        best_epoch=result.best_epoch,
        final_energy=result.final_energy,
        best_energy=result.best_energy,
        rms_gradient=result.rms_gradient,
        max_gradient=result.max_gradient,
        exploded=result.exploded,
        converged=result.converged,
        epochs_completed=result.epochs_completed,
        steps_submitted=result.steps_submitted,
        initialization_steps=result.initialization_steps,
        selected_segment_epochs_completed=(
            result.selected_segment_epochs_completed
        ),
        energy_unit="kJ/mol",
        backend_energy_unit=result.backend_energy_unit,
        termination_reason=result.termination_reason,
        terminal_converged=result.terminal_converged,
        energy_changes=tuple(result.energy_changes),
        max_displacements=tuple(result.max_displacements),
        epoch_energies=tuple(result.epoch_energies),
        rules=_execution_report(result.rules),
    )
