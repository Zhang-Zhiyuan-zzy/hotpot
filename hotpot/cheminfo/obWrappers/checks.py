"""Read-only numerical checks for Open Babel force-field states."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .contracts import OptimizationCheckReport, OptimizationFailure
from .native import _native_module, _native_molecule_data
from .reports import _execution_report
from .settings import (
    TORSION_REPAIR_ANGLE_RADIANS,
    TORSION_SINGULARITY_THRESHOLD,
)


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("check_optimization_state",)


def check_optimization_state(
    mol: "Molecule",
    forcefield: str,
    *,
    previous_coordinates: Optional[ArrayLike] = None,
    previous_energy_kj_mol: Optional[float] = None,
    singularity_threshold: float = TORSION_SINGULARITY_THRESHOLD,
    repair_angle_radians: float = TORSION_REPAIR_ANGLE_RADIANS,
) -> OptimizationCheckReport:
    """Measure one force-field state without modifying ``mol``."""
    previous = None
    if previous_coordinates is not None:
        previous = np.ascontiguousarray(previous_coordinates, dtype=np.float64)
    result = _native_module().check_optimization_state(
        _native_molecule_data(mol),
        forcefield,
        previous,
        previous_energy_kj_mol,
        singularity_threshold,
        repair_angle_radians,
    )
    measurements = result.measurements
    return OptimizationCheckReport(
        evaluated_coordinates=np.asarray(
            result.evaluated_coordinates, dtype=np.float64
        ),
        energy=measurements.energy_kj_mol,
        energy_unit="kJ/mol",
        backend_energy_unit=result.backend_energy_unit,
        rms_gradient=measurements.gradients.rms_kj_mol_angstrom,
        max_gradient=measurements.gradients.maximum_kj_mol_angstrom,
        gradient_unit="kJ/(mol*angstrom)",
        energy_change=measurements.energy_change_kj_mol,
        max_displacement=measurements.maximum_displacement_angstrom,
        finite_coordinates=measurements.finite_coordinates,
        exploded=measurements.exploded,
        failure=OptimizationFailure[result.failure.name],
        rules=_execution_report(result.rules),
    )
