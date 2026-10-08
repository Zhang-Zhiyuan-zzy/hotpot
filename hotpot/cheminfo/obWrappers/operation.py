"""Independent native Open Babel force-field operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .contracts import SingleOptimizationReport
from .native import _native_module, _native_molecule_data
from .reports import _execution_report
from .settings import (
    TORSION_REPAIR_ANGLE_RADIANS,
    TORSION_SINGULARITY_THRESHOLD,
)


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("single_optimize",)


def single_optimize(
    mol: "Molecule",
    forcefield: str,
    steps: int,
    *,
    singularity_threshold: float = TORSION_SINGULARITY_THRESHOLD,
    repair_angle_radians: float = TORSION_REPAIR_ANGLE_RADIANS,
) -> SingleOptimizationReport:
    """Run one native steepest-descent operation and update ``mol``."""
    result = _native_module().single_optimize(
        _native_molecule_data(mol),
        forcefield,
        steps,
        singularity_threshold,
        repair_angle_radians,
    )
    coordinates = np.asarray(result.coordinates, dtype=np.float64)
    mol.coordinates = coordinates
    return SingleOptimizationReport(
        coordinates=coordinates,
        energy=result.energy_kj_mol,
        energy_unit="kJ/mol",
        backend_energy_unit=result.backend_energy_unit,
        exploded=result.exploded,
        rules=_execution_report(result.rules),
    )
