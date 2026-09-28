"""Rule-aware preparation and validation around Open Babel force fields."""

from __future__ import annotations

from math import isfinite

from openbabel import openbabel as ob

from .contracts import (
    ForceFieldStateReport,
    OptimizationPreparationReport,
    RuleExecutionReport,
    RuleStage,
)
from .settings import (
    TORSION_REPAIR_ANGLE_RADIANS,
    TORSION_SINGULARITY_THRESHOLD,
)
from .snapshot import (
    _apply_coordinate_changes,
    _execution_report,
    _optimization_plan,
)


__all__ = ("prepare_optimization", "validate_forcefield_state")


def prepare_optimization(
    obmol: ob.OBMol,
    forcefield: str,
    *,
    singularity_threshold: float = TORSION_SINGULARITY_THRESHOLD,
    repair_angle_radians: float = TORSION_REPAIR_ANGLE_RADIANS,
) -> OptimizationPreparationReport:
    """Apply deterministic coordinate guards before native UFF setup."""
    if forcefield.upper() != "UFF":
        return OptimizationPreparationReport(
            forcefield=forcefield,
            rules=RuleExecutionReport(RuleStage.PRE_FORCEFIELD_SETUP),
        )

    rules = _execution_report(
        _optimization_plan(
            obmol,
            singularity_threshold=singularity_threshold,
            repair_angle_radians=repair_angle_radians,
        )
    )
    _apply_coordinate_changes(obmol, rules)
    return OptimizationPreparationReport(forcefield=forcefield, rules=rules)


def validate_forcefield_state(
    backend: ob.OBForceField,
    obmol: ob.OBMol,
) -> ForceFieldStateReport:
    """Return finite-energy and finite-gradient facts for a set-up backend."""
    energy = float(backend.Energy(True))
    nonfinite_atom_indices = []
    for atom in ob.OBMolAtomIter(obmol):
        gradient = backend.GetGradient(atom)
        if not all(
            isfinite(component)
            for component in (
                gradient.GetX(),
                gradient.GetY(),
                gradient.GetZ(),
            )
        ):
            nonfinite_atom_indices.append(atom.GetIdx() - 1)
    return ForceFieldStateReport(
        energy=energy,
        finite_energy=isfinite(energy),
        finite_gradients=not nonfinite_atom_indices,
        nonfinite_gradient_atom_indices=tuple(nonfinite_atom_indices),
    )
