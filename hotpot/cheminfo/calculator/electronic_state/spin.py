"""Named spin-inference policies for explicit molecular structures."""

from __future__ import annotations

from ...core import Molecule
from . import contracts

__all__ = [
    "LowestSpinEstimator",
    "infer_lowest_spin",
    "require_complete_explicit_atoms",
]


def require_complete_explicit_atoms(mol: Molecule) -> None:
    """Require every stored hydrogen to exist in the molecular atom list."""
    for atom in mol.heavy_atoms:
        explicit_hydrogen_count = len(atom.hydrogens)
        if atom.implicit_hydrogens and (
            explicit_hydrogen_count != atom.implicit_hydrogens
        ):
            raise contracts.IncompleteExplicitAtomError(
                f"Atom {atom.idx} ({atom.symbol}) stores "
                f"{atom.implicit_hydrogens} implicit hydrogens but has "
                f"{explicit_hydrogen_count} explicit hydrogen atoms"
            )


def infer_lowest_spin(mol: Molecule, charge: int) -> contracts.SpinInferenceResult:
    """Infer the lowest spin allowed by electron-count parity.

    The molecule must contain the exact explicit atom list that will be sent to
    the numerical backend. This policy is an execution default, not a physical
    ground-state prediction.
    """
    require_complete_explicit_atoms(mol)
    electron_count = sum(atom.atomic_number for atom in mol.atoms) - charge
    if electron_count < 0:
        raise ValueError("Total charge leaves a negative electron count")
    unpaired_electrons = electron_count % 2
    return contracts.SpinInferenceResult(
        unpaired_electrons=unpaired_electrons,
        multiplicity=unpaired_electrons + 1,
        electron_count=electron_count,
        source=contracts.SpinInferenceSource.LOWEST_SPIN_PARITY,
        assumptions=(
            "Applied the lowest-spin parity policy to the complete explicit "
            "nuclear charge and total molecular charge.",
        ),
    )


class LowestSpinEstimator:
    """Replaceable estimator implementing the lowest-spin parity policy."""

    def infer(
        self,
        mol: Molecule,
        charge: int,
    ) -> contracts.SpinInferenceResult:
        """Infer a parity-consistent lowest-spin execution state."""
        return infer_lowest_spin(mol, charge)
