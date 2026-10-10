"""Hydrogen-representation invariants for charge inference."""

from __future__ import annotations

import hotpot
import pytest


@pytest.mark.parametrize("smiles", ("C", "CCO", "[BH4-]"))
def test_materializing_hydrogens_preserves_total_and_heavy_atom_charges(
    smiles: str,
) -> None:
    from hotpot.cheminfo.calculator import infer_charge

    implicit_mol = hotpot.read_mol(smiles, "smi")
    implicit_result = infer_charge(implicit_mol)

    explicit_mol = hotpot.read_mol(smiles, "smi")
    explicit_mol.add_hydrogens(rm_polar_hs=False)
    explicit_result = infer_charge(explicit_mol)
    explicit_heavy_charges = tuple(
        explicit_result.atom_formal_charges[atom.idx]
        for atom in explicit_mol.heavy_atoms
    )

    assert explicit_result.total_charge == implicit_result.total_charge
    assert explicit_heavy_charges == implicit_result.atom_formal_charges
    assert all(
        explicit_result.atom_formal_charges[atom.idx] == 0
        for atom in explicit_mol.hydrogens
    )


def test_partially_materialized_hydrogens_are_rejected_as_ambiguous() -> None:
    from hotpot.cheminfo.calculator import infer_charge
    from hotpot.cheminfo.calculator.electronic_state.contracts import (
        AmbiguousHydrogenRepresentationError,
    )

    mol = hotpot.read_mol("C", "smi")
    mol.atoms[0].add_hydrogen(num=1)

    with pytest.raises(AmbiguousHydrogenRepresentationError):
        infer_charge(mol)

