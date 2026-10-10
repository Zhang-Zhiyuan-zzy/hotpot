"""Target behavior for pure formal-charge inference."""

from __future__ import annotations

import hotpot
import pytest


@pytest.mark.parametrize(
    ("smiles", "expected_charges", "expected_total"),
    (
        ("CCO", (0, 0, 0), 0),
        ("[NH4+]", (1,), 1),
        ("CC(=O)[O-]", (0, 0, 0, -1), -1),
        ("C[N+](=O)[O-]", (0, 1, 0, -1), 0),
        ("[BH4-]", (-1,), -1),
        ("[Zn](Cl)Cl", (2, -1, -1), 0),
    ),
)
def test_infer_charge_covers_common_molecular_states(
    smiles: str,
    expected_charges: tuple[int, ...],
    expected_total: int,
) -> None:
    from hotpot.cheminfo.calculator import infer_charge

    mol = hotpot.read_mol(smiles, "smi")
    result = infer_charge(mol)

    assert result.atom_formal_charges == expected_charges
    assert result.total_charge == expected_total
    assert sum(fragment.charge for fragment in result.fragments) == expected_total


def test_infer_charge_does_not_mutate_the_source_molecule() -> None:
    from hotpot.cheminfo.calculator import infer_charge

    mol = hotpot.read_mol("[Zn](Cl)Cl", "smi")
    mol.properties["sentinel"] = "unchanged"
    cached_obmol = mol.to_obmol()
    before = (
        tuple(atom.formal_charge for atom in mol.atoms),
        mol.charge,
        tuple((bond.a1idx, bond.a2idx, bond.bond_order) for bond in mol.bonds),
        dict(mol.properties),
    )

    result = infer_charge(mol)

    after = (
        tuple(atom.formal_charge for atom in mol.atoms),
        mol.charge,
        tuple((bond.a1idx, bond.a2idx, bond.bond_order) for bond in mol.bonds),
        dict(mol.properties),
    )
    assert result.atom_formal_charges == (2, -1, -1)
    assert after == before
    assert mol._obmol is cached_obmol


def test_fragment_order_follows_the_smallest_original_atom_index() -> None:
    from hotpot.cheminfo.calculator import infer_charge

    mol = hotpot.read_mol("[Zn](Cl)Cl", "smi")
    result = infer_charge(mol)

    assert tuple(fragment.atom_indices for fragment in result.fragments) == (
        (0,),
        (1,),
        (2,),
    )
    assert tuple(fragment.charge for fragment in result.fragments) == (2, -1, -1)


@pytest.mark.parametrize("metal", ("Eu", "Am"))
def test_default_metal_charge_is_explicitly_reported_as_an_assumption(
    metal: str,
) -> None:
    from hotpot.cheminfo.calculator import infer_charge

    mol = hotpot.read_mol(f"[{metal}].N", "smi")
    result = infer_charge(mol)

    assert result.total_charge == 3
    assert any(metal in assumption for assumption in result.assumptions)


def test_anionic_ligand_is_summed_with_default_metal_charge() -> None:
    from hotpot.cheminfo.calculator import infer_charge

    mol = hotpot.read_mol("[Am].[Cl-]", "smi")
    result = infer_charge(mol)

    assert result.atom_formal_charges == (3, -1)
    assert result.total_charge == 2

