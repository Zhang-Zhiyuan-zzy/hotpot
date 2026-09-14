from pathlib import Path

import numpy as np
import pytest
from openbabel import openbabel as ob, pybel
from rdkit import Chem

from hotpot.cheminfo import Molecule, to_hotpot_mol


class RDKitMoleculeAdapter:
    def to_rdmol(self):
        return Chem.MolFromSmiles("CCN")


def test_hotpot_molecule_is_returned_unchanged():
    molecule = Molecule()

    assert to_hotpot_mol(molecule) is molecule


def test_smiles_and_path_use_hotpot_reader(tmp_path: Path):
    from_smiles = to_hotpot_mol("CCO")
    smiles_file = tmp_path / "molecule.smi"
    smiles_file.write_text("C[NH3+]", encoding="utf-8")
    from_path = to_hotpot_mol(smiles_file)

    assert [atom.atomic_number for atom in from_smiles.atoms] == [6, 6, 8]
    assert [atom.atomic_number for atom in from_path.atoms] == [6, 7]
    assert from_path.charge == 1


def test_rdkit_conversion_preserves_order_charge_aromaticity_and_hidden_hydrogens():
    source = Chem.MolFromSmiles("[NH3+]Cc1ccccc1[O-]")
    kekulized = Chem.Mol(source)
    Chem.Kekulize(kekulized)

    molecule = to_hotpot_mol(source)

    assert [atom.atomic_number for atom in molecule.atoms] == [
        atom.GetAtomicNum() for atom in source.GetAtoms()
    ]
    assert molecule.charge == Chem.GetFormalCharge(source) == 0
    assert [atom.formal_charge for atom in molecule.atoms] == [
        atom.GetFormalCharge() for atom in source.GetAtoms()
    ]
    assert [atom.is_aromatic for atom in molecule.atoms] == [
        atom.GetIsAromatic() for atom in source.GetAtoms()
    ]
    assert [atom.implicit_hydrogens for atom in molecule.atoms] == [
        atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
        for atom in source.GetAtoms()
    ]
    assert [bond.bond_order for bond in molecule.bonds] == [
        bond.GetBondTypeAsDouble() for bond in kekulized.GetBonds()
    ]


def test_rdkit_conversion_without_conformer_uses_zero_coordinates():
    source = Chem.MolFromSmiles("CCO")

    molecule = to_hotpot_mol(source)

    np.testing.assert_array_equal(molecule.coordinates, np.zeros((3, 3)))


def test_rdkit_conversion_preserves_conformer_and_explicit_atoms():
    source = Chem.AddHs(Chem.MolFromSmiles("CO"))
    conformer = Chem.Conformer(source.GetNumAtoms())
    expected = np.arange(source.GetNumAtoms() * 3, dtype=float).reshape(-1, 3)
    for atom_index, coordinates in enumerate(expected):
        conformer.SetAtomPosition(atom_index, coordinates)
    source.AddConformer(conformer)

    molecule = to_hotpot_mol(source)

    assert len(molecule.atoms) == source.GetNumAtoms()
    assert sum(atom.atomic_number == 1 for atom in molecule.atoms) == 4
    np.testing.assert_array_equal(molecule.coordinates, expected)


@pytest.mark.parametrize("as_pybel", [False, True])
def test_openbabel_conversion_preserves_structure_and_total_charge(as_pybel):
    pybel_molecule = pybel.readstring("smi", "[NH3+]Cc1ccccc1")
    source = pybel_molecule if as_pybel else pybel_molecule.OBMol

    molecule = to_hotpot_mol(source)

    assert [atom.atomic_number for atom in molecule.atoms] == [
        atom.GetAtomicNum() for atom in ob.OBMolAtomIter(pybel_molecule.OBMol)
    ]
    assert molecule.charge == pybel_molecule.OBMol.GetTotalCharge() == 1
    assert [atom.is_aromatic for atom in molecule.atoms] == [
        atom.IsAromatic() for atom in ob.OBMolAtomIter(pybel_molecule.OBMol)
    ]
    assert [bond.bond_order for bond in molecule.bonds] == [
        bond.GetBondOrder()
        for bond in ob.OBMolBondIter(pybel_molecule.OBMol)
    ]


def test_rdkit_conversion_protocol_uses_the_shared_dispatcher():
    molecule = to_hotpot_mol(RDKitMoleculeAdapter())

    assert [atom.atomic_number for atom in molecule.atoms] == [6, 6, 7]


def test_unsupported_molecule_type_is_rejected():
    with pytest.raises(TypeError, match="unsupported molecule type"):
        to_hotpot_mol(object())
