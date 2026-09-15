from copy import copy

import pytest
from openbabel import openbabel as ob
from rdkit import Chem

from hotpot.cheminfo.core import Bond, BondKind, Molecule
from hotpot.cheminfo.obconvert import mol2obmol, obmol2mol
from hotpot.cheminfo.rdconvert import from_rdmol, to_rdmol


def _hotpot_bond(
    *,
    bond_order=1.0,
    bond_kind=BondKind.SINGLE,
    bond_direction=None,
    bond_source=None,
    bond_source_metadata=None,
):
    mol = Molecule()
    mol.create_atom(atomic_number=7)
    mol.create_atom(atomic_number=29)
    return mol.add_bond(
        0,
        1,
        bond_order=bond_order,
        bond_kind=bond_kind,
        bond_direction=bond_direction,
        bond_source=bond_source,
        bond_source_metadata=bond_source_metadata,
    )


def test_bond_metadata_does_not_change_numeric_attribute_layout():
    bond = _hotpot_bond(
        bond_kind=BondKind.DATIVE,
        bond_direction="atom1_to_atom2",
        bond_source="rdkit",
        bond_source_metadata={"raw_bond_type": "DATIVE"},
    )

    assert len(bond.attrs) == len(Bond._attrs_enumerator) == 3
    assert bond.bond_order == 1.0
    assert bond.bond_kind is BondKind.DATIVE
    assert bond.bond_direction == "atom1_to_atom2"
    assert bond.bond_source == "rdkit"
    assert bond.bond_source_metadata == {"raw_bond_type": "DATIVE"}


def test_bond_source_metadata_is_deeply_isolated():
    metadata = {"raw": {"flags": [1, 2]}}
    bond = _hotpot_bond(bond_source_metadata=metadata)

    metadata["raw"]["flags"].append(3)
    returned = bond.bond_source_metadata
    returned["raw"]["flags"].append(4)

    assert bond.bond_source_metadata == {"raw": {"flags": [1, 2]}}


def test_bond_metadata_round_trips_through_attr_dict_copy_components_and_to_mol():
    bond = _hotpot_bond(
        bond_kind=BondKind.DATIVE,
        bond_direction="atom1_to_atom2",
        bond_source="rdkit",
        bond_source_metadata={"raw_bond_type": "DATIVE"},
    )
    molecule = bond.mol

    rebuilt = copy(molecule)
    component = molecule.components[0]
    bond_molecule = bond.to_mol()

    for candidate in (rebuilt.bonds[0], component.bonds[0], bond_molecule.bonds[0]):
        assert candidate.bond_kind is BondKind.DATIVE
        assert candidate.bond_direction == "atom1_to_atom2"
        assert candidate.bond_source == "rdkit"
        assert candidate.bond_source_metadata == {"raw_bond_type": "DATIVE"}

    bond.bond_source_metadata["raw_bond_type"] = "changed"
    assert bond.bond_source_metadata == {"raw_bond_type": "DATIVE"}
    assert rebuilt.bonds[0].bond_source_metadata == {"raw_bond_type": "DATIVE"}


def test_components_preserve_direction_when_bond_endpoints_are_reversed():
    molecule = Molecule()
    molecule.create_atom(atomic_number=7)
    molecule.create_atom(atomic_number=29)
    molecule.add_bond(
        1,
        0,
        bond_kind=BondKind.DATIVE,
        bond_direction="atom1_to_atom2",
    )

    component_bond = molecule.components[0].bonds[0]

    assert component_bond.atom1.atomic_number == 29
    assert component_bond.atom2.atomic_number == 7
    assert component_bond.bond_direction == "atom1_to_atom2"


@pytest.mark.parametrize(
    ("bond_type", "expected_kind", "expected_order", "expected_direction"),
    [
        (Chem.BondType.ZERO, BondKind.ZERO, 0.0, None),
        (Chem.BondType.SINGLE, BondKind.SINGLE, 1.0, None),
        (Chem.BondType.DOUBLE, BondKind.DOUBLE, 2.0, None),
        (Chem.BondType.TRIPLE, BondKind.TRIPLE, 3.0, None),
        (Chem.BondType.DATIVE, BondKind.DATIVE, 1.0, "atom1_to_atom2"),
    ],
)
def test_rdkit_import_preserves_supported_bond_kind(
    bond_type, expected_kind, expected_order, expected_direction
):
    source = Chem.RWMol()
    source.AddAtom(Chem.Atom(7))
    source.AddAtom(Chem.Atom(29))
    source.AddBond(0, 1, bond_type)

    molecule = from_rdmol(source.GetMol(), Molecule())
    bond = molecule.bonds[0]

    assert bond.bond_order == expected_order
    assert bond.bond_kind is expected_kind
    assert bond.bond_direction == expected_direction
    assert bond.bond_source == "rdkit"
    assert bond.bond_source_metadata["raw_bond_type"] == str(bond_type)


@pytest.mark.parametrize(
    ("bond_kind", "bond_order", "expected_type"),
    [
        (BondKind.ZERO, 0.0, Chem.BondType.ZERO),
        (BondKind.SINGLE, 1.0, Chem.BondType.SINGLE),
        (BondKind.DOUBLE, 2.0, Chem.BondType.DOUBLE),
        (BondKind.TRIPLE, 3.0, Chem.BondType.TRIPLE),
        (BondKind.DATIVE, 1.0, Chem.BondType.DATIVE),
    ],
)
def test_rdkit_export_preserves_supported_bond_kind(
    bond_kind, bond_order, expected_type
):
    direction = "atom1_to_atom2" if bond_kind is BondKind.DATIVE else None
    bond = _hotpot_bond(
        bond_order=bond_order,
        bond_kind=bond_kind,
        bond_direction=direction,
    )

    exported = to_rdmol(bond.mol, kekulize=False)

    assert exported.GetBondWithIdx(0).GetBondType() == expected_type


def test_rdkit_import_and_export_preserve_reverse_dative_direction():
    source = Chem.MolFromSmiles("N<-[Cu]")

    molecule = from_rdmol(source, Molecule())
    exported = to_rdmol(molecule, kekulize=False)
    bond = exported.GetBondWithIdx(0)

    assert molecule.bonds[0].atom1.atomic_number == 29
    assert molecule.bonds[0].atom2.atomic_number == 7
    assert molecule.bonds[0].bond_direction == "atom1_to_atom2"
    assert bond.GetBondType() == Chem.BondType.DATIVE
    assert exported.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 29
    assert exported.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 7


def test_rdkit_dativeone_subtype_round_trips_when_supported():
    source = Chem.RWMol()
    source.AddAtom(Chem.Atom(7))
    source.AddAtom(Chem.Atom(29))
    source.AddBond(0, 1, Chem.BondType.DATIVEONE)

    molecule = from_rdmol(source.GetMol(), Molecule())
    exported = to_rdmol(molecule, kekulize=False)

    assert molecule.bonds[0].bond_kind is BondKind.DATIVE
    assert exported.GetBondWithIdx(0).GetBondType() == Chem.BondType.DATIVEONE


def test_rdkit_export_rejects_dative_bond_without_direction():
    bond = _hotpot_bond(bond_kind=BondKind.DATIVE)

    with pytest.raises(ValueError, match="requires a bond direction"):
        to_rdmol(bond.mol, kekulize=False)


def test_openbabel_import_treats_order_zero_as_unknown_and_records_raw_flags():
    source = ob.OBMol()
    source.NewAtom().SetAtomicNum(7)
    source.NewAtom().SetAtomicNum(29)
    source.AddBond(1, 2, 0)

    molecule = obmol2mol(source, Molecule())
    bond = molecule.bonds[0]

    assert bond.bond_order == 0.0
    assert bond.bond_kind is BondKind.UNKNOWN
    assert bond.bond_direction is None
    assert bond.bond_source == "openbabel"
    assert bond.bond_source_metadata["raw_bond_order"] == 0
    assert bond.bond_source_metadata["raw_flags"] == 0
    assert bond.bond_source_metadata["is_aromatic"] is False
    assert bond.bond_source_metadata["is_amide"] is False
    assert bond.bond_source_metadata["is_in_ring"] is False


def test_openbabel_export_rejects_dative_bond_instead_of_writing_single():
    bond = _hotpot_bond(
        bond_kind=BondKind.DATIVE,
        bond_direction="atom1_to_atom2",
    )

    with pytest.raises(ValueError, match="cannot represent dative bonds losslessly"):
        mol2obmol(bond.mol)


def test_openbabel_export_preserves_explicit_aromatic_kind():
    bond = _hotpot_bond(
        bond_order=1.5,
        bond_kind=BondKind.AROMATIC,
    )

    exported, _ = mol2obmol(bond.mol)

    assert exported.GetBond(1, 2).IsAromatic()
