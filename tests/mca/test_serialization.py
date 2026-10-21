from dataclasses import FrozenInstanceError, replace
import pickle

import pytest
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem

from hotpot import read_mol
from hotpot.cheminfo.core import BondKind, Molecule
from mca.graph_adapter import MoleculeGraph
from mca.serialization import (
    MCA_MOLECULE_DTO_VERSION,
    hotpot_mol_from_dto,
    normalize_molecule,
    rdmol_from_dto,
    validate_dto,
)
from mca.site_detection import find_nucleophilic_sites
from tests.smarts_conformance.coordination_cases import (
    ethylenediamine_chelate,
    metal_star,
    tertiary_amine,
)


def _site_records(molecule):
    return tuple(
        (site.atom_index, site.site_type)
        for site in find_nucleophilic_sites(molecule)
    )


def test_dto_is_versioned_immutable_and_pickleable():
    dto = normalize_molecule("CCN")
    restored = pickle.loads(pickle.dumps(dto))

    assert restored == dto
    assert restored.version == MCA_MOLECULE_DTO_VERSION
    assert restored.smiles == "CCN"
    assert restored.formal_charge == 0
    assert restored.atom_symbols == ("C", "C", "N")
    assert restored.atom_count == 3
    with pytest.raises(FrozenInstanceError):
        restored.atom_count = 4


@pytest.mark.parametrize(
    "value",
    [
        "CCN",
        read_mol("CCN"),
        Chem.MolFromSmiles("CCN"),
        pybel.readstring("smi", "CCN"),
        pybel.readstring("smi", "CCN").OBMol,
        MoleculeGraph(
            atomic_numbers=(6, 6, 7),
            bonds=((0, 1, 1.0), (1, 2, 1.0)),
        ),
    ],
    ids=("smiles", "hotpot", "rdkit", "pybel", "obmol", "graph"),
)
def test_supported_inputs_share_the_same_dto_graph_semantics(value):
    dto = normalize_molecule(value)
    native_mol = hotpot_mol_from_dto(dto)
    feature_mol = rdmol_from_dto(dto)

    assert dto.atom_symbols == ("C", "C", "N")
    assert _site_records(native_mol) == ((2, "Amine"),)
    assert tuple(atom.GetSymbol() for atom in feature_mol.GetAtoms()) == (
        "C",
        "C",
        "N",
    )


def test_iterable_rdmol_adapter_is_normalized_once():
    class Adapter:
        def __init__(self):
            self.calls = 0

        def __iter__(self):
            return iter(())

        def to_rdmol(self):
            self.calls += 1
            return Chem.MolFromSmiles("CCN")

    adapter = Adapter()
    dto = normalize_molecule(adapter)

    assert adapter.calls == 1
    assert dto.atom_symbols == ("C", "C", "N")


def test_rdkit_stereochemistry_and_coordinates_survive_the_dto():
    chiral = Chem.MolFromSmiles("C[C@H](O)F")
    alkene = Chem.MolFromSmiles("F/C=C/F")
    conformer = Chem.Conformer(chiral.GetNumAtoms())
    conformer.Set3D(True)
    for index in range(chiral.GetNumAtoms()):
        conformer.SetAtomPosition(index, (float(index), 1.0, 2.0))
    chiral.AddConformer(conformer)

    restored_chiral = rdmol_from_dto(normalize_molecule(chiral))
    restored_alkene = rdmol_from_dto(normalize_molecule(alkene))

    assert "CHI_TETRAHEDRAL_CCW" in {
        str(atom.GetChiralTag()) for atom in restored_chiral.GetAtoms()
    }
    assert "STEREOE" in {
        str(bond.GetStereo()) for bond in restored_alkene.GetBonds()
    }
    assert tuple(restored_chiral.GetConformer().GetAtomPosition(2)) == (
        2.0,
        1.0,
        2.0,
    )


def test_aromatic_atoms_bonds_and_site_order_survive_both_representations():
    dto = normalize_molecule(Chem.MolFromSmiles("c1ccncc1"))
    native_mol = hotpot_mol_from_dto(dto)
    feature_mol = rdmol_from_dto(dto)

    assert all(atom.is_aromatic for atom in native_mol.atoms)
    assert all(bond.bond_kind is BondKind.AROMATIC for bond in native_mol.bonds)
    assert all(atom.GetIsAromatic() for atom in feature_mol.GetAtoms())
    assert all(bond.GetIsAromatic() for bond in feature_mol.GetBonds())
    assert _site_records(native_mol) == ((3, "Pyridine_like_nitrogen"),)


@pytest.mark.parametrize(
    "molecule",
    [
        tertiary_amine(),
        tertiary_amine(0.0),
        tertiary_amine(1.0),
        ethylenediamine_chelate(),
        metal_star(6),
    ],
    ids=("free", "zero-order", "single", "chelate", "metal-star"),
)
def test_coordination_graph_round_trip_preserves_site_semantics(molecule):
    dto = normalize_molecule(molecule)
    restored = hotpot_mol_from_dto(pickle.loads(pickle.dumps(dto)))

    assert _site_records(restored) == _site_records(molecule)
    assert tuple(atom.implicit_hydrogens for atom in restored.atoms) == tuple(
        atom.implicit_hydrogens for atom in molecule.atoms
    )
    assert tuple(
        (bond.a1idx, bond.a2idx, bond.bond_order, bond.bond_kind)
        for bond in restored.bonds
    ) == tuple(
        (bond.a1idx, bond.a2idx, bond.bond_order, bond.bond_kind)
        for bond in molecule.bonds
    )
    assert tuple(
        (bond.a1idx, bond.a2idx)
        for bond in restored.bonds
        if bond.is_metal_ligand_bond
    ) == tuple(
        (bond.a1idx, bond.a2idx)
        for bond in molecule.bonds
        if bond.is_metal_ligand_bond
    )


def test_dative_graph_keeps_the_existing_hotpot_smiles_failure():
    molecule = Molecule()
    molecule.create_atom(atomic_number=7, implicit_hydrogens=3)
    molecule.create_atom(atomic_number=29)
    molecule.add_bond(
        0,
        1,
        1.0,
        bond_kind=BondKind.DATIVE,
        bond_direction="atom1_to_atom2",
    )

    with pytest.raises(ValueError):
        normalize_molecule(molecule)


def test_hotpot_isotope_keeps_site_symmetry_distinction():
    molecule = read_mol("NN")
    molecule.atoms[1].isotope = 15

    restored = hotpot_mol_from_dto(normalize_molecule(molecule))

    assert restored.atoms[1].isotope == 15
    assert _site_records(restored) == _site_records(molecule) == (
        (0, "Amine"),
        (1, "Amine"),
    )


def test_validation_preserves_existing_target_domain_checks():
    validate_dto(normalize_molecule("CCN"), position=0)
    charged = normalize_molecule("[NH4+]")
    with pytest.raises(ValueError, match="outside the validated domain"):
        validate_dto(charged, position=4)
    validate_dto(charged, position=4, allow_charged=True)

    with pytest.raises(ValueError, match="Explicit hydrogen atoms"):
        validate_dto(
            normalize_molecule(Chem.AddHs(Chem.MolFromSmiles("CN"))),
            position=2,
        )

    unlocalized = ob.OBMol()
    unlocalized.NewAtom().SetAtomicNum(6)
    unlocalized.SetTotalCharge(2)
    with pytest.raises(ValueError, match="does not match"):
        validate_dto(normalize_molecule(unlocalized), position=7)


def test_unknown_dto_version_is_rejected_by_both_rebuilders():
    dto = replace(normalize_molecule("CCN"), version=999)

    with pytest.raises(ValueError, match="Unsupported MCA molecule DTO version"):
        hotpot_mol_from_dto(dto)
    with pytest.raises(ValueError, match="Unsupported MCA molecule DTO version"):
        rdmol_from_dto(dto)
