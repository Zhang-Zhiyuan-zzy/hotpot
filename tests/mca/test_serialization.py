from dataclasses import FrozenInstanceError, replace
import pickle

import pytest
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem

from hotpot import read_mol
from hotpot.cheminfo.core import BondKind, Molecule
from hotpot.cheminfo.operations.inference.mca.results import MCA_MOLECULE_DTO_VERSION
from hotpot.cheminfo.operations.inference.mca.serialization import hotpot_mol_from_dto, normalize_molecule, rdmol_from_dto, validate_dto
from hotpot.cheminfo.operations.inference.mca.site_detection import find_nucleophilic_sites
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


class _MoleculeAdapter:
    def to_rdmol(self):
        return Chem.MolFromSmiles("CCN")


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
        _MoleculeAdapter(),
    ],
    ids=("smiles", "hotpot", "rdkit", "pybel", "obmol", "adapter"),
)
def test_supported_inputs_share_the_same_dto_semantics(value):
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


def test_transfer_records_have_one_canonical_owner():
    dto = normalize_molecule("CCN")
    assert type(dto).__module__ == "hotpot.cheminfo.operations.inference.mca.results"
    assert _site_records(hotpot_mol_from_dto(dto)) == ((2, "Amine"),)


def test_own_dto_normalization_reuses_identity_without_preparation(monkeypatch):
    from hotpot.cheminfo.operations.inference.mca import serialization

    dto = normalize_molecule("CN")
    monkeypatch.setattr(
        serialization,
        "to_hotpot_mol",
        lambda value: pytest.fail("DTO must not be converted or prepared again"),
    )
    assert normalize_molecule(dto) is dto


def test_cpu_preprocess_reuses_dto_and_keeps_domain_checks():
    from hotpot.cheminfo.operations.inference.mca.cpu_backend import MCACpuPredictor

    predictor = object.__new__(MCACpuPredictor)
    predictor.allow_charged = False
    predictor.max_atoms = 512
    predictor.conformer_seed = 42
    dto = normalize_molecule("CN")
    task = predictor._preprocess_task(dto, 5)
    assert task.molecule is dto
    assert task.molecule_index == 5
    assert task.sites is None
    assert task.max_atoms == 512
    assert task.conformer_seed == 42
    with pytest.raises(ValueError, match="position 6.*outside the validated domain"):
        predictor._preprocess_task(normalize_molecule("C[NH3+]"), 6)


@pytest.mark.parametrize("values", [None, (None, 0.0, -0.25)])
def test_dto_charge_round_trip_preserves_absence_zero_and_row_mapping(values):
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._results import _import_partial_charges

    molecule = read_mol("CCN")
    if values is not None:
        _import_partial_charges(molecule, values, producer=ATOM_SOURCE_CODES["explicit"])
    expected = (None,) * 3 if values is None else values

    dto = pickle.loads(pickle.dumps(normalize_molecule(molecule)))
    restored = hotpot_mol_from_dto(dto)

    assert tuple(atom.partial_charge for atom in dto.atoms) == expected
    assert tuple(atom.partial_charge for atom in restored.atoms) == expected
    assert tuple(atom.symbol for atom in restored.atoms) == ("C", "C", "N")
    assert _site_records(restored) == ((2, "Amine"),)
    if values is not None:
        assert restored._partial_charge_result.payload.provenance == (
            ATOM_SOURCE_CODES["default"], ATOM_SOURCE_CODES["mca_dto"],
            ATOM_SOURCE_CODES["mca_dto"],
        )


def test_stale_charge_is_unavailable_in_dto_without_replacing_historical_selection():
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._results import _import_partial_charges

    molecule = read_mol("CCN")
    _import_partial_charges(molecule, (-0.25, 0.0, 0.25), producer=ATOM_SOURCE_CODES["explicit"])
    molecule.atoms[0].x = 1.0
    selected = molecule._partial_charge_result

    dto = normalize_molecule(molecule)
    restored = hotpot_mol_from_dto(dto)

    assert tuple(atom.partial_charge for atom in dto.atoms) == (None,) * 3
    assert tuple(atom.partial_charge for atom in restored.atoms) == (None,) * 3
    assert molecule._partial_charge_result is selected
    assert selected.invalidated_by
    assert selected.payload.values == (-0.25, 0.0, 0.25)


def test_canonical_isotope_only_DTO_can_be_reexported_without_losing_symmetry():
    molecule = read_mol("NN")
    molecule.atoms[1].isotope = 15
    first = normalize_molecule(molecule)
    restored = hotpot_mol_from_dto(first)
    second = normalize_molecule(restored)
    assert tuple(atom.isotope for atom in first.atoms) == tuple(atom.isotope for atom in second.atoms) == (0, 15)
    assert _site_records(restored) == ((0, "Amine"), (1, "Amine"))
