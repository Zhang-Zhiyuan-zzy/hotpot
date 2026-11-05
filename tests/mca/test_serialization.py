from dataclasses import FrozenInstanceError, replace
import pickle

import numpy as np
import pytest
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem

from hotpot import read_mol
from hotpot.cheminfo.adapters.inference.features import mol_to_unimolv2
from hotpot.cheminfo.operations.conversion import to_hotpot_mol
from hotpot.cheminfo.operations.inference.mca.pipeline import build_prediction
from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.cheminfo.structure.molecule import Molecule
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


@pytest.mark.parametrize('kind', ['smiles', 'hotpot', 'rdkit', 'obmol', 'pybel'])
def test_neutral_acid_smiles_and_predictions_keep_the_original_atom_rows(kind):
    text = 'CC(=O)O'
    value = {
        'smiles': lambda: text,
        'hotpot': lambda: read_mol(text, 'smi'),
        'rdkit': lambda: Chem.MolFromSmiles(text),
        'obmol': lambda: pybel.readstring('smi', text).OBMol,
        'pybel': lambda: pybel.readstring('smi', text),
    }[kind]()

    dto = normalize_molecule(value)
    validate_dto(dto, position=0)
    graph = hotpot_mol_from_dto(dto)
    feature = rdmol_from_dto(dto)
    public = Chem.MolFromSmiles(dto.smiles)
    sites = _site_records(graph)
    # Synthetic row values exercise the real result join without model inference.
    prediction = build_prediction(dto, sites, (100., 101., 102., 103.), 'synthetic')

    assert public is not None
    assert dto.formal_charge == Chem.GetFormalCharge(public) == 0
    assert sum(atom.GetTotalNumHs() for atom in public.GetAtoms()) == 4
    assert Chem.MolToSmiles(public) == Chem.MolToSmiles(Chem.MolFromSmiles(text))
    assert dto.atom_symbols == ('C', 'C', 'O', 'O')
    assert tuple(atom.formal_charge for atom in dto.atoms) == (0, 0, 0, 0)
    assert tuple(atom.implicit_hydrogens for atom in dto.atoms) == (3, 0, 0, 1)
    assert tuple(
        (bond.begin_atom_index, bond.end_atom_index, bond.bond_order, bond.bond_kind)
        for bond in dto.bonds
    ) == ((0, 1, 1., 'single'), (1, 2, 2., 'double'), (1, 3, 1., 'single'))
    assert tuple(atom.GetSymbol() for atom in feature.GetAtoms()) == dto.atom_symbols
    assert tuple(atom.GetFormalCharge() for atom in feature.GetAtoms()) == (0, 0, 0, 0)
    assert tuple(atom.GetTotalNumHs() for atom in feature.GetAtoms()) == (3, 0, 0, 1)
    assert tuple(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
        for bond in feature.GetBonds()
    ) == ((0, 1, 1.), (1, 2, 2.), (1, 3, 1.))
    assert sites == ((2, 'Carboxylic acid'), (1, 'double_bond'), (3, 'atom_with_lone_pair'))
    assert prediction.smiles == dto.smiles
    assert prediction.formal_charge == dto.formal_charge
    assert tuple(
        (atom.atom_index, atom.element, atom.mca_kj_mol)
        for atom in prediction.atom_predictions
    ) == ((0, 'C', 100.), (1, 'C', 101.), (2, 'O', 102.), (3, 'O', 103.))
    assert tuple(
        (site.atom_index, site.element, site.site_type, site.mca_kj_mol)
        for site in prediction.sites
    ) == (
        (2, 'O', 'Carboxylic acid', 102.),
        (1, 'C', 'double_bond', 101.),
        (3, 'O', 'atom_with_lone_pair', 103.),
    )


@pytest.mark.parametrize('text', [
    'CC(=O)O', '[13CH3]C(=O)O', 'F[C@](Cl)(Br)I', 'F/C=C/F',
])
@pytest.mark.parametrize('kind', ['hotpot', 'rdkit'])
def test_public_smiles_keeps_the_existing_rdkit_feature_binary_and_tensors(text, kind):
    original = Chem.MolFromSmiles(text)
    conformer = Chem.Conformer(original.GetNumAtoms())
    conformer.Set3D(True)
    for index in range(original.GetNumAtoms()):
        conformer.SetAtomPosition(index, (float(index), 1., 2.))
    original.AddConformer(conformer)
    value = original if kind == 'rdkit' else to_hotpot_mol(original)
    # Capture the established feature boundary before public SMILES generation.
    expected = Chem.Mol(original if kind == 'rdkit' else value.to_rdmol())
    if kind == 'hotpot':
        Chem.SanitizeMol(expected)
    expected_binary = expected.ToBinary(Chem.PropertyPickleOptions.AllProps)
    expected_features = mol_to_unimolv2(Chem.Mol(expected))

    dto = normalize_molecule(value)
    restored = rdmol_from_dto(dto)
    actual_features = mol_to_unimolv2(restored)

    assert dto.rdkit_binary == expected_binary
    assert Chem.MolToSmiles(restored) == Chem.MolToSmiles(original)
    assert actual_features.keys() == expected_features.keys()
    for name in expected_features:
        np.testing.assert_array_equal(actual_features[name], expected_features[name])


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


def test_unmapped_nonmetal_dative_graph_keeps_hotpot_smiles_failure():
    molecule = Molecule()
    molecule.create_atom(atomic_number=7, implicit_hydrogens=3)
    molecule.create_atom(atomic_number=5)
    molecule.add_bond(
        0,
        1,
        1.0,
        bond_kind=BondKind.DATIVE,
        bond_direction="atom1_to_atom2",
    )

    with pytest.raises(ValueError, match="cannot represent dative bonds losslessly"):
        normalize_molecule(molecule)



def test_mapped_metal_dative_graph_keeps_facts_through_dto_and_public_smiles():
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

    dto = pickle.loads(pickle.dumps(normalize_molecule(molecule)))
    restored = hotpot_mol_from_dto(dto)
    feature = rdmol_from_dto(dto)
    public = Chem.MolFromSmiles(dto.smiles)

    assert dto.atom_symbols == ("N", "Cu")
    assert tuple(
        (atom.atomic_number, atom.formal_charge, atom.implicit_hydrogens, atom.valence)
        for atom in dto.atoms
    ) == ((7, 0, 3, 3), (29, 0, 0, 2))
    assert tuple(
        (bond.begin_atom_index, bond.end_atom_index, bond.bond_order,
         bond.bond_kind, bond.bond_direction)
        for bond in dto.bonds
    ) == ((0, 1, 1.0, "dative", "atom1_to_atom2"),)
    for graph in (molecule, restored):
        assert tuple(
            (atom.atomic_number, atom.formal_charge, atom.implicit_hydrogens, atom.valence)
            for atom in graph.atoms
        ) == ((7, 0, 3, 3), (29, 0, 0, 2))
        assert tuple(
            (bond.a1idx, bond.a2idx, bond.bond_order, bond.bond_kind, bond.bond_direction)
            for bond in graph.bonds
        ) == ((0, 1, 1.0, BondKind.DATIVE, "atom1_to_atom2"),)
        assert _site_records(graph) == ()
    assert tuple(atom.GetSymbol() for atom in feature.GetAtoms()) == dto.atom_symbols
    assert tuple(atom.GetTotalNumHs() for atom in feature.GetAtoms()) == (3, 0)
    assert feature.GetBondWithIdx(0).GetBondType() == Chem.BondType.DATIVE
    # The registered external projection is SINGLE; DTO facts stay authoritative.
    assert dto.smiles == "[NH3][Cu]"
    assert public is not None
    assert tuple(atom.GetSymbol() for atom in public.GetAtoms()) == dto.atom_symbols
    assert tuple(atom.GetTotalNumHs() for atom in public.GetAtoms()) == (3, 0)
    projected = pybel.readstring("smi", dto.smiles).OBMol
    assert projected.GetBond(0).GetBondOrder() == 1


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
