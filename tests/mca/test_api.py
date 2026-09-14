import pytest
from openbabel import pybel
from rdkit import Chem

from hotpot import read_mol

from mca import MoleculeGraph, MoleculePrediction
from mca import api as api_module


def test_existing_result_constructor_remains_compatible():
    prediction = MoleculePrediction("C", 0, (), "fp16")

    assert prediction.atom_predictions == ()
    assert prediction.to_dict()["atom_predictions"] == []


def test_cpu_smiles_and_batch_prediction(predictor):
    single = predictor.predict("C1CCCCN1")
    batch = predictor.predict(["C1CCCCN1", "c1ccncc1"])

    assert predictor.runtime.device == "cpu"
    assert len(single.atom_predictions) == 6
    assert [atom.atom_index for atom in single.atom_predictions] == list(range(6))
    assert len(single.sites) == 1
    assert single.sites[0].mca_kj_mol == pytest.approx(503.25, abs=1e-6)
    assert single.sites[0].mca_kj_mol == pytest.approx(
        single.atom_predictions[single.sites[0].atom_index].mca_kj_mol
    )
    assert [len(item.sites) for item in batch] == [1, 4]
    assert [len(item.atom_predictions) for item in batch] == [6, 6]


def test_all_atom_rows_are_collated_in_bounded_batches(predictor, monkeypatch):
    original = api_module.collate_site_rows
    row_counts = []

    def record_rows(features, molecule_indices, atom_indices):
        row_counts.append(len(atom_indices))
        return original(features, molecule_indices, atom_indices)

    monkeypatch.setattr(api_module, "collate_site_rows", record_rows)
    monkeypatch.setattr(predictor, "batch_size", 3)
    predictor.predict(["C1CCCCN1", "c1ccncc1"])

    assert row_counts == [3, 3, 3, 3]


def test_supported_inputs_keep_atom_and_site_indices_aligned(predictor):
    graph = MoleculeGraph(
        atomic_numbers=[6, 6, 7],
        bonds=[(0, 1, 1.0), (1, 2, 1.0)],
    )
    hotpot_mol = read_mol("CCN")
    inputs = [
        "CCN",
        hotpot_mol,
        Chem.MolFromSmiles("CCN"),
        pybel.readstring("smi", "CCN").OBMol,
        graph,
    ]

    predictions = predictor.predict(inputs)
    expected_atom_labels = [
        (atom.atom_index, atom.element)
        for atom in predictions[0].atom_predictions
    ]
    expected_atom_values = [
        atom.mca_kj_mol for atom in predictions[0].atom_predictions
    ]
    expected_site_labels = [
        (site.atom_index, site.element, site.site_type)
        for site in predictions[0].sites
    ]
    expected_site_values = [site.mca_kj_mol for site in predictions[0].sites]

    for prediction in predictions[1:]:
        assert [
            (atom.atom_index, atom.element)
            for atom in prediction.atom_predictions
        ] == expected_atom_labels
        assert [
            atom.mca_kj_mol for atom in prediction.atom_predictions
        ] == pytest.approx(expected_atom_values)
        assert [
            (site.atom_index, site.element, site.site_type)
            for site in prediction.sites
        ] == expected_site_labels
        assert [site.mca_kj_mol for site in prediction.sites] == pytest.approx(
            expected_site_values
        )


def test_iterable_rdmol_adapter_is_a_single_molecule(predictor):
    class Adapter:
        def __iter__(self):
            return iter(())

        def to_rdmol(self):
            return Chem.MolFromSmiles("CCN")

    prediction = predictor.predict(Adapter())

    assert [atom.element for atom in prediction.atom_predictions] == ["C", "C", "N"]


@pytest.mark.parametrize(
    "molecule",
    [
        pybel.readstring("smi", "CCN"),
        pybel.readstring("smi", "CCN").OBMol,
    ],
)
def test_openbabel_objects_are_single_molecule_inputs(predictor, molecule):
    prediction = predictor.predict(molecule)

    assert len(prediction.atom_predictions) == 3
    assert prediction.sites[0].site_type == "Amine"


def test_zero_and_multiple_detected_sites(predictor):
    no_sites, multiple_sites = predictor.predict(["C", "CC(=O)C"])

    assert no_sites.sites == ()
    assert len(multiple_sites.sites) > 1


def test_charged_molecule_requires_explicit_opt_in(predictor):
    with pytest.raises(ValueError, match="outside the validated domain"):
        predictor.predict("[NH4+]")


@pytest.mark.parametrize(
    "molecule",
    [
        Chem.AddHs(Chem.MolFromSmiles("CN")),
        "[H]CN",
        "[C:1]([H])(C#N)=C=[N-]",
        "[N:1]#C[H]",
    ],
)
def test_explicit_hydrogen_atoms_are_rejected_as_unsupported_targets(
    predictor, molecule
):
    with pytest.raises(ValueError, match="cannot be MCA targets"):
        predictor.predict(molecule)


def test_smiles_is_parsed_once_through_hotpot(predictor, monkeypatch):
    def reject_rdkit_smiles_parse(*args, **kwargs):
        raise AssertionError("SMILES must not be parsed a second time with RDKit")

    monkeypatch.setattr(Chem, "MolFromSmiles", reject_rdkit_smiles_parse)

    prediction = predictor.predict("CCN")

    assert len(prediction.atom_predictions) == 3


def test_string_file_path_is_parsed_by_its_extension(predictor, tmp_path):
    source = Chem.MolFromSmiles("CCN")
    mol_file = tmp_path / "molecule.mol"
    mol_file.write_text(Chem.MolToMolBlock(source), encoding="utf-8")

    prediction = predictor.predict(str(mol_file))

    assert [atom.element for atom in prediction.atom_predictions] == ["C", "C", "N"]


def test_rdkit_stereochemistry_reaches_the_featurizer(predictor, monkeypatch):
    original = api_module.mol_to_unimolv2
    observed = []

    def record_stereochemistry(mol, max_atoms):
        observed.append(
            (
                tuple(str(atom.GetChiralTag()) for atom in mol.GetAtoms()),
                tuple(str(bond.GetStereo()) for bond in mol.GetBonds()),
            )
        )
        return original(mol, max_atoms)

    monkeypatch.setattr(api_module, "mol_to_unimolv2", record_stereochemistry)
    predictor.predict(
        [
            Chem.MolFromSmiles("C[C@H](O)F"),
            Chem.MolFromSmiles("F/C=C/F"),
        ]
    )

    assert "CHI_TETRAHEDRAL_CCW" in observed[0][0]
    assert "STEREOE" in observed[1][1]
