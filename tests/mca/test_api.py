from copy import copy
import hashlib

import numpy as np
import pytest
from mca import MoleculeGraph, MoleculePrediction, predict_mca
from mca import cpu_backend as cpu_backend_module
from mca import pipeline as pipeline_module
from mca.conformer import ensure_3d_conformer
from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem

from hotpot import read_mol


def test_existing_result_constructor_remains_compatible():
    prediction = MoleculePrediction("C", 0, (), "fp16")

    assert prediction.atom_predictions == ()
    assert prediction.to_dict()["atom_predictions"] == []


def test_total_entry_requires_an_explicit_backend():
    with pytest.raises(TypeError, match="backend"):
        predict_mca("C")


def test_total_entry_dispatches_to_the_cpu_backend():
    prediction = predict_mca("C", backend="cpu")

    assert prediction.atom_predictions[0].element == "C"


def test_cpu_smiles_and_batch_prediction(predictor):
    single = predictor.predict("C1CCCCN1")
    batch = predictor.predict(["C1CCCCN1", "c1ccncc1"])

    assert predictor.runtime.session.get_providers() == ["CPUExecutionProvider"]
    assert "atom_index" not in predictor.runtime.input_dtypes
    assert len(single.atom_predictions) == 6
    assert [atom.atom_index for atom in single.atom_predictions] == list(range(6))
    assert len(single.sites) == 1
    assert single.sites[0].mca_kj_mol == pytest.approx(503.25, abs=1e-6)
    assert single.sites[0].mca_kj_mol == pytest.approx(
        single.atom_predictions[single.sites[0].atom_index].mca_kj_mol
    )
    assert [len(item.sites) for item in batch] == [1, 1]
    assert batch[1].sites[0].site_type == "Pyridine_like_nitrogen"
    assert [len(item.atom_predictions) for item in batch] == [6, 6]


def test_repeated_molecule_rows_are_collapsed_within_each_batch(
    predictor, monkeypatch
):
    original = cpu_backend_module.collate_molecule_rows
    row_counts = []

    def record_rows(features, molecule_indices, max_atoms=None):
        row_counts.append(len(molecule_indices))
        return original(features, molecule_indices, max_atoms)

    monkeypatch.setattr(cpu_backend_module, "collate_molecule_rows", record_rows)
    monkeypatch.setattr(predictor, "batch_size", 3)
    predictor.predict(["C1CCCCN1", "c1ccncc1"])

    assert row_counts == [1, 1]


def test_cpu_provider_has_batch_specific_goldens_and_exact_site_join(predictor):
    molecules = [
        ensure_3d_conformer(Chem.MolFromSmiles(smiles))
        for smiles in (
            "C",
            "C1CCCCN1",
            "CC(=O)C",
            "c1ccncc1",
            "CCCCCCCCCC",
        )
    ]
    expected_hashes = {
        5: "1b39a16d8c84f718184d89ed623ba56ec69aa5ffeacd6eb84138ed5949538a7f",
        8: "ddeb4f8635c47a7f40aa8fff8576a5dbf94c8c2bb525381752d952d0a44c4f11",
        64: "6e60a9f38359c05702f503f76bbbd710476469eac0d3837377a7044b5d73ec92",
    }

    for batch_size in (5, 8, 64):
        current = copy(predictor)
        current.batch_size = batch_size
        results = current.predict(molecules)
        atom_values = np.asarray(
            [
                atom.mca_kj_mol
                for result in results
                for atom in result.atom_predictions
            ],
            dtype=np.float64,
        )
        assert hashlib.sha256(atom_values.tobytes()).hexdigest() == expected_hashes[batch_size]
        for result in results:
            for site in result.sites:
                assert site.mca_kj_mol == result.atom_predictions[site.atom_index].mca_kj_mol


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


def test_opted_in_charge_is_reported_from_hotpot_molecule(predictor, monkeypatch):
    monkeypatch.setattr(predictor, "allow_charged", True)

    prediction = predictor.predict("[NH4+]")

    assert prediction.formal_charge == 1


def test_unlocalized_total_charge_is_rejected(predictor):
    molecule = ob.OBMol()
    molecule.NewAtom().SetAtomicNum(6)
    molecule.SetTotalCharge(2)

    with pytest.raises(ValueError, match="does not match"):
        predictor.predict(molecule)


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
    original = pipeline_module.mol_to_unimolv2
    observed = []

    def record_stereochemistry(mol, max_atoms):
        observed.append(
            (
                tuple(str(atom.GetChiralTag()) for atom in mol.GetAtoms()),
                tuple(str(bond.GetStereo()) for bond in mol.GetBonds()),
            )
        )
        return original(mol, max_atoms)

    monkeypatch.setattr(pipeline_module, "mol_to_unimolv2", record_stereochemistry)
    predictor.predict(
        [
            Chem.MolFromSmiles("C[C@H](O)F"),
            Chem.MolFromSmiles("F/C=C/F"),
        ]
    )

    assert "CHI_TETRAHEDRAL_CCW" in observed[0][0]
    assert "STEREOE" in observed[1][1]
