import pytest
from rdkit import Chem

from mca import MoleculeGraph, MoleculePrediction
from mca import api as api_module


class HotpotMoleculeAdapter:
    def to_rdmol(self):
        return Chem.MolFromSmiles("CCN")


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


def test_graph_and_hotpot_protocol_inputs(predictor):
    graph = MoleculeGraph(
        atomic_numbers=[6, 6, 7],
        bonds=[(0, 1, 1.0), (1, 2, 1.0)],
    )

    assert predictor.predict(graph).sites
    assert predictor.predict(HotpotMoleculeAdapter()).sites


def test_charged_molecule_requires_explicit_opt_in(predictor):
    with pytest.raises(ValueError, match="outside the validated domain"):
        predictor.predict("[NH4+]")


def test_explicit_hydrogen_atoms_are_rejected_as_unsupported_targets(predictor):
    molecule = Chem.AddHs(Chem.MolFromSmiles("CN"))

    with pytest.raises(ValueError, match="cannot be MCA targets"):
        predictor.predict(molecule)
