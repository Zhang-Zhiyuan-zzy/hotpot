import pytest
from rdkit import Chem

from mca import MoleculeGraph


class HotpotMoleculeAdapter:
    def to_rdmol(self):
        return Chem.MolFromSmiles("CCN")


def test_cpu_smiles_and_batch_prediction(predictor):
    single = predictor.predict("C1CCCCN1")
    batch = predictor.predict(["C1CCCCN1", "c1ccncc1"])

    assert predictor.runtime.device == "cpu"
    assert len(single.sites) == 1
    assert single.sites[0].mca_kj_mol == pytest.approx(503.25, abs=1e-6)
    assert [len(item.sites) for item in batch] == [1, 4]


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
