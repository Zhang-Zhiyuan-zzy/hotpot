from copy import copy

import numpy as np
import pytest
from mca import cpu_backend as cpu_backend_module
from mca.cpu_backend import MCACpuPredictor
from rdkit import Chem

from mca.conformer import ensure_3d_conformer


@pytest.fixture(scope="module")
def cpu_backend():
    backend = MCACpuPredictor()
    yield backend
    backend.close()


def test_cpu_backend_preserves_atom_site_join_and_golden(cpu_backend):
    prediction = cpu_backend.predict("C1CCCCN1")

    assert cpu_backend.runtime.session.get_providers() == ["CPUExecutionProvider"]
    assert "atom_index" not in cpu_backend.runtime.input_dtypes
    assert [atom.atom_index for atom in prediction.atom_predictions] == list(range(6))
    assert prediction.sites[0].site_type == "Amine"
    assert prediction.sites[0].mca_kj_mol == pytest.approx(503.25, abs=1e-6)
    assert prediction.sites[0].mca_kj_mol == pytest.approx(
        prediction.atom_predictions[prediction.sites[0].atom_index].mca_kj_mol
    )


def test_cpu_backend_iterates_one_global_flattened_batch_plan(cpu_backend):
    molecules = [
        ensure_3d_conformer(Chem.MolFromSmiles(smiles))
        for smiles in ("C", "C1CCCCN1", "CC(=O)C", "c1ccncc1")
    ]
    direct = copy(cpu_backend)
    iterated = copy(cpu_backend)
    direct.batch_size = 5
    iterated.batch_size = 5

    direct_results = direct.predict(molecules)
    iterated_results = list(iterated.iter_predict(iter(molecules)))

    for expected, actual in zip(direct_results, iterated_results):
        assert np.array_equal(
            [atom.mca_kj_mol for atom in expected.atom_predictions],
            [atom.mca_kj_mol for atom in actual.atom_predictions],
        )
        assert expected.sites == actual.sites


def test_cpu_backend_cache_key_includes_logical_chunk_padding(
    cpu_backend, monkeypatch
):
    original = cpu_backend_module.collate_molecule_rows
    calls = []

    def record_rows(features, molecule_indices, max_atoms=None):
        calls.append((tuple(molecule_indices), max_atoms))
        return original(features, molecule_indices, max_atoms)

    monkeypatch.setattr(cpu_backend_module, "collate_molecule_rows", record_rows)
    backend = copy(cpu_backend)
    backend.batch_size = 7
    backend.predict(["CCCCCCCCCC", "C1CCCCN1"])

    assert calls == [((0,), 10), ((1,), 10), ((1,), 6)]


def test_cpu_backend_close_is_idempotent(monkeypatch):
    backend = object.__new__(MCACpuPredictor)
    backend._closed = False
    calls = []
    backend.runtime = type("Runtime", (), {"close": lambda self: calls.append(True)})()

    backend.close()
    backend.close()

    assert calls == [True]


def test_cpu_iter_predict_consumes_a_bounded_prefix(cpu_backend):
    consumed = []

    def molecules():
        for index in range(5):
            consumed.append(index)
            yield "C"

    backend = copy(cpu_backend)
    backend.batch_size = 2
    predictions = backend.iter_predict(molecules())

    assert next(predictions).smiles == "C"
    assert consumed == [0, 1]
