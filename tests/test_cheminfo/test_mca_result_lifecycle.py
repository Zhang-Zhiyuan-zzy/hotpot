"""Selected MCA ownership contracts with deterministic predictors and no model."""

from copy import copy, deepcopy
from dataclasses import FrozenInstanceError, replace
import importlib
import pickle
from threading import Thread

import pytest

from hotpot import read_mol
from hotpot.cheminfo.operations.inference.mca.results import (
    AtomPrediction,
    MoleculePrediction,
    SitePrediction,
)
from hotpot.cheminfo.operations.inference.mca.results import MoleculeDTO
from hotpot.cheminfo.operations.inference.mca import mca
from hotpot.cheminfo.structure._commit import _FactTransaction, _current_transaction
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES


mca_package = importlib.import_module("hotpot.cheminfo.operations.inference.mca")
operation = importlib.import_module("hotpot.cheminfo.operations.inference.mca.attachment")


def _prediction(molecule, offset=0.0):
    atoms = tuple(
        AtomPrediction(index, element, float(offset + index + 10))
        for index, element in enumerate(molecule.atom_symbols)
    )
    sites = tuple(
        SitePrediction(atom.atom_index, atom.element, "fixture", atom.mca_kj_mol)
        for atom in atoms if atom.element in {"N", "O"}
    )
    return MoleculePrediction(
        molecule.smiles, molecule.formal_charge, sites, "fixture", atoms,
    )


@pytest.fixture
def predictor(monkeypatch):
    calls = []

    def predict(molecule, **options):
        assert isinstance(molecule, MoleculeDTO)
        prediction = _prediction(molecule, 100.0 * len(calls))
        calls.append((molecule, options, prediction))
        return prediction

    monkeypatch.setitem(mca_package.__dict__, "predict_mca", predict)
    return calls


def _values(prediction):
    return [atom.mca_kj_mol for atom in prediction.atom_predictions]


def _assert_current(molecule, prediction):
    assert molecule._mca_result.payload == prediction
    assert molecule._mca_result.invalidated_by == ()
    assert molecule._mca_result.source.owner is molecule._owner_token
    assert [atom.mca for atom in molecule.atoms] == _values(prediction)
    assert molecule.mca_sites == {
        molecule.atoms[site.atom_index]: site.mca_kj_mol
        for site in prediction.sites
    }


def test_normalizes_once_captures_after_preparation_and_predicts_without_selection_lock(
    monkeypatch, predictor,
):
    molecule = read_mol("CCN")
    serialization = importlib.import_module("hotpot.cheminfo.operations.inference.mca.serialization")
    normalize = serialization.normalize_molecule
    results = importlib.import_module("hotpot.cheminfo.structure._results")
    capture, check = operation._capture_mca_source, results._check_source
    predict = mca_package.__dict__["predict_mca"]
    select = operation._select_mca_result
    expected_uids = tuple(atom._uid for atom in molecule.atoms)
    events = []
    prepared = []

    def normalize_once(value):
        assert value is molecule
        assert molecule._mutation_lock._is_owned()
        result = normalize(value)
        prepared.append(result)
        events.append("normalize")
        return result

    def capture_prepared(value):
        assert value is molecule and len(prepared) == 1
        assert [atom.implicit_hydrogens for atom in molecule.atoms] == [
            atom.implicit_hydrogens for atom in prepared[0].atoms
        ]
        events.append("capture")
        source = capture(value)
        assert tuple(domain for domain, _ in source.dependencies) == ("mca_representation", "geometry")
        return source

    def predict_once(value, **options):
        assert value is prepared[0]
        assert not molecule._mutation_lock._is_owned()
        assert _current_transaction(molecule) is None
        events.append("predict")
        return predict(value, **options)

    def check_once(value, source):
        assert _current_transaction(value) is not None
        events.append("check")
        return check(value, source)

    def select_bound(value, payload, source, atom_uids):
        assert value is molecule and atom_uids == expected_uids
        assert payload is predictor[0][2]
        return select(value, payload, source, atom_uids)

    monkeypatch.setattr(operation, "_select_mca_result", select_bound)
    monkeypatch.setattr(serialization, "normalize_molecule", normalize_once)
    monkeypatch.setattr(operation, "_capture_mca_source", capture_prepared)
    monkeypatch.setattr(results, "_check_source", check_once)
    monkeypatch.setitem(mca_package.__dict__, "predict_mca", predict_once)
    prediction = mca(molecule, backend="cpu", n_jobs=2)

    assert events == ["normalize", "capture", "predict", "check"]
    assert len(predictor) == 1
    assert predictor[0][1] == {"backend": "cpu", "n_jobs": 2}
    assert prediction is predictor[0][2]
    assert molecule._mca_result.payload is prediction
    _assert_current(molecule, prediction)


def test_absence_reads_and_current_reads_never_predict_or_hash(monkeypatch, predictor):
    molecule = read_mol("CCN")
    with pytest.raises(AttributeError, match="hotpot.cheminfo.operations.inference.mca.mca"):
        _ = molecule.atoms[0].mca
    with pytest.raises(AttributeError, match="hotpot.cheminfo.operations.inference.mca.mca"):
        _ = molecule.mca_sites
    assert predictor == []
    prediction = mca(molecule, backend="cpu")

    def unexpected(*args, **kwargs):
        pytest.fail("MCA access must not predict, capture, or check molecular content")

    for name in (
        "hotpot.cheminfo.structure._identity",
        "hotpot.cheminfo.structure._results",
        "hotpot.cheminfo.core",
        "hotpot.cheminfo.operations.inference.mca.attachment",
    ):
        module = importlib.import_module(name)
        for attribute in ("_capture_source", "_check_source", "_content_identities"):
            if attribute in module.__dict__:
                monkeypatch.setattr(module, attribute, unexpected)
    monkeypatch.setitem(mca_package.__dict__, "predict_mca", unexpected)
    for _ in range(25):
        _assert_current(molecule, prediction)
        sites = molecule.mca_sites
        sites.clear()
    assert len(predictor) == 1


def test_one_payload_annotations_and_historical_values_survive_replacement(predictor):
    molecule = read_mol("CCN")
    annotation = {"owner": "caller"}
    molecule.properties["mca"] = annotation
    first = mca(molecule, backend="cpu")
    old_attachment = molecule._mca_result
    second = mca(molecule, backend="gpu")

    assert molecule._mca_result is not old_attachment
    assert molecule._mca_result.payload is second
    assert first is predictor[0][2] and second is predictor[1][2]
    assert _values(first) == [10.0, 11.0, 12.0]
    assert _values(second) == [110.0, 111.0, 112.0]
    assert molecule.properties["mca"] is annotation
    assert "_mca_sites" not in molecule.__dict__
    assert all("_mca" not in atom.__dict__ for atom in molecule.atoms)
    with pytest.raises(FrozenInstanceError):
        first.atom_predictions[0].mca_kj_mol = 0.0
    with pytest.raises(AttributeError):
        molecule.atoms[0].mca = 0.0
    with pytest.raises(AttributeError):
        molecule.mca_sites = {}
    _assert_current(molecule, second)


@pytest.mark.parametrize("mapping", ["missing", "duplicate", "reordered", "element", "site", "site_element", "negative_site"])
def test_invalid_mapping_retains_exact_previous_selection(monkeypatch, predictor, mapping):
    molecule = read_mol("CCN")
    previous = mca(molecule, backend="cpu")
    attachment = molecule._mca_result
    predict = mca_package.__dict__["predict_mca"]

    def invalid(value, **options):
        prediction = predict(value, **options)
        atoms, sites = prediction.atom_predictions, prediction.sites
        if mapping == "missing":
            atoms = atoms[:-1]
        elif mapping == "duplicate":
            atoms = atoms[:-1] + (atoms[0],)
        elif mapping == "reordered":
            atoms = tuple(reversed(atoms))
        elif mapping == "element":
            atoms = (replace(atoms[0], element="N"),) + atoms[1:]
        elif mapping == "site":
            sites = (replace(sites[0], atom_index=len(atoms)),)
        elif mapping == "site_element":
            sites = (replace(sites[0], element="O"),)
        else:
            sites = (replace(sites[0], atom_index=-1),)
        return replace(prediction, atom_predictions=atoms, sites=sites)

    monkeypatch.setitem(mca_package.__dict__, "predict_mca", invalid)
    with pytest.raises(ValueError, match="MCA .*bindings"):
        mca(molecule, backend="cpu")
    assert len(predictor) == 2
    assert molecule._mca_result is attachment
    _assert_current(molecule, previous)


@pytest.mark.parametrize("domain", ["geometry", "mca_representation"])
def test_source_mutation_during_prediction_rejects_selection_once(monkeypatch, predictor, domain):
    molecule = read_mol("CCN")
    previous = mca(molecule, backend="cpu")
    predict = mca_package.__dict__["predict_mca"]
    changed_attachment = []

    def mutate(value, **options):
        prediction = predict(value, **options)
        if domain == "geometry":
            molecule.atoms[0].coordinates = (1.0, 2.0, 3.0)
        else:
            molecule.atoms[0].formal_charge = 1
        changed_attachment.append(molecule._mca_result)
        return prediction

    monkeypatch.setitem(mca_package.__dict__, "predict_mca", mutate)
    with pytest.raises(RuntimeError, match=domain):
        mca(molecule, backend="cpu")
    assert len(predictor) == 2
    assert molecule._mca_result is changed_attachment[0]
    assert molecule._mca_result.payload is previous
    with pytest.raises(RuntimeError, match=domain):
        _ = molecule.atoms[0].mca
    with pytest.raises(RuntimeError, match=domain):
        _ = molecule.mca_sites
    assert _values(previous) == [10.0, 11.0, 12.0]


def test_other_thread_can_mutate_while_predictor_runs(monkeypatch, predictor):
    molecule = read_mol("CCN")
    predict = mca_package.__dict__["predict_mca"]

    def predict_with_writer(value, **options):
        writer = Thread(
            target=setattr,
            args=(molecule.atoms[0], "coordinates", (3.0, 2.0, 1.0)),
            daemon=True,
        )
        writer.start()
        writer.join(timeout=5)
        assert not writer.is_alive(), "prediction held the molecular mutation lock"
        return predict(value, **options)

    monkeypatch.setitem(mca_package.__dict__, "predict_mca", predict_with_writer)
    with pytest.raises(RuntimeError, match="geometry"):
        mca(molecule, backend="cpu")
    assert len(predictor) == 1
    assert molecule._mca_result is None
    assert tuple(molecule.atoms[0].coordinates) == (3.0, 2.0, 1.0)


def test_predictor_failure_propagates_without_replacing_selection(monkeypatch, predictor):
    molecule = read_mol("CCN")
    previous = mca(molecule, backend="cpu")
    attachment = molecule._mca_result
    failure = RuntimeError("fixture predictor failed")
    calls = []

    def fail(value, **options):
        calls.append(value)
        raise failure

    monkeypatch.setitem(mca_package.__dict__, "predict_mca", fail)
    with pytest.raises(RuntimeError) as caught:
        mca(molecule, backend="cpu")
    assert caught.value is failure
    assert len(calls) == 1
    assert molecule._mca_result is attachment
    _assert_current(molecule, previous)


@pytest.mark.parametrize("poisoned", [False, True])
def test_selection_after_invalidation_then_outer_rollback_restores_exact_payload(predictor, poisoned):
    molecule = read_mol("CCN")
    previous = mca(molecule, backend="cpu")
    attachment = molecule._mca_result
    coordinates = tuple(molecule.atoms[0].coordinates)
    failure = ValueError("rollback selected result")
    with pytest.raises(RuntimeError if poisoned else ValueError):
        with _FactTransaction(molecule, producer=ATOM_SOURCE_CODES["explicit"]):
            molecule.atoms[0].coordinates = (1.0, 2.0, 3.0)
            replacement = mca(molecule, backend="cpu")
            assert molecule._mca_result.payload is replacement
            assert molecule._mca_result.invalidated_by == ()
            if poisoned:
                try:
                    with _FactTransaction(molecule, producer=ATOM_SOURCE_CODES["explicit"]):
                        raise failure
                except ValueError:
                    pass
            else:
                raise failure
    assert len(predictor) == 2
    assert molecule._mca_result is attachment
    assert tuple(molecule.atoms[0].coordinates) == coordinates
    _assert_current(molecule, previous)


def test_exact_copy_preserves_payload_and_rebinds_once_without_coupled_currentness(predictor):
    molecule = read_mol("CCN")
    prediction = mca(molecule, backend="cpu")
    clone = copy(molecule)
    assert clone._mca_result.payload is prediction
    assert clone._mca_result.source.owner is not molecule._mca_result.source.owner
    _assert_current(clone, prediction)
    clone.atoms[0].coordinates = (1.0, 2.0, 3.0)
    with pytest.raises(RuntimeError, match="geometry"):
        _ = clone.atoms[0].mca
    _assert_current(molecule, prediction)
    assert len(predictor) == 1


@pytest.mark.parametrize("transport", [deepcopy, lambda molecule: pickle.loads(pickle.dumps(molecule))], ids=["deepcopy", "pickle"])
@pytest.mark.parametrize("stale", [False, True])
def test_transport_preserves_payload_status_and_shared_owner_token(predictor, transport, stale):
    molecule = read_mol("CCN")
    prediction = mca(molecule, backend="cpu")
    if stale:
        molecule.atoms[0].coordinates = (1.0, 2.0, 3.0)
    restored = transport(molecule)
    assert restored._mca_result.payload == prediction
    assert restored._mca_result.source.owner is restored._owner_token
    assert restored._mca_result.invalidated_by == molecule._mca_result.invalidated_by
    assert all(atom.mol is restored for atom in restored.atoms)
    if stale:
        with pytest.raises(RuntimeError, match="geometry"):
            _ = restored.atoms[0].mca
        with pytest.raises(RuntimeError, match="geometry"):
            _ = restored.mca_sites
    else:
        _assert_current(restored, prediction)
    assert len(predictor) == 1


def test_exact_copy_of_stale_result_retains_stale_marker(predictor):
    molecule = read_mol("CCN")
    prediction = mca(molecule, backend="cpu")
    molecule.atoms[0].coordinates = (1.0, 2.0, 3.0)
    clone = molecule.copy()
    assert clone._mca_result.payload is prediction
    assert clone._mca_result.invalidated_by == molecule._mca_result.invalidated_by
    assert clone._mca_result.source.owner is clone._owner_token
    with pytest.raises(RuntimeError, match="geometry"):
        _ = clone.mca_sites
    assert len(predictor) == 1


def test_component_projection_drops_selection_even_for_single_component(predictor):
    for smiles in ("CCN", "CCN.CO"):
        molecule = read_mol(smiles, fmt="smi")
        prediction = mca(molecule, backend="cpu")
        for component in molecule.components:
            assert component._mca_result is None
            with pytest.raises(AttributeError, match="hotpot.cheminfo.operations.inference.mca.mca"):
                _ = component.mca_sites
            with pytest.raises(AttributeError, match="hotpot.cheminfo.operations.inference.mca.mca"):
                _ = component.atoms[0].mca
        _assert_current(molecule, prediction)
    assert len(predictor) == 2


@pytest.mark.parametrize("change", ["merge", "drop"])
def test_merge_and_atom_removal_mark_whole_graph_prediction_stale(predictor, change):
    molecule = read_mol("CCN")
    prediction = mca(molecule, backend="cpu")
    if change == "merge":
        molecule.add_component(read_mol("CO"))
    else:
        molecule.remove_atom(molecule.atoms[0])
    assert molecule._mca_result.payload is prediction
    with pytest.raises(RuntimeError, match="mca_representation"):
        _ = molecule.atoms[0].mca
    with pytest.raises(RuntimeError, match="mca_representation"):
        _ = molecule.mca_sites
    assert len(predictor) == 1


@pytest.mark.parametrize("change", ["remove", "replace"])
def test_prediction_cannot_select_rows_after_graph_membership_changes(monkeypatch, predictor, change):
    molecule = read_mol("CCN")
    previous = mca(molecule, backend="cpu")
    predict = mca_package.__dict__["predict_mca"]
    selected_after_change = []

    def change_membership(value, **options):
        prediction = predict(value, **options)
        molecule.remove_atom(molecule.atoms[-1])
        if change == "replace":
            molecule.create_atom(atomic_number=7)
        selected_after_change.append(molecule._mca_result)
        return prediction

    monkeypatch.setitem(mca_package.__dict__, "predict_mca", change_membership)
    with pytest.raises(RuntimeError, match="mca_representation"):
        mca(molecule, backend="cpu")
    assert len(predictor) == 2
    assert molecule._mca_result is selected_after_change[0]
    assert molecule._mca_result.payload is previous
    assert _values(previous) == [10.0, 11.0, 12.0]
