"""Selected MCA ownership, cheap reads and factual lifecycle boundaries."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import pickle

import numpy as np
import pytest

from hotpot import Atom, Molecule
from hotpot.cheminfo.AImodels.mca.result_types import (
    AtomPrediction, MoleculePrediction, SitePrediction,
)
from hotpot.cheminfo.structure import _identity, _results
from hotpot.cheminfo.structure._commit import (
    _FactTransaction, _commit_atom_sparse, _commit_stereo, _commit_topology,
)
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
from hotpot.cheminfo.structure._identity import _capture_source
from hotpot.cheminfo.structure.conformer import (
    _import_frame_values, _capture_frame_records, _install_frame_records,
)
from hotpot.cheminfo.structure._results import (
    _ResultAttachment, _capture_mca_source, _prepare_atom_result_updates,
    _select_mca_result, _import_partial_charges,
)
from hotpot.cheminfo.structure.stereo import AtomTransportRecord, StereoPayload


def _molecule():
    mol = Molecule()
    Atom(mol, atomic_number=6, coordinates=(0.0, 0.0, 0.0))
    Atom(mol, atomic_number=8, coordinates=(1.4, 0.0, 0.0))
    mol.add_bond(0, 1)
    return mol


def _prediction(mol, offset=0.0):
    atoms = tuple(AtomPrediction(row, atom.symbol, offset + 10.0 + row)
                  for row, atom in enumerate(mol.atoms))
    return MoleculePrediction("CO", 0, (SitePrediction(1, "O", "oxygen", atoms[1].mca_kj_mol),),
                              "test", atoms)


def _select(mol, prediction=None):
    prediction = _prediction(mol) if prediction is None else prediction
    return _select_mca_result(mol, prediction, _capture_mca_source(mol),
                              tuple(atom._uid for atom in mol.atoms))


def test_absence_and_exact_payload_selection_keep_annotation_independent():
    mol = _molecule()
    mol.properties["mca"] = "caller annotation"
    with pytest.raises(AttributeError, match=r"calculator\.mca"):
        _ = mol.atoms[0].mca
    with pytest.raises(AttributeError, match=r"calculator\.mca"):
        _ = mol.mca_sites
    prediction = _select(mol)
    assert mol._mca_result.payload is prediction
    assert [atom.mca for atom in mol.atoms] == [10.0, 11.0]
    assert mol.mca_sites == {mol.atoms[1]: 11.0}
    detached_sites = mol.mca_sites
    detached_sites.clear()
    assert mol.mca_sites == {mol.atoms[1]: 11.0}
    assert mol.properties["mca"] == "caller annotation"
    assert "_mca_sites" not in mol.__dict__
    assert all("_mca" not in atom.__dict__ for atom in mol.atoms)
    with pytest.raises(FrozenInstanceError):
        mol._mca_result.payload = None


def test_scalar_and_site_reads_never_hash_or_check_sources(monkeypatch):
    mol = _molecule()
    _select(mol)

    def forbidden(*args, **kwargs):
        pytest.fail("a selected-result getter entered source hashing")

    for module in (_identity, _results):
        monkeypatch.setattr(module, "_capture_source", forbidden)
        monkeypatch.setattr(module, "_check_source", forbidden)
    for _ in range(20):
        assert [atom.mca for atom in mol.atoms] == [10.0, 11.0]
        assert mol.mca_sites == {mol.atoms[1]: 11.0}


def test_duplicate_site_indices_fail_selection_without_overwriting_current_payload():
    mol = _molecule()
    prediction = _select(mol)
    before = mol._mca_result
    duplicate = replace(prediction, sites=prediction.sites + prediction.sites)
    with pytest.raises(ValueError, match="MCA site.*bindings"):
        _select(mol, duplicate)
    assert mol._mca_result is before
    assert mol.mca_sites == {mol.atoms[1]: 11.0}


def test_native_hook_prepares_only_subscribed_replacements_without_fact_writes():
    mol = _molecule()
    prediction = _select(mol)
    before = mol._mca_result
    assert _prepare_atom_result_updates(mol, ("partial_charge", "id", "valence"), None) == ()
    updates = _prepare_atom_result_updates(mol, ("x", "formal_charge"), None)
    assert len(updates) == 1
    name, previous, after = updates[0]
    assert name == "_mca_result" and previous is before
    assert after.payload is prediction and after.source is before.source
    assert after.invalidated_by == ("mca_representation", "geometry")
    assert mol._mca_result is before
    assert mol.atoms[1].mca == 11.0


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda mol: _commit_atom_sparse(mol, mol.atoms[0], isotope=13), id="isotope"),
    pytest.param(lambda mol: setattr(mol, "charge", 1), id="charge-declaration"),
    pytest.param(lambda mol: mol.bonds[0]._set_bond_metadata(
        bond_kind=mol.bonds[0].bond_kind, bond_direction="atom1_to_atom2"), id="bond-direction"),
    pytest.param(lambda mol: mol.bonds[0]._set_bond_metadata(
        bond_kind=mol.bonds[0].bond_kind, bond_source="supplied"), id="bond-source"),
    pytest.param(lambda mol: mol.bonds[0]._set_bond_metadata(
        bond_kind=mol.bonds[0].bond_kind, bond_source_metadata={"producer": "user"}), id="bond-metadata"),
    pytest.param(lambda mol: _commit_stereo(mol, _capture_source(mol), StereoPayload(
        1, (), (AtomTransportRecord(0, 13, None, None, None, "test"),), ())), id="stereo"),
])
def test_python_representation_changes_leave_historical_payload_and_stale_direct_reads(mutation):
    mol = _molecule()
    prediction = _select(mol)
    mutation(mol)
    assert mol._mca_result.payload is prediction
    assert "mca_representation" in mol._mca_result.invalidated_by
    with pytest.raises(RuntimeError, match=r"mca_representation.*calculator\.mca"):
        _ = mol.atoms[1].mca
    with pytest.raises(RuntimeError, match=r"mca_representation.*calculator\.mca"):
        _ = mol.mca_sites
    assert prediction.atom_predictions[1].mca_kj_mol == 11.0


def test_annotations_constraints_partial_charge_and_frac_valence_retain_selection():
    mol = _molecule()
    _select(mol)
    before = mol._mca_result
    mol.properties["mca"] = {"caller": "data"}
    mol.identifier = "renamed"
    mol.atoms[0].id = 100
    mol.atoms[0].constraint = True
    mol.bonds[0].constraint = True
    charges = _import_partial_charges(mol, (0.25, None), producer=ATOM_SOURCE_CODES['explicit'])
    assert charges.values == (0.25, None)
    assert mol.atoms[0].partial_charge == 0.25 and mol.atoms[1].partial_charge is None
    assert mol._mca_result is before
    _commit_atom_sparse(mol, mol.atoms[0], frac_valence=2.5)
    mol.conformer_add(mol.coordinates)
    assert mol._mca_result is before
    assert mol.atoms[1].mca == 11.0


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda mol: Atom(mol, atomic_number=1), id="direct-registration"),
    pytest.param(lambda mol: mol.add_atom(Atom(atomic_number=1)), id="atom-add"),
    pytest.param(lambda mol: mol.remove_atom(0), id="atom-remove"),
    pytest.param(lambda mol: mol.remove_bond(mol.bonds[0]), id="bond-remove"),
    pytest.param(lambda mol: mol.add_component(_molecule()), id="merge"),
])
def test_topology_changes_invalidate_whole_prediction(mutation):
    mol = _molecule()
    prediction = _select(mol)
    mutation(mol)
    assert mol._mca_result.payload is prediction
    assert set(mol._mca_result.invalidated_by) == {"mca_representation", "geometry"}
    with pytest.raises(RuntimeError, match="calculator.mca"):
        _ = mol.mca_sites


def test_current_membership_check_rejects_removed_atom_access():
    mol = _molecule()
    _select(mol)
    removed = mol.atoms[0]
    mol.remove_atom(removed)
    with pytest.raises(RuntimeError, match=r"membership.*calculator\.mca"):
        _ = removed.mca


def test_complete_install_marks_selection_even_with_retained_dependency_state():
    mol = _molecule()
    _select(mol)
    record, metadata = mol._capture_factual_record(include_frames=False)
    from hotpot.cheminfo.structure._dependencies import _dependency_state
    _commit_topology(mol, _capture_source(mol), record, atom_bindings=tuple(mol.atoms),
                     bond_bindings=tuple(mol.bonds), dependency_state=_dependency_state(mol._atom_table),
                     bond_metadata=metadata)
    with pytest.raises(RuntimeError, match="mca_representation"):
        _ = mol.atoms[0].mca


def test_invalidation_then_selection_then_abort_restores_exact_original_attachment():
    mol = _molecule()
    original_payload = _select(mol)
    original_attachment = mol._mca_result
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1):
            _commit_atom_sparse(mol, mol.atoms[0], isotope=13)
            replacement = _select(mol, _prediction(mol, offset=100.0))
            assert mol.atoms[1].mca == 111.0
            raise KeyboardInterrupt("abort after selection")
    assert mol._mca_result is original_attachment
    assert mol._mca_result.payload is original_payload
    assert replacement.atom_predictions[1].mca_kj_mol == 111.0
    assert "isotope" not in mol.atoms[0].__dict__
    assert mol.atoms[1].mca == 11.0


def test_direct_registration_abort_restores_membership_and_original_attachment():
    mol = _molecule()
    _select(mol)
    original = mol._mca_result
    original_atoms = tuple(mol.atoms)
    with pytest.raises(RuntimeError, match="abort registration"):
        with _FactTransaction(mol, producer=1):
            Atom(mol, atomic_number=1)
            raise RuntimeError("abort registration")
    assert tuple(mol.atoms) == original_atoms
    assert mol._mca_result is original
    assert mol.atoms[1].mca == 11.0


def test_caught_nested_selection_failure_poison_restores_prior_selection():
    mol = _molecule()
    _select(mol)
    original = mol._mca_result
    bad = replace(_prediction(mol), atom_predictions=())
    with pytest.raises(RuntimeError, match="nested factual"):
        with _FactTransaction(mol, producer=1):
            _select(mol, _prediction(mol, offset=100.0))
            try:
                _select(mol, bad)
            except ValueError:
                pass
    assert mol._mca_result is original
    assert mol.atoms[1].mca == 11.0


def test_exact_copy_rebinds_once_and_preserves_payload_reference_and_marker():
    mol = _molecule()
    prediction = _select(mol)
    mol.properties["note"] = "original"
    copied = mol.copy()
    assert copied._mca_result.payload is prediction
    assert copied._mca_result.source.owner is copied._owner_token
    assert copied._owner_token is not mol._owner_token
    assert copied.atoms[1].mca == 11.0
    copied.properties["note"] = "copy"
    assert mol.properties["note"] == "original"
    _commit_atom_sparse(mol, mol.atoms[0], isotope=13)
    stale = mol.copy()
    assert stale._mca_result.payload is prediction
    assert stale._mca_result.invalidated_by == mol._mca_result.invalidated_by
    with pytest.raises(RuntimeError, match="mca_representation"):
        _ = stale.mca_sites
    assert copied.atoms[1].mca == 11.0


def test_copy_rejects_mismatched_current_source_before_rebinding():
    mol = _molecule()
    prediction = _select(mol)
    foreign = _molecule()
    mol._mca_result = _ResultAttachment(prediction, _capture_mca_source(foreign))
    with pytest.raises(RuntimeError, match="owner"):
        mol.copy()


def test_component_projection_drops_whole_graph_prediction():
    mol = _molecule()
    _select(mol)
    for component in mol.components:
        assert component._mca_result is None
        with pytest.raises(AttributeError, match="calculator.mca"):
            _ = component.mca_sites
    assert mol.atoms[1].mca == 11.0


@pytest.mark.parametrize("transport", [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
@pytest.mark.parametrize("stale", [False, True])
def test_pickle_and_deepcopy_share_restored_owner_token_with_attachment(transport, stale):
    mol = _molecule()
    prediction = _select(mol)
    if stale:
        _commit_atom_sparse(mol, mol.atoms[0], isotope=13)
    restored, restored_atom = transport((mol, mol.atoms[1]))
    assert restored_atom is restored.atoms[1]
    assert restored._mca_result.source.owner is restored._owner_token
    assert restored._mca_result.payload == prediction
    assert restored._mca_result.invalidated_by == mol._mca_result.invalidated_by
    if stale:
        with pytest.raises(RuntimeError, match="mca_representation"):
            _ = restored_atom.mca
    else:
        assert restored_atom.mca == 11.0
        assert restored.mca_sites == {restored_atom: 11.0}


@pytest.mark.parametrize("change", ["declared_spin", "direct_clear", "owned_clear", "load", "install_frames"])
def test_spin_only_changes_retain_mca_source_reads_and_copy_while_generic_source_changes(change):
    mol = _molecule()
    coordinates = mol.coordinates
    _import_frame_values(mol, (np.stack((coordinates, coordinates)), None, None, None,
                               None, None, np.asarray([1, 3]), None, None),
                         0, producer="test-spin")
    mol.conformer_load(0)
    prediction = _select(mol)
    original = mol._mca_result
    representation = _capture_source(mol, ("representation",))
    if change == "declared_spin":
        mol.spin_mult = 3
    elif change == "direct_clear":
        mol.conformers.clear()
    elif change == "owned_clear":
        mol.conformer_clear()
    elif change == "load":
        mol.conformer_load(1)
    else:
        frames, generated, selection = _capture_frame_records(mol)
        observations = frames[1].observations
        _install_frame_records(mol, frames, generated, (1, ((observations, 1), ())))
    with pytest.raises(RuntimeError, match="'representation'"):
        _identity._check_source(mol, representation)
    _identity._check_source(mol, original.source)
    assert mol._mca_result is original
    assert mol.atoms[1].mca == 11.0
    assert mol.mca_sites == {mol.atoms[1]: 11.0}
    copied = mol.copy()
    assert copied.spin_mult == mol.spin_mult
    assert copied._mca_result.payload is prediction
    assert copied._mca_result.invalidated_by == ()
    assert copied.atoms[1].mca == 11.0


def test_spin_only_frame_change_then_rollback_preserves_exact_attachment_and_frame_selection():
    mol = _molecule()
    coordinates = mol.coordinates
    _import_frame_values(mol, (np.stack((coordinates, coordinates)), None, None, None,
                               None, None, np.asarray([1, 3]), None, None),
                         0, producer="test-spin")
    mol.conformer_load(0)
    _select(mol)
    original = mol._mca_result
    with pytest.raises(RuntimeError, match="abort frame"):
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["conformer_load"]):
            mol.conformer_load(1)
            assert mol._mca_result is original
            assert mol.spin_mult == 3
            raise RuntimeError("abort frame")
    assert mol._conformers_index == 0
    assert mol.spin_mult == 1
    assert mol._mca_result is original
    assert mol.atoms[1].mca == 11.0


def test_spin_declaration_during_prediction_keeps_mca_selection_source_current(monkeypatch):
    from importlib import import_module
    from hotpot.cheminfo.calculator import mca

    mol = _molecule()
    calls = []

    def predict(molecule, **options):
        representation = _capture_source(mol, ("representation",))
        mol.spin_mult = 3
        with pytest.raises(RuntimeError, match="'representation'"):
            _identity._check_source(mol, representation)
        prediction = _prediction(mol)
        calls.append(prediction)
        return prediction

    mca_package = import_module("hotpot.cheminfo.AImodels.mca")
    monkeypatch.setitem(mca_package.__dict__, "predict_mca", predict)
    prediction = mca(mol, backend="cpu")
    assert calls == [prediction]
    assert mol._mca_result.payload is prediction
    assert mol._mca_result.invalidated_by == ()
    assert mol.atoms[1].mca == 11.0


def _bond_signed_zero_fixture():
    mol = _molecule()
    bond = mol.bonds[0]
    bond.bond_order = 0.0
    bond._set_bond_metadata(bond_kind=bond.bond_kind,
                            bond_source_metadata={"nested": {"value": 0.0}})
    prediction = _select(mol)
    return mol, bond, prediction


def _negative_zero_bond_write(bond, change):
    if change == "order":
        bond.bond_order = -0.0
    else:
        bond._set_bond_metadata(bond_kind=bond.bond_kind,
                                bond_source_metadata={"nested": {"value": -0.0}})


@pytest.mark.parametrize("change", ["order", "metadata"])
def test_bond_signed_zero_source_changes_mark_stale_and_reselection_retains_constraint_only_writes(change):
    mol, bond, prediction = _bond_signed_zero_fixture()
    original = mol._mca_result
    historical = prediction.to_dict()
    _negative_zero_bond_write(bond, change)
    assert mol._mca_result.payload is prediction
    assert mol._mca_result.invalidated_by == ("mca_representation",)
    assert prediction.to_dict() == historical
    with pytest.raises(RuntimeError, match="mca_representation"):
        _identity._check_source(mol, original.source)
    with pytest.raises(RuntimeError, match="mca_representation"):
        _ = mol.atoms[0].mca
    with pytest.raises(RuntimeError, match="mca_representation"):
        _ = mol.mca_sites
    _select(mol)
    selected = mol._mca_result
    bond.constraint = True
    assert mol._mca_result is selected
    _identity._check_source(mol, selected.source)
    assert mol.atoms[1].mca == 11.0


@pytest.mark.parametrize("change", ["order", "metadata"])
def test_bond_signed_zero_invalidation_and_facts_rollback_together(change):
    mol, bond, prediction = _bond_signed_zero_fixture()
    original = mol._mca_result
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1):
            _negative_zero_bond_write(bond, change)
            assert mol._mca_result.invalidated_by == ("mca_representation",)
            raise KeyboardInterrupt("abort signed-zero Bond write")
    assert not np.signbit(bond.bond_order)
    assert not np.signbit(bond.bond_source_metadata["nested"]["value"])
    assert mol._mca_result is original
    assert mol._mca_result.payload is prediction
    _identity._check_source(mol, original.source)
    assert mol.atoms[1].mca == 11.0
