"""Source/generated frame ownership, historical projections and atomic selection."""

from dataclasses import FrozenInstanceError, replace
from importlib import import_module
import inspect
import pickle
from typing import Union

import numpy as np
import pytest

from hotpot import Atom, Molecule
from hotpot.utils import types
from hotpot.cheminfo.conformers.contracts import ConformerBatch
from hotpot.cheminfo.structure import _identity
from hotpot.cheminfo.structure._builder import _thaw_value, _frame_records as _validate_frame_records
from hotpot.cheminfo.structure._commit import _FactTransaction, _native_fail_after
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from hotpot.cheminfo.structure._results import _prepare_atom_result_updates
from hotpot.cheminfo.structure.conformer import (
    Conformers, _capture_frame_records, _frame_records_from_values, _import_frame_values,
    _install_frame_records, _publish_conformer_ensemble, _select_frame_observations,
)


_NAMES = Conformers._attrs


def _molecule():
    mol = Molecule()
    Atom(mol, atomic_number=6, coordinates=(0.0, 0.0, 0.0))
    Atom(mol, atomic_number=8, coordinates=(1.4, 0.0, 0.0))
    mol.add_bond(0, 1)
    return mol


def _import(mol, **observations):
    coords = np.stack((mol.coordinates, mol.coordinates + (0.25, 0, 0)))
    values = (coords, *(observations.get(name) for name in _NAMES[1:]))
    _import_frame_values(mol, values, 0, producer="test-source", method="supplied",
                         units=(("coordinates", "angstrom"), ("energy", "eV")))
    return coords


def _publish(mol, energies=(1.0, 2.0)):
    coords = np.stack((mol.coordinates + (0, 0.25, 0), mol.coordinates + (0, 0.5, 0)))
    return _publish_conformer_ensemble(
        mol, ConformerBatch(coords, (4, 9)), {"energy": energies}, 1,
        input_source=_capture_source(mol, ("geometry", "source_frames")),
        method="test-optimizer", parameters=(("seed", 12),), units=(("energy", "kJ/mol"),),
    )


def test_standalone_empty_collection_and_public_spelling():
    frames = Conformers()
    assert Conformers.__module__ == "hotpot.cheminfo.core"
    assert import_module(Conformers.__module__).Conformers is Conformers
    assert len(frames) == 0 and list(frames) == []
    assert set(frames[:]) == set(_NAMES)
    assert frames[:]["coordinates"].shape == (0, 0, 3)


def test_public_add_signature_preserves_runtime_annotation_objects():
    signature = inspect.signature(Conformers.add)
    assert tuple(signature.parameters) == ("self", "coords", "energy")
    assert signature.parameters["self"].annotation is inspect.Parameter.empty
    assert signature.parameters["coords"].annotation is types.ArrayLike
    assert signature.parameters["energy"].annotation == Union[types.ArrayLike, float]
    assert signature.parameters["energy"].default is None
    assert signature.return_annotation is inspect.Signature.empty


def test_standalone_add_appends_energy_and_keeps_absence_distinct_from_zero():
    frames = Conformers()
    coords = np.arange(6.0).reshape(2, 3)
    frames.add(coords)
    frames.add(coords + 1, 0.0)
    frames.add(coords + 2, [4.5])
    assert len(frames) == 3
    assert frames[0]["energy"] is None
    assert frames[1]["energy"] == 0.0
    assert frames[2]["energy"] == 4.5
    np.testing.assert_equal(frames[:]["energy"], [np.nan, 0.0, 4.5])
    assert all(set(frames[row]) == set(_NAMES) for row in range(3))
    assert len(list(frames)) == 3
    frames.clear()
    assert len(frames) == 0 and list(frames) == []


@pytest.mark.parametrize("coords,energy", [
    (np.ones((2, 2)), None), (np.ones((2, 2, 4)), None),
    (np.ones((2, 2, 3)), [1.0]), (np.ones((3, 3)), None),
])
def test_standalone_malformed_shape_or_energy_failure_is_atomic(coords, energy):
    frames = Conformers()
    frames.add(np.ones((2, 3)), 2.5)
    before = frames._source_frames
    with pytest.raises(ValueError):
        frames.add(coords, energy)
    assert frames._source_frames is before
    assert frames.energy(0) == 2.5


def test_bound_add_is_a_projection_and_never_selects_or_changes_current_facts():
    mol = _molecule()
    original = mol.coordinates
    mol.conformers.add(original + 10, 0.0)
    assert vars(mol.conformers) == {"_owner": mol}
    np.testing.assert_equal(mol.coordinates, original)
    assert mol.energy is None and mol._frame_result is None
    assert mol.conformer_get(0)["energy"] == 0.0
    mol.conformers.clear()
    assert mol.conformers_number == 0 and mol.energy is None


def test_bound_invalid_preparation_preserves_exact_source_owner_and_selection():
    mol = _molecule()
    _import(mol, energy=np.array([0.0, 2.0]))
    before = mol._source_frames, mol._frame_result, mol._conformers_index
    with pytest.raises(ValueError):
        mol.conformer_add(np.ones((3, 2, 3)), [4.0, 5.0])
    assert mol._source_frames is before[0] and mol._frame_result is before[1]
    assert mol._conformers_index == before[2]


def test_projection_values_and_nested_observations_are_isolated():
    mol = _molecule()
    _import(mol, energy=[0.0, 2.0], force=np.arange(12.0).reshape(2, 2, 3),
            thermo={"nested": [np.array([3.0])]})
    first = mol.conformer_get(0)
    first["coordinates"][0, 0] = 100
    first["force"][0, 0] = 100
    first["thermo"]["nested"].append("caller")
    again = mol.conformer_get(0)
    assert again["coordinates"][0, 0] == 0
    assert again["force"][0, 0] == 0
    assert len(again["thermo"]["nested"]) == 1
    np.testing.assert_equal(mol.conformer_get(slice(None))["energy"], [0.0, 2.0])


def test_source_and_generated_buffers_remain_immutable_after_pickle():
    mol = _molecule()
    _import(mol, energy=[0.0, 2.0])
    generated = _publish(mol)
    for value in (mol._source_frames[0], generated):
        restored = pickle.loads(pickle.dumps(value))
        array = restored.coordinates if hasattr(restored, "coordinates") else restored.batch.coordinates
        with pytest.raises(ValueError):
            array.flags.writeable = True
        with pytest.raises(FrozenInstanceError):
            restored.atom_uids = ()
        raw = _thaw_value(restored.observations.values[0])
        if isinstance(raw, np.ndarray):
            with pytest.raises(ValueError):
                raw.flags.writeable = True


def test_shared_observation_rows_preserve_last_only_charges_zero_and_unmapped_history():
    mol = _molecule()
    coords = np.stack((mol.coordinates, mol.coordinates + 1))
    values = (coords, np.array([0.0, 5.0, 9.0]), np.array([0.0, 0.2]),
              0.0, None, None, 1, None, None)
    with pytest.warns(RuntimeWarning, match=r"frame_observation_unmapped field=energy.*2.*3"):
        _import_frame_values(mol, values, 0, producer="test-Gaussian")
    assert mol._source_frames[0].observations is mol._source_frames[1].observations
    assert mol.conformer_get(0)["energy"] is None
    assert mol.conformer_get(0)["partial_charge"] is None
    assert mol.conformer_get(1)["energy"] is None
    np.testing.assert_equal(mol.conformer_get(1)["partial_charge"], [0.0, 0.2])
    assert mol.conformer_get(0)["gibbs"] == 0.0
    np.testing.assert_equal(_thaw_value(mol._source_frames[0].observations.values[0]), [0, 5, 9])
    assert all(atom.partial_charge is None for atom in mol.atoms)


def test_explicit_selector_maps_histories_without_inference():
    mol = _molecule()
    coords = np.stack((mol.coordinates, mol.coordinates + 1))
    _import_frame_values(mol, (coords, [3, 7, 9], None, None, None, None, None, None, None),
                         0, producer="test", frame_indices={"energy": (2, 0)})
    assert mol.conformer_get(0)["energy"] == 9
    assert mol.conformer_get(1)["energy"] == 3


def test_source_order_precedes_ensemble_attachment_and_original_member_order():
    mol = _molecule()
    _import(mol, energy=[0.0, 2.0])
    sources = mol._source_frames
    first = _publish(mol, (None, 5.0))
    second = _publish(mol, (7.0, 8.0))
    assert mol._source_frames is sources
    assert tuple(item.payload for item in mol._conformer_results) == (first, second)
    assert first.batch.ordinals == second.batch.ordinals == (4, 9)
    assert [mol.conformer_get(row)["energy"] for row in range(6)] == [0, 2, None, 5, 7, 8]
    np.testing.assert_equal(mol.conformer_get(slice(None))["energy"], [0, 2, np.nan, 5, 7, 8])
    assert mol.energy == 8
    mol.conformer_load(2)
    assert mol.energy is None
    assert not mol._conformer_results[0].invalidated_by
    mol.conformer_load(3)
    assert mol.energy == 5


def test_generated_publication_leaves_source_identity_current_and_tracks_historical_input():
    mol = _molecule()
    mol.conformer_add(mol.coordinates, 2.0)
    source = _capture_source(mol, ("source_frames",))
    generated = _publish(mol)
    _check_source(mol, source)
    assert [name for name, _ in generated.input_source.dependencies] == ["geometry", "source_frames"]
    attachment = mol._conformer_results[0]
    assert [name for name, _ in attachment.source.dependencies] == ["frame_representation", "cell", "constraints"]
    _check_source(mol, attachment.source)
    mol.conformer_load(1)
    _check_source(mol, attachment.source)
    assert mol.energy == 1


@pytest.mark.parametrize("change", [
    lambda mol: setattr(mol.atoms[0], "formal_charge", 1),
    lambda mol: setattr(mol.atoms[0], "x_constraint", True),
    lambda mol: mol.create_crystal(10, 11, 12, 90, 90, 90),
])
def test_generated_member_stale_rejection_preserves_payload_and_current_geometry(change):
    mol = _molecule()
    generated = _publish(mol)
    change(mol)
    before = mol.coordinates
    with pytest.raises(RuntimeError, match="conformer_result_stale"):
        mol.conformer_load(0)
    np.testing.assert_equal(mol.coordinates, before)
    assert mol._conformer_results[0].payload is generated
    assert mol.conformer_get(0)["energy"] == 1


@pytest.mark.parametrize("write", [
    lambda mol: setattr(mol.atoms[0], "x", 0.5),
    lambda mol: setattr(mol.atoms[0], "coordinates", (0.5, 0.0, 0.0)),
    lambda mol: setattr(mol, "coordinates", mol.coordinates + 1),
    lambda mol: mol.updata_atoms(x=[0.5, 1.4]),
])
def test_native_geometry_writes_stale_selected_observations_without_staling_ensemble(write):
    mol = _molecule()
    generated = _publish(mol)
    historical = generated.observations
    write(mol)
    with pytest.raises(RuntimeError, match="frame_observation_stale.*geometry"):
        _ = mol.energy
    assert mol._frame_result.payload[0] is historical
    assert mol._conformer_results[0].invalidated_by == ()
    mol.conformer_load(1)
    assert mol.energy == 2


def test_native_result_callback_returns_closed_attribute_updates_for_ensemble_tuple():
    mol = _molecule()
    generated = _publish(mol)
    before = mol._conformer_results
    updates = _prepare_atom_result_updates(mol, ("x_constraint",), None)
    assert {name for name, _, _ in updates} == {"_frame_result", "_conformer_results"}
    replacement = next(after for name, _, after in updates if name == "_conformer_results")
    assert type(replacement) is tuple and replacement[0].payload is generated
    assert replacement[0].invalidated_by == ("constraints",)
    assert mol._conformer_results is before


def test_current_observation_reads_do_not_hash_sources_or_walk_all_atoms(monkeypatch):
    mol = _molecule()
    _select_frame_observations(mol, {"energy": 0.0}, producer="single-point")

    def forbidden(*args):
        pytest.fail("current energy read entered source capture or Atom traversal")

    from hotpot.cheminfo.structure import conformer
    monkeypatch.setattr(conformer, "_capture_source", forbidden)
    monkeypatch.setattr(conformer, "_uids", forbidden)
    monkeypatch.setattr(_identity, "_check_source", forbidden)
    for _ in range(20):
        assert mol.energy == 0.0


def test_single_field_assignment_keeps_bounded_per_field_attribution_and_other_values():
    mol = _molecule()
    _select_frame_observations(mol, {"energy": 1.0, "force": np.ones((2, 3))},
                               producer="optimizer", method="named-method", parameters=(("seed", 42),))
    original = _thaw_value(mol._frame_result.payload[0].provenance)["fields"]["force"]
    for value in range(12):
        mol.energy = value
    assert mol.energy == 11
    np.testing.assert_equal(mol.force, np.ones((2, 3)))
    attribution = _thaw_value(mol._frame_result.payload[0].provenance)["fields"]
    assert attribution["force"] == original
    assert attribution["energy"]["producer"] == "explicit"
    assert len(attribution) == 2
    assert "energy" not in mol.__dict__ and "force" not in mol.__dict__


@pytest.mark.parametrize("operation", ["load", "clear", "publish", "add"])
def test_frame_operations_rollback_exact_owners_attachments_and_cursor(operation):
    mol = _molecule()
    _import(mol, energy=[3.0, 4.0])
    _publish(mol)
    owners = mol._source_frames, mol._conformer_results, mol._frame_result, mol._conformers_index
    coords = mol.coordinates
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1):
            if operation == "load":
                mol.conformer_load(0)
            elif operation == "clear":
                mol.conformers.clear()
            elif operation == "publish":
                _publish(mol)
            else:
                mol.conformers.add(mol.coordinates + 4, 9.0)
            raise KeyboardInterrupt("abort frame operation")
    assert mol._source_frames is owners[0] and mol._conformer_results is owners[1]
    assert mol._frame_result is owners[2] and mol._conformers_index == owners[3]
    np.testing.assert_equal(mol.coordinates, coords)


def test_interruption_during_native_frame_load_restores_exact_selection():
    mol = _molecule()
    _import(mol, energy=[3.0, 4.0])
    attachment = mol._frame_result
    coords = mol.coordinates
    with pytest.raises(KeyboardInterrupt):
        with _native_fail_after("value"):
            mol.conformer_load(1)
    assert mol._frame_result is attachment and mol._conformers_index == 0
    np.testing.assert_equal(mol.coordinates, coords)
    assert mol.energy == 3


def test_load_observations_never_changes_atom_charges_net_charge_or_spin_declaration():
    mol = _molecule()
    _import(mol, energy=[3.0, 4.0], partial_charge=np.array([[0.1, -0.1], [0.2, -0.2]]),
            spin_mult=np.array([1, 3]))
    mol.charge = 2
    mol.spin_mult = 5
    mol.conformer_load(1)
    assert mol.charge == 2 and mol.spin_mult == 5
    assert [atom.partial_charge for atom in mol.atoms] == [None, None]
    np.testing.assert_equal(mol.partial_charge, [0.2, -0.2])


def test_partial_source_subjects_survive_merge_without_invented_coordinates():
    left, right = _molecule(), _molecule()
    left.conformer_add(left.coordinates, 3.0)
    right.conformer_add(right.coordinates + 2, 4.0)
    left.add_component(right)
    assert len(left._source_frames) == 2
    assert [len(frame.atom_uids) for frame in left._source_frames] == [2, 2]
    assert set(left._source_frames[0].atom_uids).isdisjoint(left._source_frames[1].atom_uids)
    with pytest.raises(ValueError, match="frame_atom_mapping_incomplete"):
        left.conformer_get(0)
    with pytest.raises(ValueError, match="frame_atom_mapping_incomplete"):
        left.conformer_load(1)


def test_new_atom_does_not_gain_invented_source_coordinates():
    mol = _molecule()
    mol.conformer_add(mol.coordinates, 3.0)
    source_uids = mol._source_frames[0].atom_uids
    Atom(mol, atomic_number=1)
    assert mol._source_frames[0].atom_uids == source_uids
    with pytest.raises(ValueError, match="frame_atom_mapping_incomplete"):
        mol.conformer_load(0)


def test_copy_preserves_only_source_frames_and_source_selection():
    mol = _molecule()
    _import(mol, energy=[3.0, 4.0])
    mol.conformer_load(1)
    copied = mol.copy()
    assert len(copied._source_frames) == 2 and copied._conformer_results == ()
    assert copied.energy == 4
    assert copied._frame_result.source.owner is copied._owner_token
    assert copied._source_frames[0].coordinates is not mol._source_frames[0].coordinates
    _publish(mol)
    generated_selected_copy = mol.copy()
    assert len(generated_selected_copy._source_frames) == 2
    assert generated_selected_copy._conformer_results == ()
    assert generated_selected_copy.energy is None


def test_component_sources_subset_atom_fields_and_do_not_become_fragment_energies():
    mol = _molecule()
    mol.remove_bond(mol.bonds[0])
    _import(mol, energy=[3.0, 4.0], partial_charge=np.array([[0.1, -0.1], [0.2, -0.2]]))
    components = mol.components
    assert len(components) == 2
    for component in components:
        assert len(component._source_frames) == 2
        assert component.conformer_get(0)["energy"] is None
        assert component.conformer_get(0)["partial_charge"].shape == (1,)
        evidence = _thaw_value(component._source_frames[0].observations.provenance)
        assert dict(evidence["whole_system_observations"])["energy"]
        assert component.energy is None


def test_canonical_record_roundtrip_preserves_roles_ordinals_markers_and_shared_observations():
    mol = _molecule()
    _import(mol, energy=[0.0, 4.0])
    _publish(mol)
    mol.atoms[0].formal_charge = 1
    sources, ensembles, selection = _capture_frame_records(mol)
    assert sources[0].atom_uids == tuple(range(2))
    assert sources[0].observations is sources[1].observations
    clone = _molecule()
    view = clone.conformers
    _install_frame_records(clone, sources, ensembles, selection)
    assert clone.conformers is view and vars(view) == {"_owner": clone}
    assert len(clone._source_frames) == 2 and len(clone._conformer_results) == 1
    assert clone._conformer_results[0].payload.batch.ordinals == (4, 9)
    assert clone._conformer_results[0].invalidated_by == ("frame_representation",)
    assert clone._frame_result.invalidated_by == ("frame_representation",)
    assert clone._conformer_results[0].source.owner is clone._owner_token
    assert clone._frame_result.payload[0] is clone._conformer_results[0].payload.observations


def test_pure_preparation_validates_canonical_subjects_and_preserves_negative_selection():
    coordinates = np.arange(12.0).reshape(2, 2, 3)
    source, generated, selection = _frame_records_from_values(
        (0, 1), (coordinates, [0.0, 3.0], None, None, None, None, 1, None, None),
        -1, producer="fixture")
    assert selection[0] == -1
    _validate_frame_records(source, generated, selection, 2)
    invalid = replace(source[0], atom_uids=(0, 4))
    with pytest.raises(ValueError, match="canonical Atom rows"):
        _validate_frame_records((invalid,), generated, selection, 2)


def test_live_removal_projects_surviving_coordinates_and_atom_observations_without_fragment_energy():
    mol = _molecule()
    coordinates = _import(mol, energy=[3.0, 4.0],
                          partial_charge=np.array([[0.1, -0.1], [0.2, -0.2]]))
    raw_sources = mol._source_frames
    mol.remove_atom(0)
    observed = mol.conformer_get(1)
    np.testing.assert_equal(observed["coordinates"], coordinates[1, 1:])
    np.testing.assert_equal(observed["partial_charge"], [-0.2])
    assert observed["energy"] is None
    mol.conformer_load(1)
    assert mol.energy is None
    np.testing.assert_equal(mol.partial_charge, [-0.2])
    assert mol._source_frames is raw_sources
    evidence = _thaw_value(mol._frame_result.payload[0].provenance)
    assert dict(evidence["whole_system_observations"])["energy"]
    Atom(mol, atomic_number=7, coordinates=(3.0, 0.0, 0.0))
    before = mol.coordinates
    with pytest.raises(ValueError, match="frame_atom_mapping_incomplete"):
        mol.conformer_load(0)
    np.testing.assert_equal(mol.coordinates, before)


@pytest.mark.parametrize("mutation", ["remove", "replace"])
def test_record_capture_after_removed_or_replaced_subject_preserves_whole_evidence(mutation):
    mol = _molecule()
    _import(mol, energy=[3.0, 4.0])
    _publish(mol)
    old_subjects = tuple(atom._uid for atom in mol.atoms)
    mol.remove_atom(0)
    if mutation == "replace":
        Atom(mol, atomic_number=7, coordinates=(3.0, 0.0, 0.0))
    source, generated, selection = _capture_frame_records(mol)
    assert all(0 <= uid < len(mol.atoms) for frame in source for uid in frame.atom_uids)
    assert len(source[0].atom_uids) == 1
    payload, marker = generated[0]
    assert payload.atom_uids == (0,)
    assert payload.batch.coordinates.shape == (2, 1, 3)
    assert "frame_representation" in marker
    evidence = _thaw_value(payload.observations.provenance)
    assert dict(evidence["whole_system_observations"])["energy"]
    assert evidence["whole_system_coordinates"][0] == old_subjects
    _validate_frame_records(source, generated, selection, len(mol.atoms))
    restored = pickle.loads(pickle.dumps(mol))
    with pytest.raises((RuntimeError, ValueError), match="conformer_result_stale|frame_atom_mapping_incomplete"):
        restored.conformer_load(2)


@pytest.mark.parametrize("route", ["scalar", "fields", "prepared"])
def test_constraint_publication_marks_only_changed_boolean_constraints_and_rolls_back(route):
    from hotpot.cheminfo.structure._commit import _commit_atom_facts
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._mutation import _normalize_atom_fields

    mol = _molecule()
    _publish(mol)
    selected, ensembles = mol._frame_result, mol._conformer_results
    # False changes presence/producer metadata in an initially empty cell, while
    # retaining the exact constraint-domain value consumed by the calculation.
    mol.atoms[0].x_constraint = False
    assert mol._frame_result is selected and mol._conformer_results is ensembles
    cell = mol._constraints.atom_axes[mol.atoms[0]._uid]
    with pytest.raises(KeyboardInterrupt, match="abort constraint"):
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["explicit"]):
            if route == "scalar":
                mol.atoms[0].x_constraint = True
            elif route == "fields":
                mol.atoms[0].constraint = True
            else:
                _commit_atom_facts(mol, _normalize_atom_fields(
                    _capture_source(mol), (0,), (mol.atoms[0]._uid,), {"x_constraint": [1.0]},
                    producer=ATOM_SOURCE_CODES["explicit"]))
            assert mol._frame_result.invalidated_by == ("constraints",)
            assert mol._conformer_results[0].invalidated_by == ("constraints",)
            raise KeyboardInterrupt("abort constraint")
    assert mol._constraints.atom_axes[mol.atoms[0]._uid] is cell
    assert mol._frame_result is selected and mol._conformer_results is ensembles
    assert not mol.atoms[0].x_constraint
    assert mol.energy == 2.0


def test_constraint_raw_value_presence_and_producer_changes_preserve_boolean_applicability():
    from hotpot.cheminfo.structure._commit import _commit_atom_facts
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._mutation import _normalize_atom_fields

    mol = _molecule()
    mol.atoms[0].x_constraint = True
    _publish(mol)
    selected, ensembles = mol._frame_result, mol._conformer_results
    source = _capture_source(mol, ("constraints",))
    _commit_atom_facts(mol, _normalize_atom_fields(
        _capture_source(mol), (0,), (mol.atoms[0]._uid,), {"x_constraint": [-2.5]},
        producer=ATOM_SOURCE_CODES["rdkit"], present=False))
    _check_source(mol, source)
    assert mol.atoms[0].x_constraint
    assert mol._frame_result is selected and mol._conformer_results is ensembles
    assert mol.energy == 2.0


@pytest.mark.parametrize("outer", [False, True])
def test_constraint_native_result_publication_fault_restores_exact_cells_and_attachments(outer):
    from contextlib import nullcontext

    mol = _molecule()
    _publish(mol)
    selected, ensembles = mol._frame_result, mol._conformer_results
    cell = mol._constraints.atom_axes[mol.atoms[0]._uid]
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1) if outer else nullcontext():
            with _native_fail_after("result_invalidation"):
                mol.atoms[0].x_constraint = True
    assert mol._constraints.atom_axes[mol.atoms[0]._uid] is cell
    assert mol._frame_result is selected and mol._conformer_results is ensembles
    assert not mol.atoms[0].x_constraint
    assert mol.energy == 2.0


def test_public_constraint_column_denial_preserves_exact_cells_and_result_attachments():
    mol = _molecule()
    _publish(mol)
    selected, ensembles = mol._frame_result, mol._conformer_results
    cells = tuple(mol._constraints.atom_axes[atom._uid] for atom in mol.atoms)
    coordinates = mol.coordinates
    with pytest.raises(AttributeError):
        mol.updata_atoms(x_constraint=[True, False])
    assert all(mol._constraints.atom_axes[atom._uid] is cell for atom, cell in zip(mol.atoms, cells))
    assert mol._frame_result is selected and mol._conformer_results is ensembles
    np.testing.assert_equal(mol.coordinates, coordinates)
    assert not any(atom.x_constraint for atom in mol.atoms)
    assert mol.energy == 2.0


@pytest.mark.parametrize("raw_value,value", [(-2.5, True), (-0.0, False)])
def test_bond_constraint_boolean_noop_preserves_applicability(raw_value, value):
    mol = _molecule()
    bond = mol.bonds[0]
    bond.constraint = raw_value
    _publish(mol)
    selected, ensembles = mol._frame_result, mol._conformer_results
    source = _capture_source(mol, ("constraints",))

    bond.constraint = value

    _check_source(mol, source)
    assert bond.constraint is value
    assert bond._constraint_values[bond._uid] == np.float64(value).tobytes()
    assert mol._frame_result is selected and mol._conformer_results is ensembles
    assert mol.energy == 2.0
    mol.conformer_load(0)
    assert mol.energy == 1.0


@pytest.mark.parametrize("raw_value,value", [(-2.5, False), (-0.0, True)])
def test_bond_constraint_boolean_flip_invalidates_selected_and_generated_results(raw_value, value):
    mol = _molecule()
    bond = mol.bonds[0]
    bond.constraint = raw_value
    _publish(mol)
    source = _capture_source(mol, ("constraints",))

    bond.constraint = value

    assert bond.constraint is value
    with pytest.raises(RuntimeError, match="constraints"):
        _check_source(mol, source)
    assert mol._frame_result.invalidated_by == ("constraints",)
    assert mol._conformer_results[0].invalidated_by == ("constraints",)
    with pytest.raises(RuntimeError, match="frame_observation_stale"):
        _ = mol.energy
    with pytest.raises(RuntimeError, match="conformer_result_stale"):
        mol.conformer_load(0)


@pytest.mark.parametrize("raw_value,value,markers", [
    (-2.5, True, ()), (-0.0, False, ()),
    (-2.5, False, ("constraints",)), (-0.0, True, ("constraints",)),
])
def test_bond_constraint_outer_rollback_restores_exact_cell_and_attachments(raw_value, value, markers):
    mol = _molecule()
    bond = mol.bonds[0]
    bond.constraint = raw_value
    _publish(mol)
    selected, ensembles = mol._frame_result, mol._conformer_results
    cell = bond._constraint_values[bond._uid]
    source = _capture_source(mol, ("constraints",))

    with pytest.raises(KeyboardInterrupt, match="abort Bond constraint"):
        with _FactTransaction(mol, producer=1):
            bond.constraint = value
            assert bond._constraint_values[bond._uid] == np.float64(value).tobytes()
            assert mol._frame_result.invalidated_by == markers
            assert mol._conformer_results[0].invalidated_by == markers
            raise KeyboardInterrupt("abort Bond constraint")

    assert bond._constraint_values[bond._uid] is cell
    assert mol._frame_result is selected and mol._conformer_results is ensembles
    assert bond.constraint is bool(raw_value)
    _check_source(mol, source)
    assert mol.energy == 2.0
