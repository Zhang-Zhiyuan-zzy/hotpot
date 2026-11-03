"""Canonical source/result lifecycle through carrier transactions and transport."""

from dataclasses import fields
import pickle

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.conformers.contracts import ConformerBatch
from hotpot.cheminfo.structure._builder import _restore_molecule_record
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._identity import _capture_source
from hotpot.cheminfo.structure.conformer import (
    CoordinateFrame, _import_frame_values, _install_frame_records, _publish_conformer_ensemble,
)


def _molecule(numbers=(6, 8, 7)):
    mol = Molecule()
    for row, number in enumerate(numbers):
        mol.create_atom(atomic_number=number, coordinates=(row * 1.4, 0., 0.))
    return mol


def _sources(mol):
    coordinates = np.stack((mol.coordinates, mol.coordinates + 1.))
    n = len(mol.atoms)
    values = (coordinates, np.array([0., -3.]),
              np.arange(2 * n, dtype=float).reshape(2, n), None,
              np.ones((2, n, 3)), None, np.array([1, 3]), None, None)
    _import_frame_values(mol, values, 0, producer=1, method='supplied-test')
    return coordinates


def _generated(mol):
    batch = ConformerBatch(np.stack((mol.coordinates + 2., mol.coordinates + 3.)), (2, 5))
    _publish_conformer_ensemble(
        mol, batch, (np.array([-4., -5.]), None, None, None, None, None, None, None),
        1, input_source=_capture_source(mol), method='lifecycle-test',
        parameters=(('seed', 11),), units=(('energy', 'kJ/mol'),))


def test_observation_descriptors_select_values_without_orphan_namespace_slots():
    mol = _molecule()
    observations = dict(energy=0., partial_charge=np.array([0., 1., -1.]),
                        gibbs=1., force=np.ones((3, 3)), zero_point=0.,
                        thermo={'temperature': 298.15}, capacity=np.array([1., 2.]))
    for name, value in observations.items():
        setattr(mol, name, value)
    for name, value in observations.items():
        assert name not in mol.__dict__
        observed = getattr(mol, name)
        if isinstance(value, dict):
            assert observed == value
        else:
            np.testing.assert_array_equal(observed, value)
    assert mol.conformers_number == 0
    assert all(atom.partial_charge is None for atom in mol.atoms)


def test_frame_owner_journal_restores_exact_attachments_sources_cursor_and_coordinates():
    mol = _molecule()
    _sources(mol)
    _generated(mol)
    before = (mol._source_frames, mol._conformer_results, mol._frame_result,
              mol._conformers_index, mol._conformers)
    coordinates = mol.coordinates.copy()
    with pytest.raises(KeyboardInterrupt, match='rollback'):
        with _FactTransaction(mol, producer=1):
            mol.conformer_load(0)
            mol.conformer_clear()
            mol.conformer_add(coordinates + 8., 99.)
            raise KeyboardInterrupt('rollback')
    assert mol._source_frames is before[0]
    assert mol._conformer_results is before[1]
    assert mol._frame_result is before[2]
    assert mol._conformers_index == before[3]
    assert mol._conformers is before[4]
    np.testing.assert_array_equal(mol.coordinates, coordinates)
    assert mol.energy == -5.


def test_whole_copy_preserves_source_values_and_drops_generated_observations():
    mol = _molecule()
    coordinates = _sources(mol)
    _generated(mol)
    clone = mol.copy()
    assert clone.conformers_number == 2
    assert len(clone._source_frames) == 2
    assert clone._conformer_results == ()
    assert clone._frame_result is None
    np.testing.assert_array_equal(clone.coordinates, mol.coordinates)
    np.testing.assert_array_equal(clone.conformer_get(1)['coordinates'], coordinates[1])
    clone.conformer_load(0)
    assert clone.energy == 0.
    clone.conformer_clear()
    assert mol.conformers_number == 4


def test_selected_source_copy_rebinds_only_matching_geometry_and_context():
    mol = _molecule()
    _sources(mol)
    mol.conformer_load(0)
    clone = mol.copy()
    assert clone.energy == 0.
    mol.atoms[0].x += 0.5
    changed = mol.copy()
    assert changed._frame_result is None
    assert changed.conformers_number == 2


def test_components_subset_source_subjects_and_keep_whole_evidence_out_of_fragment_energy():
    mol = _molecule()
    mol.add_bond(0, 1)
    coordinates = _sources(mol)
    fragments = mol.components
    assert sorted(len(fragment.atoms) for fragment in fragments) == [1, 2]
    for fragment in fragments:
        rows = (0, 1) if len(fragment.atoms) == 2 else (2,)
        assert fragment.conformers_number == 2
        for index in range(2):
            projected = fragment.conformer_get(index)
            np.testing.assert_array_equal(projected['coordinates'], coordinates[index, rows, :])
            assert projected['energy'] is None
            np.testing.assert_array_equal(projected['partial_charge'],
                                          np.arange(6.).reshape(2, 3)[index, rows])
            assert fragment._source_frames[index].provenance
        fragment.conformer_load(0)
        assert fragment.energy is None


def test_nonempty_merge_retains_partial_subject_sources_without_invented_coordinates():
    first, second = _molecule((6,)), _molecule((8, 7))
    _sources(first)
    _sources(second)
    first.add_component(second)
    assert first.conformers_number == 4
    assert [len(frame.atom_uids) for frame in first._source_frames] == [1, 1, 2, 2]
    assert first._frame_result is None
    for index in range(4):
        before = first.coordinates.copy()
        with pytest.raises((ValueError, RuntimeError), match='(?i)(membership|complete|mapping|subject)'):
            first.conformer_load(index)
        np.testing.assert_array_equal(first.coordinates, before)


def test_empty_merge_can_rebind_complete_incoming_sources():
    source = _molecule()
    coordinates = _sources(source)
    target = Molecule()
    target.add_component(source)
    assert target.conformers_number == 2
    target.conformer_load(1)
    np.testing.assert_array_equal(target.coordinates, coordinates[1])
    assert target.energy == -3.
    target.conformer_clear()
    assert source.conformers_number == 2


def test_added_atom_leaves_source_incomplete_and_bond_edit_retains_source_coordinates():
    mol = _molecule((6, 8))
    coordinates = _sources(mol)
    mol.add_bond(0, 1)
    np.testing.assert_array_equal(mol.conformer_get(0)['coordinates'], coordinates[0])
    mol.create_atom(atomic_number=1, coordinates=(4., 0., 0.))
    assert len(mol._source_frames[0].atom_uids) == 2
    with pytest.raises((ValueError, RuntimeError), match='(?i)(membership|complete|mapping|subject)'):
        mol.conformer_load(0)
    np.testing.assert_array_equal(mol.coordinates[-1], [4., 0., 0.])


def test_pickle_rebinds_canonical_sources_results_and_selected_observations_to_real_uids():
    mol = _molecule((1, 6, 8))
    mol.remove_atom(0)
    _sources(mol)
    _generated(mol)
    record, _ = mol._capture_factual_record()
    assert len(fields(record)) == 25
    assert record.__reduce__()[1][1] == 4
    with pytest.raises(ValueError, match='version 4'):
        _restore_molecule_record(type(record), 3, record.__reduce__()[1][2])
    restored = pickle.loads(pickle.dumps(mol))
    uids = tuple(atom._uid for atom in restored.atoms)
    assert uids == (1, 2)
    assert restored._next_atom_uid == mol._next_atom_uid
    assert [frame.atom_uids for frame in restored._source_frames] == [uids, uids]
    assert restored._conformer_results[0].payload.atom_uids == uids
    assert restored._conformer_results[0].source.owner is restored._owner_token
    assert restored._frame_result.source.owner is restored._owner_token
    assert restored.energy == -5.
    restored.conformer_load(0)
    assert restored.energy == 0.


@pytest.mark.parametrize('existing_cell', [False, True])
def test_cell_change_outer_rollback_restores_exact_cell_and_frame_currentness(existing_cell):
    mol = _molecule()
    if existing_cell:
        mol.create_crystal(10., 11., 12., 90., 90., 90.)
    _sources(mol)
    _generated(mol)
    before = (mol.crystal, mol._source_frames, mol._conformer_results,
              mol._frame_result, mol._conformers, mol._conformers_index)
    coordinates = mol.coordinates.copy()
    with pytest.raises(KeyboardInterrupt, match='cell rollback'):
        with _FactTransaction(mol, producer=1):
            changed = mol.create_crystal(15., 16., 17., 90., 90., 90.)
            assert mol.crystal is changed and changed is not before[0]
            assert mol._conformer_results[0].payload is before[2][0].payload
            assert 'cell' in mol._conformer_results[0].invalidated_by
            assert 'cell' in mol._frame_result.invalidated_by
            with pytest.raises(RuntimeError, match='frame_observation_stale.*cell'):
                _ = mol.energy
            raise KeyboardInterrupt('cell rollback')
    assert mol.crystal is before[0]
    assert mol._source_frames is before[1]
    assert mol._conformer_results is before[2]
    assert mol._frame_result is before[3]
    assert mol._conformers is before[4]
    assert mol._conformers_index == before[5]
    np.testing.assert_array_equal(mol.coordinates, coordinates)
    assert mol.energy == -5.
    mol.conformer_load(before[5])
    assert mol.energy == -5.


def test_identical_cell_publication_retains_exact_owners_and_current_generated_selection():
    mol = _molecule()
    cell = mol.create_crystal(10., 11., 12., 90., 90., 90.)
    _sources(mol)
    _generated(mol)
    before = (mol._source_frames, mol._conformer_results,
              mol._frame_result, mol._conformers, mol._conformers_index)
    coordinates = mol.coordinates.copy()
    assert mol.create_crystal(10., 11., 12., 90., 90., 90.) is cell
    assert mol.crystal is cell
    assert mol._source_frames is before[0]
    assert mol._conformer_results is before[1]
    assert mol._frame_result is before[2]
    assert mol._conformers is before[3]
    assert mol._conformers_index == before[4]
    assert mol._conformer_results[0].invalidated_by == ()
    assert mol._frame_result.invalidated_by == ()
    np.testing.assert_array_equal(mol.coordinates, coordinates)
    assert mol.energy == -5.
    mol.conformer_load(before[4])
    assert mol.energy == -5.


@pytest.mark.parametrize('cursor', [0, -1])
@pytest.mark.parametrize('with_generated', [False, True])
@pytest.mark.parametrize('energy', [3., None])
def test_source_append_preserves_selected_member_and_whole_copy(cursor, with_generated, energy):
    mol = _molecule()
    coordinates = mol.coordinates.copy()
    mol.conformer_add(coordinates, energy)
    if with_generated:
        _generated(mol)
    cursor = cursor - 2 if with_generated and cursor < 0 else cursor
    mol.conformer_load(cursor)
    before = (mol._source_frames, mol._conformer_results, mol._frame_result, mol._conformers)
    assert mol._capture_factual_record()[0].frame_selection[0] == cursor
    unchanged = mol.copy()
    assert unchanged._conformers_index == (-1 if cursor < 0 else 0)
    assert unchanged.energy == energy

    mol.conformer_add(np.stack((coordinates + 4., coordinates + 5.)), [4., 5.])

    expected_cursor = cursor - 2 if cursor < 0 else cursor
    assert mol._conformers_index == expected_cursor
    assert mol._source_frames[0] is before[0][0]
    assert mol._conformer_results is before[1]
    assert mol._frame_result is before[2]
    assert mol._conformers is before[3]
    assert mol._capture_factual_record()[0].frame_selection[0] == expected_cursor
    np.testing.assert_array_equal(mol.coordinates, coordinates)
    np.testing.assert_array_equal(mol.conformer_get(expected_cursor)['coordinates'], coordinates)
    assert mol.energy == energy
    assert mol.conformer_get(expected_cursor)['energy'] == energy

    clone = mol.copy()
    assert clone._conformers_index == (-3 if cursor < 0 else 0)
    assert clone._conformer_results == ()
    assert clone._frame_result.payload[0] is clone._source_frames[0].observations
    assert clone.energy == energy
    np.testing.assert_array_equal(clone.coordinates, coordinates)
    np.testing.assert_array_equal(clone.conformer_get(clone._conformers_index)['coordinates'], coordinates)


@pytest.mark.parametrize('cursor', [2, 3, -2, -1])
@pytest.mark.parametrize('energies', [(-4., -5.), None])
def test_source_append_preserves_selected_generated_member(cursor, energies):
    mol = _molecule()
    _sources(mol)
    batch = ConformerBatch(np.stack((mol.coordinates + 2., mol.coordinates + 3.)), (2, 5))
    _publish_conformer_ensemble(
        mol, batch, {'energy': energies}, 1,
        input_source=_capture_source(mol), method='append-test')
    mol.conformer_load(cursor)
    before = (mol._source_frames, mol._conformer_results, mol._frame_result, mol._conformers)
    coordinates = mol.coordinates.copy()
    energy = mol.energy
    assert mol._capture_factual_record()[0].frame_selection[0] == cursor

    mol.conformer_add(np.stack((coordinates + 4., coordinates + 5.)), [4., 5.])

    expected_cursor = cursor + 2 if cursor >= 0 else cursor
    assert mol._conformers_index == expected_cursor
    assert all(current is previous for current, previous in zip(mol._source_frames, before[0]))
    assert mol._conformer_results is before[1]
    assert mol._frame_result is before[2]
    assert mol._conformers is before[3]
    assert mol._conformer_results[0].invalidated_by == ()
    assert mol._frame_result.invalidated_by == ()
    assert mol._capture_factual_record()[0].frame_selection[0] == expected_cursor
    np.testing.assert_array_equal(mol.coordinates, coordinates)
    np.testing.assert_array_equal(mol.conformer_get(expected_cursor)['coordinates'], coordinates)
    assert mol.conformer_get(expected_cursor)['energy'] == energy
    assert mol.energy == energy
    clone = mol.copy()
    assert clone._frame_result is None
    assert clone.energy is None
    np.testing.assert_array_equal(clone.coordinates, coordinates)


@pytest.mark.parametrize('cursor', [1, -1])
def test_source_append_and_copy_preserve_cursor_without_observations(cursor):
    mol = _molecule()
    coordinates = mol.coordinates.copy()
    uids = tuple(atom._uid for atom in mol.atoms)
    sources = tuple(CoordinateFrame(uids, coordinates + offset, None, None, ('tuple', ()))
                    for offset in (0., 1.))
    _install_frame_records(mol, sources, (), (0, None))
    mol.conformer_load(cursor)
    coordinates = mol.coordinates.copy()
    before = (mol._source_frames, mol._conformer_results, mol._conformers)

    mol.conformer_add(coordinates + 4., 4.)

    expected_cursor = cursor - 1 if cursor < 0 else cursor
    assert mol._conformers_index == expected_cursor
    assert mol._frame_result is None
    assert mol.energy is None
    assert all(current is previous for current, previous in zip(mol._source_frames, before[0]))
    assert mol._conformer_results is before[1]
    assert mol._conformers is before[2]
    np.testing.assert_array_equal(mol.coordinates, coordinates)
    clone = mol.copy()
    assert clone._conformers_index == expected_cursor
    assert clone._frame_result is None
    assert clone.energy is None
    np.testing.assert_array_equal(clone.conformer_get(expected_cursor)['coordinates'], coordinates)


@pytest.mark.parametrize('generated', [False, True])
@pytest.mark.parametrize('negative', [False, True])
def test_source_append_outer_rollback_restores_exact_owners_and_cursor(generated, negative):
    mol = _molecule()
    _sources(mol)
    _generated(mol)
    cursor = 3 if generated else 1
    cursor = cursor - mol.conformers_number if negative else cursor
    mol.conformer_load(cursor)
    before = (mol._source_frames, mol._conformer_results, mol._frame_result,
              mol._conformers_index, mol._conformers)
    coordinates = mol.coordinates.copy()
    energy = mol.energy

    with pytest.raises(KeyboardInterrupt, match='append rollback'):
        with _FactTransaction(mol, producer=1):
            mol.conformer_add(np.stack((coordinates + 4., coordinates + 5.)), [4., 5.])
            expected_cursor = cursor + (2 if generated and not negative else
                                        -2 if not generated and negative else 0)
            assert mol._conformers_index == expected_cursor
            assert mol._conformer_results is before[1]
            assert mol._frame_result is before[2]
            assert mol._conformers is before[4]
            np.testing.assert_array_equal(mol.coordinates, coordinates)
            raise KeyboardInterrupt('append rollback')

    assert mol._source_frames is before[0]
    assert mol._conformer_results is before[1]
    assert mol._frame_result is before[2]
    assert mol._conformers_index == before[3]
    assert mol._conformers is before[4]
    np.testing.assert_array_equal(mol.coordinates, coordinates)
    np.testing.assert_array_equal(mol.conformer_get(cursor)['coordinates'], coordinates)
    assert mol.energy == energy
