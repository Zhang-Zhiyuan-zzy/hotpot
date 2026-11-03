"""Exact occurrence rows share the existing table and native atomic writer."""

from contextlib import nullcontext
from dataclasses import replace
import pickle

import numpy as np
import pytest

from hotpot import Atom, Molecule
from hotpot.cheminfo.stereo import (
    AtomTransportRecord, HydrogenTransportState, StereoPayload, set_stereo_payload,
)
from hotpot.cheminfo.structure._atom_table import _AtomFactTable, _seal_stereo_rows
from hotpot.cheminfo.structure._commit import _FactTransaction, _fact_engine, _native_fail_after
from hotpot.cheminfo.structure._fields import ATOM_FIELDS, FIELD_INDEX, TABLE_FIELDS
from hotpot.cheminfo.structure._source_identity import _SourceIdentity
from hotpot.cheminfo.structure._identity import _capture_source
from hotpot.cheminfo.structure._mutation import (
    _AtomFactUpdate, _normalize_atom_facts, _normalize_atom_fields, _normalize_atom_scalar,
)


_OCCURRENCES = (
    (2**130 + 1, -2**90, True, 2**80 + 3, -7, False, ""),
    (-2**140, None, False, None, None, None, None),
    (None, None, True, None, None, None, "source"),
    (None, None, False, None, None, None, None),
)


def _molecule():
    mol = Molecule()
    Atom(mol, atomic_number=6, coordinates=(0.0, 0.0, 0.0))
    Atom(mol, atomic_number=8, coordinates=(1.4, 0.0, 0.0))
    mol.add_bond(0, 1)
    return mol


def _candidate(private_values, rows=(0,)):
    return _AtomFactUpdate(_SourceIdentity(object(), ()), rows, rows,
                           (), (), (), (), private_values=private_values)


def _update(mol, private_values, *, rows=(0,), x=None):
    uids = tuple(mol.atoms[row]._uid for row in rows)
    update = _normalize_atom_fields(_capture_source(mol), rows, uids,
                                    {} if x is None else {"x": x}, producer=1)
    return replace(update, private_values=private_values)


def _numeric_state(table):
    return (table.matrix().tobytes(), table.presence().tobytes(), table.sources().tobytes(),
            table._group_status.tobytes(), tuple(frozenset(seed) for seed in table._dirty_seeds))


def test_private_candidate_seals_order_duplicates_types_and_unbounded_integers():
    cells = [list(cell) for cell in (*_OCCURRENCES, _OCCURRENCES[0])]
    supplied = [cells, []]
    update = _candidate(supplied, rows=(0, 1))
    cells[0][0] = 0
    cells.append(list(_OCCURRENCES[1]))
    supplied[1].append(list(_OCCURRENCES[0]))

    assert update.private_values == ((*_OCCURRENCES, _OCCURRENCES[0]), ())
    assert type(update.private_values[0][0][0]) is int
    assert type(update.private_values[0][0][2]) is bool
    assert type(update.private_values[0][0][5]) is bool
    assert update.private_values[0][0][6] == ""
    assert update.private_values[0][1][6] is None
    assert update.private_values[0][2][2] and not update.private_values[0][3][2]
    with pytest.raises(TypeError):
        update.private_values[0][0][0] = 1
    assert pickle.loads(pickle.dumps(update)).private_values == update.private_values


@pytest.mark.parametrize("cell,error", [
    ((None,) * 6, ValueError),
    ((True, None, False, None, None, None, None), TypeError),
    ((np.int64(13), None, False, None, None, None, None), TypeError),
    ((None, 1.0, False, None, None, None, None), TypeError),
    ((None, None, 1, None, None, None, None), TypeError),
    ((None, None, True, False, None, None, None), TypeError),
    ((None, None, True, None, None, 0, None), TypeError),
    ((None, None, False, 0, None, None, None), ValueError),
    ((None, None, False, None, None, None, 7), TypeError),
])
def test_private_candidate_rejects_malformed_closed_occurrences(cell, error):
    with pytest.raises(error):
        _candidate(((cell,),))


def test_private_candidate_distinguishes_no_update_from_empty_replacement_and_checks_alignment():
    assert _candidate(()).private_values == ()
    assert _candidate(((),)).private_values == ((),)
    with pytest.raises(ValueError, match="align"):
        _candidate(((), ()))


def test_table_private_rows_follow_growth_remap_take_copy_and_pickle():
    table = _AtomFactTable(np.zeros((3, len(TABLE_FIELDS))))
    table._stereo_facts[:] = list(_seal_stereo_rows((_OCCURRENCES, (), (_OCCURRENCES[1],))))
    original = table._stereo_facts[0]
    table.append(np.zeros((2, len(TABLE_FIELDS))))
    assert len(table._stereo_facts) == len(table._blocks[0])
    assert table._stereo_facts[0] is original
    assert table._stereo_facts[3:5] == [(), ()]
    table.remap((2, 0))
    assert table._stereo_facts[:2] == [(_OCCURRENCES[1],), _OCCURRENCES]
    assert table._stereo_facts[1] is original
    assert table._stereo_facts[2:] == [()] * (len(table._stereo_facts) - 2)
    selected = table.take((1, 0))
    clone = table.copy()
    assert selected._stereo_facts == [_OCCURRENCES, (_OCCURRENCES[1],)]
    assert clone._stereo_facts == table._stereo_facts[:2]
    assert clone._stereo_facts is not table._stereo_facts
    assert clone._stereo_facts[1] is original
    restored = pickle.loads(pickle.dumps(table))
    assert len(restored._stereo_facts) == len(restored._blocks[0])
    assert restored._stereo_facts == table._stereo_facts
    restored._stereo_facts[0] = ()
    assert table._stereo_facts[0] == (_OCCURRENCES[1],)


def test_detached_private_occurrences_remain_immutable_without_pointer_byte_sealing():
    mol = _molecule()
    atom = mol.atoms[0]
    facts = tuple(AtomTransportRecord(
        0, isotope, radical,
        HydrogenTransportState(implicit, bracket, no_implicit) if present else None,
        None, namespace)
        for isotope, radical, present, implicit, bracket, no_implicit, namespace in _OCCURRENCES)
    set_stereo_payload(mol, StereoPayload(1, (), facts, ()))
    table = mol._atom_table
    mol.remove_atom(atom)
    detached = atom._detached_table
    assert type(detached._stereo_facts) is list
    assert detached._stereo_facts[0] == _OCCURRENCES
    assert detached._stereo_facts is not table._stereo_facts
    with pytest.raises(TypeError):
        detached._stereo_facts[0][0][0] = 0
    with pytest.raises(ValueError):
        detached._blocks[0].flags.writeable = True


def test_all_native_numeric_candidate_constructors_initialize_empty_private_values():
    mol = _molecule()
    source = _capture_source(mol)
    uid = mol.atoms[0]._uid
    updates = (
        _normalize_atom_scalar(source, 0, uid, "x", 1.5),
        _normalize_atom_fields(source, (0,), (uid,), {"x": [1.5]}),
        _normalize_atom_facts(source, (0,), (uid,), {"x": [1.5]}),
    )
    assert all(update.private_values == () for update in updates)
    assert len(ATOM_FIELDS) == 19 and len(TABLE_FIELDS) == 16
    assert tuple(FIELD_INDEX[name] for name in ATOM_FIELDS[16:]) == (16, 17, 18)


def test_native_private_only_commit_preserves_numeric_slots_and_invalidates_representation():
    mol = _molecule()
    mol.energy = 5.0
    table = mol._atom_table
    before = _numeric_state(table)
    update = _update(mol, (_OCCURRENCES,))
    _fact_engine()._commit_atom_facts(mol, update)
    assert table._stereo_facts[0] is update.private_values[0]
    assert table._stereo_facts[1] == ()
    assert _numeric_state(table) == before
    assert mol._frame_result.invalidated_by == ("frame_representation",)
    with pytest.raises(RuntimeError, match="frame_observation_stale"):
        _ = mol.energy


def test_native_equal_private_rows_preserve_exact_selected_result_attachment():
    mol = _molecule()
    _fact_engine()._commit_atom_facts(mol, _update(mol, (_OCCURRENCES,)))
    mol.energy = 5.0
    selected = mol._frame_result
    _fact_engine()._commit_atom_facts(mol, _update(mol, (_OCCURRENCES,)))
    assert mol._frame_result is selected and mol.energy == 5.0


def test_native_mixed_commit_publishes_aligned_numeric_and_private_rows():
    mol = _molecule()
    update = _update(mol, ((_OCCURRENCES[1],), _OCCURRENCES), rows=(1, 0), x=(8.0, 9.0))
    _fact_engine()._commit_atom_facts(mol, update)
    assert mol.atoms[0].x == 9.0 and mol.atoms[1].x == 8.0
    assert mol._atom_table._stereo_facts[1] is update.private_values[0]
    assert mol._atom_table._stereo_facts[0] is update.private_values[1]
    _fact_engine()._commit_atom_facts(mol, _update(mol, ((),)))
    assert mol._atom_table._stereo_facts[0] == ()


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("boundary,occurrence", [
    ("value", 2), ("source", 2), ("presence", 2), ("result_invalidation", 1),
])
def test_native_mixed_failure_restores_exact_private_cells_numeric_state_and_results(outer, boundary, occurrence):
    mol = _molecule()
    _fact_engine()._commit_atom_facts(mol, _update(mol, ((_OCCURRENCES[1],),)))
    mol.energy = 5.0
    table = mol._atom_table
    owner, cell = table._stereo_facts, table._stereo_facts[0]
    selected, ensembles = mol._frame_result, mol._conformer_results
    before = _numeric_state(table)
    update = _update(mol, (_OCCURRENCES,), x=(9.0,))
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1) if outer else nullcontext():
            with _native_fail_after(boundary, occurrence=occurrence):
                _fact_engine()._commit_atom_facts(mol, update)
    assert table._stereo_facts is owner and owner[0] is cell
    assert _numeric_state(table) == before
    assert mol._frame_result is selected and mol._conformer_results is ensembles
    assert mol.energy == 5.0


def test_native_private_enrollment_failure_leaves_exact_owners_untouched():
    mol = _molecule()
    mol.energy = 5.0
    owner = mol._atom_table._stereo_facts
    selected = mol._frame_result
    update = _update(mol, (_OCCURRENCES,))
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1):
            with _native_fail_after("enrollment"):
                _fact_engine()._commit_atom_facts(mol, update)
    assert mol._atom_table._stereo_facts is owner and owner == [(), ()]
    assert mol._frame_result is selected and mol.energy == 5.0


def test_outer_rollback_retains_original_private_list_after_table_remap():
    mol = _molecule()
    _fact_engine()._commit_atom_facts(mol, _update(mol, ((_OCCURRENCES[1],),)))
    mol.energy = 5.0
    table = mol._atom_table
    owner, cell = table._stereo_facts, table._stereo_facts[0]
    selected = mol._frame_result
    before = _numeric_state(table)
    update = _update(mol, (_OCCURRENCES,), x=(9.0,))
    with pytest.raises(RuntimeError, match="abort mixed publication"):
        with _FactTransaction(mol, producer=1) as transaction:
            transaction.touch_topology()
            _fact_engine()._commit_atom_facts(mol, update)
            table.remap((1, 0))
            assert table._stereo_facts is not owner
            raise RuntimeError("abort mixed publication")
    assert table._stereo_facts is owner and owner[0] is cell
    assert _numeric_state(table) == before
    assert mol._frame_result is selected and mol.energy == 5.0
