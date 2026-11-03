"""Finite status propagation, selective reads and UID remap contracts."""

import pickle

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.structure._atom_table import _AtomFactTable
from hotpot.cheminfo.structure._commit import _FactTransaction, _native_fail_after
from hotpot.cheminfo.structure._dependencies import (
    GROUP_INDEX, GROUPS, _FactStatus, _dependency_state, _ensure_group,
    _install_dependency_state, _mark_topology_dirty, _remap_dependency_state,
    _seed,
)


def _molecule():
    mol = Molecule()
    for _ in range(5):
        mol.create_atom(atomic_number=6, formal_charge=0, implicit_hydrogens=0)
    for first, second in ((0, 1), (1, 2), (3, 4)):
        mol.add_bond(first, second)
    mol._atom_table._group_status[:5] = _FactStatus.CURRENT
    for seeds in mol._atom_table._dirty_seeds:
        seeds.clear()
    return mol


def test_remote_first_read_expands_upstream_seeds_before_current_byte():
    mol = _molecule()
    _seed(mol, GROUP_INDEX['VALENCE'], (mol.atoms[0]._uid,))
    assert mol._atom_table._group_status[2, GROUP_INDEX['AROMATIC_REPRESENTATION']] == _FactStatus.CURRENT
    observed = _ensure_group(mol, 'AROMATIC_REPRESENTATION', rows=(2, 4))
    assert observed.tolist() == [_FactStatus.NEEDS_RECALC, _FactStatus.CURRENT]
    assert all(not seeds for seeds in mol._atom_table._dirty_seeds[:3])
    assert _ensure_group(mol, 'AROMATIC_REPRESENTATION', rows=(2, 4)).tolist() == observed.tolist()


def test_unrelated_group_read_does_not_consume_or_expand_representation_seed():
    mol = _molecule()
    _seed(mol, GROUP_INDEX['VALENCE'], (mol.atoms[0]._uid,))
    before = _dependency_state(mol._atom_table)
    assert _ensure_group(mol, 'FORMAL_CHARGE_COVERAGE').tolist() == [0] * 5
    after = _dependency_state(mol._atom_table)
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1] == after[1]


def test_component_invalidation_preserves_absent_unresolved_and_conflict():
    mol = _molecule()
    index = GROUP_INDEX['AROMATIC_REPRESENTATION']
    mol._atom_table._group_status[:3, index] = (_FactStatus.CONFLICT, _FactStatus.ABSENT, _FactStatus.UNRESOLVED)
    _seed(mol, index, (mol.atoms[0]._uid,))
    assert _ensure_group(mol, 'AROMATIC_REPRESENTATION').tolist() == [4, 2, 3, 0, 0]


def test_breaking_edge_seeds_both_resulting_components():
    mol = _molecule()
    first, second = mol.atoms[0], mol.atoms[1]
    with _FactTransaction(mol, producer=1):
        _mark_topology_dirty(mol, (first._uid, second._uid))
        mol.remove_bond(mol.bonds[0])
    states = _ensure_group(mol, 'AROMATIC_REPRESENTATION')
    assert states.tolist() == [1, 1, 1, 0, 0]


def test_copy_and_subset_remap_keep_dirty_conflicting_state_and_rename_seeds():
    mol = _molecule()
    table = mol._atom_table
    table._group_status[2, 2] = _FactStatus.CONFLICT
    _seed(mol, 0, (mol.atoms[0]._uid, mol.atoms[2]._uid, mol.atoms[4]._uid))
    rows = (2, 0, 3)
    mapped = _remap_dependency_state(table, tuple(atom._uid for atom in mol.atoms), (20, 21, 22), rows=rows)
    destination = table.take(rows)
    _install_dependency_state(destination, mapped)
    assert destination._group_status[0, 2] == _FactStatus.CONFLICT
    assert destination._dirty_seeds[0] == {20, 21}
    destination._dirty_seeds[0].clear()
    assert len(table._dirty_seeds[0]) == 3


def test_pending_status_expansion_rolls_back_on_baseexception():
    mol = _molecule()
    _seed(mol, 0, (mol.atoms[0]._uid,))
    before = _dependency_state(mol._atom_table)
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1):
            _ensure_group(mol, 'AROMATIC_REPRESENTATION')
            raise KeyboardInterrupt('after expansion')
    after = _dependency_state(mol._atom_table)
    np.testing.assert_array_equal(before[0], after[0])
    assert before[1] == after[1]


def test_table_state_is_owned_and_serializes_only_logical_rows():
    table = _AtomFactTable(np.ones((4, 16)))
    table.remap((0,))
    table._dirty_seeds[2].add(19)
    captured = table.__getstate__()
    assert captured[5].shape == (1, len(GROUPS))
    table._blocks[0][0, 0] = 900
    table._group_status[0, 2] = _FactStatus.CONFLICT
    table._dirty_seeds[2].clear()
    restored = object.__new__(_AtomFactTable)
    restored.__setstate__(captured)
    assert restored.raw(0, 'atomic_number') == 1
    assert restored._group_status[0, 2] == _FactStatus.CURRENT
    assert restored._dirty_seeds[2] == {19}
    roundtrip = pickle.loads(pickle.dumps(restored))
    assert roundtrip._dirty_seeds[2] == {19}


@pytest.mark.parametrize('status,seeds', [
    (np.zeros((2, 4)), ((),) * 4),
    (np.zeros((1, 3), dtype=np.uint8), ((),) * 4),
    (np.full((1, 4), 5, dtype=np.uint8), ((),) * 4),
    (np.zeros((1, 4), dtype=np.uint8), ((),)),
])
def test_invalid_dependency_state_is_rejected_before_publication(status, seeds):
    table = _AtomFactTable(np.ones((1, 16)))
    with pytest.raises(ValueError):
        _install_dependency_state(table, (status, seeds))
    assert table._group_status.tolist() == [[2, 0, 0, 0]]


@pytest.mark.parametrize('writer', ('scalar', 'batch'))
def test_neighbour_dependency_scope_expands_before_remote_h_read_without_reaching_two_hops(writer):
    mol = _molecule()
    if writer == 'scalar':
        mol.atoms[0].atomic_number = 7
    else:
        mol.updata_atoms(atoms=(mol.atoms[0],), atomic_number=(7,))
    assert _ensure_group(mol, 'IMPLICIT_H', rows=(1, 2, 4)).tolist() == [
        _FactStatus.NEEDS_RECALC, _FactStatus.CURRENT, _FactStatus.CURRENT]
    assert mol._atom_table._group_status[:5, GROUP_INDEX['VALENCE']].tolist() == [1, 1, 0, 0, 0]


@pytest.mark.parametrize('rows,strict_error', [
    ((), False), ((0,), False), ((2, 1), True), ((2, 0, 2, 1), True),
])
def test_aromatic_terminal_selection_does_not_prepare_other_components(rows, strict_error):
    mol = _molecule()
    table = mol._atom_table
    group = GROUP_INDEX['AROMATIC_REPRESENTATION']
    table._group_status[:5, group] = (
        _FactStatus.CURRENT, _FactStatus.UNRESOLVED, _FactStatus.CONFLICT,
        _FactStatus.NEEDS_RECALC, _FactStatus.ABSENT,
    )
    before = _dependency_state(table)
    chemistry._prepare_aromatic_representation(mol, rows=rows)
    if strict_error:
        with pytest.raises(RuntimeError, match='Aromatic representation is unresolved or conflicting'):
            chemistry._require_aromatic_representation(mol, rows=rows)
    else:
        assert chemistry._require_aromatic_representation(mol, rows=rows) is None
    after = _dependency_state(table)
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1] == before[1]
    assert table._group_status.dtype == np.uint8


@pytest.mark.parametrize('rows', [(3,), (4,), (4, 3, 4)])
def test_aromatic_required_pending_rows_prepare_their_component_and_keep_terminal_findings(rows):
    mol = _molecule()
    table = mol._atom_table
    group = GROUP_INDEX['AROMATIC_REPRESENTATION']
    table._group_status[:5, group] = (
        _FactStatus.CURRENT, _FactStatus.UNRESOLVED, _FactStatus.CONFLICT,
        _FactStatus.NEEDS_RECALC, _FactStatus.ABSENT,
    )
    facts = table.matrix().tobytes(), table.presence().tobytes(), table.sources().tobytes()
    assert chemistry._require_aromatic_representation(mol, rows=rows) is None
    assert table._group_status[:5, group].tolist() == [0, 3, 4, 0, 0]
    assert (table.matrix().tobytes(), table.presence().tobytes(), table.sources().tobytes()) == facts


@pytest.mark.parametrize('rows', [(), (2,), (4, 2, 4)])
def test_aromatic_preparation_consumes_upstream_seeds_even_for_terminal_selection(rows):
    mol = _molecule()
    table = mol._atom_table
    group = GROUP_INDEX['AROMATIC_REPRESENTATION']
    table._group_status[3:5, group] = (_FactStatus.CONFLICT, _FactStatus.UNRESOLVED)
    _seed(mol, GROUP_INDEX['VALENCE'], (mol.atoms[0]._uid,))
    untouched = table._group_status[3:5].copy()
    assert chemistry._require_aromatic_representation(mol, rows=()) is None
    assert all(table._dirty_seeds[index] for index in range(3))
    chemistry._prepare_aromatic_representation(mol, rows=rows)
    assert all(not seeds for seeds in table._dirty_seeds[:3])
    assert table._group_status[:3, group].tolist() == [0, 0, 0]
    np.testing.assert_array_equal(table._group_status[3:5], untouched)


@pytest.mark.parametrize('boundary', ['status', 'seed_clear'])
@pytest.mark.parametrize('exception', [KeyboardInterrupt, MemoryError])
def test_aromatic_preparation_restores_dirty_expansion_on_native_interruption(boundary, exception):
    mol = _molecule()
    table = mol._atom_table
    _seed(mol, GROUP_INDEX['VALENCE'], (mol.atoms[0]._uid,))
    before = _dependency_state(table)
    owners = tuple(table._dirty_seeds)
    facts = table.matrix().tobytes(), table.presence().tobytes(), table.sources().tobytes()
    with pytest.raises(exception):
        with _native_fail_after(boundary, exception=exception):
            chemistry._prepare_aromatic_representation(mol, rows=(2,))
    after = _dependency_state(table)
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1] == before[1]
    assert all(left is right for left, right in zip(owners, table._dirty_seeds))
    assert (table.matrix().tobytes(), table.presence().tobytes(), table.sources().tobytes()) == facts
    assert chemistry._require_aromatic_representation(mol, rows=(2,)) is None
    assert all(not seeds for seeds in table._dirty_seeds[:3])


def test_aromatic_preparation_cannot_publish_after_caught_joined_failure():
    mol = _molecule()
    table = mol._atom_table
    _seed(mol, GROUP_INDEX['VALENCE'], (mol.atoms[0]._uid,))
    before = _dependency_state(table)
    with pytest.raises(RuntimeError, match='nested factual transaction failed'):
        with _FactTransaction(mol, producer=1):
            with pytest.raises(MemoryError):
                with _native_fail_after('seed_clear', exception=MemoryError):
                    chemistry._prepare_aromatic_representation(mol, rows=(2,))
            # This second call succeeds locally but must not unpoison its owner.
            chemistry._prepare_aromatic_representation(mol, rows=(2, 2))
            assert all(not seeds for seeds in table._dirty_seeds[:3])
    after = _dependency_state(table)
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1] == before[1]
