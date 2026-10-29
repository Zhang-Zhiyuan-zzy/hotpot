"""Finite status propagation, selective reads and UID remap contracts."""

import pickle

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.structure._atom_table import _AtomFactTable
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._dependencies import (
    GROUP_INDEX, GROUPS, _FactStatus, _dependency_state, _ensure_group,
    _install_dependency_state, _mark_topology_dirty, _remap_dependency_state,
    _seed,
)


def _molecule():
    mol = Molecule()
    for _ in range(5):
        mol.create_atom(attrs_array=np.zeros(19), update_electron_config=False)
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
    table = _AtomFactTable(np.ones((4, 19)))
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
    table = _AtomFactTable(np.ones((1, 19)))
    with pytest.raises(ValueError):
        _install_dependency_state(table, (status, seeds))
    assert table._group_status.tolist() == [[0] * 4]
