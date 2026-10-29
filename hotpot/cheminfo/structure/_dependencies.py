"""Finite Atom dependency declarations and UID-scoped pending invalidation.

This mechanism records validity only. Stage 3 retains the legacy eager chemistry;
no evaluator here invents a chemical result or marks a stale value current.
"""

from enum import IntEnum
from types import MappingProxyType

import numpy as np

from ._fields import FIELD_INDEX, FIELD_PRESENCE_BITS

__all__ = ()


class _FactStatus(IntEnum):
    CURRENT = 0
    NEEDS_RECALC = 1
    ABSENT = 2
    UNRESOLVED = 3
    CONFLICT = 4


GROUPS = ("VALENCE", "IMPLICIT_H", "AROMATIC_REPRESENTATION", "FORMAL_CHARGE_COVERAGE")
GROUP_INDEX = MappingProxyType({group: index for index, group in enumerate(GROUPS)})
GROUP_FIELDS = ("valence", "implicit_hydrogens", "is_aromatic", "formal_charge")
GROUP_UPSTREAM = MappingProxyType({
    "VALENCE": (),
    "IMPLICIT_H": ("VALENCE",),
    "AROMATIC_REPRESENTATION": ("VALENCE", "IMPLICIT_H"),
    "FORMAL_CHARGE_COVERAGE": (),
})
GROUP_SCOPE = MappingProxyType({
    "VALENCE": "atom", "IMPLICIT_H": "atom",
    "AROMATIC_REPRESENTATION": "component", "FORMAL_CHARGE_COVERAGE": "molecule",
})
GROUP_READ_POLICY = MappingProxyType({group: "report_status_without_evaluator" for group in GROUPS})
_GROUP_DOWNSTREAM = ((0, 1, 2), (1, 2), (2,), (3,))
_FIELD_GROUPS = MappingProxyType({
    "atomic_number": (0, 1, 2, 3),
    "formal_charge": (0, 1, 2, 3),
    "valence": (1, 2),
    "implicit_hydrogens": (2, 3),
    "is_aromatic": (0, 1, 2),
})
_MATERIALIZED_GROUP = MappingProxyType({field: index for index, field in enumerate(GROUP_FIELDS)})
_FIELD_EFFECTS = MappingProxyType({
    field: (tuple(group for group in _FIELD_GROUPS.get(field, ())
                  if group != _MATERIALIZED_GROUP.get(field)),
            _MATERIALIZED_GROUP.get(field))
    for field in FIELD_INDEX
})


def _status_from_presence(presence):
    if len(presence) == 1:
        mask = int(presence[0])
        return np.asarray([[0 if mask & FIELD_PRESENCE_BITS[FIELD_INDEX[field]] else 2
                            for field in GROUP_FIELDS]], dtype=np.uint8)
    status = np.empty((len(presence), len(GROUPS)), dtype=np.uint8)
    for index, field in enumerate(GROUP_FIELDS):
        status[:, index] = np.where(
            presence & FIELD_PRESENCE_BITS[FIELD_INDEX[field]],
            _FactStatus.CURRENT, _FactStatus.ABSENT,
        )
    return status


def _initialize_dependencies(table):
    table._group_status = _status_from_presence(table._presence[:len(table)])
    table._dirty_seeds = tuple(set() for _ in GROUPS)


def _dependency_state(table):
    return (table._group_status[:len(table)].copy(),
            tuple(frozenset(seed) for seed in table._dirty_seeds))


def _install_dependency_state(table, state):
    status, seeds = state
    candidate = np.asarray(status)
    if candidate.shape != (len(table), len(GROUPS)) or candidate.dtype.kind not in "iu":
        raise ValueError("Dependency status must align with Atom rows and finite groups")
    if candidate.size and (candidate.min() < 0 or candidate.max() > max(_FactStatus)):
        raise ValueError("Dependency status contains an undeclared state")
    if len(seeds) != len(GROUPS):
        raise ValueError("Dependency seeds must align with finite groups")
    prepared = np.empty((len(table._blocks[0]), len(GROUPS)), dtype=np.uint8)
    prepared[:len(table)] = candidate
    table._group_status = prepared
    table._dirty_seeds = tuple(set(seed) for seed in seeds)


def _remap_dependency_state(table, old_uids, new_uids, *, rows=None):
    rows = tuple(range(len(table))) if rows is None else tuple(rows)
    old_uids, new_uids = tuple(old_uids), tuple(new_uids)
    if len(old_uids) != len(table) or len(rows) != len(new_uids):
        raise ValueError("Dependency remap must align source and destination membership")
    mapping = {old_uids[row]: uid for row, uid in zip(rows, new_uids)}
    return (table._group_status[list(rows)].copy(),
            tuple(frozenset(mapping[uid] for uid in seeds if uid in mapping)
                  for seeds in table._dirty_seeds))


def _transaction(mol):
    from ._commit import _current_transaction
    return _current_transaction(mol)


def _seed(mol, group, uids):
    table = mol._atom_table
    uids = tuple(uids)
    transaction = _transaction(mol)
    # Queue downstream groups now, so consuming an upstream seed cannot hide
    # its effect from a later remote-first read of another group.
    for index in _GROUP_DOWNSTREAM[group]:
        if transaction is not None:
            transaction.touch_dependency_seeds(index, uids)
        table._dirty_seeds[index].update(uids)


def _mark_atom_facts(mol, rows, fields, *, transaction, atom_uids=None):
    table = mol._atom_table
    if len(fields) == 1:
        affected, materialized_group = _FIELD_EFFECTS[fields[0]]
        materialized = () if materialized_group is None else (materialized_group,)
    else:
        materialized = {_MATERIALIZED_GROUP[field] for field in fields if field in _MATERIALIZED_GROUP}
        affected = set()
        for field in fields:
            affected.update(_FIELD_GROUPS.get(field, ()))
        affected.difference_update(materialized)
    if not affected and not materialized:
        return
    uids = tuple(mol._atoms[row]._uid for row in rows) if atom_uids is None else atom_uids
    if len(rows) == 1:
        uid, row = uids[0], rows[0]
        for group in affected:
            seeds = table._dirty_seeds[group]
            if uid not in seeds:
                transaction.touch_dependency_seeds(group, uids)
                seeds.add(uid)
        for group in materialized:
            if int(table._group_status[row, group]) != 0:
                transaction.touch_dependency_status(rows, (group,))
                table._group_status[row, group] = 0
            seeds = table._dirty_seeds[group]
            if uid in seeds:
                transaction.touch_dependency_seeds(group, uids)
                seeds.remove(uid)
        return
    uid_set = set(uids)
    for group in affected:
        seeds = table._dirty_seeds[group]
        added = uid_set.difference(seeds)
        if added:
            transaction.touch_dependency_seeds(group, added)
            seeds.update(added)
    selection = np.asarray(rows, dtype=np.intp)
    for group in materialized:
        changed_rows = selection[table._group_status[selection, group] != 0]
        if len(changed_rows):
            transaction.touch_dependency_status(changed_rows, (group,))
            table._group_status[changed_rows, group] = 0
        seeds = table._dirty_seeds[group]
        removed = uid_set.intersection(seeds)
        if removed:
            transaction.touch_dependency_seeds(group, removed)
            seeds.difference_update(removed)


def _mark_topology_dirty(mol, atom_uids):
    for group in range(len(GROUPS)):
        _seed(mol, group, atom_uids)


def _mark_declarations_dirty(mol):
    _seed(mol, GROUP_INDEX["FORMAL_CHARGE_COVERAGE"], (atom._uid for atom in mol._atoms))


def _upstream_groups(group):
    selected = set()
    def visit(name):
        for upstream in GROUP_UPSTREAM[name]:
            visit(upstream)
        selected.add(GROUP_INDEX[name])
    visit(group)
    return tuple(index for index in range(len(GROUPS)) if index in selected)


def _component_rows(mol, seeds):
    # The sole index supplies all traversal. The temporary UID map is local to
    # the first dirty read, never retained on a carrier.
    by_uid = {atom._uid: atom for atom in mol._atoms}
    pending = [by_uid[uid] for uid in seeds if uid in by_uid]
    seen = set()
    while pending:
        atom = pending.pop()
        entry = mol._topology_index.get(id(atom))
        if entry is None or entry[0] in seen:
            continue
        seen.add(entry[0])
        for bond in entry[1]:
            pending.append(bond.atom2 if bond.atom1 is atom else bond.atom1)
    return tuple(seen)


def _expand_dirty(mol, group):
    table = mol._atom_table
    transaction = _transaction(mol)
    for index in _upstream_groups(group):
        seeds = table._dirty_seeds[index]
        if not seeds:
            continue
        scope = GROUP_SCOPE[GROUPS[index]]
        if scope == "component":
            rows = _component_rows(mol, seeds)
        elif scope == "molecule":
            rows = tuple(range(len(table)))
        else:
            rows = tuple(row for row, atom in enumerate(mol._atoms) if atom._uid in seeds)
        if transaction is not None:
            transaction.touch_dependency_status(rows, (index,))
            transaction.touch_dependency_seeds(index, tuple(seeds))
        for row in rows:
            if table._group_status[row, index] == _FactStatus.CURRENT:
                table._group_status[row, index] = _FactStatus.NEEDS_RECALC
        seeds.clear()


def _ensure_group(mol, group, *, rows=None):
    """Expand pending upstream invalidation and report the actual finite state."""
    with mol._mutation_lock:
        _expand_dirty(mol, group)
        selection = slice(0, len(mol._atom_table)) if rows is None else list(rows)
        return mol._atom_table._group_status[selection, GROUP_INDEX[group]].copy()
