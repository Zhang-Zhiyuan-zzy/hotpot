"""Finite Atom dependency declarations and UID-scoped pending invalidation.

This module records validity only. Named chemistry operations prepare the
declared groups; a field write never executes a scientific rule.
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
    "VALENCE": "neighbourhood", "IMPLICIT_H": "neighbourhood",
    "AROMATIC_REPRESENTATION": "component", "FORMAL_CHARGE_COVERAGE": "molecule",
})
GROUP_READ_POLICY = MappingProxyType({
    'VALENCE': 'pure_local_projection_without_value_storage',
    'IMPLICIT_H': 'prepare_source_aware_hydrogens',
    'AROMATIC_REPRESENTATION': 'prepare_coupled_aromatic_representation',
    'FORMAL_CHARGE_COVERAGE': 'check_explicit_coverage_without_redistribution',
})
_GROUP_DOWNSTREAM = ((0, 1, 2), (1, 2), (2,), (3,))
_FIELD_GROUPS = MappingProxyType({
    "atomic_number": (0, 1, 2, 3),
    "formal_charge": (0, 1, 2, 3),
    "valence": (1, 2),
    "implicit_hydrogens": (2, 3),
    "is_aromatic": (0, 1, 2),
    'x': (2,), 'y': (2,), 'z': (2,),
})
_MATERIALIZED_GROUP = MappingProxyType({
    'implicit_hydrogens': GROUP_INDEX['IMPLICIT_H'],
    'is_aromatic': GROUP_INDEX['AROMATIC_REPRESENTATION'],
})
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


def _seed(mol, group, uids):
    from ._commit import _fact_engine
    # Queue downstream groups now, so consuming an upstream seed cannot hide
    # its effect from a later remote-first read of another group.
    _fact_engine()._update_dependency_seeds(mol, _GROUP_DOWNSTREAM[group], tuple(uids), True)


def _mark_topology_dirty(mol, atom_uids):
    for group in range(len(GROUPS)):
        _seed(mol, group, atom_uids)


def _mark_bond_representation_dirty(mol, atom_uids):
    _seed(mol, GROUP_INDEX['VALENCE'], atom_uids)


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


def _neighbour_rows(mol, seeds):
    rows = set()
    for atom in mol._atoms:
        if atom._uid in seeds:
            row, bonds = mol._topology_index[id(atom)]
            rows.add(row)
            rows.update(mol._topology_index[id(bond.atom2 if bond.atom1 is atom else bond.atom1)][0]
                        for bond in bonds)
    return tuple(sorted(rows))


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
    from ._commit import _fact_engine
    for index in _upstream_groups(group):
        seeds = table._dirty_seeds[index]
        if not seeds:
            continue
        scope = GROUP_SCOPE[GROUPS[index]]
        if scope == "neighbourhood":
            rows = _neighbour_rows(mol, seeds)
        elif scope == "component":
            rows = _component_rows(mol, seeds)
        elif scope == "molecule":
            rows = tuple(range(len(table)))
        else:
            rows = tuple(row for row, atom in enumerate(mol._atoms) if atom._uid in seeds)
        _fact_engine()._clear_dependency_seeds(mol, index, rows)


def _ensure_group(mol, group, *, rows=None):
    """Expand pending upstream invalidation and report the actual finite state."""
    with mol._mutation_lock:
        _expand_dirty(mol, group)
        selection = slice(0, len(mol._atom_table)) if rows is None else list(rows)
        return mol._atom_table._group_status[selection, GROUP_INDEX[group]].copy()
