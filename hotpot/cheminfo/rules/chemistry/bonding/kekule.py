"""The sole retained greedy per-ring assignment, returning one coupled value."""

from ....structure.snapshots import _AromaticAssignment
from .aromaticity import _determine_ring_aromaticity, _neighbours
from .bond_order import _kind_from_order

__all__ = ()


def _calculate_aromatic_assignment(snapshot, *, kekule):
    cyclic = {row for ring in snapshot.rings for row in ring}
    aromatic = [flag if row in cyclic else False for row, flag in enumerate(snapshot.aromatic)]
    orders = [bond[3] for bond in snapshot.bonds]
    neighbours = _neighbours(snapshot)
    edges = {frozenset((bond[1], bond[2])): index for index, bond in enumerate(snapshot.bonds)}
    assigned = set()
    for ring, has_3d in zip(snapshot.rings, snapshot.ring_has_3d):
        result = _determine_ring_aromaticity(snapshot, ring, has_3d, aromatic, neighbours=neighbours)
        if result is None:
            return None
        for row in ring:
            aromatic[row] = result
        if not kekule or not result:
            continue
        ring_bonds = tuple(edges[frozenset((first, second))]
                           for first, second in zip(ring, (*ring[1:], ring[0])))
        assigned.update(ring_bonds)
        for index in ring_bonds:
            orders[index] = 1.0
        for index in ring_bonds:
            first, second = snapshot.bonds[index][1:3]
            if all(snapshot.atomic_numbers[row] not in (5, 8, 16) and orders[adjacent] == 1
                   for row in (first, second) for _, adjacent in neighbours[row]):
                orders[index] = 2.0
    bond_aromatic = [False] * len(snapshot.bonds)
    for ring in snapshot.rings:
        if all(aromatic[row] for row in ring):
            for first, second in zip(ring, (*ring[1:], ring[0])):
                bond_aromatic[edges[frozenset((first, second))]] = True
    return _AromaticAssignment(snapshot.source, snapshot.atom_uids,
                               tuple(bond[0] for bond in snapshot.bonds), tuple(aromatic),
                               tuple(bond_aromatic), tuple(orders),
                               tuple(bond[4] if index not in assigned or bond[4] in ('zero', 'dative')
                                     else _kind_from_order(order) for index, (bond, order) in enumerate(zip(snapshot.bonds, orders))),
                               tuple(range(len(aromatic))), "kekule" if kekule else "aromaticity")


def _check_joint_ring_kekulization(numbers, heavy_orders, hydrogens):
    for number, order, hydrogen in zip(numbers, heavy_orders, hydrogens):
        if number == 6 and order != 3:
            return False
        if number in (7, 15):
            if order == 2 and hydrogen != 1:
                return False
            if order == 3 and hydrogen != 0:
                return False
        if number in (5, 8, 16) and order == 2:
            return False
    return True


def _aromatic_flags_assignment(snapshot, flags):
    edges = {frozenset((bond[1], bond[2])): index for index, bond in enumerate(snapshot.bonds)}
    bond_flags = [False] * len(snapshot.bonds)
    for ring in snapshot.rings:
        if all(flags[row] for row in ring):
            for first, second in zip(ring, (*ring[1:], ring[0])):
                bond_flags[edges[frozenset((first, second))]] = True
    return _AromaticAssignment(snapshot.source, snapshot.atom_uids,
        tuple(bond[0] for bond in snapshot.bonds), tuple(flags), tuple(bond_flags),
        tuple(bond[3] for bond in snapshot.bonds), tuple(bond[4] for bond in snapshot.bonds),
        tuple(range(len(flags))), 'aromaticity')
