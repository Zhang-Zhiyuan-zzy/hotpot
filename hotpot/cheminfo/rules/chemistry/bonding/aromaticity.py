"""The existing per-ring aromaticity model over an immutable bonding input."""

from itertools import product

import numpy as np

from ...mathematics.geometry import Cycle, measure_planarity
from ..elements import _is_metal

__all__ = ()

AROMATIC_PLANARITY_RELATIVE_TOLERANCE = 0.03


def _neighbours(snapshot):
    neighbours = [[] for _ in snapshot.atomic_numbers]
    for index, (_, first, second, _, _, _) in enumerate(snapshot.bonds):
        neighbours[first].append((second, index))
        neighbours[second].append((first, index))
    return tuple(tuple(sorted(entries)) for entries in neighbours)


def _determine_ring_aromaticity(snapshot, ring, has_3d, aromatic, *, neighbours=None):
    if all(aromatic[row] for row in ring):
        return True
    numbers = snapshot.atomic_numbers
    if any(_is_metal(numbers[row]) for row in ring):
        return False
    neighbours = _neighbours(snapshot) if neighbours is None else neighbours
    if not has_3d:
        electrons = 0
        for row in ring:
            number = numbers[row]
            hydrogen = snapshot.implicit_hydrogens[row]
            if hydrogen is None:
                return None
            count = sum(numbers[other] != 1 for other, _ in neighbours[row]) + hydrogen
            if number == 6:
                if count != 3:
                    return False
                electrons += 1
            elif number in (7, 15):
                if count == 3:
                    electrons += 2
                elif count == 2:
                    electrons += 1
                else:
                    return False
            elif number in (8, 16):
                if count != 2:
                    return False
                electrons += 2
            elif number != 5:
                return False
        return (electrons - 2) % 4 == 0
    planarity = measure_planarity(Cycle(snapshot.coordinates[row] for row in ring))
    if not (planarity.length_scale > 0.0 and np.isfinite(planarity.maximum_deviation)
            and planarity.maximum_deviation / planarity.length_scale < AROMATIC_PLANARITY_RELATIVE_TOLERANCE):
        return False
    electrons = []
    for row in ring:
        number, count = numbers[row], len(neighbours[row])
        if number == 6:
            if count > 3:
                return False
            electrons.append((1,))
        elif number in (7, 15):
            if count > 3:
                return False
            electrons.append((1, 2))
        elif number in (8, 16):
            if count != 2:
                return False
            electrons.append((2,))
    return any((sum(assignment) - 2) % 4 == 0 for assignment in product(*electrons))
