"""Legacy per-ring aromaticity perception, isolated from Core."""

from __future__ import annotations

from itertools import product

import numpy as np

from ..geometry import measure_planarity
from ._protocols import RingLike
from .settings import AROMATIC_PLANARITY_RELATIVE_TOLERANCE


__all__ = ("determine_ring_aromaticity",)


def determine_ring_aromaticity(
        ring: RingLike,
        inplace: bool = False,
) -> bool:
    """Run the pre-existing Hotpot per-ring aromaticity procedure."""
    if ring.is_aromatic:
        return True

    def _neutral_mol_check() -> bool:
        if not ring.has_3d:
            pi_electron = 0
            for atom in ring._atoms:
                if atom.atomic_number == 6:
                    if len(atom.heavy_neighbours) + atom.implicit_hydrogens != 3:
                        return False
                    pi_electron += 1

                elif atom.atomic_number in (7, 15):
                    if len(atom.heavy_neighbours) + atom.implicit_hydrogens == 3:
                        pi_electron += 2
                    elif len(atom.heavy_neighbours) + atom.implicit_hydrogens == 2:
                        pi_electron += 1
                    else:
                        return False

                elif atom.atomic_number in (8, 16):
                    if len(atom.heavy_neighbours) + atom.implicit_hydrogens != 2:
                        return False
                    pi_electron += 2

                elif atom.atomic_number == 5:
                    pi_electron += 0

                else:
                    return False

            return (pi_electron - 2) % 4 == 0

        planarity = measure_planarity(ring.geometry_cycle)
        if not (
            planarity.length_scale > 0.0
            and np.isfinite(planarity.maximum_deviation)
            and planarity.maximum_deviation / planarity.length_scale
            < AROMATIC_PLANARITY_RELATIVE_TOLERANCE
        ):
            return False

        pi_electrons = []
        for atom in ring._atoms:
            if atom.atomic_number == 6:
                if len(atom.neigh_idx) > 3:
                    return False
                pi_electrons.append((1,))

            elif atom.atomic_number in (7, 15):
                if len(atom.neigh_idx) > 3:
                    return False
                pi_electrons.append((1, 2))

            elif atom.atomic_number in (8, 16):
                if len(atom.neigh_idx) != 2:
                    return False
                pi_electrons.append((2,))

        return any(
            (sum(electron_assignment) - 2) % 4 == 0
            for electron_assignment in product(*pi_electrons)
        )

    judge = False if ring.has_metal else _neutral_mol_check()
    if inplace:
        ring.is_aromatic = judge
    return judge
