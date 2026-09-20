"""Legacy Kekule bond assignment, isolated from Core."""

from __future__ import annotations

from ._protocols import JointRingLike, RingLike
from .aromaticity import determine_ring_aromaticity


__all__ = (
    "check_joint_ring_kekulization",
    "kekulize_joint_ring",
    "kekulize_ring",
)


def check_joint_ring_kekulization(joint_ring: JointRingLike) -> bool:
    """Run the pre-existing atom-rule check for a joint ring system."""
    for atom in joint_ring.atoms:
        if atom.atomic_number == 6:
            if atom.sum_heavy_cov_orders != 3:
                return False
        if atom.atomic_number in (7, 15):
            if atom.sum_heavy_cov_orders == 2 and atom.implicit_hydrogens != 1:
                return False
            if atom.sum_heavy_cov_orders == 3 and atom.implicit_hydrogens != 0:
                return False
        if atom.atomic_number in (5, 8, 16):
            if atom.sum_heavy_cov_orders == 2:
                return False
    return True


def kekulize_joint_ring(joint_ring: JointRingLike) -> None:
    """Preserve the currently unimplemented joint-ring operation."""
    raise NotImplementedError


def kekulize_ring(ring: RingLike) -> None:
    """Run the pre-existing greedy per-ring Kekule assignment."""
    if not determine_ring_aromaticity(ring, inplace=True):
        return

    for bond in ring._bonds:
        bond.bond_order = 1

    for bond in ring._bonds:
        if all(
            end_atom.atomic_number not in (5, 8, 16)
            and adjacent_bond.bond_order == 1
            for end_atom in bond.atoms
            for adjacent_bond in end_atom.bonds
        ):
            bond.bond_order = 2
