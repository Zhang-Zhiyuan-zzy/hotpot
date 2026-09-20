"""Legacy molecule-level aromaticity and Kekule workflow."""

from __future__ import annotations

from typing import Optional

from ._protocols import MoleculeLike
from .aromaticity import determine_ring_aromaticity
from .assignment import kekulize_ring


__all__ = (
    "kekulize_molecule_rings",
    "perceive_ligand_ring_aromaticity",
)


def perceive_ligand_ring_aromaticity(
        mol: MoleculeLike,
        force: Optional[bool] = None,
) -> None:
    """Run the pre-existing ligand-ring perception gate used by valence setup."""
    if force is not False:
        rings = mol.ligand_rings
        if force or (rings and not any(ring.is_aromatic for ring in rings)):
            for ring in rings:
                determine_ring_aromaticity(ring, inplace=True)


def kekulize_molecule_rings(mol: MoleculeLike) -> None:
    """Run the pre-existing hide, per-ring assignment, and recover workflow."""
    mol.hide_metal_ligand_bonds()
    for ring in mol.rings:
        kekulize_ring(ring)
    mol.recover_hided_metal_ligand_bonds()
