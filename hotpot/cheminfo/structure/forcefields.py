"""Immutable molecular-topology values shared by force-field consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


__all__ = ("AtomTopologySignature", "BondTopology", "TopologyReference")


@dataclass(frozen=True)
class AtomTopologySignature:
    """Stable identity and chemistry for an atom present before optimization."""

    index: int
    atom_id: int
    atomic_number: int
    formal_charge: int


@dataclass(frozen=True, order=True)
class BondTopology:
    """One bond in a topology revision, addressed by atom indices."""

    atom_indices: Tuple[int, int]
    bond_order: float
    bond_kind: str

    def __post_init__(self) -> None:
        atom1_index, atom2_index = sorted(self.atom_indices)
        object.__setattr__(self, "atom_indices", (atom1_index, atom2_index))

    @classmethod
    def from_bond(cls, bond: "Bond") -> "BondTopology":
        """Capture one Hotpot bond without retaining mutable objects."""
        return cls(
            atom_indices=(int(bond.atom1.idx), int(bond.atom2.idx)),
            bond_order=float(bond.bond_order),
            bond_kind=str(bond.bond_kind.value),
        )


@dataclass(frozen=True)
class TopologyReference:
    """Immutable topology snapshot used by force-field transactions."""

    atoms: Tuple[AtomTopologySignature, ...]
    bonds: Tuple[BondTopology, ...]
    allow_added_hydrogens: bool = True
