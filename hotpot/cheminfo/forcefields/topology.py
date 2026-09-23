"""Immutable molecular-topology snapshots for force-field workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from ..core import Atom, Bond, Molecule


__all__ = (
    "AtomTopologySignature",
    "BondTopologySignature",
    "TopologyReference",
    "capture_topology",
)


@dataclass(frozen=True)
class AtomTopologySignature:
    """Stable identity and chemistry for an atom present before optimization."""

    index: int
    atom_id: int
    atomic_number: int
    formal_charge: int


@dataclass(frozen=True)
class BondTopologySignature:
    """Stable topology for a bond present before optimization."""

    atom_indices: Tuple[int, int]
    bond_order: float
    bond_kind: str


@dataclass(frozen=True)
class TopologyReference:
    """Immutable topology snapshot used by force-field transactions."""

    atoms: Tuple[AtomTopologySignature, ...]
    bonds: Tuple[BondTopologySignature, ...]
    allow_added_hydrogens: bool = True


def _atom_index_map(atoms: Sequence["Atom"]) -> dict[int, int]:
    """Map each atom object's identity to its molecular atom-table position."""
    return {id(atom): index for index, atom in enumerate(atoms)}


def _bond_endpoint_indices(
    bond: "Bond",
    atom_indices: Mapping[int, int],
) -> Tuple[int, int]:
    """Return the atom-table positions of a bond's ordered atoms."""
    return atom_indices[id(bond.atom1)], atom_indices[id(bond.atom2)]


def _atom_identity(atom: "Atom") -> Tuple[int, int, int]:
    """Return the stable chemical identity used by topology transactions."""
    return int(atom.id), int(atom.atomic_number), int(atom.formal_charge)


def _bond_identity(
    bond: "Bond",
    atom_indices: Mapping[int, int],
) -> Tuple[Tuple[int, int], float, str]:
    """Return orientation-independent endpoints, order, and kind for a bond."""
    first, second = sorted(_bond_endpoint_indices(bond, atom_indices))
    return (first, second), float(bond.bond_order), bond.bond_kind.value


def _topology_bond_signature(
    bond: "Bond",
    atom_indices: Mapping[int, int],
) -> BondTopologySignature:
    endpoints, bond_order, bond_kind = _bond_identity(bond, atom_indices)
    return BondTopologySignature(
        atom_indices=endpoints,
        bond_order=bond_order,
        bond_kind=bond_kind,
    )


def capture_topology(
    mol: "Molecule",
    *,
    allow_added_hydrogens: bool = True,
) -> TopologyReference:
    """Capture immutable topology expected to survive a force-field workflow."""
    atoms = tuple(mol.atoms)
    atom_indices = _atom_index_map(atoms)
    atom_signatures = tuple(
        AtomTopologySignature(
            index=index,
            atom_id=int(atom.id),
            atomic_number=int(atom.atomic_number),
            formal_charge=int(atom.formal_charge),
        )
        for index, atom in enumerate(atoms)
    )
    bond_signatures = tuple(sorted(
        (_topology_bond_signature(bond, atom_indices) for bond in mol.bonds),
        key=lambda signature: signature.atom_indices,
    ))
    return TopologyReference(
        atom_signatures,
        bond_signatures,
        allow_added_hydrogens=allow_added_hydrogens,
    )
