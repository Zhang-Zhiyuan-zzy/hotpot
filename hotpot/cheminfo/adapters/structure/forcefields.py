"""Read-only carrier capture for force-field topology consumers."""

from __future__ import annotations

from typing import Iterator, Mapping, Optional, Sequence, Tuple

from ...structure.forcefields import AtomTopologySignature, BondTopology, TopologyReference


__all__ = ("capture_topology",)


def _atom_index_map(atoms: Sequence["Atom"]) -> dict[int, int]:
    """Map each atom object's identity to its molecular atom-table position."""
    return {id(atom): index for index, atom in enumerate(atoms)}


def _bond_endpoint_indices(
    bond: "Bond",
    atom_indices: Mapping[int, int],
) -> Tuple[int, int]:
    """Return the atom-table positions of a bond's ordered atoms."""
    return atom_indices[id(bond.atom1)], atom_indices[id(bond.atom2)]


def _bond_key(bond: "Bond") -> Tuple[int, int]:
    first, second = sorted((int(bond.atom1.idx), int(bond.atom2.idx)))
    return first, second


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
) -> BondTopology:
    endpoints, bond_order, bond_kind = _bond_identity(bond, atom_indices)
    return BondTopology(
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


def _iter_metal_donor_pairs(
    mol: "Molecule",
) -> Iterator[Tuple["Atom", "Atom"]]:
    """Yield the metal and donor atoms of each explicit coordination bond."""
    for bond in mol.bonds:
        if not bond.is_metal_ligand_bond:
            continue
        if bond.atom1.is_metal:
            yield bond.atom1, bond.atom2
        else:
            yield bond.atom2, bond.atom1


def capture_atom_pair_acceptance_values(
    distances,
    *,
    capture_radii: bool,
    overlap_tolerance: float,
) -> Tuple[Tuple[Tuple[int, int], bool, float, Optional[float]], ...]:
    """Detach measured pair values, reading radii only where consumed."""
    pairs = []
    for distance in distances:
        measured = float(distance.measurement.distance)
        bonded = distance.target.bonded
        radius_sum = None
        if capture_radii and not bonded and measured > overlap_tolerance:
            radius_sum = (
                float(distance.target.first.atom.covalent_radius)
                + float(distance.target.second.atom.covalent_radius)
            )
        pairs.append((
            (distance.target.first.key, distance.target.second.key),
            bonded,
            measured,
            radius_sum,
        ))
    return tuple(pairs)


def capture_bond_acceptance_values(
    mol: "Molecule",
    atoms: Sequence["Atom"],
    *,
    capture_radii: bool,
) -> Tuple[Tuple[int, Tuple[int, int], Tuple[int, int], Optional[float], Optional[bool]], ...]:
    """Capture bond sequence positions, endpoint rows/labels and scoped radii."""
    atom_indices = _atom_index_map(atoms)
    bonds = []
    for bond_index, bond in enumerate(mol.bonds):
        first, second = _bond_endpoint_indices(bond, atom_indices)
        labels = (int(atoms[first].idx), int(atoms[second].idx))
        radius_sum = None
        metal_ligand = None
        if capture_radii:
            radius_sum = (
                float(atoms[first].covalent_radius)
                + float(atoms[second].covalent_radius)
            )
            if radius_sum > 0.0:
                metal_ligand = bond.is_metal_ligand_bond
        bonds.append((bond_index, (first, second), labels, radius_sum, metal_ligand))
    return tuple(bonds)


def capture_coordination_acceptance_values(
    mol: "Molecule",
    atoms: Sequence["Atom"],
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]:
    """Capture isolated metals, oriented donor rows and participating labels."""
    atom_indices = _atom_index_map(atoms)
    metals = tuple(index for index, atom in enumerate(atoms) if atom.is_metal)
    pairs = tuple(
        (atom_indices[id(metal)], atom_indices[id(donor)])
        for metal, donor in _iter_metal_donor_pairs(mol)
    )
    rows = set(metals).union(donor for _, donor in pairs)
    labels = tuple((index, int(atoms[index].idx)) for index in sorted(rows))
    return metals, pairs, labels


def capture_acceptance_bond_positions(
    mol: "Molecule",
) -> Tuple[Tuple[Tuple[int, int], int], ...]:
    """Join normalized endpoint labels to current bond sequence positions."""
    return tuple((_bond_key(bond), index) for index, bond in enumerate(mol.bonds))
