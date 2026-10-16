"""Scientific-state orchestration and immutable molecular-fact extraction."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Optional, Tuple

from ..core import Bond, BondKind, Molecule
from .contracts import (
    ComponentIdentity,
    HydrogenInventory,
    MolecularFacts,
    MolecularFactsIdentity,
    SemanticBond,
    SemanticBondKind,
)

__all__ = ["extract_molecular_facts"]


# Canonical chemical-boundary serialization.


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atom_map(mol: Molecule) -> Tuple[int, ...]:
    atom_ids = tuple(int(atom.id) for atom in mol.atoms)
    if len(set(atom_ids)) == len(atom_ids) and all(atom_id >= 0 for atom_id in atom_ids):
        return atom_ids
    return tuple(range(len(mol.atoms)))


def _semantic_bond_kind(bond: Bond) -> SemanticBondKind:
    if bond.bond_kind is BondKind.DATIVE:
        return "coordination"
    if bond.bond_kind is BondKind.UNKNOWN or bond.is_metal_ligand_bond:
        return "unknown"
    return "covalent"


def _semantic_bonds(mol: Molecule) -> Tuple[SemanticBond, ...]:
    bonds = []
    for bond in mol.bonds:
        atom_indices = tuple(sorted((int(bond.a1idx), int(bond.a2idx))))
        order = None if bond.bond_kind is BondKind.UNKNOWN else float(bond.bond_order)
        bonds.append(
            SemanticBond(
                atom_indices=atom_indices,
                order=order,
                aromatic=bond.bond_kind is BondKind.AROMATIC,
                kind=_semantic_bond_kind(bond),
            )
        )
    return tuple(sorted(bonds, key=lambda item: item.atom_indices))


def _component_indices(
    atom_count: int,
    bonds: Iterable[SemanticBond],
) -> Tuple[Tuple[int, ...], ...]:
    adjacency = [set() for _ in range(atom_count)]
    for bond in bonds:
        begin, end = bond.atom_indices
        adjacency[begin].add(end)
        adjacency[end].add(begin)

    components = []
    unseen = set(range(atom_count))
    while unseen:
        pending = [min(unseen)]
        component = set()
        while pending:
            atom_index = pending.pop()
            if atom_index in component:
                continue
            component.add(atom_index)
            unseen.discard(atom_index)
            pending.extend(adjacency[atom_index] - component)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _component_identities(
    component_kind: str,
    components: Tuple[Tuple[int, ...], ...],
    atom_map: Tuple[int, ...],
) -> Tuple[ComponentIdentity, ...]:
    return tuple(
        ComponentIdentity(
            identity=_sha256(
                {
                    "component_kind": component_kind,
                    "mapped_atom_ids": sorted(atom_map[index] for index in component),
                }
            ),
            atom_indices=component,
        )
        for component in components
    )


def _coordinates(
    mol: Molecule,
) -> Optional[Tuple[Tuple[float, float, float], ...]]:
    if not mol.has_3d:
        return None
    return tuple(
        tuple(float(value) for value in atom.coordinates)
        for atom in mol.atoms
    )


def _chemical_identity_payload(facts: MolecularFacts) -> object:
    atom_records = sorted(
        (
            facts.atom_map[index],
            atomic_number,
            facts.mass_numbers[index],
            facts.hydrogen_inventory.explicit_counts[index],
            facts.hydrogen_inventory.implicit_counts[index],
        )
        for index, atomic_number in enumerate(facts.atomic_numbers)
    )
    bond_records = sorted(
        (
            min(facts.atom_map[begin], facts.atom_map[end]),
            max(facts.atom_map[begin], facts.atom_map[end]),
            bond.order,
            bond.aromatic,
            bond.kind,
        )
        for bond in facts.semantic_bonds
        for begin, end in (bond.atom_indices,)
    )
    return {
        "atoms": atom_records,
        "bonds": bond_records,
        "source_components": sorted(
            sorted(facts.atom_map[index] for index in component.atom_indices)
            for component in facts.source_components
        ),
        "covalent_components": sorted(
            sorted(facts.atom_map[index] for index in component.atom_indices)
            for component in facts.covalent_components
        ),
    }


# Public fact extraction.


def extract_molecular_facts(mol: Molecule) -> MolecularFacts:
    """Extract immutable chemical facts without mutating ``mol``."""
    atoms = tuple(mol.atoms)
    atomic_numbers = tuple(int(atom.atomic_number) for atom in atoms)
    mass_numbers = tuple(
        int(atom.isotope) if getattr(atom, "isotope", None) else None
        for atom in atoms
    )
    atom_map = _atom_map(mol)
    semantic_bonds = _semantic_bonds(mol)
    source_components = _component_identities(
        "source",
        _component_indices(len(atoms), semantic_bonds),
        atom_map,
    )
    covalent_bonds = tuple(
        bond for bond in semantic_bonds if bond.kind != "coordination"
    )
    covalent_components = _component_identities(
        "covalent",
        _component_indices(len(atoms), covalent_bonds),
        atom_map,
    )
    hydrogen_inventory = HydrogenInventory(
        explicit_counts=tuple(
            sum(neighbour.atomic_number == 1 for neighbour in atom.neighbours)
            for atom in atoms
        ),
        implicit_counts=tuple(
            int(atom.implicit_hydrogens) if atom.implicit_hydrogens else None
            for atom in atoms
        ),
    )
    coordinates = _coordinates(mol)
    provisional = MolecularFacts(
        identity=MolecularFactsIdentity(chemical_identity_digest=""),
        geometry_digest=_sha256(coordinates) if coordinates is not None else None,
        atomic_numbers=atomic_numbers,
        mass_numbers=mass_numbers,
        atom_map=atom_map,
        source_components=source_components,
        covalent_components=covalent_components,
        hydrogen_inventory=hydrogen_inventory,
        semantic_bonds=semantic_bonds,
        coordinates=coordinates,
        nuclear_charge_sum=sum(atomic_numbers),
    )
    return MolecularFacts(
        identity=MolecularFactsIdentity(
            chemical_identity_digest=_sha256(_chemical_identity_payload(provisional))
        ),
        geometry_digest=provisional.geometry_digest,
        atomic_numbers=provisional.atomic_numbers,
        mass_numbers=provisional.mass_numbers,
        atom_map=provisional.atom_map,
        source_components=provisional.source_components,
        covalent_components=provisional.covalent_components,
        hydrogen_inventory=provisional.hydrogen_inventory,
        semantic_bonds=provisional.semantic_bonds,
        coordinates=provisional.coordinates,
        nuclear_charge_sum=provisional.nuclear_charge_sum,
    )
