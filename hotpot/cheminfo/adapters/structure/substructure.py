"""Read and pack prepared molecular values for native substructure search."""

from __future__ import annotations

from ...structure.substructure import SmartsSemantics


def _pack_target_values(mol, *, needs_atom_objects, needs_bond_objects, derived_requirements):
    atoms_by_key = {}
    for atom in mol.atoms:
        atoms_by_key[atom.idx] = atom

    bonds_by_edge = {}
    for bond in mol.bonds:
        edge = tuple(sorted((bond.a1idx, bond.a2idx)))
        bonds_by_edge[edge] = bond

    keys = tuple(sorted(atoms_by_key))
    edges = tuple(sorted(bonds_by_edge))
    needs_full_rings = (
        _needs_rings(derived_requirements["full"])
        or derived_requirements["bond_aromatic"]
    )
    full_rings = mol.rings if needs_full_rings else ()
    ligand_rings = (
        mol.ligand_rings
        if _needs_rings(derived_requirements["ligand"])
        else ()
    )
    full_ring_sizes = (
        _pack_ring_sizes(full_rings, keys)
        if _needs_rings(derived_requirements["full"])
        else {}
    )
    ligand_ring_sizes = (
        _pack_ring_sizes(ligand_rings, keys)
        if _needs_rings(derived_requirements["ligand"])
        else {}
    )
    aromatic_edges = (
        _pack_aromatic_edges(full_rings)
        if derived_requirements["bond_aromatic"]
        else set()
    )
    return {
        "keys": keys,
        "edges": edges,
        "atoms": tuple(
            _pack_atom(
                atoms_by_key[key],
                needs_atom_objects,
                derived_requirements,
                full_ring_sizes.get(key, ()),
                ligand_ring_sizes.get(key, ()),
            )
            for key in keys
        ),
        "bonds": tuple(
            _pack_bond(
                bonds_by_edge[edge],
                needs_bond_objects,
                derived_requirements["bond_fields"],
                edge in aromatic_edges,
            )
            for edge in edges
        ),
    }


def _pack_atom(
    atom,
    needs_object,
    derived_requirements,
    full_ring_sizes,
    ligand_ring_sizes,
):
    field_names = derived_requirements["atom_fields"]
    fields = {
        name: _pack_atom_field(atom, name)
        for name in field_names
    }
    if derived_requirements["total_h"]:
        fields["implicit_hydrogens"] = int(atom.implicit_hydrogens)
        fields["explicit_hydrogens"] = int(atom.explicit_hydrogens)
    return {
        "fields": fields,
        "full": _pack_semantics(
            atom,
            SmartsSemantics.FULL_GRAPH,
            derived_requirements["full"],
            full_ring_sizes,
        ),
        "ligand": _pack_semantics(
            atom,
            SmartsSemantics.LIGAND_SKELETON,
            derived_requirements["ligand"],
            ligand_ring_sizes,
        ),
        "object": atom if needs_object else None,
    }


def _pack_semantics(atom, semantics, required_fields, ring_sizes):
    record = {}
    if "degree" in required_fields:
        record["degree"] = int(semantics.degree(atom))
    if "connectivity" in required_fields:
        record["connectivity"] = int(semantics.connectivity(atom))
    if "valence" in required_fields:
        record["valence"] = _native_number(semantics.valence(atom))
    if "ring_count" in required_fields:
        record["ring_count"] = len(ring_sizes)
    if "ring_sizes" in required_fields:
        record["ring_sizes"] = ring_sizes
    return record


def _pack_bond(bond, needs_object, required_fields, is_aromatic):
    fields = {}
    if "bond_order" in required_fields:
        fields["bond_order"] = float(bond.bond_order)
    if "is_aromatic" in required_fields:
        fields["is_aromatic"] = bool(is_aromatic)
    return {
        "fields": fields,
        "bond_kind": bond.bond_kind.value,
        "is_aromatic": bool(is_aromatic),
        "object": bond if needs_object else None,
    }


def _pack_atom_field(atom, name):
    value = getattr(atom, name)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    return _native_number(value)


def _needs_rings(required_fields):
    return bool({"ring_count", "ring_sizes"}.intersection(required_fields))


def _native_number(value):
    return int(value) if isinstance(value, int) else float(value)


def _pack_ring_sizes(rings, keys):
    sizes = {key: [] for key in keys}
    for ring in rings:
        size = len(ring)
        for atom in ring.atoms:
            sizes[atom.idx].append(size)
    return {key: tuple(values) for key, values in sizes.items()}


def _pack_aromatic_edges(rings):
    return {
        tuple(sorted((bond.a1idx, bond.a2idx)))
        for ring in rings
        if ring.is_aromatic
        for bond in ring.bonds
    }


def _exact_topology_label_values(mol, keys, edges):
    atoms = {atom.idx: atom for atom in mol.atoms}
    bonds = {
        tuple(sorted((bond.a1idx, bond.a2idx))): bond
        for bond in mol.bonds
    }
    node_values = tuple(
        (
            int(atoms[key].atomic_number),
            int(atoms[key].formal_charge),
            bool(atoms[key].is_aromatic),
            int(atoms[key].implicit_hydrogens),
            int(getattr(atoms[key], "isotope", 0)),
        )
        for key in keys
    )
    edge_values = tuple(
        (
            "aromatic",
            bonds[edge].bond_kind.value,
        )
        if bonds[edge].is_aromatic
        else (
            "bond_order",
            float(bonds[edge].bond_order),
            bonds[edge].bond_kind.value,
        )
        for edge in edges
    )
    return _integer_labels(node_values), _integer_labels(edge_values)


def _integer_labels(values):
    labels = {
        value: label
        for label, value in enumerate(sorted(set(values), key=repr))
    }
    return tuple(labels[value] for value in values)
