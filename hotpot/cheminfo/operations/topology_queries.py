"""Prepare current topology once for the retained deterministic graph rules."""

from ..rules.mathematics import graph
from ..rules.mathematics.graph.paths import shortest_path
from ..rules.mathematics import topology


__all__ = ()


def _topology(mol, selection=None):
    from ..structure._selection import _selected_bonds

    index = mol._topology_index
    return len(mol._atoms), tuple((index[id(bond.atom1)][0], index[id(bond.atom2)][0])
                                  for bond in _selected_bonds(mol, selection))


def _simple_paths(mol, source, target, cutoff):
    with mol._mutation_lock:
        return topology._simple_paths(*_topology(mol), source, target, cutoff)


def _longest_path(mol):
    with mol._mutation_lock:
        return topology._longest_path(*_topology(mol))


def _canonical_tree(mol):
    with mol._mutation_lock:
        return topology._canonical_tree(*_topology(mol))


def _shortest_path_indices(mol, first, second):
    with mol._mutation_lock:
        return shortest_path(*_topology(mol), mol._topology_index[id(first)][0],
                             mol._topology_index[id(second)][0])


def _shortest_path(mol, first, second):
    with mol._mutation_lock:
        return [mol._atoms[row] for row in _shortest_path_indices(mol, first, second)]


def _cycle_basis(mol):
    with mol._mutation_lock:
        count, edges = _topology(mol)
        return list(graph._cycle_basis(range(count), edges))


def _all_cycle_basis(mol):
    with mol._mutation_lock:
        count, edges = _topology(mol)
        return [list(basis) for basis in graph._all_cycle_basis(range(count), edges)]


def _ring_cycles(mol, *, ligand_skeleton=False, max_size=None,
                 max_cycles=graph.DEFAULT_RELEVANT_CYCLE_LIMIT):
    index = mol._topology_index
    return graph.relevant_cycles(
        ((index[id(bond._atoms[0])][0], index[id(bond._atoms[1])][0])
         for bond in mol._bonds if not ligand_skeleton or not bond.is_metal_ligand_bond),
        max_size=max_size, max_cycles=max_cycles)


def _rings_from_cycles(mol, cls, cycles):
    from .chemistry import _prepare_aromatic_representation
    from ..views._source import (
        _capture_view_source, _geometry_snapshot, _initialize_view_source,
        _member_geometry, _ring_representation,
    )
    with mol._mutation_lock:
        _prepare_aromatic_representation(mol)
        atoms = tuple(mol._atoms)
        uids = tuple(atom._uid for atom in atoms)
        index = mol._topology_index
        bonds = {frozenset((index[id(bond._atoms[0])][0], index[id(bond._atoms[1])][0])): bond
                 for bond in mol._bonds}
        source = _capture_view_source(mol)
        geometry = _geometry_snapshot(mol, atoms)[1]
        result = []
        for rows in cycles:
            rows = tuple(rows)
            ring = object.__new__(cls)
            members = tuple(atoms[row] for row in rows)
            edges = tuple(bonds.get(frozenset((first, second)))
                          for first, second in zip(rows, (*rows[1:], rows[0])))
            object.__setattr__(ring, '_atoms', members)
            object.__setattr__(ring, '_bonds', edges)
            _initialize_view_source(ring, source, tuple(uids[row] for row in rows),
                                    _member_geometry(geometry, rows, len(atoms)))
            object.__setattr__(ring, '_view_representation', _ring_representation(mol, members, edges))
            result.append(ring)
        return result


def _ring_views(mol, cls, *, ligand_skeleton=False, max_size=None,
                max_cycles=graph.DEFAULT_RELEVANT_CYCLE_LIMIT):
    with mol._mutation_lock:
        return _rings_from_cycles(mol, cls, _ring_cycles(mol,
            ligand_skeleton=ligand_skeleton, max_size=max_size, max_cycles=max_cycles))


def _project_molecule(mol, molecule_type, *, selection, rows=None):
    from ..structure._commit import _commit_topology
    from ..structure.conformer import _copy_source_frames
    from ..structure._dependencies import _mark_topology_dirty, _remap_dependency_state
    from ..structure._identity import _capture_source
    from ..structure._selection import _selected_bonds

    with mol._mutation_lock:
        rows = tuple(range(len(mol._atoms))) if rows is None else tuple(rows)
        selected = _selected_bonds(mol, selection)
        selected_ids = {id(bond) for bond in selected}
        retained_ids = {id(mol._atoms[row]) for row in rows}
        excluded_uids = {atom._uid for bond in mol._bonds
                         if id(bond) not in selected_ids
                         or any(id(member) not in retained_ids for member in bond._atoms)
                         for atom in bond._atoms}
        record, metadata = mol._capture_factual_record(
            rows=rows, include_frames=False, selection=selection)
        state = _remap_dependency_state(
            mol._atom_table, tuple(atom._uid for atom in mol._atoms),
            tuple(range(len(rows))), rows=rows)
        excluded_rows = tuple(target for target, row in enumerate(rows)
                              if mol._atoms[row]._uid in excluded_uids)
    projected = molecule_type()
    _commit_topology(projected, _capture_source(projected), record,
        atom_bindings=(None,) * record.atom_count,
        bond_bindings=(None,) * len(record.bond_endpoints),
        dependency_state=state, bond_metadata=metadata)
    if excluded_rows:
        _mark_topology_dirty(projected, tuple(projected._atoms[row]._uid
                                             for row in excluded_rows))
    with mol._mutation_lock:
        _copy_source_frames(mol, projected, rows=rows)
    return projected


def _components(mol, molecule_type, *, with_bindings=False, selection=None):
    from ..rules.mathematics.graph.components import connected_components

    with mol._mutation_lock:
        members = tuple(mol._atoms)
        selections = connected_components(*_topology(mol, selection))
        result = []
        for rows in selections:
            component = _project_molecule(mol, molecule_type, selection=selection, rows=rows)
            result.append((component, tuple(members[row] for row in rows))
                          if with_bindings else component)
        return result


def atom_idx_pair_to_bond_idx(mol: "Molecule", *atom_idx_pair: tuple[int, int]) -> list[int]:
    bond_atom_idx_pair = [frozenset((b.a1idx, b.a2idx)) for b in mol.bonds]
    atom_idx_pair = [frozenset(a_pair) for a_pair in atom_idx_pair]
    return [bond_atom_idx_pair.index(a_pair) for a_pair in atom_idx_pair]


def _rings_for_scope(mol, ring_scope, *, max_size=None,
                     max_cycles=graph.DEFAULT_RELEVANT_CYCLE_LIMIT):
    if ring_scope == "full_graph":
        return mol._materialize_rings(max_size=max_size, max_cycles=max_cycles)
    if ring_scope == "ligand_skeleton":
        return mol._materialize_rings(ligand_skeleton=True, max_size=max_size,
                                      max_cycles=max_cycles)
    raise ValueError(f"Unsupported ring scope: {ring_scope!r}")


def _aromatic_joint_rings(mol):
    rings = mol.rings
    joint_rings = []
    while rings:
        ring = rings.pop()
        joint_ring = ring.joint_ring()
        if joint_ring:
            to_remove = []
            for candidate in rings:
                if candidate in joint_ring:
                    to_remove.append(candidate)
            for candidate in to_remove:
                rings.remove(candidate)
            joint_rings.append(joint_ring)
    return joint_rings


def _rings_small(mol):
    from ..rules.chemistry.descriptors.rings import _is_small_ring
    return [ring for ring in mol.rings if _is_small_ring(len(ring))]


def _ring_joint_with(ring, other, ring_type):
    if not isinstance(other, ring_type):
        raise TypeError(f"expected Ring but got {type(other)}")
    from ..views._source import _check_ring_source
    with ring.mol._mutation_lock:
        _check_ring_source(ring, prepare=True)
        if other.mol is not ring.mol:
            return False
        _check_ring_source(other, prepare=True)
        intersect_bonds = list(set(ring.bonds) & set(other.bonds))
    from ..rules.chemistry.descriptors.rings import _rings_are_joint
    return _rings_are_joint(len(intersect_bonds))


def _joint_ring(ring, aromatic, joint_ring_type):
    from ..views._source import _check_ring_source
    with ring.mol._mutation_lock:
        _check_ring_source(ring, prepare=True)
        if aromatic and not ring.is_aromatic:
            return None
        rings = []
        for candidate in ring.mol.rings:
            if candidate == ring:
                rings.append(candidate)
            elif ring.joint_with(candidate) and (not aromatic or candidate.is_aromatic):
                rings.append(candidate)
        if rings:
            return joint_ring_type(*rings)


def _c_bonds(mol):
    return [bond for bond in mol.bonds if bond.is_metal_ligand_bond]
