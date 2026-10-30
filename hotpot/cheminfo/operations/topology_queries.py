"""Prepare current topology once for the retained deterministic graph rules."""

from ..rules.mathematics import graph
from ..rules.mathematics.graph.paths import shortest_path
from ..rules.mathematics import topology


__all__ = ()


def _topology(mol):
    index = mol._topology_index
    return len(mol._atoms), tuple((index[id(bond.atom1)][0], index[id(bond.atom2)][0])
                                  for bond in mol._bonds)


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


def _components(mol, molecule_type, *, with_bindings=False):
    from ..rules.mathematics.graph.components import connected_components
    from ..structure._commit import _commit_topology
    from ..structure._dependencies import _remap_dependency_state
    from ..structure._identity import _capture_source

    with mol._mutation_lock:
        members = tuple(mol._atoms)
        selections = connected_components(*_topology(mol))
        captures = [(rows, mol._capture_factual_record(rows=rows, include_frames=False),
                     _remap_dependency_state(mol._atom_table, tuple(atom._uid for atom in mol._atoms),
                                             tuple(range(len(rows))), rows=tuple(rows))) for rows in selections]
    result = []
    for rows, (record, metadata), state in captures:
        component = molecule_type()
        _commit_topology(component, _capture_source(component), record,
            atom_bindings=(None,) * record.atom_count,
            bond_bindings=(None,) * len(record.bond_endpoints),
            dependency_state=state, bond_metadata=metadata)
        result.append((component, tuple(members[row] for row in rows))
                      if with_bindings else component)
    return result
