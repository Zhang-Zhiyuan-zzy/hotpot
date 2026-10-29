"""Prepare molecular geometry queries and commit explicit coordinate changes."""

import numpy as np

from ..rules.mathematics.view_geometry import _angle_rows, _torsion_rows
from ..structure._commit import _commit_coordinates
from ..structure._identity import _capture_source
from ..structure._mutation import _normalize_atom_facts
from ..views._source import _capture_view_source


__all__ = ()


def _sequence_views(mol, scope, cls, materialize):
    with mol._mutation_lock:
        atoms = tuple(mol._atoms)
        index = mol._topology_index
        adjacency = tuple(tuple(index[id(bond.atom2 if bond.atom1 is atom else bond.atom1)][0]
                                for bond in index[id(atom)][1]) for atom in atoms)
        edges = tuple((index[id(bond.atom1)][0], index[id(bond.atom2)][0]) for bond in mol._bonds)
        rows = list(_angle_rows(adjacency) if scope == "angles" else _torsion_rows(edges, adjacency))
        constrained = getattr(mol._constraints, scope)
        if constrained:
            uid_rows = {atom._uid: row for row, atom in enumerate(atoms)}
            seen = set(rows)
            for members in constrained:
                entry = tuple(uid_rows[uid] for uid in members)
                if entry not in seen:
                    rows.append(entry)
                    seen.add(entry)
        complete = tuple(mol._bonds) + tuple(mol._hided_metal_bonds) + tuple(mol._hided_covalent_bonds)
        bonds = {frozenset((id(bond.atom1), id(bond.atom2))): bond for bond in complete}
        source = _capture_view_source(mol)
        return [materialize(cls, mol, row, bonds, source=source) for row in rows]


def _angle_views(mol, cls, materialize):
    return _sequence_views(mol, "angles", cls, materialize)


def _torsion_views(mol, cls, materialize):
    return _sequence_views(mol, "torsions", cls, materialize)


def _pair_distances(mol):
    from ..rules.mathematics.view_geometry import _pair_distances as calculate
    return calculate(mol.coordinates)


def _distance_matrix(mol):
    from ..rules.mathematics.view_geometry import _distance_matrix as calculate
    return calculate(mol.coordinates)


def _bond_length(bond):
    from ..rules.mathematics.view_geometry import _pair_distance
    with bond.mol._mutation_lock:
        return float(_pair_distance(bond.atom1.vector, bond.atom2.vector))


def _translate(mol, vector):
    with mol._mutation_lock:
        coordinates = mol._atom_table.coordinates() + np.asarray(vector)
        rows = tuple(range(len(mol._atoms)))
        update = _normalize_atom_facts(_capture_source(mol), rows,
                                       tuple(atom._uid for atom in mol._atoms),
                                       {"coordinates": coordinates})
        _commit_coordinates(mol, update)
