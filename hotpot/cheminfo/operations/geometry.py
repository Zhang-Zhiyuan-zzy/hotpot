"""Prepare molecular geometry queries and commit explicit coordinate changes."""

import numpy as np

from ..rules.mathematics.view_geometry import _angle_rows, _torsion_rows
from ..structure._commit import _commit_coordinates
from ..structure._identity import _capture_source
from ..structure._mutation import _normalize_atom_facts
from ..views._source import _capture_view_source, _geometry_snapshot


__all__ = ()


def _sequence_views(mol, scope, cls, materialize):
    with mol._mutation_lock:
        atoms = tuple(mol._atoms)
        uids = tuple(atom._uid for atom in atoms)
        index = mol._topology_index
        adjacency = tuple(tuple(index[id(bond._atoms[1] if bond._atoms[0] is atom else bond._atoms[0])][0]
                                for bond in index[id(atom)][1]) for atom in atoms)
        edges = tuple((index[id(bond._atoms[0])][0], index[id(bond._atoms[1])][0]) for bond in mol._bonds)
        rows = list(_angle_rows(adjacency) if scope == "angles" else _torsion_rows(edges, adjacency))
        constrained = getattr(mol._constraints, scope)
        if constrained:
            uid_rows = {uid: row for row, uid in enumerate(uids)}
            seen = set(rows)
            for members in constrained:
                entry = tuple(uid_rows[uid] for uid in members)
                if entry not in seen:
                    rows.append(entry)
                    seen.add(entry)
        complete = tuple(mol._bonds)
        bonds = {}
        for bond in complete:
            bonds.setdefault(frozenset((id(bond._atoms[0]), id(bond._atoms[1]))), bond)
        source = _capture_view_source(mol)
        geometry = _geometry_snapshot(mol, atoms)[1]
        return [materialize(cls, atoms, row, bonds, source=source, uids=uids, geometry=geometry)
                for row in rows]


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


def _is_disorder(subject, minimum_separation=0.5):
    from ..adapters.structure import geometry as adapter
    return any(pair.measurement.distance < minimum_separation
               for pair in adapter.measure_atom_pair_distances(subject, pair_scope="all"))


def _molecule_has_3d(mol):
    from ..rules.chemistry.descriptors.geometry import _has_distinct_rows
    if len(mol.atoms) < 2:
        return False
    reference = tuple(mol.atoms[0].coordinates)
    return any(_has_distinct_rows((reference, tuple(atom.coordinates)))
               for atom in mol.atoms[1:])


def _ring_has_3d(ring):
    from ..rules.chemistry.descriptors.geometry import _has_distinct_rows
    from ..views._source import _view_coordinates
    with ring.mol._mutation_lock:
        return _has_distinct_rows(tuple(tuple(row) for row in _view_coordinates(ring)))


def _constraint_bonds_angles(mol, exclude_metal_bonds):
    if exclude_metal_bonds:
        for bond in mol.bonds:
            if not bond.has_metal:
                bond.constraint = True
        for angle in mol.angles:
            if not angle.has_metal:
                angle.constraint = True


def _random_point_on_sphere(radius, rng):
    from ..rules.mathematics.sampling import _point_on_sphere
    random = np.random.rand if rng is None else rng.random
    return _point_on_sphere(radius, random(), random())


def _bond_segment_distance(bond, other_bond):
    from ..rules.mathematics import geometry
    return geometry.segment_segment_distance(bond.bond_segment, other_bond.bond_segment)


def _ring_relation_to_bond(ring, bond, not_in_same_molecule):
    from ..adapters.structure import geometry as adapter
    if ring.mol is not bond.mol or bond not in ring.mol.bonds:
        raise not_in_same_molecule(ring, bond)
    return adapter.determine_bond_ring_relation(ring, bond)


def _ring_geometry_cycle(ring):
    from ..adapters.structure import geometry as adapter
    return adapter.cycle_from_ring(ring)


def _closest_ring_edge_to_bond(ring, bond, not_in_same_molecule):
    from ..adapters.structure import geometry as adapter
    from ..rules.mathematics import geometry
    if ring.mol is not bond.mol or bond not in ring.mol.bonds:
        raise not_in_same_molecule(ring, bond)
    measurement = geometry.closest_cycle_edge(ring.geometry_cycle, bond.bond_segment)
    if measurement is None:
        return None
    return adapter.RingEdgeDistance(bond=ring.bonds[measurement.edge_index],
                                    measurement=measurement)
