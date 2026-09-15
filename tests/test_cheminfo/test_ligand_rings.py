import numpy as np

from hotpot.cheminfo.core import Molecule


def _chelating_molecule():
    molecule = Molecule()
    for atomic_number in (29, 7, 6, 6, 7):
        molecule.create_atom(atomic_number=atomic_number)
    molecule.add_bonds(
        (
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 0, 1.0),
            (2, 4, 1.0),
        )
    )
    molecule._update_graph(clear_conformers=False)
    molecule.conformer_add(np.arange(15, dtype=float).reshape(5, 3))
    molecule.conformer_add(np.arange(15, 30, dtype=float).reshape(5, 3))
    return molecule


def _ring_atom_indices(rings):
    return [tuple(atom.idx for atom in ring.atoms) for ring in rings]


def test_ligand_rings_filters_metal_edges_without_mutating_molecule():
    molecule = _chelating_molecule()
    original_bonds = tuple(molecule.bonds)
    original_graph = molecule.graph
    original_graph_edges = tuple(molecule.graph.edges)
    original_rings = _ring_atom_indices(molecule.rings)
    original_ring_cache = molecule._rings
    original_ring_objects = tuple(original_ring_cache)
    original_conformer_array = molecule.conformers._coordinates
    original_conformers = molecule.conformers._coordinates.copy()

    ligand_rings = molecule.ligand_rings

    assert {frozenset(ring) for ring in _ring_atom_indices(ligand_rings)} == {
        frozenset((2, 3, 4))
    }
    assert all(
        current is original
        for current, original in zip(molecule.bonds, original_bonds)
    )
    assert molecule.graph is original_graph
    assert tuple(molecule.graph.edges) == original_graph_edges
    assert _ring_atom_indices(molecule.rings) == original_rings
    assert molecule._rings is original_ring_cache
    assert all(
        current is original
        for current, original in zip(molecule._rings, original_ring_objects)
    )
    assert molecule.conformers._coordinates is original_conformer_array
    np.testing.assert_array_equal(molecule.conformers._coordinates, original_conformers)
    assert molecule._hided_metal_bonds == []
