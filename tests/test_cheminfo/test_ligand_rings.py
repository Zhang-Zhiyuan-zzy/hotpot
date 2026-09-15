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


def _ring_atom_sets(rings):
    return {frozenset(atom.idx for atom in ring.atoms) for ring in rings}


def _three_membered_carbon_chain():
    molecule = Molecule()
    for _ in range(3):
        molecule.create_atom(atomic_number=6)
    molecule.add_bond(0, 1, 1.0)
    molecule.add_bond(1, 2, 1.0)
    return molecule


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


def test_repeated_ligand_ring_reads_are_equivalent_and_non_mutating():
    molecule = _chelating_molecule()
    bonds = tuple(molecule.bonds)
    graph = molecule.graph
    graph_edges = frozenset(frozenset(edge) for edge in graph.edges)
    full_rings = _ring_atom_sets(molecule.rings)

    first = _ring_atom_sets(molecule.ligand_rings)
    second = _ring_atom_sets(molecule.ligand_rings)

    assert first == second == {frozenset((2, 3, 4))}
    assert tuple(molecule.bonds) == bonds
    assert molecule.graph is graph
    assert frozenset(frozenset(edge) for edge in graph.edges) == graph_edges
    assert _ring_atom_sets(molecule.rings) == full_rings


def test_ligand_rings_refresh_after_public_topology_changes():
    molecule = _three_membered_carbon_chain()

    assert _ring_atom_sets(molecule.ligand_rings) == set()

    closing_bond = molecule.add_bond(2, 0, 1.0)
    assert _ring_atom_sets(molecule.ligand_rings) == {frozenset((0, 1, 2))}

    molecule.remove_bond(closing_bond)
    assert _ring_atom_sets(molecule.ligand_rings) == set()


def test_ligand_rings_refresh_after_endpoint_metallicity_changes_in_place():
    molecule = _three_membered_carbon_chain()
    molecule.add_bond(2, 0, 1.0)
    original_bonds = tuple(molecule.bonds)
    original_edges = frozenset(frozenset(edge) for edge in molecule.graph.edges)

    assert _ring_atom_sets(molecule.ligand_rings) == {frozenset((0, 1, 2))}

    molecule.atoms[0].atomic_number = 29
    assert _ring_atom_sets(molecule.ligand_rings) == set()

    molecule.atoms[0].atomic_number = 6
    assert _ring_atom_sets(molecule.ligand_rings) == {frozenset((0, 1, 2))}
    assert tuple(molecule.bonds) == original_bonds
    assert frozenset(frozenset(edge) for edge in molecule.graph.edges) == original_edges
