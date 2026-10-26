"""Contracts for NetworkX-free direct topology algorithms."""

import numpy as np
import pytest

from hotpot.cheminfo.core import InternalCoordinates, Molecule
from hotpot.cheminfo.graph.components import connected_components
from hotpot.cheminfo.graph.paths import (
    all_simple_paths,
    depth_first_path,
    longest_simple_path,
    shortest_path,
)


EDGES = ((0, 1), (1, 2), (1, 3), (2, 3))


def test_components_include_isolates_and_have_one_total_order():
    assert connected_components(6, reversed(EDGES)) == (
        (0, 1, 2, 3),
        (4,),
        (5,),
    )


def test_shortest_path_uses_ascending_neighbors_and_reports_disconnection():
    diamond = ((0, 2), (2, 3), (0, 1), (1, 3))
    assert shortest_path(5, diamond, 0, 3) == [0, 1, 3]
    assert shortest_path(5, diamond, 0, 0) == [0]
    assert shortest_path(5, diamond, 0, 4) == []


def test_simple_paths_are_stable_under_edge_order_and_honor_cutoff():
    expected = [[0, 1, 2], [0, 1, 3, 2]]
    assert all_simple_paths(4, EDGES, 0, 2) == expected
    assert all_simple_paths(4, reversed(EDGES), 0, 2) == expected
    assert all_simple_paths(4, EDGES, 0, 2, cutoff=2) == [[0, 1, 2]]


def test_longest_path_uses_the_first_deterministic_tie():
    assert longest_simple_path(4, EDGES) == [0, 1, 3, 2]
    assert longest_simple_path(4, EDGES, source=0) == [0, 1, 3, 2]
    assert longest_simple_path(4, EDGES, source=0, target=3) == [0, 1, 2, 3]


def test_longest_path_selects_a_component_in_disconnected_topology():
    assert longest_simple_path(2, ()) == [0]
    assert longest_simple_path(3, ((1, 2),)) == [1, 2]


def test_depth_first_path_preserves_frozen_visit_events_without_graph_object():
    assert depth_first_path(
        4,
        EDGES,
        start_node=0,
        scope_nodes={1, 2, 3},
        max_depth=3,
    ) == [0, 1, 2]
    assert depth_first_path(4, EDGES, start_node=0, scope_nodes=None) is None
    assert depth_first_path(
        4,
        EDGES,
        start_node=0,
        scope_nodes=set(),
        min_depth=1,
    ) == [0]


def test_molecule_path_aliases_share_disconnection_semantics():
    molecule = Molecule()
    molecule._create_atom(atomic_number=6)
    molecule._create_atom(atomic_number=6)

    assert molecule.shortest_path_index(*molecule.atoms) == []
    assert molecule.shortest_paths_indices(*molecule.atoms) == []
    assert molecule.longest_path() == [0]


def test_zmatrix_includes_one_isolate_and_rejects_disconnected_topology():
    no_edges = np.empty((0, 2), dtype=int)
    np.testing.assert_array_equal(
        InternalCoordinates.calc_zmat_idx(no_edges, node_count=1),
        np.array([[0, -1, -1, -1]], dtype=int),
    )
    assert InternalCoordinates.calc_zmat(no_edges, np.zeros((1, 3))).shape == (1, 3)

    with pytest.raises(RuntimeError, match="incorrect Z-Matirx"):
        InternalCoordinates.calc_zmat_idx(no_edges, node_count=2)
