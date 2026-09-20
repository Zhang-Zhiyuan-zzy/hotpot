"""Regression fence for migrating ``hotpot.cheminfo.graph`` to a package.

These tests intentionally record the observable behaviour of the former
``graph.py`` module.  Behavioural fixes belong in later, dedicated changes.
"""

import importlib

import networkx as nx
import numpy as np


def _graph_api():
    return importlib.import_module("hotpot.cheminfo.graph")


def test_module_and_direct_imports_expose_the_legacy_api():
    graph = _graph_api()

    from hotpot.cheminfo.graph import (
        GraphSpectrum,
        adj2laplacian,
        atoms_electron_configurations,
        calc_electron_config,
        calc_spectrum,
        graph_dfs_path,
        graph_dfs_paths,
        linkmat2adj,
    )

    expected = {
        "GraphSpectrum": GraphSpectrum,
        "adj2laplacian": adj2laplacian,
        "atoms_electron_configurations": atoms_electron_configurations,
        "calc_electron_config": calc_electron_config,
        "calc_spectrum": calc_spectrum,
        "graph_dfs_path": graph_dfs_path,
        "graph_dfs_paths": graph_dfs_paths,
        "linkmat2adj": linkmat2adj,
    }
    assert {name: getattr(graph, name) for name in expected} == expected


def test_electron_configuration_outputs_are_unchanged():
    graph = _graph_api()

    assert graph.calc_electron_config(1) == (0, [1, 0, 0, 0])
    assert graph.calc_electron_config(10) == (1, [2, 6, 0, 0])
    assert graph.calc_electron_config(26) == (3, [2, 6, 0, 0])
    assert graph.calc_electron_config(26, length=2) == (3, [2, 6])

    configurations = graph.atoms_electron_configurations([1, 6, 8])
    np.testing.assert_array_equal(
        configurations,
        np.array(
            [
                [0, 1, 1],
                [1, 2, 2],
                [0, 2, 4],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    )


def test_link_matrix_conversion_preserves_normal_and_empty_shapes():
    graph = _graph_api()

    adjacency = graph.linkmat2adj(4, np.array([[0, 1], [1, 2]], dtype=int))
    np.testing.assert_array_equal(
        adjacency,
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert adjacency.dtype == np.dtype(float)

    empty = graph.linkmat2adj(4, np.empty((0, 2), dtype=int))
    assert empty.shape == (1, 0)
    assert empty.dtype == np.dtype(int)


def test_laplacian_conversion_preserves_normalization_and_isolated_nodes():
    graph = _graph_api()
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    np.testing.assert_allclose(
        graph.adj2laplacian(adjacency, norm=False),
        np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]),
    )
    np.testing.assert_allclose(
        graph.adj2laplacian(adjacency),
        np.array(
            [
                [1.0, -1.0 / np.sqrt(2.0), 0.0],
                [-1.0 / np.sqrt(2.0), 1.0, -1.0 / np.sqrt(2.0)],
                [0.0, -1.0 / np.sqrt(2.0), 1.0],
            ]
        ),
    )
    np.testing.assert_array_equal(graph.adj2laplacian(np.zeros((2, 2))), np.zeros((2, 2)))

    empty = graph.adj2laplacian(np.empty((1, 0)))
    assert empty.shape == (1, 0)
    assert empty.dtype == np.dtype(int)


def test_spectrum_and_graph_spectrum_behaviour_is_unchanged():
    graph = _graph_api()
    adjacency = np.array([[0, 1], [1, 0]])
    atomic_numbers = np.array([1, 1])

    spectrum = graph.calc_spectrum(adjacency, atomic_numbers)
    np.testing.assert_allclose(
        spectrum,
        np.array(
            [
                [2.0, 0.0],
                [1.0, -1.0],
                [2.0, 0.0],
                [1.0, -1.0],
                [1.0, -1.0],
                [1.0, -1.0],
            ]
        ),
    )

    first = graph.GraphSpectrum.from_adj_atoms(adjacency, atomic_numbers)
    second = graph.GraphSpectrum(spectrum.copy())
    assert first.width == 2
    assert first.vectors is first.spectrum
    assert first.similarity(second) == 1.0
    assert (first | second) == 1.0

    narrow = graph.GraphSpectrum(np.array([[1.0], [1.0]]), norm="min")
    wide = graph.GraphSpectrum(np.array([[1.0, 0.0], [0.0, 1.0]]), norm="min")
    assert narrow.similarity(wide) == 0.0


def test_depth_first_path_preserves_scope_and_depth_semantics():
    graph_api = _graph_api()
    graph = nx.Graph([(0, 1), (1, 2), (1, 3)])

    assert graph_api.graph_dfs_path(graph, start_node=0, scope_nodes={1, 2, 3}, max_deep=3) == [0, 1, 2]
    assert graph_api.graph_dfs_path(graph, start_node=0, scope_nodes=None) is None
    assert graph_api.graph_dfs_path(graph, start_node=0, scope_nodes=set(), min_deep=1) == [0]


def test_depth_first_paths_preserves_current_empty_result():
    """The legacy function defines a DFS closure but never invokes it."""
    graph_api = _graph_api()
    graph = nx.path_graph(4)

    assert graph_api.graph_dfs_paths(graph, start_node=0, scope_nodes={1, 2, 3}, min_deep=2) == []
