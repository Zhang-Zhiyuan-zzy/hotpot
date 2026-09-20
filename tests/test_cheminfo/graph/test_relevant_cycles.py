"""Public contract and independent-oracle tests for Relevant Cycles."""

from __future__ import annotations

from itertools import permutations

import networkx as nx
import numpy as np
import pytest

from hotpot.cheminfo.graph import (
    RelevantCycleLimitExceeded,
    relevant_cycles,
)

from .exhaustive_oracle import exhaustive_relevant_cycles
from .fixtures import normalized_edges, reference_graphs


@pytest.mark.parametrize(("name", "edges"), reference_graphs())
def test_relevant_cycles_match_independent_exhaustive_oracle(name, edges):
    assert relevant_cycles(edges, max_cycles=None) == exhaustive_relevant_cycles(edges), name


def test_expected_chemical_and_symmetric_ring_sets():
    cases = dict(reference_graphs())

    assert relevant_cycles(cases["square_with_diagonal"]) == (
        (0, 1, 2),
        (0, 2, 3),
    )
    assert tuple(map(len, relevant_cycles(cases["naphthalene_topology"]))) == (6, 6)
    assert tuple(map(len, relevant_cycles(cases["anthracene_topology"]))) == (6, 6, 6)
    assert tuple(map(len, relevant_cycles(cases["cubane"]))) == (4, 4, 4, 4, 4, 4)
    assert tuple(map(len, relevant_cycles(cases["complete_4"]))) == (3, 3, 3, 3)
    assert tuple(map(len, relevant_cycles(cases["theta"]))) == (4, 4, 4)


def test_numpy_edge_matrix_and_sparse_node_indices_are_supported():
    edges = np.array([(40, 70), (70, 90), (90, 40)], dtype=np.int64)
    assert relevant_cycles(edges) == ((40, 70, 90),)


def test_generator_input_is_consumed_once():
    edges = ((first, second) for first, second in ((0, 1), (1, 2), (0, 2)))
    assert relevant_cycles(edges) == ((0, 1, 2),)


def test_edge_order_and_orientation_do_not_change_results():
    edges = dict(reference_graphs())["square_with_diagonal"]
    expected = relevant_cycles(edges)
    for order in permutations(edges[:3]):
        remaining = edges[3:]
        candidate = tuple(tuple(reversed(edge)) for edge in order + remaining)
        assert relevant_cycles(candidate) == expected


def test_node_relabelling_preserves_the_cycle_set():
    edges = dict(reference_graphs())["cubane"]
    mapping = {node: 100 + 7 * node for edge in edges for node in edge}
    relabelled_edges = tuple((mapping[first], mapping[second]) for first, second in edges)
    expected = {
        frozenset(mapping[node] for node in cycle)
        for cycle in relevant_cycles(edges)
    }
    actual = {frozenset(cycle) for cycle in relevant_cycles(relabelled_edges)}
    assert actual == expected


def test_max_size_returns_an_exact_subset_of_global_relevant_cycles():
    edges = normalized_edges(tuple(nx.complete_graph(5).edges))
    all_cycles = relevant_cycles(edges, max_cycles=None)
    assert relevant_cycles(edges, max_size=3, max_cycles=None) == tuple(
        cycle for cycle in all_cycles if len(cycle) <= 3
    )
    assert relevant_cycles(edges, max_size=4, max_cycles=None) == tuple(
        cycle for cycle in all_cycles if len(cycle) <= 4
    )


def test_max_cycles_raises_instead_of_returning_a_partial_collection():
    edges = dict(reference_graphs())["complete_4"]
    with pytest.raises(RelevantCycleLimitExceeded):
        relevant_cycles(edges, max_cycles=3)


@pytest.mark.parametrize(
    ("edges", "error", "message"),
    (
        ([(0, 0)], ValueError, "self-loops"),
        ([(0, 1), (1, 0)], ValueError, "duplicate"),
        ([(0, -1)], ValueError, "nonnegative"),
        ([(0, 1, 2)], ValueError, "exactly two"),
        ([(0, 1.5)], TypeError, "integers"),
        ([(False, 1)], TypeError, "integers"),
    ),
)
def test_invalid_graph_inputs_are_rejected(edges, error, message):
    with pytest.raises(error, match=message):
        relevant_cycles(edges)


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    (
        ("max_size", 2, ValueError),
        ("max_size", -1, ValueError),
        ("max_size", 3.5, TypeError),
        ("max_cycles", -1, ValueError),
        ("max_cycles", True, TypeError),
    ),
)
def test_invalid_limits_are_rejected(keyword, value, error):
    with pytest.raises(error):
        relevant_cycles([(0, 1), (1, 2), (0, 2)], **{keyword: value})


def test_empty_edge_collection_returns_empty_without_loading_native_backend(monkeypatch):
    import hotpot.cheminfo.graph.cycles as cycle_module

    def unexpected_native_load():
        raise AssertionError("native backend should not load for an empty edge collection")

    monkeypatch.setattr(cycle_module, "_native_module", unexpected_native_load)
    assert relevant_cycles([]) == ()
