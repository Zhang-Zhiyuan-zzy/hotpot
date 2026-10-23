"""Differential contracts for the private native cycle-basis facade."""

from __future__ import annotations

import networkx as nx
import pytest

from hotpot.cheminfo.graph.cycles import _all_cycle_basis, _cycle_basis

from .cycle_basis_oracle import (
    all_simple_cycle_bases,
    canonical_cycle,
    cycle_rank,
    deterministic_paton_basis,
    simple_cycles,
)


TREE = ((0, 1), (1, 2), (1, 3))
TRIANGLE = ((0, 1), (1, 2), (2, 0))
SQUARE_WITH_DIAGONAL = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 2),
)
COMPLETE_4 = tuple(
    (first, second)
    for first in range(4)
    for second in range(first + 1, 4)
)
DISCONNECTED_TRIANGLES = TRIANGLE + ((4, 5), (5, 6), (6, 4))


def _canonical_basis(basis):
    return tuple(
        sorted(
            (canonical_cycle(cycle) for cycle in basis),
            key=lambda cycle: (len(cycle), cycle),
        )
    )


def _expected_all_bases(nodes, edges):
    default = deterministic_paton_basis(nodes, edges)
    canonical_default = _canonical_basis(default)
    remaining = tuple(
        basis
        for basis in all_simple_cycle_bases(nodes, edges)
        if basis != canonical_default
    )
    return (default,) + remaining


def _atlas_cases():
    cases = []
    for atlas_index, graph in enumerate(nx.graph_atlas_g()):
        nodes = tuple(graph.nodes)
        if len(nodes) > 5:
            continue
        edges = tuple(graph.edges)
        rank = cycle_rank(nodes, edges)
        candidates = simple_cycles(nodes, edges)
        if rank <= 3 and len(candidates) <= 8:
            cases.append(
                pytest.param(
                    nodes,
                    edges,
                    id=f"atlas-{atlas_index}-n{len(nodes)}-r{rank}",
                )
            )
    return cases


@pytest.mark.parametrize(
    ("nodes", "edges"),
    (
        pytest.param(range(4), TREE, id="tree"),
        pytest.param(range(3), TRIANGLE, id="triangle"),
        pytest.param(
            range(4),
            SQUARE_WITH_DIAGONAL,
            id="square-with-diagonal",
        ),
        pytest.param(range(4), COMPLETE_4, id="complete-four"),
        pytest.param(
            range(7),
            DISCONNECTED_TRIANGLES,
            id="disconnected-triangles",
        ),
    ),
)
def test_native_default_and_complete_bases_match_independent_oracle(
    nodes,
    edges,
):
    assert _cycle_basis(nodes, edges) == deterministic_paton_basis(nodes, edges)
    assert _all_cycle_basis(nodes, edges) == _expected_all_bases(nodes, edges)


def test_native_results_have_exact_immutable_nested_shapes():
    default = _cycle_basis(range(4), SQUARE_WITH_DIAGONAL)
    all_bases = _all_cycle_basis(range(4), SQUARE_WITH_DIAGONAL)

    assert isinstance(default, tuple)
    assert all(isinstance(cycle, tuple) for cycle in default)
    assert isinstance(all_bases, tuple)
    assert all(isinstance(basis, tuple) for basis in all_bases)
    assert all(
        isinstance(cycle, tuple)
        for basis in all_bases
        for cycle in basis
    )
    assert all_bases[0] == default


def test_acyclic_graph_has_one_empty_complete_basis():
    assert _cycle_basis(range(4), TREE) == ()
    assert _all_cycle_basis(range(4), TREE) == ((),)


def test_sparse_keys_and_isolated_nodes_are_preserved():
    nodes = (10, 20, 40, 99)
    edges = ((10, 20), (20, 40), (40, 10))

    assert _cycle_basis(nodes, edges) == deterministic_paton_basis(nodes, edges)
    assert _all_cycle_basis(nodes, edges) == _expected_all_bases(nodes, edges)


def test_default_and_complete_results_ignore_edge_input_order_and_orientation():
    shuffled = tuple(
        tuple(reversed(edge)) for edge in reversed(SQUARE_WITH_DIAGONAL)
    )

    assert _cycle_basis(range(4), shuffled) == _cycle_basis(
        range(4),
        SQUARE_WITH_DIAGONAL,
    )
    assert _all_cycle_basis(range(4), shuffled) == _all_cycle_basis(
        range(4),
        SQUARE_WITH_DIAGONAL,
    )


@pytest.mark.parametrize(("nodes", "edges"), _atlas_cases())
def test_native_bases_match_oracle_across_bounded_graph_atlas(nodes, edges):
    assert _cycle_basis(nodes, edges) == deterministic_paton_basis(nodes, edges)
    assert _all_cycle_basis(nodes, edges) == _expected_all_bases(nodes, edges)


def test_cycle_basis_rejects_self_loops_at_the_algorithm_boundary():
    with pytest.raises(ValueError, match="self-loop"):
        _cycle_basis((0,), ((0, 0),))
    with pytest.raises(ValueError, match="self-loop"):
        _all_cycle_basis((0,), ((0, 0),))


def test_complete_basis_limit_raises_instead_of_returning_a_prefix():
    complete_five = tuple(
        (first, second)
        for first in range(5)
        for second in range(first + 1, 5)
    )

    with pytest.raises(OverflowError, match="complete-basis limit"):
        _all_cycle_basis(range(5), complete_five)
