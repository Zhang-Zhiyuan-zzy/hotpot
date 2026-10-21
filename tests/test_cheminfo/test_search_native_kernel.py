from itertools import combinations, permutations

import pytest

from hotpot.cheminfo.search.native import (
    _NativeMappingSource,
    _synthetic_problem,
)
from hotpot.cheminfo.search.search import Hits, _MappingIterator


def _edges(node_count, mask):
    pairs = tuple(combinations(range(node_count), 2))
    return tuple(pair for index, pair in enumerate(pairs) if mask & (1 << index))


def _all_compatible_problem(query_keys, query_edges, target_keys, target_edges):
    return _synthetic_problem(
        query_keys,
        query_edges,
        target_keys,
        target_edges,
        [[True] * len(target_keys) for _ in query_keys],
        [[True] * len(target_edges) for _ in query_edges],
    )


def _brute_force(query_count, query_edges, target_count, target_edges):
    target_edge_set = {frozenset(edge) for edge in target_edges}
    return {
        mapping
        for mapping in permutations(range(target_count), query_count)
        if all(
            frozenset((mapping[first], mapping[second])) in target_edge_set
            for first, second in query_edges
        )
    }


def test_empty_query_emits_one_empty_mapping():
    problem = _all_compatible_problem((), (), (0, 1), ((0, 1),))
    cursor = problem.cursor()

    assert iter(cursor) is cursor
    assert list(cursor) == [()]
    assert list(problem.cursor()) == [()]


def test_sparse_query_keys_survive_the_legacy_mapping_adapter():
    problem = _all_compatible_problem(
        (0, 2),
        ((0, 2),),
        (10, 11),
        ((10, 11),),
    )
    mappings = tuple(
        _MappingIterator(
            _NativeMappingSource(problem).subgraph_monomorphisms_iter()
        )
    )

    assert {tuple(mapping.items()) for mapping in mappings} == {
        ((0, 10), (2, 11)),
        ((0, 11), (2, 10)),
    }


def test_non_induced_path_keeps_all_triangle_automorphisms_deterministically():
    problem = _all_compatible_problem(
        (0, 1, 2),
        ((0, 1), (1, 2)),
        (0, 1, 2),
        ((0, 1), (0, 2), (1, 2)),
    )
    first = tuple(problem.cursor())
    second = tuple(problem.cursor())

    assert first == second
    assert len(first) == 6
    assert set(first) == set(permutations(range(3)))


def test_disconnected_query_shares_injectivity_and_allows_cross_component_edges():
    problem = _all_compatible_problem(
        (0, 1),
        (),
        (0, 1),
        ((0, 1),),
    )

    assert set(problem.cursor()) == {(0, 1), (1, 0)}


def test_self_loop_requires_a_compatible_target_self_loop():
    accepted = _synthetic_problem(
        (0,),
        ((0, 0),),
        (4, 5),
        ((4, 4),),
        ((True, True),),
        ((True,),),
    )
    rejected = _synthetic_problem(
        (0,),
        ((0, 0),),
        (4,),
        ((4, 4),),
        ((True,),),
        ((False,),),
    )

    assert list(accepted.cursor()) == [(4,)]
    assert list(rejected.cursor()) == []


def test_synthetic_node_and_edge_compatibility_filter_candidates():
    problem = _synthetic_problem(
        (0, 1),
        ((0, 1),),
        (0, 1, 2),
        ((0, 1), (1, 2)),
        (
            (True, False, False),
            (False, True, True),
        ),
        ((False, True),),
    )

    assert list(problem.cursor()) == []

    accepted = _synthetic_problem(
        (0, 1),
        ((0, 1),),
        (0, 1, 2),
        ((0, 1), (1, 2)),
        (
            (True, False, False),
            (False, True, True),
        ),
        ((True, False),),
    )
    assert list(accepted.cursor()) == [(0, 1)]


@pytest.mark.parametrize(
    ("limit", "expected_count", "truncated"),
    ((0, 0, True), (2, 2, True), (3, 3, False), (4, 3, False)),
)
def test_python_facade_is_the_only_bounded_iteration_owner(
    limit, expected_count, truncated
):
    problem = _all_compatible_problem((0,), (), (0, 1, 2), ())
    mappings = _MappingIterator(
        _NativeMappingSource(problem).subgraph_monomorphisms_iter(),
        max_matches=limit,
    )

    assert len(tuple(mappings)) == expected_count
    assert mappings.truncated is truncated


def test_hits_consumes_the_same_unbounded_native_source_once():
    problem = _all_compatible_problem((0,), (), (0, 1, 2), ())
    hits = Hits(
        None,
        None,
        _NativeMappingSource(problem),
        get_hit=False,
        max_matches=2,
    )

    assert list(hits) == [frozenset((0,)), frozenset((1,))]
    assert hits.truncated


def test_problem_copies_inputs_and_cursor_owns_problem_lifetime():
    query_keys = [0]
    target_keys = [7, 8]
    node_compatible = [[True, True]]
    problem = _synthetic_problem(
        query_keys,
        (),
        target_keys,
        (),
        node_compatible,
        (),
    )
    cursor = problem.cursor()

    query_keys[0] = 99
    target_keys[:] = [42]
    node_compatible[0][0] = False
    del problem

    assert list(cursor) == [(7,), (8,)]


def test_graph_boundary_rejects_noncanonical_edges():
    with pytest.raises(ValueError, match="ascending undirected"):
        _all_compatible_problem((0, 1), ((1, 0),), (0, 1), ((0, 1),))
    with pytest.raises(ValueError, match="sorted lexicographically"):
        _all_compatible_problem(
            (0, 1, 2),
            ((1, 2), (0, 1)),
            (0, 1, 2),
            ((0, 1), (1, 2)),
        )


def test_small_simple_graphs_match_an_independent_injection_oracle():
    for query_count in range(4):
        query_edge_count = query_count * (query_count - 1) // 2
        for target_count in range(query_count, 5):
            target_edge_count = target_count * (target_count - 1) // 2
            for query_mask in range(1 << query_edge_count):
                query_edges = _edges(query_count, query_mask)
                for target_mask in range(1 << target_edge_count):
                    target_edges = _edges(target_count, target_mask)
                    problem = _all_compatible_problem(
                        tuple(range(query_count)),
                        query_edges,
                        tuple(range(target_count)),
                        target_edges,
                    )
                    assert set(problem.cursor()) == _brute_force(
                        query_count,
                        query_edges,
                        target_count,
                        target_edges,
                    )
