from itertools import combinations, permutations

import pytest

from hotpot.cheminfo.search import _search_native
from hotpot.cheminfo.search._search_native import NativeProblem, NativeTarget
from hotpot.cheminfo.search.native import _NativeMappingSource
from hotpot.cheminfo.search.search import Hits, _MappingIterator


_NODE_VALUE_BASE = 10_000
_EDGE_VALUE_BASE = 20_000
_NEVER_VALUE = -1


def _edges(node_count, mask):
    pairs = tuple(combinations(range(node_count), 2))
    return tuple(pair for index, pair in enumerate(pairs) if mask & (1 << index))


def _semantic():
    return {
        "degree": 0,
        "connectivity": 0,
        "valence": 0.0,
        "ring_count": 0,
        "ring_sizes": (),
    }


def _target_atom(position):
    return {
        "fields": {"atomic_number": _NODE_VALUE_BASE + position},
        "full": _semantic(),
        "ligand": _semantic(),
        "object": None,
    }


def _target_bond(position):
    return {
        "fields": {"bond_order": _EDGE_VALUE_BASE + position},
        "bond_kind": "single",
        "is_aromatic": False,
        "object": None,
    }


def _allowed_values(row, base):
    values = tuple(base + position for position, accepted in enumerate(row) if accepted)
    return values or (_NEVER_VALUE,)


def _query_node(row):
    return {
        "preflight": ("atomic_number",),
        "leaves": (
            ("native_set", "atomic_number", _allowed_values(row, _NODE_VALUE_BASE)),
        ),
    }


def _query_bond(row):
    return {
        "preflight": ("bond_order",),
        "leaves": (
            ("native_set", "bond_order", _allowed_values(row, _EDGE_VALUE_BASE)),
        ),
        "exclude_aromatic": False,
        "bond_kind": None,
    }


def _native_problem(
    query_keys,
    query_edges,
    target_keys,
    target_edges,
    node_compatible,
    edge_compatible,
):
    query = {
        "keys": tuple(query_keys),
        "edges": tuple(query_edges),
        "nodes": tuple(_query_node(row) for row in node_compatible),
        "bonds": tuple(_query_bond(row) for row in edge_compatible),
    }
    target = {
        "keys": tuple(target_keys),
        "edges": tuple(target_edges),
        "atoms": tuple(_target_atom(position) for position in range(len(target_keys))),
        "bonds": tuple(_target_bond(position) for position in range(len(target_edges))),
    }
    return NativeProblem(query, NativeTarget(target))


def _all_compatible_problem(query_keys, query_edges, target_keys, target_edges):
    return _native_problem(
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


def test_temporary_synthetic_problem_interface_is_removed():
    assert not hasattr(_search_native, "SyntheticProblem")


def test_empty_query_emits_one_empty_mapping():
    problem = _all_compatible_problem((), (), (0, 1), ((0, 1),))
    cursor = problem.cursor()

    assert iter(cursor) is cursor
    assert list(cursor) == [()]
    assert list(problem.cursor()) == [()]


def test_sparse_query_keys_survive_the_native_mapping_source():
    problem = _all_compatible_problem(
        (0, 2),
        ((0, 2),),
        (10, 11),
        ((10, 11),),
    )
    mappings = tuple(
        _MappingIterator(_NativeMappingSource(problem).iter_mappings())
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
    accepted = _native_problem(
        (0,),
        ((0, 0),),
        (4, 5),
        ((4, 4),),
        ((True, True),),
        ((True,),),
    )
    rejected = _native_problem(
        (0,),
        ((0, 0),),
        (4,),
        ((4, 4),),
        ((True,),),
        ((False,),),
    )

    assert list(accepted.cursor()) == [(4,)]
    assert list(rejected.cursor()) == []


def test_native_node_and_edge_predicates_filter_candidates():
    problem = _native_problem(
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

    accepted = _native_problem(
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
        _NativeMappingSource(problem).iter_mappings(),
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
    accepted_values = [_NODE_VALUE_BASE, _NODE_VALUE_BASE + 1]
    query = {
        "keys": query_keys,
        "edges": [],
        "nodes": [
            {
                "preflight": ["atomic_number"],
                "leaves": [("native_set", "atomic_number", accepted_values)],
            }
        ],
        "bonds": [],
    }
    target_atoms = [_target_atom(0), _target_atom(1)]
    target = {
        "keys": target_keys,
        "edges": [],
        "atoms": target_atoms,
        "bonds": [],
    }
    problem = NativeProblem(query, NativeTarget(target))
    cursor = problem.cursor()

    query_keys[0] = 99
    target_keys[:] = [42]
    accepted_values[0] = _NEVER_VALUE
    target_atoms[1]["fields"]["atomic_number"] = _NEVER_VALUE
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
