from types import SimpleNamespace

import pytest

from hotpot.cheminfo.search import native
from hotpot.cheminfo.search._search_native import NativeTarget


def _semantic():
    return {
        "degree": 0,
        "connectivity": 0,
        "valence": 0.0,
        "ring_count": 0,
        "ring_sizes": (),
    }


def _atom(atomic_number, row):
    return {
        "fields": {
            "atomic_number": atomic_number,
            "_hotpot_stereo_row": row,
        },
        "full": _semantic(),
        "ligand": _semantic(),
        "object": None,
    }


def _bond():
    return {
        "fields": {},
        "bond_kind": "single",
        "is_aromatic": False,
        "object": None,
    }


def _node(*leaves):
    return {"preflight": (), "leaves": tuple(leaves)}


def _path_target():
    return {
        "keys": (10, 20, 30),
        "edges": ((10, 20), (20, 30)),
        "atoms": (_atom(6, 10), _atom(6, 20), _atom(6, 30)),
        "bonds": (_bond(), _bond()),
    }


def _path_query(*, pin_left_to_right=False):
    left_leaves = [("native_set", "atomic_number", (6,))]
    if pin_left_to_right:
        left_leaves.append(
            ("native_set", "_hotpot_stereo_row", (30,))
        )
    left = _node(*left_leaves)
    unpinned = _node(
        ("native_set", "atomic_number", (6,)),
    )
    return {
        "keys": (1, 2, 3),
        "edges": ((1, 2), (2, 3)),
        "nodes": (left, unpinned, unpinned),
        "bonds": (
            {
                "preflight": (),
                "leaves": (),
                "exclude_aromatic": False,
                "bond_kind": None,
            },
            {
                "preflight": (),
                "leaves": (),
                "exclude_aromatic": False,
                "bond_kind": None,
            },
        ),
    }


def test_native_target_reuses_target_and_creates_each_problem(monkeypatch):
    target_calls = []
    problem_calls = []

    class FakeProblem:
        query_keys = (7, 9)

        def __init__(self, query_spec, target):
            problem_calls.append((query_spec, target))

        def cursor(self):
            return iter(((40, 50),))

    def make_target(target_spec):
        target = object()
        target_calls.append((target_spec, target))
        return target

    monkeypatch.setattr(
        native,
        "_native_module",
        lambda: SimpleNamespace(
            NativeTarget=make_target,
            NativeProblem=FakeProblem,
        ),
    )
    target_spec = {"target": "spec"}
    first_query = {"query": 1}
    second_query = {"query": 2}
    reusable = native._NativeTarget(target_spec)

    first = reusable.iter_mappings(first_query)
    second = reusable.iter_mappings(second_query)

    assert len(target_calls) == 1
    assert [query for query, _ in problem_calls] == [first_query, second_query]
    assert all(target is target_calls[0][1] for _, target in problem_calls)
    assert tuple(first) == ({7: 40, 9: 50},)
    assert tuple(second) == ({7: 40, 9: 50},)


def test_native_target_real_path_is_deterministic_and_can_pin_endpoint():
    reusable = native._NativeTarget(_path_target())

    first = tuple(reusable.iter_mappings(_path_query()))
    second = tuple(reusable.iter_mappings(_path_query()))
    pinned = tuple(
        reusable.iter_mappings(_path_query(pin_left_to_right=True))
    )

    assert first == second
    assert first == (
        {1: 10, 2: 20, 3: 30},
        {1: 30, 2: 20, 3: 10},
    )
    assert pinned == ({1: 30, 2: 20, 3: 10},)


def test_native_target_preserves_native_target_validation_error():
    malformed = _path_target() | {"atoms": (_atom(6, 10),)}

    with pytest.raises(Exception) as direct:
        NativeTarget(malformed)
    with pytest.raises(type(direct.value), match=str(direct.value)):
        native._NativeTarget(malformed)
