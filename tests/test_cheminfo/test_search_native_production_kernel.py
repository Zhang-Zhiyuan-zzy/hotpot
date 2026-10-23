import gc
from types import SimpleNamespace
import weakref

import pytest

from hotpot.cheminfo.search._search_native import (
    NativeProblem,
    NativeTarget,
    _NativeAutomorphism,
)


def _semantic(*, degree=0, connectivity=0, valence=0, ring_sizes=()):
    return {
        "degree": degree,
        "connectivity": connectivity,
        "valence": valence,
        "ring_count": len(ring_sizes),
        "ring_sizes": tuple(ring_sizes),
    }


def _atom(fields, obj=None, *, full=None, ligand=None):
    return {
        "fields": fields,
        "full": full or _semantic(),
        "ligand": ligand or full or _semantic(),
        "object": obj,
    }


def _bond(fields, obj=None, *, kind="single", aromatic=False):
    return {
        "fields": fields,
        "bond_kind": kind,
        "is_aromatic": aromatic,
        "object": obj,
    }


def _node(*leaves, preflight=()):
    return {"preflight": tuple(preflight), "leaves": tuple(leaves)}


def _query(keys, edges, nodes, bonds=()):
    return {
        "keys": tuple(keys),
        "edges": tuple(edges),
        "nodes": tuple(nodes),
        "bonds": tuple(bonds),
    }


def _target(keys, edges, atoms, bonds=()):
    return {
        "keys": tuple(keys),
        "edges": tuple(edges),
        "atoms": tuple(atoms),
        "bonds": tuple(bonds),
    }


def _problem(query, target):
    return NativeProblem(query, NativeTarget(target))


def _automorphism(keys, edges, node_labels, edge_labels):
    target = NativeTarget(
        _target(
            keys,
            edges,
            tuple(_atom({}) for _ in keys),
            tuple(_bond({}) for _ in edges),
        )
    )
    return _NativeAutomorphism(target, node_labels, edge_labels)


def test_native_fixed_root_automorphism_uses_exact_integer_labels():
    symmetric = _automorphism((0, 1, 2), ((0, 1), (1, 2)), (0, 1, 0), (4, 4))
    edge_asymmetric = _automorphism(
        (0, 1, 2),
        ((0, 1), (1, 2)),
        (0, 1, 0),
        (4, 5),
    )
    node_asymmetric = _automorphism(
        (0, 1, 2),
        ((0, 1), (1, 2)),
        (0, 1, 2),
        (4, 4),
    )

    assert symmetric.equivalent(0, 2)
    assert not edge_asymmetric.equivalent(0, 2)
    assert not node_asymmetric.equivalent(0, 2)


def test_native_fixed_root_automorphism_maps_disconnected_components():
    automorphism = _automorphism(
        (0, 1, 2, 3),
        ((0, 1), (2, 3)),
        (0, 1, 0, 1),
        (8, 8),
    )

    assert automorphism.equivalent(0, 2)
    assert not automorphism.equivalent(0, 1)


def _atom_expr(op, *, field=None, value=None, semantics=None, children=()):
    return {
        "op": op,
        "field": field,
        "value": value,
        "semantics": semantics,
        "children": tuple(children),
    }


def _bond_expr(op, *, kind=None, children=()):
    return {"op": op, "kind": kind, "children": tuple(children)}


def test_native_sets_and_typed_atom_expressions_run_in_the_production_cursor():
    expression = _atom_expr(
        "and",
        children=(
            _atom_expr("field_eq", field="formal_charge", value=0),
            _atom_expr("degree_eq", value=1, semantics="full_graph"),
            _atom_expr("ring_none", semantics="full_graph"),
            _atom_expr("total_h_eq", value=3),
        ),
    )
    query = _query(
        (0,),
        (),
        (
            _node(
                ("native_set", "atomic_number", (6,)),
                ("atom_expr", expression),
                preflight=("atomic_number",),
            ),
        ),
    )
    target = _target(
        (10, 11),
        (),
        (
            _atom(
                {
                    "atomic_number": 6,
                    "formal_charge": 0,
                    "explicit_hydrogens": 0,
                    "implicit_hydrogens": 3,
                },
                full=_semantic(degree=1, connectivity=4, valence=4),
            ),
            _atom(
                {
                    "atomic_number": 8,
                    "formal_charge": 0,
                    "explicit_hydrogens": 0,
                    "implicit_hydrogens": 0,
                },
                full=_semantic(degree=1, connectivity=1, valence=2),
            ),
        ),
    )

    assert list(_problem(query, target).cursor()) == [(10,)]


def test_preflight_and_native_short_circuit_happen_before_python_callbacks():
    calls = []

    def callback(obj):
        calls.append(obj)
        return True

    query = _query(
        (0,),
        (),
        (
            _node(
                ("native_set", "atomic_number", (6,)),
                ("callable", callback),
                preflight=("custom", "atomic_number"),
            ),
        ),
    )
    target = _target(
        (0, 1),
        (),
        (
            _atom({"atomic_number": 6}, object()),
            _atom({"atomic_number": 8}, SimpleNamespace(custom=1)),
        ),
    )

    problem = _problem(query, target)
    assert calls == []
    assert list(problem.cursor()) == []
    assert calls == []


def test_host_set_and_callable_receive_the_original_whole_object_lazily():
    first = SimpleNamespace(custom="accepted")
    second = SimpleNamespace(custom="accepted")
    calls = []

    def callback(obj):
        calls.append(obj)
        return obj is second

    query = _query(
        (0,),
        (),
        (
            _node(
                ("host_set", "custom", {"accepted"}),
                ("callable", callback),
                preflight=("custom",),
            ),
        ),
    )
    target = _target(
        (3, 4),
        (),
        (
            _atom({}, first),
            _atom({}, second),
        ),
    )

    assert list(_problem(query, target).cursor()) == [(4,)]
    assert calls == [first, second]


def test_host_membership_accepts_the_original_general_python_container():
    target_object = SimpleNamespace(custom="accepted")
    query = _query(
        (0,),
        (),
        (_node(("host_set", "custom", ["accepted"]), preflight=("custom",)),),
    )
    target = _target((0,), (), (_atom({}, target_object),))

    assert list(_problem(query, target).cursor()) == [(0,)]


def test_out_of_int64_target_value_stays_available_to_lazy_host_membership():
    huge = 1 << 100
    target_object = SimpleNamespace(atomic_number=huge)
    target = _target(
        (0,),
        (),
        (_atom({"atomic_number": huge}, target_object),),
    )
    native_query = _query(
        (0,),
        (),
        (
            _node(
                ("native_set", "atomic_number", (6,)),
                preflight=("atomic_number",),
            ),
        ),
    )
    host_query = _query(
        (0,),
        (),
        (
            _node(
                ("host_set", "atomic_number", {huge}),
                preflight=("atomic_number",),
            ),
        ),
    )

    native_target = NativeTarget(target)
    assert list(NativeProblem(native_query, native_target).cursor()) == []
    assert list(NativeProblem(host_query, native_target).cursor()) == [(0,)]


def test_native_problems_share_one_prepared_target():
    query = _query((0,), (), (_node(),))
    target_spec = _target((5, 6), (), (_atom({}), _atom({})))
    target = NativeTarget(target_spec)

    first = NativeProblem(query, target)
    second = NativeProblem(query, target)

    assert list(first.cursor()) == [(5,), (6,)]
    assert list(second.cursor()) == [(5,), (6,)]


def test_native_problem_does_not_accept_an_unprepared_target_spec():
    query = _query((0,), (), (_node(),))
    target = _target((0,), (), (_atom({}),))

    with pytest.raises(TypeError):
        NativeProblem(query, target)
    with pytest.raises(TypeError):
        NativeProblem(query, None)


def test_original_callback_exception_propagates_from_cursor():
    marker = RuntimeError("predicate failed")

    def callback(_obj):
        raise marker

    query = _query((0,), (), (_node(("callable", callback)),))
    target = _target((0,), (), (_atom({}, object()),))

    with pytest.raises(RuntimeError, match="predicate failed") as caught:
        next(_problem(query, target).cursor())

    assert caught.value is marker


def test_bond_guards_and_typed_bond_expression_are_exact():
    query = _query(
        (0, 1),
        ((0, 1),),
        (_node(), _node()),
        (
            {
                "preflight": (),
                "leaves": (
                    (
                        "bond_expr",
                        _bond_expr(
                            "or",
                            children=(
                                _bond_expr("kind_is", kind="single"),
                                _bond_expr("aromatic"),
                            ),
                        ),
                    ),
                ),
                "exclude_aromatic": True,
                "bond_kind": "single",
            },
        ),
    )
    target = _target(
        (0, 1, 2),
        ((0, 1), (1, 2)),
        (_atom({}), _atom({}), _atom({})),
        (
            _bond({}, kind="single", aromatic=False),
            _bond({}, kind="single", aromatic=True),
        ),
    )

    assert set(_problem(query, target).cursor()) == {(0, 1), (1, 0)}


def test_recursive_expression_uses_the_same_native_ri_cursor_with_anchor_zero():
    nested = _query(
        (0, 1),
        ((0, 1),),
        (
            _node(
                ("native_set", "atomic_number", (6,)),
                preflight=("atomic_number",),
            ),
            _node(
                ("native_set", "atomic_number", (8,)),
                preflight=("atomic_number",),
            ),
        ),
        (
            {
                "preflight": (),
                "leaves": (),
                "exclude_aromatic": False,
                "bond_kind": None,
            },
        ),
    )
    outer = _query(
        (0,),
        (),
        (_node(("atom_expr", _atom_expr("recursive", value=nested))),),
    )
    target = _target(
        (0, 1, 2, 3),
        ((0, 1), (2, 3)),
        (
            _atom({"atomic_number": 6}),
            _atom({"atomic_number": 8}),
            _atom({"atomic_number": 6}),
            _atom({"atomic_number": 6}),
        ),
        (_bond({}), _bond({})),
    )

    native_target = NativeTarget(target)
    problem = NativeProblem(outer, native_target)
    assert list(problem.cursor()) == [(0,)]
    assert list(problem.cursor()) == [(0,)]


def test_production_cursor_owns_problem_and_python_target_lifetimes():
    class Target:
        pass

    target_object = Target()
    target_reference = weakref.ref(target_object)
    query = _query((0,), (), (_node(("callable", lambda _obj: True)),))
    target = _target((7,), (), (_atom({}, target_object),))
    native_target = NativeTarget(target)
    problem = NativeProblem(query, native_target)
    cursor = problem.cursor()

    del native_target, problem, target, target_object
    gc.collect()
    assert target_reference() is not None
    assert list(cursor) == [(7,)]

    del cursor
    gc.collect()
    assert target_reference() is None
