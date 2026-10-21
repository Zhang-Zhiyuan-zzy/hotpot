"""Private compiler for native substructure-search query programs."""

from __future__ import annotations

from collections.abc import Callable

from ..core import BondKind
from ._predicate_ir import (
    _AtomExpression,
    _AtomOp,
    _BondExpression,
)


_ATOM_NATIVE_SET_TYPES = {
    "atomic_number": (int,),
    "is_aromatic": (bool,),
}
_BOND_NATIVE_SET_TYPES = {
    "bond_order": (int, float),
    "is_aromatic": (bool,),
}
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _compile_query(substructure):
    """Snapshot a ``Substructure`` as the native executor's query schema."""
    nodes_by_key = {}
    for atom in substructure.query_atoms:
        nodes_by_key[atom.idx] = atom

    bonds_by_edge = {}
    for bond in substructure.query_bonds:
        edge = tuple(sorted((bond.a1idx, bond.a2idx)))
        bonds_by_edge[edge] = bond

    keys = tuple(sorted(nodes_by_key))
    edges = tuple(sorted(bonds_by_edge))
    return {
        "keys": keys,
        "edges": edges,
        "nodes": tuple(_compile_node(nodes_by_key[key]) for key in keys),
        "bonds": tuple(_compile_bond(bonds_by_edge[edge]) for edge in edges),
    }


def _compile_node(query_atom):
    return _compile_constraints(
        query_atom.kwargs,
        _ATOM_NATIVE_SET_TYPES,
        _AtomExpression,
        _encode_atom_expression,
        "atom_expr",
    )


def _compile_bond(query_bond):
    program = _compile_constraints(
        query_bond.kwargs,
        _BOND_NATIVE_SET_TYPES,
        _BondExpression,
        _encode_bond_expression,
        "bond_expr",
    )
    kind = query_bond._smarts_bond_kind
    return {
        "preflight": program["preflight"],
        "leaves": program["leaves"],
        "exclude_aromatic": bool(query_bond._smarts_exclude_aromatic),
        "bond_kind": kind.value if isinstance(kind, BondKind) else None,
    }


def _compile_constraints(
    kwargs, native_set_types, expression_type, encode_expression, expression_tag
):
    preflight = []
    leaves = []
    for attr, values in kwargs.items():
        if isinstance(values, Callable):
            if type(values) is expression_type:
                leaves.append((expression_tag, encode_expression(values)))
            else:
                leaves.append(("callable", values))
            continue

        preflight.append(attr)
        expected_types = native_set_types.get(attr)
        if expected_types is not None and _is_native_scalar_set(
            values, expected_types
        ):
            leaves.append(("native_set", attr, tuple(sorted(values))))
        else:
            leaves.append(("host_set", attr, values))
    return {"preflight": tuple(preflight), "leaves": tuple(leaves)}


def _is_native_scalar_set(values, expected_types) -> bool:
    if not values:
        return True
    return all(
        type(value) in expected_types
        and (
            type(value) is not int
            or _INT64_MIN <= value <= _INT64_MAX
        )
        for value in values
    )


def _encode_atom_expression(expression):
    value = expression.value
    if expression.op in {_AtomOp.FIELD_IN, _AtomOp.PERIOD_IN}:
        value = tuple(sorted(value))
    elif expression.op is _AtomOp.RECURSIVE:
        value = _compile_query(value.substructure)

    return {
        "op": expression.op.value,
        "field": expression.field.value if expression.field is not None else None,
        "value": value,
        "semantics": (
            expression.semantics.value
            if expression.semantics is not None
            else None
        ),
        "children": tuple(
            _encode_atom_expression(child) for child in expression.children
        ),
    }


def _encode_bond_expression(expression):
    return {
        "op": expression.op.value,
        "kind": expression.kind.value if expression.kind is not None else None,
        "children": tuple(
            _encode_bond_expression(child) for child in expression.children
        ),
    }


def _query_object_requirements(query_spec):
    """Return whether atom and bond Python handles are needed by a query."""
    atom_objects = any(
        _program_needs_object(program) for program in query_spec["nodes"]
    )
    bond_objects = any(
        _program_needs_object(program) for program in query_spec["bonds"]
    )

    for program in query_spec["nodes"]:
        for leaf in program["leaves"]:
            if leaf[0] == "atom_expr":
                nested_atom, nested_bond = _atom_expression_requirements(leaf[1])
                atom_objects = atom_objects or nested_atom
                bond_objects = bond_objects or nested_bond
    return atom_objects, bond_objects


def _query_derived_requirements(query_spec):
    """Return semantic-view fields and bond facts consumed by a query."""
    views = {"full": [], "ligand": []}
    atom_fields = []
    needs_bond_aromatic = False
    needs_total_h = False
    for program in query_spec["nodes"]:
        for leaf in program["leaves"]:
            if leaf[0] == "atom_expr":
                (
                    nested_aromatic,
                    nested_total_h,
                ) = _append_atom_derived_requirements(
                    leaf[1], views, atom_fields
                )
                needs_bond_aromatic = (
                    needs_bond_aromatic or nested_aromatic
                )
                needs_total_h = needs_total_h or nested_total_h
    for program in query_spec["bonds"]:
        needs_bond_aromatic = (
            needs_bond_aromatic
            or program["exclude_aromatic"]
            or any(
                leaf[0] in {"native_set", "host_set"}
                and leaf[1] == "is_aromatic"
                for leaf in program["leaves"]
            )
        )
        for leaf in program["leaves"]:
            if leaf[0] == "bond_expr" and _bond_expression_needs_aromatic(
                leaf[1]
            ):
                needs_bond_aromatic = True
    return {
        "full": tuple(views["full"]),
        "ligand": tuple(views["ligand"]),
        "atom_fields": tuple(atom_fields),
        "bond_aromatic": needs_bond_aromatic,
        "total_h": needs_total_h,
    }


def _append_atom_derived_requirements(expression, views, atom_fields):
    operation_to_field = {
        _AtomOp.DEGREE_EQ.value: "degree",
        _AtomOp.CONNECTIVITY_EQ.value: "connectivity",
        _AtomOp.VALENCE_EQ.value: "valence",
        _AtomOp.RING_ANY.value: "ring_count",
        _AtomOp.RING_COUNT_EQ.value: "ring_count",
        _AtomOp.RING_NONE.value: "ring_count",
        _AtomOp.RING_SIZE_EQ.value: "ring_sizes",
    }
    field = operation_to_field.get(expression["op"])
    if field is not None:
        view = (
            "ligand"
            if expression["semantics"] == "ligand_skeleton"
            else "full"
        )
        if field not in views[view]:
            views[view].append(field)

    if expression["op"] in {
        _AtomOp.FIELD_EQ.value,
        _AtomOp.FIELD_IN.value,
    }:
        field = expression["field"]
        if field not in atom_fields:
            atom_fields.append(field)

    needs_bond_aromatic = False
    needs_total_h = expression["op"] == _AtomOp.TOTAL_H_EQ.value
    if expression["op"] == _AtomOp.RECURSIVE.value:
        nested = _query_derived_requirements(expression["value"])
        for view in ("full", "ligand"):
            for nested_field in nested[view]:
                if nested_field not in views[view]:
                    views[view].append(nested_field)
        for nested_field in nested["atom_fields"]:
            if nested_field not in atom_fields:
                atom_fields.append(nested_field)
        needs_bond_aromatic = nested["bond_aromatic"]
        needs_total_h = nested["total_h"]
    for child in expression["children"]:
        child_aromatic, child_total_h = _append_atom_derived_requirements(
            child, views, atom_fields
        )
        needs_bond_aromatic = child_aromatic or needs_bond_aromatic
        needs_total_h = child_total_h or needs_total_h
    return needs_bond_aromatic, needs_total_h


def _bond_expression_needs_aromatic(expression):
    return expression["op"] == "aromatic" or any(
        _bond_expression_needs_aromatic(child)
        for child in expression["children"]
    )


def _program_needs_object(program) -> bool:
    return any(leaf[0] in {"host_set", "callable"} for leaf in program["leaves"])


def _atom_expression_requirements(expression):
    atom_objects = False
    bond_objects = False
    if expression["op"] == _AtomOp.RECURSIVE.value:
        atom_objects, bond_objects = _query_object_requirements(expression["value"])
    for child in expression["children"]:
        child_atom, child_bond = _atom_expression_requirements(child)
        atom_objects = atom_objects or child_atom
        bond_objects = bond_objects or child_bond
    return atom_objects, bond_objects


__all__ = ()
