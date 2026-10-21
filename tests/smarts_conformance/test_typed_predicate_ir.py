"""Typed predicate IR contracts for the active SMARTS parser."""

from dataclasses import FrozenInstanceError

import pytest

import hotpot as hp
from hotpot.cheminfo.search._predicate_ir import (
    _AtomExpression,
    _AtomField,
    _AtomOp,
    _BondExpression,
    _BondOp,
)
from hotpot.cheminfo.search.smarts import parse_bracket_atom


pytestmark = pytest.mark.smarts_core


def _atom_expression(smarts):
    return hp.Substructure.from_smarts(smarts).query_atoms[0].kwargs["predicate"]


def _walk(expression):
    yield expression
    for child in expression.children:
        yield from _walk(child)


@pytest.mark.parametrize(
    ("smarts", "expected_op"),
    (
        ("[*]", _AtomOp.TRUE),
        ("[#6]", _AtomOp.FIELD_EQ),
        ("[NG3-8]", _AtomOp.FIELD_IN),
        ("[NP3-5]", _AtomOp.PERIOD_IN),
        ("[D2]", _AtomOp.DEGREE_EQ),
        ("[X4]", _AtomOp.CONNECTIVITY_EQ),
        ("[v3]", _AtomOp.VALENCE_EQ),
        ("[R]", _AtomOp.RING_ANY),
        ("[R2]", _AtomOp.RING_COUNT_EQ),
        ("[r0]", _AtomOp.RING_NONE),
        ("[r6]", _AtomOp.RING_SIZE_EQ),
        ("[H2]", _AtomOp.TOTAL_H_EQ),
        ("[$([N])]", _AtomOp.RECURSIVE),
        ("[C;D1]", _AtomOp.AND),
        ("[C,N]", _AtomOp.OR),
        ("[!M]", _AtomOp.NOT),
    ),
)
def test_parser_emits_typed_atom_operations(smarts, expected_op):
    expression = _atom_expression(smarts)

    assert isinstance(expression, _AtomExpression)
    assert expression.op is expected_op


def test_all_supported_atom_field_primitives_are_typed():
    cases = {
        "[#6]": _AtomField.ATOMIC_NUMBER,
        "[N+]": _AtomField.FORMAL_CHARGE,
        "[a]": _AtomField.IS_AROMATIC,
        "[NG1]": _AtomField.GROUP,
        "[Ln]": _AtomField.IS_LANTHANIDE,
        "[An]": _AtomField.IS_ACTINIDE,
        "[M]": _AtomField.IS_METAL,
    }

    for smarts, expected_field in cases.items():
        fields = {
            node.field
            for node in _walk(_atom_expression(smarts))
            if node.op in {_AtomOp.FIELD_EQ, _AtomOp.FIELD_IN}
        }
        assert expected_field in fields


def test_atom_boolean_shape_repr_and_aromatic_hints_are_stable():
    expression = _atom_expression("[C,N;!R]")

    assert repr(expression) == (
        "(((C & aromatic=False) | (N & aromatic=False)) & !R)"
    )
    assert expression.op is _AtomOp.AND
    assert expression.children[0].op is _AtomOp.OR
    assert expression.children[1].op is _AtomOp.NOT
    assert _atom_expression("[c]").aromatic_states == frozenset({True})
    assert _atom_expression("[C]").aromatic_states == frozenset({False})
    assert _atom_expression("[c,C]").aromatic_states == frozenset({False, True})
    assert _atom_expression("[!c]").aromatic_states == frozenset({False, True})


def test_hydrogen_identity_ambiguity_and_extension_repr_are_stable():
    elemental_hydrogen = _atom_expression("[H]")
    hydrogen_count = _atom_expression("[C;H]")

    assert repr(elemental_hydrogen) == "(H & aromatic=False)"
    assert not any(node.op is _AtomOp.TOTAL_H_EQ for node in _walk(elemental_hydrogen))
    assert any(node.op is _AtomOp.TOTAL_H_EQ for node in _walk(hydrogen_count))
    assert repr(_atom_expression("[NP3]")) == "NP3-3"
    assert repr(_atom_expression("[NG1]")) == "NG1-1"


def test_typed_atom_boolean_evaluation_short_circuits_in_child_order():
    class Probe:
        def __init__(self):
            self.accessed = []

        @property
        def atomic_number(self):
            self.accessed.append("atomic_number")
            return 6

        @property
        def formal_charge(self):
            self.accessed.append("formal_charge")
            raise AssertionError("short-circuited child was evaluated")

    first_true = _AtomExpression(
        _AtomOp.FIELD_EQ,
        "first",
        field=_AtomField.ATOMIC_NUMBER,
        value=6,
    )
    first_false = _AtomExpression(
        _AtomOp.FIELD_EQ,
        "first",
        field=_AtomField.ATOMIC_NUMBER,
        value=7,
    )
    unreachable = _AtomExpression(
        _AtomOp.FIELD_EQ,
        "unreachable",
        field=_AtomField.FORMAL_CHARGE,
        value=0,
    )
    probe = Probe()

    assert _AtomExpression(
        _AtomOp.OR,
        "or",
        children=(first_true, unreachable),
    )(probe)
    assert probe.accessed == ["atomic_number"]

    probe.accessed.clear()
    assert not _AtomExpression(
        _AtomOp.AND,
        "and",
        children=(first_false, unreachable),
    )(probe)
    assert probe.accessed == ["atomic_number"]


def test_typed_nodes_are_frozen_but_keep_identity_equality():
    first = _atom_expression("[#6]")
    second = _atom_expression("[#6]")

    with pytest.raises(FrozenInstanceError):
        first.description = "changed"
    assert first is not second
    assert first != second


def test_public_bracket_result_remains_a_direct_callable_constraint():
    attrs = parse_bracket_atom("[#6;H3:7]")
    carbon = hp.read_mol("CC", "smi").atoms[0]

    assert list(attrs) == ["predicate"]
    assert isinstance(attrs["predicate"], _AtomExpression)
    assert attrs["predicate"](carbon)


def test_explicit_bond_shapes_stay_sets_and_guards():
    single = hp.Substructure.from_smarts("C-C").query_bonds[0]
    double = hp.Substructure.from_smarts("C=O").query_bonds[0]
    aromatic = hp.Substructure.from_smarts("c:c").query_bonds[0]
    any_bond = hp.Substructure.from_smarts("C~C").query_bonds[0]

    assert single.kwargs == {"bond_order": {1}}
    assert single._smarts_exclude_aromatic
    assert double.kwargs == {"bond_order": {2}}
    assert double._smarts_exclude_aromatic
    assert aromatic.kwargs == {"is_aromatic": {True}}
    assert any_bond.kwargs == {}


def test_multi_and_implicit_bond_predicates_are_typed_and_callable():
    multi = hp.Substructure.from_smarts("C=,:C").query_bonds[0].kwargs["predicate"]
    implicit = hp.Substructure.from_smarts("CC").query_bonds[0].kwargs["predicate"]
    single_bond = hp.read_mol("CC", "smi").bonds[0]
    double_bond = hp.read_mol("C=C", "smi").bonds[0]
    aromatic_bond = hp.read_mol("c1ccccc1", "smi").bonds[0]

    assert isinstance(multi, _BondExpression)
    assert multi.op is _BondOp.OR
    assert [child.op for child in multi.children] == [
        _BondOp.KIND_IS,
        _BondOp.AROMATIC,
    ]
    assert repr(multi) == "=,:"
    assert not multi(single_bond)
    assert multi(double_bond)
    assert multi(aromatic_bond)

    assert isinstance(implicit, _BondExpression)
    assert implicit.op is _BondOp.OR
    assert repr(implicit) == "single-or-aromatic"
    assert implicit(single_bond)
    assert not implicit(double_bond)
    assert implicit(aromatic_bond)


def test_parser_predicates_are_typed_while_programmatic_callable_stays_opaque():
    patterns = (
        "[#6;X4;H3]",
        "[C,N;!R]",
        "[#6;$([#6](=[#8])[#7])]",
        "C=,:O",
        "CC",
    )

    for pattern in patterns:
        query = hp.Substructure.from_smarts(pattern)
        predicate_values = [
            value
            for item in (*query.query_atoms, *query.query_bonds)
            for value in item.kwargs.values()
            if callable(value)
        ]
        assert predicate_values
        assert all(
            isinstance(value, (_AtomExpression, _BondExpression))
            for value in predicate_values
        )

    opaque = lambda atom: atom.atomic_number == 6
    query_atom = hp.QueryAtom(any_keyword=opaque)
    assert query_atom.kwargs["any_keyword"] is opaque
    assert not isinstance(opaque, (_AtomExpression, _BondExpression))


def test_replacing_or_removing_query_predicate_is_observed_immediately():
    query_atom = hp.Substructure.from_smarts("[#6]").query_atoms[0]
    carbon = hp.read_mol("C", "smi").atoms[0]

    assert query_atom.match(carbon)
    query_atom.kwargs["predicate"] = lambda atom: atom.atomic_number == 8
    assert not query_atom.match(carbon)
    query_atom.kwargs.pop("predicate")
    assert query_atom.match(carbon)


def test_recursive_typed_leaf_keeps_semantics_and_target_mutation_invalidation():
    atom_molecule = hp.read_mol("CN", "smi")
    atom_expression = _atom_expression("[N;$([N+])]")
    recursive = next(
        node for node in _walk(atom_expression) if node.op is _AtomOp.RECURSIVE
    )
    nitrogen = atom_molecule.atoms[1]

    assert recursive.value.semantics is hp.SmartsSemantics.FULL_GRAPH
    assert not atom_expression(nitrogen)
    nitrogen.formal_charge = 1
    assert atom_expression(nitrogen)

    bond_molecule = hp.read_mol("C=O", "smi")
    bond_expression = _atom_expression("[#6;$([#6]=[#8])]")
    carbon = bond_molecule.atoms[0]

    assert bond_expression(carbon)
    bond_molecule.bonds[0].bond_order = 1
    assert not bond_expression(carbon)
