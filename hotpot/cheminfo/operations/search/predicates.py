"""Bind lazy carrier reads to pure SMARTS expression comparisons."""

from ...structure.substructure import _AtomOp, _BondOp
from ...rules.chemistry.substructure import predicates as rules
from ..chemistry import _atom_non_graph_hydrogens


def match_atom_expression(expression, atom):
    rings = ()

    def values(leaf, ring_index=None):
        nonlocal rings
        op = leaf.op
        if op in {_AtomOp.FIELD_EQ, _AtomOp.FIELD_IN}:
            return getattr(atom, leaf.field.value)
        if op is _AtomOp.PERIOD_IN:
            return atom.atomic_number
        if op is _AtomOp.DEGREE_EQ:
            return leaf.semantics.degree(atom)
        if op is _AtomOp.CONNECTIVITY_EQ:
            return leaf.semantics.connectivity(atom)
        if op is _AtomOp.VALENCE_EQ:
            return leaf.semantics.valence(atom)
        if op in {_AtomOp.RING_ANY, _AtomOp.RING_COUNT_EQ, _AtomOp.RING_NONE}:
            return len(leaf.semantics.rings_for(atom))
        if op is _AtomOp.RING_SIZE_EQ:
            if ring_index is None:
                rings = leaf.semantics.rings_for(atom)
                return len(rings)
            return len(rings[ring_index])
        if op is _AtomOp.TOTAL_H_EQ:
            return atom.explicit_hydrogens, _atom_non_graph_hydrogens(atom)
        if op is _AtomOp.RECURSIVE:
            return leaf.value(atom)
    return rules.match_atom_expression(expression, values)


def match_bond_expression(expression, bond):
    def values(leaf):
        if leaf.op is _BondOp.KIND_IS:
            return bond.bond_kind
        if leaf.op is _BondOp.AROMATIC:
            return bond.is_aromatic
    return rules.match_bond_expression(expression, values)
