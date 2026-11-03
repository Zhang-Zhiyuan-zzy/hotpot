"""Allocate current query objects and execute anchored recursive SMARTS."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from ...structure.substructure import SmartsSemantics, _AtomExpression, _AtomOp
from ...rules.chemistry.substructure.semantics import resolve_smarts_semantics
from ...rules.chemistry.substructure.smarts import _compile_bracket_atom, parse_smarts
from .search import QueryAtom, Substructure


class _RecursiveEvaluator:
    """Anchored recursive SMARTS evaluated in a fresh query/target task."""

    def __init__(self, smarts: str, semantics: SmartsSemantics):
        self.smarts = smarts
        self.semantics = semantics
        self.substructure = substructure_from_smarts(smarts, semantics=semantics)

    def __call__(self, atom: object) -> bool:
        from ..chemistry import _require_aromatic_representation, _require_implicit_hydrogens
        from .native import _compile_problem

        with atom.mol._mutation_lock:
            _require_implicit_hydrogens(atom.mol)
            _require_aromatic_representation(atom.mol)
        problem = _compile_problem(self.substructure, atom.mol)
        anchor_position = tuple(problem.query_keys).index(0)
        anchored_indices = {mapping[anchor_position] for mapping in problem.cursor()}
        return atom.idx in anchored_indices

    def __repr__(self) -> str:
        return f"$({self.smarts})"


def parse_bracket_atom(
    expr_text: str,
    *,
    semantics: SmartsSemantics = SmartsSemantics.FULL_GRAPH,
) -> Dict[str, object]:
    """Compile a bracket atom into constraints accepted by ``QueryAtom``.

    Atom-map labels are graph metadata rather than matching constraints and are
    therefore absent from this return value. They are attached to
    ``QueryAtom.map_number`` by :func:`substructure_from_smarts`.
    """
    expression, _, _ = _compile_bracket_atom(
        expr_text, resolve_smarts_semantics(semantics)
    )
    return {"predicate": _bind_recursive(expression, resolve_smarts_semantics(semantics))}


def substructure_from_smarts(
    smarts: str,
    *,
    semantics: SmartsSemantics = SmartsSemantics.FULL_GRAPH,
) -> Substructure:
    """Build a :class:`Substructure` for Hotpot's native graph executor."""
    semantics = resolve_smarts_semantics(semantics)
    atoms, edges = parse_smarts(smarts, semantics=semantics)
    substructure = Substructure()
    for kwargs, map_number, aromatic_hint in atoms:
        kwargs = {
            name: _bind_recursive(value, semantics) if type(value) is _AtomExpression else value
            for name, value in kwargs.items()
        }
        atom = QueryAtom(sub=substructure, map_number=map_number, **kwargs)
        atom._smarts_aromatic = aromatic_hint
        substructure.add_atom(atom)
    for first, second, attrs in edges:
        substructure.add_bond(first, second, **attrs)
    return substructure


def _bind_recursive(expression, semantics):
    if expression.op is _AtomOp.RECURSIVE:
        return replace(expression, value=_RecursiveEvaluator(expression.value, semantics))
    if expression.children:
        return replace(expression, children=tuple(_bind_recursive(child, semantics) for child in expression.children))
    return expression
