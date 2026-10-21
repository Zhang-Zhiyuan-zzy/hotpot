"""SMARTS parsing for Hotpot's native substructure search.

The parser intentionally targets the graph and query objects implemented by
``hotpot.cheminfo.search``. It does not translate through another chemistry
toolkit. Standard atom expressions are compiled to small predicates and the
Hotpot coordination-chemistry extensions are handled by the same machinery.
"""

from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from weakref import WeakKeyDictionary

from ..core import BondKind
from ._predicate_ir import (
    _AtomExpression,
    _AtomField,
    _AtomOp,
    _BondExpression,
    _BondOp,
)
from ._smarts_syntax import (
    AROMATIC_LOWER,
    TokenType,
    atomic_number as _atomic_number,
    tokenize,
)
from .errors import SmartsSyntaxError, UnsupportedSmartsError
from .search import QueryAtom, Substructure
from .semantics import SmartsSemantics, resolve_smarts_semantics

METAL_TOKEN = "M"
LANTHANIDE_TOKEN = "Ln"
ACTINIDE_TOKEN = "An"
PERIOD_PREFIX = "NP"
GROUP_PREFIX = "NG"

ANY_ATOM_TOKEN = "*"
SINGLE_BOND_TOKEN = "-"
DOUBLE_BOND_TOKEN = "="
TRIPLE_BOND_TOKEN = "#"
AROMATIC_BOND_TOKEN = ":"
ANY_BOND_TOKEN = "~"
UP_BOND_TOKEN = "/"
DOWN_BOND_TOKEN = "\\"


class BondOrder(int, Enum):
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3


class AromaticFlag(Enum):
    AROMATIC = "arom"
    NON_AROMATIC = "non_arom"
    UNSET = "unset"


class _RecursiveEvaluator:
    """Anchored recursive SMARTS with per-molecule/per-atom result caching."""

    def __init__(self, smarts: str, semantics: SmartsSemantics):
        self.smarts = smarts
        self.semantics = semantics
        self.substructure = substructure_from_smarts(smarts, semantics=semantics)
        self._cache = WeakKeyDictionary()

    def __call__(self, atom: object) -> bool:
        state_signature = _molecule_search_signature(atom.mol)
        cached = self._cache.get(atom.mol)
        if cached is None or cached[0] != state_signature:
            from .native import _compile_problem

            problem = _compile_problem(self.substructure, atom.mol)
            query_keys = tuple(problem.query_keys)
            anchor_position = query_keys.index(0)
            anchored_indices = {
                mapping[anchor_position]
                for mapping in problem.cursor()
            }
            cached = (
                state_signature,
                {
                    candidate.idx: candidate.idx in anchored_indices
                    for candidate in atom.mol.atoms
                },
            )
            self._cache[atom.mol] = cached
        mol_cache = cached[1]
        return mol_cache[atom.idx]

    def __repr__(self) -> str:
        return f"$({self.smarts})"


def _molecule_search_signature(mol: object) -> Tuple[object, ...]:
    """Return the molecular state consumed by SMARTS matching predicates."""
    atoms = tuple(
        (
            atom.idx,
            atom.atomic_number,
            atom.formal_charge,
            atom.is_aromatic,
            atom.implicit_hydrogens,
        )
        for atom in mol.atoms
    )
    bonds = tuple(
        sorted(
            (
                min(bond.a1idx, bond.a2idx),
                max(bond.a1idx, bond.a2idx),
                bond.bond_order,
                bond.bond_kind,
                bond.is_aromatic,
            )
            for bond in mol.bonds
        )
    )
    return atoms, bonds


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
    return {"predicate": expression}


def substructure_from_smarts(
    smarts: str,
    *,
    semantics: SmartsSemantics = SmartsSemantics.FULL_GRAPH,
) -> Substructure:
    """Build a :class:`Substructure` for Hotpot's native graph executor."""
    semantics = resolve_smarts_semantics(semantics)
    tokens = tokenize(smarts)
    if not tokens:
        raise SmartsSyntaxError("SMARTS query must contain at least one atom")

    substructure = Substructure()
    ring_anchors: Dict[str, List[Tuple[int, Optional[Dict[str, object]]]]] = (
        defaultdict(list)
    )
    branch_stack: List[Tuple[int, int]] = []
    last_atom_index: Optional[int] = None
    pending_bond_attrs: Optional[Dict[str, object]] = None
    component_has_atom = False

    for token_type, token_text in tokens:
        if token_type in (TokenType.ATOM, TokenType.BRACKET):
            query_atom = _create_query_atom_from_token(
                substructure, token_type, token_text, semantics
            )
            substructure.add_atom(query_atom)
            current_index = len(substructure.query_atoms) - 1
            if last_atom_index is not None:
                bond_attrs = _infer_bond_attrs(
                    substructure.query_atoms[last_atom_index],
                    query_atom,
                    pending_bond_attrs,
                )
                substructure.add_bond(last_atom_index, current_index, **bond_attrs)
            last_atom_index = current_index
            pending_bond_attrs = None
            component_has_atom = True
        elif token_type == TokenType.BOND:
            if last_atom_index is None:
                raise SmartsSyntaxError(
                    f"Bond expression must follow an atom: {smarts}"
                )
            if pending_bond_attrs is not None:
                raise SmartsSyntaxError(
                    f"Consecutive bond expressions in SMARTS: {smarts}"
                )
            pending_bond_attrs = _bond_attrs_for_symbol(token_text)
        elif token_type == TokenType.BRANCH_L:
            if last_atom_index is None:
                raise SmartsSyntaxError(f"Branch '(' must follow an atom: {smarts}")
            if pending_bond_attrs is not None:
                raise SmartsSyntaxError(
                    f"Bond expression cannot precede a branch: {smarts}"
                )
            branch_stack.append((last_atom_index, len(substructure.query_atoms)))
        elif token_type == TokenType.BRANCH_R:
            if not branch_stack:
                raise SmartsSyntaxError(f"Unmatched ')' in SMARTS: {smarts}")
            branch_anchor, atom_count = branch_stack.pop()
            if len(substructure.query_atoms) == atom_count:
                raise SmartsSyntaxError(f"Empty branch in SMARTS: {smarts}")
            if pending_bond_attrs is not None:
                raise SmartsSyntaxError(
                    f"Bond expression must be followed by an atom: {smarts}"
                )
            last_atom_index = branch_anchor
            pending_bond_attrs = None
        elif token_type == TokenType.RING:
            if last_atom_index is None:
                raise SmartsSyntaxError(f"Ring label must follow an atom: {smarts}")
            pending_bond_attrs = _connect_or_anchor_ring(
                substructure,
                ring_anchors,
                token_text,
                last_atom_index,
                pending_bond_attrs,
            )
        elif token_type == TokenType.DOT:
            if branch_stack:
                raise SmartsSyntaxError(f"Dot is not allowed inside a branch: {smarts}")
            if not component_has_atom:
                raise SmartsSyntaxError(f"Empty component in SMARTS: {smarts}")
            if pending_bond_attrs is not None:
                raise SmartsSyntaxError(
                    f"Bond expression must be followed by an atom: {smarts}"
                )
            last_atom_index = None
            pending_bond_attrs = None
            component_has_atom = False

    if branch_stack:
        raise SmartsSyntaxError(f"Unclosed '(' in SMARTS: {smarts}")
    if pending_bond_attrs is not None:
        raise SmartsSyntaxError(
            f"Bond expression must be followed by an atom: {smarts}"
        )
    if any(anchors for anchors in ring_anchors.values()):
        labels = [label for label, anchors in ring_anchors.items() if anchors]
        raise SmartsSyntaxError(f"Unclosed ring label(s) {labels} in SMARTS: {smarts}")
    if not component_has_atom:
        raise SmartsSyntaxError(f"Empty component in SMARTS: {smarts}")
    return substructure


def _compile_bracket_atom(
    expr_text: str,
    semantics: SmartsSemantics,
) -> Tuple[_AtomExpression, Optional[int], Optional[bool]]:
    if not expr_text.startswith("[") or not expr_text.endswith("]"):
        raise SmartsSyntaxError(
            f"Bracket atom must start with '[' and end with ']': {expr_text!r}"
        )
    inner, map_number = _extract_atom_map(expr_text[1:-1].strip())
    if not inner:
        raise SmartsSyntaxError("Empty SMARTS bracket atom")
    expression = _parse_low_and(inner, semantics)
    aromatic_hint = (
        next(iter(expression.aromatic_states))
        if len(expression.aromatic_states) == 1
        else None
    )
    return expression, map_number, aromatic_hint


def _extract_atom_map(expr: str) -> Tuple[str, Optional[int]]:
    paren_depth = 0
    square_depth = 0
    index = 0
    while index < len(expr):
        char = expr[index]
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == ":" and paren_depth == square_depth == 0:
            match = re.fullmatch(r":(\d+)\s*", expr[index:])
            if match:
                return expr[:index].rstrip(), int(match.group(1))
        index += 1
    return expr, None


def _parse_low_and(
    expr: str, semantics: SmartsSemantics
) -> _AtomExpression:
    parts = _split_top_level(expr, ";")
    identities = [_contains_non_hydrogen_identity(part) for part in parts]
    return _combine_and(
        [
            _parse_or(
                part,
                semantics,
                any(identities[:index] + identities[index + 1 :]),
            )
            for index, part in enumerate(parts)
        ]
    )


def _parse_or(
    expr: str,
    semantics: SmartsSemantics,
    outer_identity: bool = False,
) -> _AtomExpression:
    return _combine_or(
        [
            _parse_high_and(
                part,
                semantics,
                outer_identity or _contains_non_hydrogen_identity(part),
            )
            for part in _split_top_level(expr, ",")
        ]
    )


def _parse_high_and(
    expr: str,
    semantics: SmartsSemantics,
    identity_context: bool = False,
) -> _AtomExpression:
    chunks = _split_top_level(expr, "&")
    expressions = []
    identity_seen = identity_context
    for chunk in chunks:
        index = 0
        while index < len(chunk):
            if chunk[index].isspace():
                index += 1
                continue
            negate = False
            while index < len(chunk) and chunk[index] == "!":
                negate = not negate
                index += 1
            if index == len(chunk):
                raise SmartsSyntaxError(
                    f"Missing operand after '!' in atom expression {expr!r}"
                )
            primitive, index, is_identity = _parse_atom_primitive(
                chunk, index, identity_seen, semantics
            )
            if negate:
                primitive = _negate(primitive)
            expressions.append(primitive)
            identity_seen = identity_seen or is_identity
    return _combine_and(expressions)


def _contains_non_hydrogen_identity(expr: str) -> bool:
    """Whether an AND branch establishes an atom identity independently of H."""
    index = 0
    while index < len(expr):
        char = expr[index]
        candidate = expr[index : index + 2]
        if (
            len(candidate) == 2
            and (
                (candidate[0].isupper() and candidate[1].islower())
                or candidate in AROMATIC_LOWER
            )
            and _atomic_number(candidate.capitalize())
        ):
            return True
        if expr.startswith("$(", index):
            index = _find_balanced_parenthesis(expr, index + 1) + 1
        elif char in "!,;&+@":
            index += 1
        elif char in "-=":
            index += 1
        elif char == "#" or char == "*":
            return True
        elif char in "DXvHRr":
            index += 1
            while index < len(expr) and expr[index].isdigit():
                index += 1
        elif char.isdigit():
            while index < len(expr) and expr[index].isdigit():
                index += 1
        elif char.isalpha():
            return True
        else:
            index += 1
    return False


def _split_top_level(expr: str, separator: str) -> List[str]:
    parts = []
    start = 0
    paren_depth = 0
    square_depth = 0
    for index, char in enumerate(expr):
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth -= 1
        elif char == separator and paren_depth == square_depth == 0:
            parts.append(expr[start:index].strip())
            start = index + 1
    parts.append(expr[start:].strip())
    if any(not part for part in parts):
        raise SmartsSyntaxError(
            f"Empty operand around {separator!r} in atom expression {expr!r}"
        )
    return parts


def _parse_atom_primitive(
    expr: str,
    index: int,
    identity_seen: bool,
    semantics: SmartsSemantics,
) -> Tuple[_AtomExpression, int, bool]:
    if expr.startswith("$(", index):
        end = _find_balanced_parenthesis(expr, index + 1)
        recursive_smarts = expr[index + 2 : end]
        if not recursive_smarts.strip():
            raise SmartsSyntaxError("Recursive SMARTS body must contain an atom")
        recursive = _RecursiveEvaluator(recursive_smarts, semantics)
        return (
            _AtomExpression(
                _AtomOp.RECURSIVE,
                repr(recursive),
                value=recursive,
            ),
            end + 1,
            False,
        )

    if index >= len(expr):
        raise SmartsSyntaxError(f"Missing atom primitive in {expr!r}")

    candidate = expr[index : index + 2]
    if (
        len(candidate) == 2
        and (
            (candidate[0].isupper() and candidate[1].islower())
            or candidate in AROMATIC_LOWER
        )
        and _atomic_number(candidate.capitalize())
    ):
        atomic_number = _atomic_number(candidate.capitalize())
        aromatic = candidate.islower()
        return (
            _combine_and(
                [
                    _equals("atomic_number", atomic_number, candidate),
                    _equals(
                        "is_aromatic",
                        aromatic,
                        f"aromatic={aromatic}",
                        {aromatic},
                    ),
                ]
            ),
            index + 2,
            True,
        )

    special = _parse_hotpot_extension(expr, index)
    if special is not None:
        return special

    char = expr[index]
    if char in "hx":
        raise UnsupportedSmartsError(
            f"SMARTS atom primitive {char!r} is not implemented"
        )
    if char == ANY_ATOM_TOKEN:
        return _always_true("*"), index + 1, True
    if char == "#":
        match = re.match(r"#(\d+)", expr[index:])
        if match is None:
            raise SmartsSyntaxError(f"Malformed atomic number in {expr!r}")
        number = int(match.group(1))
        return (
            _equals("atomic_number", number, f"#{number}"),
            index + len(match.group(0)),
            True,
        )
    if char in "+-":
        charge, end = _read_charge(expr, index)
        return _equals("formal_charge", charge, f"charge={charge}"), end, False
    if char == "@":
        raise UnsupportedSmartsError(
            "SMARTS atom chirality '@'/'@@' is not implemented"
        )
    if char in "DXvRr":
        return _parse_numeric_atom_primitive(expr, index, semantics)
    if char == "H" and (
        identity_seen or (index + 1 < len(expr) and expr[index + 1].isdigit())
    ):
        return _parse_hydrogen_primitive(expr, index)
    if char == "a":
        return _equals("is_aromatic", True, "a", {True}), index + 1, True
    if char == "A":
        return _equals("is_aromatic", False, "A", {False}), index + 1, True
    if char.isdigit():
        raise UnsupportedSmartsError("SMARTS isotope matching is not implemented")
    if char.isalpha():
        symbol, end = _read_element_in_expression(expr, index)
        atomic_number = _atomic_number(symbol.capitalize())
        if not atomic_number:
            raise SmartsSyntaxError(f"Unknown atom primitive {symbol!r} in {expr!r}")
        aromatic = symbol.islower()
        return (
            _combine_and(
                [
                    _equals("atomic_number", atomic_number, symbol),
                    _equals(
                        "is_aromatic",
                        aromatic,
                        f"aromatic={aromatic}",
                        {aromatic},
                    ),
                ]
            ),
            end,
            True,
        )
    raise SmartsSyntaxError(f"Unsupported atom primitive at {expr[index:]!r}")


def _parse_hotpot_extension(
    expr: str, index: int
) -> Optional[Tuple[_AtomExpression, int, bool]]:
    remaining = expr[index:]
    if remaining.startswith(PERIOD_PREFIX):
        match = re.match(r"NP(\d+)(?:-(\d+))?", remaining)
        if match is None:
            raise SmartsSyntaxError("Hotpot period extension requires a period number")
        values = _integer_range(match.group(1), match.group(2))
        if not values or min(values) < 1 or max(values) > 7:
            raise SmartsSyntaxError("Hotpot period extension must be within 1-7")
        return (
            _AtomExpression(
                _AtomOp.PERIOD_IN,
                f"NP{min(values)}-{max(values)}",
                value=frozenset(values),
            ),
            index + len(match.group(0)),
            True,
        )
    if remaining.startswith(GROUP_PREFIX):
        match = re.match(r"NG(\d+)(?:-(\d+))?", remaining)
        if match is None:
            raise SmartsSyntaxError("Hotpot group extension requires a group number")
        values = _integer_range(match.group(1), match.group(2))
        if not values or min(values) < 1 or max(values) > 18:
            raise SmartsSyntaxError("Hotpot group extension must be within 1-18")
        return (
            _in_values("group", values, f"NG{min(values)}-{max(values)}"),
            index + len(match.group(0)),
            True,
        )
    for token, attr in (
        (LANTHANIDE_TOKEN, "is_lanthanide"),
        (ACTINIDE_TOKEN, "is_actinide"),
    ):
        if remaining.startswith(token):
            return _equals(attr, True, token), index + len(token), True
    if remaining.startswith(METAL_TOKEN):
        next_char = remaining[1:2]
        candidate = remaining[:2]
        if not (next_char.islower() and _atomic_number(candidate)):
            return _equals("is_metal", True, METAL_TOKEN), index + 1, True
    return None


def _parse_numeric_atom_primitive(
    expr: str, index: int, semantics: SmartsSemantics
) -> Tuple[_AtomExpression, int, bool]:
    code = expr[index]
    match = re.match(r"[DXvRr](\d*)", expr[index:])
    digits = match.group(1)
    end = index + len(match.group(0))
    if code == "D":
        value = int(digits) if digits else 1
        return (
            _AtomExpression(
                _AtomOp.DEGREE_EQ,
                f"D{value}",
                value=value,
                semantics=semantics,
            ),
            end,
            False,
        )
    if code == "X":
        value = int(digits) if digits else 1
        return (
            _AtomExpression(
                _AtomOp.CONNECTIVITY_EQ,
                f"X{value}",
                value=value,
                semantics=semantics,
            ),
            end,
            False,
        )
    if code == "v":
        value = int(digits) if digits else 1
        return (
            _AtomExpression(
                _AtomOp.VALENCE_EQ,
                f"v{value}",
                value=value,
                semantics=semantics,
            ),
            end,
            False,
        )
    if code == "R":
        if not digits:
            return (
                _AtomExpression(
                    _AtomOp.RING_ANY,
                    "R",
                    semantics=semantics,
                ),
                end,
                False,
            )
        count = int(digits)
        return (
            _AtomExpression(
                _AtomOp.RING_COUNT_EQ,
                f"R{count}",
                value=count,
                semantics=semantics,
            ),
            end,
            False,
        )
    if not digits:
        return (
            _AtomExpression(
                _AtomOp.RING_ANY,
                "r",
                semantics=semantics,
            ),
            end,
            False,
        )
    size = int(digits)
    if size == 0:
        return (
            _AtomExpression(
                _AtomOp.RING_NONE,
                "r0",
                semantics=semantics,
            ),
            end,
            False,
        )
    return (
        _AtomExpression(
            _AtomOp.RING_SIZE_EQ,
            f"r{size}",
            value=size,
            semantics=semantics,
        ),
        end,
        False,
    )


def _parse_hydrogen_primitive(
    expr: str, index: int
) -> Tuple[_AtomExpression, int, bool]:
    match = re.match(r"H(\d*)", expr[index:])
    count = int(match.group(1)) if match.group(1) else 1
    end = index + len(match.group(0))
    return (
        _AtomExpression(
            _AtomOp.TOTAL_H_EQ,
            f"H{count}",
            value=count,
        ),
        end,
        False,
    )


def _find_balanced_parenthesis(expr: str, opening_index: int) -> int:
    depth = 1
    index = opening_index + 1
    while index < len(expr):
        if expr[index] == "(":
            depth += 1
        elif expr[index] == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise SmartsSyntaxError(f"Unclosed recursive SMARTS in {expr!r}")


def _read_element_in_expression(expr: str, index: int) -> Tuple[str, int]:
    if index + 1 < len(expr) and expr[index + 1].islower():
        candidate = expr[index : index + 2]
        if _atomic_number(candidate.capitalize()):
            return candidate, index + 2
    return expr[index], index + 1


def _read_charge(expr: str, index: int) -> Tuple[int, int]:
    sign_char = expr[index]
    sign = 1 if sign_char == "+" else -1
    end = index
    while end < len(expr) and expr[end] == sign_char:
        end += 1
    repeated = end - index
    digit_start = end
    while end < len(expr) and expr[end].isdigit():
        end += 1
    if end > digit_start:
        return sign * int(expr[digit_start:end]), end
    return sign * repeated, end


def _integer_range(start: str, end: Optional[str]) -> Set[int]:
    first = int(start)
    last = int(end) if end is not None else first
    low, high = sorted((first, last))
    return set(range(low, high + 1))


def _always_true(description: str) -> _AtomExpression:
    return _AtomExpression(_AtomOp.TRUE, description)


def _equals(
    attr: str,
    value: object,
    description: str,
    aromatic_states: Set[bool] = frozenset((False, True)),
) -> _AtomExpression:
    return _AtomExpression(
        _AtomOp.FIELD_EQ,
        description,
        field=_AtomField(attr),
        value=value,
        aromatic_states=frozenset(aromatic_states),
    )


def _in_values(attr: str, values: Set[int], description: str) -> _AtomExpression:
    return _AtomExpression(
        _AtomOp.FIELD_IN,
        description,
        field=_AtomField(attr),
        value=frozenset(values),
    )


def _combine_and(expressions: List[_AtomExpression]) -> _AtomExpression:
    if not expressions:
        return _always_true("*")
    if len(expressions) == 1:
        return expressions[0]
    states = {False, True}
    for expression in expressions:
        states.intersection_update(expression.aromatic_states)
    return _AtomExpression(
        _AtomOp.AND,
        "(" + " & ".join(map(repr, expressions)) + ")",
        children=tuple(expressions),
        aromatic_states=frozenset(states),
    )


def _combine_or(expressions: List[_AtomExpression]) -> _AtomExpression:
    if len(expressions) == 1:
        return expressions[0]
    states = set()
    for expression in expressions:
        states.update(expression.aromatic_states)
    return _AtomExpression(
        _AtomOp.OR,
        "(" + " | ".join(map(repr, expressions)) + ")",
        children=tuple(expressions),
        aromatic_states=frozenset(states),
    )


def _negate(expression: _AtomExpression) -> _AtomExpression:
    return _AtomExpression(
        _AtomOp.NOT,
        f"!{expression!r}",
        children=(expression,),
    )


def _create_query_atom_from_token(
    substructure: Substructure,
    token_type: TokenType,
    text: str,
    semantics: SmartsSemantics,
) -> QueryAtom:
    if token_type == TokenType.BRACKET:
        expression, map_number, aromatic_hint = _compile_bracket_atom(
            text, semantics
        )
        query_atom = QueryAtom(
            sub=substructure, map_number=map_number, predicate=expression
        )
        query_atom._smarts_aromatic = aromatic_hint
        return query_atom
    return _create_query_atom_from_symbol(substructure, text)


def _create_query_atom_from_symbol(
    substructure: Substructure, symbol: str
) -> QueryAtom:
    if symbol == ANY_ATOM_TOKEN:
        query_atom = QueryAtom(sub=substructure)
        query_atom._smarts_aromatic = None
        return query_atom
    if symbol == "a":
        query_atom = QueryAtom(sub=substructure, is_aromatic={True})
        query_atom._smarts_aromatic = True
        return query_atom
    if symbol == "A":
        query_atom = QueryAtom(sub=substructure, is_aromatic={False})
        query_atom._smarts_aromatic = False
        return query_atom
    if symbol.lower() in AROMATIC_LOWER and symbol.islower():
        atomic_number = _atomic_number(symbol.capitalize())
        query_atom = QueryAtom(
            sub=substructure,
            atomic_number={atomic_number},
            is_aromatic={True},
        )
        query_atom._smarts_aromatic = True
        return query_atom
    atomic_number = _atomic_number(symbol)
    if not atomic_number:
        raise SmartsSyntaxError(f"Unknown atom symbol in SMARTS: {symbol!r}")
    query_atom = QueryAtom(
        sub=substructure,
        atomic_number={atomic_number},
        is_aromatic={False},
    )
    query_atom._smarts_aromatic = False
    return query_atom


def _bond_attrs_for_symbol(symbol: str) -> Dict[str, object]:
    alternatives = symbol.split(",")
    if any(token in {UP_BOND_TOKEN, DOWN_BOND_TOKEN} for token in alternatives):
        raise UnsupportedSmartsError(
            "SMARTS directional bonds '/' and '\\' are not implemented"
        )
    if len(alternatives) == 1:
        token = alternatives[0]
        if token == SINGLE_BOND_TOKEN:
            return {
                "bond_order": {BondOrder.SINGLE.value},
                "_smarts_bond_kind": BondKind.SINGLE,
                "_smarts_exclude_aromatic": True,
            }
        if token == DOUBLE_BOND_TOKEN:
            return {
                "bond_order": {BondOrder.DOUBLE.value},
                "_smarts_bond_kind": BondKind.DOUBLE,
                "_smarts_exclude_aromatic": True,
            }
        if token == TRIPLE_BOND_TOKEN:
            return {
                "bond_order": {BondOrder.TRIPLE.value},
                "_smarts_bond_kind": BondKind.TRIPLE,
                "_smarts_exclude_aromatic": True,
            }
        if token == AROMATIC_BOND_TOKEN:
            return {"is_aromatic": {True}}
        if token == ANY_BOND_TOKEN:
            return {}
        raise SmartsSyntaxError(f"Unsupported bond symbol: {symbol}")
    predicates = tuple(_bond_predicate(token) for token in alternatives)
    return {
        "predicate": _BondExpression(
            _BondOp.OR,
            symbol,
            children=predicates,
        )
    }


def _bond_predicate(symbol: str) -> _BondExpression:
    if symbol in {UP_BOND_TOKEN, DOWN_BOND_TOKEN}:
        raise UnsupportedSmartsError(
            "SMARTS directional bonds '/' and '\\' are not implemented"
        )
    if symbol == SINGLE_BOND_TOKEN:
        return _BondExpression(_BondOp.KIND_IS, symbol, kind=BondKind.SINGLE)
    if symbol == DOUBLE_BOND_TOKEN:
        return _BondExpression(_BondOp.KIND_IS, symbol, kind=BondKind.DOUBLE)
    if symbol == TRIPLE_BOND_TOKEN:
        return _BondExpression(_BondOp.KIND_IS, symbol, kind=BondKind.TRIPLE)
    if symbol == AROMATIC_BOND_TOKEN:
        return _BondExpression(_BondOp.AROMATIC, symbol)
    if symbol == ANY_BOND_TOKEN:
        return _BondExpression(_BondOp.TRUE, symbol)
    raise SmartsSyntaxError(f"Unsupported bond symbol: {symbol}")


def _get_aromatic_flag(query_atom: QueryAtom) -> AromaticFlag:
    hint = getattr(query_atom, "_smarts_aromatic", None)
    if hint is True:
        return AromaticFlag.AROMATIC
    if hint is False:
        return AromaticFlag.NON_AROMATIC
    aromatic_values = query_atom.kwargs.get("is_aromatic")
    if aromatic_values == {True}:
        return AromaticFlag.AROMATIC
    if aromatic_values == {False}:
        return AromaticFlag.NON_AROMATIC
    return AromaticFlag.UNSET


def _infer_bond_attrs(
    first_atom: QueryAtom,
    second_atom: QueryAtom,
    pending_bond_attrs: Optional[Dict[str, object]],
) -> Dict[str, object]:
    if pending_bond_attrs is not None:
        return pending_bond_attrs
    first_state = _get_aromatic_flag(first_atom)
    second_state = _get_aromatic_flag(second_atom)
    if first_state is second_state is AromaticFlag.AROMATIC:
        return {"is_aromatic": {True}}
    return {
        "predicate": _BondExpression(
            _BondOp.OR,
            "single-or-aromatic",
            children=(
                _BondExpression(
                    _BondOp.KIND_IS,
                    SINGLE_BOND_TOKEN,
                    kind=BondKind.SINGLE,
                ),
                _BondExpression(_BondOp.AROMATIC, AROMATIC_BOND_TOKEN),
            ),
        )
    }


def _connect_or_anchor_ring(
    substructure: Substructure,
    ring_anchors: Dict[str, List[Tuple[int, Optional[Dict[str, object]]]]],
    ring_label: str,
    current_atom_index: int,
    pending_bond_attrs: Optional[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    anchors = ring_anchors[ring_label]
    if not anchors:
        anchors.append((current_atom_index, pending_bond_attrs))
        return None
    start_index, opening_bond_attrs = anchors.pop()
    if start_index == current_atom_index:
        raise SmartsSyntaxError(f"Ring label {ring_label} creates a self-loop")
    if any(
        {bond.a1idx, bond.a2idx} == {start_index, current_atom_index}
        for bond in substructure.query_bonds
    ):
        raise SmartsSyntaxError(f"Ring label {ring_label} creates a duplicate edge")
    if opening_bond_attrs is not None and pending_bond_attrs is not None:
        raise SmartsSyntaxError(
            f"Ring bond specified at both ends of label {ring_label}"
        )
    explicit_attrs = pending_bond_attrs or opening_bond_attrs
    bond_attrs = _infer_bond_attrs(
        substructure.query_atoms[start_index],
        substructure.query_atoms[current_atom_index],
        explicit_attrs,
    )
    substructure.add_bond(start_index, current_atom_index, **bond_attrs)
    return None
