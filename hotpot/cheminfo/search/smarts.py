"""SMARTS parsing for Hotpot's NetworkX substructure search.

The parser intentionally targets the graph and query objects implemented by
``hotpot.cheminfo.search``. It does not translate through another chemistry
toolkit. Standard atom expressions are compiled to small predicates and the
Hotpot coordination-chemistry extensions are handled by the same machinery.
"""

from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple
from weakref import WeakKeyDictionary

from networkx.algorithms import isomorphism
from openbabel import openbabel as ob

from .search import QueryAtom, Searcher, Substructure


class TokenType(str, Enum):
    ATOM = "ATOM"
    BRACKET = "BRACKET"
    BOND = "BOND"
    BRANCH_L = "BRANCH_L"
    BRANCH_R = "BRANCH_R"
    RING = "RING"
    DOT = "DOT"


BOND_CHARS: Set[str] = {"-", "=", "#", ":", "/", "\\", "~"}
AROMATIC_LOWER: Set[str] = {"b", "c", "n", "o", "p", "s", "se", "as"}

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


class _Predicate:
    """A named callable which keeps query representations deterministic."""

    def __init__(self, func: Callable[[object], bool], description: str):
        self._func = func
        self.description = description

    def __call__(self, obj: object) -> bool:
        return bool(self._func(obj))

    def __repr__(self) -> str:
        return self.description


class _AtomExpression(_Predicate):
    def __init__(
        self,
        func: Callable[[object], bool],
        description: str,
        aromatic_states: Set[bool] = frozenset((False, True)),
    ):
        super().__init__(func, description)
        self.aromatic_states = frozenset(aromatic_states)


class _RecursivePredicate:
    """Anchored recursive SMARTS with per-molecule/per-atom result caching."""

    def __init__(self, smarts: str):
        self.smarts = smarts
        self.substructure = substructure_from_smarts(smarts)
        self._cache = WeakKeyDictionary()

    def __call__(self, atom: object) -> bool:
        state_signature = _molecule_search_signature(atom.mol)
        cached = self._cache.get(atom.mol)
        if cached is None or cached[0] != state_signature:
            matcher = isomorphism.GraphMatcher(
                atom.mol.atom_bond_graph,
                self.substructure.construct_graph(),
                node_match=Searcher._node_match,
                edge_match=Searcher._edge_match,
            )
            anchored_indices = {
                mol_index
                for mapping in matcher.subgraph_monomorphisms_iter()
                for mol_index, query_index in mapping.items()
                if query_index == 0
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
                bond.is_aromatic,
            )
            for bond in mol.bonds
        )
    )
    return atoms, bonds


def tokenize(smarts: str) -> List[Tuple[TokenType, str]]:
    """Tokenize graph-level SMARTS syntax."""
    tokens: List[Tuple[TokenType, str]] = []
    index = 0
    while index < len(smarts):
        char = smarts[index]
        if char.isspace():
            index += 1
        elif char == "[":
            closing = _find_closing_bracket_index(smarts, index)
            tokens.append((TokenType.BRACKET, smarts[index : closing + 1]))
            index = closing + 1
        elif char == "(":
            tokens.append((TokenType.BRANCH_L, char))
            index += 1
        elif char == ")":
            tokens.append((TokenType.BRANCH_R, char))
            index += 1
        elif char == ".":
            tokens.append((TokenType.DOT, char))
            index += 1
        elif char in BOND_CHARS:
            bond, index = _read_bond_token(smarts, index)
            tokens.append((TokenType.BOND, bond))
        elif char.isdigit() or _is_multi_digit_ring_start(smarts, index):
            ring, index = _read_ring_token(smarts, index)
            tokens.append((TokenType.RING, ring))
        elif char.isalpha() or char == ANY_ATOM_TOKEN:
            atom, index = _read_atom_token(smarts, index)
            tokens.append((TokenType.ATOM, atom))
        else:
            raise ValueError(
                f"Unsupported character in SMARTS: {char!r} at position {index}"
            )
    return tokens


def parse_bracket_atom(expr_text: str) -> Dict[str, object]:
    """Compile a bracket atom into constraints accepted by ``QueryAtom``.

    Atom-map labels are graph metadata rather than matching constraints and are
    therefore absent from this return value. They are attached to
    ``QueryAtom.map_number`` by :func:`substructure_from_smarts`.
    """
    expression, _, _ = _compile_bracket_atom(expr_text)
    return {"predicate": expression}


def substructure_from_smarts(smarts: str) -> Substructure:
    """Build a :class:`Substructure` without leaving Hotpot's graph backend."""
    tokens = tokenize(smarts)
    substructure = Substructure()
    ring_anchors: Dict[
        str, List[Tuple[int, Optional[Dict[str, object]]]]
    ] = defaultdict(list)
    branch_stack: List[Tuple[int, int]] = []
    last_atom_index: Optional[int] = None
    pending_bond_attrs: Optional[Dict[str, object]] = None

    for token_type, token_text in tokens:
        if token_type in (TokenType.ATOM, TokenType.BRACKET):
            query_atom = _create_query_atom_from_token(
                substructure, token_type, token_text
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
        elif token_type == TokenType.BOND:
            if pending_bond_attrs is not None:
                raise ValueError(f"Consecutive bond expressions in SMARTS: {smarts}")
            pending_bond_attrs = _bond_attrs_for_symbol(token_text)
        elif token_type == TokenType.BRANCH_L:
            if last_atom_index is None:
                raise ValueError(f"Branch '(' must follow an atom: {smarts}")
            branch_stack.append((last_atom_index, len(substructure.query_atoms)))
        elif token_type == TokenType.BRANCH_R:
            if not branch_stack:
                raise ValueError(f"Unmatched ')' in SMARTS: {smarts}")
            branch_anchor, atom_count = branch_stack.pop()
            if len(substructure.query_atoms) == atom_count:
                raise ValueError(f"Empty branch in SMARTS: {smarts}")
            if pending_bond_attrs is not None:
                raise ValueError(f"Bond expression must be followed by an atom: {smarts}")
            last_atom_index = branch_anchor
            pending_bond_attrs = None
        elif token_type == TokenType.RING:
            if last_atom_index is None:
                raise ValueError(f"Ring label must follow an atom: {smarts}")
            pending_bond_attrs = _connect_or_anchor_ring(
                substructure,
                ring_anchors,
                token_text,
                last_atom_index,
                pending_bond_attrs,
            )
        elif token_type == TokenType.DOT:
            if branch_stack:
                raise ValueError(f"Dot is not allowed inside a branch: {smarts}")
            if pending_bond_attrs is not None:
                raise ValueError(f"Bond expression must be followed by an atom: {smarts}")
            last_atom_index = None
            pending_bond_attrs = None

    if branch_stack:
        raise ValueError(f"Unclosed '(' in SMARTS: {smarts}")
    if pending_bond_attrs is not None:
        raise ValueError(f"Bond expression must be followed by an atom: {smarts}")
    if any(anchors for anchors in ring_anchors.values()):
        labels = [label for label, anchors in ring_anchors.items() if anchors]
        raise ValueError(f"Unclosed ring label(s) {labels} in SMARTS: {smarts}")
    return substructure


def _find_closing_bracket_index(text: str, start_index: int) -> int:
    square_depth = 1
    paren_depth = 0
    index = start_index + 1
    while index < len(text):
        char = text[index]
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
            if paren_depth < 0:
                raise ValueError(f"Unmatched ')' in bracket atom: {text}")
        elif char == "[":
            if paren_depth == 0:
                raise ValueError(
                    f"Nested '[' is only valid in recursive SMARTS: {text}"
                )
            square_depth += 1
        elif char == "]":
            square_depth -= 1
            if square_depth == 0:
                return index
        index += 1
    raise ValueError(f"Unclosed '[' in SMARTS: {text}")


def _read_bond_token(smarts: str, index: int) -> Tuple[str, int]:
    parts = [smarts[index]]
    index += 1
    while index < len(smarts) and smarts[index] == ",":
        if index + 1 >= len(smarts) or smarts[index + 1] not in BOND_CHARS:
            raise ValueError(f"Malformed bond OR expression in SMARTS: {smarts}")
        parts.extend((",", smarts[index + 1]))
        index += 2
    return "".join(parts), index


def _is_multi_digit_ring_start(smarts: str, index: int) -> bool:
    return (
        smarts[index] == "%"
        and len(smarts[index + 1 : index + 3]) == 2
        and smarts[index + 1 : index + 3].isdigit()
    )


def _read_ring_token(smarts: str, index: int) -> Tuple[str, int]:
    if smarts[index] == "%":
        end = index + 1
        while end < len(smarts) and smarts[end].isdigit():
            end += 1
        return smarts[index:end], end
    return smarts[index], index + 1


def _read_atom_token(smarts: str, index: int) -> Tuple[str, int]:
    if smarts[index] == ANY_ATOM_TOKEN:
        return ANY_ATOM_TOKEN, index + 1
    char = smarts[index]
    if char.isupper() and index + 1 < len(smarts):
        candidate = smarts[index : index + 2]
        if smarts[index + 1].islower() and ob.GetAtomicNum(candidate):
            return candidate, index + 2
    if char.islower() and index + 1 < len(smarts):
        candidate = smarts[index : index + 2]
        if candidate in AROMATIC_LOWER:
            return candidate, index + 2
    return char, index + 1


def _compile_bracket_atom(
    expr_text: str,
) -> Tuple[_AtomExpression, Optional[int], Optional[bool]]:
    if not expr_text.startswith("[") or not expr_text.endswith("]"):
        raise ValueError(
            f"Bracket atom must start with '[' and end with ']': {expr_text!r}"
        )
    inner, map_number = _extract_atom_map(expr_text[1:-1].strip())
    if not inner:
        raise ValueError("Empty SMARTS bracket atom")
    expression = _parse_low_and(inner)
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


def _parse_low_and(expr: str) -> _AtomExpression:
    parts = _split_top_level(expr, ";")
    identities = [_contains_non_hydrogen_identity(part) for part in parts]
    return _combine_and(
        [
            _parse_or(part, any(identities[:index] + identities[index + 1 :]))
            for index, part in enumerate(parts)
        ]
    )


def _parse_or(expr: str, outer_identity: bool = False) -> _AtomExpression:
    return _combine_or(
        [
            _parse_high_and(
                part, outer_identity or _contains_non_hydrogen_identity(part)
            )
            for part in _split_top_level(expr, ",")
        ]
    )


def _parse_high_and(expr: str, identity_context: bool = False) -> _AtomExpression:
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
            primitive, index, is_identity = _parse_atom_primitive(
                chunk, index, identity_seen
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
        raise ValueError(f"Empty operand around {separator!r} in atom expression {expr!r}")
    return parts


def _parse_atom_primitive(
    expr: str, index: int, identity_seen: bool
) -> Tuple[_AtomExpression, int, bool]:
    if expr.startswith("$(", index):
        end = _find_balanced_parenthesis(expr, index + 1)
        recursive_smarts = expr[index + 2 : end]
        recursive = _RecursivePredicate(recursive_smarts)
        return _AtomExpression(recursive, repr(recursive)), end + 1, False

    special = _parse_hotpot_extension(expr, index)
    if special is not None:
        return special

    char = expr[index]
    if char == ANY_ATOM_TOKEN:
        return _always_true("*"), index + 1, True
    if char == "#":
        match = re.match(r"#(\d+)", expr[index:])
        if match is None:
            raise ValueError(f"Malformed atomic number in {expr!r}")
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
        raise NotImplementedError("SMARTS atom chirality '@'/'@@' is not implemented")
    if char in "DXvRr":
        return _parse_numeric_atom_primitive(expr, index)
    if char == "H" and (
        identity_seen or (index + 1 < len(expr) and expr[index + 1].isdigit())
    ):
        return _parse_hydrogen_primitive(expr, index)
    if char == "a":
        return _equals("is_aromatic", True, "a", {True}), index + 1, True
    if char == "A":
        return _equals("is_aromatic", False, "A", {False}), index + 1, True
    if char.isdigit():
        raise NotImplementedError("SMARTS isotope matching is not implemented")
    if char.isalpha():
        symbol, end = _read_element_in_expression(expr, index)
        atomic_number = ob.GetAtomicNum(symbol.capitalize())
        if not atomic_number:
            raise ValueError(f"Unknown atom primitive {symbol!r} in {expr!r}")
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
    raise ValueError(f"Unsupported atom primitive at {expr[index:]!r}")


def _parse_hotpot_extension(
    expr: str, index: int
) -> Optional[Tuple[_AtomExpression, int, bool]]:
    remaining = expr[index:]
    if remaining.startswith(PERIOD_PREFIX):
        match = re.match(r"NP(\d+)(?:-(\d+))?", remaining)
        if match is not None:
            values = _integer_range(match.group(1), match.group(2))
            return (
                _atom_predicate(
                    lambda atom, allowed=frozenset(values): _period_number(
                        atom.atomic_number
                    )
                    in allowed,
                    f"NP{min(values)}-{max(values)}",
                ),
                index + len(match.group(0)),
                True,
            )
    if remaining.startswith(GROUP_PREFIX):
        match = re.match(r"NG(\d+)(?:-(\d+))?", remaining)
        if match is not None:
            values = _integer_range(match.group(1), match.group(2))
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
        if not (next_char.islower() and ob.GetAtomicNum(candidate)):
            return _equals("is_metal", True, METAL_TOKEN), index + 1, True
    return None


def _parse_numeric_atom_primitive(
    expr: str, index: int
) -> Tuple[_AtomExpression, int, bool]:
    code = expr[index]
    match = re.match(r"[DXvRr](\d*)", expr[index:])
    digits = match.group(1)
    end = index + len(match.group(0))
    if code == "D":
        value = int(digits) if digits else 1
        return (
            _atom_predicate(lambda atom, n=value: len(atom.neighbours) == n, f"D{value}"),
            end,
            False,
        )
    if code == "X":
        value = int(digits) if digits else 1
        return (
            _atom_predicate(
                lambda atom, n=value: len(atom.neighbours) + atom.implicit_hydrogens == n,
                f"X{value}",
            ),
            end,
            False,
        )
    if code == "v":
        value = int(digits) if digits else 1
        return (
            _atom_predicate(
                lambda atom, n=value: atom.sum_bond_orders + atom.implicit_hydrogens == n,
                f"v{value}",
            ),
            end,
            False,
        )
    if code == "R":
        if not digits:
            return _equals("in_ring", True, "R"), end, False
        count = int(digits)
        return (
            _atom_predicate(lambda atom, n=count: len(atom.rings) == n, f"R{count}"),
            end,
            False,
        )
    if not digits:
        return _equals("in_ring", True, "r"), end, False
    size = int(digits)
    return (
        _atom_predicate(
            lambda atom, n=size: any(len(ring) == n for ring in atom.rings),
            f"r{size}",
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
        _atom_predicate(
            lambda atom, n=count: atom.explicit_hydrogens + atom.implicit_hydrogens == n,
            f"H{count}",
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
    raise ValueError(f"Unclosed recursive SMARTS in {expr!r}")


def _read_element_in_expression(expr: str, index: int) -> Tuple[str, int]:
    if index + 1 < len(expr) and expr[index + 1].islower():
        candidate = expr[index : index + 2]
        if ob.GetAtomicNum(candidate.capitalize()):
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


def _period_number(atomic_number: int) -> int:
    for period, upper_bound in enumerate((2, 10, 18, 36, 54, 86, 118), start=1):
        if atomic_number <= upper_bound:
            return period
    return 0


def _atom_predicate(
    func: Callable[[object], bool], description: str
) -> _AtomExpression:
    return _AtomExpression(func, description)


def _always_true(description: str) -> _AtomExpression:
    return _AtomExpression(lambda atom: True, description)


def _equals(
    attr: str,
    value: object,
    description: str,
    aromatic_states: Set[bool] = frozenset((False, True)),
) -> _AtomExpression:
    return _AtomExpression(
        lambda atom, name=attr, expected=value: getattr(atom, name) == expected,
        description,
        aromatic_states,
    )


def _in_values(attr: str, values: Set[int], description: str) -> _AtomExpression:
    accepted = frozenset(values)
    return _AtomExpression(
        lambda atom, name=attr, allowed=accepted: getattr(atom, name) in allowed,
        description,
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
        lambda atom, predicates=tuple(expressions): all(p(atom) for p in predicates),
        "(" + " & ".join(map(repr, expressions)) + ")",
        states,
    )


def _combine_or(expressions: List[_AtomExpression]) -> _AtomExpression:
    if len(expressions) == 1:
        return expressions[0]
    states = set()
    for expression in expressions:
        states.update(expression.aromatic_states)
    return _AtomExpression(
        lambda atom, predicates=tuple(expressions): any(p(atom) for p in predicates),
        "(" + " | ".join(map(repr, expressions)) + ")",
        states,
    )


def _negate(expression: _AtomExpression) -> _AtomExpression:
    return _AtomExpression(
        lambda atom, predicate=expression: not predicate(atom),
        f"!{expression!r}",
    )


def _create_query_atom_from_token(
    substructure: Substructure, token_type: TokenType, text: str
) -> QueryAtom:
    if token_type == TokenType.BRACKET:
        expression, map_number, aromatic_hint = _compile_bracket_atom(text)
        query_atom = QueryAtom(
            sub=substructure, map_number=map_number, predicate=expression
        )
        query_atom._smarts_aromatic = aromatic_hint
        return query_atom
    return _create_query_atom_from_symbol(substructure, text)


def _create_query_atom_from_symbol(substructure: Substructure, symbol: str) -> QueryAtom:
    if symbol == ANY_ATOM_TOKEN:
        query_atom = QueryAtom(sub=substructure)
        query_atom._smarts_aromatic = None
        return query_atom
    if symbol.lower() in AROMATIC_LOWER and symbol.islower():
        atomic_number = ob.GetAtomicNum(symbol.capitalize())
        query_atom = QueryAtom(
            sub=substructure,
            atomic_number={atomic_number},
            is_aromatic={True},
        )
        query_atom._smarts_aromatic = True
        return query_atom
    atomic_number = ob.GetAtomicNum(symbol)
    if not atomic_number:
        raise ValueError(f"Unknown atom symbol in SMARTS: {symbol!r}")
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
        raise NotImplementedError("SMARTS directional bonds '/' and '\\' are not implemented")
    if len(alternatives) == 1:
        token = alternatives[0]
        if token == SINGLE_BOND_TOKEN:
            return {"bond_order": {BondOrder.SINGLE.value}}
        if token == DOUBLE_BOND_TOKEN:
            return {"bond_order": {BondOrder.DOUBLE.value}}
        if token == TRIPLE_BOND_TOKEN:
            return {"bond_order": {BondOrder.TRIPLE.value}}
        if token == AROMATIC_BOND_TOKEN:
            return {"is_aromatic": {True}}
        if token == ANY_BOND_TOKEN:
            return {}
        raise ValueError(f"Unsupported bond symbol: {symbol}")
    predicates = tuple(_bond_predicate(token) for token in alternatives)
    return {
        "predicate": _Predicate(
            lambda bond, choices=predicates: any(choice(bond) for choice in choices),
            symbol,
        )
    }


def _bond_predicate(symbol: str) -> Callable[[object], bool]:
    if symbol in {UP_BOND_TOKEN, DOWN_BOND_TOKEN}:
        raise NotImplementedError("SMARTS directional bonds '/' and '\\' are not implemented")
    if symbol == SINGLE_BOND_TOKEN:
        return lambda bond: bond.bond_order == BondOrder.SINGLE.value
    if symbol == DOUBLE_BOND_TOKEN:
        return lambda bond: bond.bond_order == BondOrder.DOUBLE.value
    if symbol == TRIPLE_BOND_TOKEN:
        return lambda bond: bond.bond_order == BondOrder.TRIPLE.value
    if symbol == AROMATIC_BOND_TOKEN:
        return lambda bond: bond.is_aromatic
    if symbol == ANY_BOND_TOKEN:
        return lambda bond: True
    raise ValueError(f"Unsupported bond symbol: {symbol}")


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
        "predicate": _Predicate(
            lambda bond: bond.bond_order == BondOrder.SINGLE.value or bond.is_aromatic,
            "single-or-aromatic",
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
    if opening_bond_attrs is not None and pending_bond_attrs is not None:
        raise ValueError(f"Ring bond specified at both ends of label {ring_label}")
    explicit_attrs = pending_bond_attrs or opening_bond_attrs
    bond_attrs = _infer_bond_attrs(
        substructure.query_atoms[start_index],
        substructure.query_atoms[current_atom_index],
        explicit_attrs,
    )
    substructure.add_bond(start_index, current_atom_index, **bond_attrs)
    return None
