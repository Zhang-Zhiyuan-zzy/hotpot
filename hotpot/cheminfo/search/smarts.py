"""
@File Name:        smarts
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/12/7 12:30
@Project:          Hotpot
"""
from __future__ import annotations
import re
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

from openbabel import openbabel as ob

from .search import QueryAtom, Substructure


class TokenType(str, Enum):
    ATOM = "ATOM"
    BRACKET = "BRACKET"
    BOND = "BOND"
    BRANCH_L = "BRANCH_L"
    BRANCH_R = "BRANCH_R"
    RING = "RING"


BOND_CHARS: Set[str] = {"-", "=", "#", ":", "/", "\\", "~"}

AROMATIC_LOWER: Set[str] = set("bcnops")

RING_OPENERS: str = "([{"
RING_CLOSERS: str = ")]}"

NEGATION_PREFIX: str = "!"
AND_SEPARATORS: str = ";&"

METAL_TOKEN: str = "M"
NON_METAL_TOKEN: str = "!M"
LANTHANIDE_TOKEN: str = "Ln"
ACTINIDE_TOKEN: str = "An"

PERIOD_PREFIX: str = "NP"
GROUP_PREFIX: str = "NG"

RING_FLAG_TOKEN: str = "R"
RING_SIZE_TOKEN: str = "r"
DEGREE_TOKEN: str = "D"
VALENCE_TOKEN: str = "v"
CONNECTIVITY_TOKEN: str = "X"
AROMATIC_ANY_TOKEN: str = "a"
ALIPHATIC_ANY_TOKEN: str = "A"
HYDROGEN_TOKEN: str = "H"

ATOMIC_NUMBER_PREFIX: str = "#"

RECURSIVE_SMARTS_PREFIX: str = "$("

ANY_ATOM_TOKEN: str = "*"

AROMATIC_BOND_TOKEN: str = ":"
ANY_BOND_TOKEN: str = "~"

SINGLE_BOND_TOKEN: str = "-"
DOUBLE_BOND_TOKEN: str = "="
TRIPLE_BOND_TOKEN: str = "#"
UP_BOND_TOKEN: str = "/"
DOWN_BOND_TOKEN: str = "\\"


class BondOrder(int, Enum):
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3


class AromaticFlag(Enum):
    AROMATIC = "arom"
    NON_AROMATIC = "non_arom"
    UNSET = "unset"


def tokenize(smarts: str) -> List[Tuple[TokenType, str]]:
    """
    Tokenize a SMARTS pattern into a flat token stream.

    Args:
        smarts: SMARTS pattern string.

    Returns:
        Ordered list of (TokenType, text) tuples.
    """
    tokens: List[Tuple[TokenType, str]] = []
    index = 0
    size = len(smarts)

    while index < size:
        char = smarts[index]

        if char.isspace():
            index += 1
            continue

        if char == "[":
            closing_index = _find_closing_bracket_index(smarts, index)
            tokens.append((TokenType.BRACKET, smarts[index:closing_index + 1]))
            index = closing_index + 1
            continue

        if char == "(":
            tokens.append((TokenType.BRANCH_L, char))
            index += 1
            continue

        if char == ")":
            tokens.append((TokenType.BRANCH_R, char))
            index += 1
            continue

        if char in BOND_CHARS:
            tokens.append((TokenType.BOND, char))
            index += 1
            continue

        if char.isdigit() or _is_multi_digit_ring_start(smarts, index):
            ring_token, index = _read_ring_token(smarts, index)
            tokens.append((TokenType.RING, ring_token))
            continue

        if char.isalpha():
            atom_token, index = _read_atom_token(smarts, index)
            tokens.append((TokenType.ATOM, atom_token))
            continue

        raise ValueError(f"Unsupported character in SMARTS: '{char}' at position {index}")

    return tokens


def parse_bracket_atom(expr_text: str) -> Dict[str, object]:
    """
    Parse a full bracket atom expression like "[C;H2]" into attribute constraints.

    Args:
        expr_text: Bracket expression including the square brackets.

    Returns:
        Dictionary of attribute constraints that can be passed into QueryAtom.
    """
    if not expr_text.startswith("[") or not expr_text.endswith("]"):
        raise ValueError(f"Bracket atom must start with '[' and end with ']': {expr_text!r}")

    expr = expr_text[1:-1].strip()

    if RECURSIVE_SMARTS_PREFIX in expr:
        raise NotImplementedError("Recursive SMARTS '$(' is not supported")

    if "," not in expr:
        return _parse_and_block(expr)

    return _parse_or_block(expr)


def substructure_from_smarts(smarts: str) -> Substructure:
    """
    Build a Substructure object from a SMARTS pattern.

    Args:
        smarts: SMARTS pattern string.

    Returns:
        Substructure instance containing query atoms and bonds.
    """
    tokens = tokenize(smarts)
    substructure = Substructure()
    ring_anchors: Dict[str, List[int]] = defaultdict(list)
    branch_stack: List[int] = []
    last_atom_index: Optional[int] = None
    pending_bond_attrs: Optional[Dict[str, object]] = None

    for token_type, token_text in tokens:
        if token_type in (TokenType.ATOM, TokenType.BRACKET):
            query_atom = _create_query_atom_from_token(substructure, token_type, token_text)
            substructure.add_atom(query_atom)
            current_index = len(substructure.query_atoms) - 1

            if last_atom_index is not None:
                first_atom = substructure.query_atoms[last_atom_index]
                second_atom = query_atom
                bond_attrs = _infer_bond_attrs(first_atom, second_atom, pending_bond_attrs)
                substructure.add_bond(last_atom_index, current_index, **bond_attrs)

            last_atom_index = current_index
            pending_bond_attrs = None
            continue

        if token_type == TokenType.BOND:
            pending_bond_attrs = _bond_attrs_for_symbol(token_text)
            continue

        if token_type == TokenType.BRANCH_L:
            if last_atom_index is None:
                raise ValueError(f"Branch '(' must follow an atom: {smarts}")
            branch_stack.append(last_atom_index)
            continue

        if token_type == TokenType.BRANCH_R:
            if not branch_stack:
                raise ValueError(f"Unmatched ')' in SMARTS: {smarts}")
            last_atom_index = branch_stack.pop()
            pending_bond_attrs = None
            continue

        if token_type == TokenType.RING:
            if last_atom_index is None:
                raise ValueError(f"Ring digit must follow an atom: {smarts}")
            _connect_or_anchor_ring(
                substructure=substructure,
                ring_anchors=ring_anchors,
                ring_label=token_text,
                current_atom_index=last_atom_index,
            )
            continue

        raise ValueError(f"Unexpected token type {token_type} in SMARTS parser")

    if branch_stack:
        raise ValueError(f"Unclosed '(' in SMARTS: {smarts}")

    for label, indices in ring_anchors.items():
        if indices:
            raise ValueError(f"Unclosed ring digit {label} in SMARTS: {smarts}")

    return substructure


def _find_closing_bracket_index(text: str, start_index: int) -> int:
    depth = 1
    index = start_index + 1
    size = len(text)

    while index < size and depth > 0:
        char = text[index]
        if char == "[":
            raise ValueError(f"Nested '[' is not allowed in bracket atom: {text[start_index:index + 1]}")
        if char == "]":
            depth -= 1
            if depth == 0:
                return index
        index += 1

    raise ValueError(f"Unclosed '[' in SMARTS: {text}")


def _is_multi_digit_ring_start(smarts: str, index: int) -> bool:
    if smarts[index] != "%":
        return False
    lookahead = smarts[index + 1 : index + 3]
    return len(lookahead) == 2 and lookahead.isdigit()


def _read_ring_token(smarts: str, index: int) -> Tuple[str, int]:
    if smarts[index] == "%":
        next_index = index + 1
        size = len(smarts)
        while next_index < size and smarts[next_index].isdigit():
            next_index += 1
        return smarts[index:next_index], next_index

    return smarts[index], index + 1


def _read_atom_token(smarts: str, index: int) -> Tuple[str, int]:
    char = smarts[index]
    size = len(smarts)

    if char.isupper() and index + 1 < size and smarts[index + 1].islower():
        return smarts[index : index + 2], index + 2

    return char, index + 1


def _parse_or_block(expr: str) -> Dict[str, object]:
    parts = _split_top_level(expr, {","})
    atomic_numbers: Set[int] = set()
    common_attrs: Dict[str, object] = {}

    for part in parts:
        primitive_attrs = _parse_and_block(part)
        keys = set(primitive_attrs)
        unsupported_keys = keys - {"atomic_number", "is_aromatic"}

        if unsupported_keys:
            raise NotImplementedError(
                f"Only simple element OR is supported, got: [{expr}]"
            )

        atomic_number = primitive_attrs.get("atomic_number")
        if atomic_number is not None:
            if isinstance(atomic_number, set):
                atomic_numbers.update(atomic_number)
            else:
                atomic_numbers.add(int(atomic_number))

        if "is_aromatic" in primitive_attrs:
            previous_aromaticity = common_attrs.get("is_aromatic")
            current_aromaticity = primitive_attrs["is_aromatic"]
            if previous_aromaticity is not None and previous_aromaticity != current_aromaticity:
                raise NotImplementedError(f"Inconsistent aromatic flags in OR list: [{expr}]")
            common_attrs["is_aromatic"] = current_aromaticity

    if not atomic_numbers:
        raise NotImplementedError(f"Unsupported OR expression: [{expr}]")

    common_attrs["atomic_number"] = atomic_numbers
    return common_attrs


def _parse_and_block(expr: str) -> Dict[str, object]:
    parts = _split_top_level(expr, set(AND_SEPARATORS))
    merged: Dict[str, object] = {}

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith(NEGATION_PREFIX):
            raise NotImplementedError(f"Negation with '!' is not supported: [{expr}]")

        primitive_attrs = _parse_atom_primitive(part)
        for key, value in primitive_attrs.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"Conflicting attribute '{key}' in AND block: '{expr}'")
            merged[key] = value

    return merged


def _split_top_level(expr: str, separators: Set[str]) -> List[str]:
    parts: List[str] = []
    level = 0
    last_index = 0

    for index, char in enumerate(expr):
        if char in RING_OPENERS:
            level += 1
        elif char in RING_CLOSERS:
            level -= 1
        elif level == 0 and char in separators:
            parts.append(expr[last_index:index])
            last_index = index + 1

    parts.append(expr[last_index:])
    return [part.strip() for part in parts if part.strip()]


def _parse_atom_primitive(text: str) -> Dict[str, object]:
    if not text:
        return {}

    special_attrs = _parse_special_primitive(text)
    if special_attrs:
        return special_attrs

    return _parse_standard_primitive(text)


def _parse_special_primitive(text: str) -> Dict[str, object]:
    if text == METAL_TOKEN:
        return {"is_metal": True}

    if text == NON_METAL_TOKEN:
        return {"is_metal": False}

    if text == LANTHANIDE_TOKEN:
        return {"is_lanthanide": True}

    if text == ACTINIDE_TOKEN:
        return {"is_actinide": True}

    period_match = re.fullmatch(rf"{PERIOD_PREFIX}(\d+)(?:-(\d+))?", text)
    if period_match:
        return {"period": _build_integer_range_set(period_match)}

    group_match = re.fullmatch(rf"{GROUP_PREFIX}(\d+)(?:-(\d+))?", text)
    if group_match:
        return {"group": _build_integer_range_set(group_match)}

    return {}


def _build_integer_range_set(match: re.Match[str]) -> Set[int]:
    start = int(match.group(1))
    end_group = match.group(2)
    end = int(end_group) if end_group is not None else start

    low = min(start, end)
    high = max(start, end)

    return set(range(low, high + 1))


def _parse_standard_primitive(text: str) -> Dict[str, object]:
    attributes: Dict[str, object] = {}
    segment = text.strip()
    if not segment:
        return attributes

    index = 0
    size = len(segment)

    index = _parse_isotope_prefix(segment, index, attributes)

    while index < size:
        char = segment[index]

        if char == ATOMIC_NUMBER_PREFIX:
            index = _parse_atomic_number(segment, index, attributes)
            continue

        if char.isalpha():
            index = _parse_alpha_code(segment, index, attributes)
            continue

        if char in {"+", "-"}:
            index = _parse_charge(segment, index, attributes)
            continue

        remaining = segment[index:].strip()
        if remaining:
            attributes.setdefault("unparsed", set()).add(remaining)
        break

    return attributes


def _parse_isotope_prefix(segment: str, start_index: int, attributes: Dict[str, object]) -> int:
    index = start_index
    size = len(segment)

    while index < size and segment[index].isdigit():
        index += 1

    if index > start_index and index < size and segment[index].isalpha():
        attributes["isotope"] = int(segment[start_index:index])
        return start_index

    return start_index


def _parse_atomic_number(segment: str, index: int, attributes: Dict[str, object]) -> int:
    size = len(segment)
    index += 1
    start = index

    while index < size and segment[index].isdigit():
        index += 1

    if index == start:
        raise ValueError(f"Malformed atomic number in bracket primitive: {segment!r}")

    attributes["atomic_number"] = int(segment[start:index])
    return index


def _parse_alpha_code(segment: str, index: int, attributes: Dict[str, object]) -> int:
    char = segment[index]

    if char == HYDROGEN_TOKEN and "atomic_number" in attributes:
        return _parse_hydrogen_count(segment, index + 1, attributes)

    if char == DEGREE_TOKEN:
        return _parse_integer_property(segment, index + 1, "degree", segment)

    if char == VALENCE_TOKEN:
        return _parse_integer_property(segment, index + 1, "valence", segment)

    if char == CONNECTIVITY_TOKEN:
        return _parse_integer_property(segment, index + 1, "connectivity", segment)

    if char == RING_SIZE_TOKEN:
        return _parse_ring_property(segment, index + 1, attributes)

    if char == RING_FLAG_TOKEN:
        attributes["in_ring"] = True
        return index + 1

    if char == ALIPHATIC_ANY_TOKEN:
        attributes["is_aromatic"] = False
        return index + 1

    if char == AROMATIC_ANY_TOKEN:
        attributes["is_aromatic"] = True
        return index + 1

    if char == "@":
        return _parse_chiral_flag(segment, index, attributes)

    return _parse_element_symbol(segment, index, attributes)


def _parse_hydrogen_count(segment: str, index: int, attributes: Dict[str, object]) -> int:
    size = len(segment)
    start = index

    while index < size and segment[index].isdigit():
        index += 1

    count_text = segment[start:index]
    attributes["h_count"] = int(count_text) if count_text else 1
    return index


def _parse_integer_property(
    segment: str,
    index: int,
    key: str,
    context: str,
) -> int:
    size = len(segment)
    start = index

    while index < size and segment[index].isdigit():
        index += 1

    if index == start:
        raise ValueError(f"Malformed {key} in bracket primitive: {context!r}")

    attributes_value = int(segment[start:index])
    return_index = index

    if key == "degree":
        context_dict_key = "degree"
    elif key == "valence":
        context_dict_key = "valence"
    else:
        context_dict_key = "connectivity"

    attributes_key = context_dict_key
    return_index = index

    segment_attributes = {attributes_key: attributes_value}
    segment_attributes.update(segment_attributes)
    return return_index


def _parse_ring_property(segment: str, index: int, attributes: Dict[str, object]) -> int:
    size = len(segment)
    start = index

    while index < size and segment[index].isdigit():
        index += 1

    if index > start:
        attributes["ring_size"] = int(segment[start:index])
    else:
        attributes["in_ring"] = True

    return index


def _parse_chiral_flag(segment: str, index: int, attributes: Dict[str, object]) -> int:
    size = len(segment)
    start = index

    while index < size and segment[index] == "@":
        index += 1

    count = index - start
    attributes["chiral_flag"] = attributes.get("chiral_flag", 0) + count
    return index


def _parse_element_symbol(segment: str, index: int, attributes: Dict[str, object]) -> int:
    size = len(segment)
    char = segment[index]

    if char.isupper():
        next_index = index + 1
        if next_index < size and segment[next_index].islower():
            symbol = segment[index : next_index + 1]
            index = next_index + 1
        else:
            symbol = char
            index = next_index
    else:
        symbol = char
        index += 1

    attributes["atomic_number"] = ob.GetAtomicNum(symbol)
    attributes["is_aromatic"] = symbol.islower()

    return index


def _parse_charge(segment: str, index: int, attributes: Dict[str, object]) -> int:
    size = len(segment)
    start = index

    while index < size and segment[index] in {"+", "-"} and index - start < 2:
        index += 1

    signs = segment[start:index]
    magnitude_start = index

    while index < size and segment[index].isdigit():
        index += 1

    magnitude_text = segment[magnitude_start:index]
    base = signs.count("+") - signs.count("-")

    if magnitude_text:
        magnitude = int(magnitude_text)
        charge = -abs(magnitude) if base < 0 else abs(magnitude)
    else:
        charge = base

    attributes["formal_charge"] = charge
    return index


def _create_query_atom_from_token(
    substructure: Substructure,
    token_type: TokenType,
    text: str,
) -> QueryAtom:
    if token_type == TokenType.BRACKET:
        attrs = parse_bracket_atom(text)
        normalized = _normalize_attribute_values(attrs)
        return QueryAtom(sub=substructure, **normalized)

    return _create_query_atom_from_symbol(substructure, text)


def _normalize_attribute_values(attrs: Dict[str, object]) -> Dict[str, Set[object]]:
    normalized: Dict[str, Set[object]] = {}
    for key, value in attrs.items():
        if isinstance(value, (set, list, tuple)):
            normalized[key] = set(value)
        else:
            normalized[key] = {value}
    return normalized


def _create_query_atom_from_symbol(substructure: Substructure, symbol: str) -> QueryAtom:
    if symbol == ANY_ATOM_TOKEN:
        return QueryAtom(sub=substructure)

    attributes: Dict[str, Set[object]] = {}

    if symbol.islower():
        if symbol == AROMATIC_ANY_TOKEN:
            attributes["is_aromatic"] = {True}
        else:
            attributes["atomic_number"] = {ob.GetAtomicNum(symbol.upper())}
            attributes["is_aromatic"] = {True}
        return QueryAtom(sub=substructure, **attributes)

    if symbol == ALIPHATIC_ANY_TOKEN:
        attributes["is_heavy_atom"] = {True}
        return QueryAtom(sub=substructure, **attributes)

    attributes["atomic_number"] = {ob.GetAtomicNum(symbol)}
    return QueryAtom(sub=substructure, **attributes)


def _bond_attrs_for_symbol(symbol: str) -> Dict[str, Set[Union[int, bool]]]:
    if symbol == SINGLE_BOND_TOKEN:
        return {"bond_order": {BondOrder.SINGLE.value}}

    if symbol == DOUBLE_BOND_TOKEN:
        return {"bond_order": {BondOrder.DOUBLE.value}}

    if symbol == TRIPLE_BOND_TOKEN:
        return {"bond_order": {BondOrder.TRIPLE.value}}

    if symbol == AROMATIC_BOND_TOKEN:
        return {"bond_order": {BondOrder.SINGLE.value, BondOrder.DOUBLE.value}, "is_aromatic": {True}}

    if symbol in {UP_BOND_TOKEN, DOWN_BOND_TOKEN}:
        return {"bond_order": {BondOrder.SINGLE.value}}

    if symbol == ANY_BOND_TOKEN:
        return {}

    raise ValueError(f"Unsupported bond symbol: {symbol}")


def _get_aromatic_flag(query_atom: QueryAtom) -> AromaticFlag:
    kwargs = getattr(query_atom, "kwargs", {})
    aromatic_value = kwargs.get("is_aromatic")

    if aromatic_value is None:
        return AromaticFlag.UNSET

    if True in aromatic_value:
        return AromaticFlag.AROMATIC

    if False in aromatic_value:
        return AromaticFlag.NON_AROMATIC

    return AromaticFlag.UNSET


def _infer_bond_attrs(
    first_atom: QueryAtom,
    second_atom: QueryAtom,
    pending_bond_attrs: Optional[Dict[str, object]],
) -> Dict[str, object]:
    first_state = _get_aromatic_flag(first_atom)
    second_state = _get_aromatic_flag(second_atom)

    if first_state is AromaticFlag.AROMATIC and second_state is AromaticFlag.AROMATIC:
        return {
            "bond_order": {BondOrder.SINGLE.value, BondOrder.DOUBLE.value},
            "is_aromatic": {True},
        }

    if pending_bond_attrs is not None:
        return pending_bond_attrs

    if _is_one_aromatic_and_other_unset(first_state, second_state):
        return {}

    if _is_one_aromatic_and_other_non_aromatic(first_state, second_state):
        return {"bond_order": {BondOrder.SINGLE.value}}

    return {"bond_order": {BondOrder.SINGLE.value}}


def _is_one_aromatic_and_other_unset(first: AromaticFlag, second: AromaticFlag) -> bool:
    return (
        (first is AromaticFlag.AROMATIC and second is AromaticFlag.UNSET)
        or (second is AromaticFlag.AROMATIC and first is AromaticFlag.UNSET)
    )


def _is_one_aromatic_and_other_non_aromatic(first: AromaticFlag, second: AromaticFlag) -> bool:
    return (
        (first is AromaticFlag.AROMATIC and second is AromaticFlag.NON_AROMATIC)
        or (second is AromaticFlag.AROMATIC and first is AromaticFlag.NON_AROMATIC)
    )


def _connect_or_anchor_ring(
    substructure: Substructure,
    ring_anchors: Dict[str, List[int]],
    ring_label: str,
    current_atom_index: int,
) -> None:
    indices = ring_anchors[ring_label]
    if not indices:
        indices.append(current_atom_index)
        return

    start_index = indices.pop()
    first_atom = substructure.query_atoms[start_index]
    second_atom = substructure.query_atoms[current_atom_index]
    bond_attrs = _infer_bond_attrs(first_atom, second_atom, pending_bond_attrs=None)
    substructure.add_bond(start_index, current_atom_index, **bond_attrs)