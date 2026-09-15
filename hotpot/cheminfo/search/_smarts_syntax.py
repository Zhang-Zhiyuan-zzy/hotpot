"""Lexical primitives for Hotpot's active SMARTS parser.

The internal lexer retains source spans for diagnostics.  The public
``tokenize`` adapter deliberately preserves Hotpot's historical list of
``(TokenType, text)`` tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Tuple

from ..elements import Element
from .errors import SmartsSyntaxError


class TokenType(str, Enum):
    ATOM = "ATOM"
    BRACKET = "BRACKET"
    BOND = "BOND"
    BRANCH_L = "BRANCH_L"
    BRANCH_R = "BRANCH_R"
    RING = "RING"
    DOT = "DOT"


@dataclass(frozen=True)
class Token:
    """A SMARTS token with its half-open source span."""

    type: TokenType
    text: str
    start: int
    end: int


BOND_CHARS: Set[str] = {"-", "=", "#", ":", "/", "\\", "~"}
AROMATIC_LOWER: Set[str] = {"b", "c", "n", "o", "p", "s", "se", "as"}
ANY_ATOM_TOKEN = "*"

_ATOMIC_NUMBERS = {
    symbol: atomic_number
    for atomic_number, symbol in enumerate(Element.symbols)
    if atomic_number
}


def atomic_number(symbol: str) -> int:
    """Return an atomic number for a correctly cased element symbol."""
    return _ATOMIC_NUMBERS.get(symbol, 0)


def tokenize_with_spans(smarts: str) -> List[Token]:
    """Tokenize graph-level SMARTS syntax while retaining source spans."""
    tokens: List[Token] = []
    index = 0
    while index < len(smarts):
        start = index
        char = smarts[index]
        if char.isspace():
            index += 1
            continue
        if char == "[":
            index = _find_closing_bracket_index(smarts, index) + 1
            token_type = TokenType.BRACKET
        elif char == "(":
            index += 1
            token_type = TokenType.BRANCH_L
        elif char == ")":
            index += 1
            token_type = TokenType.BRANCH_R
        elif char == ".":
            index += 1
            token_type = TokenType.DOT
        elif char in BOND_CHARS:
            _, index = _read_bond_token(smarts, index)
            token_type = TokenType.BOND
        elif char.isdigit() or _is_multi_digit_ring_start(smarts, index):
            _, index = _read_ring_token(smarts, index)
            token_type = TokenType.RING
        elif char.isalpha() or char == ANY_ATOM_TOKEN:
            _, index = _read_atom_token(smarts, index)
            token_type = TokenType.ATOM
        else:
            raise SmartsSyntaxError(
                f"Unsupported character in SMARTS: {char!r} at position {index}"
            )
        tokens.append(Token(token_type, smarts[start:index], start, index))
    return tokens


def tokenize(smarts: str) -> List[Tuple[TokenType, str]]:
    """Return tokens in the historical public representation."""
    return [(token.type, token.text) for token in tokenize_with_spans(smarts)]


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
                raise SmartsSyntaxError(f"Unmatched ')' in bracket atom: {text}")
        elif char == "[":
            if paren_depth == 0:
                raise SmartsSyntaxError(
                    f"Nested '[' is only valid in recursive SMARTS: {text}"
                )
            square_depth += 1
        elif char == "]":
            square_depth -= 1
            if square_depth == 0:
                return index
        index += 1
    raise SmartsSyntaxError(f"Unclosed '[' in SMARTS: {text}")


def _read_bond_token(smarts: str, index: int) -> Tuple[str, int]:
    parts = [smarts[index]]
    index += 1
    while index < len(smarts) and smarts[index] == ",":
        if index + 1 >= len(smarts) or smarts[index + 1] not in BOND_CHARS:
            raise SmartsSyntaxError(f"Malformed bond OR expression in SMARTS: {smarts}")
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
        if smarts[index + 1].islower() and atomic_number(candidate):
            return candidate, index + 2
    if char.islower() and index + 1 < len(smarts):
        candidate = smarts[index : index + 2]
        if candidate in AROMATIC_LOWER:
            return candidate, index + 2
    return char, index + 1
