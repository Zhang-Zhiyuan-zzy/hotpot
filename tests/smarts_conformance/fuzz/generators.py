"""Bounded deterministic generators for the SMARTS robustness runner."""

from __future__ import annotations

import random
import string
from typing import Tuple


ATOMS = (
    "C",
    "N",
    "O",
    "F",
    "Cl",
    "Br",
    "*",
    "c",
    "n",
    "[#6]",
    "[#7]",
    "[C,N]",
    "[!#6]",
    "[C;H0]",
    "[C;D1]",
    "[C;X4]",
    "[C;v4]",
    "[#6;!R]",
)
BONDS = ("-", "=", "#", "~")
RINGS = ("C1CCCCC1", "c1ccccc1", "C1CC1", "N1CCCC1")
LOGICAL = (
    "[C,N]",
    "[C;H0]",
    "[#6&!R]",
    "[!#6;!#1]",
    "[C,N;H1]",
)
RECURSIVE = (
    "[$([#6]-[#8])]",
    "[C;$([#6]-[#7])]",
    "[N;!$(N-C=O)]",
)

INVALID_MUTATORS: Tuple[Tuple[str, str, str], ...] = (
    ("leading_empty_component", "prefix", "."),
    ("trailing_empty_component", "suffix", "."),
    ("double_empty_component", "suffix", "..C"),
    ("leading_bond", "prefix", "="),
    ("dangling_bond", "suffix", "-"),
    ("unclosed_branch", "suffix", "("),
    ("unmatched_branch", "suffix", ")"),
    ("empty_branch", "suffix", "()"),
    ("unclosed_ring", "suffix", "%99"),
    ("unsupported_character", "suffix", "\x00"),
)

ROBUSTNESS_ALPHABET = (
    string.ascii_letters
    + string.digits
    + "[]()$!&,;:+-#:=~.%/\\* \t"
    + "\x00\x01\x7f"
    + "λ中"
)


def generate_valid_smarts(rng: random.Random, max_atoms: int = 8) -> str:
    """Generate a query from grammar fragments accepted by the target dialect."""

    form = rng.choice(("atom", "linear", "branch", "ring", "logic", "recursive"))
    if form == "atom":
        return rng.choice(ATOMS)
    if form == "ring":
        return rng.choice(RINGS)
    if form == "logic":
        return rng.choice(LOGICAL)
    if form == "recursive":
        return rng.choice(RECURSIVE)

    atom_count = rng.randint(2, max(2, max_atoms))
    atoms = [rng.choice(ATOMS) for _ in range(atom_count)]
    if form == "linear":
        query = atoms[0]
        for atom in atoms[1:]:
            query += rng.choice(BONDS) + atom
        return query

    branch_atom = atoms.pop()
    query = atoms[0] + "(" + rng.choice(BONDS) + branch_atom + ")"
    for atom in atoms[1:]:
        query += rng.choice(BONDS) + atom
    return query


def mutate_valid_smarts(
    rng: random.Random,
    valid_seed: str,
) -> Tuple[str, str, str, int, str]:
    """Apply one named local corruption to a known-valid generated query."""

    mutation, placement, payload = rng.choice(INVALID_MUTATORS)
    position = 0 if placement == "prefix" else len(valid_seed)
    mutated = valid_seed[:position] + payload + valid_seed[position:]
    return mutation, valid_seed, mutated, position, payload


def generate_structured_stress_smarts(
    rng: random.Random, max_depth: int
) -> Tuple[str, str]:
    """Generate a valid bounded query with one deliberately deep structure."""

    depth = rng.randint(2, max(2, max_depth))
    form = rng.choice(("branch", "recursive", "logic", "rings"))
    if form == "branch":
        return form, "C(" * depth + "C" + ")" * depth
    if form == "recursive":
        query = "C"
        for _ in range(depth):
            query = f"[C;$({query})]"
        return form, query
    if form == "logic":
        return form, "[" + ";".join("#6,#7" for _ in range(depth)) + "]"
    return form, ".".join("C1CCCCC1" for _ in range(min(depth, 8)))


def generate_robustness_text(rng: random.Random, max_length: int = 96) -> str:
    """Generate arbitrary text with bounded size; no acceptance is prescribed."""

    length = rng.randint(0, max_length)
    return "".join(rng.choice(ROBUSTNESS_ALPHABET) for _ in range(length))
