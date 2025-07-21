# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : lexer
 Created   : 2025/7/7 19:46
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------

A compact, *self-contained* module that:

    1. Tokenises a SMARTS string (simple lexer).
    2. Builds a full molecular graph while giving top priority to ring
       closures so we know which bonds generate cycles.
    3. Breaks (“cuts”) those cycle-producing ring bonds, yielding a set
       of *acyclic* sub-graphs (fragments).
    4. Stores an explicit *link-table* describing how fragments are
       connected by the removed ring bonds.
    5. Converts every fragment into NetworkX cells (nodes/edges with
       attribute dictionaries).

This is *not* a feature-complete SMARTS implementation (that would run
to thousands of lines) – it intentionally supports a high-value subset:

    • element / wildcard atoms:  C  c  N  [O;H1]  *
    • bond symbols:             -  =  #  :  /  \  ~
    • ring digits 1-9 and %nn
    • branching with ( ... )
    • dot ‘.’ for disconnected components

You can extend the lexer / attribute builders later without touching
the splitting logic.
===========================================================
"""
import logging
import re
from typing import Union
from collections import defaultdict
from enum import Enum, auto

from openbabel import openbabel as ob
import sympy as sp

# from ..elements import elements
from hotpot.cheminfo.elements import elements

# from .._logic_tuple import AndTuple, OrTuple, NotTuple
from hotpot.cheminfo.search._logic_tuple import AndTuple, OrTuple, NotTuple
from hotpot.cheminfo.search.logic import AndDict, sor, sand, MutexBool, MutexValue, Chiral


__all__ = [
    'extract_brackets',
    'extract_numeric_pairs',
    'validate_brackets',
    'tokenize_smarts'
]

# Definition of patterns
_bracket_pattern = re.compile(r'\[([^\]]*?)\]')
_num_pattern = re.compile(r'%\d\d+|\d')

# Bond symbols defined in SMARTS (Daylight theory)
BOND_SYMBOLS = {'-', '=', '#', ':', '/', '\\', '.'}

class TokenType(Enum):
    BRACKET    = "bracket"          # [...] atom expression
    CYCLE_ST   = "cycle_start"      # start numeric closure token
    CYCLE_ED   = "cycle_end"        # end numeric closure token
    BRANCH_ST  = "branch"           # start (...) branch
    BRANCH_ED  = "branch_end"       # end (...) branch
    BOND       = "bond"             # -, =, #, :, /, \, .
    ATOM       = "atom"             # element or aromatic symbol
    OTHER      = "other"            # fallback


class SegmentType(Enum):
    NUMERIC = "numeric"
    OTHER = "other"


class AttrType(Enum):
    # The discrete type
    In = auto()                # value     in (v1, v2, ...)
    NotIn = auto()             # value not in (v1, v2, ...)
    Is = auto()                # value     is ...
    NotIs = auto()             # value not is ...
    Equal = auto()             # value      = ...
    NotEqual = auto()          # value     != ...

    # The continuous type
    LessThan = auto()          #        value <  max
    LessEqual = auto()         #        value <= max
    GreaterThan = auto()       #        value >  min
    GreaterEqual = auto()      #        value >= min
    GtAndLt = auto()           # min <  value <  max
    GeAndLt = auto()           # min <= value <  max
    GtAndLe = auto()           # min <  value <= max
    GeAndLe = auto()           # min <= value <= max
    LtOrGt = auto()            # value <  min or value >  max
    LeOrGt = auto()            # value <= min or value >  max
    LtOrGe = auto()            # value <  min or value >= max
    LeOrGe = auto()            # value <= min or value >= min


class MutexChiral(MutexValue):
    def _value_check(self, value):
        if not isinstance(value, Chiral):
            raise TypeError(f'value {value} is not a Chiral type')

    def __bool__(self):
        if self.value is Chiral.CIS or self.value is Chiral.TRANS:
            return True
        return False

    def __int__(self):
        if self.value is Chiral.CIS:
            return 1
        elif self.value is Chiral.TRANS:
            return -1
        else:
            return 0

    @classmethod
    def create_from_AT_num(cls, AT_num: int):
        """ Create an instance from the @ count in SMARTS or SMILES string """
        if AT_num == 0:
            raise cls(Chiral.UnSpecified)
        elif AT_num == 1:
            return cls(Chiral.CIS)
        elif AT_num == 2:
            return cls(Chiral.TRANS)
        else:
            raise ValueError(f'AT_num {AT_num} is not in 0, 1, 2')


def validate_brackets(smarts: str) -> None:
    """
    Raise ValueError if the SMARTS string contains:
      - mismatched number of '[' and ']' characters, or
      - nested brackets, i.e., any '[' or ']' inside a [...] region.
    """
    open_count = smarts.count('[')
    close_count = smarts.count(']')
    if open_count != close_count:
        raise ValueError(f"Unmatched brackets: {open_count} '[' vs {close_count} ']'")

    # Check for nested bracket characters inside any [...] region
    for match in _bracket_pattern.finditer(smarts):
        inner = match.group(1)
        if '[' in inner or ']' in inner:
            raise ValueError(f"Invalid nested bracket in SMARTS: '[{inner}]'")


def extract_brackets(smarts: str) -> list[tuple[int, int, str]]:
    """
    Return a list of tuples with:
      (start_index_of '[', end_index_of ']', inner_text)
    for each non‐nested [content] in the input string.
    """
    return [
        (match.start(), match.end(), match.group(0))
        for match in _bracket_pattern.finditer(smarts)
    ]


def extract_numeric_pairs(
        smarts: str,
        bracket_regions: list[tuple[int, int, str]]
) -> list[tuple[int, int, str, str]]:
    """
    1. Find all \d and %\d\d+ occurrences and record their positions and numeric value.
    2. Remove those inside [...] regions.
    3. Ensure each numeric value has an even count, else RuntimeError.
    4. Pair them (0-1, 2-3, ...) by proximity, record pair start/end positions,
       the numeric string, and the substring between the two occurrences.
    5. Return a list of (pair_start, pair_end, number_str, enclosed_content),
       sorted by pair_start.
    """
    # regex for single-digit or % followed by at least two digits
    occurrences_by_value: dict[str, list[tuple[int, int]]] = {}

    # collect and filter occurrences
    for m in _num_pattern.finditer(smarts):
        start = m.start()
        end = m.end()  # inclusive index of last digit or %
        number_str = m.group(0)
        # extract numeric part without '%'

        # skip if inside any bracket region
        inside_bracket = any(start >= bstart and end <= bend
                             for bstart, bend, _ in bracket_regions)
        if inside_bracket:
            continue

        occurrences_by_value.setdefault(number_str, []).append((start, end))

    # prepare result list
    paired_list: list[tuple[int, int, str, str]] = []

    # pair occurrences for each numeric value
    for number_str, occs in occurrences_by_value.items():
        if len(occs) % 2 != 0:
            raise RuntimeError(f"Unmatched occurrences for number {number_str}: {len(occs)} found in {smarts}")
        # sort by start position
        occs.sort(key=lambda x: x[0])
        # form pairs
        for i in range(0, len(occs), 2):
            start1, end1 = occs[i]
            start2, end2 = occs[i+1]
            # extract content between the two matches
            enclosed = smarts[end1+1:start2]
            paired_list.append((start1, end2, number_str, enclosed))

    # sort all pairs by their start position
    paired_list.sort(key=lambda x: x[0])
    return paired_list


def extract_branches(smarts: str, bracket_regions: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """
    Identify branch parentheses '()' outside any bracket_regions.
    Returns list of (start_index, end_index, inner_text).
    Raises ValueError on unbalanced or nested errors.
    """
    spans: list[tuple[int, int, str]] = []
    stack: list[int] = []

    def in_bracket(idx: int) -> bool:
        return any(bst <= idx < bed for bst, bed, _ in bracket_regions)

    for i, ch in enumerate(smarts):
        if ch == '(' and not in_bracket(i):
            stack.append(i)
        elif ch == ')' and not in_bracket(i):
            if not stack:
                raise ValueError(f"Unmatched ')' at position {i+1}/{len(smarts)}, SMARTS: {smarts}")
            start = stack.pop()
            spans.append((start, i, smarts[start+1:i]))
    if stack:
        raise ValueError(f"Unmatched '(' at positions {stack}")
    return sorted(spans, key=lambda x: x[0])


def tokenize_smarts(smarts: str) -> list[tuple[int, int, TokenType, str]]:
    """
    Split the SMARTS string into a sequence of tokens:
      1. [...] expressions (BRACKET)
      2. merged ring closures (CYCLE)
      3. branch expressions (...) (BRANCH)
      4. bonds (-,=,#,:,/,\\,.) (BOND)
      5. atom symbols (uppercase+optional lowercase, or single lowercase aromatic) (ATOM)
    Returns list of (start, end, TokenType, text).
    """
    # 1. identify bracket and numeric closures
    bracket_regions = extract_brackets(smarts)
    numeric_pairs   = extract_numeric_pairs(smarts, bracket_regions)

    # 2. identify branches
    branch_regions = extract_branches(smarts, bracket_regions)

    tokens: list[tuple[int, int, TokenType, str]] = []
    for st, ed, s, closer in numeric_pairs:
        tokens.append((st, st + len(s), TokenType.CYCLE_ST, s))
        tokens.append((ed - len(s), ed, TokenType.CYCLE_ED, s))

    for st, ed, closer in bracket_regions:
        tokens.append((st, ed, TokenType.BRACKET, closer))

    for st, ed, closer in branch_regions:
        tokens.append((st, st + 1, TokenType.BRANCH_ST, '('))
        tokens.append((ed, ed + 1, TokenType.BRANCH_ED, ')'))

    tokens = sorted(tokens, key=lambda x: x[0])

    # Split the smarts and validate
    txt_in_gap = []
    anchor = 0
    for st, ed, _, _ in tokens:
        assert anchor <= st
        txt_in_gap.append((anchor, st, smarts[anchor: st]))
        anchor = ed

    if anchor < len(smarts):
        txt_in_gap.append((anchor, len(smarts), smarts[anchor:]))

    def tokenize_gap(text: str, offset: int):
        i = 0
        L = len(text)
        tks = []
        while i < L:
            ch = text[i]
            abs_i = offset + i
            # bond?
            if ch in BOND_SYMBOLS:
                tks.append((abs_i, abs_i+1, TokenType.BOND, ch))
                i += 1
            # wildcard
            elif ch == '*':
                tks.append((abs_i, abs_i+1, TokenType.ATOM, ch))
                i += 1
            # uppercase atom symbol
            elif re.match(r'[A-Z]', ch):
                if i+1 < L and re.match(r'[a-z]', text[i+1]):
                    atom = text[i:i+2]
                    tks.append((abs_i, abs_i+2, TokenType.ATOM, atom))
                    i += 2
                else:
                    tks.append((abs_i, abs_i+1, TokenType.ATOM, ch))
                    i += 1
            # lowercase aromatic atom
            elif re.match(r'[bcnops]', ch):
                tks.append((abs_i, abs_i+1, TokenType.ATOM, ch))
                i += 1
            # ignore whitespace
            elif ch.isspace():
                i += 1
            else:
                tks.append((abs_i, abs_i+1, TokenType.OTHER, ch))
                i += 1

        return tks

    for st, ed, txt in txt_in_gap:
        tokens.extend(tokenize_gap(txt, st))

    tokens.sort(key=lambda x: x[0])
    return validate_token_continuous(smarts, tokens)


def validate_token_continuous(
        smarts: str,
        tokens: list[tuple[int, int, TokenType, str]]
) -> list[tuple[int, int, TokenType, str]]:
    if not tokens:
        raise ValueError(f"Empty token list")

    assert tokens[0][0] == 0, f"The tokens not start at 0, actual start is {tokens[0][0]}"
    assert tokens[-1][1] == len(smarts), f"The tokens should end at the length of smarts, got {tokens[-1][1]}"

    for i, (tk1, tk2) in enumerate(zip(tokens[:-1], tokens[1:])):
        assert tk1[1] == tk2[0], f'end of {i}th Token is not equal to start of {i+1}th Token'

    logging.debug(f"Tokens is continuous: {smarts}")
    return tokens


# Simple element lookup for atomic numbers
_ELEMENTS = tuple(elements.symbols)
_AROMATIC_ATOMS = {"h": 1, "b": 5, "c": 6, "n": 7, "o": 8, "p": 15, "s": 16, "f": 9}  # lowercase = aromatic

ELEMENTS = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
    "h": 1, "b": 5, "c": 6, "n": 7, "o": 8, "p": 15, "s": 16, "f": 9  # lowercase = aromatic
    # ... extend as needed
}

BOND_SYMBOL_ATTRS = {
    "-": {"bond_order": 1},
    "=": {"bond_order": 2},
    "#": {"bond_order": 3},
    ":": {"bond_order": (1, 2), "is_aromatic": True},
    "/": {"bond_order": 1, "stereo": "up"},
    "/?": {"bond_order": 1, "stereo": "not_down"},
    "\\?": {"bond_order": 1, "stereo": "not_up"},
    "\\": {"bond_order": 1, "stereo": "down"},
    "~": {},
    "@": {'in_ring': True}
}

# ------------- Bracket atom attribute parsing ----------------
# Definition of matcher
_hydrogen_catch = re.compile(
    r'H'                     # H(...)
    r'((\d*)(\+?)|'          # H2+, H2, H+
    r'\?|'                   # H?, one or zero
    r'\*|'                    # H*, at least one
    r'\{(\d*)?(-(\d+))?})',  # H{5}, H{,7}, H{3,7}, specify the ranges
)
def _parse_hydrogen_counts(s: str):
    m = _hydrogen_catch.fullmatch(s)
    if not m:
        return None, m

    if isinstance(g2 := m.group(2), str):  # In option1
        h_count = 1 if not g2 else int(g2)
        if m.group(3) == '+':
            return sp.Interval(h_count, sp.oo), m
        else:
            return h_count, m

    elif (g4 := m.group(4)) is None:  # Option 2&3:
        assert isinstance(g1 := m.group(1), str) and len(g1) == 1
        if g1 == '?':
            return sp.FiniteSet(0, 1), m
        elif g1 == '*':
            return sp.Interval(0, sp.oo), m
        else:
            raise AssertionError(f'Unknown hydrogen catch pattern: {m.group()}')

    else:  # Option 4
        assert isinstance(g4, str) and g4.isdigit(), f'Unknown hydrogen catch pattern: {m.group()}'
        if (g6 := m.group(6)).isdigit():
            return sp.Interval(int(g4), int(g6)), m
        else:
            return sp.Interval(0, int(g4)), m


def parse_bracket_atom(text: str) -> AndDict:
    """
    Parse a SMARTS bracket atom, e.g. "[nH+1;R;X3,!r,a]" into an attribute dict.
    Returns:
        dict with standardized named properties.
    """
    assert text.startswith('[') and text.endswith(']')
    expr = text[1:-1]

    def _tokenize_top(expr, seps):
        # Returns list of (subexpr, sep) where sep is after subexpr or '' for last
        result = []
        level = 0
        last = 0
        for i, c in enumerate(expr):
            if c in "([{":
                level += 1
            elif c in ")]}":
                level -= 1
            elif level == 0 and c in seps:
                result.append((expr[last:i], c))
                last = i+1
        result.append((expr[last:], ''))
        return result

    def _parse_atom_primitive(s: str) -> AndDict:
        attr = AndDict()
        s = s.strip()
        # Isotope: 13C, 2H, ...
        m = re.match(r'^(\d+)([A-Za-z][a-z]?)', s)
        if m:
            attr['isotope'].add(int(m.group(1)))
            s = s[m.end(1):]

        # Atomic number: #6
        m = re.match(r'^#(\d+)', s)
        if m:
            attr['atomic_number'].add(int(m.group(1)))
            s = s[m.end(0):]

        # Atom symbol (with aromaticity)
        m = re.match(r'^([A-Za-z][a-z]?)', s)
        if m:
            el = m.group(1)
            attr['atomic_number'].add(ob.GetAtomicNum(el))
            attr['is_aromatic'].add(MutexBool(el.islower()))
            s = s[m.end(1):]
        # Chiral: @/@@
        if '@' in s:
            attr['chiral'] = MutexChiral.create_from_AT_num(s.count('@'))
            s = s.replace('@', '')

        # Num hydrogens: H, H1, H2, ...
        hydrogen_count, m = _parse_hydrogen_counts(s)
        if hydrogen_count is not None:
            attr['hydrogen_counts'].add(hydrogen_count)
            s = s[m.end():]

        # Charge: +2, -1, ++, --
        m = re.match(r'([+-]{1,2})(\d*)', s)
        if m:
            sign = 1 if "+" in m.group(1) else -1
            magnitude = m.group(2)
            charges = m.group(1).count('+') - m.group(1).count('-')
            attr['charge'].add(sign * (int(magnitude) if magnitude else abs(charges)))
            s = s[m.end(0):]

        # Degree: Dn
        m = re.match(r'D(\d+)', s)
        if m:
            attr['degree'].add(int(m.group(1)))
            s = s[m.end(0):]

        # Valence: v<n>
        m = re.match(r'v(\d+)', s)
        if m:
            attr['valence'].add(int(m.group(1)))
            s = s[m.end(0):]

        # Connectivity: Xn
        m = re.match(r'X(\d+)', s)
        if m:
            attr['connectivity'].add(int(m.group(1)))
            s = s[m.end(0):]

        # Ring membership/size: r<n>
        m = re.match(r'r(\d+)?', s)
        if m:
            if m.group(1):
                attr['ring_size'].add(int(m.group(1)))
            else:
                attr['in_ring'].add(MutexBool(True))
            s = s[m.end(0):]

        # 'R' (any ring, or chain), and 'A' (aliphatic atom), 'a' (aromatic atom)
        if s.startswith('R'):
            attr['in_ring'].add(MutexBool(True))
            s = s[1:]
        if s.startswith('A'):
            attr['is_aromatic'].add(MutexBool(False))
            s = s[1:]
        if s.startswith('a'):
            attr['is_aromatic'].add(MutexBool(True))
            s = s[1:]

        # TODO: NotImplement
        # Logical feature, e.g. %(...)
        m = re.match(r'%\(([^)]*)\)', s)  # extra SMARTS feature
        if m:
            attr['extra_feature'] = m.group(1).strip()
            s = s[m.end(0):]

        if s:
            attr['unparsed'] = s

        return attr

    # Precedence (low to high): ',', ';'/'&', '!', e.g., 'C,N;O&!H'
    # 1. Split on ',' at top level (OR)
    hi_and_cells = [seg[0] for seg in _tokenize_top(expr, ';')]
    if len(hi_and_cells) > 1:
        return sand(parse_bracket_atom(f'[{cell}]') for cell in hi_and_cells)

    or_cells = [seg[0] for seg in _tokenize_top(expr, ',')]
    if len(or_cells) > 1:
        return sor(parse_bracket_atom(f'[{cell}]') for cell in or_cells)

    # 2. Split on ';' or '&' at top level (AND)
    and_cells = [seg[0] for seg in _tokenize_top(expr, '&')]
    if len(and_cells) > 1:
        return sand(parse_bracket_atom(f'[{cell}]') for cell in and_cells)

    # 3. Handle top-level !
    if expr.startswith('!'):
        return ~parse_bracket_atom(f'[{expr[1:]}]')

    # 4. Otherwise, parse simple
    # multiple ! inside (e.g. [!C;R], [C;!r;X3])
    return _parse_atom_primitive(expr)


# ------------- Atom symbol attributes (unbracketed, simple atoms/aromatics) ---------------
def parse_simple_atom(atom: str):
    att = {}
    if atom == '*':  # wildcard
        return att

    # Aromatic single letter: c n o p s
    if atom in "cnopsfab":
        att["atomic_number"] = ob.GetAtomicNum(atom.upper())  # c=6, etc.
        att["is_aromatic"] = True
    # Uppercase (possibly two-letter) atom: C, N, Cl, Br...
    elif atom in _ELEMENTS:
        att["atomic_number"] = ob.GetAtomicNum(atom)

    return att


# ------------- Interpreter wrapper ---------------
def interpret_tokens(
        tokens: list[tuple[int, int, TokenType, str]]
):
    atoms: list[dict] = []
    bonds: list[tuple[int, int], dict] = []
    living_branch = []
    living_rings = defaultdict(list)

    atom_anchor = None
    specified_bond_attr = None
    for st, ed, ttype, text in tokens:
        if ttype in (TokenType.ATOM, TokenType.BRACKET):
            atom = parse_simple_atom(text) if ttype == TokenType.ATOM else parse_bracket_atom(text)

            if atoms:
                last_atom_idx = atom_anchor or len(atoms) - 1
                bond_attr = specified_bond_attr or {'bond_order': 1}

                if 'is_aromatic' not in bond_attr:
                    is_bond_aromatic = atoms[-1].get('is_aromatic', False) and atom.get('is_aromatic', False)
                    if is_bond_aromatic:
                        bond_attr['is_aromatic'] = True

                bond = (last_atom_idx, len(atoms)), bond_attr
                bonds.append(bond)

            atoms.append(atom)
            specified_bond_attr = None
            atom_anchor = None

        elif ttype == TokenType.BOND:
            specified_bond_attr = BOND_SYMBOL_ATTRS.get(text, {})

        elif ttype == TokenType.BRANCH_ST:
            living_branch.append(len(atoms)-1)
        elif ttype == TokenType.BRANCH_ED:
            if not living_branch:
                raise RuntimeError(f'not found matched start branch token `(` for end token `)`')
            atom_anchor = living_branch.pop()

        elif ttype == TokenType.CYCLE_ST:
            living_rings[text].append(len(atoms)-1)
        elif ttype == TokenType.CYCLE_ED:
            if text not in living_rings:
                raise RuntimeError(f'not found matched cycle sign {text}, with out the key')
            if not living_rings[text]:
                raise RuntimeError(f'not found matched cycle sign {text}, with out the anchor')

            cycle_start = living_rings[text].pop()
            bonds.append(((cycle_start, len(atoms) - 1), {'bond_order': 1}))

        elif ttype == TokenType.OTHER:
            raise NotImplementedError(f"not implemented for {ttype}")

        else:
            raise NotImplementedError(f"not implemented for unknown type")

    # Check the regularization
    assert len(living_branch) == 0, "some branches is not closed"
    for sign, cycle_anchor in living_branch:
        if cycle_anchor:
            raise RuntimeError(f'cycle closure {sign} is not closed')

    return atoms, bonds


def test_extract_numeric_pairs():
    valid_numeric_smarts = [
        "c%11ccc2c(c%11)[nH]c3c2[C@@H]4CC[C@H](C(=O)O)N4c3C(=O)O"]

    for smarts in valid_numeric_smarts:
        # For this test, we assume no brackets, so bracket_regions is empty
        bracket_regions = []
        try:
            pairs = extract_numeric_pairs(smarts, bracket_regions)
            print(f"\nSMARTS: {smarts}")
            if pairs:
                for p in pairs:
                    print(f"  Pair pos=({p[0]}, {p[1]})  num='{p[2]}'  enclosed='{p[3]}'")
            else:
                print("  No numeric pairs found.")
        except RuntimeError as e:
            print(f"\nSMARTS: {smarts}\n  ERROR: {e}")


def test_tokenize():

    for smarts in mega_smarts:
        # smarts = "c%11ccc2c(c%11)[nH]c3c2[C@@H]4CC[C@H](C(=O)O)N4c3C[#6](=O)O"
        tokens = tokenize_smarts(smarts)
        atoms, bonds = interpret_tokens(tokens)

        print(atoms[0])
        print(atoms[1])
        print(bonds[0])
        print(smarts)


# Mega-SMARTS for stress-testing a parser
_mega_smarts = [
    # 1 – densely functionalised sugar-purine hybrid with recursion
    r"[$([C@@H]1OC[C@@H](O1)[C@H](O[$(*n2cnc3c2ncnc3N)])CO)]",

    # 2 – fused tetracyclic steroid core + explicit stereochemistry & ring indices
    r"C[C@]12CC[C@H]3[C@@H]1CC[C@@]4(C)[C@H]3CC[C@@]24C(=O)O",

    # 3 – nitrated, dichlorinated aromatic with tetrahedral halocarbon side chain
    r"[nH]1c(c(c(c(c1[N+](=O)[O-])Cl)Cl)-[C@H](F)[C@@](Br)(I)Cl)",

    # 4 – quaternary ammonium / phosphate zwitterion with logic lists & degree/valence
    r"[N+;D4;v4]([O-])([O-])([O-])[O-]-[$([C;D3]),$([O;D2])]",

    # 5 – recursive pattern: atom attached to BOTH a sulfide and an α,β-unsaturated carbonyl
    r"[$(*[S;X2]),$(*C=CC(=O))]",

    # 6 – seven-membered aromatic heteroring with alternating isotopes & charges
    r"[13c]1[n-][14c][nH][15c][n+][c]1",

    # 7 – internal alkyne flanked by E/Z-specified alkenes and explicit atom mapping
    r"[F:1]/C=C\[C#C]/C=C\[F:2]",

    # 8 – peptide triad with protected N-terminus, using component-level grouping
    r"(BocN(C(=O)OC(C)(C)C).C(=O)NCC(=O)N).(C(=O)O)>>(peptide-Boc).CO2",

    # 9 – metallated porphyrin core (square-planar) with generic metal wildcard
    r"[n]1c([n]c([n]c1-[M])=[N-])=c2[c][n][c]([n][c]2=[N-])-[M]",

    # 10 – bicyclic bridgehead expressing AND/OR bonding primitives & ring queries
    r"[C;R2;D3]1(-[C;R;!#6])C[C@@H]2C1CC2",

    # 11 – esterification reaction SMARTS with complete atom maps
    r"[C:1](=[O:2])[O-:3].[O:4][C:5]>>[C:1](=[O:2])[O:4][C:5].[O-:3]",

    # 12 – lactamisation (intramolecular) with optional (?1) map class
    r"([N:?1]CC(=O)O)>>[N:?1]C(=O)CO",

    # 13 – perfluoro-tert-butyl substituent as a side-chain wildcard
    r"C(C(F)(F)F)(C(F)(F)F)(C(F)(F)F)F",

    # 14 – arsenate analogue with variable bond wildcard and charge constraints
    r"[As;X4;D4](~O)(~O)(~O)[O-]",

    # 15 –  macrocyclic (14-membered) depsipeptide with alternating ester/amide
    r"O=C1NC(=O)OC[C@@H](C)NC(=O)OC[C@@H](C)NC(=O)OC1",

    # 16 – sulfonamide with both cis and trans geometric descriptors
    r"F/C=C\C(=O)N[S@@](=O)(=O)C\C=C/F",

    # 17 – recursive + ring test: atom in a 5-ring *and* connected to an exocyclic carbonyl
    r"[$(*[R5]),$(*C(=O))]",

    # 18 – triple list/logic example covering all halogens except iodine
    r"[F,Cl,Br][C;!R]=,#[C;R][F,Cl,Br]",

    # 19 – fullerene fragment with explicit ring closures up to 12
    r"C1C2C3C4C5C6C7C8C9C%10C%11C%12C1C2C3C4C5C6C7C8C9C%10C%11C%12",

    # 20 – charged diazonium with aromatic attachment and variable connectivity
    r"[N+;X2]#[N][c;a;D2]-[C;R0]",

    # 21 – hypervalent iodine reagent (IBX surrogate) with mixed valent states
    r"O=I(=O)(O[C@@H]1CCCCC1)O",

    # 22 – boronic-acid pinacol ester using grouped OR’s and ring enumeration
    r"B(O[C@H]1OC(C)(C)C(O)C1(C)C)O",

    # 23 – polyether crown fragment with repeating –OCC– unit (recursive repetition)
    r"[$(OCC)]3O",

    # 24 – chelated bidentate ligand SMARTS enclosing metal placeholder
    r"c1ccc(cc1)N=C(N)c2ccccc2>>[M]-c1ccc(N=C(N)c2ccccc2)cc1",

    # 25 – intentionally nasty: reaction SMARTS mixing mapped & unmapped atoms + multiple dots
    r"[C:1]=[O:2].[O-].[Na+]>>[C:1](=[O:2])[O-].[Na+]"
]


mega_smarts = [
    '[CH+,NH{1-3}][CH+,SH2]'
]

custom_smarts = [
    r''
]

if __name__ == "__main__":
    # test_extract_numeric_pairs()
    test_tokenize()
