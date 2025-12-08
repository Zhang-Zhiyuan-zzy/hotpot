"""
@File Name:        smarts
@Project:
@Author:           Zhiyuan Zhang
@Created On:       2025/12/6 13:40
@Project:          Hotpot
"""
# smarts_parser.py

import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from hotpot.cheminfo.core import Atom, Bond, Molecule
from .search import Substructure, QueryAtom  # adjust import path to your project layout


# --------- Lexing: minimal SMARTS tokenization ----------

class TokenType:
    ATOM = "ATOM"          # C, c, Cl, [..]
    BRACKET = "BRACKET"    # full [ ... ] text
    BOND = "BOND"          # -, =, #, :, /, \, ~  (implicit bond handled at syntax level)
    BRANCH_L = "BRANCH_L"  # (
    BRANCH_R = "BRANCH_R"  # )
    RING = "RING"          # ring digits 1-9 or %10 etc.


BOND_CHARS = set("-=#:\/\\")


def tokenize(smarts: str) -> List[Tuple[TokenType, str]]:
    """
    Minimal SMARTS tokenizer:
      - [ ... ] as BRACKET
      - 1/2-char element symbols: ATOM
      - lowercase c n o s p ... as aromatic ATOM
      - -,=,#,:,/,\ as BOND
      - digits / %nnn as RING
      - ( ) as BRANCH

    No support for reaction SMARTS (>>), atom maps, or other advanced features.
    """
    tokens: List[Tuple[str, str]] = []
    i = 0
    n = len(smarts)

    while i < n:
        ch = smarts[i]

        # skip whitespace
        if ch.isspace():
            i += 1
            continue

        # bracket atom
        if ch == '[':
            j = i + 1
            depth = 1
            while j < n and depth > 0:
                if smarts[j] == '[':
                    # nested [ ] is explicitly forbidden in this simplified parser
                    raise ValueError(f"Nested '[' not allowed in bracket atom: {smarts[i:j+1]}")
                if smarts[j] == ']':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth != 0:
                raise ValueError(f"Unclosed '[' in SMARTS: {smarts}")
            tokens.append((TokenType.BRACKET, smarts[i:j+1]))
            i = j + 1
            continue

        # branches
        if ch == '(':
            tokens.append((TokenType.BRANCH_L, ch))
            i += 1
            continue
        if ch == ')':
            tokens.append((TokenType.BRANCH_R, ch))
            i += 1            # syntax layer will validate bracket pairing
            continue

        # bonds
        if ch in BOND_CHARS:
            tokens.append((TokenType.BOND, ch))
            i += 1
            continue

        # ring digits: 1-9 or %10, %11 ...
        if ch.isdigit() or (ch == '%' and i + 2 < n and smarts[i+1:i+3].isdigit()):
            if ch == '%':
                j = i + 1
                while j < n and smarts[j].isdigit():
                    j += 1
                tokens.append((TokenType.RING, smarts[i:j]))
                i = j
            else:
                tokens.append((TokenType.RING, ch))
                i += 1
            continue

        # atom symbols: uppercase + optional lowercase; or single lowercase aromatic symbol
        if ch.isalpha():
            if ch.isupper():
                if i + 1 < n and smarts[i+1].islower():
                    tokens.append((TokenType.ATOM, smarts[i:i+2]))
                    i += 2
                else:
                    tokens.append((TokenType.ATOM, ch))
                    i += 1
            else:
                # lowercase aromatic b c n o p s ...
                tokens.append((TokenType.ATOM, ch))
                i += 1
            continue

        raise ValueError(f"Unsupported character in SMARTS: '{ch}' at position {i}")

    return tokens


# --------- One-layer logic parsing of bracket atoms ----------

AROMATIC_LOWER = set("bcnops")


def _parse_atom_primitive_(s: str) -> Dict[str, object]:
    """
    Parse a single [ ... ] primitive without top-level logical operators,
    using a small state machine over the string.

    Supported:
      - isotope:       13C
      - atomic number: #6
      - element:       C, c, N, Cl ...
      - stereo:        @ / @@   (record as simple count / flag)
      - H / Hn:        implicit hydrogen count (stored as 'h_count')
      - charge:        +, -, +2, -1, ++, -- (stored as 'formal_charge')
      - Dn:            degree
      - v<n>:          valence
      - Xn:            connectivity
      - r<n> / r:      ring size / in_ring
      - R:             in_ring
      - A / a:         non-aromatic / aromatic

    Example:  "OX2H" → O, X2, H1  yields
        {
            "atomic_number": 8,
            "is_aromatic": False,
            "connectivity": 2,
            "h_count": 1,
        }
    """
    attrs: Dict[str, object] = {}
    s = s.strip()
    if not s:
        return attrs

    i = 0
    n = len(s)

    # 1) optional isotope: digits followed by element-like token
    start_i = i
    while i < n and s[i].isdigit():
        i += 1
    if i > start_i and i < n and s[i].isalpha():
        # we treat preceding digits as isotope
        attrs["isotope"] = int(s[start_i:i])

    # if we parsed isotope, we *do not* consume the element yet; we just
    # recorded isotope and will parse the element in the main loop.
    # Reset i back to start of possible element / #Z / etc.
    i = start_i if "isotope" in attrs else 0

    while i < n:
        ch = s[i]

        # explicit atomic number: #6
        if ch == '#':
            i += 1
            j = i
            while j < n and s[j].isdigit():
                j += 1
            if j == i:
                raise ValueError(f"Malformed atomic number in bracket primitive: {s!r}")
            attrs["atomic_number"] = int(s[i:j])
            i = j
            continue

        # element symbol (one or two letters)
        if ch.isalpha():
            # handle element vs property code by longest match
            # element can be 1 or 2 letters; property codes are single letters.
            # We treat a leading letter+optional second lowercase as element
            # unless it is one of our single-letter property codes without
            # following lowercase (H, D, X, v, r, R, A, a).
            # To keep behavior close to canonical SMARTS, we do:
            #   - first check for standard property codes at this position
            #   - otherwise accept it as element.
            # However, 'H' must be treated as hydrogen *property* if
            # it appears after an element; as standalone 'H' we interpret
            # it as element hydrogen only when no other atom has been set.
            # Here we choose a simple rule:
            #   - if we already have atomic_number and this is 'H',
            #     treat as hydrogen-count property. Otherwise see below.

            # Hydrogen-count property H / Hn
            if ch == 'H' and "atomic_number" in attrs:
                i += 1
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                num_txt = s[i:j]
                attrs["h_count"] = int(num_txt) if num_txt else 1
                i = j
                continue

            # Degree: Dn
            if ch == 'D':
                i += 1
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                if j == i:
                    raise ValueError(f"Malformed degree Dn in bracket primitive: {s!r}")
                attrs["degree"] = int(s[i:j])
                i = j
                continue

            # Valence: v<n>
            if ch == 'v':
                i += 1
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                if j == i:
                    raise ValueError(f"Malformed valence v<n> in bracket primitive: {s!r}")
                attrs["valence"] = int(s[i:j])
                i = j
                continue

            # Connectivity: Xn
            if ch == 'X':
                i += 1
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                if j == i:
                    raise ValueError(f"Malformed connectivity Xn in bracket primitive: {s!r}")
                attrs["connectivity"] = int(s[i:j])
                i = j
                continue

            # Ring: r<n> or plain r
            if ch == 'r':
                i += 1
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                if j > i:
                    attrs["ring_size"] = int(s[i:j])
                    i = j
                else:
                    attrs["in_ring"] = True
                continue

            # In-ring: R
            if ch == 'R':
                attrs["in_ring"] = True
                i += 1
                continue

            # Aromaticity flags A / a
            if ch == 'A':
                attrs["is_aromatic"] = False
                i += 1
                continue
            if ch == 'a':
                attrs["is_aromatic"] = True
                i += 1
                continue

            # Stereo: @ / @@ (count all '@')
            if ch == '@':
                # consume continuous '@'
                j = i
                while j < n and s[j] == '@':
                    j += 1
                at_cnt = j - i
                attrs["chiral_flag"] = attrs.get("chiral_flag", 0) + at_cnt
                i = j
                continue

            # If we reach here, it's an element symbol.
            # One or two letters: first uppercase, optional second lowercase.
            if ch.isupper():
                j = i + 1
                if j < n and s[j].islower():
                    el = s[i:j+1]
                    i = j + 1
                else:
                    el = ch
                    i = j
            else:
                # lowercase element form (aromatic)
                el = ch
                i += 1

            from openbabel import openbabel as ob
            attrs["atomic_number"] = ob.GetAtomicNum(el)
            attrs["is_aromatic"] = el.islower()
            continue

        # charge: +, -, +2, -1, ++, --
        if ch in "+-":
            j = i
            # one or two signs
            while j < n and s[j] in "+-" and j - i < 2:
                j += 1
            signs = s[i:j]
            k = j
            while k < n and s[k].isdigit():
                k += 1
            magnitude_txt = s[j:k]
            base = signs.count('+') - signs.count('-')
            if magnitude_txt:
                charge = int(magnitude_txt)
                if base < 0:
                    charge = -abs(charge)
                else:
                    charge = abs(charge)
            else:
                charge = base
            attrs["formal_charge"] = charge
            i = k
            continue

        # Anything else we cannot parse: treat the rest as unparsed and stop
        remaining = s[i:].strip()
        if remaining:
            attrs.setdefault("unparsed", set()).add(remaining)
        break

    return attrs


def _parse_atom_primitive(s: str) -> Dict[str, object]:
    """
    Parse a single [ ... ] primitive without top-level logical operators,
    using a small state machine over the string.

    Supported:
      - isotope:       13C
      - atomic number: #6
      - element:       C, c, N, Cl ...
      - stereo:        @ / @@   (record as simple count / flag)
      - H / Hn:        implicit hydrogen count (stored as 'h_count')
      - charge:        +, -, +2, -1, ++, -- (stored as 'formal_charge')
      - Dn:            degree
      - v<n>:          valence
      - Xn:            connectivity
      - r<n> / r:      ring size / in_ring
      - R:             in_ring
      - A / a:         non-aromatic / aromatic

      Extended custom tokens:
      - M / !M:        metal / non‑metal (Molecule.is_metal)
      - Ln:            lanthanide (Molecule.is_lanthanide)
      - An:            actinide (Molecule.is_actinide)
      - NPn / NPm-n:   period == n or m..n (Molecule.period)
      - NGn / NGm-n:   group  == n or m..n (Molecule.group)
    """
    attrs: Dict[str, object] = {}
    s = s.strip()
    if not s:
        return attrs

    # --- special-case patterns that are easier to detect by regex/startswith ---

    # metal / non‑metal: [M] / [!M]
    if s == "M":
        attrs["is_metal"] = True
        return attrs
    if s == "!M":
        attrs["is_metal"] = False
        return attrs

    # lanthanides / actinides
    if s == "Ln":
        attrs["is_lanthanide"] = True
        return attrs
    if s == "An":
        attrs["is_actinide"] = True
        return attrs

    # period selector: NP3, NP3-5
    m = re.fullmatch(r"NP(\d+)(?:-(\d+))?", s)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if lo > hi:
            lo, hi = hi, lo
        # 保存为整数集合，parse_bracket_atom 会再包装成 set(...)
        attrs["period"] = set(range(lo, hi + 1))
        return attrs

    # group selector: NG4, NG2-4
    m = re.fullmatch(r"NG(\d+)(?:-(\d+))?", s)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if lo > hi:
            lo, hi = hi, lo
        attrs["group"] = set(range(lo, hi + 1))
        return attrs

    # 下面是原来的状态机逻辑
    i = 0
    n = len(s)

    # 1) optional isotope: digits followed by element-like token
    start_i = i
    while i < n and s[i].isdigit():
        i += 1
    if i > start_i and i < n and s[i].isalpha():
        attrs["isotope"] = int(s[start_i:i])

    # if we parsed isotope, we *do not* consume the element yet
    i = start_i if "isotope" in attrs else 0

    while i < n:
        ch = s[i]

        # explicit atomic number: #6
        if ch == '#':
            i += 1
            j = i
            while j < n and s[j].isdigit():
                j += 1
            if j == i:
                raise ValueError(f"Malformed atomic number in bracket primitive: {s!r}")
            attrs["atomic_number"] = int(s[i:j])
            i = j
            continue

        # element symbol or property codes
        if ch.isalpha():
            # Hydrogen-count property H / Hn (only if an element already parsed)
            if ch == 'H' and "atomic_number" in attrs:
                i += 1
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                num_txt = s[i:j]
                attrs["h_count"] = int(num_txt) if num_txt else 1
                i = j
                continue

            # Degree: Dn
            if ch == 'D':
                i += 1
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                if j == i:
                    raise ValueError(f"Malformed degree Dn in bracket primitive: {s!r}")
                attrs["degree"] = int(s[i:j])
                i = j
                continue

            # Valence: v<n>
            if ch == 'v':
                i += 1
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                if j == i:
                    raise ValueError(f"Malformed valence v<n> in bracket primitive: {s!r}")
                attrs["valence"] = int(s[i:j])
                i = j
                continue

            # Connectivity: Xn
            if ch == 'X':
                i += 1
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                if j == i:
                    raise ValueError(f"Malformed connectivity Xn in bracket primitive: {s!r}")
                attrs["connectivity"] = int(s[i:j])
                i = j
                continue

            # Ring: r<n> or plain r
            if ch == 'r':
                i += 1
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                if j > i:
                    attrs["ring_size"] = int(s[i:j])
                    i = j
                else:
                    attrs["in_ring"] = True
                continue

            # In-ring: R
            if ch == 'R':
                attrs["in_ring"] = True
                i += 1
                continue

            # Aromaticity flags A / a
            if ch == 'A':
                attrs["is_aromatic"] = False
                i += 1
                continue
            if ch == 'a':
                attrs["is_aromatic"] = True
                i += 1
                continue

            # Stereo: @ / @@ (count all '@')
            if ch == '@':
                j = i
                while j < n and s[j] == '@':
                    j += 1
                at_cnt = j - i
                attrs["chiral_flag"] = attrs.get("chiral_flag", 0) + at_cnt
                i = j
                continue

            # If we reach here, it's an element symbol.
            if ch.isupper():
                j = i + 1
                if j < n and s[j].islower():
                    el = s[i:j+1]
                    i = j + 1
                else:
                    el = ch
                    i = j
            else:
                el = ch
                i += 1

            from openbabel import openbabel as ob
            attrs["atomic_number"] = ob.GetAtomicNum(el)
            attrs["is_aromatic"] = el.islower()
            continue

        # charge: +, -, +2, -1, ++, --
        if ch in "+-":
            j = i
            while j < n and s[j] in "+-" and j - i < 2:
                j += 1
            signs = s[i:j]
            k = j
            while k < n and s[k].isdigit():
                k += 1
            magnitude_txt = s[j:k]
            base = signs.count('+') - signs.count('-')
            if magnitude_txt:
                charge = int(magnitude_txt)
                charge = -abs(charge) if base < 0 else abs(charge)
            else:
                charge = base
            attrs["formal_charge"] = charge
            i = k
            continue

        # Anything else we cannot parse: treat the rest as unparsed and stop
        remaining = s[i:].strip()
        if remaining:
            attrs.setdefault("unparsed", set()).add(remaining)
        break

    return attrs


def parse_bracket_atom(expr_text: str) -> Dict[str, object]:
    """
    One-layer parsing of a bracket atom [ ... ]:

      expr = text[1:-1]

      - Top-level ',' means OR.
        For this simplified implementation, we only support:
          * element lists: [C,N,O] -> atomic_number in {6, 7, 8}
        Other property OR combinations are intentionally not implemented.

      - Top-level ';' / '&' means AND: all primitives must hold.

      - Primitives can start with '!': in a full implementation this would
        be a negation of that primitive, but here we prefer to raise an error
        instead of silently approximating, to keep semantics predictable.

    The returned dict is shaped to plug directly into QueryAtom(**attrs) and
    primarily includes one-layer attributes like atomic_number, is_aromatic,
    formal_charge, h_count, etc.
    """
    assert expr_text.startswith('[') and expr_text.endswith(']')
    expr = expr_text[1:-1].strip()

    if '$(' in expr:
        raise NotImplementedError("Recursive SMARTS '$(' not supported in one-layer parser")

    # top-level split helper
    def _split_top(s: str, seps: str):
        res = []
        level = 0
        last = 0
        for i, c in enumerate(s):
            if c in "([{":
                level += 1
            elif c in ")]}":
                level -= 1
            elif level == 0 and c in seps:
                res.append(s[last:i])
                last = i + 1
        res.append(s[last:])
        return [part.strip() for part in res if part.strip()]

    # simple case: no OR at top level
    if ',' not in expr:
        return _parse_and_block(expr)

    # OR present: we only support element lists, e.g. [C,N,O]
    parts = _split_top(expr, ',')
    atomic_nums = set()
    common_attrs: Dict[str, object] = {}

    for p in parts:
        prim_attrs = _parse_and_block(p)  # already guaranteed to be OR-free
        # Only allow OR over atomic_number / is_aromatic.
        keys = set(prim_attrs.keys())
        other = keys - {"atomic_number", "is_aromatic"}
        if other:
            raise NotImplementedError(
                f"Only simple element OR like [C,N,O] is supported for ',', got: [{expr}]"
            )

        z = prim_attrs.get("atomic_number")
        if z is not None:
            atomic_nums.add(z)

        # if is_aromatic is present, it must be consistent across all parts
        if "is_aromatic" in prim_attrs:
            if "is_aromatic" in common_attrs and common_attrs["is_aromatic"] != prim_attrs["is_aromatic"]:
                raise NotImplementedError(f"Inconsistent aromatic flags in OR list: [{expr}]")
            common_attrs["is_aromatic"] = prim_attrs["is_aromatic"]

    if not atomic_nums:
        raise NotImplementedError(f"Unsupported OR expression: [{expr}]")

    # return a dict with atomic_number as a set to feed into QueryAtom
    common_attrs["atomic_number"] = atomic_nums
    return common_attrs


def _parse_and_block(expr: str) -> Dict[str, object]:
    """
    Parse an AND-only block (no top-level ','):

      Examples:
        'C;H2', 'nH+1;R;X3', 'C;!r'

    In this reduced implementation, negation with '!' is not supported:
      - [!C] or similar patterns will raise NotImplementedError rather than
        being approximated, to avoid confusing semantics.

    If you want to support [!C], you can push something like 'not_atomic_number'
    into the attrs and implement it inside QueryAtom.match.
    """
    # split on ';' / '&' at top level
    parts = []
    level = 0
    last = 0
    for i, c in enumerate(expr):
        if c in "([{":
            level += 1
        elif c in ")]}":
            level -= 1
        elif level == 0 and c in ";&":
            parts.append(expr[last:i])
            last = i + 1
    parts.append(expr[last:])

    merged: Dict[str, object] = {}
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.startswith('!'):
            raise NotImplementedError(
                f"NOT '!' in bracket atom not supported yet: [{expr}]"
            )
        prim_attrs = _parse_atom_primitive(p)
        for k, v in prim_attrs.items():
            if k in merged and merged[k] != v:
                raise ValueError(
                    f"Conflicting attribute '{k}' in AND block: '{expr}'"
                )
            merged[k] = v
    return merged


def _get_aromatic_flag(qa: QueryAtom):
    """
    Interpret aromatic constraint on a QueryAtom.

    Returns one of:
      "arom"     -> is_aromatic explicitly contains True
      "non_arom" -> is_aromatic explicitly contains False (and not True)
      "unset"    -> no is_aromatic constraint

    This assumes QueryAtom stores its attributes in a 'kwargs' dict; adjust
    if your implementation differs.
    """
    arom = getattr(qa, "kwargs", {}).get("is_aromatic", None)
    if arom is None:
        return "unset"
    if True in arom:
        return "arom"
    if False in arom:
        return "non_arom"
    return "unset"


def _infer_bond_attrs(
    qa1: QueryAtom,
    qa2: QueryAtom,
    pending_bond_attrs: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """
    Decide final bond attributes based on:

      - aromatic constraints on both end atoms
      - explicitly specified bond symbol (if any)

    Rules:

      1) If both atoms are explicitly aromatic ("arom" & "arom"):
           -> treat the bond as aromatic:
                {"bond_order": {1, 2}, "is_aromatic": {True}}
           This overrides any explicit bond symbol.

      2) Otherwise, if there is an explicit pending_bond_attrs:
           -> use pending_bond_attrs as-is.

      3) No explicit bond, one side "arom" and the other "unset":
           -> unconstrained bond: {}.

      4) No explicit bond, one side "arom" and the other "non_arom":
           -> default single bond: {"bond_order": {1}}.

      5) All other "no explicit bond" cases:
           -> default single bond: {"bond_order": {1}}.
    """
    state1 = _get_aromatic_flag(qa1)
    state2 = _get_aromatic_flag(qa2)

    # 1) both aromatic -> aromatic bond, highest priority (overrides explicit)
    if state1 == "arom" and state2 == "arom":
        return {"bond_order": {1, 2}, "is_aromatic": {True}}

    # 2) explicit bond takes precedence in all other cases
    if pending_bond_attrs is not None:
        return pending_bond_attrs

    # 3) no explicit bond: one aromatic, other unset -> unconstrained bond
    if ((state1 == "arom" and state2 == "unset") or
        (state2 == "arom" and state1 == "unset")):
        return {}

    # 4) no explicit bond: one aromatic, other explicitly non-aromatic
    if ((state1 == "arom" and state2 == "non_arom") or
        (state2 == "arom" and state1 == "non_arom")):
        return {"bond_order": {1}}

    # 5) everything else -> default single bond
    return {"bond_order": {1}}


def substructure_from_smarts(smarts: str) -> Substructure:
    """
    Parse a simplified SMARTS pattern into a Substructure:

      Supported:
        - atoms, bracket atoms
        - explicit bonds
        - implicit single bonds
        - branches ()
        - ring digits 1-9 / %nn

      Not supported:
        - reaction SMARTS (>>)
        - disconnected components '.'
        - atom mapping and other advanced features
    """
    tokens = tokenize(smarts)
    sub = Substructure()
    atoms: List[QueryAtom] = []
    ring_anchors: Dict[str, int] = defaultdict(list)  # digit -> atom index list
    branch_stack: List[int] = []     # branch return points
    last_atom_idx: Optional[int] = None
    pending_bond_attrs: Optional[Dict[str, object]] = None

    def _make_qatom_from_symbol(sym: str) -> QueryAtom:
        """
        Build a QueryAtom from a non-bracket symbol: C, c, N, Cl, a, A, * ...
        """
        from openbabel import openbabel as ob

        # '*' means "any atom": no constraints
        if sym == '*':
            return QueryAtom(sub=sub)

        attrs = {}

        # lowercase = aromatic symbol space
        if sym.islower():
            if sym == 'a':
                # 'a' = any aromatic atom: constrain only aromaticity
                attrs["is_aromatic"] = {True}
            else:
                # concrete aromatic element: c, n, o, s, p ...
                attrs["atomic_number"] = {ob.GetAtomicNum(sym.upper())}
                attrs["is_aromatic"] = {True}
        else:
            # uppercase = aliphatic / generic heavy atom space
            if sym == 'A':
                # 'A' = any non-hydrogen atom; this flag is for QueryAtom.match
                attrs["is_heavy_atom"] = {True}
            else:
                attrs["atomic_number"] = {ob.GetAtomicNum(sym)}
            # do not force is_aromatic here; leave it to actual atom properties
        return QueryAtom(sub=sub, **attrs)

    def _make_qatom_from_bracket(text: str) -> QueryAtom:
        """
        Build a QueryAtom from a full bracket expression [ ... ].
        """
        attrs = parse_bracket_atom(text)
        # normalize scalar attributes into sets for QueryAtom
        norm = {}
        for k, v in attrs.items():
            if isinstance(v, set) or isinstance(v, (list, tuple)):
                norm[k] = set(v)
            else:
                norm[k] = {v}
        return QueryAtom(sub=sub, **norm)

    def _bond_attrs_for_symbol(sym: str) -> Dict[str, object]:
        """
        Map a bond symbol to bond attribute constraints.
        """
        if sym == '-':
            return {"bond_order": {1}}
        if sym == '=':
            return {"bond_order": {2}}
        if sym == '#':
            return {"bond_order": {3}}
        if sym == ':':
            # aromatic / conjugated bond
            return {"bond_order": {1, 2}, "is_aromatic": {True}}
        if sym in ('/', '\\'):
            # simplified: treat as single bond, ignore stereo for now
            return {"bond_order": {1}}
        if sym == '~':
            # any bond: no constraints
            return {}
        raise ValueError(f"Unsupported bond symbol: {sym}")

    # main token traversal
    for ttype, text in tokens:
        if ttype in (TokenType.ATOM, TokenType.BRACKET):
            # create QueryAtom
            if ttype == TokenType.ATOM:
                qa = _make_qatom_from_symbol(text)
            else:
                qa = _make_qatom_from_bracket(text)

            sub.add_atom(qa)
            this_idx = len(sub.query_atoms) - 1

            # connect with previous atom (implicit or explicit bond)
            if last_atom_idx is not None:
                qa1 = sub.query_atoms[last_atom_idx]
                qa2 = qa
                bond_attrs = _infer_bond_attrs(qa1, qa2, pending_bond_attrs)
                sub.add_bond(last_atom_idx, this_idx, **bond_attrs)

            last_atom_idx = this_idx
            pending_bond_attrs = None

        elif ttype == TokenType.BOND:
            pending_bond_attrs = _bond_attrs_for_symbol(text)

        elif ttype == TokenType.BRANCH_L:
            if last_atom_idx is None:
                raise ValueError(f"Branch '(' cannot appear before first atom in SMARTS: {smarts}")
            branch_stack.append(last_atom_idx)

        elif ttype == TokenType.BRANCH_R:
            if not branch_stack:
                raise ValueError(f"Unmatched ')' in SMARTS: {smarts}")
            # return to branch anchor
            last_atom_idx = branch_stack.pop()
            pending_bond_attrs = None

        elif ttype == TokenType.RING:
            if last_atom_idx is None:
                raise ValueError(f"Ring digit without atom in SMARTS: {smarts}")
            key = text  # '1' or '%10'
            if key not in ring_anchors or ring_anchors[key] == []:
                ring_anchors[key] = [last_atom_idx]
            else:
                start_idx = ring_anchors[key].pop()
                qa1 = sub.query_atoms[start_idx]
                qa2 = sub.query_atoms[last_atom_idx]
                # ring closures never have an explicit bond token here
                bond_attrs = _infer_bond_attrs(qa1, qa2, None)
                sub.add_bond(start_idx, last_atom_idx, **bond_attrs)
        else:
            raise ValueError(f"Unexpected token type {ttype} in simplified parser")

    if branch_stack:
        raise ValueError(f"Unclosed '(' in SMARTS: {smarts}")
    for k, v in ring_anchors.items():
        if v:
            raise ValueError(f"Unclosed ring digit {k} in SMARTS: {smarts}")

    return sub