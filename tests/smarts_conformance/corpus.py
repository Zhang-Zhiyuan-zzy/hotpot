"""Deterministic, offline SMARTS conformance corpora.

The large sets below are generated from small, reviewable grammar tables. This
keeps provenance and intent visible while exercising the parser over the
periodic table, primitive boundaries, graph syntax, and positive/near-negative
semantic pairs. Generation is deterministic and performs no I/O.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "1.0"
DIALECT = "hotpot-daylight-core-2026-09"
CURATED_SOURCE = {
    "kind": "curated_generated",
    "name": "Hotpot SMARTS conformance grammar tables",
    "license": "same-as-project",
    "version": "2026-09-15",
    "reviewed": True,
}

ELEMENT_SYMBOLS: Tuple[str, ...] = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)


def _parse_case(
    case_id: str,
    smarts: str,
    classification: str,
    outcome: str,
    features: Sequence[str],
    expected_phase: Sequence[str] = (),
) -> Dict[str, object]:
    case: Dict[str, object] = {
        "id": case_id,
        "smarts": smarts,
        "classification": classification,
        "expected_outcome": outcome,
        "features": tuple(features),
        "dialect": DIALECT,
        "source": dict(CURATED_SOURCE),
    }
    if expected_phase:
        case["expected_phase"] = tuple(expected_phase)
    return case


def _semantic_case(
    case_id: str,
    smarts: str,
    target: str,
    matched: bool,
    features: Sequence[str],
) -> Dict[str, object]:
    return {
        "id": case_id,
        "smarts": smarts,
        "classification": "semantic_case",
        "target": {"format": "smiles", "text": target},
        "options": {},
        "expected": {"matched": matched},
        "features": tuple(features),
        "dialect": DIALECT,
        "source": dict(CURATED_SOURCE),
        "oracle": {
            "kind": "curated",
            "basis": "single-feature positive or near-negative pair",
            "reviewed": True,
        },
    }


def _build_valid_parse_cases() -> Tuple[Mapping[str, object], ...]:
    cases: List[Mapping[str, object]] = []

    for atomic_number in range(1, 119):
        cases.append(
            _parse_case(
                f"parse.atomic_number.{atomic_number:03d}",
                f"[#{atomic_number}]",
                "valid_core",
                "accept",
                ("bracket_atom", "atomic_number", "periodic_table"),
            )
        )
    for atomic_number, symbol in enumerate(ELEMENT_SYMBOLS, start=1):
        cases.append(
            _parse_case(
                f"parse.element.{atomic_number:03d}",
                f"[{symbol}]",
                "valid_core",
                "accept",
                ("bracket_atom", "element_identity", "periodic_table"),
            )
        )

    for code in ("D", "X", "v", "R", "r"):
        for value in range(13):
            cases.append(
                _parse_case(
                    f"parse.primitive.{code}.{value:02d}",
                    f"[C;{code}{value}]",
                    "valid_core",
                    "accept",
                    ("atom_primitive", code, "numeric_boundary"),
                )
            )
    for value in range(9):
        cases.append(
            _parse_case(
                f"parse.primitive.H.{value:02d}",
                f"[C;H{value}]",
                "valid_core",
                "accept",
                ("atom_primitive", "H", "hydrogen_count"),
            )
        )
    for map_number in range(1, 33):
        cases.append(
            _parse_case(
                f"parse.atom_map.{map_number:03d}",
                f"[C:{map_number}]",
                "valid_core",
                "accept",
                ("atom_map", "metadata", "multi_digit"),
            )
        )

    charge_forms = (
        "[C+]",
        "[C++]",
        "[C+++]",
        "[C+1]",
        "[C+2]",
        "[C+3]",
        "[N-]",
        "[N--]",
        "[N---]",
        "[N-1]",
        "[N-2]",
        "[N-3]",
    )
    for index, smarts in enumerate(charge_forms, start=1):
        cases.append(
            _parse_case(
                f"parse.charge.{index:03d}",
                smarts,
                "valid_core",
                "accept",
                ("formal_charge", "equivalent_spelling"),
            )
        )

    logical = (
        "[C,N]",
        "[C;H3;R0]",
        "[C&H3]",
        "[C!R]",
        "[C;!R]",
        "[!C]",
        "[!!C]",
        "[C,N;H1]",
        "[C;H1,N;H2]",
        "[C,N,O,S]",
        "[C;D1;X4]",
        "[C&D1&X4]",
        "[C;!R;H3]",
        "[#6,#7;!R]",
        "[C;$(C=O)]",
        "[C;!$(C=O)]",
        "[#6;$([#6](=[#8])[#7])]",
        "[N;!$(N-C=O)]",
        "[C;$(C(-O)-N)]",
        "[C;$(C1CC1)]",
        "[C;D1,D2]",
        "[C;R0,R1]",
        "[C;H0,H1,H2,H3]",
        "[C;v3,v4]",
        "[C;X3,X4]",
        "[C;!D0]",
        "[C;!!!R]",
        "[C;!!!!R]",
        "[C;D1&X4;!R]",
        "[C,N;!$(C=O);R0]",
    )
    for index, smarts in enumerate(logical, start=1):
        cases.append(
            _parse_case(
                f"parse.logic.{index:03d}",
                smarts,
                "valid_core",
                "accept",
                (
                    "atom_logic",
                    "precedence",
                    "recursive" if "$(" in smarts else "boolean",
                ),
            )
        )

    bond_patterns = (
        "C-C",
        "C=C",
        "C#N",
        "C:C",
        "C~N",
        "C-,=C",
        "C=,#N",
        "C-,:c",
        "C=,~O",
        "c:c",
        "cc",
        "C-C-C",
        "C=C-C",
        "C#C-C",
        "N-C=O",
        "O=C-O",
        "C(=O)N",
        "C(-O)(-N)-C",
        "C1CCCCC1",
        "c1ccccc1",
        "C1=CC=CC=C1",
        "C%10CCCCC%10",
        "C1CC1C1CC1",
        "C1CCC2CCCCC2C1",
        "C.C",
        "[Na+].[Cl-]",
        "C.C.O",
        "[#6]-[#8]",
        "[#6]=[#8]",
        "[#6]~[#7]",
    )
    for index, smarts in enumerate(bond_patterns, start=1):
        cases.append(
            _parse_case(
                f"parse.graph.{index:03d}",
                smarts,
                "valid_core",
                "accept",
                (
                    "graph",
                    "bond",
                    "ring"
                    if any(ch.isdigit() for ch in smarts)
                    else "linear_or_branch",
                ),
            )
        )

    extension_patterns = (
        "[M]",
        "[!M]",
        "[Ln]",
        "[An]",
        "[NP1]",
        "[NP2]",
        "[NP3]",
        "[NP4]",
        "[NP5]",
        "[NP6]",
        "[NP7]",
        "[NP1-3]",
        "[NP3-5]",
        "[NP5-3]",
        "[NG1]",
        "[NG2]",
        "[NG3]",
        "[NG4]",
        "[NG5]",
        "[NG6]",
        "[NG7]",
        "[NG8]",
        "[NG9]",
        "[NG10]",
        "[NG11]",
        "[NG12]",
        "[NG13]",
        "[NG14]",
        "[NG15]",
        "[NG16]",
        "[NG17]",
        "[NG18]",
        "[NG1-2]",
        "[NG3-8]",
        "[NG8-3]",
        "[Ln]-O",
        "[An]~O",
        "[M;NP4]",
        "[M;NG8-12]",
        "[!M;#6]",
    )
    for index, smarts in enumerate(extension_patterns, start=1):
        cases.append(
            _parse_case(
                f"parse.extension.{index:03d}",
                smarts,
                "valid_extension",
                "accept",
                ("hotpot_extension", "metal_or_periodic_table"),
            )
        )
    return tuple(cases)


def _build_invalid_parse_cases() -> Tuple[Mapping[str, object], ...]:
    cases: List[Mapping[str, object]] = []

    for number in range(1, 31):
        cases.append(
            _parse_case(
                f"invalid.unclosed_bracket.{number:03d}",
                f"[#{number}",
                "invalid_syntax",
                "reject",
                ("unclosed_bracket", "atomic_number"),
                ("tokenize",),
            )
        )
    for number in range(1, 31):
        atom = f"[#{number}]"
        cases.append(
            _parse_case(
                f"invalid.unclosed_branch.{number:03d}",
                atom + "(",
                "invalid_syntax",
                "reject",
                ("unclosed_branch",),
                ("query_compile",),
            )
        )
        cases.append(
            _parse_case(
                f"invalid.unmatched_branch.{number:03d}",
                atom + ")",
                "invalid_syntax",
                "reject",
                ("unmatched_branch",),
                ("query_compile",),
            )
        )
        cases.append(
            _parse_case(
                f"invalid.unclosed_ring.{number:03d}",
                atom + str(number % 10),
                "invalid_syntax",
                "reject",
                ("unclosed_ring",),
                ("query_compile",),
            )
        )

    for atom_index, atom in enumerate(("C", "N", "O", "[C]", "[#6]"), start=1):
        for bond_index, bond in enumerate(("-", "=", "#", ":", "~"), start=1):
            cases.append(
                _parse_case(
                    f"invalid.dangling_bond.{atom_index:02d}.{bond_index:02d}",
                    atom + bond,
                    "invalid_syntax",
                    "reject",
                    ("dangling_bond", f"bond_{bond}"),
                    ("query_compile",),
                )
            )

    consecutive_bonds = (
        "C-=O",
        "C-#N",
        "C-:c",
        "C-~N",
        "C=-O",
        "C=#N",
        "C=:c",
        "C=~N",
        "C#-C",
        "C#=C",
        "C#:c",
        "C#~N",
        "C:-C",
        "C:=C",
        "C:#C",
        "C:~N",
        "C~-C",
        "C~=C",
        "C~#N",
        "C~:c",
    )
    for index, smarts in enumerate(consecutive_bonds, start=1):
        cases.append(
            _parse_case(
                f"invalid.consecutive_bond.{index:03d}",
                smarts,
                "invalid_syntax",
                "reject",
                ("consecutive_bond",),
                ("query_compile",),
            )
        )

    malformed_logic = (
        "[C,]",
        "[,C]",
        "[C,,N]",
        "[C;]",
        "[;C]",
        "[C;;N]",
        "[C&]",
        "[&C]",
        "[C&&N]",
        "[!]",
        "[C;!]",
        "[C,!]",
        "[C&!]",
        "[C;,,N]",
        "[C,&N]",
        "[C;,N]",
        "[C&;N]",
        "[C,;N]",
        "[$()]",
        "[C;$()]",
        "[C;$([C)]",
        "[C;$([C])",
        "[C;$((C)]",
    )
    for index, smarts in enumerate(malformed_logic, start=1):
        cases.append(
            _parse_case(
                f"invalid.logic.{index:03d}",
                smarts,
                "invalid_syntax",
                "reject",
                ("atom_logic", "missing_operand"),
                ("tokenize", "query_compile"),
            )
        )

    malformed_maps = (
        "[C:]",
        "[C:-1]",
        "[C:+1]",
        "[C:abc]",
        "[C:1:2]",
        "[C::1]",
        "[C:1x]",
        "[C: 1]",
        "[:1]",
        "[C:\u0000]",
    )
    for index, smarts in enumerate(malformed_maps, start=1):
        cases.append(
            _parse_case(
                f"invalid.atom_map.{index:03d}",
                smarts,
                "invalid_syntax",
                "reject",
                ("atom_map", "malformed"),
                ("tokenize", "query_compile"),
            )
        )

    punctuation = ("", " ", ".", "..", "C.", ".C", "C..N", "(", ")", "[]")
    for index, smarts in enumerate(punctuation, start=1):
        cases.append(
            _parse_case(
                f"invalid.empty_or_punctuation.{index:03d}",
                smarts,
                "invalid_syntax",
                "reject",
                ("empty_component", "punctuation"),
                ("tokenize", "query_compile"),
            )
        )
    return tuple(cases)


def _append_semantic_pairs(
    cases: List[Mapping[str, object]],
    prefix: str,
    smarts: str,
    targets: Iterable[Tuple[str, bool]],
    features: Sequence[str],
) -> None:
    for index, (target, expected) in enumerate(targets, start=1):
        cases.append(
            _semantic_case(
                f"semantic.{prefix}.{index:03d}", smarts, target, expected, features
            )
        )


def _build_semantic_cases() -> Tuple[Mapping[str, object], ...]:
    cases: List[Mapping[str, object]] = []

    for atomic_number, symbol in enumerate(ELEMENT_SYMBOLS, start=1):
        other_symbol = ELEMENT_SYMBOLS[atomic_number % len(ELEMENT_SYMBOLS)]
        targets = ((f"[{symbol}]", True), (f"[{other_symbol}]", False))
        _append_semantic_pairs(
            cases,
            f"element.{atomic_number:03d}",
            f"[{symbol}]",
            targets,
            ("element_identity", "positive_and_near_negative"),
        )
        _append_semantic_pairs(
            cases,
            f"atomic_number.{atomic_number:03d}",
            f"[#{atomic_number}]",
            targets,
            ("atomic_number", "positive_and_near_negative"),
        )

    semantic_rules = (
        (
            "charge.ammonium",
            "[N+]",
            (("[NH4+]", True), ("N", False), ("[NH2-]", False)),
            ("formal_charge", "hydrogen"),
        ),
        (
            "charge.oxide",
            "[O-]",
            (("[O-]", True), ("O", False), ("[O+]", False)),
            ("formal_charge",),
        ),
        (
            "charge.carbon",
            "[C-]",
            (("[CH3-]", True), ("C", False), ("[CH3+]", False)),
            ("formal_charge",),
        ),
        (
            "charge.double_positive",
            "[N+2]",
            (("[N+2]", True), ("[N+]", False), ("N", False)),
            ("formal_charge", "multi_digit"),
        ),
        (
            "bond.single_cc",
            "C-C",
            (("CC", True), ("C=C", False), ("C#C", False), ("CO", False)),
            ("single_bond", "identity"),
        ),
        (
            "bond.double_cc",
            "C=C",
            (("C=C", True), ("CC", False), ("C#C", False), ("C=O", False)),
            ("double_bond", "identity"),
        ),
        (
            "bond.triple_cc",
            "C#C",
            (("C#C", True), ("CC", False), ("C=C", False), ("C#N", False)),
            ("triple_bond", "identity"),
        ),
        (
            "bond.carbonyl",
            "C=O",
            (("CC=O", True), ("CCO", False), ("C=N", False), ("O=O", False)),
            ("double_bond", "heteroatom"),
        ),
        (
            "bond.nitrile",
            "C#N",
            (("CC#N", True), ("CC=N", False), ("N#N", False), ("C#C", False)),
            ("triple_bond", "heteroatom"),
        ),
        (
            "bond.any_co",
            "C~O",
            (("CO", True), ("C=O", True), ("C#O", True), ("CN", False)),
            ("any_bond",),
        ),
        (
            "bond.or_cc",
            "C-,=C",
            (("CC", True), ("C=C", True), ("C#C", False), ("CO", False)),
            ("bond_or",),
        ),
        (
            "bond.aromatic",
            "c:c",
            (("c1ccccc1", True), ("C1CCCCC1", False), ("C=C", False)),
            ("aromatic_bond",),
        ),
        (
            "graph.branch",
            "C(=O)N",
            (("CC(=O)N", True), ("CC(=O)O", False), ("CCN", False), ("NC=O", True)),
            ("branch", "carbonyl"),
        ),
        (
            "graph.ring5",
            "C1CCCC1",
            (("C1CCCC1", True), ("C1CCCCC1", False), ("CCCCC", False)),
            ("ring_closure", "ring_size"),
        ),
        (
            "graph.ring6",
            "C1CCCCC1",
            (("C1CCCCC1", True), ("C1CCCC1", False), ("CCCCCC", False)),
            ("ring_closure", "ring_size"),
        ),
        (
            "graph.disconnected",
            "[Na+].[Cl-]",
            (
                ("[Na+].[Cl-]", True),
                ("[Na+]", False),
                ("[Cl-]", False),
                ("[K+].[Cl-]", False),
            ),
            ("disconnected_query", "charge"),
        ),
        (
            "aromatic.carbon",
            "[c]",
            (("c1ccccc1", True), ("C1CCCCC1", False), ("C", False)),
            ("aromaticity",),
        ),
        (
            "aromatic.aliphatic",
            "[C]",
            (("C1CCCCC1", True), ("c1ccccc1", False), ("C", True)),
            ("aromaticity",),
        ),
        (
            "aromatic.nitrogen",
            "[n]",
            (("n1ccccc1", True), ("N1CCCCC1", False), ("c1ccccc1", False)),
            ("aromaticity", "heteroatom"),
        ),
        (
            "primitive.degree1",
            "[C;D1]",
            (("CC", True), ("C(C)C", True), ("C", False)),
            ("degree",),
        ),
        (
            "primitive.degree4",
            "[C;D4]",
            (("C(C)(C)(C)C", True), ("C(C)(C)C", False), ("CC", False)),
            ("degree",),
        ),
        (
            "primitive.connectivity4",
            "[C;X4]",
            (("C", True), ("CC", True), ("C=C", False)),
            ("connectivity", "implicit_hydrogen"),
        ),
        (
            "primitive.valence4",
            "[C;v4]",
            (("C", True), ("C=C", True), ("C#N", True), ("[C-]", False)),
            ("valence",),
        ),
        (
            "primitive.hydrogen3",
            "[C;H3]",
            (("CC", True), ("C=C", False), ("C#C", False)),
            ("hydrogen_count",),
        ),
        (
            "primitive.ring",
            "[C;R]",
            (("C1CCCCC1", True), ("CCCCCC", False), ("c1ccccc1", False)),
            ("ring_membership", "aliphatic"),
        ),
        (
            "primitive.ring0",
            "[C;R0]",
            (("CCCCCC", True), ("C1CCCCC1", False), ("c1ccccc1", False)),
            ("ring_count",),
        ),
        (
            "primitive.ring5",
            "[C;r5]",
            (("C1CCCC1", True), ("C1CCCCC1", False), ("CCCCC", False)),
            ("ring_size",),
        ),
        (
            "logic.or",
            "[C,N]",
            (("C", True), ("N", True), ("O", False)),
            ("atom_logic", "or"),
        ),
        (
            "logic.not",
            "[!C]",
            (("N", True), ("O", True), ("C", False), ("c1ccccc1", True)),
            ("atom_logic", "not"),
        ),
        (
            "logic.double_not",
            "[!!C]",
            (("C", True), ("N", False), ("c1ccccc1", False)),
            ("atom_logic", "double_not"),
        ),
        (
            "logic.low_and",
            "[C,N;H1]",
            (("C#C", True), ("N=C", True), ("C", False), ("N", False)),
            ("atom_logic", "precedence"),
        ),
        (
            "recursive.carbonyl",
            "[C;$([C]=O)]",
            (("CC=O", True), ("CCO", False), ("C=N", False)),
            ("recursive", "anchored"),
        ),
        (
            "recursive.nonamide_n",
            "[N;!$(N-C=O)]",
            (("CCN", True), ("CC(=O)N", False), ("N", True)),
            ("recursive", "negation"),
        ),
        (
            "extension.metal",
            "[M]",
            (("[Fe]", True), ("[Na]", True), ("C", False)),
            ("hotpot_extension", "metal"),
        ),
        (
            "extension.nonmetal",
            "[!M]",
            (("C", True), ("O", True), ("[Fe]", False), ("[Na]", False)),
            ("hotpot_extension", "metal"),
        ),
        (
            "extension.lanthanide",
            "[Ln]",
            (("[Eu]", True), ("[La]", True), ("[Am]", False), ("C", False)),
            ("hotpot_extension", "lanthanide"),
        ),
        (
            "extension.actinide",
            "[An]",
            (("[Am]", True), ("[U]", True), ("[Eu]", False), ("C", False)),
            ("hotpot_extension", "actinide"),
        ),
        (
            "extension.period3",
            "[NP3]",
            (("[Na]", True), ("[Cl]", True), ("[C]", False), ("[K]", False)),
            ("hotpot_extension", "period"),
        ),
        (
            "extension.group1",
            "[NG1]",
            (("[Na]", True), ("[K]", True), ("[Mg]", False), ("[Cl]", False)),
            ("hotpot_extension", "group"),
        ),
    )
    for prefix, smarts, targets, features in semantic_rules:
        _append_semantic_pairs(cases, prefix, smarts, targets, features)

    # Map-labelled queries ensure mapping metadata does not alter semantics.
    for map_number in range(1, 33):
        _append_semantic_pairs(
            cases,
            f"atom_map.{map_number:03d}",
            f"[C:{map_number}]",
            (("CC", True), ("N", False)),
            ("atom_map", "metadata", "positive_and_near_negative"),
        )
    return tuple(cases)


VALID_PARSE_CASES = _build_valid_parse_cases()
INVALID_PARSE_CASES = _build_invalid_parse_cases()
SEMANTIC_CASES = _build_semantic_cases()

CORPUS_COUNTS = {
    "valid_parse": len(VALID_PARSE_CASES),
    "invalid_parse": len(INVALID_PARSE_CASES),
    "semantic": len(SEMANTIC_CASES),
    "total": len(VALID_PARSE_CASES) + len(INVALID_PARSE_CASES) + len(SEMANTIC_CASES),
}


__all__ = [
    "CORPUS_COUNTS",
    "CURATED_SOURCE",
    "DIALECT",
    "ELEMENT_SYMBOLS",
    "INVALID_PARSE_CASES",
    "SCHEMA_VERSION",
    "SEMANTIC_CASES",
    "VALID_PARSE_CASES",
]
