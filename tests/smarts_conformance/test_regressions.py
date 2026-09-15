"""Strict intended-contract tests for defects found by the conformance audit.

These tests intentionally remain red until production fixes are authorized.
They are not xfailed because doing so would hide the strict conformance result.
"""

import pytest
from openbabel import openbabel as ob

import hotpot as hp


pytestmark = [pytest.mark.smarts_core, pytest.mark.smarts_known_failure]


def accepts(smarts):
    try:
        hp.Substructure.from_smarts(smarts)
    except (ValueError, NotImplementedError):
        return False
    return True


def has_match(smiles, smarts):
    return bool(hp.read_mol(smiles, "smi").search_substructure(smarts))


def test_smarts_parse_boundary_001_rejects_empty_query_and_empty_components():
    malformed = ("", " ", ".", ".C", "C.", "C..O")

    assert [query for query in malformed if accepts(query)] == [], (
        "SMARTS-PARSE-BOUNDARY-001"
    )


def test_smarts_parse_graph_002_rejects_leading_bond_self_loop_and_duplicate_edge():
    malformed = ("=C", "C11", "C1C1", "C-(C)")

    assert [query for query in malformed if accepts(query)] == [], (
        "SMARTS-PARSE-GRAPH-002"
    )


def test_smarts_recursive_001_rejects_empty_recursive_query():
    malformed = ("[$()]", "[C;$()]")

    assert [query for query in malformed if accepts(query)] == [], (
        "SMARTS-RECURSIVE-001"
    )


@pytest.mark.parametrize("smarts", ["[!]", "[!!]", "[C!]", "[C&!]"])
def test_smarts_diagnostic_001_never_leaks_index_error(smarts):
    with pytest.raises(ValueError, match="."):
        hp.Substructure.from_smarts(smarts)


def test_smarts_atom_wildcard_001_supports_documented_bare_a_and_A():
    assert has_match("c1ccccc1", "a"), "SMARTS-ATOM-WILDCARD-001: bare a"
    assert has_match("CC", "A"), "SMARTS-ATOM-WILDCARD-001: bare A"


def test_smarts_atom_element_001_all_bracket_symbols_equal_atomic_number_queries():
    mismatches = []
    for atomic_number in range(1, 119):
        symbol = ob.GetSymbol(atomic_number)
        target = f"[{symbol}]"
        atomic_number_match = has_match(target, f"[#{atomic_number}]")
        try:
            symbol_match = has_match(target, target)
        except ValueError:
            symbol_match = False
        if symbol_match != atomic_number_match:
            mismatches.append((atomic_number, symbol))

    assert mismatches == [], "SMARTS-ATOM-ELEMENT-001"


def test_smarts_atom_h_001_classifies_lowercase_h_and_x_as_unsupported():
    for smarts in ("[h]", "[h1]", "[x]", "[x2]"):
        with pytest.raises(NotImplementedError):
            hp.Substructure.from_smarts(smarts)


def test_smarts_bond_arom_001_explicit_numeric_bonds_exclude_aromatic_bonds():
    false_positives = [
        smarts
        for smarts in ("*-*", "*=*", "[#6]-[#6]", "[#6]=[#6]", "c-c", "c=c")
        if has_match("c1ccccc1", smarts)
    ]

    assert false_positives == [], "SMARTS-BOND-AROM-001"


def test_smarts_ring_r0_001_r0_matches_acyclic_atoms():
    assert has_match("C", "[C;r0]"), "SMARTS-RING-r0-001"


def test_smarts_ext_range_001_rejects_incomplete_and_out_of_range_extensions():
    malformed = ("[NP]", "[NP0]", "[NP8]", "[NG0]", "[NG19]")

    assert [query for query in malformed if accepts(query)] == [], (
        "SMARTS-EXT-RANGE-001"
    )


@pytest.mark.parametrize(
    ("smiles", "smarts", "expected_bond_count"),
    [("CC", "C.C", 0), ("C1CC1", "CCC", 2)],
)
def test_smarts_hit_bonds_001_contains_only_query_mapped_edges(
    smiles, smarts, expected_bond_count
):
    hits = hp.read_mol(smiles, "smi").search_substructure(smarts)

    assert len(hits) == 1
    assert len(hits[0].bonds) == expected_bond_count, "SMARTS-HIT-BONDS-001"


def test_smarts_query_atom_001_can_be_constructed_from_a_hotpot_atom():
    atom = hp.read_mol("C", "smi").atoms[0]

    query_atom = hp.QueryAtom.from_atom(atom)

    assert query_atom.match(atom), "SMARTS-QUERY-ATOM-001"
