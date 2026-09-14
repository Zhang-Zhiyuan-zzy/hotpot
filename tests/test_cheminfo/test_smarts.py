"""Focused tests for the NetworkX-backed SMARTS implementation."""

import pytest

import hotpot as hp
from hotpot.cheminfo.AImodels.mca.site_detection import NUCLEOPHILE_RULES
from hotpot.cheminfo.search.smarts import TokenType, parse_bracket_atom, tokenize


def molecule(smiles):
    return hp.read_mol(smiles, "smi")


def has_match(smiles, smarts):
    return bool(molecule(smiles).search_substructure(smarts))


def test_tokenizer_preserves_recursive_atoms_bond_or_and_components():
    tokens = tokenize("[#6;!$(C([OX2])N):1]=,:[#8].[Na+]")
    assert tokens == [
        (TokenType.BRACKET, "[#6;!$(C([OX2])N):1]"),
        (TokenType.BOND, "=,:"),
        (TokenType.BRACKET, "[#8]"),
        (TokenType.DOT, "."),
        (TokenType.BRACKET, "[Na+]"),
    ]


def test_atom_map_is_metadata_not_a_match_constraint():
    sub = hp.Substructure.from_smarts("[O:7]=[C:2]")
    assert [atom.map_number for atom in sub.query_atoms] == [7, 2]
    assert all("map_number" not in atom.kwargs for atom in sub.query_atoms)
    assert has_match("CC=O", "[O:7]=[C:2]")


def test_public_bracket_parser_returns_a_query_constraint():
    attrs = parse_bracket_atom("[#6;X4;H3:1]")
    query = hp.QueryAtom(**attrs)
    methyl = molecule("CC").atoms[0]
    assert query.match(methyl)


@pytest.mark.parametrize(
    ("smiles", "smarts", "expected"),
    [
        ("CCO", "[OX2H1]", True),
        ("CCO", "[OX1]", False),
        ("CCO", "[CD1H3]", True),
        ("CCO", "[CX4H3]", True),
        ("CC#N", "[NX1v3]", True),
        ("CC=O", "[H1R0]", True),
        ("c1ccccc1", "[cH1]", True),
        ("c1ccccc1", "[C]", False),
        ("C1CCCC1", "[C;R;r5]", True),
        ("CCO", "[C;R0]", True),
        ("[NH4+]", "[N+1H4X4]", True),
    ],
)
def test_atom_primitives_follow_smarts_graph_semantics(smiles, smarts, expected):
    assert has_match(smiles, smarts) is expected


def test_boolean_logic_and_anchored_recursive_smarts():
    assert has_match("CC(=O)N", "[O,N;!R]")
    assert has_match("CC(=O)N", "[#6;$([#6](=[#8])[#7]):1]")
    assert not has_match("CC(=O)N", "[#6;D3;!$([#6](=[#8])[#7])]")
    assert not has_match("CC(=O)N", "[N;!$(N-C=O)]")
    assert has_match("CCN", "[N;!$(N-C=O)]")


def test_bond_or_branch_ring_and_dot_components():
    assert has_match("CC=O", "[#8]=,:[#6]")
    assert has_match("c1ccccc1", "[#6]=,:[#6]")
    assert has_match("CC(=O)N", "C(=O)N")
    assert has_match("c1ccccc1", "c1ccccc1")
    assert has_match("[Na+].[O-]C(=O)C", "[Na+].[O-]C(=O)")


@pytest.mark.parametrize(
    ("smiles", "smarts", "expected"),
    [
        ("[Mn]", "[Mn]", True),
        ("[Mn]", "[M]", True),
        ("[Mn]", "[Mg]", False),
        ("[Mg]", "[Mg]", True),
        ("[Na]", "[Na]", True),
        ("[Np]", "[Np]", True),
        ("[Eu]", "[Ln]", True),
        ("[Eu]O", "[Ln]O", True),
        ("[Eu]", "[An]", False),
        ("[Am]", "[An]", True),
        ("[Am]O", "[An]O", True),
        ("C", "[!M]", True),
        ("[Na]", "[!M]", False),
        ("[Na]", "[NP3]", True),
        ("[Na]", "[NG1]", True),
        ("[Mn]", "[NP3-5]", True),
        ("[Mn]", "[NG3-8]", True),
    ],
)
def test_hotpot_metal_and_periodic_table_extensions(smiles, smarts, expected):
    assert has_match(smiles, smarts) is expected


def test_every_mca_rule_compiles_with_mapped_target_atom():
    assert len(NUCLEOPHILE_RULES) == 24
    for name, smarts in NUCLEOPHILE_RULES:
        sub = hp.Substructure.from_smarts(smarts)
        assert sub.query_atoms, name
        assert sub.query_atoms[0].map_number == 1, name


def test_mca_carboxylic_acid_recursive_h1_branch_matches_formic_acid():
    carboxylic_acid = dict(NUCLEOPHILE_RULES)["Carboxylic acid"]
    assert has_match("O=CO", carboxylic_acid)
    assert not has_match("CC=O", carboxylic_acid)
