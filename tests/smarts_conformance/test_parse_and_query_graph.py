import pytest
import networkx as nx

import hotpot as hp
from hotpot.cheminfo.search._smarts_syntax import tokenize_with_spans
from hotpot.cheminfo.search.smarts import TokenType, tokenize


pytestmark = pytest.mark.smarts_core


@pytest.mark.smarts_smoke
@pytest.mark.parametrize(
    ("smarts", "atom_count", "bond_count"),
    [
        ("C", 1, 0),
        ("[#6]", 1, 0),
        ("*", 1, 0),
        ("[a]", 1, 0),
        ("[A]", 1, 0),
        ("C(=O)N", 3, 2),
        ("C1CCCCC1", 6, 6),
        ("C%12CCCCC%12", 6, 6),
        ("[Na+].[O-]", 2, 0),
        ("[#6;$([#6](=[#8])[#7]):17]", 1, 0),
    ],
)
def test_representative_core_queries_compile(smarts, atom_count, bond_count):
    query = hp.Substructure.from_smarts(smarts)

    assert len(query.query_atoms) == atom_count
    assert len(query.query_bonds) == bond_count


def test_token_stream_preserves_nested_recursive_expression_and_graph_tokens():
    assert tokenize("[#6;!$(C([OX2])N):1]=,:[#8].[Na+]") == [
        (TokenType.BRACKET, "[#6;!$(C([OX2])N):1]"),
        (TokenType.BOND, "=,:"),
        (TokenType.BRACKET, "[#8]"),
        (TokenType.DOT, "."),
        (TokenType.BRACKET, "[Na+]"),
    ]


def test_internal_token_spans_retain_source_positions():
    tokens = tokenize_with_spans(" C(=O)N")

    assert [(token.type, token.text, token.start, token.end) for token in tokens] == [
        (TokenType.ATOM, "C", 1, 2),
        (TokenType.BRANCH_L, "(", 2, 3),
        (TokenType.BOND, "=", 3, 4),
        (TokenType.ATOM, "O", 4, 5),
        (TokenType.BRANCH_R, ")", 5, 6),
        (TokenType.ATOM, "N", 6, 7),
    ]


def test_public_parser_errors_keep_legacy_base_classes():
    with pytest.raises(hp.SmartsSyntaxError):
        hp.Substructure.from_smarts("C(")
    with pytest.raises(hp.UnsupportedSmartsError):
        hp.Substructure.from_smarts("C/C")
    assert issubclass(hp.SmartsSyntaxError, ValueError)
    assert issubclass(hp.UnsupportedSmartsError, NotImplementedError)


def test_branch_query_graph_has_expected_topology_and_bond_constraints():
    query = hp.Substructure.from_smarts("[C:1](=[O:2])[N:3]")
    graph = query.construct_graph()

    assert list(graph.nodes) == [0, 1, 2]
    assert {frozenset(edge) for edge in graph.edges} == {
        frozenset((0, 1)),
        frozenset((0, 2)),
    }
    assert [atom.map_number for atom in query.query_atoms] == [1, 2, 3]
    assert query.query_bonds[0].kwargs == {"bond_order": {2}}
    assert repr(query.query_bonds[1].kwargs["predicate"]) == "single-or-aromatic"


def test_aromatic_ring_query_graph_retains_aromatic_nodes_and_edges():
    query = hp.Substructure.from_smarts("c1ccccc1")
    graph = query.construct_graph()

    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 6
    assert all(atom.kwargs["is_aromatic"] == {True} for atom in query.query_atoms)
    assert all(bond.kwargs["is_aromatic"] == {True} for bond in query.query_bonds)


def test_disconnected_query_graph_has_two_components_and_no_synthetic_edge():
    query = hp.Substructure.from_smarts("[Na+].[O-]C=O")
    graph = query.construct_graph()

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 2
    assert sorted(len(component) for component in nx.connected_components(graph)) == [
        1,
        3,
    ]


def test_atom_map_is_query_metadata_only():
    mapped = hp.Substructure.from_smarts("[#6:7]-[#8:12]")
    unmapped = hp.Substructure.from_smarts("[#6]-[#8]")
    target = hp.read_mol("CCO", "smi")

    assert [atom.map_number for atom in mapped.query_atoms] == [7, 12]
    assert all("map_number" not in atom.kwargs for atom in mapped.query_atoms)
    assert [hit.atom_indices for hit in hp.Searcher(mapped).search(target)] == [
        hit.atom_indices for hit in hp.Searcher(unmapped).search(target)
    ]


@pytest.mark.parametrize(
    "smarts",
    [
        "C-",
        "C==O",
        "C()",
        "C(=)N",
        "C1CC",
        "[C",
        "[C;;H1]",
        "[C,,N]",
        "[C;$([N])",
        "C(.N)",
    ],
)
@pytest.mark.smarts_smoke
def test_unambiguous_malformed_queries_raise_value_error(smarts):
    with pytest.raises(ValueError):
        hp.Substructure.from_smarts(smarts)


@pytest.mark.parametrize("smarts", ["[13C]", "[C@H]", "[C@@H]", "C/C", r"C\C"])
def test_documented_unsupported_features_raise_not_implemented(smarts):
    with pytest.raises(NotImplementedError):
        hp.Substructure.from_smarts(smarts)
