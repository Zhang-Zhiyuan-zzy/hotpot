"""Self-tests which prevent the conformance adapter becoming the oracle bug."""

import pytest

from . import adapter


pytestmark = [pytest.mark.smarts_core]


def test_parse_reports_summary_and_preserves_atom_maps():
    result = adapter.parse_smarts("[C:1](=O)N.[Na+]")

    assert result.accepted
    assert result.phase == "query_compile"
    assert result.error_code is None
    assert result.error_position is None
    assert result.ast_or_query_summary == adapter.QuerySummary(
        atom_count=4,
        bond_count=2,
        component_count=2,
        atom_map_numbers=(1, None, None, None),
    )


def test_parse_separates_tokenize_compile_and_unsupported_failures():
    lexical = adapter.parse_smarts("C\u0000")
    syntactic = adapter.parse_smarts("C(")
    unsupported = adapter.parse_smarts("C/C")

    assert (lexical.accepted, lexical.phase, lexical.error_code) == (
        False,
        "tokenize",
        "invalid_syntax",
    )
    assert lexical.error_position is not None
    assert (syntactic.accepted, syntactic.phase, syntactic.error_code) == (
        False,
        "query_compile",
        "invalid_syntax",
    )
    assert syntactic.error_position is None
    assert (unsupported.accepted, unsupported.phase, unsupported.error_code) == (
        False,
        "query_compile",
        "unsupported_feature",
    )


def test_parse_does_not_swallow_unknown_exceptions(monkeypatch):
    def fail(_text):
        raise RuntimeError("unexpected parser failure")

    monkeypatch.setattr(adapter, "tokenize", fail)
    with pytest.raises(RuntimeError, match="unexpected parser failure"):
        adapter.parse_smarts("C")


def test_prepare_target_uses_hotpot_reader_and_stable_current_indices():
    result = adapter.prepare_target("CCO")

    assert result.accepted
    assert result.phase == "target_prepare"
    assert result.error_code is None
    assert result.atom_identity_map == (0, 1, 2)
    assert [atom.atomic_number for atom in result.molecule.atoms] == [6, 6, 8]


def test_prepare_target_structures_known_reader_rejection():
    result = adapter.prepare_target("not-a-smiles")

    assert not result.accepted
    assert result.phase == "target_prepare"
    assert result.error_code == "invalid_target"
    assert result.atom_identity_map == ()


def test_match_normalizes_query_order_and_target_sets_separately():
    result = adapter.match_smarts("C-C", "CCC")

    assert result.matched
    assert result.raw_embedding_count == 4
    assert result.embeddings == ((0, 1), (1, 0), (1, 2), (2, 1))
    assert result.unique_target_atom_sets == ((0, 1), (1, 2))
    assert result.truncated is None


def test_match_retains_failure_phase():
    query_failure = adapter.match_smarts("C(", "CC")
    target_failure = adapter.match_smarts("C", "not-a-smiles")

    assert not query_failure.query_accepted
    assert query_failure.target_accepted is None
    assert query_failure.phase == "query_compile"
    assert target_failure.query_accepted
    assert target_failure.target_accepted is False
    assert target_failure.phase == "target_prepare"


def test_search_is_an_explicit_linear_scan_baseline():
    records = (
        {"id": "ethanol", "smiles": "CCO"},
        {"id": "amine", "smiles": "CCN"},
        {"id": "invalid", "smiles": "not-a-smiles"},
    )
    result = adapter.search_smarts("C-O", records)

    assert result.strategy == "linear_scan"
    assert result.matched_record_ids == ("ethanol",)
    assert set(result.per_record_matches) == {"ethanol", "amine", "invalid"}
    assert result.errors == {"invalid": "invalid_target"}
    assert result.truncated is None


def test_unexposed_options_fail_explicitly():
    with pytest.raises(NotImplementedError, match="use_chirality"):
        adapter.match_smarts("C", "C", {"use_chirality": False})
