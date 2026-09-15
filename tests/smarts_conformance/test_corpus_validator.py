"""Tests for corpus scale, schema, provenance, and duplicate governance."""

import re
from pathlib import Path

import pytest

from .corpus import (
    CORPUS_COUNTS,
    INVALID_PARSE_CASES,
    SEMANTIC_CASES,
    VALID_PARSE_CASES,
)
from .corpus_validator import (
    assert_valid_corpora,
    load_manifest,
    validate_corpora,
    validate_manifest,
)


pytestmark = [pytest.mark.smarts_core]


def _base_case(**updates):
    case = {
        "id": "test.case.001",
        "smarts": "C",
        "classification": "valid_core",
        "expected_outcome": "accept",
        "features": ("atom",),
        "dialect": "test",
        "source": {
            "kind": "curated",
            "name": "validator self-test",
            "version": "1",
            "license": "same-as-project",
            "reviewed": True,
        },
    }
    case.update(updates)
    return case


def test_bundled_corpus_is_valid_and_meets_initial_scale_goals():
    assert_valid_corpora()
    assert CORPUS_COUNTS["valid_parse"] >= 300
    assert CORPUS_COUNTS["invalid_parse"] >= 150
    assert CORPUS_COUNTS["semantic"] >= 300
    assert CORPUS_COUNTS == {
        "valid_parse": len(VALID_PARSE_CASES),
        "invalid_parse": len(INVALID_PARSE_CASES),
        "semantic": len(SEMANTIC_CASES),
        "total": len(VALID_PARSE_CASES)
        + len(INVALID_PARSE_CASES)
        + len(SEMANTIC_CASES),
    }


def test_manifest_matches_generated_corpus_and_records_provenance():
    manifest = load_manifest()

    assert validate_manifest() == ()
    assert manifest["provenance"]["reviewed"] is True
    assert manifest["provenance"]["external_corpus_imported"] is False


def test_feature_matrix_quantifies_every_declared_feature():
    text = Path(__file__).with_name("feature_matrix.yaml").read_text()
    feature_count = len(re.findall(r"^  - name:", text, flags=re.MULTILINE))
    count_blocks = re.findall(r"^    counts: \{([^}]*)\}", text, flags=re.MULTILINE)

    assert feature_count > 0
    assert len(count_blocks) == feature_count
    required_fields = (
        "positive_parse",
        "definite_negative",
        "semantic_pairs",
        "near_negative_pairs",
        "differential_cases",
        "fuzz_seeds",
    )
    for block in count_blocks:
        assert all(
            re.search(rf"\b{field}:\s*\d+\b", block) for field in required_fields
        )


def test_validator_reports_all_duplicate_and_contract_errors():
    first = _base_case()
    second = _base_case(expected_outcome="reject")
    issues = validate_corpora({"sample": (first, second)})
    codes = {issue.code for issue in issues}

    assert "duplicate_id" in codes
    assert "outcome" in codes


def test_validator_rejects_external_cases_without_full_provenance():
    case = _base_case(
        source={
            "kind": "external",
            "name": "upstream sample",
            "version": "1",
            "license": "unknown",
            "reviewed": False,
        }
    )
    issues = validate_corpora({"external": (case,)})

    assert {issue.code for issue in issues} == {"external_provenance"}
    assert len(issues) == 3


def test_validator_keeps_disputed_cases_out_of_core_assertions():
    disputed = _base_case(
        classification="dialect_disputed",
        expected_outcome=None,
        enforce_core=True,
    )
    issues = validate_corpora({"disputed": (disputed,)})

    assert any(issue.code == "disputed_core" for issue in issues)


def test_validator_requires_known_failure_exit_metadata():
    case = _base_case(known_failure={"issue_id": "SMARTS-1", "reason": "known"})
    issues = validate_corpora({"known": (case,)})

    assert sum(issue.code == "known_failure" for issue in issues) == 2


def test_assert_valid_corpora_raises_with_actionable_location():
    with pytest.raises(ValueError, match="broken:<missing>:required_field"):
        assert_valid_corpora({"broken": ({},)})
