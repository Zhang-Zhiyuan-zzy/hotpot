"""Strict execution of every generated corpus case.

Failures are aggregated so one malformed input cannot hide later findings.
"""

import json
from pathlib import Path

import pytest

from .audit_corpus import build_report


pytestmark = pytest.mark.smarts_core


KNOWN_MISMATCHES_PATH = Path(__file__).with_name("corpus") / "known_mismatches.json"


@pytest.fixture(scope="module")
def corpus_report():
    return build_report()


def _expected_mismatches():
    document = json.loads(KNOWN_MISMATCHES_PATH.read_text())
    assert document["schema_version"] == 1
    expected = {}
    for issue_id, issue in document["issues"].items():
        assert issue["reason"]
        assert issue["owner"]
        assert issue["added"]
        assert issue["exit_condition"]
        for corpus_name, case_ids in issue["cases"].items():
            classification = issue["classifications"][corpus_name]
            for case_id in case_ids:
                key = (corpus_name, case_id)
                assert key not in expected
                expected[key] = (classification, issue_id)
    return expected


def test_full_corpus_has_exactly_the_reviewed_known_mismatches(corpus_report):
    expected = _expected_mismatches()
    actual = {}
    for corpus_name, failures in corpus_report["failures"].items():
        for failure in failures:
            key = (corpus_name, failure["case_id"])
            issue_id = expected.get(key, (None, "<unreviewed>"))[1]
            actual[key] = (failure["classification"], issue_id)

    assert actual == expected


@pytest.mark.smarts_known_failure
@pytest.mark.parametrize("corpus_name", ["valid_parse", "invalid_parse", "semantic"])
def test_full_corpus_matches_frozen_contract(corpus_report, corpus_name):
    failures = corpus_report["failures"][corpus_name]
    summary = [
        {
            "case_id": failure["case_id"],
            "classification": failure["classification"],
            "smarts": failure["smarts"],
            "target_smiles": failure.get("target_smiles"),
            "actual": failure["actual"],
        }
        for failure in failures
    ]

    assert summary == [], f"{corpus_name} conformance mismatches"
