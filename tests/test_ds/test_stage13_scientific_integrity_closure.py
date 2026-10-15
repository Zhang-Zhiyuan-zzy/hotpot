from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotpot.ds.gates import BenchmarkGate
from hotpot.ds.public._assessment import (
    assess_case,
    assess_evidence,
    summarize_assessments,
    validate_identity_schema,
)
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA
from hotpot.ds.public.bundle import Bundle


def _case() -> dict:
    return {
        "id": "case-1",
        "local_id": "case-1",
        "public_index": 0,
        "identifiers": {"sample": "one"},
        "metadata": {"supplier": "test"},
        "core_info": {"sample": "one"},
        "structures": [],
        "observations": [
            {
                "observation_id": "observation-1",
                "property": "spin",
                "value": 3,
            }
        ],
        "claims": [],
        "evidence": [],
        "verifications": [],
        "attestations": [],
    }


def _primary_evidence(**values) -> dict:
    result = {
        "evidence_id": "evidence-1",
        "target": "observation-1",
        "relation": "supports",
        "source_kind": "primary_literature",
        "doi": "10.1000/example",
        "source_lineage_id": "experiment:one",
        "locator": "Table 1",
    }
    result.update(values)
    return result


def _human_verification(**values) -> dict:
    result = {
        "verification_id": "verification-1",
        "evidence_id": "evidence-1",
        "method": "Human",
        "verifier": "curator@example.org",
    }
    result.update(values)
    return result


def _assess_case(case: dict, assets=()) -> dict:
    return assess_case(
        case,
        core_labels=["spin"],
        core_info=["sample"],
        required_structure_roles=[],
        asset_assessments=assets,
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
    )


def _bundle_with_file_evidence(
    tmp_path: Path, *, present: bool, expected_sha256: str | None = None
) -> Bundle:
    root = tmp_path / ("present" if present else "missing")
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    case = _case()
    evidence = _primary_evidence(
        source="file:evidence/paper.txt",
        path="evidence/paper.txt",
    )
    if expected_sha256 is not None:
        evidence["expected_sha256"] = expected_sha256
    case["evidence"] = [evidence]
    case["verifications"] = [_human_verification()]
    intake.write_text(json.dumps(case) + "\n", encoding="utf-8")
    if present:
        source = root / "evidence" / "paper.txt"
        source.write_text("spin = 3\n", encoding="utf-8")
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "ds_name": "integrity",
                "release_id": f"integrity-{'present' if present else 'missing'}",
                "release_serial": 1,
                "version": "1.0",
                "title": "Scientific integrity fixture",
                "description": "Scientific integrity fixture",
                "license": "CC0-1.0",
                "access_policy": "open",
                "provenance": {"created_by": "test"},
                "core_labels": ["spin"],
                "core_info": ["sample"],
                "intake": {
                    "format": "jsonl",
                    "path": "evidence/intake/cases.jsonl",
                },
                "identity_schema": {
                    "components": [
                        {
                            "key": "sample",
                            "path": "core_info.sample",
                            "required": True,
                            "normalizer": "text",
                        }
                    ]
                },
                "required_structure_roles": [],
            }
        ),
        encoding="utf-8",
    )
    return Bundle(root, "integrity", db=tmp_path / "datasets.sqlite3")


def test_generation_asset_assessment_controls_file_evidence_support(
    tmp_path: Path,
) -> None:
    present = _bundle_with_file_evidence(tmp_path / "ok", present=True)
    present.synchronize()
    assert present.cases[0].evidence_level == "E3"

    missing = _bundle_with_file_evidence(tmp_path / "absent", present=False)
    missing.synchronize()
    assert missing.cases[0].evidence_level == "E0"
    assert missing.cases[0].archive_class == "ordinary"
    assert "evidence_asset_missing" in {
        item["code"] for item in missing.cases[0].reasons
    }

    mismatch = _bundle_with_file_evidence(
        tmp_path / "mismatch", present=True, expected_sha256="0" * 64
    )
    mismatch.synchronize()
    assert mismatch.cases[0].evidence_level == "E0"
    assert "evidence_asset_hash_mismatch" in {
        item["code"] for item in mismatch.cases[0].reasons
    }
    assessment = mismatch.cases[0].assessment["asset_assessments"][0]
    assert assessment["record_id"] == "evidence-1"
    assert assessment["collection"] == "evidence"


def test_http_scheme_never_implies_primary_literature() -> None:
    case = _case()
    case["evidence"] = [
        {
            "evidence_id": "evidence-1",
            "target": "observation-1",
            "relation": "supports",
            "url": "https://example.org/report",
            "locator": "p. 1",
        }
    ]
    case["verifications"] = [_human_verification()]

    undeclared = assess_evidence(case)
    assert undeclared["aggregate"] == "E2"
    assert undeclared["reasons"][0]["code"] == "evidence_source_kind_undeclared"

    case["evidence"][0]["source_kind"] = "secondary_literature"
    assert assess_evidence(case)["aggregate"] == "E2"
    case["evidence"][0]["source_kind"] = "primary_literature"
    assert assess_evidence(case)["aggregate"] == "E3"


def test_ai_verification_requires_explicit_tool_identity_and_version() -> None:
    case = _case()
    case["evidence"] = [_primary_evidence()]
    case["verifications"] = [
        {
            "verification_id": "verification-1",
            "evidence_id": "evidence-1",
            "method": "AI",
            "verifier": "automation-service",
            "procedure_version": "primary-review/current",
            "verified_at": "2026-10-15T09:00:00+08:00",
        }
    ]

    assert assess_evidence(case)["aggregate"] == "E2"
    invalid = _assess_case(case)
    assert invalid["archive_class"] == "incomplete"
    assert invalid["integrity_status"] == "invalid"
    assert "ai_verification_metadata_incomplete" in {
        item["code"] for item in invalid["reasons"]
    }

    case["verifications"][0].update(
        {"tool": "review-agent", "tool_version": "2026.10"}
    )
    assert assess_evidence(case)["aggregate"] == "E3"


def test_invalid_source_kind_is_incomplete_but_undeclared_kind_only_lowers_evidence() -> None:
    case = _case()
    case["evidence"] = [_primary_evidence(source_kind="web")]
    case["verifications"] = [_human_verification()]
    invalid = _assess_case(case)
    assert invalid["archive_class"] == "incomplete"
    assert invalid["integrity_status"] == "invalid"

    del case["evidence"][0]["source_kind"]
    undeclared = _assess_case(case)
    assert undeclared["archive_class"] == "ordinary"
    assert undeclared["evidence_level"] == "E2"


@pytest.mark.parametrize(
    ("normalizer", "value_kind"),
    [
        ("text", "number"),
        ("casefold", "json"),
        ("number", "text"),
        ("boolean", "number"),
        ("set", "json"),
        ("json", "set"),
    ],
)
def test_identity_normalizer_value_kind_pairs_are_closed(
    normalizer: str, value_kind: str
) -> None:
    with pytest.raises(ValueError, match="value_kind does not match"):
        validate_identity_schema(
            {
                "components": [
                    {
                        "key": "sample",
                        "path": "core_info.sample",
                        "normalizer": normalizer,
                        "value_kind": value_kind,
                    }
                ]
            }
        )


def test_assessment_summary_has_only_named_decision_sections() -> None:
    case = _case()
    first = _assess_case(case)
    case["attestations"] = [
        {
            "attestation_id": "attestation-1",
            "authority": "User",
            "actor": "owner",
            "approved": True,
        }
    ]
    second = _assess_case(case)

    summary = summarize_assessments([first, second])

    assert set(summary) == {
        "cases",
        "integrity",
        "completeness",
        "evidence",
        "quality",
        "benchmark",
        "attestation",
        "conflict",
        "reasons",
    }
    assert summary["cases"] == 2
    assert summary["integrity"]["status"] == {"valid": 2}
    assert summary["completeness"]["status"] == {"complete": 2}
    assert summary["attestation"]["status"] == {"approved": 1, "none": 1}
    assert summary["conflict"]["status"] == {"clear": 2}


def test_case_api_rejects_ai_verification_without_tool_fields_atomically(
    tmp_path: Path,
) -> None:
    bundle = _bundle_with_file_evidence(tmp_path, present=True)
    case = bundle.cases[0]
    before = bundle.status["source_revision"]

    with pytest.raises(ValueError, match="tool, tool_version"):
        case.verify_evidence(
            "evidence-1",
            method="AI",
            verifier="automation-service",
            details={"procedure_version": "review/current"},
        )

    assert bundle.status["source_revision"] == before
    assert len(bundle._state.get_case("case-1")["verifications"]) == 1
