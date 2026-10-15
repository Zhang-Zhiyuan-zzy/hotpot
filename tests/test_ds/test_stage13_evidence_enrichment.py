from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA
from hotpot.ds.public.bundle import Bundle


def _bundle(tmp_path: Path) -> Bundle:
    root = tmp_path / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    intake.write_text(
        json.dumps(
            {
                "id": "case-1",
                "local_id": "case-1",
                "public_index": 0,
                "core_info": {"sample": "one"},
                "observations": [
                    {
                        "observation_id": "observation-1",
                        "property": "spin",
                        "value": 3,
                        "schema_version": "current",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "ds_name": "evidence",
                "release_id": "evidence-r1",
                "release_serial": 1,
                "version": "1.0",
                "title": "Evidence enrichment fixture",
                "description": "Evidence enrichment fixture",
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
                "property_definitions": {
                    "spin": {
                        "type": "integer",
                        "schema_version": "current",
                        "unit": None,
                    }
                },
                "property_comparators": {"spin": {"kind": "exact"}},
            }
        ),
        encoding="utf-8",
    )
    return Bundle(root, "evidence", db=tmp_path / "datasets.sqlite3")


def test_public_case_api_can_reach_e4_with_auditable_provenance(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    case = bundle.cases[0]
    first = case.add_evidence(
        "observation-1",
        "doi:10.1000/first",
        locator="Table 1",
        details={
            "source_kind": "primary_literature",
            "source_lineage_id": "experiment:first",
        },
    )
    second = case.add_evidence(
        "observation-1",
        "https://doi.org/10.1000/second",
        locator="Table 2",
        details={
            "source_kind": "primary_literature",
            "source_lineage_id": "experiment:second",
        },
    )
    human = case.verify_evidence(
        first,
        method="Human",
        verifier="reviewer@example.org",
    )
    ai = case.verify_evidence(
        second,
        method="AI",
        verifier="review-tool/2026-10-14",
        details={
            "tool": "review-tool",
            "tool_version": "2026-10-14",
            "procedure_version": "primary-source-review/current",
            "verified_at": "2026-10-14T09:30:00+08:00",
        },
    )

    evidence = {item["evidence_id"]: item for item in case.evidence}
    assert evidence[first]["doi"] == "10.1000/first"
    assert evidence[first]["scientific_work_id"] == "doi:10.1000/first"
    assert evidence[second]["doi"] == "10.1000/second"
    assert evidence[second]["source_lineage_id"] == "experiment:second"
    facts = bundle._state.get_case("case-1")
    verifications = {
        item["verification_id"]: item for item in facts["verifications"]
    }
    assert verifications[human]["procedure_version"].endswith("/current")
    assert verifications[human]["verified_at"].endswith("Z")
    assert verifications[ai]["procedure_version"] == "primary-source-review/current"
    assert verifications[ai]["tool"] == "review-tool"
    assert verifications[ai]["tool_version"] == "2026-10-14"
    assert verifications[ai]["verified_at"] == "2026-10-14T09:30:00+08:00"

    bundle.synchronize()

    assert case.evidence_level == "E4"


def test_replacement_does_not_inherit_an_obsolete_source_identity(
    tmp_path: Path,
) -> None:
    case = _bundle(tmp_path).cases[0]
    old_id = case.add_evidence(
        "observation-1",
        "doi:10.1000/old",
        locator="p. 1",
        details={"source_lineage_id": "result:old"},
    )

    new_id = case.replace_evidence(
        old_id,
        "doi:10.1000/new",
        reason="Corrected citation",
        locator="p. 2",
        details={"source_lineage_id": "result:new"},
    )

    evidence = {item["evidence_id"]: item for item in case.evidence}
    assert evidence[old_id]["status"] == "superseded"
    assert evidence[old_id]["doi"] == "10.1000/old"
    assert evidence[new_id]["doi"] == "10.1000/new"
    assert evidence[new_id]["scientific_work_id"] == "doi:10.1000/new"
    assert evidence[new_id]["source_lineage_id"] == "result:new"


def test_evidence_detail_schema_is_closed_and_mutation_is_atomic(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    before = bundle.status["source_revision"]

    with pytest.raises(ValueError, match="unsupported evidence detail fields"):
        bundle.cases[0].add_evidence(
            "observation-1",
            "doi:10.1000/example",
            details={"evidence_level": "E4"},
        )

    assert bundle.status["source_revision"] == before
    assert bundle.cases[0].evidence == ()
