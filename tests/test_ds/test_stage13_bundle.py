from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA, BundleAlreadyBuiltError
from hotpot.ds.public import _bundle_state as bundle_state_module
from hotpot.ds.public.bundle import (
    Bundle,
    MissingCertificateError,
    SealedIntegrityError,
    SealedReleaseError,
    load_bundle,
)
from hotpot.ds.public.case import UnsynchronizedBundleError


def _case(case_id: str, index: int, *, approved: bool = False) -> dict:
    case = {
        "id": case_id,
        "local_id": f"local-{case_id}",
        "public_index": index,
        "identifiers": {"CSD_identifier": case_id.upper()},
        "core_info": {"sample": case_id},
        "observations": [
            {
                "observation_id": f"observation-{case_id}",
                "property": "spin",
                "value": index + 1,
            }
        ],
        "structures": [
            {
                "structure_id": f"structure-{case_id}",
                "role": "primary",
                "format": "smi",
                "path": f"structures/{case_id}.smi",
                "primary": True,
            }
        ],
    }
    if approved:
        case["attestations"] = [
            {
                "attestation_id": f"attestation-{case_id}",
                "authority": "User",
                "actor": "curator@example.test",
                "approved": True,
            }
        ]
    return case


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    cases = [_case("ordinary", 0), _case("benchmark", 1, approved=True)]
    intake.write_text(
        "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8"
    )
    structures = root / "structures"
    structures.mkdir()
    (structures / "ordinary.smi").write_text("O\n", encoding="utf-8")
    (structures / "benchmark.smi").write_text("N\n", encoding="utf-8")
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "demo",
        "release_id": "demo-r1",
        "release_serial": 1,
        "version": "1.0",
        "title": "Demo",
        "description": "Bundle lifecycle test",
        "license": "CC0-1.0",
        "access_policy": "open",
        "provenance": {"created_by": "test"},
        "core_labels": ["spin"],
        "core_info": ["sample"],
        "intake": {"format": "jsonl", "path": "evidence/intake/cases.jsonl"},
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
        "required_structure_roles": ["primary"],
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_build_load_synchronize_and_mutation_state_flow(tmp_path: Path) -> None:
    root = _root(tmp_path)
    database = tmp_path / "datasets.sqlite3"
    bundle = Bundle(root, "demo", db=database)

    assert bundle.mode == "new"
    assert bundle.status["next_action"] == "synchronize"
    assert len(bundle.cases) == 2
    with pytest.raises(UnsynchronizedBundleError):
        _ = bundle.summary
    with pytest.raises(BundleAlreadyBuiltError):
        Bundle(root, "demo", db=database)

    reopened = load_bundle(root, "demo", db=database)
    result = reopened.synchronize()
    assert result["is_synchronized"] is True
    assert reopened.is_synchronized is True
    assert reopened.status["next_action"] == "certify"
    assert len(reopened.ordinary) == 1
    assert len(reopened.benchmark) == 1
    assert reopened.cases[1].benchmark is True

    reopened.cases[0].set_core_info("note", "edited")
    assert reopened.is_synchronized is False
    assert reopened.status["next_action"] == "synchronize"
    with pytest.raises(UnsynchronizedBundleError):
        _ = reopened.cases[0].quality_level


def test_external_asset_change_advances_source_revision_on_sync(tmp_path: Path) -> None:
    root = _root(tmp_path)
    bundle = Bundle(root, "demo", db=tmp_path / "datasets.sqlite3")
    bundle.synchronize()
    original_revision = bundle.status["source_revision"]

    (root / "structures" / "ordinary.smi").write_text("[OH-]\n", encoding="utf-8")
    assert bundle.is_synchronized is False
    bundle.synchronize()

    assert bundle.status["source_revision"] == original_revision + 1
    assert bundle.is_synchronized is True


def test_runtime_policy_drift_requires_owner_migration_for_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = Bundle(_root(tmp_path), "demo", db=tmp_path / "datasets.sqlite3")
    bundle.synchronize()
    monkeypatch.setattr(
        bundle_state_module, "runtime_policy_is_current", lambda policy: False
    )

    assert bundle.status["runtime_policy_current"] is False
    assert bundle.status["is_synchronized"] is False
    assert bundle.status["next_action"] == "owner_policy_migration"


def test_runtime_policy_drift_requires_successor_for_sealed_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = Bundle(_root(tmp_path), "demo", db=tmp_path / "datasets.sqlite3")
    bundle.synchronize()
    bundle.write_certificate()
    bundle.ingest()
    monkeypatch.setattr(
        bundle_state_module, "runtime_policy_is_current", lambda policy: False
    )

    assert bundle.status["runtime_policy_current"] is False
    assert bundle.status["release_state"] == "sealed"
    assert bundle.status["next_action"] == "create_successor_release"


def test_stale_certificate_generation_is_retained_until_certificate_replacement(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    bundle = Bundle(root, "demo", db=tmp_path / "datasets.sqlite3")
    first = bundle.synchronize()["generation_id"]
    bundle.write_certificate()

    bundle.cases[0].set_core_info("curation_note", "reviewed")
    second = bundle.synchronize()["generation_id"]

    generations = root / "assessments" / "generations"
    assert first != second
    assert (generations / first).is_dir()
    assert (generations / second).is_dir()

    bundle.write_certificate()

    assert not (generations / first).exists()
    assert (generations / second).is_dir()


def test_certify_ingest_restore_retry_and_seal(tmp_path: Path) -> None:
    root = _root(tmp_path)
    database = tmp_path / "datasets.sqlite3"
    bundle = Bundle(root, "demo", db=database)
    bundle.synchronize()

    with pytest.raises(MissingCertificateError):
        bundle.ingest()
    certificate = bundle.write_certificate()
    restored = Bundle.from_certificate(certificate, db=database)
    assert restored.summary == bundle.summary

    first = restored.ingest()
    second = restored.ingest()
    assert second["receipt_id"] == first["receipt_id"]
    assert restored.status["release_state"] == "sealed"
    assert restored.status["ingest_ready"] is True
    assert restored.status["next_action"] == "none"
    assert restored.write_certificate() == certificate
    with pytest.raises(SealedReleaseError):
        restored.cases[0].set_core_info("forbidden", True)


def test_sealed_receipt_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    database = tmp_path / "datasets.sqlite3"
    bundle = Bundle(root, "demo", db=database)
    bundle.synchronize()
    bundle.write_certificate()
    bundle.ingest()

    certificate = root / "certificate.json"
    envelope = json.loads(certificate.read_text(encoding="utf-8"))
    envelope["envelope_sha256"] = "0" * 64
    certificate.write_text(json.dumps(envelope), encoding="utf-8")

    assert bundle.status["release_state"] == "sealed_integrity_error"
    with pytest.raises(UnsynchronizedBundleError):
        _ = bundle.cases[0].assessment
    with pytest.raises(SealedIntegrityError):
        bundle.cases[0].set_core_info("forbidden", True)
    with pytest.raises(SealedIntegrityError):
        bundle.write_certificate()


def test_sealed_release_uses_frozen_governance_and_reports_newer_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    database = tmp_path / "datasets.sqlite3"
    bundle = Bundle(root, "demo", db=database)
    bundle.synchronize()
    bundle.write_certificate()
    first = bundle.ingest()
    frozen = bundle.status

    monkeypatch.setattr(
        bundle_state_module,
        "_trusted_governance_binding",
        lambda: (
            frozen["assessment_governance_registry_revision"] + 1,
            "f" * 64,
        ),
    )

    status = bundle.status
    assert status["governance_registry_revision"] == (
        status["assessment_governance_registry_revision"] + 1
    )
    assert status["governance_update_available"] is True
    assert status["is_synchronized"] is True
    assert status["certificate_state"] == "current"
    assert status["ingest_ready"] is True
    assert bundle.certificate_stale is False

    retry = bundle.ingest()
    assert retry["receipt_id"] == first["receipt_id"]
