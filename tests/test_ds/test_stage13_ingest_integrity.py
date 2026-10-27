from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from hotpot.ds.gates import BenchmarkGate
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA
from hotpot.ds.public._generation import AssessmentGenerationError
from hotpot.ds.public.bundle import (
    Bundle,
    ExistingDatasetPolicyError,
    SealedIntegrityError,
)
from hotpot.ds.public.case import UnsynchronizedBundleError
from hotpot.ds.public.certificate import CertificateIntegrityError
from hotpot.ds.storage import TargetStore


def _bundle_root(
    parent: Path,
    *,
    release_id: str,
    release_serial: int,
    version: str,
) -> Path:
    root = parent / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    case = {
        "id": f"case-{release_serial}",
        "local_id": f"local-{release_serial}",
        "public_index": 0,
        "identifiers": {"CSD_identifier": f"CASE{release_serial}"},
        "core_info": {"sample": "same-sample"},
        "observations": [
            {
                "observation_id": f"observation-{release_serial}",
                "property": "spin",
                "value": 1,
            }
        ],
        "structures": [
            {
                "structure_id": f"structure-{release_serial}",
                "role": "primary",
                "format": "smi",
                "path": "structures/sample.smi",
                "primary": True,
            }
        ],
        "attestations": [
            {
                "attestation_id": f"attestation-{release_serial}",
                "authority": "User",
                "actor": "curator@example.test",
                "approved": True,
            }
        ],
    }
    intake.write_text(json.dumps(case) + "\n", encoding="utf-8")
    structures = root / "structures"
    structures.mkdir()
    (structures / "sample.smi").write_text("O\n", encoding="utf-8")
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "demo",
        "release_id": release_id,
        "release_serial": release_serial,
        "version": version,
        "title": "Demo",
        "description": "Ingestion integrity test",
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


def _ingest_first_release(root: Path, database: Path, *, gates=None) -> Bundle:
    bundle = Bundle(root, "demo", db=database, gates=gates)
    bundle.synchronize()
    bundle.write_certificate()
    bundle.ingest()
    return bundle


def test_existing_dataset_mode_cannot_unlock_when_target_disappears(
    tmp_path: Path,
) -> None:
    database = tmp_path / "datasets.sqlite3"
    first = _bundle_root(
        tmp_path / "first", release_id="demo-r1", release_serial=1, version="1.0"
    )
    _ingest_first_release(first, database)
    successor_root = _bundle_root(
        tmp_path / "second", release_id="demo-r2", release_serial=2, version="2.0"
    )
    successor = Bundle(successor_root, "demo", db=database)

    assert successor.mode == "existing"
    database.unlink()
    assert successor.mode == "existing"
    with pytest.raises(ExistingDatasetPolicyError):
        successor.use_gate(BenchmarkGate())


@pytest.mark.external
def test_existing_dataset_successor_materializes_persisted_custom_checker(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "project_checker.py"
    source = (
        "def project_rule(case, bundle):\n"
        "    return True, 'accepted by persisted project policy'\n"
    )
    module_path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("project_checker", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    benchmark_gate = BenchmarkGate(checklist={"project_rule": module.project_rule})

    database = tmp_path / "datasets.sqlite3"
    first = _bundle_root(
        tmp_path / "first", release_id="demo-r1", release_serial=1, version="1.0"
    )
    _ingest_first_release(first, database, gates=[benchmark_gate])

    successor_root = _bundle_root(
        tmp_path / "second", release_id="demo-r2", release_serial=2, version="2.0"
    )
    successor = Bundle(successor_root, "demo", db=database)
    definition = successor.gates["benchmark"].info["checklist"]["project_rule"]
    artifact = (
        successor_root
        / "policy"
        / "checkers"
        / f"{definition['source_sha256']}.py"
    )

    assert successor.mode == "existing"
    assert artifact.read_text(encoding="utf-8") == source
    assert successor.synchronize()["is_synchronized"] is True


@pytest.mark.parametrize("changed_asset", ["certificate", "facts", "generation"])
def test_ingest_compare_and_swap_rejects_source_change_before_target_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_asset: str,
) -> None:
    root = _bundle_root(
        tmp_path, release_id="demo-r1", release_serial=1, version="1.0"
    )
    database = tmp_path / "datasets.sqlite3"
    bundle = Bundle(root, "demo", db=database)
    bundle.synchronize()
    bundle.write_certificate()
    original_spool = TargetStore._spool_cases

    def spool_then_change(store, *args, **kwargs):
        result = original_spool(store, *args, **kwargs)
        if changed_asset == "certificate":
            with bundle.certificate_path.open("ab") as stream:
                stream.write(b"\n")
        elif changed_asset == "facts":
            with sqlite3.connect(root / "cases" / "facts.sqlite3") as connection:
                connection.execute(
                    "UPDATE cases SET payload_json=payload_json || ' ' WHERE case_id=?",
                    ("case-1",),
                )
        else:
            with bundle._current_generation.manifest_path.open("ab") as stream:
                stream.write(b"\n")
        return result

    monkeypatch.setattr(TargetStore, "_spool_cases", spool_then_change)
    with pytest.raises(
        (
            AssessmentGenerationError,
            CertificateIntegrityError,
            SealedIntegrityError,
            UnsynchronizedBundleError,
        )
    ):
        bundle.ingest()

    assert TargetStore(database).load_release_metadata("demo", "demo-r1") is None


def test_sealed_receipt_binds_included_case_content_digest(tmp_path: Path) -> None:
    root = _bundle_root(
        tmp_path, release_id="demo-r1", release_serial=1, version="1.0"
    )
    database = tmp_path / "datasets.sqlite3"
    bundle = _ingest_first_release(root, database)
    receipt = bundle._authoritative_receipt()

    assert receipt is not None
    assert len(receipt["included_case_content_digest"]) == 64

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE ingestion_receipts SET included_case_content_digest=? "
            "WHERE dataset_id=? AND release_id=?",
            ("0" * 64, "demo", "demo-r1"),
        )

    with pytest.raises(
        SealedIntegrityError,
        match="receipt does not match current certified content",
    ):
        bundle._authoritative_receipt()
