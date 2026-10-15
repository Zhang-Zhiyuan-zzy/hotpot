from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from pathlib import Path

import pytest

import hotpot.ds.public.certificate as certificate_module
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA, BundleState
from hotpot.ds.public._generation import AssessmentGeneration
from hotpot.ds.public.bundle import Bundle
from hotpot.ds.public.certificate import (
    CertificateIntegrityError,
    load_certificate_envelope,
    verify_bundle_certificate,
    write_bundle_certificate,
)


def _bundle(tmp_path: Path) -> Bundle:
    root = tmp_path / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    case = {
        "id": "case-1",
        "local_id": "case-1",
        "public_index": 0,
        "identifiers": {"sample": "one"},
        "core_info": {"sample": "one"},
        "structures": [
            {
                "structure_id": "structure-1",
                "role": "primary",
                "format": "smi",
                "path": "structures/shared.smi",
                "primary": True,
            },
            {
                "structure_id": "structure-2",
                "role": "alternate",
                "format": "smi",
                "path": "structures/shared.smi",
            },
        ],
        "observations": [
            {
                "observation_id": "observation-1",
                "property": "spin",
                "value": 1,
            }
        ],
    }
    intake.write_text(json.dumps(case) + "\n", encoding="utf-8")
    structures = root / "structures"
    structures.mkdir()
    (structures / "shared.smi").write_text("O\n", encoding="utf-8")
    control = root / "evidence" / "control.json"
    control.write_text('{"source":"curated"}\n', encoding="utf-8")
    control_bytes = control.read_bytes()
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "schema": BUNDLE_SCHEMA,
                "ds_name": "certificate-efficiency",
                "release_id": "certificate-efficiency-r1",
                "release_serial": 1,
                "version": "1.0",
                "title": "Certificate efficiency fixture",
                "description": "Certificate efficiency fixture",
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
                "control_assets": [
                    {
                        "category": "provenance",
                        "name": "curation-record",
                        "path": "evidence/control.json",
                        "sha256": hashlib.sha256(control_bytes).hexdigest(),
                        "size": len(control_bytes),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    bundle = Bundle(
        root,
        "certificate-efficiency",
        db=tmp_path / "datasets.sqlite3",
    )
    bundle.synchronize()
    return bundle


def test_certificate_uses_one_locked_snapshot_and_one_hash_per_unique_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    expected_source_digest = bundle._state.source_digest
    generation_opens = 0
    source_captures = 0
    bound_hashes: Counter[Path] = Counter()
    source_digest_calls = 0
    original_open = AssessmentGeneration._open_without_lease.__func__
    original_capture = certificate_module._capture_canonical_state
    original_hash = certificate_module._stable_hash_file
    original_source_digest = BundleState._compute_source_digest

    def counted_open(cls, *args, **kwargs):
        nonlocal generation_opens
        generation_opens += 1
        return original_open(cls, *args, **kwargs)

    def counted_capture(*args, **kwargs):
        nonlocal source_captures
        source_captures += 1
        return original_capture(*args, **kwargs)

    def counted_hash(path: Path):
        bound_hashes[Path(path)] += 1
        return original_hash(path)

    def counted_source_digest(*args, **kwargs):
        nonlocal source_digest_calls
        source_digest_calls += 1
        return original_source_digest(*args, **kwargs)

    def forbidden_state_connect(*args, **kwargs):
        raise AssertionError("certificate capture must not call BundleState.connect")

    with monkeypatch.context() as patch:
        patch.setattr(
            AssessmentGeneration,
            "_open_without_lease",
            classmethod(counted_open),
        )
        patch.setattr(
            certificate_module, "_capture_canonical_state", counted_capture
        )
        patch.setattr(certificate_module, "_stable_hash_file", counted_hash)
        patch.setattr(BundleState, "_compute_source_digest", counted_source_digest)
        patch.setattr(BundleState, "connect", forbidden_state_connect)
        path = write_bundle_certificate(bundle)

    envelope = load_certificate_envelope(path)
    generation_manifest = bundle.root / envelope["certificate"]["generation"][
        "manifest"
    ]["path"]
    facts = bundle.root / "cases" / "facts.sqlite3"
    shared_asset = bundle.root / "structures" / "shared.smi"
    control_asset = bundle.root / "evidence" / "control.json"

    assert source_captures == 1
    assert source_digest_calls == 0
    assert generation_opens == 1
    assert bound_hashes[facts] == 1
    assert bound_hashes[generation_manifest] == 1
    assert bound_hashes[shared_asset] == 1
    assert bound_hashes[control_asset] == 1
    assert max(bound_hashes.values()) == 1
    assert envelope["certificate"]["source"]["digest"] == expected_source_digest
    assert [
        item["sha256"]
        for item in envelope["certificate"]["scientific_assets"]["items"]
    ] == [
        envelope["certificate"]["scientific_assets"]["items"][0]["sha256"]
    ] * 2
    assert verify_bundle_certificate(bundle) == envelope


@pytest.mark.parametrize("target", ["scientific", "control"])
def test_prepublication_cas_rejects_asset_tampering_and_preserves_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    bundle = _bundle(tmp_path)
    path = write_bundle_certificate(bundle)
    original_certificate = path.read_bytes()
    asset = {
        "scientific": bundle.root / "structures" / "shared.smi",
        "control": bundle.root / "evidence" / "control.json",
    }[target]
    original_atomic_write = certificate_module._atomic_write

    def tampering_atomic_write(path, value, *, before_replace=None):
        def tamper_then_check():
            asset.write_text("N\n", encoding="utf-8")
            before_replace()

        return original_atomic_write(
            path, value, before_replace=tamper_then_check
        )

    monkeypatch.setattr(
        certificate_module, "_atomic_write", tampering_atomic_write
    )

    with pytest.raises(CertificateIntegrityError, match="changed before publication"):
        write_bundle_certificate(bundle)

    assert path.read_bytes() == original_certificate
    assert not list(bundle.root.glob(".certificate.json.*.tmp"))


@pytest.mark.parametrize("mutation", ["add", "remove", "replace"])
def test_prepublication_cas_rejects_generation_tree_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    bundle = _bundle(tmp_path)
    path = write_bundle_certificate(bundle)
    original_certificate = path.read_bytes()
    envelope = load_certificate_envelope(path)
    generation_directory = (
        bundle.root
        / envelope["certificate"]["generation"]["manifest"]["path"]
    ).parent
    rogue = generation_directory / "rogue.txt"
    existing = generation_directory / "summary.json"
    existing_bytes = existing.read_bytes()
    original_atomic_write = certificate_module._atomic_write

    def tampering_atomic_write(path, value, *, before_replace=None):
        def tamper_then_check():
            if mutation == "add":
                rogue.write_text("unlisted\n", encoding="utf-8")
            elif mutation == "remove":
                existing.unlink()
            else:
                existing.write_bytes(existing_bytes + b"tampered")
            before_replace()

        return original_atomic_write(
            path, value, before_replace=tamper_then_check
        )

    monkeypatch.setattr(
        certificate_module, "_atomic_write", tampering_atomic_write
    )

    with pytest.raises(CertificateIntegrityError, match="generation tree changed"):
        write_bundle_certificate(bundle)

    assert path.read_bytes() == original_certificate
    assert not list(bundle.root.glob(".certificate.json.*.tmp"))
    if mutation == "add":
        rogue.unlink()
    else:
        existing.write_bytes(existing_bytes)
    assert verify_bundle_certificate(bundle) == envelope


def test_bundle_lock_serializes_supported_edits_with_certificate_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    start_edit = threading.Event()
    edit_attempted = threading.Event()
    edit_finished = threading.Event()
    original_atomic_write = certificate_module._atomic_write

    def edit_case() -> None:
        start_edit.wait()
        edit_attempted.set()
        bundle.cases[0].set_core_info("note", "concurrent edit")
        edit_finished.set()

    editor = threading.Thread(target=edit_case)
    editor.start()

    def observing_atomic_write(path, value, *, before_replace=None):
        def observe_then_check():
            start_edit.set()
            assert edit_attempted.wait(timeout=5)
            assert not edit_finished.wait(timeout=0.2)
            before_replace()

        return original_atomic_write(
            path, value, before_replace=observe_then_check
        )

    monkeypatch.setattr(
        certificate_module, "_atomic_write", observing_atomic_write
    )
    path = write_bundle_certificate(bundle)
    editor.join(timeout=5)

    assert not editor.is_alive()
    assert edit_finished.is_set()
    assert load_certificate_envelope(path)["certificate"]["dataset"] == bundle.name
    assert bundle.is_synchronized is False


def test_certificate_rejects_symlinked_control_asset(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    control = bundle.root / "evidence" / "control.json"
    target = bundle.root / "evidence" / "control-target.json"
    target.write_bytes(control.read_bytes())
    control.unlink()
    control.symlink_to(target.name)

    with pytest.raises(ValueError, match="regular Bundle file"):
        write_bundle_certificate(bundle)
