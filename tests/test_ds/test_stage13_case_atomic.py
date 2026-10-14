from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA
from hotpot.ds.public.bundle import Bundle, load_bundle
from hotpot.ds.public.case import Case


def _bundle_root(tmp_path: Path) -> Path:
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "atomic",
        "release_id": "atomic-r1",
        "release_serial": 1,
        "version": "1.0",
        "title": "Atomic mutation test",
        "description": "Atomic mutation test",
        "license": "CC0-1.0",
        "access_policy": "open",
        "provenance": {"created_by": "test"},
        "core_labels": [],
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
        "required_structure_roles": [],
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _audit_actions(root: Path) -> list[str]:
    with sqlite3.connect(root / "cases" / "facts.sqlite3") as connection:
        return [
            row[0]
            for row in connection.execute(
                "SELECT action FROM audit_events ORDER BY source_revision,rowid"
            )
        ]


def test_concurrent_case_edits_merge_against_latest_facts(tmp_path: Path) -> None:
    root = _bundle_root(tmp_path)
    database = tmp_path / "datasets.sqlite3"
    first = Bundle(root, "atomic", db=database)
    second = load_bundle(root, "atomic", db=database)
    start = threading.Barrier(2)

    def edit(bundle: Bundle, name: str, value: int) -> None:
        start.wait()
        bundle.cases[0].set_core_info(name, value)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(edit, first, "left", 1),
            executor.submit(edit, second, "right", 2),
        ]
        for future in futures:
            future.result()

    facts = first.cases[0].core_info
    assert facts == {"sample": "one", "left": 1, "right": 2}
    assert first.status["source_revision"] == 2
    assert _audit_actions(root)[-2:] == ["core_info.set", "core_info.set"]


def test_stale_evidence_view_cannot_authorize_later_verification(
    tmp_path: Path,
) -> None:
    root = _bundle_root(tmp_path)
    database = tmp_path / "datasets.sqlite3"
    first = Bundle(root, "atomic", db=database)
    second = load_bundle(root, "atomic", db=database)
    evidence_id = first.cases[0].add_evidence("case-1", "doi:10.1/example")
    stale_view = first.cases[0].evidence

    second.cases[0].withdraw_evidence(evidence_id, reason="retracted")
    before = first.status["source_revision"]
    with pytest.raises(KeyError, match=evidence_id):
        first.cases[0].verify_evidence(
            evidence_id,
            method="Human",
            verifier="curator",
        )

    assert stale_view[0]["status"] == "active"
    assert first.cases[0].evidence[0]["status"] == "withdrawn"
    assert first.status["source_revision"] == before
    assert first._state.get_case("case-1")["verifications"] == []
    assert _audit_actions(root)[-1] == "evidence.withdraw"


def test_verification_and_withdrawal_are_serialized_at_transaction_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _bundle_root(tmp_path)
    database = tmp_path / "datasets.sqlite3"
    verifier_bundle = Bundle(root, "atomic", db=database)
    withdrawer_bundle = load_bundle(root, "atomic", db=database)
    evidence_id = verifier_bundle.cases[0].add_evidence(
        "case-1", "doi:10.1/example"
    )

    verification_inside_transaction = threading.Event()
    release_verification = threading.Event()
    withdrawal_attempted = threading.Event()
    original_active_item = Case._active_item
    original_locked = withdrawer_bundle._locked

    def paused_active_item(facts, collection, id_key, item_id):
        if threading.current_thread().name.startswith("verifier"):
            verification_inside_transaction.set()
            assert release_verification.wait(timeout=5)
        return original_active_item(facts, collection, id_key, item_id)

    @contextmanager
    def observed_lock():
        withdrawal_attempted.set()
        with original_locked():
            yield

    monkeypatch.setattr(Case, "_active_item", staticmethod(paused_active_item))
    monkeypatch.setattr(withdrawer_bundle, "_locked", observed_lock)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="verifier") as verify_pool:
        verification = verify_pool.submit(
            verifier_bundle.cases[0].verify_evidence,
            evidence_id,
            method="Human",
            verifier="curator",
        )
        assert verification_inside_transaction.wait(timeout=5)
        with ThreadPoolExecutor(max_workers=1) as withdraw_pool:
            withdrawal = withdraw_pool.submit(
                withdrawer_bundle.cases[0].withdraw_evidence,
                evidence_id,
                reason="retracted after verification",
            )
            assert withdrawal_attempted.wait(timeout=5)
            assert not withdrawal.done()
            release_verification.set()
            verification_id = verification.result(timeout=5)
            withdrawal.result(timeout=5)

    facts = verifier_bundle._state.get_case("case-1")
    assert facts["verifications"][0]["verification_id"] == verification_id
    assert facts["evidence"][0]["status"] == "withdrawn"
    assert _audit_actions(root)[-2:] == ["evidence.verify", "evidence.withdraw"]
