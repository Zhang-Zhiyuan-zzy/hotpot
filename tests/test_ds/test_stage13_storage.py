from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from hotpot.ds.storage import (
    TARGET_SCHEMA_ID,
    PolicyMismatchError,
    ReceiptConflictError,
    SnapshotError,
    TargetStore,
    TargetStoreError,
    asset_inventory_digest,
    case_content_digest,
    case_set_digest,
)
from hotpot.ds.public._implementation import runtime_policy


def _digest_json(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha(path: Path):
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _case(
    case_id,
    public_index,
    archive_class="ordinary",
    asset=None,
    structure_id=None,
):
    structures = []
    if asset is not None:
        digest, size = _sha(asset)
        structures.append(
            {
                "structure_id": structure_id or f"structure-{case_id}",
                "format": "xyz",
                "role": "experimental",
                "primary": True,
                "source_path": asset,
                "sha256": digest,
                "size": size,
            }
        )
    return {
        "id": case_id,
        "public_index": public_index,
        "local_id": f"local-{case_id}",
        "facts": {"core_info": {"sample": case_id}},
        "assessment": {
            "archive_class": archive_class,
            "quality_level": "B2" if archive_class != "incomplete" else "B0",
            "evidence_level": "E2" if archive_class != "incomplete" else "E0",
            "benchmark": archive_class == "benchmark",
            "reasons": [],
        },
        "identifiers": [{"namespace": "CSD", "value": case_id.upper()}],
        "structures": structures,
    }


def _snapshot(
    store,
    cases,
    *,
    release="release-1",
    version="1.0",
    release_serial=None,
    policy=None,
    include_incomplete=False,
):
    cases = list(cases)
    policy = policy or {"gates": {"quality": "default"}}
    policy = {
        "schema": "hotpot-ds-policy/current",
        **policy,
        "runtime_policy": runtime_policy(),
    }
    included = [
        case
        for case in cases
        if case["assessment"]["archive_class"] != "incomplete"
        or include_incomplete
    ]
    excluded = [case for case in cases if case not in included]
    for case in cases:
        case["assets"] = [
            {
                "collection": "structures",
                "declaration_id": structure["structure_id"],
                "asset_index": index,
                "role": structure["role"],
                "status": "active",
                "source_path": structure["source_path"],
                "sha256": structure["sha256"],
                "size": structure["size"],
            }
            for index, structure in enumerate(case["structures"])
        ]
    assets = [asset for case in included for asset in case["assets"]]
    if store.path.is_file():
        metadata = store.metadata()
        established = store.read_policy("spin")
        target_context = {
            "canonical_path": str(store.path),
            "store_state": "existing",
            "store_uuid": metadata["store_uuid"],
            "schema_id": metadata["schema_id"],
            "dataset_state": "present" if established else "absent",
            "dataset_policy_revision": None if established is None else established["revision"],
            "dataset_policy_digest": None if established is None else established["digest"],
        }
    else:
        target_context = {
            "canonical_path": str(store.path),
            "store_state": "absent",
            "store_uuid": None,
            "schema_id": TARGET_SCHEMA_ID,
            "dataset_state": "absent",
            "dataset_policy_revision": None,
            "dataset_policy_digest": None,
        }
    marker = f"{release}:{version}"
    return {
        "target_context": target_context,
        "bundle": {
            "dataset": "spin",
            "release": release,
            "version": version,
            "release_serial": release_serial,
            "title": "Spin states",
            "description": "Curated spin-state records",
            "metadata": {"marker": marker},
        },
        "policy": {
            "revision": 1,
            "digest": _digest_json(policy),
            "value": policy,
        },
        "digests": {
            "bundle": hashlib.sha256(f"bundle:{marker}".encode()).hexdigest(),
            "certificate": hashlib.sha256(f"certificate:{marker}".encode()).hexdigest(),
            "generation": hashlib.sha256(f"generation:{marker}".encode()).hexdigest(),
            "case_content": case_content_digest(cases),
            "included_cases": case_set_digest(case["id"] for case in included),
            "excluded_cases": case_set_digest(case["id"] for case in excluded),
            "asset_inventory": asset_inventory_digest(assets),
        },
        "cases": cases,
    }


def test_initialize_records_stable_uuid_and_current_schema(tmp_path):
    store = TargetStore(tmp_path / "datasets.sqlite3")

    first = store.initialize()
    second = store.initialize()

    assert first == second
    assert first["schema_id"] == TARGET_SCHEMA_ID
    assert first["store_uuid"]


def test_case_content_must_match_receipt_bound_snapshot(tmp_path):
    store = TargetStore(tmp_path / "datasets.sqlite3")
    snapshot = _snapshot(store, [_case("case-a", 0)])
    snapshot["cases"][0]["facts"]["core_info"]["sample"] = "changed"

    with pytest.raises(SnapshotError, match="Case content digest"):
        store.ingest(snapshot)


def test_equal_coordinate_bytes_do_not_merge_case_or_structure_identity(tmp_path):
    source = tmp_path / "same.xyz"
    source.write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    cases = [
        _case("case-a", 0, asset=source, structure_id="structure-a"),
        _case("case-b", 1, asset=source, structure_id="structure-b"),
    ]
    store = TargetStore(tmp_path / "datasets.sqlite3")

    store.ingest(_snapshot(store, cases))

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM target_assets").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM structure_asset_refs").fetchone()[0] == 2
        assert {
            row[0]
            for row in connection.execute(
                "SELECT structure_id FROM structure_asset_refs ORDER BY structure_id"
            )
        } == {"structure-a", "structure-b"}
    assert [case["id"] for case in store.iter_cases("spin", "release-1")] == [
        "case-a",
        "case-b",
    ]


def test_receipt_rejects_tampered_frozen_runtime_policy(tmp_path) -> None:
    store = TargetStore(tmp_path / "datasets.sqlite3")
    store.ingest(_snapshot(store, [_case("case-a", 0)]))
    with sqlite3.connect(store.path) as connection:
        raw = connection.execute(
            "SELECT binding_json FROM ingestion_receipts"
        ).fetchone()[0]
        binding = json.loads(raw)
        binding["runtime_policy"]["runtime"]["python"]["version"] = "tampered"
        connection.execute(
            "UPDATE ingestion_receipts SET binding_json=?",
            (json.dumps(binding, separators=(",", ":"), sort_keys=True),),
        )

    with pytest.raises(TargetStoreError, match="runtime policy"):
        store.load_release_metadata("spin", "release-1")


def test_exact_receipt_retry_survives_source_removal_and_digest_change_conflicts(
    tmp_path,
):
    source = tmp_path / "source.xyz"
    source.write_text("1\n\nEu 0 0 0\n", encoding="utf-8")
    store = TargetStore(tmp_path / "datasets.sqlite3")
    snapshot = _snapshot(store, [_case("case-a", 0, asset=source)])

    first = store.ingest(snapshot)
    source.unlink()
    retry = store.ingest(snapshot)

    assert retry == first
    stored = list(store.iter_cases("spin", "release-1"))[0]["structures"][0]["path"]
    assert stored.is_file()
    assert stored.read_text(encoding="utf-8") == "1\n\nEu 0 0 0\n"

    changed = dict(snapshot)
    changed["digests"] = dict(snapshot["digests"])
    changed["digests"]["bundle"] = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(ReceiptConflictError):
        store.ingest(changed)


def test_existing_dataset_accepts_only_exact_persisted_policy(tmp_path):
    store = TargetStore(tmp_path / "datasets.sqlite3")
    store.ingest(_snapshot(store, [_case("case-a", 0)]))
    changed_policy = {"gates": {"quality": "strict"}}

    with pytest.raises(PolicyMismatchError):
        store.ingest(
            _snapshot(
                store,
                [_case("case-b", 0)],
                release="release-2",
                version="2.0",
                policy=changed_policy,
            )
        )


def test_incomplete_partition_requires_reason_and_default_reads_exclude_it(tmp_path):
    cases = [_case("ordinary", 7), _case("broken", 11, "incomplete")]
    store = TargetStore(tmp_path / "datasets.sqlite3")
    forced = _snapshot(store, cases, include_incomplete=True)

    with pytest.raises(SnapshotError, match="non-empty reason"):
        store.ingest(forced, include_incomplete=True)

    result = store.ingest(
        forced,
        include_incomplete=True,
        reason="Retain supplier record pending structure recovery.",
    )

    assert result.included_count == 2
    assert result.excluded_count == 0
    assert [case["id"] for case in store.iter_cases("spin", "release-1")] == [
        "ordinary"
    ]
    assert [
        case["id"]
        for case in store.iter_cases("spin", "release-1", include_incomplete=True)
    ] == ["ordinary", "broken"]
    assert [
        case["id"]
        for case in store.iter_cases(
            "spin", "release-1", archive_class="incomplete"
        )
    ] == ["broken"]


def test_excluded_incomplete_cases_are_receipt_bound_but_not_stored(tmp_path):
    cases = [_case("ordinary", 2), _case("broken", 8, "incomplete")]
    store = TargetStore(tmp_path / "datasets.sqlite3")

    result = store.ingest(_snapshot(store, cases))

    assert (result.included_count, result.excluded_count) == (1, 1)
    assert [
        case["id"]
        for case in store.iter_cases("spin", "release-1", include_incomplete=True)
    ] == ["ordinary"]
    release = store.load_release_metadata("spin", "release-1")
    assert release["receipt"]["excluded_cases_digest"] == case_set_digest(["broken"])


def test_release_serial_and_public_indexes_remain_stable(tmp_path):
    store = TargetStore(tmp_path / "datasets.sqlite3")
    first = store.ingest(
        _snapshot(store, [_case("case-a", 42)], release_serial=7)
    )
    second = store.ingest(
        _snapshot(store, [_case("case-b", 4)], release="release-2", version="2.0")
    )

    assert first.release_serial == 7
    assert second.release_serial == 8
    assert list(store.iter_cases("spin", "release-1"))[0]["public_index"] == 42
    assert list(store.iter_cases("spin", "release-2"))[0]["public_index"] == 4
    assert [item["release_serial"] for item in store.inspect_dataset("spin")["releases"]] == [
        7,
        8,
    ]


def test_target_owns_materialized_evidence_and_withdrawn_assets(tmp_path):
    source = tmp_path / "supplier-note.txt"
    source.write_text("reported spin = 3\n", encoding="utf-8")
    digest, size = _sha(source)
    checker = tmp_path / "checker.py"
    checker.write_text("def check(case, context):\n    return True\n", encoding="utf-8")
    checker_digest, checker_size = _sha(checker)
    store = TargetStore(tmp_path / "datasets.sqlite3")
    snapshot = _snapshot(store, [_case("case-a", 0)])
    snapshot["cases"][0]["assets"].append(
        {
            "collection": "evidence",
            "declaration_id": "evidence-1",
            "asset_index": 0,
            "role": "supplier_text",
            "status": "withdrawn",
            "source_path": source,
            "sha256": digest,
            "size": size,
        }
    )
    snapshot["control_assets"] = [
        {
            "category": "checker",
            "name": "checker.py",
            "source_path": checker,
            "sha256": checker_digest,
            "size": checker_size,
        }
    ]
    snapshot["digests"]["asset_inventory"] = asset_inventory_digest(
        snapshot["cases"][0]["assets"] + snapshot["control_assets"]
    )
    snapshot["digests"]["case_content"] = case_content_digest(snapshot["cases"])

    store.ingest(snapshot)
    source.unlink()
    checker.unlink()

    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT collection,declaration_id,status,content_sha256 "
            "FROM scientific_asset_refs"
        ).fetchone()
    assert row == ("evidence", "evidence-1", "withdrawn", digest)
    with sqlite3.connect(store.path) as connection:
        control = connection.execute(
            "SELECT category,name,content_sha256 FROM release_control_asset_refs"
        ).fetchone()
    assert control == ("checker", "checker.py", checker_digest)
    owned = store.asset_root / "objects" / "sha256" / digest[:2] / digest
    assert owned.read_text(encoding="utf-8") == "reported spin = 3\n"
    owned_checker = (
        store.asset_root
        / "objects"
        / "sha256"
        / checker_digest[:2]
        / checker_digest
    )
    assert owned_checker.read_text(encoding="utf-8").startswith("def check")


def test_existing_target_uuid_is_rechecked_before_ingest(tmp_path):
    store = TargetStore(tmp_path / "datasets.sqlite3")
    store.initialize()
    snapshot = _snapshot(store, [_case("case-a", 0)])
    original_uuid = snapshot["target_context"]["store_uuid"]

    store.path.unlink()
    replacement_uuid = store.initialize()["store_uuid"]
    assert replacement_uuid != original_uuid

    with pytest.raises(ReceiptConflictError, match="UUID"):
        store.ingest(snapshot)


def test_one_external_identifier_may_name_several_cases(tmp_path):
    store = TargetStore(tmp_path / "datasets.sqlite3")
    cases = [_case("case-a", 0), _case("case-b", 1)]
    cases[0]["identifiers"] = [{"namespace": "CSD", "value": "DOQRAC"}]
    cases[1]["identifiers"] = [{"namespace": "CSD", "value": "DOQRAC"}]

    store.ingest(_snapshot(store, cases))

    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM identifiers WHERE namespace='CSD' AND value='DOQRAC'"
        ).fetchone()[0] == 2
