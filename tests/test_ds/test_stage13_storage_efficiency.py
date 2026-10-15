from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import hotpot.ds.storage as storage_module
from hotpot.ds.public._implementation import runtime_policy
from hotpot.ds.storage import (
    TARGET_SCHEMA_ID,
    SnapshotError,
    TargetStore,
    asset_inventory_digest,
    case_content_digest,
    case_set_digest,
)


def _json_digest(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _asset(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "source_path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def _case(case_id: str, public_index: int, source: Path) -> dict:
    asset = _asset(source)
    structure_id = f"structure-{case_id}"
    return {
        "id": case_id,
        "public_index": public_index,
        "local_id": f"local-{case_id}",
        "facts": {"core_info": {"sample": case_id}},
        "assessment": {
            "archive_class": "ordinary",
            "quality_level": "B2",
            "evidence_level": "E2",
            "benchmark": False,
            "reasons": [],
        },
        "identifiers": [{"namespace": "CSD", "value": case_id.upper()}],
        "structures": [
            {
                "structure_id": structure_id,
                "format": "xyz",
                "role": "experimental",
                "primary": True,
                **asset,
            }
        ],
        "assets": [
            {
                "collection": "structures",
                "declaration_id": structure_id,
                "asset_index": 0,
                "role": "experimental",
                "status": "active",
                **asset,
            }
        ],
    }


def _snapshot(store: TargetStore, cases: list[dict], control_assets=()) -> dict:
    policy = {
        "schema": "hotpot-ds-policy/current",
        "gates": {"quality": "default"},
        "runtime_policy": runtime_policy(),
    }
    all_assets = [asset for case in cases for asset in case["assets"]]
    all_assets.extend(control_assets)
    marker = "release-1:1.0"
    return {
        "target_context": {
            "canonical_path": str(store.path),
            "store_state": "absent",
            "store_uuid": None,
            "schema_id": TARGET_SCHEMA_ID,
            "dataset_state": "absent",
            "dataset_policy_revision": None,
            "dataset_policy_digest": None,
        },
        "bundle": {
            "dataset": "spin",
            "release": "release-1",
            "version": "1.0",
            "release_serial": None,
            "title": "Spin states",
            "description": "Storage efficiency fixture",
            "metadata": {},
        },
        "policy": {
            "revision": 1,
            "digest": _json_digest(policy),
            "value": policy,
        },
        "digests": {
            "bundle": hashlib.sha256(f"bundle:{marker}".encode()).hexdigest(),
            "certificate": hashlib.sha256(
                f"certificate:{marker}".encode()
            ).hexdigest(),
            "generation": hashlib.sha256(
                f"generation:{marker}".encode()
            ).hexdigest(),
            "case_content": case_content_digest(cases),
            "included_cases": case_set_digest(case["id"] for case in cases),
            "excluded_cases": case_set_digest(()),
            "asset_inventory": asset_inventory_digest(all_assets),
        },
        "control_assets": list(control_assets),
        "cases": cases,
    }


def test_ingest_verifies_and_copies_each_unique_asset_once(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "shared.xyz"
    source.write_text("1\n\nEu 0 0 0\n", encoding="utf-8")
    shared = _asset(source)
    control = {
        "category": "checker",
        "name": "shared.xyz",
        **shared,
    }
    cases = [_case("case-a", 0, source), _case("case-b", 1, source)]
    store = TargetStore(tmp_path / "datasets.sqlite3")
    calls = []
    original = TargetStore._store_asset

    def counted_store_asset(self, descriptor):
        calls.append(descriptor["sha256"])
        return original(self, descriptor)

    monkeypatch.setattr(TargetStore, "_store_asset", counted_store_asset)
    store.ingest(_snapshot(store, cases, [control]))

    assert calls == [shared["sha256"]]
    with sqlite3.connect(store.path) as connection:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "target_assets",
                "structure_asset_refs",
                "scientific_asset_refs",
                "release_control_asset_refs",
            )
        }
    assert counts == {
        "target_assets": 1,
        "structure_asset_refs": 2,
        "scientific_asset_refs": 2,
        "release_control_asset_refs": 1,
    }


def test_ingest_commit_does_not_reparse_case_json(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.xyz"
    source.write_text("1\n\nAm 0 0 0\n", encoding="utf-8")
    store = TargetStore(tmp_path / "datasets.sqlite3")
    snapshot = _snapshot(store, [_case("case-a", 0, source)])

    def forbidden_loads(*args, **kwargs):
        raise AssertionError("ingest reparsed normalized Case JSON")

    monkeypatch.setattr(
        storage_module,
        "json",
        SimpleNamespace(dumps=json.dumps, loads=forbidden_loads),
    )
    result = store.ingest(snapshot)

    assert result.included_count == 1


def test_ingest_publishes_cases_with_constant_set_based_statements(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.xyz"
    source.write_text("1\n\nEu 0 0 0\n", encoding="utf-8")
    store = TargetStore(tmp_path / "datasets.sqlite3")
    cases = [_case(f"case-{index}", index, source) for index in range(100)]
    statements = []
    original_connect = TargetStore._connect

    def traced_connect(self):
        connection = original_connect(self)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(TargetStore, "_connect", traced_connect)
    store.ingest(_snapshot(store, cases))

    normalized = [statement.lstrip().upper() for statement in statements]
    case_inserts = sum(
        statement.startswith("INSERT INTO CASES(") for statement in normalized
    )
    assert case_inserts == 1
    assert sum(
        statement.startswith("INSERT INTO CASE_ASSESSMENTS(")
        for statement in normalized
    ) == 1
    assert sum(
        statement.startswith("INSERT INTO IDENTIFIERS(")
        for statement in normalized
    ) == 1


def test_spool_content_digest_still_binds_published_case_facts(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.xyz"
    source.write_text("1\n\nEu 0 0 0\n", encoding="utf-8")
    store = TargetStore(tmp_path / "datasets.sqlite3")
    snapshot = _snapshot(store, [_case("case-a", 0, source)])
    original_spool = TargetStore._spool_cases

    def tampered_spool(self, spool_path, *args, **kwargs):
        counts = original_spool(self, spool_path, *args, **kwargs)
        with sqlite3.connect(spool_path) as spool:
            spool.execute(
                "UPDATE spool_cases SET content_json=json_set("
                "content_json,'$.facts.core_info.sample','tampered')"
            )
        return counts

    monkeypatch.setattr(TargetStore, "_spool_cases", tampered_spool)

    with pytest.raises(SnapshotError, match="Case content digest"):
        store.ingest(snapshot)

    assert store.load_release_metadata("spin", "release-1") is None
