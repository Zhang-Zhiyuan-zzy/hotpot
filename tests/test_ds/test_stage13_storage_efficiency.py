from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from hotpot.ds.public._implementation import runtime_policy
from hotpot.ds.storage import (
    TARGET_SCHEMA_ID,
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
