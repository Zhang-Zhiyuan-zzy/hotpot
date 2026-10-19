from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from hotpot.ds._store_copy import StoreCopyError, copy_store
from hotpot.ds.cli import build_parser, run
from hotpot.ds.public._implementation import runtime_policy
from hotpot.ds.public.catalog import load
from hotpot.ds.storage import (
    TARGET_SCHEMA_ID,
    TargetStore,
    asset_inventory_digest,
    case_content_digest,
    case_set_digest,
)


def _json_digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_package(tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    package = tmp_path / "source" / "hotpot"
    store_root = package / "dataset" / "data" / "store"
    generation = store_root / "generations" / "source-generation"
    database = generation / "datasets.sqlite3"
    source_structure = tmp_path / "eu.xyz"
    structure_bytes = b"1\n\nEu 0 0 0\n"
    source_structure.write_bytes(structure_bytes)
    digest = hashlib.sha256(structure_bytes).hexdigest()
    case = {
        "id": "case-eu",
        "public_index": 4,
        "local_id": "local-eu",
        "facts": {
            "core_info": {"sample": "eu-complex"},
            "labels": {"spin": 3},
        },
        "assessment": {
            "archive_class": "benchmark",
            "quality_level": "B4",
            "evidence_level": "E4",
            "benchmark": True,
            "benchmark_status": "admitted",
            "reasons": [],
        },
        "identifiers": [{"namespace": "CSD", "value": "EUCASE"}],
        "structures": [
            {
                "structure_id": "structure-eu",
                "format": "xyz",
                "role": "experimental",
                "primary": True,
                "source_path": source_structure,
                "sha256": digest,
                "size": len(structure_bytes),
            }
        ],
        "assets": [
            {
                "collection": "structures",
                "declaration_id": "structure-eu",
                "asset_index": 0,
                "role": "experimental",
                "status": "active",
                "source_path": source_structure,
                "sha256": digest,
                "size": len(structure_bytes),
            }
        ],
    }
    policy = {
        "schema": "hotpot-ds-policy/current",
        "gates": {"quality": "test"},
        "runtime_policy": runtime_policy(),
    }
    store = TargetStore(database)
    store.ingest(
        {
            "target_context": {
                "canonical_path": str(database.resolve()),
                "store_state": "absent",
                "store_uuid": None,
                "schema_id": TARGET_SCHEMA_ID,
                "dataset_state": "absent",
                "dataset_policy_revision": None,
                "dataset_policy_digest": None,
            },
            "bundle": {
                "dataset": "spin",
                "release": "spin.v001",
                "version": "1.0",
                "release_serial": 1,
                "title": "Spin states",
                "description": "Store-copy fixture",
                "metadata": {"core_labels": ["spin"]},
            },
            "policy": {
                "revision": 1,
                "digest": _json_digest(policy),
                "value": policy,
            },
            "digests": {
                "bundle": hashlib.sha256(b"bundle").hexdigest(),
                "certificate": hashlib.sha256(b"certificate").hexdigest(),
                "generation": hashlib.sha256(b"generation").hexdigest(),
                "case_content": case_content_digest([case]),
                "included_cases": case_set_digest([case["id"]]),
                "excluded_cases": case_set_digest([]),
                "asset_inventory": asset_inventory_digest(case["assets"]),
            },
            "cases": [case],
        }
    )
    active = store_root / "active"
    active.parent.mkdir(parents=True, exist_ok=True)
    active.symlink_to(Path("generations") / generation.name, target_is_directory=True)
    return package, active / "datasets.sqlite3", database, structure_bytes


def _destination_database(package: Path) -> Path:
    return package / "dataset" / "data" / "store" / "active" / "datasets.sqlite3"


def _target_package(tmp_path: Path) -> Path:
    package = tmp_path / "target" / "hotpot"
    (package / "dataset").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    return package


def _only_cas_path(database: Path) -> Path:
    store = TargetStore(database)
    with store._connect() as connection:
        relative_path = connection.execute(
            "SELECT relative_path FROM target_assets"
        ).fetchone()[0]
    return store.asset_root / relative_path


def test_copy_store_publishes_relocatable_current_store(tmp_path: Path) -> None:
    _, source_active_db, source_generation_db, structure_bytes = _source_package(
        tmp_path
    )
    source_store = TargetStore(source_generation_db)
    source_metadata = source_store.metadata()
    source_cas = _only_cas_path(source_generation_db)
    with sqlite3.connect(source_generation_db) as connection:
        source_binding = connection.execute(
            "SELECT binding_json FROM ingestion_receipts"
        ).fetchone()[0]
    historical_path = json.loads(source_binding)["target_context"]["canonical_path"]
    target_package = _target_package(tmp_path)

    assert source_active_db.resolve() == source_generation_db
    result = copy_store(source_active_db, target_package)

    destination = _destination_database(target_package)
    store_root = target_package / "dataset" / "data" / "store"
    assert (store_root / "active").is_symlink()
    active_target = (store_root / "active").readlink()
    assert active_target.parts[0] == "generations"
    assert destination.is_file()
    assert result["store_uuid"] == source_metadata["store_uuid"]
    assert TargetStore(destination).metadata() == source_metadata

    copied_cas = _only_cas_path(destination)
    assert copied_cas.read_bytes() == source_cas.read_bytes() == structure_bytes
    dataset = load("spin", db=destination)
    assert dataset.info["size"] == 1
    assert dataset.CSD_identifier["EUCASE"].label == {"spin": 3}
    assert dataset[0].structures.path("xyz").read_bytes() == structure_bytes

    with sqlite3.connect(destination) as connection:
        copied_binding = connection.execute(
            "SELECT binding_json FROM ingestion_receipts"
        ).fetchone()[0]
    assert copied_binding == source_binding
    assert json.loads(copied_binding)["target_context"]["canonical_path"] == (
        historical_path
    )


def test_copy_cli_emits_the_closed_store_replica_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, source_database, _, _ = _source_package(tmp_path)
    target_package = _target_package(tmp_path)

    args = build_parser().parse_args(
        ["copy", str(target_package), "--db", str(source_database)]
    )
    assert run(args) == 0

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["key"] == "hotpot.ds.copy"
    assert envelope["channel"] == "default"
    payload = envelope["payload"]
    assert payload["target_package"] == str(target_package.resolve())
    assert payload["destination_database"] == str(
        _destination_database(target_package)
    )
    assert payload["counts"] == {
        "datasets": 1,
        "releases": 1,
        "cases": 1,
        "assets": 1,
        "asset_bytes": len(b"1\n\nEu 0 0 0\n"),
    }
    assert payload["audit"] == {"passed": True}


@pytest.mark.parametrize(
    "occupied", ["active-file", "active-directory", "active-link"]
)
def test_copy_store_refuses_an_occupied_target_without_changes(
    tmp_path: Path, occupied: str
) -> None:
    _, source_database, _, _ = _source_package(tmp_path)
    target_package = _target_package(tmp_path)
    store_root = target_package / "dataset" / "data" / "store"
    store_root.mkdir(parents=True)
    active = store_root / "active"
    if occupied == "active-file":
        active.write_bytes(b"do not replace")
    elif occupied == "active-directory":
        active.mkdir()
        (active / "owner-marker").write_bytes(b"do not replace")
    elif occupied == "active-link":
        active.symlink_to("missing-generation")
    before_kind = (
        "symlink"
        if active.is_symlink()
        else "directory"
        if active.is_dir()
        else "file"
    )
    before_value = (
        active.readlink()
        if active.is_symlink()
        else (active / "owner-marker").read_bytes()
        if active.is_dir()
        else active.read_bytes()
    )

    with pytest.raises(StoreCopyError, match="target|active|empty"):
        copy_store(source_database, target_package)

    after_kind = (
        "symlink"
        if active.is_symlink()
        else "directory"
        if active.is_dir()
        else "file"
    )
    after_value = (
        active.readlink()
        if active.is_symlink()
        else (active / "owner-marker").read_bytes()
        if active.is_dir()
        else active.read_bytes()
    )
    assert (after_kind, after_value) == (before_kind, before_value)
    assert not (store_root / "generations").exists()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_copy_store_rejects_invalid_cas_before_publication(
    tmp_path: Path, damage: str
) -> None:
    _, source_database, source_generation_db, _ = _source_package(tmp_path)
    cas_path = _only_cas_path(source_generation_db)
    if damage == "missing":
        cas_path.unlink()
    else:
        cas_path.write_bytes(b"corrupt")
    target_package = _target_package(tmp_path)

    with pytest.raises(StoreCopyError, match="asset|CAS|integrity|missing"):
        copy_store(source_database, target_package)

    store_root = target_package / "dataset" / "data" / "store"
    assert not (store_root / "active").exists()
    assert not (store_root / "active").is_symlink()
    generations = store_root / "generations"
    assert not generations.exists() or not tuple(generations.iterdir())
