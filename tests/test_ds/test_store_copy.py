from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import zlib
from pathlib import Path

import pytest

import hotpot
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
    included_case_content_digest,
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
    source_evidence = tmp_path / "chemist-note.txt"
    evidence_bytes = b"chemist-verified spin assignment\n"
    source_evidence.write_bytes(evidence_bytes)
    evidence_digest = hashlib.sha256(evidence_bytes).hexdigest()
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
            },
            {
                "collection": "evidence",
                "declaration_id": "chemist-note",
                "asset_index": 0,
                "role": "annotation",
                "status": "active",
                "source_path": source_evidence,
                "sha256": evidence_digest,
                "size": len(evidence_bytes),
            },
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
                "included_case_content": case_content_digest([case]),
                "included_cases": case_set_digest([case["id"]]),
                "excluded_cases": case_set_digest([]),
                "asset_inventory": asset_inventory_digest(case["assets"]),
            },
            "control_assets": [
                {
                    "category": "provenance",
                    "name": "chemist-note",
                    "source_path": source_evidence,
                    "sha256": evidence_digest,
                    "size": len(evidence_bytes),
                }
            ],
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
    running = Path(hotpot.__file__).resolve().parent
    (package / "dataset").mkdir(parents=True)
    shutil.copy2(running / "__init__.py", package / "__init__.py")
    shutil.copy2(running / "__version__.py", package / "__version__.py")
    shutil.copy2(
        running / "dataset" / "__init__.py",
        package / "dataset" / "__init__.py",
    )
    (package / "cheminfo").mkdir()
    shutil.copy2(
        running / "cheminfo" / "__init__.py",
        package / "cheminfo" / "__init__.py",
    )
    shutil.copytree(running / "ds", package / "ds")
    return package


def _only_cas_path(database: Path) -> Path:
    store = TargetStore(database)
    with store._connect() as connection:
        relative_path = connection.execute(
            "SELECT t.relative_path FROM target_assets t JOIN structure_asset_refs r "
            "USING(content_sha256) WHERE r.structure_id='structure-eu'"
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


def test_copied_store_can_become_the_single_target_for_a_later_ingest(
    tmp_path: Path,
) -> None:
    _, source_database, _, _ = _source_package(tmp_path)
    target_package = _target_package(tmp_path)
    copy_store(source_database, target_package)
    destination = _destination_database(target_package)
    store = TargetStore(destination)
    metadata = store.metadata()
    case = {
        "id": "case-oxidation",
        "public_index": 0,
        "local_id": "local-oxidation",
        "facts": {"labels": {"oxidation_state": 3}},
        "assessment": {
            "archive_class": "ordinary",
            "quality_level": "B2",
            "evidence_level": "E2",
            "benchmark": False,
            "reasons": [],
        },
        "identifiers": [{"namespace": "CSD", "value": "OXCASE"}],
        "structures": [],
        "assets": [],
    }
    policy = {
        "schema": "hotpot-ds-policy/current",
        "gates": {"quality": "test"},
        "runtime_policy": runtime_policy(),
    }
    store.ingest(
        {
            "target_context": {
                "canonical_path": str(store.path),
                "store_state": "existing",
                "store_uuid": metadata["store_uuid"],
                "schema_id": TARGET_SCHEMA_ID,
                "dataset_state": "absent",
                "dataset_policy_revision": None,
                "dataset_policy_digest": None,
            },
            "bundle": {
                "dataset": "oxidation",
                "release": "oxidation.v001",
                "version": "1.0",
                "release_serial": 1,
                "title": "Oxidation states",
                "description": "Post-copy ingest fixture",
                "metadata": {"core_labels": ["oxidation_state"]},
            },
            "policy": {
                "revision": 1,
                "digest": _json_digest(policy),
                "value": policy,
            },
            "digests": {
                "bundle": hashlib.sha256(b"oxidation-bundle").hexdigest(),
                "certificate": hashlib.sha256(b"oxidation-certificate").hexdigest(),
                "generation": hashlib.sha256(b"oxidation-generation").hexdigest(),
                "case_content": case_content_digest([case]),
                "included_case_content": included_case_content_digest([case]),
                "included_cases": case_set_digest([case["id"]]),
                "excluded_cases": case_set_digest([]),
                "asset_inventory": asset_inventory_digest([]),
            },
            "cases": [case],
        }
    )

    assert load("oxidation", db=destination).CSD_identifier["OXCASE"].label == {
        "oxidation_state": 3
    }


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
        "assets": 2,
        "asset_bytes": len(b"1\n\nEu 0 0 0\n")
        + len(b"chemist-verified spin assignment\n"),
    }
    assert payload["audit"] == {"passed": True}


def test_copy_store_rejects_a_directory_that_is_not_the_current_hotpot_runtime(
    tmp_path: Path,
) -> None:
    _, source_database, _, _ = _source_package(tmp_path)
    package = tmp_path / "fake" / "hotpot"
    (package / "dataset").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(StoreCopyError, match="Hotpot package|runtime|importable"):
        copy_store(source_database, package)

    assert not (package / "dataset" / "data").exists()


def test_copy_store_rejects_noncurrent_schema_definitions(tmp_path: Path) -> None:
    _, source_database, source_generation_db, _ = _source_package(tmp_path)
    with sqlite3.connect(source_generation_db) as connection:
        connection.execute(
            "CREATE TRIGGER altered_current_schema AFTER INSERT ON identifiers "
            "BEGIN SELECT 1; END"
        )
    target_package = _target_package(tmp_path)

    with pytest.raises(StoreCopyError, match="schema"):
        copy_store(source_database, target_package)

    assert not (
        target_package / "dataset" / "data" / "store" / "active"
    ).exists()


def test_copy_store_rejects_a_different_ds_runtime(tmp_path: Path) -> None:
    _, source_database, _, _ = _source_package(tmp_path)
    target_package = _target_package(tmp_path)
    with (target_package / "ds" / "storage.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# different checkout\n")

    with pytest.raises(StoreCopyError, match="exact current ds runtime"):
        copy_store(source_database, target_package)

    assert not (target_package / "dataset" / "data").exists()


@pytest.mark.parametrize(
    "tamper", ["facts_with_matching_row_digest", "identifier", "assessment"]
)
def test_copy_store_rejects_unbound_persisted_case_content(
    tmp_path: Path, tamper: str
) -> None:
    _, source_database, source_generation_db, _ = _source_package(tmp_path)
    with sqlite3.connect(source_generation_db) as connection:
        if tamper == "facts_with_matching_row_digest":
            facts = {
                "core_info": {"sample": "eu-complex"},
                "labels": {"spin": 99},
            }
            canonical = json.dumps(
                facts,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            connection.execute(
                "UPDATE cases SET source_payload_blob=?,source_payload_digest=?",
                (zlib.compress(canonical, level=6), hashlib.sha256(canonical).digest()),
            )
        elif tamper == "identifier":
            connection.execute(
                "UPDATE identifiers SET value='ALTERED-CSD-IDENTIFIER'"
            )
        else:
            connection.execute("DELETE FROM case_assessments")
    target_package = _target_package(tmp_path)

    with pytest.raises(StoreCopyError, match="Case|content|assessment|receipt"):
        copy_store(source_database, target_package)

    assert not (
        target_package / "dataset" / "data" / "store" / "active"
    ).exists()


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE structure_asset_refs SET role='computed'",
        "UPDATE structure_asset_refs SET format='mol2'",
        "UPDATE structure_asset_refs SET is_primary=0",
        "UPDATE structure_asset_refs SET content_sha256=(SELECT content_sha256 "
        "FROM scientific_asset_refs WHERE declaration_id='chemist-note')",
        "UPDATE scientific_asset_refs SET role='reclassified' "
        "WHERE declaration_id='chemist-note'",
        "UPDATE scientific_asset_refs SET content_sha256=(SELECT content_sha256 "
        "FROM structure_asset_refs) WHERE declaration_id='chemist-note'",
        "UPDATE release_control_asset_refs SET name='other-provenance'",
    ],
    ids=[
        "structure-role",
        "structure-format",
        "structure-primary",
        "structure-object-reassociation",
        "scientific-role",
        "scientific-object-reassociation",
        "release-control-reference",
    ],
)
def test_copy_store_rejects_tampered_asset_reference_relationships(
    tmp_path: Path, statement: str
) -> None:
    _, source_database, source_generation_db, _ = _source_package(tmp_path)
    with sqlite3.connect(source_generation_db) as connection:
        connection.execute(statement)
    target_package = _target_package(tmp_path)

    with pytest.raises(StoreCopyError, match="formal release content|receipt"):
        copy_store(source_database, target_package)

    assert not (
        target_package / "dataset" / "data" / "store" / "active"
    ).exists()


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE datasets SET title='Altered dataset title'",
        "UPDATE datasets SET metadata_json='{}'",
        "UPDATE releases SET metadata_json='{}'",
    ],
    ids=["dataset-title", "dataset-metadata", "release-metadata"],
)
def test_copy_store_rejects_tampered_dataset_or_release_metadata(
    tmp_path: Path, statement: str
) -> None:
    _, source_database, source_generation_db, _ = _source_package(tmp_path)
    with sqlite3.connect(source_generation_db) as connection:
        connection.execute(statement)
    target_package = _target_package(tmp_path)

    with pytest.raises(StoreCopyError, match="formal release content|receipt"):
        copy_store(source_database, target_package)

    assert not (
        target_package / "dataset" / "data" / "store" / "active"
    ).exists()


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
