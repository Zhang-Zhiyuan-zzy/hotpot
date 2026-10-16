from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import pytest

import hotpot.ds._bundle_cleanup as cleanup_module
from hotpot.ds._bundle_cleanup import BundleCleanupError, remove_bundle_intermediates
from hotpot.ds.cli import build_parser, run
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA
from hotpot.ds.public.bundle import Bundle
from hotpot.ds.storage import TargetSchemaError, TargetStore


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sealed_bundle(tmp_path: Path) -> tuple[Bundle, Path, Path]:
    root = tmp_path / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    structures = root / "structures"
    structures.mkdir()
    structure = b"O\n"
    (structures / "sample.smi").write_bytes(structure)
    source = root / "evidence" / "original" / "chemist.txt"
    source.parent.mkdir(parents=True)
    source.write_text("curator classification\n", encoding="utf-8")
    migration = root / "evidence" / "migration" / "identity-migration.jsonl"
    migration.parent.mkdir()
    migration.write_text('{"legacy":"case-1"}\n', encoding="utf-8")
    source_inventory = migration.parent / "legacy-source-inventory.json"
    source_inventory.write_text('{"sources":[]}\n', encoding="utf-8")
    provenance = migration.parent / "migration-provenance.json"
    provenance.write_text('{"method":"curated"}\n', encoding="utf-8")
    supplemental = migration.parent / "supplier-notes.json"
    supplemental.write_text('{"note":"original"}\n', encoding="utf-8")
    case = {
        "id": "case-1",
        "local_id": "local-1",
        "public_index": 0,
        "core_info": {"sample": "sample-1"},
        "observations": [
            {"observation_id": "observation-1", "property": "spin", "value": 1}
        ],
        "structures": [
            {
                "structure_id": "structure-1",
                "role": "primary",
                "format": "smi",
                "path": "structures/sample.smi",
                "primary": True,
            }
        ],
    }
    intake.write_text(json.dumps(case) + "\n", encoding="utf-8")
    def control_asset(category: str, name: str, path: Path) -> dict:
        payload = path.read_bytes()
        return {
            "category": category,
            "name": name,
            "path": path.relative_to(root).as_posix(),
            "sha256": _digest(payload),
            "size": len(payload),
        }

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "demo",
        "release_id": "demo-r1",
        "release_serial": 1,
        "version": "1.0",
        "title": "Demo",
        "description": "Bundle cleanup fixture",
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
        "control_assets": [
            control_asset("migration", "identity-migration", migration),
            control_asset("migration", "legacy-source-inventory", source_inventory),
            control_asset("migration", "migration-provenance", provenance),
            control_asset("migration", "supplier-notes", supplemental),
        ],
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    database = tmp_path / "datasets.sqlite3"
    bundle = Bundle(root, "demo", db=database)
    bundle.synchronize()
    bundle.write_certificate()
    bundle.ingest()
    cache = root / "assessments" / "cache"
    cache.mkdir()
    (cache / "reproducible.bin").write_bytes(b"derived")
    return bundle, root, database


def test_cleanup_removes_only_allowlisted_reproducible_data_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, root, database = _sealed_bundle(tmp_path)
    generation = bundle._current_generation.manifest_path.parent
    result = remove_bundle_intermediates(root, "demo", database)

    assert result["already_clean"] is False
    assert result["removed_count"] > 3
    assert result["removed_bytes"] > 0
    assert not (root / "cases" / "facts.sqlite3").exists()
    assert not generation.exists()
    assert not (
        root / "evidence" / "migration" / "identity-migration.jsonl"
    ).exists()
    assert not (root / "assessments" / "cache").exists()
    for relative in (
        "bundle.json",
        "certificate.json",
        "ingestion-receipt.json",
        "evidence/intake/cases.jsonl",
        "evidence/original/chemist.txt",
        "evidence/migration/legacy-source-inventory.json",
        "evidence/migration/migration-provenance.json",
        "evidence/migration/supplier-notes.json",
        "structures/sample.smi",
        "policy/policy.json",
        "audit/events.jsonl",
    ):
        assert (root / relative).is_file()

    cleanup = json.loads((root / "bundle-cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["state"] == "complete"
    assert cleanup["receipt"]["receipt_id"] == result["receipt_id"]
    assert cleanup["target_audit"]["case_count"] == 1
    assert cleanup["removed_count"] == len(cleanup["deleted"])
    assert cleanup["removed_bytes"] == sum(item["bytes"] for item in cleanup["deleted"])
    stored = TargetStore(database).get_case("demo", "demo-r1", "case-1")
    assert stored["facts"]["observations"][0]["value"] == 1
    with TargetStore(database)._connect() as connection:
        controls = {
            (row["category"], row["name"])
            for row in connection.execute(
                "SELECT category,name FROM release_control_asset_refs"
            )
        }
    assert {
        ("provenance", "legacy-source-inventory"),
        ("provenance", "migration-provenance"),
    } <= controls
    assert not {
        name
        for category, name in controls
        if category == "migration" or name == "identity-migration"
    }

    monkeypatch.setattr(
        cleanup_module,
        "load_bundle",
        lambda *args, **kwargs: pytest.fail("idempotent cleanup reopened the Bundle"),
    )
    repeated = remove_bundle_intermediates(root, "demo", database)
    assert repeated == {**result, "already_clean": True}


def test_cleanup_rejects_a_bundle_without_an_authoritative_ingestion_receipt(
    tmp_path: Path,
) -> None:
    bundle, root, database = _sealed_bundle(tmp_path)
    (root / "ingestion-receipt.json").unlink()
    with TargetStore(database)._connect() as connection:
        connection.execute("DELETE FROM ingestion_receipts")
        connection.commit()

    with pytest.raises(BundleCleanupError, match="successfully ingested"):
        remove_bundle_intermediates(root, "demo", database)
    assert (root / "cases" / "facts.sqlite3").is_file()
    assert bundle.certificate_path.is_file()


def test_cleanup_rejects_corrupt_target_owned_assets(tmp_path: Path) -> None:
    _bundle, root, database = _sealed_bundle(tmp_path)
    store = TargetStore(database)
    with store._connect() as connection:
        row = connection.execute(
            "SELECT relative_path FROM target_assets ORDER BY content_sha256 LIMIT 1"
        ).fetchone()
    (store.asset_root / row["relative_path"]).write_bytes(b"corrupt")

    with pytest.raises(BundleCleanupError, match="target-owned asset"):
        remove_bundle_intermediates(root, "demo", database)
    assert (root / "cases" / "facts.sqlite3").is_file()
    assert not (root / "bundle-cleanup.json").exists()


def test_cleanup_rejects_corrupt_compressed_target_facts(tmp_path: Path) -> None:
    _bundle, root, database = _sealed_bundle(tmp_path)
    store = TargetStore(database)
    with store._connect() as connection:
        connection.execute(
            "UPDATE cases SET source_payload_blob=?",
            (b"not-zlib",),
        )
        connection.commit()

    with pytest.raises(BundleCleanupError, match="corrupt compressed Case facts"):
        remove_bundle_intermediates(root, "demo", database)
    assert (root / "cases" / "facts.sqlite3").is_file()
    assert not (root / "bundle-cleanup.json").exists()


def test_cleanup_explicitly_rejects_an_old_target_schema(tmp_path: Path) -> None:
    _bundle, root, database = _sealed_bundle(tmp_path)
    store = TargetStore(database)
    with store._connect() as connection:
        connection.execute(
            "UPDATE store_metadata SET schema_id='hotpot-ds-store/legacy'"
        )
        connection.commit()

    with pytest.raises(TargetSchemaError, match="current Hotpot schema"):
        remove_bundle_intermediates(root, "demo", database)
    assert (root / "cases" / "facts.sqlite3").is_file()
    assert not (root / "bundle-cleanup.json").exists()


def test_resumed_cleanup_revalidates_target_before_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bundle, root, database = _sealed_bundle(tmp_path)
    unlink = cleanup_module._unlink_inventory
    monkeypatch.setattr(
        cleanup_module,
        "_unlink_inventory",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        remove_bundle_intermediates(root, "demo", database)
    cleanup = json.loads((root / "bundle-cleanup.json").read_text(encoding="utf-8"))
    assert cleanup["state"] == "planned"
    monkeypatch.setattr(cleanup_module, "_unlink_inventory", unlink)

    store = TargetStore(database)
    with store._connect() as connection:
        row = connection.execute(
            "SELECT source_payload_blob FROM cases"
        ).fetchone()
        facts = json.loads(zlib.decompress(row["source_payload_blob"]))
        facts["core_info"]["sample"] = "changed-after-plan"
        canonical = cleanup_module._canonical_bytes(facts)
        connection.execute(
            "UPDATE cases SET source_payload_blob=?,source_payload_digest=?",
            (zlib.compress(canonical, level=6), hashlib.sha256(canonical).digest()),
        )
        connection.commit()

    with pytest.raises(BundleCleanupError, match="target changed"):
        remove_bundle_intermediates(root, "demo", database)
    assert (root / "cases" / "facts.sqlite3").is_file()
    assert json.loads(
        (root / "bundle-cleanup.json").read_text(encoding="utf-8")
    )["state"] == "planned"


def test_resumed_cleanup_revalidates_certificate_before_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bundle, root, database = _sealed_bundle(tmp_path)
    unlink = cleanup_module._unlink_inventory
    monkeypatch.setattr(
        cleanup_module,
        "_unlink_inventory",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        remove_bundle_intermediates(root, "demo", database)
    monkeypatch.setattr(cleanup_module, "_unlink_inventory", unlink)

    certificate = root / "certificate.json"
    certificate.write_bytes(certificate.read_bytes() + b" ")

    with pytest.raises(BundleCleanupError, match="certificate"):
        remove_bundle_intermediates(root, "demo", database)
    assert (root / "cases" / "facts.sqlite3").is_file()
    assert json.loads(
        (root / "bundle-cleanup.json").read_text(encoding="utf-8")
    )["state"] == "planned"


def test_idempotent_cleanup_manifest_cannot_expand_the_delete_allowlist(
    tmp_path: Path,
) -> None:
    _bundle, root, database = _sealed_bundle(tmp_path)
    remove_bundle_intermediates(root, "demo", database)
    path = root / "bundle-cleanup.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    bundle_bytes = (root / "bundle.json").read_bytes()
    document["deleted"].append(
        {
            "path": "bundle.json",
            "sha256": _digest(bundle_bytes),
            "bytes": len(bundle_bytes),
            "kind": "file",
            "category": "canonical-facts",
        }
    )
    document["removed_count"] += 1
    document["removed_bytes"] += len(bundle_bytes)
    document["manifest_sha256"] = cleanup_module._manifest_digest(document)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(BundleCleanupError, match="unsafe deletion record"):
        remove_bundle_intermediates(root, "demo", database)
    assert (root / "bundle.json").read_bytes() == bundle_bytes


@pytest.mark.parametrize(
    "name",
    ["legacy-source-inventory.json", "migration-provenance.json"],
)
def test_cleanup_manifest_never_deletes_source_inventory_or_provenance(
    tmp_path: Path,
    name: str,
) -> None:
    _bundle, root, database = _sealed_bundle(tmp_path)
    remove_bundle_intermediates(root, "demo", database)
    cleanup_path = root / "bundle-cleanup.json"
    document = json.loads(cleanup_path.read_text(encoding="utf-8"))
    protected = root / "evidence" / "migration" / name
    payload = protected.read_bytes()
    document["deleted"].append(
        {
            "path": protected.relative_to(root).as_posix(),
            "sha256": _digest(payload),
            "bytes": len(payload),
            "kind": "file",
            "category": "derived-control",
        }
    )
    document["removed_count"] += 1
    document["removed_bytes"] += len(payload)
    document["manifest_sha256"] = cleanup_module._manifest_digest(document)
    cleanup_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(BundleCleanupError, match="unsafe deletion record"):
        remove_bundle_intermediates(root, "demo", database)
    assert protected.read_bytes() == payload


def test_bundle_rm_cli_uses_bound_target_and_emits_closed_envelope(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bundle, root, database = _sealed_bundle(tmp_path)
    parser = build_parser()
    monkeypatch.chdir(root)
    command = parser.parse_args(["bundle", "rm", "--all"])
    assert run(command) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["key"] == "hotpot.ds.bundle.rm"
    assert envelope["channel"] == "default"
    assert envelope["payload"]["bundle"]["dataset"] == "demo"
    assert envelope["payload"]["already_clean"] is False

    asserted = parser.parse_args(
        [
            "bundle",
            "--root",
            str(root),
            "--dataset",
            "demo",
            "--db",
            str(database),
            "rm",
            "--all",
        ]
    )
    assert run(asserted) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["payload"]["already_clean"] is True
