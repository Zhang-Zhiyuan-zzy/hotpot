"""Focused tests for the one-time current-runtime rebuild operator."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from hotpot.ds.gates._persistence import default_gates, serialize_gate_configs
from hotpot.ds.public._implementation import runtime_policy


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "plan"
    / "13_dataset_bundle_governance"
    / "rebuild_current_runtime.py"
)


@pytest.fixture(scope="module")
def rebuild_module():
    specification = importlib.util.spec_from_file_location(
        "stage13_rebuild_current_runtime", SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_bundle_closure_copies_only_referenced_bytes(tmp_path, rebuild_module):
    source = tmp_path / "source"
    partial = tmp_path / "candidate.partial"
    structure = b"2\nexample\nH 0 0 0\nH 0 0 1\n"
    control = b'{"owner":"Hotpot"}\n'
    (source / "assets").mkdir(parents=True)
    (source / "assets" / "molecule.xyz").write_bytes(structure)
    (source / "assets" / "unreferenced.xyz").write_bytes(b"not copied")
    (source / "evidence").mkdir()
    (source / "evidence" / "control.json").write_bytes(control)
    case = {
        "id": "case-1",
        "local_id": "one",
        "structures": [
            {
                "structure_id": "s1",
                "representations": [
                    {"path": "assets/molecule.xyz", "format": "xyz"}
                ],
            }
        ],
        "evidence": [{"path": "evidence/intentionally-missing.txt"}],
    }
    (source / "cases.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
    _write_json(
        source / "bundle.json",
        {
            "ds_name": "fixture",
            "release_id": "fixture.v001",
            "intake": {"format": "jsonl", "path": "cases.jsonl"},
            "control_assets": [
                {
                    "category": "owner",
                    "name": "decision",
                    "path": "evidence/control.json",
                    "sha256": rebuild_module.hashlib.sha256(control).hexdigest(),
                    "size": len(control),
                }
            ],
        },
    )

    closure = rebuild_module._copy_bundle_closure(source, partial)

    assert (partial / "assets" / "molecule.xyz").read_bytes() == structure
    assert not (partial / "assets" / "unreferenced.xyz").exists()
    assert not (partial / "evidence" / "intentionally-missing.txt").exists()
    assert rebuild_module._observe_bundle_closure(partial, closure) == closure
    missing = {item["path"]: item for item in closure["entries"]}
    assert missing["evidence/intentionally-missing.txt"]["state"] == "missing"


def test_plan_is_read_only_and_binds_final_paths(tmp_path, rebuild_module):
    repository = tmp_path / "repo"
    data = repository / "hotpot" / "dataset" / "data"
    store = data / "store"
    generation = store / "generations" / rebuild_module.SOURCE_GENERATION
    generation.mkdir(parents=True)
    database = generation / "datasets.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE superseded_layout(marker TEXT)")
    connection.execute("INSERT INTO superseded_layout VALUES('opaque rollback anchor')")
    policy = {
        "schema": "hotpot-ds-policy/current",
        "gates": serialize_gate_configs(default_gates()),
        "identity_schema": {},
        "required_structure_roles": [],
        "property_definitions": {},
        "property_comparators": {},
        "runtime_policy": runtime_policy(),
    }
    for spec in rebuild_module.DATASETS:
        root = data / "bundles" / "stage13-migration" / spec.source_relative
        manifest = {
            "schema": "hotpot-ds-bundle/current",
            "ds_name": spec.name,
            "dataset_id": spec.name,
            "release_id": spec.release,
            "intake": {"format": "jsonl", "path": "cases.jsonl"},
            "provenance": {"legacy": {"fixture": True}},
        }
        _write_json(
            root / "bundle.json",
            manifest,
        )
        (root / "cases.jsonl").write_text("", encoding="utf-8")
        (root / "cases").mkdir()
        facts = sqlite3.connect(root / "cases" / "facts.sqlite3")
        facts.executescript(
            """
            PRAGMA user_version=2;
            CREATE TABLE audit_events(id TEXT);
            CREATE TABLE cases(id TEXT);
            CREATE TABLE control(singleton INTEGER PRIMARY KEY,policy_revision INTEGER);
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value_json TEXT);
            CREATE TABLE policy(singleton INTEGER PRIMARY KEY,policy_json TEXT);
            INSERT INTO control VALUES(1,7);
            """
        )
        facts.execute(
            "INSERT INTO metadata VALUES('manifest',?)",
            (json.dumps(manifest, sort_keys=True, separators=(",", ":")),),
        )
        facts.execute(
            "INSERT INTO policy VALUES(1,?)",
            (json.dumps(policy, sort_keys=True, separators=(",", ":")),),
        )
        facts.commit()
        facts.close()
        _write_json(root / "policy" / "policy.json", policy)
        _write_json(
            root / "migration-export.json",
            {
                "schema": "hotpot-ds-legacy-migration/current",
                "phase": "export",
                "dataset": spec.name,
                "release": spec.release,
            },
        )
        _write_json(
            root / "migration-parity.json",
            {
                "schema": "hotpot-ds-legacy-migration/current",
                "phase": "source-export-parity",
                "dataset": spec.name,
                "release": spec.release,
                "passed": True,
            },
        )
    connection.commit()
    connection.close()
    os.symlink(
        os.path.relpath(generation, store),
        store / "active",
    )

    operator = rebuild_module.Operator(
        repository, rebuild_module.RUN_ID, rebuild_module.SOURCE_GENERATION
    )
    plan = operator.plan()

    assert plan["mode"] == "read-only-plan"
    assert Path(plan["candidate_database"]) == operator.output_database
    assert all(".partial" not in item["candidate"] for item in plan["datasets"])
    assert all(item["source_policy_revision"] == 7 for item in plan["datasets"])
    assert not operator.state_root.exists()
    assert not operator.output_generation.exists()


def test_migration_proof_binds_intake_facts_and_identity_files(
    tmp_path, rebuild_module, monkeypatch
):
    root = tmp_path / "legacy"
    spec = rebuild_module.DatasetSpec("fixture", "fixture.v001", "legacy")
    root.mkdir()
    intake = b'{"id":"case-1"}\n'
    inventory = b'{"legacy":"identity"}\n'
    migration = b'{"legacy":"mapping"}\n'
    (root / "cases.jsonl").write_bytes(intake)
    (root / "evidence").mkdir()
    (root / "evidence" / "inventory.jsonl").write_bytes(inventory)
    (root / "evidence" / "migration.jsonl").write_bytes(migration)
    manifest = {
        "schema": "hotpot-ds-bundle/current",
        "ds_name": spec.name,
        "release_id": spec.release,
        "intake": {"format": "jsonl", "path": "cases.jsonl"},
        "provenance": {"legacy": {"database_sha256": "a" * 64}},
        "control_assets": [
            {
                "category": "migration",
                "name": "identity-inventory",
                "path": "evidence/inventory.jsonl",
                "sha256": rebuild_module.hashlib.sha256(inventory).hexdigest(),
                "size": len(inventory),
            },
            {
                "category": "migration",
                "name": "identity-migration",
                "path": "evidence/migration.jsonl",
                "sha256": rebuild_module.hashlib.sha256(migration).hexdigest(),
                "size": len(migration),
            },
        ],
    }
    _write_json(root / "bundle.json", manifest)
    (root / "cases").mkdir()
    with sqlite3.connect(root / "cases" / "facts.sqlite3") as facts:
        facts.executescript(
            """
            PRAGMA user_version=2;
            CREATE TABLE audit_events(id TEXT);
            CREATE TABLE cases(id TEXT);
            CREATE TABLE control(id TEXT);
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value_json TEXT);
            CREATE TABLE policy(id TEXT);
            INSERT INTO cases VALUES('case-1');
            """
        )
        facts.execute(
            "INSERT INTO metadata VALUES('manifest',?)",
            (json.dumps(manifest, sort_keys=True, separators=(",", ":")),),
        )
    source_inventory = {"records": {"count": 1, "row_set_sha256": "b" * 64}}
    export = {
        "schema": "hotpot-ds-legacy-migration/current",
        "phase": "export",
        "dataset": spec.name,
        "release": spec.release,
        "case_count": 1,
        "cases_sha256": rebuild_module.hashlib.sha256(intake).hexdigest(),
        "source_inventory": source_inventory,
        "identity_inventory": {
            "path": "evidence/inventory.jsonl",
            "sha256": rebuild_module.hashlib.sha256(inventory).hexdigest(),
            "size": len(inventory),
        },
        "identity_migration": {
            "path": "evidence/migration.jsonl",
            "sha256": rebuild_module.hashlib.sha256(migration).hexdigest(),
            "size": len(migration),
        },
    }
    parity = {
        "schema": "hotpot-ds-legacy-migration/current",
        "phase": "source-export-parity",
        "dataset": spec.name,
        "release": spec.release,
        "passed": True,
        "case_count": 1,
        "unique_case_ids": 1,
        "stable_source_snapshot": True,
        "control_bindings_valid": True,
        "identity_inventory_coverage_complete": True,
        "identity_new_ids_valid": True,
        "persistent_indices_contiguous": True,
        "mismatches": {},
        "case_mismatches": [],
        "identity_file_mismatches": [],
        "source_inventory": source_inventory,
        "export_inventory": source_inventory,
        "source_database_sha256": "a" * 64,
    }
    _write_json(root / "migration-export.json", export)
    _write_json(root / "migration-parity.json", parity)
    operator = rebuild_module.Operator(
        tmp_path / "repo", rebuild_module.RUN_ID, rebuild_module.SOURCE_GENERATION
    )
    operator.source_bundles = tmp_path
    operator.output_bundles = tmp_path / "candidate-bundles"
    candidate = operator._output_root(spec)
    closure = rebuild_module._copy_bundle_closure(root, candidate)

    proof = rebuild_module._migration_proof_binding(root, spec, closure)
    assert proof["case_count"] == 1
    assert proof["cases_sha256"] == export["cases_sha256"]

    artifact = tmp_path / "closure.json"
    rebuild_module._atomic_json(artifact, closure)
    artifact_sha256, artifact_size, _ = rebuild_module._sha256_file(artifact)
    copy_result = {
        "candidate": str(candidate.resolve()),
        "closure_artifact": str(artifact.resolve()),
        "closure_artifact_sha256": artifact_sha256,
        "closure_artifact_size": artifact_size,
        "closure_digest": closure["closure_digest"],
        "legacy_proof": proof,
    }
    state = {
        "phases": {
            f"copy:{spec.name}": {"result": copy_result},
            f"process:{spec.name}": {"result": {}},
            "parity": {
                "result": {
                    "artifact_sha256": "a" * 64,
                    "semantic": {"semantic_digest": "b" * 64},
                    "source_anchor": {"sha256": "c" * 64},
                    "target_database": {"sha256": "d" * 64},
                    "target_cas": {"sha256": "e" * 64},
                }
            },
        }
    }
    monkeypatch.setattr(rebuild_module, "DATASETS", (spec,))
    monkeypatch.setattr(operator, "_validate_parity_phase", lambda _result: None)
    monkeypatch.setattr(
        operator, "_validate_process_phase", lambda _spec, _result: None
    )
    operator._preactivation_recheck(state)

    export["cases_sha256"] = "0" * 64
    _write_json(root / "migration-export.json", export)
    with pytest.raises(rebuild_module.RebuildError, match="does not bind"):
        rebuild_module._migration_proof_binding(root, spec, closure)
    with pytest.raises(rebuild_module.RebuildError, match="does not bind"):
        operator._preactivation_recheck(state)


def test_cas_verification_is_bidirectional(tmp_path, rebuild_module):
    database = tmp_path / "datasets.sqlite3"
    asset_root = Path(str(database) + ".assets")
    payload = b"content-addressed bytes"
    digest = rebuild_module.hashlib.sha256(payload).hexdigest()
    relative = f"objects/sha256/{digest[:2]}/{digest}"
    path = asset_root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE target_assets(
          content_sha256 TEXT PRIMARY KEY,size INTEGER,relative_path TEXT
        );
        CREATE TABLE structure_asset_refs(content_sha256 TEXT);
        CREATE TABLE scientific_asset_refs(content_sha256 TEXT);
        CREATE TABLE release_control_asset_refs(
          content_sha256 TEXT,category TEXT,name TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO target_assets VALUES(?,?,?)", (digest, len(payload), relative)
    )
    connection.execute("INSERT INTO scientific_asset_refs VALUES(?)", (digest,))
    connection.commit()
    connection.close()

    result = rebuild_module._verify_cas(database)
    assert result["count"] == 1
    assert result["bytes"] == len(payload)
    lock = path.with_name(f".{digest}.lock")
    lock.touch()
    assert rebuild_module._verify_cas(database)["lock_files"] == 1

    extra = asset_root / "extra"
    extra.write_bytes(b"not in database")
    with pytest.raises(rebuild_module.RebuildError, match="database/filesystem"):
        rebuild_module._verify_cas(database)
    extra.unlink()
    lock.write_bytes(b"not a valid lock")
    with pytest.raises(rebuild_module.RebuildError, match="invalid lock"):
        rebuild_module._verify_cas(database)


def test_cas_rejects_non_regular_nodes(tmp_path, rebuild_module):
    database = tmp_path / "datasets.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE target_assets(
              content_sha256 TEXT PRIMARY KEY,size INTEGER,relative_path TEXT
            );
            CREATE TABLE structure_asset_refs(content_sha256 TEXT);
            CREATE TABLE scientific_asset_refs(content_sha256 TEXT);
            CREATE TABLE release_control_asset_refs(
              content_sha256 TEXT,category TEXT,name TEXT
            );
            """
        )
    asset_root = Path(str(database) + ".assets")
    asset_root.mkdir()
    fifo = asset_root / "unexpected-fifo"
    os.mkfifo(fifo)

    with pytest.raises(rebuild_module.RebuildError, match="non-regular"):
        rebuild_module._verify_cas(database)


def test_cas_rejects_unreferenced_database_objects(tmp_path, rebuild_module):
    database = tmp_path / "datasets.sqlite3"
    asset_root = Path(str(database) + ".assets")
    payload = b"orphan"
    digest = rebuild_module.hashlib.sha256(payload).hexdigest()
    relative = f"objects/sha256/{digest[:2]}/{digest}"
    path = asset_root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE target_assets(
              content_sha256 TEXT PRIMARY KEY,size INTEGER,relative_path TEXT
            );
            CREATE TABLE structure_asset_refs(content_sha256 TEXT);
            CREATE TABLE scientific_asset_refs(content_sha256 TEXT);
            CREATE TABLE release_control_asset_refs(
              content_sha256 TEXT,category TEXT,name TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO target_assets VALUES(?,?,?)",
            (digest, len(payload), relative),
        )

    with pytest.raises(rebuild_module.RebuildError, match="unreferenced"):
        rebuild_module._verify_cas(database)


def _compact_receipt_store(path, rebuild_module):
    from hotpot.ds.storage import (
        TARGET_SCHEMA_ID,
        TargetStore,
        asset_inventory_digest,
        case_content_digest,
        case_set_digest,
    )

    spec = rebuild_module.DatasetSpec("fixture", "fixture.v001", "fixture")
    store = TargetStore(path)
    facts = {
        "id": "case-1",
        "local_id": "local-1",
        "public_index": 0,
        "identifiers": {"CSD": "AAAAAA"},
        "metadata": {
            "supplier_grade": "gold",
            "migration_rows": {"large": ["derived"] * 20},
        },
        "labels": {"spin": 3},
        "evidence": [{"quote": "S = 3", "doi": "10.1/example"}],
    }
    assessment = {
        "quality_level": "B1",
        "evidence_level": "E1",
        "archive_class": "ordinary",
        "benchmark": False,
        "reasons": ["supplier_text"],
        "gate_results": {"large": ["derived"] * 20},
    }
    case = {
        "id": "case-1",
        "public_index": 0,
        "local_id": "local-1",
        "facts": facts,
        "assessment": assessment,
        "identifiers": [{"namespace": "CSD", "value": "AAAAAA"}],
        "structures": [],
        "assets": [],
    }
    policy = {
        "schema": "hotpot-ds-policy/current",
        "gates": serialize_gate_configs(default_gates()),
        "identity_schema": {},
        "required_structure_roles": [],
        "property_definitions": {},
        "property_comparators": {},
        "runtime_policy": runtime_policy(),
    }
    snapshot = {
        "target_context": {
            "canonical_path": str(path.resolve()),
            "store_state": "absent",
            "store_uuid": None,
            "schema_id": TARGET_SCHEMA_ID,
            "dataset_state": "absent",
            "dataset_policy_revision": None,
            "dataset_policy_digest": None,
        },
        "bundle": {
            "dataset": spec.name,
            "release": spec.release,
            "version": "1",
            "release_serial": 1,
            "title": "Fixture",
            "description": "Compact fixture",
            "metadata": {},
        },
        "policy": {
            "revision": 1,
            "digest": rebuild_module._json_digest(policy),
            "value": policy,
        },
        "digests": {
            "bundle": "1" * 64,
            "certificate": "2" * 64,
            "generation": "3" * 64,
            "case_content": case_content_digest([case]),
            "included_cases": case_set_digest(["case-1"]),
            "excluded_cases": case_set_digest([]),
            "asset_inventory": asset_inventory_digest([]),
        },
        "control_assets": [],
        "cases": [case],
    }
    store.ingest(
        snapshot,
        include_incomplete=True,
        reason=rebuild_module.INCOMPLETE_REASON,
    )
    projection = {
        "count": 1,
        "case_content_sha256": snapshot["digests"]["case_content"],
        "case_set_sha256": snapshot["digests"]["included_cases"],
    }
    return store, spec, projection


def test_compact_receipt_binds_retained_facts_and_final_labels(
    tmp_path, rebuild_module
):
    store, spec, projection = _compact_receipt_store(
        tmp_path / "receipt.sqlite3", rebuild_module
    )

    result = rebuild_module._validate_target_receipt(
        store, store.path, spec, projection
    )
    loaded = store.get_case(spec.name, spec.release, "case-1")

    assert result["included_count"] == 1
    assert loaded["facts"]["metadata"] == {"supplier_grade": "gold"}
    assert loaded["facts"]["evidence"][0]["quote"] == "S = 3"
    assert loaded["facts"]["labels"] == {"spin": 3}
    assert loaded["assessment"] == {
        "quality_level": "B1",
        "evidence_level": "E1",
        "archive_class": "ordinary",
        "benchmark": False,
        "reasons": ["supplier_text"],
    }


def test_bundle_to_compact_store_parity_is_casewise_and_drops_only_derived_data(
    tmp_path, rebuild_module, monkeypatch
):
    from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA
    from hotpot.ds.public.bundle import Bundle
    from hotpot.ds.storage import TargetStore

    spec = rebuild_module.DatasetSpec("fixture", "fixture.v001", "fixture")

    def make_root(parent: Path) -> Path:
        root = parent / "fixture" / spec.release
        intake = root / "cases.jsonl"
        intake.parent.mkdir(parents=True)
        case = {
            "id": "case-1",
            "local_id": "local-1",
            "public_index": 0,
            "identifiers": {"CSD": "AAAAAA"},
            "core_info": {"sample": "one"},
            "metadata": {
                "supplier_classification": "gold",
                "migration_rows": {"derived": [1, 2, 3]},
                "migration_schema": "legacy-v1",
            },
            "observations": [
                {
                    "observation_id": "observation-1",
                    "property": "spin",
                    "value": 3,
                }
            ],
            "evidence": [
                {
                    "evidence_id": "evidence-1",
                    "quote": "S = 3",
                    "doi": "10.1/example",
                    "path": "evidence/quote.txt",
                }
            ],
            "structures": [
                {
                    "structure_id": "structure-1",
                    "role": "primary",
                    "format": "smi",
                    "path": "structures/one.smi",
                    "primary": True,
                }
            ],
        }
        intake.write_text(json.dumps(case) + "\n", encoding="utf-8")
        (root / "structures").mkdir()
        (root / "structures" / "one.smi").write_text("O\n", encoding="utf-8")
        (root / "evidence").mkdir()
        (root / "evidence" / "quote.txt").write_text(
            "Reported S = 3\n", encoding="utf-8"
        )
        _write_json(
            root / "bundle.json",
            {
                "schema": BUNDLE_SCHEMA,
                "ds_name": spec.name,
                "release_id": spec.release,
                "release_serial": 1,
                "version": "1.0",
                "title": "Fixture",
                "description": "Durable parity fixture",
                "license": "CC0-1.0",
                "access_policy": "open",
                "provenance": {"created_by": "test"},
                "core_labels": ["spin"],
                "core_info": ["sample"],
                "intake": {"format": "jsonl", "path": "cases.jsonl"},
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
            },
        )
        return root

    source_root = make_root(tmp_path / "source")
    output_root = make_root(tmp_path / "output")
    source = Bundle(source_root, spec.name, db=tmp_path / "source.sqlite3")
    output = Bundle(output_root, spec.name, db=tmp_path / "target.sqlite3")
    source.synchronize()
    output.synchronize()
    output.write_certificate()
    output.ingest(include_incomplete=True, reason=rebuild_module.INCOMPLETE_REASON)
    source_assessments = [
        {
            "id": row["public"]["id"],
            "public_index": row["public"]["public_index"],
            "assessment": {
                key: row["public"]["assessment"].get(key)
                for key in (
                    "quality_level",
                    "evidence_level",
                    "archive_class",
                    "benchmark",
                )
            },
        }
        for row in rebuild_module._bundle_projection_rows(source_root, spec)
    ]
    monkeypatch.setattr(
        rebuild_module,
        "_source_assessment_rows",
        lambda root, source_spec: iter(source_assessments),
    )

    result = rebuild_module._compare_durable_cases(
        source_root,
        output_root,
        TargetStore(tmp_path / "target.sqlite3"),
        spec,
    )
    loaded = TargetStore(tmp_path / "target.sqlite3").get_case(
        spec.name, spec.release, "case-1"
    )

    assert result["count"] == 1
    assert result["public_target_reads_compared"] is True
    assert result["source_final_assessments_compared"] is True
    assert result["original_facts_compared_in_full"] is False
    assert {"migration_rows", "migration_schema"} <= set(
        result["omitted_migration_fields"]
    )
    assert "gate_results" in result["omitted_assessment_fields"]
    assert loaded["facts"]["metadata"] == {"supplier_classification": "gold"}
    assert loaded["facts"]["evidence"][0]["quote"] == "S = 3"
    assert loaded["structures"][0]["sha256"]
    with sqlite3.connect(tmp_path / "target.sqlite3") as connection:
        collections = {
            row[0]
            for row in connection.execute(
                "SELECT collection FROM scientific_asset_refs ORDER BY collection"
            )
        }
    assert collections == {"evidence", "structures"}


def test_schema_v2_source_assessment_reader_is_migration_local(
    tmp_path, rebuild_module
):
    spec = rebuild_module.DatasetSpec("fixture", "fixture.v001", "fixture")
    root = tmp_path / "fixture"
    generation = root / "assessments" / "generations" / "pending"
    generation.mkdir(parents=True)
    assessment_path = generation / "assessment.sqlite3"
    assessment = {
        "quality_level": "B2",
        "evidence_level": "E3",
        "archive_class": "benchmark",
        "benchmark": True,
        "reasons": ["verified source"],
    }
    with sqlite3.connect(assessment_path) as connection:
        connection.executescript(
            "PRAGMA user_version=1;"
            "CREATE TABLE case_assessments("
            "case_id TEXT,public_index INTEGER,quality_level TEXT,"
            "evidence_level TEXT,archive_class TEXT,benchmark INTEGER,"
            "assessment_json TEXT);"
        )
        connection.execute(
            "INSERT INTO case_assessments VALUES(?,?,?,?,?,?,?)",
            (
                "case-1",
                0,
                "B2",
                "E3",
                "benchmark",
                1,
                json.dumps(assessment, sort_keys=True, separators=(",", ":")),
            ),
        )
    payload = assessment_path.read_bytes()
    logical = {
        "schema": "hotpot-ds-assessment-generation/current",
        "source": {"revision": 1, "digest": "1" * 64},
        "policy": {"revision": 1, "digest": "2" * 64},
        "governance": {"revision": 1, "digest": "3" * 64},
        "runtime_policy": {"schema": "legacy-test"},
        "ordered_assessment_digest": "4" * 64,
        "summary_digest": "5" * 64,
        "files": [
            {
                "path": "assessment.sqlite3",
                "sha256": rebuild_module.hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        ],
    }
    generation_digest = rebuild_module._json_digest(logical)
    generation_id = "generation-" + generation_digest[:24]
    final_generation = generation.with_name(generation_id)
    generation.rename(final_generation)
    _write_json(
        final_generation / "generation.json",
        {
            **logical,
            "generation_id": generation_id,
            "generation_digest": generation_digest,
        },
    )
    (root / "cases").mkdir(parents=True)
    with sqlite3.connect(root / "cases" / "facts.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE control(singleton INTEGER PRIMARY KEY,"
            "current_generation_id TEXT,current_generation_digest TEXT)"
        )
        connection.execute(
            "INSERT INTO control VALUES(1,?,?)",
            (generation_id, generation_digest),
        )

    assert list(rebuild_module._source_assessment_rows(root, spec)) == [
        {
            "id": "case-1",
            "public_index": 0,
            "assessment": {
                "quality_level": "B2",
                "evidence_level": "E3",
                "archive_class": "benchmark",
                "benchmark": True,
            },
        }
    ]
    manifest_path = final_generation / "generation.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["summary_digest"] = "9" * 64
    _write_json(manifest_path, tampered)
    with pytest.raises(rebuild_module.RebuildError, match="logical digest"):
        list(rebuild_module._source_assessment_rows(root, spec))


def test_compact_receipt_binding_duplicates_must_match_columns(
    tmp_path, rebuild_module
):
    store, spec, projection = _compact_receipt_store(
        tmp_path / "receipt.sqlite3", rebuild_module
    )
    with sqlite3.connect(store.path) as connection:
        binding = json.loads(
            connection.execute("SELECT binding_json FROM ingestion_receipts").fetchone()[0]
        )
        binding["certificate_digest"] = "9" * 64
        connection.execute(
            "UPDATE ingestion_receipts SET binding_json=?",
            (json.dumps(binding, sort_keys=True, separators=(",", ":")),),
        )
    with pytest.raises(rebuild_module.RebuildError, match="certificate_digest"):
        rebuild_module._validate_target_receipt(store, store.path, spec, projection)


def test_cas_rejects_reproducible_control_categories(tmp_path, rebuild_module):
    database = tmp_path / "datasets.sqlite3"
    asset_root = Path(str(database) + ".assets")
    payload = b"reproducible facts database"
    digest = rebuild_module.hashlib.sha256(payload).hexdigest()
    relative = f"objects/sha256/{digest[:2]}/{digest}"
    path = asset_root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE target_assets(content_sha256 TEXT PRIMARY KEY,size INTEGER,relative_path TEXT);
            CREATE TABLE structure_asset_refs(content_sha256 TEXT);
            CREATE TABLE scientific_asset_refs(content_sha256 TEXT);
            CREATE TABLE release_control_asset_refs(content_sha256 TEXT,category TEXT,name TEXT);
            """
        )
        connection.execute(
            "INSERT INTO target_assets VALUES(?,?,?)", (digest, len(payload), relative)
        )
        connection.execute(
            "INSERT INTO release_control_asset_refs VALUES(?,?,?)",
            (digest, "facts", "facts.sqlite3"),
        )

    with pytest.raises(rebuild_module.RebuildError, match="reproducible"):
        rebuild_module._verify_cas(database)


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "two/parts", "UPPER"])
def test_operator_rejects_unsafe_generation_components(
    tmp_path, rebuild_module, unsafe
):
    with pytest.raises(rebuild_module.RebuildError):
        rebuild_module.Operator(tmp_path, unsafe, rebuild_module.SOURCE_GENERATION)
    with pytest.raises(rebuild_module.RebuildError):
        rebuild_module.Operator(tmp_path, rebuild_module.RUN_ID, unsafe)


@pytest.mark.parametrize(
    "managed_name",
    ("output_bundles", "output_generation", "state_path", "partial"),
)
def test_operator_rejects_dangling_symlinks_in_output_namespace(
    tmp_path, rebuild_module, managed_name
):
    repository = tmp_path / "repo"
    operator = rebuild_module.Operator(
        repository, rebuild_module.RUN_ID, rebuild_module.SOURCE_GENERATION
    )
    if managed_name == "partial":
        spec = rebuild_module.DATASETS[0]
        managed = operator._output_root(spec).with_name(spec.release + ".partial")
    else:
        managed = getattr(operator, managed_name)
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.symlink_to("missing-target")

    with pytest.raises(rebuild_module.RebuildError, match="symlink"):
        operator._assert_output_namespace()


def test_operator_rejects_symlinked_rebuild_lock(tmp_path, rebuild_module):
    operator = rebuild_module.Operator(
        tmp_path / "repo", rebuild_module.RUN_ID, rebuild_module.SOURCE_GENERATION
    )
    operator.store_root.mkdir(parents=True)
    operator.lock_path.symlink_to("missing-lock")

    with pytest.raises(rebuild_module.RebuildError, match="symlink"):
        with operator.locked():
            pass


def _minimal_operator_tree(tmp_path, rebuild_module):
    repository = tmp_path / "repo"
    operator = rebuild_module.Operator(
        repository, rebuild_module.RUN_ID, rebuild_module.SOURCE_GENERATION
    )
    operator.source_generation_root.mkdir(parents=True)
    operator.output_generation.mkdir(parents=True)
    for database in (operator.source_database, operator.output_database):
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE marker(value INTEGER)")
            connection.execute(
                "CREATE TABLE store_metadata("
                "singleton INTEGER PRIMARY KEY,store_uuid TEXT,schema_id TEXT)"
            )
            connection.execute(
                "INSERT INTO store_metadata VALUES(1,'fixture','hotpot-ds-store/current')"
            )
    for spec in rebuild_module.DATASETS:
        for root in (
            operator.source_bundles / spec.source_relative,
            operator._output_root(spec),
        ):
            root.mkdir(parents=True)
            (root / ".bundle.lock").touch()
    operator.store_root.mkdir(parents=True, exist_ok=True)
    os.symlink(
        os.path.relpath(operator.source_generation_root, operator.store_root),
        operator.store_root / "active",
    )
    return operator


def test_publication_barrier_excludes_source_candidate_and_bundle_writers(
    tmp_path, rebuild_module
):
    operator = _minimal_operator_tree(tmp_path, rebuild_module)

    with operator._publication_barrier():
        competing = sqlite3.connect(operator.output_database, timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                competing.execute("BEGIN IMMEDIATE")
        finally:
            competing.close()
        source_writer = sqlite3.connect(operator.source_database, timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                source_writer.execute("BEGIN IMMEDIATE")
        finally:
            source_writer.close()
        lock_path = operator._output_root(rebuild_module.DATASETS[0]) / ".bundle.lock"
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                rebuild_module.fcntl.flock(
                    descriptor,
                    rebuild_module.fcntl.LOCK_EX | rebuild_module.fcntl.LOCK_NB,
                )
        finally:
            os.close(descriptor)


def test_activation_and_rollback_are_atomic_and_recoverable(
    tmp_path, rebuild_module
):
    operator = _minimal_operator_tree(tmp_path, rebuild_module)
    state = operator._new_state()
    state["phases"]["ready"] = {
        "status": "complete",
        "result": {"activation_ready": True},
    }
    operator._save_state(state)
    operator._preactivation_recheck = lambda _state: {"fixture": "proved"}

    activated = operator.activate(operator.run_id)
    assert (operator.store_root / "active").resolve() == operator.output_generation
    assert activated["status"] == "activated"

    rolled_back = operator.rollback(activated["activation_id"])
    assert (operator.store_root / "active").resolve() == operator.source_generation_root
    assert rolled_back["status"] == "rolled-back"
    assert Path(rolled_back["rollback_record"]).is_file()


def test_activation_and_rollback_resume_after_swap_before_finalize(
    tmp_path, rebuild_module
):
    operator = _minimal_operator_tree(tmp_path, rebuild_module)
    state = operator._new_state()
    state["phases"]["ready"] = {
        "status": "complete",
        "result": {"activation_ready": True},
    }
    operator._save_state(state)
    proof_calls = []

    def prove(_state):
        proof_calls.append("proved")
        return {"fixture": "proved"}

    operator._preactivation_recheck = prove
    operator._flush_candidate_publication = lambda: None

    finalize_activation = operator._finalize_activation
    operator._finalize_activation = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("injected after activation swap")
    )
    with pytest.raises(RuntimeError, match="injected"):
        operator.activate(operator.run_id)
    assert (operator.store_root / "active").resolve() == operator.output_generation
    operator._finalize_activation = finalize_activation
    activated = operator.activate(operator.run_id)
    assert len(proof_calls) == 2
    assert activated["status"] == "activated"
    assert json.loads(operator.state_path.read_text())["activation"]["status"] == "activated"

    finalize_rollback = operator._finalize_rollback
    operator._finalize_rollback = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("injected after rollback swap")
    )
    with pytest.raises(RuntimeError, match="injected"):
        operator.rollback(activated["activation_id"])
    assert (operator.store_root / "active").resolve() == operator.source_generation_root
    operator._finalize_rollback = finalize_rollback
    rolled_back = operator.rollback(activated["activation_id"])
    assert len(proof_calls) == 4
    assert rolled_back["status"] == "rolled-back"
