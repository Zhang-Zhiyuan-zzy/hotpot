"""Focused tests for the one-time current-runtime rebuild operator."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest


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
    connection.executescript(
        """
        CREATE TABLE store_metadata(singleton INTEGER PRIMARY KEY, store_uuid TEXT, schema_id TEXT);
        CREATE TABLE dataset_policies(
          dataset_id TEXT PRIMARY KEY, revision INTEGER, policy_json TEXT, policy_digest TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO store_metadata VALUES(1,'fixture','hotpot-ds-store/current')"
    )
    policy = {
        "schema": "hotpot-ds-policy/current",
        "gates": [],
        "runtime_policy": {"digest": "old"},
    }
    for spec in rebuild_module.DATASETS:
        connection.execute(
            "INSERT INTO dataset_policies VALUES(?,?,?,?)",
            (
                spec.name,
                1,
                json.dumps(policy, sort_keys=True, separators=(",", ":")),
                rebuild_module._json_digest(policy),
            ),
        )
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
            CREATE TABLE control(id TEXT);
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value_json TEXT);
            CREATE TABLE policy(id TEXT);
            """
        )
        facts.execute(
            "INSERT INTO metadata VALUES('manifest',?)",
            (json.dumps(manifest, sort_keys=True, separators=(",", ":")),),
        )
        facts.commit()
        facts.close()
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
    os.symlink("generations/stage13-current", store / "active")

    operator = rebuild_module.Operator(
        repository, rebuild_module.RUN_ID, rebuild_module.SOURCE_GENERATION
    )
    plan = operator.plan()

    assert plan["mode"] == "read-only-plan"
    assert Path(plan["candidate_database"]) == operator.output_database
    assert all(".partial" not in item["candidate"] for item in plan["datasets"])
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
                    "source_database": {"sha256": "c" * 64},
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


def _receipt_projection_store(path, rebuild_module):
    spec = rebuild_module.DatasetSpec("fixture", "fixture.v001", "fixture")
    facts = {
        "id": "case-1",
        "local_id": "local-1",
        "public_index": 0,
        "identifiers": {"CSD": "AAAAAA"},
    }
    assessment = {
        "quality_level": "B1",
        "evidence_level": "E1",
        "archive_class": "ordinary",
        "benchmark": False,
    }
    identifiers = [{"namespace": "CSD", "value": "AAAAAA"}]
    content = {
        "id": "case-1",
        "public_index": 0,
        "local_id": "local-1",
        "facts": facts,
        "assessment": assessment,
        "identifiers": identifiers,
    }
    semantics = {
        "case_content_digest": rebuild_module._stream_json_array_digest([content]),
        "included_cases_digest": rebuild_module._stream_json_array_digest(["case-1"]),
        "excluded_cases_digest": rebuild_module._stream_json_array_digest([]),
        "include_incomplete": 1,
        "reason": rebuild_module.INCOMPLETE_REASON,
        "included_count": 1,
        "excluded_count": 0,
    }
    binding = {
        "dataset": spec.name,
        "release": spec.release,
        "store_uuid": "store-fixture",
        "bundle_digest": "1" * 64,
        "certificate_digest": "2" * 64,
        "policy_digest": "3" * 64,
        "generation_digest": "4" * 64,
        "case_content_digest": semantics["case_content_digest"],
        "included_cases_digest": semantics["included_cases_digest"],
        "excluded_cases_digest": semantics["excluded_cases_digest"],
        "asset_inventory_digest": "5" * 64,
        "include_incomplete": True,
        "reason": rebuild_module.INCOMPLETE_REASON,
    }
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE cases(
              dataset_id TEXT,release_id TEXT,case_id TEXT,public_index INTEGER,
              local_id TEXT,source_payload_json TEXT
            );
            CREATE TABLE case_assessments(
              dataset_id TEXT,release_id TEXT,case_id TEXT,assessment_json TEXT
            );
            CREATE TABLE identifiers(
              dataset_id TEXT,release_id TEXT,case_id TEXT,namespace TEXT,value TEXT
            );
            CREATE TABLE ingestion_receipts(
              receipt_id TEXT,store_uuid TEXT,dataset_id TEXT,release_id TEXT,
              bundle_digest TEXT,certificate_digest TEXT,policy_digest TEXT,
              generation_digest TEXT,case_content_digest TEXT,
              included_cases_digest TEXT,excluded_cases_digest TEXT,
              asset_inventory_digest TEXT,include_incomplete INTEGER,reason TEXT,
              included_count INTEGER,excluded_count INTEGER,binding_json TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO cases VALUES(?,?,?,?,?,?)",
            (
                spec.name,
                spec.release,
                "case-1",
                0,
                "local-1",
                json.dumps(facts, sort_keys=True, separators=(",", ":")),
            ),
        )
        connection.execute(
            "INSERT INTO case_assessments VALUES(?,?,?,?)",
            (
                spec.name,
                spec.release,
                "case-1",
                json.dumps(assessment, sort_keys=True, separators=(",", ":")),
            ),
        )
        connection.execute(
            "INSERT INTO identifiers VALUES(?,?,?,?,?)",
            (spec.name, spec.release, "case-1", "CSD", "AAAAAA"),
        )
        connection.execute(
            "INSERT INTO ingestion_receipts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "receipt-" + rebuild_module._json_digest(binding),
                binding["store_uuid"],
                spec.name,
                spec.release,
                binding["bundle_digest"],
                binding["certificate_digest"],
                binding["policy_digest"],
                binding["generation_digest"],
                semantics["case_content_digest"],
                semantics["included_cases_digest"],
                semantics["excluded_cases_digest"],
                binding["asset_inventory_digest"],
                semantics["include_incomplete"],
                semantics["reason"],
                semantics["included_count"],
                semantics["excluded_count"],
                json.dumps(binding, sort_keys=True, separators=(",", ":")),
            ),
        )
    return spec, semantics


def test_receipt_semantics_are_recomputed_from_case_rows(tmp_path, rebuild_module):
    database = tmp_path / "receipt.sqlite3"
    spec, expected = _receipt_projection_store(database, rebuild_module)

    with rebuild_module._readonly_connection(database) as connection:
        assert rebuild_module._receipt_semantics(connection, spec) == expected

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE ingestion_receipts SET case_content_digest=?",
            ("0" * 64,),
        )
    with rebuild_module._readonly_connection(database) as connection:
        with pytest.raises(rebuild_module.RebuildError, match="stored Case content"):
            rebuild_module._receipt_semantics(connection, spec)


def test_receipt_binding_duplicates_must_match_columns(tmp_path, rebuild_module):
    database = tmp_path / "receipt.sqlite3"
    spec, _expected = _receipt_projection_store(database, rebuild_module)

    with rebuild_module._readonly_connection(database) as connection:
        rebuild_module._validate_receipt(connection, spec)

    with sqlite3.connect(database) as connection:
        binding = json.loads(
            connection.execute(
                "SELECT binding_json FROM ingestion_receipts"
            ).fetchone()[0]
        )
        binding["certificate_digest"] = "9" * 64
        connection.execute(
            "UPDATE ingestion_receipts SET binding_json=?",
            (json.dumps(binding, sort_keys=True, separators=(",", ":")),),
        )
    with rebuild_module._readonly_connection(database) as connection:
        with pytest.raises(rebuild_module.RebuildError, match="certificate_digest"):
            rebuild_module._validate_receipt(connection, spec)


def test_runtime_control_transition_requires_independent_artifacts(
    tmp_path, rebuild_module
):
    spec = rebuild_module.DatasetSpec("fixture", "fixture.v001", "fixture")
    databases = [tmp_path / "source.sqlite3", tmp_path / "target.sqlite3"]
    identities = (
        ("facts", "facts.sqlite3"),
        ("generation", "assessment.sqlite3"),
        ("generation", "generation.json"),
        ("certificate", "certificate.json"),
    )
    for index, database in enumerate(databases):
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE release_control_asset_refs("
                "dataset_id TEXT,release_id TEXT,category TEXT,name TEXT,"
                "content_sha256 TEXT)"
            )
            connection.executemany(
                "INSERT INTO release_control_asset_refs VALUES(?,?,?,?,?)",
                (
                    (spec.name, spec.release, category, name, str(index + 1) * 64)
                    for category, name in identities
                ),
            )
    with (
        rebuild_module._readonly_connection(databases[0]) as source,
        rebuild_module._readonly_connection(databases[1]) as target,
    ):
        transition = rebuild_module._assert_runtime_controls_regenerated(
            source, target, spec
        )
        assert set(transition) == {
            f"{category}/{name}" for category, name in identities
        }

    with sqlite3.connect(databases[1]) as connection:
        connection.execute(
            "UPDATE release_control_asset_refs SET content_sha256=? "
            "WHERE category='facts'",
            ("1" * 64,),
        )
    with (
        rebuild_module._readonly_connection(databases[0]) as source,
        rebuild_module._readonly_connection(databases[1]) as target,
    ):
        with pytest.raises(rebuild_module.RebuildError, match="reused"):
            rebuild_module._assert_runtime_controls_regenerated(source, target, spec)


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


def test_publication_barrier_excludes_bundle_and_database_writers(
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
            source_writer.execute("BEGIN IMMEDIATE")
            source_writer.execute("INSERT INTO marker VALUES(1)")
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                source_writer.commit()
            source_writer.rollback()
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
