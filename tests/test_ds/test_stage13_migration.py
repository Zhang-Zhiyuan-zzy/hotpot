import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from hotpot.ds.gates import BenchmarkGate, EvidenceGate, QualityGate
from hotpot.ds.gates._persistence import serialize_gate_configs
from hotpot.ds.public.bundle import SealedIntegrityError
from hotpot.ds.storage import TargetStore
from hotpot.ds.tools.migrate_legacy_store import (
    DatasetBootstrap,
    LegacyMigrationError,
    LegacyNormalizedStore,
    MigrationParityError,
    _REQUIRED_COLUMNS,
    _normalize_identifiers,
    activate_store_generation,
    build_parser,
    create_rollback_artifact,
    export_legacy_release,
    migrate_legacy_store,
    load_bootstraps,
    publish_cutover_pointer,
    verify_bundle_export,
    verify_rollback_artifact,
    verify_target_parity,
)


def _insert(connection, table, **values):
    columns = tuple(values)
    connection.execute(
        f"INSERT INTO {table}({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
        tuple(values[column] for column in columns),
    )


def _legacy_store(tmp_path: Path):
    database = tmp_path / "legacy.sqlite3"
    asset_root = tmp_path / "legacy-assets"
    coordinate = asset_root / "structures" / "objects" / "aa" / "sample.xyz"
    coordinate.parent.mkdir(parents=True)
    coordinate.write_text("3\nwater\nO 0 0 0\nH 0 0 1\nH 0 1 0\n", encoding="utf-8")
    digest = hashlib.sha256(coordinate.read_bytes()).hexdigest()
    structure_id = "a" * 64
    record_id = "legacy-release:record:sample-1"
    entity_id = f"entity:structure:{structure_id}"

    connection = sqlite3.connect(database)
    for table, columns in _REQUIRED_COLUMNS.items():
        connection.execute(
            f"CREATE TABLE {table}({','.join(columns)})"
        )
    _insert(connection, "schema_metadata", key="schema_version", value="1")
    _insert(
        connection,
        "schema_metadata",
        key="record_standard",
        value="hotpot-dataset/current",
    )
    _insert(
        connection,
        "datasets",
        dataset_id="legacy-demo",
        title="Legacy demo",
        description="Synthetic normalized legacy release",
    )
    policy = json.dumps(
        {
            "schema_version": "legacy",
            "generic": {},
            "datasets": {"legacy-demo": {"core_labels": ["oxidation_state"]}},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    _insert(
        connection,
        "dataset_policies",
        dataset_id="legacy-demo",
        revision=1,
        config_json=policy,
        config_sha256=hashlib.sha256(policy.encode()).hexdigest(),
        updated_at="2026-01-01 00:00:00",
    )
    _insert(
        connection,
        "dataset_releases",
        release_id="legacy-demo.v001",
        dataset_id="legacy-demo",
        version="001",
        release_serial=1,
        quality_level="B0_collection",
        core_labels_json='["oxidation_state"]',
        license="test-only",
        access_policy="local",
        provenance_json='{"supplier":"fixture"}',
        quality_json='{"core_labels":["oxidation_state"]}',
        ingestion_id=1,
    )
    _insert(
        connection,
        "subsets",
        subset_id="legacy-demo.v001:gold",
        release_id="legacy-demo.v001",
        parent_subset_id=None,
        name="supplier-gold",
        quality_level="B3_curated",
        selection_json="{}",
        quality_json="{}",
    )
    metadata = json.dumps(
        {
            "source_evidence_grade": "gold_supplier",
            "source_fields": {"sample_name": "sample-1"},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    _insert(
        connection,
        "records",
        record_id=record_id,
        release_id="legacy-demo.v001",
        public_index=0,
        local_id="sample-1",
        metadata_json=metadata,
    )
    _insert(
        connection,
        "subset_memberships",
        subset_id="legacy-demo.v001:gold",
        record_id=record_id,
    )
    _insert(
        connection,
        "entities",
        entity_id="entity:record:sample-1",
        entity_kind="chemical_record",
        parent_entity_id=None,
        metadata_json="{}",
    )
    _insert(
        connection,
        "entities",
        entity_id=entity_id,
        entity_kind="molecular_structure",
        parent_entity_id=None,
        metadata_json="{}",
    )
    _insert(
        connection,
        "external_identifiers",
        entity_id=entity_id,
        namespace="CSD_identifier",
        value="FIXTURE01",
        version=None,
    )
    _insert(
        connection,
        "record_entities",
        record_id=record_id,
        entity_id="entity:record:sample-1",
        role="record_subject",
    )
    _insert(
        connection,
        "record_entities",
        record_id=record_id,
        entity_id=entity_id,
        role="geometry",
    )
    _insert(
        connection,
        "assets",
        asset_id="asset:fixture",
        uri="structures/objects/aa/sample.xyz",
        media_type="chemical/x-xyz",
        content_sha256=digest,
        access_policy="local",
        metadata_json='{"format":"xyz"}',
    )
    _insert(
        connection,
        "structures",
        structure_id=structure_id,
        entity_id=entity_id,
        coordinate_hash=structure_id,
        atom_count=3,
        formal_charge=0,
        metadata_json="{}",
    )
    _insert(
        connection,
        "structure_assets",
        structure_id=structure_id,
        asset_id="asset:fixture",
        role="coordinates",
    )
    _insert(
        connection,
        "ingestion_assets",
        ingestion_id=1,
        asset_id="asset:fixture",
    )
    _insert(
        connection,
        "property_types",
        property_type_id="oxidation_state",
        name="Oxidation state",
        value_kind="integer",
        canonical_unit=None,
        schema_version="legacy",
        metadata_json="{}",
    )
    _insert(
        connection,
        "property_types",
        property_type_id="spin_category",
        name="Reported spin category",
        value_kind="category",
        canonical_unit=None,
        schema_version="legacy",
        metadata_json="{}",
    )
    _insert(
        connection,
        "observations",
        observation_id="observation:sample-1:oxidation",
        record_id=record_id,
        subject_entity_id=entity_id,
        property_type_id="oxidation_state",
        value_json="2",
        unit=None,
        uncertainty_json="{}",
        conditions_json='{"phase":"solid"}',
        status="active",
        evidence_level="E2_explicit_reported",
        method_json='{"method":"supplier-text"}',
        metadata_json="{}",
    )
    _insert(
        connection,
        "observation_subset_memberships",
        subset_id="legacy-demo.v001:gold",
        observation_id="observation:sample-1:oxidation",
    )
    _insert(
        connection,
        "sources",
        source_id="source:fixture",
        source_type="article",
        citation="Fixture article",
        doi="10.0000/fixture",
        license="test-only",
        access_policy="public",
        metadata_json="{}",
    )
    _insert(
        connection,
        "evidence_items",
        evidence_id="evidence:fixture",
        source_id="source:fixture",
        evidence_type="reported_text",
        locator="Table 1",
        quote="The metal is divalent.",
        matched_text="divalent",
        metadata_json="{}",
    )
    _insert(
        connection,
        "observation_evidence",
        observation_id="observation:sample-1:oxidation",
        evidence_id="evidence:fixture",
        role="supports",
    )
    _insert(
        connection,
        "claims",
        claim_id="claim:sample-1:reported-spin",
        record_id=record_id,
        subject_entity_id=entity_id,
        property_type_id="spin_category",
        text="The supplier text says high spin, but the claim is quarantined.",
        status="unresolved",
        evidence_level="E3_supplier_claim",
        metadata_json='{"quarantined":true,"conflict_status":"supplier_disputed"}',
    )
    _insert(
        connection,
        "claim_evidence",
        claim_id="claim:sample-1:reported-spin",
        evidence_id="evidence:fixture",
        role="reported_by",
    )
    connection.commit()
    connection.close()
    return database, asset_root, digest, structure_id, record_id


def _bootstrap():
    return DatasetBootstrap(
        ds_name="legacy-demo",
        core_labels=("oxidation_state",),
        core_info=(),
        identity_schema={"components": []},
        property_definitions={
            "oxidation_state": {"type": "integer", "unit": None},
            "spin_category": {
                "type": "category",
                "unit": None,
                "allowed_values": ["high spin", "low spin"],
            },
        },
        property_comparators={
            "oxidation_state": {"kind": "exact"},
            "spin_category": {"kind": "categorical"},
        },
        required_structure_roles=(),
        gates=(EvidenceGate(), QualityGate(), BenchmarkGate()),
        owner="fixture-owner",
        decision="Preserve source facts and reassess under current default Gates.",
        include_incomplete=True,
    )


def test_export_preserves_source_rows_ids_indices_and_asset_hash(tmp_path):
    database, asset_root, digest, old_structure_id, record_id = _legacy_store(tmp_path)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    root = tmp_path / "bundle"

    report = export_legacy_release(
        database,
        asset_root,
        "legacy-demo.v001",
        root,
        _bootstrap(),
    )
    parity = verify_bundle_export(database, "legacy-demo.v001", root)

    assert report["case_count"] == 1
    assert parity["passed"] is True
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    case = json.loads((root / "cases.jsonl").read_text().splitlines()[0])
    assert case["id"] == record_id
    assert case["public_index"] == 0
    assert case["identifiers"] == {"CSD": "FIXTURE01"}
    assert case["structures"][0]["structure_id"] != old_structure_id
    assert case["structures"][0]["legacy_structure_id"] == old_structure_id
    representation = case["structures"][0]["representations"][0]
    assert representation["expected_sha256"] == digest
    assert hashlib.sha256((root / representation["path"]).read_bytes()).hexdigest() == digest
    assert case["metadata"]["migration_rows"]["observations"][0]["value_json"] == "2"
    assert len(case["observations"]) == 1
    assert case["claims"][0]["claim_id"] == "claim:sample-1:reported-spin"
    assert case["claims"][0]["status"] == "unresolved"
    assert case["claims"][0]["claimed_evidence_level"] == "E3_supplier_claim"
    assert case["verifications"] == []
    assert case["claimed_benchmark"] is True
    assert case["attestations"][0]["authority"] == "Other"
    controls = {
        item["name"]: item for item in report["migration_controls"]
    }
    source_inventory = controls["legacy-source-inventory"]
    assert hashlib.sha256((root / source_inventory["path"]).read_bytes()).hexdigest() == source_inventory["sha256"]
    identity_mappings = [
        json.loads(line)
        for line in (root / "evidence" / "migration" / "identity-migration.jsonl")
        .read_text()
        .splitlines()
    ]
    assert any(
        item.get("namespace_rewrite")
        == {"legacy": "CSD_identifier", "current": "CSD"}
        for item in identity_mappings
    )


def test_export_parity_distinguishes_declared_missing_and_corrupt_assets(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    source = next(asset_root.rglob("sample.xyz"))
    source.unlink()
    missing_root = tmp_path / "missing-bundle"
    export_legacy_release(
        database,
        asset_root,
        "legacy-demo.v001",
        missing_root,
        _bootstrap(),
    )

    missing = verify_bundle_export(database, "legacy-demo.v001", missing_root)
    assert missing["passed"] is True
    assert missing["asset_parity"]["source_missing"] == 1
    missing_case = json.loads((missing_root / "cases.jsonl").read_text())
    assert missing_case["structures"][0]["representations"][0]["source_missing"] is True

    database, asset_root, _, _, _ = _legacy_store(tmp_path / "corrupt")
    corrupt_root = tmp_path / "corrupt-bundle"
    export_legacy_release(
        database,
        asset_root,
        "legacy-demo.v001",
        corrupt_root,
        _bootstrap(),
    )
    case = json.loads((corrupt_root / "cases.jsonl").read_text())
    copied = corrupt_root / case["structures"][0]["representations"][0]["path"]
    copied.unlink()
    with pytest.raises(MigrationParityError, match="copied asset integrity failure"):
        verify_bundle_export(database, "legacy-demo.v001", corrupt_root)


def test_export_parity_rejects_declared_asset_hash_mismatch(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("UPDATE assets SET content_sha256=?", ("f" * 64,))
    connection.commit()
    connection.close()
    root = tmp_path / "bundle"
    export_legacy_release(
        database,
        asset_root,
        "legacy-demo.v001",
        root,
        _bootstrap(),
    )

    with pytest.raises(MigrationParityError, match="declared asset hash mismatch"):
        verify_bundle_export(database, "legacy-demo.v001", root)


def test_export_parity_rejects_same_count_operative_fact_tampering(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    root = tmp_path / "bundle"
    export_legacy_release(database, asset_root, "legacy-demo.v001", root, _bootstrap())
    case = json.loads((root / "cases.jsonl").read_text())
    case["observations"][0]["value"] = 3
    (root / "cases.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")

    with pytest.raises(MigrationParityError, match="per-Case operative fact mismatch"):
        verify_bundle_export(database, "legacy-demo.v001", root)


def test_cross_case_relation_is_quarantined_and_endpoints_are_normalized(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    connection = sqlite3.connect(database)
    _insert(
        connection,
        "entity_relations",
        subject_entity_id="entity:record:sample-1",
        predicate="legacy-cross-case-link",
        object_entity_id="entity:outside-this-case",
        metadata_json='{"supplier":"unresolved"}',
    )
    _insert(
        connection,
        "entity_relations",
        subject_entity_id="entity:record:sample-1",
        predicate="has-geometry",
        object_entity_id="entity:structure:" + "a" * 64,
        metadata_json="{}",
    )
    connection.commit()
    connection.close()
    root = tmp_path / "bundle"
    export_legacy_release(database, asset_root, "legacy-demo.v001", root, _bootstrap())
    parity = verify_bundle_export(database, "legacy-demo.v001", root)
    case = json.loads((root / "cases.jsonl").read_text())

    assert parity["identity_indeterminate_rows"] == 1
    assert case["metadata"]["quarantined_entity_relations"][0]["predicate"] == "legacy-cross-case-link"
    operative = [
        relation
        for entity in case["entities"]
        for relation in entity["relations"]
    ]
    assert operative[0]["subject_entity_id"].startswith(case["id"] + ":entity:")
    assert operative[0]["object_entity_id"].startswith(case["id"] + ":entity:")


def test_legacy_reader_rejects_schema_drift(tmp_path):
    database, _, _, _, _ = _legacy_store(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE records ADD COLUMN obsolete_compatibility_field")
    connection.commit()
    connection.close()

    with pytest.raises(Exception, match="columns differ"):
        LegacyNormalizedStore(database).connect()


def test_rollback_and_atomic_cutover_pointer_require_passing_parity(tmp_path):
    database, asset_root, digest, _, _ = _legacy_store(tmp_path)
    rollback = tmp_path / "rollback"
    snapshot = create_rollback_artifact(database, asset_root, rollback)
    verification = verify_rollback_artifact(rollback)
    assert snapshot["asset_failures"] == 0
    assert verification["passed"] is True
    assert (rollback / "legacy-store.sqlite3").is_file()
    copied = next((rollback / "assets" / "objects").rglob(f"{digest}.*"))
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == digest

    target = tmp_path / "generation" / "new-store.sqlite3"
    target.parent.mkdir()
    TargetStore(target).initialize()
    pointer = tmp_path / "active-store.json"
    with pytest.raises(MigrationParityError):
        publish_cutover_pointer(
            pointer,
            target,
            rollback,
            [{"passed": False, "release": "legacy-demo.v001"}],
        )
    assert not pointer.exists()

    value = publish_cutover_pointer(
        pointer,
        target,
        rollback,
        [{"passed": True, "release": "legacy-demo.v001"}],
    )
    assert value["target_schema"] == "hotpot-ds-store/current"
    assert json.loads(pointer.read_text())["target_database"] == str(target.resolve())


def test_migration_source_is_the_immutable_rollback_snapshot(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    rollback_root = tmp_path / "rollback"
    create_rollback_artifact(database, asset_root, rollback_root)
    snapshot = verify_rollback_artifact(rollback_root)
    connection = sqlite3.connect(database)
    connection.execute("UPDATE observations SET value_json='99'")
    connection.commit()
    connection.close()

    root = tmp_path / "bundle"
    export_legacy_release(
        snapshot["snapshot_database"],
        snapshot["snapshot_asset_root"],
        "legacy-demo.v001",
        root,
        _bootstrap(),
    )
    parity = verify_bundle_export(
        snapshot["snapshot_database"],
        "legacy-demo.v001",
        root,
        snapshot["snapshot_asset_root"],
    )

    assert parity["stable_source_snapshot"] is True
    assert json.loads((root / "cases.jsonl").read_text())["observations"][0]["value"] == 2


def test_rollback_preserves_declared_source_missing_as_a_negative_fact(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    next(asset_root.rglob("sample.xyz")).unlink()
    rollback_root = tmp_path / "rollback"
    create_rollback_artifact(database, asset_root, rollback_root)

    verification = verify_rollback_artifact(rollback_root)

    assert verification["passed"] is True
    assert verification["missing_negative_facts"] == 1


def test_export_never_targets_nonempty_bundle_folder(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "keep.txt").write_text("user data")
    with pytest.raises(LegacyMigrationError, match="not empty"):
        export_legacy_release(
            database,
            asset_root,
            "legacy-demo.v001",
            root,
            _bootstrap(),
        )


def test_complete_migration_synchronizes_certifies_ingests_and_then_cuts_over(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    target = tmp_path / "generation" / "current.sqlite3"
    target.parent.mkdir()
    pointer = tmp_path / "active-store.json"
    active = tmp_path / "active"
    old_generation = tmp_path / "old-generation"
    old_generation.mkdir()
    active.symlink_to(old_generation, target_is_directory=True)

    result = migrate_legacy_store(
        database,
        asset_root,
        tmp_path / "work",
        target,
        tmp_path / "rollback",
        {"legacy-demo": _bootstrap()},
        cutover_pointer=pointer,
        active_store_link=active,
    )

    assert result["passed"] is True
    assert result["release_order"] == [
        {
            "position": 0,
            "dataset": "legacy-demo",
            "release": "legacy-demo.v001",
            "source_case_count": 1,
        }
    ]
    assert json.loads((tmp_path / "work" / "migration-plan.json").read_text())[
        "activation"
    ] == "only_after_all_releases_pass"
    assert result["releases"][0]["synchronization"]["is_synchronized"] is True
    assert Path(result["releases"][0]["certificate"]).is_file()
    assert result["releases"][0]["receipt"]["included_count"] == 1
    assert result["releases"][0]["target_parity"]["passed"] is True
    assert pointer.is_file()
    assert active.is_symlink()
    assert result["activation"]["previous_generation"] == str(old_generation.resolve())
    assert TargetStore(active / target.name).metadata()["schema_id"] == "hotpot-ds-store/current"
    assert TargetStore(target).load_release_metadata(
        "legacy-demo", "legacy-demo.v001"
    )["counts"] == {"ordinary": 1, "benchmark": 0, "incomplete": 0}
    stored = TargetStore(target).get_case(
        "legacy-demo", "legacy-demo.v001", "legacy-release:record:sample-1"
    )
    assert len(stored["facts"]["observations"]) == 1
    assert len(stored["facts"]["claims"]) == 1
    assert stored["assessment"]["benchmark"] is False
    connection = sqlite3.connect(target)
    categories = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT category FROM release_control_asset_refs"
        )
    }
    connection.close()
    assert {
        "bundle", "certificate", "facts", "generation", "governance", "migration"
    } <= categories


def test_owner_bootstrap_file_round_trips_current_gate_and_property_policy(tmp_path):
    bootstrap = _bootstrap()
    path = tmp_path / "bootstrap.json"
    path.write_text(
        json.dumps(
            {
                "schema": "hotpot-ds-migration-bootstrap/current",
                "datasets": {
                    bootstrap.ds_name: {
                        "core_labels": list(bootstrap.core_labels),
                        "core_info": list(bootstrap.core_info),
                        "identity_schema": bootstrap.identity_schema,
                        "property_definitions": bootstrap.property_definitions,
                        "property_comparators": bootstrap.property_comparators,
                        "required_structure_roles": list(bootstrap.required_structure_roles),
                        "gates": serialize_gate_configs(bootstrap.gates),
                        "owner": bootstrap.owner,
                        "decision": bootstrap.decision,
                        "include_incomplete": bootstrap.include_incomplete,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_bootstraps(path)["legacy-demo"]
    assert loaded.property_definitions == bootstrap.property_definitions
    assert tuple(gate.digest for gate in loaded.gates) == tuple(
        gate.digest for gate in bootstrap.gates
    )
    assert build_parser().parse_args(
        ["legacy.sqlite3", "assets", "work", "target.sqlite3", "rollback", "--bootstrap", str(path)]
    ).bootstrap == path


def test_target_parity_accounts_for_certificate_bound_excluded_incomplete_cases(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    bootstrap = replace(
        _bootstrap(),
        core_info=("required_but_missing",),
        include_incomplete=False,
    )
    target = tmp_path / "generation" / "current.sqlite3"
    target.parent.mkdir()

    result = migrate_legacy_store(
        database,
        asset_root,
        tmp_path / "work",
        target,
        tmp_path / "rollback",
        {"legacy-demo": bootstrap},
    )

    parity = result["releases"][0]["target_parity"]
    assert parity["passed"] is True
    assert parity["source_case_count"] == 1
    assert parity["target_case_count"] == 0
    assert parity["certified_included_case_count"] == 0
    assert parity["certified_excluded_case_count"] == 1
    connection = sqlite3.connect(target)
    controls = connection.execute(
        "SELECT category,name FROM release_control_asset_refs "
        "WHERE dataset_id='legacy-demo' AND release_id='legacy-demo.v001' "
        "AND category='migration' ORDER BY name"
    ).fetchall()
    connection.close()
    assert [name for _, name in controls] == [
        "identity-migration",
        "legacy-identity-relationships",
        "legacy-source-inventory",
    ]


def test_target_parity_rejects_same_count_stored_fact_tampering(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    target = tmp_path / "generation" / "current.sqlite3"
    target.parent.mkdir()
    result = migrate_legacy_store(
        database,
        asset_root,
        tmp_path / "work",
        target,
        tmp_path / "rollback",
        {"legacy-demo": _bootstrap()},
    )
    migrated = result["releases"][0]
    connection = sqlite3.connect(target)
    payload = json.loads(
        connection.execute("SELECT source_payload_json FROM cases").fetchone()[0]
    )
    payload["observations"][0]["value"] = 99
    connection.execute(
        "UPDATE cases SET source_payload_json=?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
    )
    connection.commit()
    connection.close()

    with pytest.raises(MigrationParityError, match="does not exactly bind"):
        verify_target_parity(
            target,
            migrated["bundle_root"],
            migrated["source_parity"],
        )


def test_target_parity_rejects_well_formed_placeholder_receipt_digest(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    target = tmp_path / "generation" / "current.sqlite3"
    target.parent.mkdir()
    result = migrate_legacy_store(
        database,
        asset_root,
        tmp_path / "work",
        target,
        tmp_path / "rollback",
        {"legacy-demo": _bootstrap()},
    )
    migrated = result["releases"][0]
    connection = sqlite3.connect(target)
    connection.execute(
        "UPDATE ingestion_receipts SET included_cases_digest=?",
        ("a" * 64,),
    )
    connection.commit()
    connection.close()

    with pytest.raises(SealedIntegrityError, match="receipt|certified content"):
        verify_target_parity(
            target,
            migrated["bundle_root"],
            migrated["source_parity"],
        )


def test_activation_failure_preserves_existing_active_generation(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    target = tmp_path / "generation" / "current.sqlite3"
    target.parent.mkdir()
    result = migrate_legacy_store(
        database,
        asset_root,
        tmp_path / "work",
        target,
        tmp_path / "rollback",
        {"legacy-demo": _bootstrap()},
    )
    old_generation = tmp_path / "old-generation"
    old_generation.mkdir()
    active = tmp_path / "active"
    active.symlink_to(old_generation, target_is_directory=True)
    connection = sqlite3.connect(target)
    relative = connection.execute(
        "SELECT relative_path FROM target_assets ORDER BY content_sha256 LIMIT 1"
    ).fetchone()[0]
    connection.close()
    (TargetStore(target).asset_root / relative).unlink()

    with pytest.raises(MigrationParityError, match="target assets"):
        activate_store_generation(
            active,
            target,
            tmp_path / "rollback",
            [result["releases"][0]["target_parity"]],
        )

    assert active.resolve() == old_generation.resolve()


def test_activation_rejects_a_generation_beneath_the_active_path(tmp_path):
    database, asset_root, _, _, _ = _legacy_store(tmp_path)
    active = tmp_path / "active"
    target = active / "generation" / "current.sqlite3"
    target.parent.mkdir(parents=True)
    result = migrate_legacy_store(
        database,
        asset_root,
        tmp_path / "work",
        target,
        tmp_path / "rollback",
        {"legacy-demo": _bootstrap()},
    )

    with pytest.raises(LegacyMigrationError, match="active store path|outside"):
        activate_store_generation(
            active,
            target,
            tmp_path / "rollback",
            [result["releases"][0]["target_parity"]],
        )

    assert not active.is_symlink()


def test_shared_legacy_csd_identifier_is_preserved_without_merging_cases():
    collision = {("CSD_identifier", "DOQRAC")}
    first = {
        "record_entities": [{"entity_id": "entity:first"}],
        "external_identifiers": [
            {
                "entity_id": "entity:first",
                "namespace": "CSD_identifier",
                "value": "DOQRAC",
            }
        ],
    }
    second = {
        "record_entities": [{"entity_id": "entity:second"}],
        "external_identifiers": [
            {
                "entity_id": "entity:second",
                "namespace": "CSD_identifier",
                "value": "DOQRAC",
            }
        ],
    }

    assert _normalize_identifiers(first, collision) == {"CSD": "DOQRAC"}
    assert _normalize_identifiers(second, collision) == {"CSD": "DOQRAC"}
