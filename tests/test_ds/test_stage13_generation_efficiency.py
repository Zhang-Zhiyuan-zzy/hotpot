from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from hotpot.ds.gates._persistence import default_gates, serialize_gate_configs
from hotpot.ds.governance import default_registry_snapshot, verify_registry
from hotpot.ds.public import _generation as generation_module
from hotpot.ds.public._assessment import canonical_json, summarize_assessments
from hotpot.ds.public._bundle_state import (
    BUNDLE_SCHEMA,
    BundleState,
    _SynchronizationSnapshot,
)
from hotpot.ds.public._generation import (
    AssessmentGeneration,
    AssetSourceChangedError,
    _DIGEST_QUERIES,
    _assess_cases,
    _assess_conflicts,
    _assessment_policy,
    _initialize_assessment_database,
    _initialize_work_database,
    _stream_inputs,
)
from hotpot.ds.public._implementation import runtime_policy
from hotpot.ds.public.bundle import Bundle


def _case(index: int) -> dict:
    return {
        "id": f"case-{index:04d}",
        "local_id": f"local-{index:04d}",
        "public_index": index,
        "core_info": {"sample": f"sample-{index:04d}"},
        "observations": [
            {
                "observation_id": f"observation-{index:04d}",
                "property": "spin",
                "value": index % 3,
            }
        ],
    }


def _state(tmp_path: Path, size: int) -> BundleState:
    root = tmp_path / f"bundle-{size}"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    intake.write_text(
        "".join(json.dumps(_case(index)) + "\n" for index in range(size)),
        encoding="utf-8",
    )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "generation-efficiency",
        "release_id": f"generation-efficiency-r{size}",
        "version": "1.0",
        "title": "Generation efficiency",
        "description": "Bounded-query generation fixture",
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
        "required_structure_roles": [],
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return BundleState.build(
        root,
        "generation-efficiency",
        {
            "schema": "hotpot-ds-policy/current",
            "gates": serialize_gate_configs(default_gates()),
            "runtime_policy": runtime_policy(),
        },
        {"mode": "new", "store_id": None},
    )


def _shared_asset_state(tmp_path: Path, size: int) -> tuple[BundleState, str]:
    root = tmp_path / f"shared-asset-bundle-{size}"
    asset = root / "structures" / "shared.smi"
    asset.parent.mkdir(parents=True)
    asset.write_text("O\n", encoding="utf-8")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    cases = []
    for index in range(size):
        expected = (
            digest
            if index % 3 == 0
            else "0" * 64
            if index % 3 == 1
            else None
        )
        structure = {
            "structure_id": f"structure-{index:04d}",
            "role": "primary" if index % 2 == 0 else "secondary",
            "path": "structures/shared.smi",
            "format": "smi",
        }
        if expected is not None:
            structure["expected_sha256"] = expected
        case = _case(index)
        case["structures"] = [structure]
        cases.append(case)
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    intake.write_text(
        "".join(json.dumps(case) + "\n" for case in cases),
        encoding="utf-8",
    )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "shared-asset-efficiency",
        "release_id": f"shared-asset-efficiency-r{size}",
        "version": "1.0",
        "title": "Shared asset efficiency",
        "description": "One scientific structure referenced by many Cases",
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
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return (
        BundleState.build(
            root,
            "shared-asset-efficiency",
            {
                "schema": "hotpot-ds-policy/current",
                "gates": serialize_gate_configs(default_gates()),
                "runtime_policy": runtime_policy(),
            },
            {"mode": "new", "store_id": None},
        ),
        digest,
    )


class _SelectCounter:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.selects: list[str] = []

    def execute(self, statement: str, parameters=()):
        if statement.lstrip().upper().startswith("SELECT"):
            self.selects.append(" ".join(statement.split()))
        return self.connection.execute(statement, parameters)

    def executemany(self, statement: str, parameters):
        return self.connection.executemany(statement, parameters)


@pytest.mark.parametrize("size", [3, 24])
def test_case_support_queries_are_bounded_independent_of_case_count(
    tmp_path: Path, size: int
) -> None:
    state = _state(tmp_path, size)
    assessment = sqlite3.connect(":memory:")
    work = sqlite3.connect(":memory:")
    _initialize_assessment_database(assessment)
    _initialize_work_database(work)
    policy = state.policy
    manifest = state.manifest
    with assessment, work:
        _stream_inputs(
            state.root,
            state,
            assessment,
            work,
            policy["identity_schema"],
            policy["required_structure_roles"],
            tmp_path,
            policy["runtime_policy"],
        )
        _assess_conflicts(
            assessment,
            work,
            {},
            "identity-schema",
            policy["property_definitions"],
            policy["property_comparators"],
        )
        (
            evidence_policy,
            quality_policy,
            benchmark_policy,
            _,
            property_definitions,
            property_comparators,
        ) = _assessment_policy(policy)
        counted_assessment = _SelectCounter(assessment)
        counted_work = _SelectCounter(work)
        _assess_cases(
            state,
            counted_assessment,
            counted_work,
            manifest,
            evidence_policy,
            quality_policy,
            benchmark_policy,
            property_definitions,
            property_comparators,
            verify_registry(default_registry_snapshot()),
            None,
        )

    assert len(counted_work.selects) == 1
    assert len(counted_assessment.selects) == 2
    assert assessment.execute("SELECT COUNT(*) FROM case_assessments").fetchone()[0] == size


def test_generation_build_uses_two_ordered_case_passes_and_validates_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, 12)
    iter_calls = 0
    validation_calls = 0
    original_iter = _SynchronizationSnapshot.iter_cases
    original_open = AssessmentGeneration._open_without_lease.__func__

    def counted_iter(self):
        nonlocal iter_calls
        iter_calls += 1
        yield from original_iter(self)

    def counted_open(cls, *args, **kwargs):
        nonlocal validation_calls
        validation_calls += 1
        return original_open(cls, *args, **kwargs)

    monkeypatch.setattr(_SynchronizationSnapshot, "iter_cases", counted_iter)
    monkeypatch.setattr(
        AssessmentGeneration,
        "_open_without_lease",
        classmethod(counted_open),
    )

    with state.synchronization_snapshot() as input_snapshot:
        generation = AssessmentGeneration.build(
            state.root,
            state,
            state.policy,
            default_registry_snapshot(),
            input_snapshot=input_snapshot,
        )

    assert generation.summary["cases"] == 12
    assert iter_calls == 2
    assert validation_calls == 1


@pytest.mark.parametrize("size", [3, 100])
def test_shared_structure_is_snapshotted_hashed_and_parsed_once_per_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: int,
) -> None:
    state, digest = _shared_asset_state(tmp_path, size)
    snapshot_and_hash_calls = 0
    parse_calls = 0
    parser_binding_calls = 0
    original_snapshot = generation_module._snapshot_asset
    original_cache_binding = generation_module._AssetAssessmentCache.parser_binding

    def counted_snapshot(*args, **kwargs):
        nonlocal snapshot_and_hash_calls
        snapshot_and_hash_calls += 1
        return original_snapshot(*args, **kwargs)

    def counted_parse(path: Path, fmt: str) -> bool:
        nonlocal parse_calls
        parse_calls += 1
        assert path.read_bytes() == b"O\n"
        assert fmt == "smi"
        return True

    def counted_parser_binding(cache, fmt):
        nonlocal parser_binding_calls
        if fmt not in cache._parser_bindings:
            parser_binding_calls += 1
        return original_cache_binding(cache, fmt)

    monkeypatch.setattr(generation_module, "_snapshot_asset", counted_snapshot)
    monkeypatch.setattr(generation_module, "_structure_is_parseable", counted_parse)
    monkeypatch.setattr(
        generation_module._AssetAssessmentCache,
        "parser_binding",
        counted_parser_binding,
    )

    with state.synchronization_snapshot() as input_snapshot:
        generation = AssessmentGeneration.build(
            state.root,
            state,
            state.policy,
            default_registry_snapshot(),
            input_snapshot=input_snapshot,
        )

    database = generation.manifest_path.parent / "assessment.sqlite3"
    with sqlite3.connect(database) as connection:
        assets = {
            value["case_id"]: value
            for (payload,) in connection.execute(
                "SELECT assessment_json FROM asset_assessments"
            )
            for value in [json.loads(payload)]
        }

    assert snapshot_and_hash_calls == 1
    assert parse_calls == 1
    assert parser_binding_calls == 1
    assert len(assets) == size
    assert len({asset["asset_id"] for asset in assets.values()}) == size
    for index in range(size):
        asset = assets[f"case-{index:04d}"]
        expected_hash = (
            "unchecked"
            if index % 3 == 2
            else "match"
            if index % 3 == 0
            else "mismatch"
        )
        assert asset["structure_id"] == f"structure-{index:04d}"
        assert asset["role"] == ("primary" if index % 2 == 0 else "secondary")
        assert asset["observed_sha256"] == digest
        assert asset["hash_result"] == expected_hash
        assert asset["parse_result"] == "valid"
        assert asset["effective_required"] is (index % 2 == 0)
        assert ("asset_hash_mismatch" in asset["reasons"]) is (index % 3 == 1)
        assert asset["parser_binding_digest"] is not None


def test_equal_bytes_at_distinct_paths_are_independently_snapshotted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "distinct-paths"
    root.mkdir()
    for name in ("first.smi", "second.smi"):
        (root / name).write_text("O\n", encoding="utf-8")
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    calls = 0
    original_snapshot = generation_module._snapshot_asset

    def counted_snapshot(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(generation_module, "_snapshot_asset", counted_snapshot)
    monkeypatch.setattr(generation_module, "_structure_is_parseable", lambda *_: True)
    policy = runtime_policy()
    with generation_module._AssetAssessmentCache(snapshots, policy) as cache:
        for index, name in enumerate(("first.smi", "second.smi")):
            generation_module._assess_asset(
                root,
                {
                    "asset_id": f"asset-{index}",
                    "case_id": f"case-{index}",
                    "collection": "structures",
                    "record_id": f"structure-{index}",
                    "structure_id": f"structure-{index}",
                    "role": "primary",
                    "path": name,
                    "format": "smi",
                    "expected_sha256": None,
                },
                ["primary"],
                snapshots,
                policy,
                cache,
            )

    assert calls == 2
    assert list(snapshots.iterdir()) == []


@pytest.mark.parametrize("preserve_stat_signature", [False, True])
def test_shared_asset_cache_rejects_source_change_before_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preserve_stat_signature: bool,
) -> None:
    source = tmp_path / "shared.smi"
    source.write_text("O\n", encoding="utf-8")
    if preserve_stat_signature:
        source_stat = source.stat()
        source_identity = source_stat.st_dev, source_stat.st_ino
        stat_signature = generation_module._stat_signature
        original_signature = stat_signature(source_stat)

        def colliding_stat_signature(value):
            if (value.st_dev, value.st_ino) == source_identity:
                return original_signature
            return stat_signature(value)

        monkeypatch.setattr(generation_module, "_stat_signature", colliding_stat_signature)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    policy = runtime_policy()
    declaration = {
        "asset_id": "asset-1",
        "case_id": "case-1",
        "collection": "structures",
        "record_id": "structure-1",
        "structure_id": "structure-1",
        "role": "primary",
        "path": "shared.smi",
        "format": "smi",
        "expected_sha256": None,
    }
    monkeypatch.setattr(generation_module, "_structure_is_parseable", lambda *_: True)

    with pytest.raises(AssetSourceChangedError, match="changed"):
        with generation_module._AssetAssessmentCache(snapshots, policy) as cache:
            generation_module._assess_asset(
                tmp_path,
                declaration,
                ["primary"],
                snapshots,
                policy,
                cache,
            )
            source.write_text("N\n", encoding="utf-8")
            generation_module._assess_asset(
                tmp_path,
                declaration,
                ["primary"],
                snapshots,
                policy,
                cache,
            )

    assert list(snapshots.iterdir()) == []


def test_bundle_synchronize_performs_exactly_prepare_and_publish_source_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, 10)
    bundle = Bundle._from_state(state, tmp_path / "target.sqlite3")
    proof_calls = 0
    original = BundleState._compute_source_digest

    def counted_proof(self, connection, manifest, **kwargs):
        nonlocal proof_calls
        proof_calls += 1
        return original(self, connection, manifest, **kwargs)

    monkeypatch.setattr(BundleState, "_compute_source_digest", counted_proof)

    result = bundle.synchronize()

    assert result["is_synchronized"] is True
    assert proof_calls == 2


def _reference_ordered_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for table, query in _DIGEST_QUERIES:
        encoded_table = table.encode("utf-8")
        digest.update(len(encoded_table).to_bytes(8, "big"))
        digest.update(encoded_table)
        for row in connection.execute(query):
            encoded = canonical_json([table, *row]).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def test_runtime_protocol_matches_generation_schema_and_digest_order() -> None:
    protocol = runtime_policy()["assessment"]["store_protocol"]

    assert (
        protocol["sqlite_user_version"]
        == generation_module._DATABASE_SCHEMA_VERSION
    )
    assert protocol["ordered_digest_tables"] == [
        table for table, _query in _DIGEST_QUERIES
    ]


def test_generation_products_match_reference_with_two_case_scans_and_no_vacuum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path, 18)
    statements: list[str] = []
    original_connect = generation_module.sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(generation_module.sqlite3, "connect", traced_connect)
    with state.synchronization_snapshot() as input_snapshot:
        generation = AssessmentGeneration.build(
            state.root,
            state,
            state.policy,
            default_registry_snapshot(),
            input_snapshot=input_snapshot,
        )

    case_scans = [
        statement
        for statement in statements
        if "FROM case_assessments ORDER BY public_index,case_id" in statement
    ]
    assert len(case_scans) == 2
    assert not any(
        statement.lstrip().upper().startswith("VACUUM")
        for statement in statements
    )

    database = generation.manifest_path.parent / "assessment.sqlite3"
    with original_connect(database) as connection:
        assert _reference_ordered_digest(connection) == generation.manifest[
            "ordered_assessment_digest"
        ]
        reference_summary = summarize_assessments(
            json.loads(row[0])
            for row in connection.execute(
                "SELECT assessment_json FROM case_assessments "
                "ORDER BY public_index,case_id"
            )
        )
        for archive_class in ("incomplete", "ordinary", "benchmark"):
            expected = [
                row[0]
                for row in connection.execute(
                    "SELECT case_id FROM case_assessments "
                    "WHERE archive_class=? ORDER BY public_index,case_id",
                    (archive_class,),
                )
            ]
            observed = (
                generation.manifest_path.parent / f"{archive_class}.ids"
            ).read_text(encoding="utf-8").splitlines()
            assert observed == expected
    assert generation.summary == reference_summary
