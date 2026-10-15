from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from hotpot.ds.gates._persistence import default_gates, serialize_gate_configs
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA, BundleState
from hotpot.ds.public import _generation as generation_module
from hotpot.ds.public._generation import (
    AssessmentGeneration,
    AssetSourceChangedError,
    GenerationIntegrityError,
    _assess_conflicts,
    _initialize_assessment_database,
    _initialize_work_database,
)
from hotpot.ds.public._assessment import (
    canonical_json,
    compare_observations,
)
from hotpot.ds.public._implementation import runtime_policy


def _policy() -> dict:
    return {
        "schema": "hotpot-ds-policy/current",
        "gates": serialize_gate_configs(default_gates()),
        "runtime_policy": runtime_policy(),
    }


def _structure(case_id: str, path: str, **values) -> dict:
    return {
        "structure_id": f"structure-{case_id}-{len(path)}",
        "role": values.pop("role", "primary"),
        "path": path,
        "format": values.pop("format", "smi"),
        **values,
    }


def _case(
    case_id: str,
    index: int,
    *,
    structures: list[dict] | None = None,
    user_approval: bool = False,
) -> dict:
    result = {
        "id": case_id,
        "local_id": f"local-{case_id}",
        "public_index": index,
        "identifiers": {"CSD": case_id.upper()},
        "metadata": {"supplier": "test"},
        "core_info": {"sample": case_id},
        "observations": [
            {
                "observation_id": f"observation-{case_id}",
                "property": "spin",
                "value": index + 1,
            }
        ],
        "structures": structures or [],
    }
    if user_approval:
        result["attestations"] = [
            {
                "attestation_id": f"attestation-{case_id}",
                "authority": "User",
                "actor": "curator@example.test",
                "approved": True,
            }
        ]
    return result


def _root(
    tmp_path: Path,
    cases: list[dict],
    *,
    required_roles: list[str] | None = None,
) -> Path:
    root = tmp_path / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    intake.write_text(
        "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8"
    )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "demo",
        "release_id": "demo-r1",
        "version": "1.0",
        "title": "Demo",
        "description": "Assessment generation test",
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
        "required_structure_roles": required_roles or [],
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _build_state(root: Path) -> tuple[BundleState, dict]:
    policy = _policy()
    state = BundleState.build(
        root,
        "demo",
        policy_json=policy,
        target_context={"mode": "new", "store_id": None},
    )
    return state, state.policy


def _registry() -> dict:
    from hotpot.ds.governance import default_registry_snapshot

    return default_registry_snapshot()


def test_build_writes_three_indexes_and_does_not_publish_current_pointer(
    tmp_path: Path,
) -> None:
    cases = [
        _case("incomplete", 0),
        _case("ordinary", 1, structures=[_structure("ordinary", "structures/o.smi")]),
        _case(
            "benchmark",
            2,
            structures=[_structure("benchmark", "structures/b.smi")],
            user_approval=True,
        ),
    ]
    root = _root(tmp_path, cases, required_roles=["primary"])
    structures = root / "structures"
    structures.mkdir()
    (structures / "o.smi").write_text("O\n", encoding="utf-8")
    (structures / "b.smi").write_text("N\n", encoding="utf-8")
    state, policy = _build_state(root)

    generation = AssessmentGeneration.build(root, state, policy, _registry())

    assert list(generation.iter_archive_ids("incomplete")) == ["incomplete"]
    assert list(generation.iter_archive_ids("ordinary")) == ["ordinary"]
    assert list(generation.iter_archive_ids("benchmark")) == ["benchmark"]
    assert generation.get_case("incomplete")["archive_class"] == "incomplete"
    assert generation.get_case("benchmark")["benchmark_status"] == "admitted"
    assert generation.summary["archive_class"] == {
        "benchmark": 1,
        "incomplete": 1,
        "ordinary": 1,
    }
    assert state.status["current_generation_id"] is None
    assert not list((root / "assessments").rglob("*-wal"))
    assert not list((root / "assessments").rglob("*-shm"))


def test_generation_digest_is_deterministic_and_existing_generation_is_reused(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, [_case("case-1", 0)])
    state, policy = _build_state(root)

    first = AssessmentGeneration.build(root, state, policy, _registry())
    second = AssessmentGeneration.build(root, state, policy, _registry())

    assert first.id == second.id
    assert first.digest == second.digest
    assert len(list((root / "assessments" / "generations").iterdir())) == 1
    assert [item["case_id"] for item in second.iter_cases()] == ["case-1"]


def test_asset_assessments_record_missing_hash_mismatch_and_parse_failure(
    tmp_path: Path,
) -> None:
    cases = [
        _case(
            "assets",
            0,
            structures=[
                _structure("missing", "structures/missing.smi", role="missing"),
                _structure(
                    "mismatch",
                    "structures/valid.smi",
                    role="mismatch",
                    expected_sha256="0" * 64,
                ),
                _structure("invalid", "structures/invalid.smi", role="invalid"),
            ],
        )
    ]
    root = _root(tmp_path, cases)
    (root / "structures").mkdir()
    (root / "structures" / "valid.smi").write_text("O\n", encoding="utf-8")
    (root / "structures" / "invalid.smi").write_text(
        "this is not smiles ###\n", encoding="utf-8"
    )
    state, policy = _build_state(root)

    generation = AssessmentGeneration.build(root, state, policy, _registry())
    result = generation.get_case("assets")
    assets = {item["role"]: item for item in result["asset_assessments"]}

    assert assets["missing"]["presence"] == "missing"
    assert assets["missing"]["hash_result"] == "unchecked"
    assert assets["mismatch"]["presence"] == "present"
    assert assets["mismatch"]["hash_result"] == "mismatch"
    assert assets["mismatch"]["parse_result"] == "valid"
    assert assets["invalid"]["parse_result"] == "invalid"
    parser = generation.manifest["runtime_policy"]["parsers"]["entries"][0]
    parser_digest = parser["source"]["digest"]
    assert assets["mismatch"]["parser_implementation_digest"] == parser_digest
    assert assets["invalid"]["parser_implementation_digest"] == parser_digest
    assert assets["mismatch"]["parser_protocol"] == parser["protocol"]
    assert assets["mismatch"]["parser_binding_digest"] == parser["digest"]
    assert assets["mismatch"]["parser_distribution_identity"] == parser[
        "distributions"
    ]


def test_asset_collections_and_representations_are_all_assessed(tmp_path: Path) -> None:
    case = _case(
        "multi",
        0,
        structures=[
            {
                "structure_id": "structure-multi",
                "role": "primary",
                "assets": [{"path": "structures/a.smi", "format": "smi"}],
                "representations": {
                    "xyz": {"path": "structures/a.xyz"},
                },
            }
        ],
    )
    root = _root(tmp_path, [case])
    (root / "structures").mkdir()
    (root / "structures" / "a.smi").write_text("O\n", encoding="utf-8")
    (root / "structures" / "a.xyz").write_text(
        "1\nwater oxygen\nO 0 0 0\n", encoding="utf-8"
    )
    state, policy = _build_state(root)

    assets = AssessmentGeneration.build(
        root, state, policy, _registry()
    ).get_case("multi")["asset_assessments"]

    assert sorted((item["format"], item["presence"]) for item in assets) == [
        ("smi", "present"),
        ("xyz", "present"),
    ]


def test_open_rejects_inventory_and_database_tampering(tmp_path: Path) -> None:
    root = _root(tmp_path, [_case("case-1", 0)])
    state, policy = _build_state(root)
    generation = AssessmentGeneration.build(root, state, policy, _registry())
    directory = root / "assessments" / "generations" / generation.id

    (directory / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(GenerationIntegrityError, match="content changed"):
        AssessmentGeneration.open(root, generation.id, generation.digest)


def test_open_rejects_incomplete_manifest_inventory(tmp_path: Path) -> None:
    root = _root(tmp_path, [_case("case-1", 0)])
    state, policy = _build_state(root)
    generation = AssessmentGeneration.build(root, state, policy, _registry())
    directory = root / "assessments" / "generations" / generation.id
    unexpected = directory / "unlisted.txt"
    unexpected.write_text("not in generation.json", encoding="utf-8")

    with pytest.raises(GenerationIntegrityError, match="inventory is incomplete"):
        AssessmentGeneration.open(root, generation.id)


def test_open_checks_expected_digest_and_get_case_errors(tmp_path: Path) -> None:
    root = _root(tmp_path, [_case("case-1", 0)])
    state, policy = _build_state(root)
    generation = AssessmentGeneration.build(root, state, policy, _registry())

    reopened = AssessmentGeneration.open(root, generation.id, generation.digest)
    assert reopened.manifest["source"]["digest"] == state.source_digest
    assert reopened.get_case("case-1")["input_digest"] == hashlib.sha256(
        json.dumps(
            state.get_case("case-1"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(KeyError):
        reopened.get_case("absent")
    with pytest.raises(GenerationIntegrityError, match="expected digest"):
        AssessmentGeneration.open(root, generation.id, "f" * 64)


def test_generation_exposes_certificate_manifest_binding(tmp_path: Path) -> None:
    root = _root(tmp_path, [_case("case-1", 0)])
    state, policy = _build_state(root)
    generation = AssessmentGeneration.build(root, state, policy, _registry())

    assert generation.manifest_path == (
        root / "assessments" / "generations" / generation.id / "generation.json"
    )
    assert generation.manifest_digest == hashlib.sha256(
        generation.manifest_path.read_bytes()
    ).hexdigest()


def test_open_rejects_tampered_frozen_runtime_policy(tmp_path: Path) -> None:
    root = _root(tmp_path, [_case("case-1", 0)])
    state, policy = _build_state(root)
    generation = AssessmentGeneration.build(root, state, policy, _registry())
    manifest = json.loads(generation.manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_policy"]["runtime"]["python"]["version"] = "tampered"
    generation.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(GenerationIntegrityError, match="runtime policy"):
        AssessmentGeneration.open(root, generation.id, generation.digest)


def test_hotpot_authority_is_verified_on_a_derived_copy_only(tmp_path: Path) -> None:
    case = _case("claimed-hotpot", 0)
    case["attestations"] = [
        {
            "attestation_id": "attestation-claimed-hotpot",
            "authority": "Hotpot",
            "actor": "untrusted-caller",
            "approved": True,
        }
    ]
    root = _root(tmp_path, [case])
    state, policy = _build_state(root)

    assessed = AssessmentGeneration.build(
        root, state, policy, _registry()
    ).get_case("claimed-hotpot")

    assert assessed["benchmark"] is False
    assert assessed["governance_assessments"] == [
        {
            "attestation_id": "attestation-claimed-hotpot",
            "reason": "invalid_attestation",
            "signature_verified": False,
        }
    ]
    assert "signature_verified" not in state.get_case("claimed-hotpot")[
        "attestations"
    ][0]


def _conflict_databases() -> tuple[sqlite3.Connection, sqlite3.Connection]:
    assessment = sqlite3.connect(":memory:")
    work = sqlite3.connect(":memory:")
    _initialize_assessment_database(assessment)
    _initialize_work_database(work)
    return assessment, work


def _insert_candidate(
    work: sqlite3.Connection,
    case_id: str,
    observation_id: str,
    value: object,
) -> dict:
    observation = {
        "observation_id": observation_id,
        "property": "spin",
        "value": value,
    }
    work.execute(
        "INSERT INTO candidates VALUES (?,?,?,?,?)",
        ("sample", "spin", case_id, observation_id, canonical_json(observation)),
    )
    return observation


def test_streamed_conflicts_preserve_pair_order_digest_and_case_links() -> None:
    assessment, work = _conflict_databases()
    rows = [
        ("case-b", "observation-2", "low"),
        ("case-a", "observation-2", "high"),
        ("case-a", "observation-1", "low"),
        ("case-c", "observation-1", "low"),
    ]
    observations = {
        (case_id, observation_id): _insert_candidate(
            work, case_id, observation_id, value
        )
        for case_id, observation_id, value in rows
    }

    _assess_conflicts(assessment, work, {}, "identity-schema")

    ordered_rows = sorted(observations)
    ordered_digest = hashlib.sha256()
    expected_exceptions = []
    consistent = 0
    for left_index, left_key in enumerate(ordered_rows):
        for right_key in ordered_rows[left_index + 1 :]:
            result = compare_observations(
                observations[left_key], observations[right_key], {}
            )
            pair = {
                "sample_key": "sample",
                "identity_schema_digest": "identity-schema",
                "property_key": "spin",
                "left_case_id": left_key[0],
                "right_case_id": right_key[0],
                "left_observation_id": left_key[1],
                "right_observation_id": right_key[1],
                **result,
            }
            encoded = canonical_json(pair).encode("utf-8")
            ordered_digest.update(len(encoded).to_bytes(8, "big"))
            ordered_digest.update(encoded)
            if result["outcome"] == "consistent":
                consistent += 1
            else:
                expected_exceptions.append(pair)

    group = json.loads(
        assessment.execute(
            "SELECT assessment_json FROM conflict_groups"
        ).fetchone()[0]
    )
    assert group == {
        "sample_key": "sample",
        "identity_schema_digest": "identity-schema",
        "property_key": "spin",
        "candidate_count": 6,
        "consistent_count": consistent,
        "exception_count": len(expected_exceptions),
        "ordered_results_digest": ordered_digest.hexdigest(),
    }
    observed_exceptions = [
        json.loads(row[0])
        for row in assessment.execute(
            "SELECT assessment_json FROM conflict_assessments ORDER BY conflict_id"
        )
    ]
    assert observed_exceptions == sorted(
        expected_exceptions,
        key=lambda pair: "conflict-"
        + hashlib.sha256(canonical_json(pair).encode("utf-8")).hexdigest()[:32],
    )
    expected_links = sum(
        len({pair["left_case_id"], pair["right_case_id"]})
        for pair in expected_exceptions
    )
    assert assessment.execute(
        "SELECT count(*) FROM conflict_case_links"
    ).fetchone()[0] == expected_links


def test_large_conflict_group_is_streamed_and_case_lookup_uses_primary_index() -> None:
    assessment, work = _conflict_databases()
    size = 256
    for index in range(size):
        _insert_candidate(
            work,
            f"case-{index:04d}",
            f"observation-{index:04d}",
            "same-value",
        )

    _assess_conflicts(assessment, work, {}, "identity-schema")

    group = json.loads(
        assessment.execute(
            "SELECT assessment_json FROM conflict_groups"
        ).fetchone()[0]
    )
    assert group["candidate_count"] == size * (size - 1) // 2
    assert group["consistent_count"] == group["candidate_count"]
    assert group["exception_count"] == 0
    assert assessment.execute(
        "SELECT count(*) FROM conflict_assessments"
    ).fetchone()[0] == 0

    candidate_plan = " ".join(
        row[3]
        for row in work.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT case_id,observation_id,observation_json
            FROM candidates
            WHERE sample_key=? AND property_key=?
              AND (case_id,observation_id)>(?,?)
            ORDER BY case_id,observation_id
            """,
            ("sample", "spin", "case-0000", "observation-0000"),
        )
    )
    assert "PRIMARY KEY" in candidate_plan
    assert "sample_key=? AND property_key=?" in candidate_plan

    case_lookup_plan = " ".join(
        row[3]
        for row in assessment.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT conflict_assessments.assessment_json
            FROM conflict_case_links
            JOIN conflict_assessments USING(conflict_id)
            WHERE conflict_case_links.case_id=?
            ORDER BY conflict_case_links.conflict_id
            """,
            ("case-0000",),
        )
    )
    assert "conflict_case_links" in case_lookup_plan
    assert "case_id=?" in case_lookup_plan


def test_asset_snapshot_rejects_a_b_a_change_between_hash_and_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "structure.smi"
    source.write_text("O\n", encoding="utf-8")
    original_mtime = source.stat().st_mtime_ns
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    parsed = {}

    def mutate_source_while_parsing(snapshot: Path, fmt: str) -> bool:
        source.write_text("N\n", encoding="utf-8")
        parsed["path"] = snapshot
        parsed["bytes"] = snapshot.read_bytes()
        parsed["format"] = fmt
        source.write_text("O\n", encoding="utf-8")
        os.utime(source, ns=(original_mtime, original_mtime + 1_000_000))
        return True

    monkeypatch.setattr(
        generation_module,
        "_structure_is_parseable",
        mutate_source_while_parsing,
    )

    with pytest.raises(AssetSourceChangedError, match="changed"):
        generation_module._assess_asset(
            tmp_path,
            {
                "asset_id": "asset-1",
                "case_id": "case-1",
                "collection": "structures",
                "record_id": "structure-1",
                "structure_id": "structure-1",
                "role": "primary",
                "path": "structure.smi",
                "format": "smi",
                "expected_sha256": None,
            },
            [],
            snapshots,
            runtime_policy(),
        )

    assert parsed["path"] != source
    assert parsed["bytes"] == b"O\n"
    assert parsed["format"] == "smi"
    assert source.read_bytes() == b"O\n"
    assert list(snapshots.iterdir()) == []
