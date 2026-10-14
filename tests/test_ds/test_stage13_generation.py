from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hotpot.ds.gates._persistence import default_gates, serialize_gate_configs
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA, BundleState
from hotpot.ds.public._generation import (
    AssessmentGeneration,
    GenerationIntegrityError,
)


def _policy() -> dict:
    return {"gates": serialize_gate_configs(default_gates())}


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
    return state, policy


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

    result = AssessmentGeneration.build(root, state, policy, _registry()).get_case(
        "assets"
    )
    assets = {item["role"]: item for item in result["asset_assessments"]}

    assert assets["missing"]["presence"] == "missing"
    assert assets["missing"]["hash_result"] == "unchecked"
    assert assets["mismatch"]["presence"] == "present"
    assert assets["mismatch"]["hash_result"] == "mismatch"
    assert assets["mismatch"]["parse_result"] == "valid"
    assert assets["invalid"]["parse_result"] == "invalid"


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
