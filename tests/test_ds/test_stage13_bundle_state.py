from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import hotpot.ds.public._bundle_state as bundle_state_module
from hotpot.ds.public._bundle_state import (
    BUNDLE_SCHEMA,
    BundleAlreadyBuiltError,
    BundleNotBuiltError,
    BundleState,
    BundleStateError,
    ConcurrentStateError,
)


def _case(
    case_id: str = "case-1",
    *,
    public_index: int | None = None,
    structure_path: str | None = None,
) -> dict:
    value = {
        "id": case_id,
        "local_id": f"local-{case_id}",
        "identifiers": {"CSD": case_id.upper()},
        "core_info": {"metal": "Eu"},
        "observations": [
            {
                "observation_id": f"observation-{case_id}",
                "property_type": "spin",
                "value": 3,
            }
        ],
    }
    if public_index is not None:
        value["public_index"] = public_index
    if structure_path is not None:
        value["structures"] = [
            {
                "structure_id": f"structure-{case_id}",
                "role": "primary",
                "path": structure_path,
                "format": "smi",
            }
        ]
    return value


def _root(tmp_path: Path, cases: list[dict] | None = None) -> Path:
    root = tmp_path / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    values = [_case()] if cases is None else cases
    intake.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "demo",
        "release_id": "demo-r1",
        "version": "1.0",
        "title": "Demo",
        "description": "Bundle state test",
        "license": "CC0-1.0",
        "access_policy": "open",
        "provenance": {"created_by": "test"},
        "core_labels": ["spin"],
        "core_info": ["metal"],
        "intake": {"format": "jsonl", "path": "evidence/intake/cases.jsonl"},
        "identity_schema": {"fields": ["core_info.metal"]},
        "required_structure_roles": [],
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _build(root: Path) -> BundleState:
    return BundleState.build(
        root,
        "demo",
        policy_json={"gates": {"quality": {"name": "default"}}},
        target_context={"mode": "new", "store_id": None},
    )


def test_build_and_validation_only_open_are_distinct(tmp_path: Path) -> None:
    root = _root(tmp_path)
    with pytest.raises(BundleNotBuiltError):
        BundleState.open(root, "demo")

    state = _build(root)
    assert len(state) == 1
    assert state.manifest["dataset_id"] == "demo"
    assert state.manifest["release_id"] == "demo-r1"

    with pytest.raises(BundleAlreadyBuiltError):
        _build(root)
    reopened = BundleState.open(root, "demo")
    assert reopened.get_case("case-1")["public_index"] == 0


def test_manifest_is_exactly_one_release_and_uses_safe_paths(tmp_path: Path) -> None:
    root = _root(tmp_path)
    manifest_path = root / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("release_id")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleStateError, match="release_id"):
        _build(root)

    root = _root(tmp_path / "unsafe")
    manifest_path = root / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["intake"]["path"] = "../cases.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleStateError, match="safe POSIX"):
        _build(root)

    root = _root(tmp_path / "serial")
    manifest_path = root / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_serial"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _build(root).manifest["release_serial"] == 7


def test_manifest_property_policy_and_case_entities_are_canonical_facts(
    tmp_path: Path,
) -> None:
    case = _case()
    case["entities"] = [
        {"entity_id": "entity-eu", "role": "metal", "element": "Eu"}
    ]
    root = _root(tmp_path, [case])
    manifest_path = root / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["property_definitions"] = {"spin": {"type": "integer"}}
    manifest["property_comparators"] = {"spin": {"absolute_tolerance": 0}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    state = _build(root)

    assert state.manifest["property_definitions"]["spin"]["type"] == "integer"
    assert state.manifest["property_comparators"]["spin"]["absolute_tolerance"] == 0
    assert state.get_case("case-1")["entities"][0]["entity_id"] == "entity-eu"


@pytest.mark.parametrize(
    "cases, message",
    [
        ([_case("same"), _case("same")], "duplicate"),
        ([_case("first", public_index=7), _case("second", public_index=7)], "duplicate"),
    ],
)
def test_duplicate_case_identity_or_index_is_rejected(
    tmp_path: Path, cases: list[dict], message: str
) -> None:
    with pytest.raises(BundleStateError, match=message):
        _build(_root(tmp_path, cases))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda case: case.update({"quality_level": "B4"}),
        lambda case: case["observations"][0].update({"evidence_level": "E4"}),
        lambda case: case.update(
            {
                "claims": [
                    {"claim_id": "claim-1", "text": "expert", "benchmark": True}
                ]
            }
        ),
    ],
)
def test_operative_assessment_fields_are_not_source_facts(
    tmp_path: Path, mutator
) -> None:
    case = _case()
    mutator(case)
    with pytest.raises(BundleStateError, match="operative assessment field"):
        _build(_root(tmp_path, [case]))


def test_claimed_assessment_fields_are_preserved_as_nonoperative_facts(
    tmp_path: Path,
) -> None:
    case = _case()
    case["claimed_quality_level"] = "supplier-gold"
    case["observations"][0]["claimed_evidence_level"] = "supplier-primary"
    state = _build(_root(tmp_path, [case]))
    loaded = state.get_case("case-1")
    assert loaded["claimed_quality_level"] == "supplier-gold"
    assert loaded["observations"][0]["claimed_evidence_level"] == "supplier-primary"


def test_hotpot_signature_verification_cannot_be_supplied_as_a_fact(tmp_path: Path) -> None:
    case = _case()
    case["attestations"] = [
        {
            "attestation_id": "attestation-1",
            "authority": "Hotpot",
            "approved": True,
            "signature_verified": True,
        }
    ]
    with pytest.raises(BundleStateError, match="derived trust result"):
        _build(_root(tmp_path, [case]))


def test_actual_asset_bytes_participate_in_source_digest(tmp_path: Path) -> None:
    root = _root(
        tmp_path,
        [_case(structure_path="structures/primary.smi")],
    )
    structure = root / "structures" / "primary.smi"
    structure.parent.mkdir()
    structure.write_text("O\n", encoding="utf-8")
    state = _build(root)
    before = state.source_digest
    assert state.status["recorded_source_digest"] == before

    structure.write_text("N\n", encoding="utf-8")
    after = state.source_digest
    assert after != before
    assert state.status["recorded_source_digest"] == before

    refreshed = state.refresh_source_state()
    assert refreshed["changed"] is True
    assert refreshed["source_revision"] == 1
    assert state.status["recorded_source_digest"] == after
    assert state.refresh_source_state()["changed"] is False


def test_built_manifest_identity_cannot_be_changed_out_of_band(tmp_path: Path) -> None:
    root = _root(tmp_path)
    state = _build(root)
    frozen = state.status
    case = state.get_case("case-1")
    manifest_path = root / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_id"] = "different-release"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleStateError, match="immutable built manifest"):
        BundleState.open(root, "demo")

    for read in (
        lambda: state.manifest,
        lambda: state.status,
        lambda: state.source_digest,
        lambda: state.policy,
        lambda: state.target_context,
        lambda: len(state),
        lambda: state.get_case("case-1"),
        lambda: tuple(state.iter_cases()),
        lambda: state.refresh_source_state(),
        lambda: state.refresh_governance_state(),
        lambda: state.mutate_case("case-1", case, "no_op", {}),
        lambda: state.publish_generation(
            "generation-1",
            "a" * 64,
            expected_source_revision=frozen["source_revision"],
            expected_source_digest=frozen["source_digest"],
            expected_policy_revision=frozen["policy_revision"],
            expected_policy_digest=frozen["policy_digest"],
            expected_governance_registry_revision=frozen[
                "governance_registry_revision"
            ],
            expected_governance_registry_digest=frozen[
                "governance_registry_digest"
            ],
        ),
    ):
        with pytest.raises(BundleStateError, match="immutable built manifest"):
            read()


def test_manifest_formatting_is_not_an_out_of_band_semantic_change(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    state = _build(root)
    manifest_path = root / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert state.manifest["release_id"] == "demo-r1"
    assert state.status["source_digest"] == state.status["recorded_source_digest"]


def test_case_mutation_commits_revision_and_audit_atomically(tmp_path: Path) -> None:
    state = _build(_root(tmp_path))
    before = state.status
    with state.connect() as connection:
        audit_count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    invalid = state.get_case("case-1")
    invalid["archive_class"] = "benchmark"
    with pytest.raises(BundleStateError):
        state.mutate_case("case-1", invalid, "invalid", {})
    assert state.status["source_revision"] == before["source_revision"]
    with state.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == audit_count

    changed = state.get_case("case-1")
    changed["core_info"]["metal"] = "Am"
    result = state.mutate_case(
        "case-1", changed, "core_info_changed", {"reason": "corrected"}
    )
    assert result["source_revision"] == before["source_revision"] + 1
    assert state.get_case("case-1")["core_info"]["metal"] == "Am"
    with state.connect() as connection:
        event = connection.execute(
            "SELECT action, details_json FROM audit_events ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    assert event["action"] == "core_info_changed"
    assert json.loads(event["details_json"])["reason"] == "corrected"


def test_validation_only_open_does_not_repair_readable_projections(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    _build(root)
    (root / "audit" / "events.jsonl").unlink()
    (root / "policy" / "policy.json").unlink()

    BundleState.open(root, "demo")
    assert not (root / "audit" / "events.jsonl").exists()
    assert not (root / "policy" / "policy.json").exists()


def test_generation_publication_is_compare_and_swap(tmp_path: Path) -> None:
    state = _build(_root(tmp_path))
    status = state.status
    result = state.publish_generation(
        "generation-1",
        "a" * 64,
        expected_source_revision=status["source_revision"],
        expected_source_digest=status["source_digest"],
        expected_policy_revision=status["policy_revision"],
        expected_policy_digest=status["policy_digest"],
        expected_governance_registry_revision=status[
            "governance_registry_revision"
        ],
        expected_governance_registry_digest=status["governance_registry_digest"],
    )
    assert result["generation_id"] == "generation-1"
    assert state.status["is_synchronized"] is True

    changed = state.get_case("case-1")
    changed["metadata"]["note"] = "later edit"
    state.mutate_case("case-1", changed, "metadata_changed", {})
    assert state.status["is_synchronized"] is False
    with pytest.raises(ConcurrentStateError):
        state.publish_generation(
            "generation-2",
            "b" * 64,
            expected_source_revision=status["source_revision"],
            expected_source_digest=status["source_digest"],
            expected_policy_revision=status["policy_revision"],
            expected_policy_digest=status["policy_digest"],
            expected_governance_registry_revision=status[
                "governance_registry_revision"
            ],
            expected_governance_registry_digest=status[
                "governance_registry_digest"
            ],
        )


def test_governance_binding_is_persisted_and_part_of_synchronization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _build(_root(tmp_path))
    initial = state.status
    assert initial["recorded_governance_registry_revision"] == initial[
        "governance_registry_revision"
    ]
    assert initial["recorded_governance_registry_digest"] == initial[
        "governance_registry_digest"
    ]

    state.publish_generation(
        "generation-1",
        "a" * 64,
        expected_source_revision=initial["source_revision"],
        expected_source_digest=initial["source_digest"],
        expected_policy_revision=initial["policy_revision"],
        expected_policy_digest=initial["policy_digest"],
        expected_governance_registry_revision=initial[
            "governance_registry_revision"
        ],
        expected_governance_registry_digest=initial[
            "governance_registry_digest"
        ],
    )
    assert state.status["is_synchronized"] is True

    advanced = (initial["governance_registry_revision"] + 1, "b" * 64)
    monkeypatch.setattr(
        bundle_state_module, "_trusted_governance_binding", lambda: advanced
    )
    stale = state.status
    assert stale["is_synchronized"] is False
    assert stale["governance_registry_revision"] == advanced[0]
    assert stale["recorded_governance_registry_revision"] == initial[
        "governance_registry_revision"
    ]

    refreshed = state.refresh_governance_state()
    assert refreshed["changed"] is True
    assert refreshed["governance_registry_revision"] == advanced[0]
    recorded = state.status
    assert recorded["recorded_governance_registry_digest"] == advanced[1]
    assert recorded["is_synchronized"] is False

    state.publish_generation(
        "generation-2",
        "c" * 64,
        expected_source_revision=recorded["source_revision"],
        expected_source_digest=recorded["source_digest"],
        expected_policy_revision=recorded["policy_revision"],
        expected_policy_digest=recorded["policy_digest"],
        expected_governance_registry_revision=recorded[
            "governance_registry_revision"
        ],
        expected_governance_registry_digest=recorded[
            "governance_registry_digest"
        ],
    )
    assert state.status["is_synchronized"] is True


def test_governance_publication_cas_rejects_a_registry_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _build(_root(tmp_path))
    frozen = state.status
    monkeypatch.setattr(
        bundle_state_module,
        "_trusted_governance_binding",
        lambda: (frozen["governance_registry_revision"] + 1, "d" * 64),
    )

    with pytest.raises(ConcurrentStateError):
        state.publish_generation(
            "generation-1",
            "a" * 64,
            expected_source_revision=frozen["source_revision"],
            expected_source_digest=frozen["source_digest"],
            expected_policy_revision=frozen["policy_revision"],
            expected_policy_digest=frozen["policy_digest"],
            expected_governance_registry_revision=frozen[
                "governance_registry_revision"
            ],
            expected_governance_registry_digest=frozen[
                "governance_registry_digest"
            ],
        )


def test_sqlite_intake_uses_the_declared_current_table(tmp_path: Path) -> None:
    root = _root(tmp_path)
    sqlite_path = root / "evidence" / "intake" / "cases.sqlite3"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            "CREATE TABLE cases(case_id TEXT PRIMARY KEY, public_index INTEGER UNIQUE, payload_json TEXT)"
        )
        case = _case(public_index=4)
        connection.execute(
            "INSERT INTO cases VALUES (?, ?, ?)",
            (case["id"], 4, json.dumps(case)),
        )
    manifest_path = root / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["intake"] = {
        "format": "sqlite",
        "path": "evidence/intake/cases.sqlite3",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    state = _build(root)
    assert state.get_case("case-1")["public_index"] == 4
