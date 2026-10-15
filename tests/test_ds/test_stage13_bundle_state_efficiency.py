from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import hotpot.ds.public._bundle_state as state_module
from hotpot.ds.gates._persistence import default_gates, serialize_gate_configs
from hotpot.ds.public._bundle_state import (
    BUNDLE_SCHEMA,
    BundleState,
    BundleStateError,
)
from hotpot.ds.public.bundle import Bundle
from hotpot.ds.public.case import UnsynchronizedBundleError
from hotpot.ds.public._implementation import runtime_policy


def _case(index: int, *, asset: str | None = None) -> dict:
    case = {
        "id": f"case-{index}",
        "local_id": f"local-{index}",
        "public_index": index,
        "core_info": {"sample": f"sample-{index}"},
        "observations": [
            {
                "observation_id": f"observation-{index}",
                "property_type": "spin",
                "value": index + 1,
            }
        ],
    }
    if asset is not None:
        case["structures"] = [
            {
                "structure_id": f"structure-{index}",
                "role": "primary",
                "format": "smi",
                "path": asset,
            }
        ]
    return case


def _root(
    tmp_path: Path,
    *,
    cases: list[dict] | None = None,
    control_path: str | None = None,
) -> Path:
    root = tmp_path / "bundle"
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    values = cases or [_case(0)]
    intake.write_text(
        "".join(json.dumps(case) + "\n" for case in values),
        encoding="utf-8",
    )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "state-efficiency",
        "release_id": "state-efficiency-r1",
        "version": "1.0",
        "title": "State efficiency",
        "description": "BundleState efficiency contract",
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
    if control_path is not None:
        path = root / control_path
        content = path.read_bytes()
        manifest["control_assets"] = [
            {
                "category": "source",
                "name": "declared-control",
                "path": control_path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        ]
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _build(root: Path) -> BundleState:
    return BundleState.build(
        root,
        "state-efficiency",
        {
            "schema": "hotpot-ds-policy/current",
            "gates": serialize_gate_configs(default_gates()),
            "runtime_policy": runtime_policy(),
        },
        {"mode": "new", "store_id": None},
    )


def test_case_mutation_marks_digest_dirty_without_a_full_source_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _build(_root(tmp_path))
    original = BundleState._compute_source_digest

    def forbidden(*args, **kwargs):
        raise AssertionError("mutation or dirty status performed a full source proof")

    monkeypatch.setattr(BundleState, "_compute_source_digest", forbidden)

    def edit(case):
        case["metadata"]["note"] = "edited"
        return {"reason": "test"}, "done"

    result = state.mutate_case("case-0", "metadata_changed", edit)
    assert result["result"] == "done"
    assert result["source_digest"] is None
    assert state.status["source_digest_state"] == "dirty"
    assert state.status["is_synchronized"] is False
    assert state.get_case("case-0")["metadata"]["note"] == "edited"

    reopened = BundleState.open(state.root, "state-efficiency")
    assert reopened.status["source_integrity"] == "dirty"
    monkeypatch.setattr(BundleState, "_compute_source_digest", original)
    revision = reopened.status["source_revision"]
    refreshed = reopened.refresh_source_state()
    assert refreshed["source_revision"] == revision
    assert reopened.status["source_digest_state"] == "current"
    assert reopened.status["recorded_source_digest"] == refreshed["source_digest"]


def test_invalid_mutation_rolls_back_facts_revision_and_audit(tmp_path: Path) -> None:
    state = _build(_root(tmp_path))
    before = state.status

    def invalid(case):
        case["archive_class"] = "benchmark"
        return {}, None

    with pytest.raises(BundleStateError):
        state.mutate_case("case-0", "invalid", invalid)

    after = state.status
    assert after["source_revision"] == before["source_revision"]
    assert after["recorded_source_digest"] == before["recorded_source_digest"]
    assert after["audit_revision"] == before["audit_revision"]
    assert after["audit_projection_current"] is True


def test_projection_append_fails_closed_and_explicit_refresh_repairs_it(
    tmp_path: Path,
) -> None:
    state = _build(_root(tmp_path))
    projection = state.root / "audit" / "events.jsonl"
    with projection.open("ab") as stream:
        stream.write(b"out-of-band\n")

    def edit(case):
        case["metadata"]["note"] = "edited"
        return {}, None

    result = state.mutate_case("case-0", "metadata_changed", edit)
    assert result["audit_projection_updated"] is False
    assert state.status["audit_projection_current"] is False

    state.refresh_audit_projection()
    assert state.status["audit_projection_current"] is True
    projected = [json.loads(line) for line in projection.read_text().splitlines()]
    with state.connect() as connection:
        authoritative_count = connection.execute(
            "SELECT COUNT(*) FROM audit_events"
        ).fetchone()[0]
    assert len(projected) == authoritative_count == 2
    assert projected[-1]["action"] == "metadata_changed"


def test_projection_is_stale_if_process_stops_before_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _build(_root(tmp_path))
    monkeypatch.setattr(
        state, "_append_audit_projection_event", lambda *args, **kwargs: False
    )

    def edit(case):
        case["metadata"]["note"] = "committed"
        return {}, None

    state.mutate_case("case-0", "metadata_changed", edit)
    reopened = BundleState.open(state.root, "state-efficiency")
    assert reopened.get_case("case-0")["metadata"]["note"] == "committed"
    assert reopened.status["audit_projection_current"] is False
    reopened.refresh_audit_projection()
    assert reopened.status["audit_projection_current"] is True


def test_projection_is_stale_if_process_stops_after_append_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _build(_root(tmp_path))
    projection = state.root / "audit" / "events.jsonl"

    def append_then_stop(event, **kwargs):
        with projection.open("ab") as stream:
            stream.write(state_module._canonical_bytes(dict(event)) + b"\n")
            stream.flush()
            state_module.os.fsync(stream.fileno())
        raise RuntimeError("simulated process stop")

    monkeypatch.setattr(state, "_append_audit_projection_event", append_then_stop)

    def edit(case):
        case["metadata"]["note"] = "committed"
        return {}, None

    with pytest.raises(RuntimeError, match="simulated process stop"):
        state.mutate_case("case-0", "metadata_changed", edit)

    reopened = BundleState.open(state.root, "state-efficiency")
    assert reopened.get_case("case-0")["metadata"]["note"] == "committed"
    assert reopened.status["audit_projection_current"] is False
    reopened.refresh_audit_projection()
    assert reopened.status["audit_projection_current"] is True


def test_batched_build_rolls_back_without_publishing_partial_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [_case(0), _case(1), _case(2)]
    cases[2]["public_index"] = 0
    root = _root(tmp_path, cases=cases)
    monkeypatch.setattr(state_module, "_CASE_INSERT_BATCH_SIZE", 2)

    with pytest.raises(BundleStateError, match="duplicate"):
        _build(root)

    assert not (root / "cases" / "facts.sqlite3").exists()
    assert not (root / "audit" / "events.jsonl").exists()
    assert not list((root / "cases").glob(".facts.*.tmp"))


def test_source_proof_detects_scientific_control_and_payload_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fixture"
    shared = root / "bundle" / "assets" / "shared.smi"
    shared.parent.mkdir(parents=True)
    shared.write_text("O\n", encoding="utf-8")
    bundle_root = _root(
        root,
        cases=[_case(0, asset="assets/shared.smi")],
        control_path="assets/shared.smi",
    )
    state = _build(bundle_root)
    recorded = state.status["recorded_source_digest"]

    shared.write_text("N\n", encoding="utf-8")
    with pytest.raises(BundleStateError, match="declared control asset"):
        _ = state.status
    shared.write_text("O\n", encoding="utf-8")

    with sqlite3.connect(state.path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM cases WHERE case_id='case-0'"
            ).fetchone()[0]
        )
        payload["metadata"]["out_of_band"] = True
        connection.execute(
            "UPDATE cases SET payload_json=? WHERE case_id='case-0'",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
        )
    status = state.status
    assert status["source_integrity"] == "changed_out_of_band"
    assert status["source_digest"] != recorded
    assert status["is_synchronized"] is False


def test_source_proof_hashes_each_declared_control_once_and_observes_cases_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "fixture"
    shared = root / "bundle" / "assets" / "shared.smi"
    shared.parent.mkdir(parents=True)
    shared.write_text("O\n", encoding="utf-8")
    state = _build(
        _root(
            root,
            cases=[_case(0, asset="assets/shared.smi"), _case(1)],
            control_path="assets/shared.smi",
        )
    )
    hashed: list[Path] = []
    observed: list[str] = []

    def digest(path: Path) -> tuple[str, int]:
        hashed.append(path)
        return state_module._digest_file(path)

    with state.connect() as connection:
        proof = state._compute_source_proof(
            connection,
            state._stored_manifest(connection),
            file_digest=digest,
            case_observer=lambda case: observed.append(case["id"]),
        )

    assert proof == state.status["recorded_source_digest"]
    assert hashed == [shared]
    assert observed == ["case-0", "case-1"]

    def forbidden(path: Path) -> tuple[str, int]:
        raise AssertionError(f"ordinary state connection hashed {path}")

    monkeypatch.setattr(state_module, "_digest_file", forbidden)
    assert state.get_case("case-0")["id"] == "case-0"
    assert state.manifest["control_assets"][0]["path"] == "assets/shared.smi"


def test_open_rejects_schema_v2_without_compatibility(tmp_path: Path) -> None:
    state = _build(_root(tmp_path))
    with sqlite3.connect(state.path) as connection:
        connection.execute("PRAGMA user_version=2")

    with pytest.raises(BundleStateError, match="current schema"):
        BundleState.open(state.root, "state-efficiency")


def test_default_source_hasher_rejects_a_file_changed_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = tmp_path / "asset.smi"
    asset.write_text("O\n", encoding="utf-8")
    real_fstat = state_module.os.fstat
    calls = 0

    def changed_after_read(descriptor: int):
        nonlocal calls
        calls += 1
        observed = real_fstat(descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_mode=observed.st_mode,
                st_size=observed.st_size + 1,
                st_mtime_ns=observed.st_mtime_ns,
                st_ctime_ns=observed.st_ctime_ns,
            )
        return observed

    monkeypatch.setattr(state_module.os, "fstat", changed_after_read)
    with pytest.raises(BundleStateError, match="changed while hashing"):
        state_module._digest_file(asset)


def test_dirty_bundle_rejects_derived_reads_certificate_and_ingest(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    bundle = Bundle(root, "state-efficiency", db=tmp_path / "datasets.sqlite3")
    bundle.synchronize()
    bundle.write_certificate()
    case = bundle.cases[0]
    case.set_core_info("sample", "edited")

    assert case.core_info["sample"] == "edited"
    assert bundle.status["source_digest_state"] == "dirty"
    with pytest.raises(UnsynchronizedBundleError):
        _ = case.quality_level
    with pytest.raises(UnsynchronizedBundleError):
        bundle.write_certificate()
    with pytest.raises(UnsynchronizedBundleError):
        bundle.ingest()


def test_batched_build_publishes_canonical_equivalent_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = [_case(index) for index in range(7)]
    validations = 0
    build_proofs: list[bool] = []
    original_validate = state_module._validate_case
    original_proof = state_module._compute_source_proof_rows

    def counted_validate(*args, **kwargs):
        nonlocal validations
        validations += 1
        return original_validate(*args, **kwargs)

    def counted_proof(*args, **kwargs):
        build_proofs.append(kwargs.get("scientific_paths") is not None)
        return original_proof(*args, **kwargs)

    monkeypatch.setattr(state_module, "_validate_case", counted_validate)
    monkeypatch.setattr(state_module, "_compute_source_proof_rows", counted_proof)
    state = _build(_root(tmp_path, cases=cases))
    assert validations == len(cases)
    assert build_proofs == [True]

    with state.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        rows = connection.execute(
            "SELECT case_id,public_index,payload_json FROM cases "
            "ORDER BY public_index,case_id"
        ).fetchall()
        direct = state._compute_source_proof(
            connection, state._stored_manifest(connection)
        )

    assert [row["case_id"] for row in rows] == [case["id"] for case in cases]
    assert all(
        row["payload_json"]
        == json.dumps(
            json.loads(row["payload_json"]),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in rows
    )
    assert direct == state.status["recorded_source_digest"]

    BundleState.open(state.root, "state-efficiency")
    assert validations == 2 * len(cases)
