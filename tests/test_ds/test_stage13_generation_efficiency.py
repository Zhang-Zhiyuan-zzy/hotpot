from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hotpot.ds.gates._persistence import default_gates, serialize_gate_configs
from hotpot.ds.governance import default_registry_snapshot, verify_registry
from hotpot.ds.public._bundle_state import (
    BUNDLE_SCHEMA,
    BundleState,
    _SynchronizationSnapshot,
)
from hotpot.ds.public._generation import (
    AssessmentGeneration,
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


class _SelectCounter:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.selects: list[str] = []

    def execute(self, statement: str, parameters=()):
        if statement.lstrip().upper().startswith("SELECT"):
            self.selects.append(" ".join(statement.split()))
        return self.connection.execute(statement, parameters)


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
