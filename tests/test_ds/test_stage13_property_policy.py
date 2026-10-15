from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotpot.ds.gates import BenchmarkGate, EvidenceGate, QualityGate
from hotpot.ds.gates._persistence import serialize_gate_configs
from hotpot.ds.governance import default_registry_snapshot
from hotpot.ds.public._assessment import compare_observations
from hotpot.ds.public._bundle_state import (
    BUNDLE_SCHEMA,
    BundleState,
    BundleStateError,
)
from hotpot.ds.public._generation import AssessmentGeneration
from hotpot.ds.public._implementation import runtime_policy
from hotpot.ds.public.bundle import (
    ExistingDatasetPolicyError,
    _assert_manifest_policy,
)


def _root(
    tmp_path: Path,
    observations: list[dict],
    *,
    definitions: dict,
    comparators: dict,
    cases: int = 1,
) -> Path:
    root = tmp_path / "bundle"
    intake = root / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    records = []
    for index in range(cases):
        records.append(
            {
                "id": f"case-{index}",
                "local_id": f"local-{index}",
                "public_index": index,
                "identifiers": {"sample": "same"},
                "metadata": {"supplier": "test"},
                "core_info": {"sample": "same"},
                "observations": [observations[index]],
            }
        )
    intake.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "property-demo",
        "release_id": "property-demo-r1",
        "version": "1.0",
        "title": "Property policy",
        "description": "Property policy test",
        "license": "CC0-1.0",
        "access_policy": "open",
        "provenance": {"created_by": "test"},
        "core_labels": ["temperature"],
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
        "property_definitions": definitions,
        "property_comparators": comparators,
        "required_structure_roles": [],
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _policy(definitions: dict, comparators: dict, *, gate_tolerance=()) -> dict:
    benchmark = (
        BenchmarkGate()
        if not gate_tolerance
        else BenchmarkGate(
            numeric_tolerances={"temperature": {"absolute": gate_tolerance[0]}}
        )
    )
    return {
        "schema": "hotpot-ds-policy/current",
        "gates": serialize_gate_configs(
            (EvidenceGate(), QualityGate(), benchmark)
        ),
        "property_definitions": definitions,
        "property_comparators": comparators,
        "runtime_policy": runtime_policy(),
    }


def test_property_policy_is_persisted_and_observation_type_and_unit_are_fenced(
    tmp_path: Path,
) -> None:
    definitions = {
        "temperature": {
            "type": "real",
            "unit": "K",
            "context_fields": ["phase"],
        }
    }
    comparators = {
        "temperature": {
            "kind": "numeric",
            "absolute_tolerance": 0.1,
            "unit_conversions": {"degC": {"scale": 1, "offset": 273.15}},
        }
    }
    observation = {
        "observation_id": "observation-0",
        "property": "temperature",
        "value": 25.0,
        "unit": "degC",
        "phase": "solid",
    }
    root = _root(
        tmp_path,
        [observation],
        definitions=definitions,
        comparators=comparators,
    )
    state = BundleState.build(
        root,
        "property-demo",
        _policy(definitions, comparators),
        {"mode": "new", "store_id": None},
    )

    assert state.policy["property_definitions"] == definitions
    assert state.policy["property_comparators"] == comparators

    invalid = dict(observation, value="twenty-five")
    invalid_root = _root(
        tmp_path / "invalid-type",
        [invalid],
        definitions=definitions,
        comparators=comparators,
    )
    with pytest.raises(BundleStateError, match="value must have type real"):
        BundleState.build(
            invalid_root,
            "property-demo",
            _policy(definitions, comparators),
            {"mode": "new", "store_id": None},
        )

    unsupported_unit = dict(observation, unit="eV")
    invalid_root = _root(
        tmp_path / "invalid-unit",
        [unsupported_unit],
        definitions=definitions,
        comparators=comparators,
    )
    with pytest.raises(BundleStateError, match="unit is not canonical or convertible"):
        BundleState.build(
            invalid_root,
            "property-demo",
            _policy(definitions, comparators),
            {"mode": "new", "store_id": None},
        )


def test_property_comparator_controls_context_units_and_tolerance() -> None:
    definitions = {
        "temperature": {
            "type": "real",
            "unit": "K",
            "context_fields": ["phase"],
        }
    }
    comparators = {
        "temperature": {
            "kind": "numeric",
            "absolute_tolerance": 0.1,
            "unit_conversions": {"degC": {"scale": 1, "offset": 273.15}},
        }
    }
    left = {
        "property": "temperature",
        "value": 25.0,
        "unit": "degC",
        "phase": "solid",
    }
    right = {
        "property": "temperature",
        "value": 298.2,
        "unit": "K",
        "phase": "solid",
    }
    result = compare_observations(
        left,
        right,
        {"temperature": {"absolute": 100}},
        property_definitions=definitions,
        property_comparators=comparators,
    )
    assert result["outcome"] == "consistent"
    assert result["difference"] == pytest.approx(0.05)
    assert result["allowed"] == pytest.approx(0.1)

    right["value"] = 298.4
    result = compare_observations(
        left,
        right,
        {"temperature": {"absolute": 100}},
        property_definitions=definitions,
        property_comparators=comparators,
    )
    assert result["outcome"] == "conflict"
    assert result["reason"] == "scientific_value_conflict"

    right["phase"] = "solution"
    result = compare_observations(
        left,
        right,
        {},
        property_definitions=definitions,
        property_comparators=comparators,
    )
    assert result["outcome"] == "noncomparable"
    assert result["reason"] == "observation_context_noncomparable"

    right.pop("phase")
    result = compare_observations(
        left,
        right,
        {},
        property_definitions=definitions,
        property_comparators=comparators,
    )
    assert result["outcome"] == "indeterminate"
    assert result["reason"] == "observation_context_indeterminate"


def test_generation_uses_property_comparator_before_benchmark_fallback(
    tmp_path: Path,
) -> None:
    definitions = {"temperature": {"type": "real", "unit": "K"}}
    comparators = {
        "temperature": {"kind": "numeric", "absolute_tolerance": 0.01}
    }
    observations = [
        {
            "observation_id": "observation-0",
            "property": "temperature",
            "value": 298.0,
            "unit": "K",
        },
        {
            "observation_id": "observation-1",
            "property": "temperature",
            "value": 298.05,
            "unit": "K",
        },
    ]
    root = _root(
        tmp_path,
        observations,
        definitions=definitions,
        comparators=comparators,
        cases=2,
    )
    policy = _policy(definitions, comparators, gate_tolerance=(100.0,))
    state = BundleState.build(
        root,
        "property-demo",
        policy,
        {"mode": "new", "store_id": None},
    )
    policy = state.policy

    generation = AssessmentGeneration.build(
        root,
        state,
        policy,
        default_registry_snapshot(),
    )

    for case_id in ("case-0", "case-1"):
        assessment = generation.get_case(case_id)
        assert assessment["archive_class"] == "incomplete"
        assert assessment["conflict_assessments"][0]["reason"] == (
            "scientific_value_conflict"
        )
        assert assessment["conflict_assessments"][0]["allowed"] == pytest.approx(
            0.01
        )


def test_existing_dataset_requires_exact_property_policy() -> None:
    manifest = {
        "identity_schema": {},
        "required_structure_roles": [],
        "property_definitions": {"spin": {"type": "integer"}},
        "property_comparators": {"spin": {"kind": "exact"}},
    }
    matching = {
        "identity_schema": {},
        "required_structure_roles": [],
        "property_definitions": {"spin": {"type": "integer"}},
        "property_comparators": {"spin": {"kind": "exact"}},
    }
    _assert_manifest_policy(manifest, matching)

    differing = dict(matching)
    differing["property_definitions"] = {"spin": {"type": "real"}}
    with pytest.raises(ExistingDatasetPolicyError, match="property definitions"):
        _assert_manifest_policy(manifest, differing)

    differing = dict(matching)
    differing["property_comparators"] = {
        "spin": {"kind": "numeric", "absolute_tolerance": 1}
    }
    with pytest.raises(ExistingDatasetPolicyError, match="property comparators"):
        _assert_manifest_policy(manifest, differing)
