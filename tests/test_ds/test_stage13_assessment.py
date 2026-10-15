import pytest

from hotpot.ds.gates import BenchmarkGate
from hotpot.ds.public._assessment import (
    assess_case,
    assess_evidence,
    compare_observations,
    find_conflicts,
    sample_match,
    validate_identity_schema,
)


def _case(case_id="c1", value=2):
    observation_id = f"{case_id}:o1"
    return {
        "id": case_id,
        "local_id": case_id,
        "identifiers": {"sample": "same"},
        "metadata": {"supplier": "test"},
        "core_info": {"phase": "solid"},
        "structures": [],
        "observations": [
            {
                "observation_id": observation_id,
                "property": "oxidation_state",
                "value": value,
                "phase": "solid",
                "status": "active",
            }
        ],
        "claims": [],
        "evidence": [],
        "verifications": [],
        "attestations": [],
    }


def test_evidence_requires_independent_verified_lineage_for_e4():
    case = _case()
    target = "c1:o1"
    case["evidence"] = [
        {
            "evidence_id": "e1",
            "target": target,
            "relation": "supports",
            "locator": "p.1",
            "scientific_work_id": "doi:one",
            "source_lineage_id": "result:one",
        },
        {
            "evidence_id": "e2",
            "target": target,
            "relation": "supports",
            "locator": "p.2",
            "scientific_work_id": "doi:two",
            "source_lineage_id": "result:two",
        },
    ]
    case["verifications"] = [
        {
            "verification_id": "v1",
            "evidence_id": "e1",
            "method": "Human",
            "verifier": "reviewer-one",
        },
        {
            "verification_id": "v2",
            "evidence_id": "e2",
            "method": "Human",
            "verifier": "reviewer-two",
        },
    ]
    assert assess_evidence(case)["aggregate"] == "E4"
    case["evidence"][1]["source_lineage_id"] = "result:one"
    result = assess_evidence(case)
    assert result["aggregate"] == "E3"
    assert result["reasons"][0]["code"] == "evidence_independence_unproven"


def test_human_verification_requires_an_attributable_reviewer():
    case = _case()
    case["evidence"] = [
        {
            "evidence_id": "e1",
            "target": "c1:o1",
            "relation": "supports",
            "locator": "p.1",
            "scientific_work_id": "doi:one",
        }
    ]
    case["verifications"] = [
        {"verification_id": "v1", "evidence_id": "e1", "method": "Human"}
    ]

    assert assess_evidence(case)["aggregate"] == "E2"


def test_contradictory_evidence_never_counts_as_support():
    case = _case()
    case["evidence"] = [
        {
            "evidence_id": "e1",
            "target": "c1:o1",
            "relation": "contradicts",
            "locator": "p.1",
            "doi": "10.1/example",
            "source_lineage_id": "result:one",
        }
    ]
    case["verifications"] = [
        {
            "verification_id": "v1",
            "evidence_id": "e1",
            "method": "Human",
            "verifier": "reviewer",
        }
    ]

    result = assess_evidence(case)

    assert result["aggregate"] == "E0"
    assert result["levels"] == {"c1:o1": "E0"}


def test_unambiguous_property_name_resolves_an_evidence_target():
    case = _case()
    case["evidence"] = [
        {
            "evidence_id": "e1",
            "target": "oxidation_state",
            "relation": "supports",
            "locator": "p.1",
            "doi": "10.1/example",
            "source_lineage_id": "result:one",
        }
    ]
    case["verifications"] = [
        {
            "verification_id": "v1",
            "evidence_id": "e1",
            "method": "Human",
            "verifier": "reviewer",
        }
    ]

    assert assess_evidence(case)["levels"] == {"c1:o1": "E3"}


def test_work_identity_normalization_prevents_false_e4():
    case = _case()
    case["evidence"] = [
        {
            "evidence_id": "e1",
            "target": "c1:o1",
            "relation": "supports",
            "locator": "p.1",
            "doi": "10.1000/SAME",
            "source_lineage_id": "result:one",
        },
        {
            "evidence_id": "e2",
            "target": "c1:o1",
            "relation": "supports",
            "locator": "p.2",
            "scientific_work_id": "https://doi.org/10.1000/same",
            "source_lineage_id": "result:two",
        },
    ]
    case["verifications"] = [
        {
            "verification_id": f"v{index}",
            "evidence_id": f"e{index}",
            "method": "Human",
            "verifier": f"reviewer-{index}",
        }
        for index in (1, 2)
    ]

    assert assess_evidence(case)["aggregate"] == "E3"


def test_derived_result_lineage_resolves_by_lineage_id_and_is_not_independent():
    case = _case()
    case["evidence"] = [
        {
            "evidence_id": "e1",
            "target": "c1:o1",
            "relation": "supports",
            "locator": "p.1",
            "doi": "10.1000/one",
            "source_lineage_id": "result:root",
        },
        {
            "evidence_id": "e2",
            "target": "c1:o1",
            "relation": "supports",
            "locator": "p.2",
            "doi": "10.1000/two",
            "source_lineage_id": "result:derived",
            "derived_from": ["result:root"],
        },
    ]
    case["verifications"] = [
        {
            "verification_id": f"v{index}",
            "evidence_id": f"e{index}",
            "method": "Human",
            "verifier": f"reviewer-{index}",
        }
        for index in (1, 2)
    ]

    assert assess_evidence(case)["aggregate"] == "E3"


def test_empty_core_policy_aggregates_all_active_targets():
    result = assess_case(
        _case(),
        core_labels=[],
        core_info=[],
        required_structure_roles=[],
        asset_assessments=[],
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
    )

    assert result["evidence_level"] == "E0"


def test_missing_core_target_is_e0_not_not_applicable():
    result = assess_case(
        _case(),
        core_labels=["spin"],
        core_info=[],
        required_structure_roles=[],
        asset_assessments=[],
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
    )

    assert result["evidence_level"] == "E0"
    assert any(
        reason["code"] == "expected_evidence_target_missing"
        for reason in result["reasons"]
    )


def test_identity_indeterminate_blocks_direct_entry_but_remains_ordinary():
    case = _case()
    case["attestations"] = [
        {
            "attestation_id": "a1",
            "authority": "User",
            "actor": "owner",
            "approved": True,
        }
    ]
    result = assess_case(
        case,
        core_labels=["oxidation_state"],
        core_info=["phase"],
        required_structure_roles=[],
        asset_assessments=[],
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
        identity_reasons=[
            {"code": "identity_indeterminate", "message": "missing sample key"}
        ],
    )

    assert result["archive_class"] == "ordinary"
    assert result["completeness"] == "complete"
    assert result["benchmark_status"] == "vetoed"
    assert result["identity_determinate"] is False


def test_broken_case_relations_are_incomplete_before_direct_entry():
    case = _case()
    case["evidence"] = [
        {
            "evidence_id": "e1",
            "target": "absent-observation",
            "relation": "supports",
        }
    ]
    case["attestations"] = [
        {
            "attestation_id": "a1",
            "authority": "User",
            "actor": "owner",
            "approved": True,
        }
    ]
    result = assess_case(
        case,
        core_labels=["oxidation_state"],
        core_info=["phase"],
        required_structure_roles=[],
        asset_assessments=[],
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
    )

    assert result["archive_class"] == "incomplete"
    assert result["benchmark"] is False
    assert any(
        reason["code"] == "evidence_target_unresolved"
        for reason in result["reasons"]
    )


def test_ambiguous_property_target_is_an_integrity_failure():
    case = _case()
    case["observations"].append(
        {
            "observation_id": "c1:o2",
            "property": "oxidation_state",
            "value": 3,
            "phase": "solid",
            "status": "active",
        }
    )
    case["evidence"] = [
        {
            "evidence_id": "e1",
            "target": "oxidation_state",
            "relation": "supports",
        }
    ]
    result = assess_case(
        case,
        core_labels=["oxidation_state"],
        core_info=["phase"],
        required_structure_roles=[],
        asset_assessments=[],
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
    )

    assert result["archive_class"] == "incomplete"
    assert any(
        reason["code"] == "evidence_target_ambiguous"
        for reason in result["reasons"]
    )


def test_unresolved_result_lineage_is_an_integrity_failure():
    case = _case()
    case["evidence"] = [
        {
            "evidence_id": "e1",
            "target": "c1:o1",
            "relation": "supports",
            "source_lineage_id": "result:derived",
            "derived_from": ["result:missing"],
        }
    ]
    result = assess_case(
        case,
        core_labels=["oxidation_state"],
        core_info=["phase"],
        required_structure_roles=[],
        asset_assessments=[],
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
    )

    assert result["archive_class"] == "incomplete"
    assert any(
        reason["code"] == "evidence_lineage_unresolved"
        for reason in result["reasons"]
    )


def test_list_supersession_is_a_valid_attestation_relation():
    case = _case()
    case["attestations"] = [
        {
            "attestation_id": "old-1",
            "authority": "User",
            "actor": "owner",
            "approved": False,
            "status": "superseded",
            "superseded_by": "new",
        },
        {
            "attestation_id": "old-2",
            "authority": "User",
            "actor": "owner",
            "approved": False,
            "status": "superseded",
            "superseded_by": "new",
        },
        {
            "attestation_id": "new",
            "authority": "User",
            "actor": "owner",
            "approved": True,
            "status": "active",
            "supersedes": ["old-1", "old-2"],
        },
    ]
    result = assess_case(
        case,
        core_labels=["oxidation_state"],
        core_info=["phase"],
        required_structure_roles=[],
        asset_assessments=[],
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
    )

    assert result["benchmark"] is True
    assert not any(
        reason["code"].startswith("supersession_")
        for reason in result["reasons"]
    )


def test_numeric_conflict_requires_comparable_context_and_tolerance():
    left = {"property": "p", "value": 1.0, "phase": "solid"}
    right = {"property": "p", "value": 1.05, "phase": "solid"}
    result = compare_observations(left, right, {"p": {"absolute": 0.1}})
    assert result["outcome"] == "consistent"
    right["value"] = 1.2
    assert compare_observations(left, right, {"p": {"absolute": 0.1}})["outcome"] == "conflict"
    right["phase"] = "solution"
    result = compare_observations(left, right, {"p": {"absolute": 0.1}})
    assert result["outcome"] == "noncomparable"
    assert result["reason"] == "observation_context_noncomparable"


def test_same_sample_conflict_propagates_to_both_cases():
    first = _case("c1", 2)
    second = _case("c2", 3)
    result = find_conflicts(
        [first, second],
        {"components": [{"key": "sample", "path": "identifiers.sample", "required": True}]},
        {"oxidation_state": {"absolute": 0.0, "relative": 0.0}},
    )
    assert result["conflicting_cases"] == ["c1", "c2"]
    assert result["exceptions"][0]["reason"] == "scientific_value_conflict"


def test_incomplete_vetoes_direct_user_benchmark_approval():
    case = _case()
    case["attestations"] = [
        {
            "attestation_id": "a1",
            "authority": "User",
            "actor": "owner",
            "approved": True,
        }
    ]
    result = assess_case(
        case,
        core_labels=["oxidation_state"],
        core_info=["phase"],
        required_structure_roles=["reported"],
        asset_assessments=[],
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
    )
    assert result["archive_class"] == "incomplete"
    assert not result["benchmark"]


def test_user_approval_alone_admits_complete_case():
    case = _case()
    case["attestations"] = [
        {
            "attestation_id": "a1",
            "authority": "User",
            "actor": "owner",
            "approved": True,
        }
    ]
    result = assess_case(
        case,
        core_labels=["oxidation_state"],
        core_info=["phase"],
        required_structure_roles=[],
        asset_assessments=[],
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
    )
    assert result["archive_class"] == "benchmark"
    assert result["benchmark_status"] == "admitted"


def _assess_with_any_structure(case, asset_assessments):
    return assess_case(
        case,
        core_labels=["oxidation_state"],
        core_info=["phase"],
        required_structure_roles=["*"],
        asset_assessments=asset_assessments,
        conflicting=False,
        benchmark_policy=BenchmarkGate().info,
    )


def test_any_structure_requirement_rejects_missing_structure():
    result = _assess_with_any_structure(_case(), [])

    assert result["archive_class"] == "incomplete"
    declared_role = result["gate_results"]["structure"]["declared_role_valid"]
    assert declared_role["passed"] is False
    assert any(
        reason["code"] == "missing_required_structure"
        and reason["details"]["role"] == "*"
        for reason in result["reasons"]
    )


def test_any_structure_requirement_rejects_invalid_representation():
    case = _case()
    case["structures"] = [{"structure_id": "s1", "role": "reported"}]
    result = _assess_with_any_structure(
        case,
        [
            {
                "structure_id": "s1",
                "presence": "present",
                "hash_result": "match",
                "parse_result": "invalid",
            }
        ],
    )

    assert result["archive_class"] == "incomplete"
    declared_role = result["gate_results"]["structure"]["declared_role_valid"]
    assert declared_role["passed"] is False
    assert any(
        reason["code"] == "invalid_required_structure"
        and reason["details"]["role"] == "*"
        for reason in result["reasons"]
    )


def test_any_structure_requirement_accepts_valid_representation_for_any_role():
    case = _case()
    case["structures"] = [{"structure_id": "s1", "role": "optimized"}]
    result = _assess_with_any_structure(
        case,
        [
            {
                "structure_id": "s1",
                "presence": "present",
                "hash_result": "match",
                "parse_result": "valid",
            }
        ],
    )

    assert result["completeness"] == "complete"
    declared_role = result["gate_results"]["structure"]["declared_role_valid"]
    assert declared_role["passed"] is True
    assert not any(
        reason["code"] in {"missing_required_structure", "invalid_required_structure"}
        for reason in result["reasons"]
    )


def test_identity_schema_is_closed_and_materializes_current_protocol():
    with pytest.raises(ValueError, match="only components"):
        validate_identity_schema({"fields": ["core_info.sample"]})
    with pytest.raises(ValueError, match="core_info or identifiers"):
        validate_identity_schema(
            {
                "components": [
                    {"key": "grade", "path": "claimed_benchmark", "required": True}
                ]
            }
        )

    schema = validate_identity_schema(
        {
            "components": [
                {
                    "key": "sample",
                    "path": "identifiers.sample",
                    "required": True,
                    "normalizer": "casefold",
                }
            ]
        }
    )
    component = schema["components"][0]
    assert component["value_kind"] == "text"
    assert component["comparator"] == "exact"
    assert component["mismatch_is_decisive"] is True
    assert len(component["implementation_digest"]) == 64
    assert len(component["protocol_digest"]) == 64


def test_sample_match_is_three_valued_and_case_reflexive():
    schema = {
        "components": [
            {
                "key": "sample",
                "path": "identifiers.sample",
                "required": True,
                "normalizer": "casefold",
            }
        ]
    }
    left = _case("left")
    right = _case("right")
    assert sample_match(left, right, schema)["outcome"] == "same"
    right["identifiers"]["sample"] = "other"
    assert sample_match(left, right, schema)["outcome"] == "different"
    right["identifiers"].clear()
    assert sample_match(left, right, schema)["outcome"] == "indeterminate"
    assert sample_match(left, left, {"components": []})["outcome"] == "same"


def test_missing_cross_case_identity_does_not_hide_same_case_conflict():
    case = _case("same-case", 1)
    case["identifiers"].clear()
    case["observations"].append(
        {
            "observation_id": "same-case:o2",
            "property": "oxidation_state",
            "value": 2,
            "phase": "solid",
            "status": "active",
        }
    )
    result = find_conflicts(
        [case],
        {
            "components": [
                {
                    "key": "sample",
                    "path": "identifiers.sample",
                    "required": True,
                    "normalizer": "text",
                }
            ]
        },
        {},
        property_definitions={"oxidation_state": {"type": "integer"}},
        property_comparators={"oxidation_state": {"kind": "exact"}},
    )

    assert result["conflicting_cases"] == ["same-case"]
    assert result["exceptions"][0]["reason"] == "scientific_value_conflict"
    assert result["identity_reasons"]["same-case"][0]["code"] == "identity_indeterminate"


def test_structure_identity_component_selects_one_declared_role():
    schema = {
        "components": [
            {
                "key": "primary_graph",
                "path": "metadata.inchi_key",
                "structure_role": "primary",
                "required": True,
                "normalizer": "casefold",
            }
        ]
    }
    left = _case("left")
    right = _case("right")
    for case in (left, right):
        case["structures"] = [
            {
                "structure_id": f"{case['id']}:structure",
                "role": "primary",
                "metadata": {"inchi_key": "ABC-DEF"},
            }
        ]

    assert sample_match(left, right, schema)["outcome"] == "same"
    right["structures"][0]["metadata"]["inchi_key"] = "XYZ-DEF"
    assert sample_match(left, right, schema)["outcome"] == "different"
    right["structures"].append(
        {
            "structure_id": "right:second",
            "role": "primary",
            "metadata": {"inchi_key": "XYZ-DEF"},
        }
    )
    assert sample_match(left, right, schema)["outcome"] == "indeterminate"


def test_cross_case_subject_ids_are_resolved_before_value_comparison():
    left = _case("left", 1)
    right = _case("right", 2)
    for case, entity_id in ((left, "left:metal"), (right, "right:metal")):
        case["entities"] = [
            {
                "entity_id": entity_id,
                "external_identifiers": [
                    {"namespace": "chemical", "value": "eu-center"}
                ],
            }
        ]
        case["observations"][0]["subject_entity_id"] = entity_id
    result = find_conflicts(
        [left, right],
        {
            "components": [
                {
                    "key": "sample",
                    "path": "identifiers.sample",
                    "required": True,
                    "normalizer": "text",
                }
            ]
        },
        {},
        property_definitions={
            "oxidation_state": {
                "type": "integer",
                "context_fields": ["subject", "phase"],
            }
        },
        property_comparators={"oxidation_state": {"kind": "exact"}},
    )

    assert result["conflicting_cases"] == ["left", "right"]
    assert result["exceptions"][0]["reason"] == "scientific_value_conflict"
