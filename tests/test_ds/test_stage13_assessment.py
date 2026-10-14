from hotpot.ds.gates import BenchmarkGate
from hotpot.ds.public._assessment import (
    assess_case,
    assess_evidence,
    compare_observations,
    find_conflicts,
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
            "locator": "p.1",
            "scientific_work_id": "doi:one",
            "source_lineage_id": "result:one",
        },
        {
            "evidence_id": "e2",
            "target": target,
            "locator": "p.2",
            "scientific_work_id": "doi:two",
            "source_lineage_id": "result:two",
        },
    ]
    case["verifications"] = [
        {"verification_id": "v1", "evidence_id": "e1", "method": "Human"},
        {"verification_id": "v2", "evidence_id": "e2", "method": "Human"},
    ]
    assert assess_evidence(case)["aggregate"] == "E4"
    case["evidence"][1]["source_lineage_id"] = "result:one"
    result = assess_evidence(case)
    assert result["aggregate"] == "E3"
    assert result["reasons"][0]["code"] == "evidence_independence_unproven"


def test_numeric_conflict_requires_comparable_context_and_tolerance():
    left = {"property": "p", "value": 1.0, "phase": "solid"}
    right = {"property": "p", "value": 1.05, "phase": "solid"}
    result = compare_observations(left, right, {"p": {"absolute": 0.1}})
    assert result["outcome"] == "consistent"
    right["value"] = 1.2
    assert compare_observations(left, right, {"p": {"absolute": 0.1}})["outcome"] == "conflict"
    right["phase"] = "solution"
    assert compare_observations(left, right, {"p": {"absolute": 0.1}})["outcome"] == "indeterminate"


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
