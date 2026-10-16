"""Versioned lossless serialization of factorized candidate spaces."""

import json
from dataclasses import replace

import pytest

from hotpot.cheminfo.calculator.charge import NetChargeCandidateEvidence
from hotpot.cheminfo.calculator.contracts import (
    CandidateBudget,
    CandidateProvenance,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.oxidation_state import CountingConvention
from hotpot.cheminfo.calculator.state_composition import (
    BranchAssignment,
    CandidateConstraint,
    CandidateDomainScope,
    CandidateDomainSummary,
    CandidateFactor,
    CandidateSearchStatus,
    CoupledScientificStateCandidate,
    DerivedStateValue,
    FactorCandidateValue,
    FactorDomainContribution,
    FactorKey,
    FrontierBound,
    MaterializationStatus,
    RejectedPathSummary,
    ScientificStateCandidateSpace,
    ScientificStateIdentity,
    StateValueOrigin,
    StateValueRef,
)
from hotpot.cheminfo.calculator.state_composition._artifacts import (
    _FRONTIER_SCHEMA,
    _candidate_budget_artifact_id,
)
from hotpot.cheminfo.calculator.state_composition._candidate_space_serialization import (
    _candidate_space_from_json,
    _candidate_space_to_json,
)


def _candidate_space() -> ScientificStateCandidateSpace:
    budget = CandidateBudget(2, 3, 4)
    provenance = CandidateProvenance(
        "charge:1",
        StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
        "serialization-test",
        ("evidence",),
        ("assumption",),
    )
    candidate = NetChargeCandidateEvidence(provenance, 1)
    factor_key = FactorKey(ScientificQuantity.NET_CHARGE, "molecule:test")
    frontier = FrontierBound("branch:1", _FRONTIER_SCHEMA, ("0", "source"))
    summary = CandidateDomainSummary(2, "continuation:1", frontier)
    derived = DerivedStateValue(
        CandidateProvenance(
            "derived:charge",
            StateValueAuthority.DERIVED_CONSTRAINT,
            "HARD-002",
            ("oxidation",),
            (),
        ),
        ScientificQuantity.NET_CHARGE,
        1,
        "HARD-002.v1",
        ("oxidation",),
        CountingConvention.IONIC,
    )
    return ScientificStateCandidateSpace(
        factors=(
            CandidateFactor(
                identity="factor:charge",
                factor_key=factor_key,
                domain_definition_id="charge-domain-v1",
                values=(
                    FactorCandidateValue(
                        candidate,
                        ("proposal:charge",),
                        (BranchAssignment("group:1", "option:1"),),
                    ),
                ),
                domain_contributions=(
                    FactorDomainContribution(
                        "proposal:charge",
                        CandidateDomainScope.DOMAIN_BOUNDED,
                        MaterializationStatus.RESOURCE_TRUNCATED,
                        summary,
                    ),
                ),
                unexpanded_domain_summary=summary,
                domain_scope=CandidateDomainScope.DOMAIN_BOUNDED,
                materialization_status=(
                    MaterializationStatus.RESOURCE_TRUNCATED
                ),
            ),
        ),
        constraints=(
            CandidateConstraint(
                "constraint:1",
                "test_constraint",
                "test-constraint-v1",
                (factor_key,),
                (
                    (
                        "nested",
                        (None, True, 1, 1.5, "value", (2,)),
                    ),
                ),
            ),
        ),
        candidate_budget=budget,
        candidate_budget_artifact_id=_candidate_budget_artifact_id(budget),
        frontier_bound_schema_version=_FRONTIER_SCHEMA,
        expanded_paths=(
            CoupledScientificStateCandidate(
                ScientificStateIdentity("state:1"),
                StateValueRef(
                    ScientificQuantity.NET_CHARGE,
                    1,
                    StateValueOrigin.RESULT,
                    "charge:1",
                ),
                ("component:1",),
                ("coordination:1",),
                "oxidation:1",
                "electronic-result:1",
                "electronic-candidate:1",
                (derived,),
                ("evidence",),
                ("assumption",),
            ),
        ),
        rejected_path_summaries=(
            RejectedPathSummary(
                ("charge:2",),
                "constraint:1",
                ("HARD-002",),
            ),
        ),
        frontier=(frontier,),
        search_status=(
            CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION
        ),
    )


def test_candidate_space_json_round_trip_is_lossless_and_canonical() -> None:
    candidate_space = _candidate_space()

    serialized = _candidate_space_to_json(candidate_space)
    restored = _candidate_space_from_json(serialized)

    assert restored == candidate_space
    assert _candidate_space_to_json(restored) == serialized


def test_candidate_space_rejects_unknown_snapshot_schema() -> None:
    payload = json.loads(_candidate_space_to_json(_candidate_space()))
    payload["schema_version"] = "unknown-v0"

    with pytest.raises(ValueError, match="Unsupported candidate-space"):
        _candidate_space_from_json(json.dumps(payload))


def test_candidate_space_rejects_tampered_budget_identity() -> None:
    candidate_space = replace(
        _candidate_space(),
        candidate_budget_artifact_id="tampered",
    )

    with pytest.raises(ValueError, match="budget artifact"):
        _candidate_space_to_json(candidate_space)


def test_candidate_space_rejects_nested_schema_drift() -> None:
    payload = json.loads(_candidate_space_to_json(_candidate_space()))
    payload["factors"][0]["unexpected"] = True

    with pytest.raises(ValueError, match="schema mismatch"):
        _candidate_space_from_json(json.dumps(payload))
