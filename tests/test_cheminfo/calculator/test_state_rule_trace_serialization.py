"""Lossless serialization of complete optional-rule traces."""

import json

from hotpot.cheminfo.calculator.charge import NetChargeCandidateEvidence
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.state_composition import (
    BranchAssignment,
    CandidateConstraint,
    CandidateDomainScope,
    CandidateDomainSummary,
    CandidateProposal,
    ConstraintGuard,
    FactorKey,
    FrontierBound,
    GuardedCandidateConstraint,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
)
from hotpot.cheminfo.calculator.state_rules import (
    EvidenceItem,
    RankAdjustment,
    RuleApplicability,
    RuleOutcome,
    RulePhase,
    RuleTrace,
)
from hotpot.cheminfo.calculator.state_rules._serialization import (
    _rule_traces_from_payload,
    _rule_traces_to_payload,
)


def _trace() -> RuleTrace:
    candidate = NetChargeCandidateEvidence(
        CandidateProvenance(
            "charge:0",
            StateValueAuthority.DOMAIN_RULE,
            "OPTIONAL-TEST",
            ("evidence:1",),
            ("assumption:1",),
        ),
        0,
    )
    factor_key = FactorKey(ScientificQuantity.NET_CHARGE, "molecule:test")
    factor = ProposedCandidateFactor(
        factor_key,
        "charge-domain-v1",
        (
            ProposedCandidateValue(
                candidate,
                (BranchAssignment("group:1", "option:1"),),
            ),
        ),
        CandidateDomainScope.DOMAIN_BOUNDED,
        MaterializationStatus.RESOURCE_TRUNCATED,
        CandidateDomainSummary(
            1,
            "continuation:1",
            FrontierBound(
                "branch:1",
                "scientific-state-frontier-v1",
                ("best",),
            ),
        ),
    )
    constraint = GuardedCandidateConstraint(
        ConstraintGuard(
            (candidate.provenance.identity,),
            (BranchAssignment("group:1", "option:1"),),
        ),
        CandidateConstraint(
            "constraint:1",
            "requires_all",
            "requires-all-v1",
            (factor_key,),
            (("candidate_ids", (candidate.provenance.identity,)),),
        ),
    )
    proposal = CandidateProposal("proposal:1", (factor,), (constraint,))
    outcome = RuleOutcome(
        "OPTIONAL-TEST",
        "1.0.0",
        RuleApplicability(True, ("IN_DOMAIN",)),
        proposal,
        (
            EvidenceItem(
                "evidence:1",
                StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
                "source:1",
                "test claim",
                "doi:test",
            ),
        ),
        (RankAdjustment("charge:0", "authority", 1.25, "test"),),
        ("assumption:1",),
        ("warning:1",),
    )
    return RuleTrace(RulePhase.CHARGE_EVIDENCE, outcome)


def test_complete_rule_trace_crosses_a_real_json_boundary() -> None:
    traces = (_trace(),)

    payload = json.loads(
        json.dumps(_rule_traces_to_payload(traces), allow_nan=False)
    )

    assert _rule_traces_from_payload(payload) == traces
