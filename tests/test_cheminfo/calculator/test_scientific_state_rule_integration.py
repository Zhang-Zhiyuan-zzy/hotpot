"""Resolver integration of dependency-layered optional rules."""

import json
from typing import Tuple

import hotpot
import pytest

from hotpot.cheminfo.calculator import resolve_scientific_state
from hotpot.cheminfo.calculator.charge import NetChargeCandidateEvidence
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    ScientificQuantity,
    StatePolicy,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.state_composition import (
    CandidateDomainScope,
    CandidateProposal,
    FactorKey,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
)
from hotpot.cheminfo.calculator.state_composition._decision_snapshot import (
    _decision_snapshot_from_json,
    _decision_snapshot_to_json,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)
from hotpot.cheminfo.calculator.state_rules import (
    InferenceRule,
    InferenceRuleRegistry,
    ResultType,
    RuleApplicability,
    RuleContext,
    RuleKind,
    RuleOutcome,
    RulePhase,
    RuleSet,
    RuleSpecification,
    RuleToggle,
    RuleValue,
)


class _ChargeRule(InferenceRule):
    specification = RuleSpecification(
        "OPTIONAL-TEST-CHARGE",
        "1.0.0",
        RuleKind.EMPIRICAL,
        RulePhase.CHARGE_EVIDENCE,
        (),
        (),
        (ResultType.CANDIDATE_PROPOSAL,),
        "test-v1",
    )

    def applicable(self, context: RuleContext) -> RuleApplicability:
        del context
        return RuleApplicability(True)

    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        del configuration
        subject = (
            "molecule:"
            f"{context.facts.identity.chemical_identity_digest}"
        )
        candidate = NetChargeCandidateEvidence(
            CandidateProvenance(
                "optional-test-charge:0",
                StateValueAuthority.DOMAIN_RULE,
                self.specification.rule_id,
            ),
            0,
        )
        proposal = CandidateProposal(
            "proposal:optional-test-charge",
            (
                ProposedCandidateFactor(
                    FactorKey(ScientificQuantity.NET_CHARGE, subject),
                    "optional-test-charge-domain-v1",
                    (ProposedCandidateValue(candidate),),
                    CandidateDomainScope.DOMAIN_BOUNDED,
                    MaterializationStatus.COMPLETE,
                    None,
                ),
            ),
            (),
        )
        return RuleOutcome(
            self.specification.rule_id,
            self.specification.rule_version,
            RuleApplicability(True),
            proposal,
            (),
            (),
            (),
            (),
        )


def _rule_set() -> RuleSet:
    registry = InferenceRuleRegistry()
    registry.register(_ChargeRule())
    return registry.activate(
        StatePolicy.COORDINATION_SCREENING_V1,
        (RuleToggle("OPTIONAL-TEST-CHARGE", True),),
    )


def test_resolver_composes_each_rule_layer_and_retains_trace() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("CO", "smi"))

    bundle = resolve_scientific_state(
        facts,
        StateRequest(net_charge=0, two_s=0),
        policy=StatePolicy.COORDINATION_SCREENING_V1,
        rule_set=_rule_set(),
    )

    assert tuple(trace.outcome.rule_id for trace in bundle.rule_trace) == (
        "OPTIONAL-TEST-CHARGE",
    )
    assert any(
        contribution.proposal_id == "proposal:optional-test-charge"
        for factor in bundle.candidate_space.factors
        for contribution in factor.domain_contributions
    )


def test_strict_resolver_rejects_a_nonempty_rule_set() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("CO", "smi"))

    with pytest.raises(ValueError, match="STRICT"):
        resolve_scientific_state(facts, rule_set=_rule_set())


def test_resolver_rule_trace_survives_decision_snapshot() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("CO", "smi"))
    bundle = resolve_scientific_state(
        facts,
        StateRequest(net_charge=0, two_s=0),
        policy=StatePolicy.COORDINATION_SCREENING_V1,
        rule_set=_rule_set(),
    )

    snapshot = _decision_snapshot_to_json(
        bundle.candidate_space,
        bundle.scientific_resolution,
        bundle.execution_decision,
        bundle.rule_trace,
    )
    _, _, _, rule_trace = _decision_snapshot_from_json(snapshot)

    assert rule_trace == bundle.rule_trace


def test_snapshot_rejects_a_trace_detached_from_candidate_space() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("CO", "smi"))
    bundle = resolve_scientific_state(
        facts,
        StateRequest(net_charge=0, two_s=0),
        policy=StatePolicy.COORDINATION_SCREENING_V1,
        rule_set=_rule_set(),
    )
    payload = json.loads(
        _decision_snapshot_to_json(
            bundle.candidate_space,
            bundle.scientific_resolution,
            bundle.execution_decision,
            bundle.rule_trace,
        )
    )
    payload["rule_trace"][0]["outcome"]["candidate_proposal"][
        "proposal_id"
    ] = "detached"

    with pytest.raises(ValueError, match="absent from candidate factors"):
        _decision_snapshot_from_json(json.dumps(payload))
