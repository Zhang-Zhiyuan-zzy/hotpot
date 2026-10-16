"""Shared oxidation-state prior mechanics."""

import hotpot

from hotpot.cheminfo.calculator.contracts import (
    CandidateBudget,
    ScientificQuantity,
    StatePolicy,
)
from hotpot.cheminfo.calculator.operation_requirements import (
    _effective_requirements,
)
from hotpot.cheminfo.calculator.state_composition import (
    compose_scientific_state_space,
)
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts
from hotpot.cheminfo.calculator.state_rules import (
    RuleContext,
    RulePhase,
    RuleToggle,
)
from hotpot.cheminfo.calculator.state_rules.neutral_l_donors import (
    NeutralLDonorRule,
)
from hotpot.cheminfo.calculator.state_rules.oxidation_priors import (
    _ElementOxidationPrior,
    _OxidationPriorArtifact,
    _OxidationStatePriorRule,
)
from hotpot.cheminfo.core import BondKind


def _facts():
    mol = hotpot.read_mol("[Eu].N", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    return extract_molecular_facts(mol)


def _artifact():
    return _OxidationPriorArtifact(
        rule_id="OPTIONAL-TEST-OS",
        rule_version="1.0.0",
        source_identity="test-os-prior",
        source_locator="test://os-prior",
        content_digest="test-os-prior-digest",
        applicability_schema="test-os-prior-v1",
        elements=(_ElementOxidationPrior(63, (2, 3), 3),),
    )


def _context(facts, budget=CandidateBudget(8, 32, 16)):
    requirements = _effective_requirements(
        StatePolicy.COORDINATION_SCREENING_V1,
        None,
    )
    donor_context = RuleContext(
        facts,
        RulePhase.COORDINATION_COUNTING,
        requirements,
        None,
        (),
        budget,
    )
    donor = NeutralLDonorRule().evaluate(
        RuleToggle("OPTIONAL-003", True),
        donor_context,
    )
    candidate_space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        candidate_proposals=(donor.candidate_proposal,),
        candidate_budget=budget,
    )
    return RuleContext(
        facts,
        RulePhase.OXIDATION_STATE_CANDIDATE,
        requirements,
        candidate_space,
        (donor,),
        budget,
    )


def test_prior_requires_local_ionic_counting_evidence() -> None:
    facts = _facts()
    empty_context = RuleContext(
        facts,
        RulePhase.OXIDATION_STATE_CANDIDATE,
        _effective_requirements(StatePolicy.COORDINATION_SCREENING_V1, None),
        None,
        (),
        CandidateBudget(8, 32, 16),
    )
    outcome = _OxidationStatePriorRule(_artifact()).evaluate(
        RuleToggle("OPTIONAL-TEST-OS", True),
        empty_context,
    )

    assert not outcome.applicability.matches


def test_prior_retains_alternatives_and_ranks_only_common_state() -> None:
    facts = _facts()
    outcome = _OxidationStatePriorRule(_artifact()).evaluate(
        RuleToggle("OPTIONAL-TEST-OS", True),
        _context(facts),
    )

    assert outcome.applicability.matches
    factor = outcome.candidate_proposal.factors[0]
    assert factor.factor_key.quantity is ScientificQuantity.OXIDATION_STATE
    assert {
        value.candidate.atom_states[0]
        for value in factor.materialized_candidates
    } == {2, 3}
    assert len(outcome.rank_adjustments) == 1
    ranked_id = outcome.rank_adjustments[0].target_identity
    assert next(
        value.candidate.atom_states[0]
        for value in factor.materialized_candidates
        if value.candidate.provenance.identity == ranked_id
    ) == 3


def test_prior_marks_budgeted_domain_as_resource_truncated() -> None:
    facts = _facts()
    outcome = _OxidationStatePriorRule(_artifact()).evaluate(
        RuleToggle("OPTIONAL-TEST-OS", True),
        _context(facts, CandidateBudget(1, 32, 16)),
    )

    factor = outcome.candidate_proposal.factors[0]
    assert len(factor.materialized_candidates) == 1
    assert factor.unexpanded_domain_summary.remaining_count_upper_bound == 1
