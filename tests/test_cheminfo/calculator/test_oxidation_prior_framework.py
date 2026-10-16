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


def _facts(smiles="[Eu].N", coordination_bonds=((0, 1),)):
    mol = hotpot.read_mol(smiles, "smi")
    for begin, end in coordination_bonds:
        mol.add_bond(begin, end, bond_kind=BondKind.DATIVE)
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


def test_prior_emits_one_independent_factor_per_metal_center() -> None:
    facts = _facts(
        "[Eu].N.[Eu].N",
        ((0, 1), (2, 3)),
    )
    outcome = _OxidationStatePriorRule(_artifact()).evaluate(
        RuleToggle("OPTIONAL-TEST-OS", True),
        _context(facts),
    )

    factors = outcome.candidate_proposal.factors
    assert tuple(factor.factor_key.subject_identity for factor in factors) == (
        f"atom:{facts.atom_map[0]}",
        f"atom:{facts.atom_map[2]}",
    )
    assert all(
        {
            next(
                state
                for state in value.candidate.atom_states
                if state is not None
            )
            for value in factor.materialized_candidates
        }
        == {2, 3}
        for factor in factors
    )
    assert all(
        sum(state is not None for state in value.candidate.atom_states) == 1
        for factor in factors
        for value in factor.materialized_candidates
    )
    assert len(outcome.rank_adjustments) == 2


def test_each_center_has_an_independent_truncated_domain_summary() -> None:
    facts = _facts(
        "[Eu].N.[Eu].N",
        ((0, 1), (2, 3)),
    )
    outcome = _OxidationStatePriorRule(_artifact()).evaluate(
        RuleToggle("OPTIONAL-TEST-OS", True),
        _context(facts, CandidateBudget(1, 32, 16)),
    )

    factors = outcome.candidate_proposal.factors
    summaries = tuple(factor.unexpanded_domain_summary for factor in factors)
    assert all(summary.remaining_count_upper_bound == 1 for summary in summaries)
    assert len({summary.continuation_identity for summary in summaries}) == 2


def test_different_local_prior_families_remain_composable() -> None:
    facts = _facts(
        "[Eu].N.[Th].N",
        ((0, 1), (2, 3)),
    )
    context = _context(facts)
    europium = _OxidationStatePriorRule(_artifact()).evaluate(
        RuleToggle("OPTIONAL-TEST-OS", True),
        context,
    )
    thorium_artifact = _OxidationPriorArtifact(
        rule_id="OPTIONAL-TEST-TH",
        rule_version="1.0.0",
        source_identity="test-th-prior",
        source_locator="test://th-prior",
        content_digest="test-th-prior-digest",
        applicability_schema="test-th-prior-v1",
        elements=(_ElementOxidationPrior(90, (4,), 4),),
    )
    thorium = _OxidationStatePriorRule(thorium_artifact).evaluate(
        RuleToggle("OPTIONAL-TEST-TH", True),
        context,
    )

    candidate_space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        candidate_proposals=(
            europium.candidate_proposal,
            thorium.candidate_proposal,
        ),
    )

    oxidation_factors = tuple(
        factor
        for factor in candidate_space.factors
        if factor.factor_key.quantity is ScientificQuantity.OXIDATION_STATE
    )
    assert tuple(
        factor.factor_key.subject_identity for factor in oxidation_factors
    ) == (
        f"atom:{facts.atom_map[0]}",
        f"atom:{facts.atom_map[2]}",
    )
