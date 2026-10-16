"""OPTIONAL-011 localized f-block Hund-prior tests."""

import hotpot

from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    CoverageStatus,
    DEFAULT_CANDIDATE_BUDGET,
    ScientificQuantity,
    StatePolicy,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.operation_requirements import (
    _effective_requirements,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CountingConvention,
    OxidationStateCandidateEvidence,
)
from hotpot.cheminfo.calculator.state_composition import (
    CandidateDomainScope,
    CandidateProposal,
    FactorKey,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
    compose_scientific_state_space,
)
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts
from hotpot.cheminfo.calculator.state_rules import (
    RuleContext,
    RulePhase,
    RuleToggle,
)
from hotpot.cheminfo.calculator.state_rules.f_block_hund import (
    FBlockHundPriorRule,
)
from hotpot.cheminfo.calculator.state_rules.ionic_configuration_spin import (
    IonicConfigurationSpinRule,
)


def _spin_context(smiles: str, oxidation_state: int):
    facts = extract_molecular_facts(hotpot.read_mol(smiles, "smi"))
    requirements = _effective_requirements(
        StatePolicy.COORDINATION_SCREENING_V1,
        None,
    )
    oxidation = OxidationStateCandidateEvidence(
        CandidateProvenance(
            f"os:{facts.atomic_numbers[0]}:{oxidation_state}",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "hund-test",
        ),
        CountingConvention.IONIC,
        (oxidation_state,),
        CoverageStatus.COMPLETE,
    )
    oxidation_proposal = CandidateProposal(
        "proposal:hund-os",
        (
            ProposedCandidateFactor(
                FactorKey(
                    ScientificQuantity.OXIDATION_STATE,
                    f"molecule:{facts.identity.chemical_identity_digest}",
                ),
                "hund-os-domain",
                (ProposedCandidateValue(oxidation),),
                CandidateDomainScope.DOMAIN_BOUNDED,
                MaterializationStatus.COMPLETE,
                None,
            ),
        ),
        (),
    )
    oxidation_space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        candidate_proposals=(oxidation_proposal,),
    )
    candidate_context = RuleContext(
        facts,
        RulePhase.ELECTRONIC_STATE_CANDIDATE,
        requirements,
        oxidation_space,
        (),
        DEFAULT_CANDIDATE_BUDGET,
    )
    spin_outcome = IonicConfigurationSpinRule().evaluate(
        RuleToggle("OPTIONAL-009", True),
        candidate_context,
    )
    spin_space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        candidate_proposals=(
            oxidation_proposal,
            spin_outcome.candidate_proposal,
        ),
    )
    return RuleContext(
        facts,
        RulePhase.ELECTRONIC_STATE_RANKING,
        requirements,
        spin_space,
        (spin_outcome,),
        DEFAULT_CANDIDATE_BUDGET,
    )


def test_eu_iii_ranks_six_without_removing_lower_spin_candidates() -> None:
    context = _spin_context("[Eu]", 3)
    outcome = FBlockHundPriorRule().evaluate(
        RuleToggle("OPTIONAL-011", True),
        context,
    )

    candidates = {
        value.candidate.provenance.identity: value.candidate
        for factor in context.candidate_space.factors
        for value in factor.values
    }
    ranked = candidates[outcome.rank_adjustments[0].target_identity]
    assert ranked.two_s == 6
    assert {
        candidate.two_s
        for candidate in candidates.values()
        if hasattr(candidate, "two_s")
    } == {0, 2, 4, 6}
    assert outcome.rank_adjustments[0].assumption_ids == (
        "execution_prior:optional-011",
    )


def test_fe_ii_is_outside_f_block_hund_registry() -> None:
    outcome = FBlockHundPriorRule().evaluate(
        RuleToggle("OPTIONAL-011", True),
        _spin_context("[Fe]", 2),
    )

    assert not outcome.applicability.matches


def test_rule_declares_ionic_configuration_dependency() -> None:
    dependency = FBlockHundPriorRule.specification.requires[0]

    assert dependency.rule_id == "OPTIONAL-009"
