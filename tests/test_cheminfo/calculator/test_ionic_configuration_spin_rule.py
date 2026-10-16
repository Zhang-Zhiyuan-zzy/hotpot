"""OPTIONAL-009 ionic-configuration spin-domain tests."""

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
from hotpot.cheminfo.calculator.state_rules.ionic_configuration_spin import (
    IonicConfigurationSpinRule,
)


def _context(smiles: str, oxidation_state: int):
    facts = extract_molecular_facts(hotpot.read_mol(smiles, "smi"))
    oxidation = OxidationStateCandidateEvidence(
        CandidateProvenance(
            f"os:{facts.atomic_numbers[0]}:{oxidation_state}",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "ionic-spin-test",
        ),
        CountingConvention.IONIC,
        (oxidation_state,),
        CoverageStatus.COMPLETE,
    )
    proposal = CandidateProposal(
        "proposal:test-os",
        (
            ProposedCandidateFactor(
                FactorKey(
                    ScientificQuantity.OXIDATION_STATE,
                    f"molecule:{facts.identity.chemical_identity_digest}",
                ),
                "test-os-domain",
                (ProposedCandidateValue(oxidation),),
                CandidateDomainScope.DOMAIN_BOUNDED,
                MaterializationStatus.COMPLETE,
                None,
            ),
        ),
        (),
    )
    candidate_space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        candidate_proposals=(proposal,),
    )
    return RuleContext(
        facts,
        RulePhase.ELECTRONIC_STATE_CANDIDATE,
        _effective_requirements(StatePolicy.COORDINATION_SCREENING_V1, None),
        candidate_space,
        (),
        DEFAULT_CANDIDATE_BUDGET,
    )


def _two_s_values(context):
    outcome = IonicConfigurationSpinRule().evaluate(
        RuleToggle("OPTIONAL-009", True),
        context,
    )
    values = outcome.candidate_proposal.factors[0].materialized_candidates
    return outcome, {value.candidate.two_s for value in values}


def test_fe_ii_d6_proposes_all_spin_only_values() -> None:
    outcome, values = _two_s_values(_context("[Fe]", 2))

    assert values == {0, 2, 4}
    assert len(outcome.candidate_proposal.constraints) == 3


def test_eu_iii_f6_proposes_all_spin_only_values() -> None:
    outcome, values = _two_s_values(_context("[Eu]", 3))

    assert values == {0, 2, 4, 6}
    assert all(
        value.candidate.parent_candidate_ids == ("os:63:3",)
        for value in outcome.candidate_proposal.factors[
            0
        ].materialized_candidates
    )


def test_unregistered_ionic_state_abstains() -> None:
    outcome = IonicConfigurationSpinRule().evaluate(
        RuleToggle("OPTIONAL-009", True),
        _context("[Fe]", 7),
    )

    assert not outcome.applicability.matches
