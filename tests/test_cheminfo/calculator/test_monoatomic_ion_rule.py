"""OPTIONAL-002 common monoatomic-ion rule tests."""

import hotpot

from hotpot.cheminfo.calculator.contracts import (
    ChemicalConfidence,
    DEFAULT_CANDIDATE_BUDGET,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificQuantity,
    StatePolicy,
)
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateInferenceResult,
)
from hotpot.cheminfo.calculator.operation_requirements import (
    _effective_requirements,
)
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts
from hotpot.cheminfo.calculator.state_composition import (
    StateValueOrigin,
    compose_scientific_state_space,
)
from hotpot.cheminfo.calculator.state_rules import (
    RuleContext,
    RulePhase,
    RuleToggle,
)
from hotpot.cheminfo.calculator.state_rules.monoatomic_ions import (
    MonoatomicIonRule,
)
from hotpot.cheminfo.core import BondKind


def _context(facts):
    return RuleContext(
        facts=facts,
        phase=RulePhase.CHARGE_EVIDENCE,
        requirements=_effective_requirements(
            StatePolicy.COORDINATION_SCREENING_V1,
            None,
        ),
        candidate_space=None,
        prior_outcomes=(),
        candidate_budget=DEFAULT_CANDIDATE_BUDGET,
    )


def test_disconnected_sodium_and_chloride_receive_bounded_ionic_domains() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Na+].[Cl-]", "smi"))
    outcome = MonoatomicIonRule().evaluate(
        RuleToggle("OPTIONAL-002", True),
        _context(facts),
    )

    assert outcome.applicability.matches
    factors = outcome.candidate_proposal.factors
    assert len(factors) == 2
    assert all(
        factor.factor_key.quantity
        is ScientificQuantity.COORDINATION_COUNTING
        for factor in factors
    )
    charges = {
        value.candidate.fragments[0].ionic_bookkeeping_charge
        for factor in factors
        for value in factor.materialized_candidates
    }
    assert charges == {-1, 1}

    space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        electronic_state_results=(
            ElectronicStateInferenceResult(
                identity="electronic:unresolved",
                facts_identity=facts.identity,
                candidates=(),
                admissibility=ScientificAdmissibility.NOT_EVALUATED,
                resolution=ResolutionStatus.UNRESOLVED,
                confidence=ChemicalConfidence.UNRESOLVED,
                findings=(),
            ),
        ),
        candidate_proposals=(outcome.candidate_proposal,),
    )
    assert space.expanded_paths[0].net_charge.value == 0
    assert space.expanded_paths[0].net_charge.origin is (
        StateValueOrigin.DERIVED_CONSTRAINT
    )


def test_coordinated_sodium_is_not_reclassified_as_counterion() -> None:
    mol = hotpot.read_mol("[Na].O", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    outcome = MonoatomicIonRule().evaluate(
        RuleToggle("OPTIONAL-002", True),
        _context(facts),
    )

    assert not outcome.applicability.matches
    assert outcome.candidate_proposal is None


def test_complete_single_ion_accounting_derives_its_system_charge() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Na+]", "smi"))
    outcome = MonoatomicIonRule().evaluate(
        RuleToggle("OPTIONAL-002", True),
        _context(facts),
    )
    space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        electronic_state_results=(
            ElectronicStateInferenceResult(
                identity="electronic:unresolved-na",
                facts_identity=facts.identity,
                candidates=(),
                admissibility=ScientificAdmissibility.NOT_EVALUATED,
                resolution=ResolutionStatus.UNRESOLVED,
                confidence=ChemicalConfidence.UNRESOLVED,
                findings=(),
            ),
        ),
        candidate_proposals=(outcome.candidate_proposal,),
    )

    assert space.expanded_paths[0].net_charge.value == 1


def test_multivalent_transition_metal_atom_is_outside_registry() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe]", "smi"))
    outcome = MonoatomicIonRule().evaluate(
        RuleToggle("OPTIONAL-002", True),
        _context(facts),
    )

    assert not outcome.applicability.matches
