"""OPTIONAL-005 ordinary lanthanide(III) prior tests."""

import hotpot

from hotpot.cheminfo.calculator.contracts import (
    DEFAULT_CANDIDATE_BUDGET,
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
from hotpot.cheminfo.calculator.state_rules.lanthanide_iii import (
    LanthanideIIIPriorRule,
)
from hotpot.cheminfo.calculator.state_rules.neutral_l_donors import (
    NeutralLDonorRule,
)
from hotpot.cheminfo.core import BondKind


def _context(symbol: str):
    mol = hotpot.read_mol(f"[{symbol}].N", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    requirements = _effective_requirements(
        StatePolicy.COORDINATION_SCREENING_V1,
        None,
    )
    counting_context = RuleContext(
        facts,
        RulePhase.COORDINATION_COUNTING,
        requirements,
        None,
        (),
        DEFAULT_CANDIDATE_BUDGET,
    )
    counting = NeutralLDonorRule().evaluate(
        RuleToggle("OPTIONAL-003", True),
        counting_context,
    )
    candidate_space = compose_scientific_state_space(
        facts,
        net_charge_results=(),
        candidate_proposals=(counting.candidate_proposal,),
    )
    return RuleContext(
        facts,
        RulePhase.OXIDATION_STATE_CANDIDATE,
        requirements,
        candidate_space,
        (counting,),
        DEFAULT_CANDIDATE_BUDGET,
    )


def _states_and_ranked_state(symbol: str):
    outcome = LanthanideIIIPriorRule().evaluate(
        RuleToggle("OPTIONAL-005", True),
        _context(symbol),
    )
    values = outcome.candidate_proposal.factors[0].materialized_candidates
    state_by_id = {
        value.candidate.provenance.identity: value.candidate.atom_states[0]
        for value in values
    }
    ranked = tuple(
        state_by_id[adjustment.target_identity]
        for adjustment in outcome.rank_adjustments
    )
    return frozenset(state_by_id.values()), ranked


def test_europium_ii_and_iii_are_retained_but_iii_is_ranked() -> None:
    states, ranked = _states_and_ranked_state("Eu")

    assert states == {2, 3}
    assert ranked == (3,)


def test_cerium_iv_is_retained_without_displacing_common_iii() -> None:
    states, ranked = _states_and_ranked_state("Ce")

    assert states == {3, 4}
    assert ranked == (3,)


def test_lanthanum_domain_contains_only_reviewed_iii_state() -> None:
    states, ranked = _states_and_ranked_state("La")

    assert states == {3}
    assert ranked == (3,)
