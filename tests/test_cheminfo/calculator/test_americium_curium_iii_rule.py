"""OPTIONAL-006 ordinary Am/Cm(III) prior tests."""

import pytest

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
from hotpot.cheminfo.calculator.state_rules.americium_curium_iii import (
    AmericiumCuriumIIIPriorRule,
)
from hotpot.cheminfo.calculator.state_rules.neutral_l_donors import (
    NeutralLDonorRule,
)
from hotpot.cheminfo.core import BondKind


def _context(symbol: str, with_counting: bool = True):
    mol = hotpot.read_mol(f"[{symbol}].N", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    requirements = _effective_requirements(
        StatePolicy.COORDINATION_SCREENING_V1,
        None,
    )
    if with_counting:
        counting = NeutralLDonorRule().evaluate(
            RuleToggle("OPTIONAL-003", True),
            RuleContext(
                facts,
                RulePhase.COORDINATION_COUNTING,
                requirements,
                None,
                (),
                DEFAULT_CANDIDATE_BUDGET,
            ),
        )
        candidate_space = compose_scientific_state_space(
            facts,
            net_charge_results=(),
            candidate_proposals=(counting.candidate_proposal,),
        )
        prior_outcomes = (counting,)
    else:
        candidate_space = None
        prior_outcomes = ()
    return RuleContext(
        facts,
        RulePhase.OXIDATION_STATE_CANDIDATE,
        requirements,
        candidate_space,
        prior_outcomes,
        DEFAULT_CANDIDATE_BUDGET,
    )


@pytest.mark.parametrize("symbol", ("Am", "Cm"))
def test_ordinary_donor_complex_proposes_and_ranks_iii(symbol: str) -> None:
    outcome = AmericiumCuriumIIIPriorRule().evaluate(
        RuleToggle("OPTIONAL-006", True),
        _context(symbol),
    )

    candidate = outcome.candidate_proposal.factors[0].materialized_candidates[
        0
    ].candidate
    assert candidate.atom_states[0] == 3
    assert outcome.rank_adjustments[0].target_identity == (
        candidate.provenance.identity
    )


def test_am_prior_abstains_without_local_ligand_counting() -> None:
    outcome = AmericiumCuriumIIIPriorRule().evaluate(
        RuleToggle("OPTIONAL-006", True),
        _context("Am", with_counting=False),
    )

    assert not outcome.applicability.matches
