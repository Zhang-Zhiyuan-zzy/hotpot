"""OPTIONAL-007 ordinary Th(IV) prior tests."""

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
from hotpot.cheminfo.calculator.state_rules.neutral_l_donors import (
    NeutralLDonorRule,
)
from hotpot.cheminfo.calculator.state_rules.thorium_iv import (
    ThoriumIVPriorRule,
)
from hotpot.cheminfo.core import BondKind


def _context():
    mol = hotpot.read_mol("[Th].N", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    requirements = _effective_requirements(
        StatePolicy.COORDINATION_SCREENING_V1,
        None,
    )
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
    return RuleContext(
        facts,
        RulePhase.OXIDATION_STATE_CANDIDATE,
        requirements,
        candidate_space,
        (counting,),
        DEFAULT_CANDIDATE_BUDGET,
    )


def test_ordinary_thorium_donor_complex_proposes_and_ranks_iv() -> None:
    outcome = ThoriumIVPriorRule().evaluate(
        RuleToggle("OPTIONAL-007", True),
        _context(),
    )

    candidate = outcome.candidate_proposal.factors[0].materialized_candidates[
        0
    ].candidate
    assert candidate.atom_states[0] == 4
    assert outcome.rank_adjustments[0].target_identity == (
        candidate.provenance.identity
    )
