"""OPTIONAL-001 supported main-group closed-shell tests."""

import hotpot

from hotpot.cheminfo.calculator.charge import (
    resolve_net_charge,
    solve_main_group_lewis,
)
from hotpot.cheminfo.calculator.contracts import (
    DEFAULT_CANDIDATE_BUDGET,
    StatePolicy,
)
from hotpot.cheminfo.calculator.electronic_state import (
    SpinContributionScope,
    infer_electronic_states,
)
from hotpot.cheminfo.calculator.operation_requirements import (
    _effective_requirements,
)
from hotpot.cheminfo.calculator.state_composition import (
    CandidateSearchStatus,
    compose_scientific_state_space,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)
from hotpot.cheminfo.calculator.state_rules import (
    RuleContext,
    RulePhase,
    RuleToggle,
)
from hotpot.cheminfo.calculator.state_rules.main_group_closed_shell import (
    MainGroupClosedShellRule,
)
from hotpot.cheminfo.core import BondKind


def _context(smiles: str, charge: int = 0):
    facts = extract_molecular_facts(hotpot.read_mol(smiles, "smi"))
    net_charge = resolve_net_charge(
        facts,
        StateRequest(net_charge=charge),
    )[0]
    lewis = solve_main_group_lewis(facts, net_charge)[0]
    candidate_space = compose_scientific_state_space(
        facts,
        net_charge_results=(net_charge,),
        lewis_results=(lewis,),
    )
    return (
        RuleContext(
            facts,
            RulePhase.ELECTRONIC_STATE_CANDIDATE,
            _effective_requirements(
                StatePolicy.COORDINATION_SCREENING_V1,
                None,
            ),
            candidate_space,
            (),
            DEFAULT_CANDIDATE_BUDGET,
        ),
        net_charge,
        lewis,
    )


def _outcome(smiles: str, charge: int = 0):
    context, _, _ = _context(smiles, charge)
    return MainGroupClosedShellRule().evaluate(
        RuleToggle("OPTIONAL-001", True),
        context,
    )


def test_ordinary_closed_shell_molecules_propose_molecular_singlets() -> None:
    for smiles in ("O", "CCO", "c1ccccc1"):
        outcome = _outcome(smiles)

        assert outcome.applicability.matches
        candidates = outcome.candidate_proposal.factors[
            0
        ].materialized_candidates
        assert candidates
        assert all(
            value.candidate.scope is SpinContributionScope.MOLECULAR
            and value.candidate.two_s == 0
            for value in candidates
        )


def test_open_shell_counterexamples_do_not_match_by_even_parity() -> None:
    oxygen = _outcome("O=O")
    carbene = _outcome("[CH2]")

    assert not oxygen.applicability.matches
    assert not carbene.applicability.matches


def test_unreviewed_hypervalent_structure_abstains() -> None:
    outcome = _outcome("P(F)(F)(F)(F)F")

    assert not outcome.applicability.matches


def test_coordination_complex_is_not_assigned_a_molecular_singlet() -> None:
    mol = hotpot.read_mol("[Eu].O", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=3))[0]
    lewis = solve_main_group_lewis(facts, net_charge)[0]
    candidate_space = compose_scientific_state_space(
        facts,
        net_charge_results=(net_charge,),
        lewis_results=(lewis,),
    )
    context = RuleContext(
        facts,
        RulePhase.ELECTRONIC_STATE_CANDIDATE,
        _effective_requirements(StatePolicy.COORDINATION_SCREENING_V1, None),
        candidate_space,
        (),
        DEFAULT_CANDIDATE_BUDGET,
    )

    outcome = MainGroupClosedShellRule().evaluate(
        RuleToggle("OPTIONAL-001", True),
        context,
    )

    assert not outcome.applicability.matches


def test_atom_order_and_kekule_form_preserve_applicability() -> None:
    aromatic = _outcome("c1ccccc1")
    kekule = _outcome("C1=CC=CC=C1")
    ethanol_forward = _outcome("CCO")
    ethanol_reverse = _outcome("OCC")

    assert aromatic.applicability == kekule.applicability
    assert ethanol_forward.applicability == ethanol_reverse.applicability


def test_molecular_contribution_composes_a_complete_electronic_state() -> None:
    context, net_charge, lewis = _context("CCO")
    outcome = MainGroupClosedShellRule().evaluate(
        RuleToggle("OPTIONAL-001", True),
        context,
    )
    unresolved = infer_electronic_states(
        context.facts,
        StateRequest(),
        net_charge_results=(net_charge,),
    )

    candidate_space = compose_scientific_state_space(
        context.facts,
        net_charge_results=(net_charge,),
        lewis_results=(lewis,),
        electronic_state_results=unresolved,
        candidate_proposals=(outcome.candidate_proposal,),
    )

    assert len(candidate_space.expanded_paths) == 1
    path = candidate_space.expanded_paths[0]
    assert path.electronic_state_candidate_identity is not None
    assert path.electronic_state_result_id == outcome.candidate_proposal.proposal_id
    assert candidate_space.search_status is CandidateSearchStatus.DECISION_CERTIFIED
