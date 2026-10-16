"""OPTIONAL-003 neutral L-type donor tests."""

import hotpot

from hotpot.cheminfo.calculator.contracts import (
    DEFAULT_CANDIDATE_BUDGET,
    StatePolicy,
)
from hotpot.cheminfo.calculator.operation_requirements import (
    _effective_requirements,
)
from hotpot.cheminfo.calculator.oxidation_state import CountingConvention
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts
from hotpot.cheminfo.calculator.state_rules import (
    RuleContext,
    RulePhase,
    RuleToggle,
)
from hotpot.cheminfo.calculator.state_rules.neutral_l_donors import (
    NeutralLDonorRule,
)
from hotpot.cheminfo.core import BondKind


def _facts(smiles: str, donor_index: int):
    mol = hotpot.read_mol(smiles, "smi")
    mol.add_bond(0, donor_index, bond_kind=BondKind.DATIVE)
    return extract_molecular_facts(mol)


def _context(facts):
    return RuleContext(
        facts=facts,
        phase=RulePhase.COORDINATION_COUNTING,
        requirements=_effective_requirements(
            StatePolicy.COORDINATION_SCREENING_V1,
            None,
        ),
        candidate_space=None,
        prior_outcomes=(),
        candidate_budget=DEFAULT_CANDIDATE_BUDGET,
    )


def _outcome(facts, configuration=()):
    return NeutralLDonorRule().evaluate(
        RuleToggle("OPTIONAL-003", True, configuration),
        _context(facts),
    )


def test_neutral_amine_is_an_ionic_zero_l_donor_by_default() -> None:
    outcome = _outcome(_facts("[Eu].N", 1))

    assert outcome.applicability.matches
    candidate = outcome.candidate_proposal.factors[0].materialized_candidates[
        0
    ].candidate
    assert candidate.convention is CountingConvention.IONIC
    assert candidate.fragments[0].ionic_bookkeeping_charge == 0
    assert candidate.fragments[0].lxz_class is None


def test_requested_counting_conventions_remain_distinct() -> None:
    outcome = _outcome(
        _facts("[Eu].COC", 2),
        (
            (
                "conventions",
                (
                    CountingConvention.IONIC.value,
                    CountingConvention.DONOR_RETURN.value,
                    CountingConvention.CBC.value,
                ),
            ),
        ),
    )

    values = outcome.candidate_proposal.factors[0].materialized_candidates
    candidates = {value.candidate.convention: value.candidate for value in values}
    assert candidates[
        CountingConvention.IONIC
    ].fragments[0].ionic_bookkeeping_charge == 0
    assert candidates[
        CountingConvention.DONOR_RETURN
    ].fragments[0].donated_electrons == 2
    assert candidates[CountingConvention.CBC].fragments[0].lxz_class == "L"


def test_pyridine_and_carbon_monoxide_signatures_match() -> None:
    pyridine = _outcome(_facts("[Eu].n1ccccc1", 1))
    carbon_monoxide = _outcome(_facts("[Eu].[C-]#[O+]", 1))

    assert pyridine.applicability.matches
    assert carbon_monoxide.applicability.matches


def test_protonated_or_anionic_donors_do_not_match_neutral_templates() -> None:
    ammonium = _outcome(_facts("[Eu].[NH4+]", 1))
    phenoxide = _outcome(_facts("[Eu].[O-]c1ccccc1", 1))

    assert not ammonium.applicability.matches
    assert not phenoxide.applicability.matches
