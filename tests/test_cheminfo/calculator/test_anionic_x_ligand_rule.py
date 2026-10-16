"""OPTIONAL-004 ordinary anionic X-type ligand tests."""

import pytest

import hotpot

from hotpot.cheminfo.calculator.contracts import (
    DEFAULT_CANDIDATE_BUDGET,
    StatePolicy,
)
from hotpot.cheminfo.calculator.operation_requirements import (
    _effective_requirements,
)
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts
from hotpot.cheminfo.calculator.state_rules import (
    RuleContext,
    RulePhase,
    RuleToggle,
)
from hotpot.cheminfo.calculator.state_rules.anionic_x_ligands import (
    AnionicXLigandRule,
)
from hotpot.cheminfo.core import BondKind


def _outcome(smiles: str, donor_index: int):
    mol = hotpot.read_mol(smiles, "smi")
    mol.add_bond(0, donor_index, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    context = RuleContext(
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
    return AnionicXLigandRule().evaluate(
        RuleToggle("OPTIONAL-004", True),
        context,
    )


@pytest.mark.parametrize(
    ("smiles", "donor_index"),
    (
        ("[Eu].[Cl-]", 1),
        ("[Eu].[H-]", 1),
        ("[Eu].[OH-]", 1),
        ("[Eu].C[O-]", 2),
        ("[Eu].CC(=O)[O-]", 4),
        ("[Eu].[C-]#N", 1),
    ),
)
def test_supported_x_ligands_propose_minus_one(
    smiles: str,
    donor_index: int,
) -> None:
    outcome = _outcome(smiles, donor_index)

    assert outcome.applicability.matches
    candidates = tuple(
        value.candidate
        for factor in outcome.candidate_proposal.factors
        for value in factor.materialized_candidates
    )
    assert {
        candidate.fragments[0].ionic_bookkeeping_charge
        for candidate in candidates
    } == {-1}


@pytest.mark.parametrize(
    ("smiles", "donor_index"),
    (
        ("[Eu].CO", 2),
        ("[Eu].CC(=O)O", 4),
    ),
)
def test_neutral_alcohol_and_acid_forms_abstain(
    smiles: str,
    donor_index: int,
) -> None:
    outcome = _outcome(smiles, donor_index)

    assert not outcome.applicability.matches


def test_bridging_halide_abstains() -> None:
    mol = hotpot.read_mol("[Eu].[Eu].[Cl-]", "smi")
    mol.add_bond(0, 2, bond_kind=BondKind.DATIVE)
    mol.add_bond(1, 2, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    context = RuleContext(
        facts,
        RulePhase.COORDINATION_COUNTING,
        _effective_requirements(StatePolicy.COORDINATION_SCREENING_V1, None),
        None,
        (),
        DEFAULT_CANDIDATE_BUDGET,
    )
    outcome = AnionicXLigandRule().evaluate(
        RuleToggle("OPTIONAL-004", True),
        context,
    )

    assert not outcome.applicability.matches
