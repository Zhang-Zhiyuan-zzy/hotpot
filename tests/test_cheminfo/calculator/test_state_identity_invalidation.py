"""Scientific evidence invalidation across molecular-boundary changes."""

from __future__ import annotations

from typing import Callable

import hotpot
import pytest

from hotpot.cheminfo.calculator.charge import resolve_net_charge
from hotpot.cheminfo.calculator.state_resolver import (
    ProvidedScientificResults,
    StateRequest,
    extract_molecular_facts,
    resolve_scientific_state,
)
from hotpot.cheminfo.calculator.state_selection import StateDecisionAction
from hotpot.cheminfo.core import Molecule


def _change_graph(mol: Molecule) -> None:
    mol.add_bond(0, 2)


def _change_protonation(mol: Molecule) -> None:
    mol.atoms[0].implicit_hydrogens -= 1


def _split_component(mol: Molecule) -> None:
    mol.remove_bond(mol.bonds[0])


@pytest.mark.parametrize(
    "mutation",
    (_change_graph, _change_protonation, _split_component),
    ids=("graph", "protonation", "component"),
)
def test_stale_scientific_evidence_is_rejected_after_boundary_change(
    mutation: Callable[[Molecule], None],
) -> None:
    source_mol = hotpot.read_mol("CCO", "smi")
    source_facts = extract_molecular_facts(source_mol)
    source_charge = resolve_net_charge(
        source_facts,
        StateRequest(net_charge=0),
    )[0]
    changed_mol = source_mol.copy()
    mutation(changed_mol)

    bundle = resolve_scientific_state(
        extract_molecular_facts(changed_mol),
        provided_results=ProvidedScientificResults(
            net_charge_results=(source_charge,),
        ),
    )

    finding_codes = {
        finding.code
        for result in bundle.net_charge_results
        for finding in result.findings
    }
    assert bundle.execution_decision.action is StateDecisionAction.INADMISSIBLE
    assert bundle.execution_decision.selected_state_identity is None
    assert "HARD-001.FACTS_IDENTITY_MISMATCH" in finding_codes
