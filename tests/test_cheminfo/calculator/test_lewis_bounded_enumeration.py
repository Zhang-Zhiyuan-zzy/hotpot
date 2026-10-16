"""Behavioral parity for bounded Lewis-state enumeration."""

from typing import Dict, List, Tuple

import hotpot
import pytest

from hotpot.cheminfo.calculator.charge.lewis import (
    _atom_charge_candidates,
    _bond_valences,
    _enumerate_assignments,
)
from hotpot.cheminfo.calculator.contracts import MolecularFacts
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts


_Assignment = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


def _reference_enumeration(
    facts: MolecularFacts,
    bond_valences: Tuple[int, ...],
    target_charge: int,
    limit: int,
) -> Tuple[Tuple[_Assignment, ...], bool]:
    states: Dict[int, List[_Assignment]] = {0: [((), (), ())]}
    truncated = False
    for atomic_number, bond_valence in zip(facts.atomic_numbers, bond_valences):
        next_states: Dict[int, List[_Assignment]] = {}
        for accumulated_charge, assignments in states.items():
            for charges, nonbonding, radicals in assignments:
                for atom_charge, atom_nonbonding, atom_radicals in (
                    _atom_charge_candidates(atomic_number, bond_valence)
                ):
                    charge = accumulated_charge + atom_charge
                    next_states.setdefault(charge, []).append(
                        (
                            charges + (atom_charge,),
                            nonbonding + (atom_nonbonding,),
                            radicals + (atom_radicals,),
                        )
                    )
        states = {}
        for charge, assignments in next_states.items():
            ordered = sorted(set(assignments))
            truncated = len(ordered) > limit or truncated
            states[charge] = ordered[:limit]
    return tuple(states.get(target_charge, ())), truncated


@pytest.mark.parametrize("smiles", ("CO", "NNO", "CCO"))
@pytest.mark.parametrize("target_charge", (-1, 0, 1))
@pytest.mark.parametrize("limit", (1, 2, 8))
def test_bounded_enumeration_matches_prior_behavior(
    smiles: str,
    target_charge: int,
    limit: int,
) -> None:
    facts = extract_molecular_facts(hotpot.read_mol(smiles, "smi"))
    bond_valences = _bond_valences(facts)
    assert bond_valences is not None

    assert _enumerate_assignments(
        facts,
        bond_valences,
        target_charge,
        limit,
    ) == _reference_enumeration(
        facts,
        bond_valences,
        target_charge,
        limit,
    )
