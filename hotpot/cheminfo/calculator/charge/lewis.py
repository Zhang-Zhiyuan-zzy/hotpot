"""Deterministic main-group Lewis allocation for immutable molecular facts."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

from ..contracts import (
    CandidateBudget,
    CandidateProvenance,
    ChemicalConfidence,
    DEFAULT_CANDIDATE_BUDGET,
    MolecularFacts,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificFinding,
    StateValueAuthority,
)
from .contracts import (
    LewisFormalChargeCandidate,
    LewisFormalChargeResult,
    NetChargeCandidateEvidence,
    NetChargeResult,
)

__all__ = ["solve_main_group_lewis"]


# Reviewed fixed-bond-order main-group domain.


_MAIN_GROUP_ELECTRON_MODELS = {
    1: (1, (2,)),
    5: (3, (6, 8)),
    6: (4, (8,)),
    7: (5, (8,)),
    8: (6, (8,)),
    9: (7, (8,)),
    14: (4, (8, 10, 12)),
    15: (5, (8, 10)),
    16: (6, (8, 10, 12)),
    17: (7, (8, 10, 12, 14)),
    33: (5, (8, 10)),
    34: (6, (8, 10, 12)),
    35: (7, (8, 10, 12, 14)),
    52: (6, (8, 10, 12)),
    53: (7, (8, 10, 12, 14)),
}

_PROVIDER_ID = "PROVIDER-001"
_PROVIDER_VERSION = "fixed-bond-main-group-v1"


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bond_valences(facts: MolecularFacts) -> Optional[Tuple[int, ...]]:
    values = [0.0] * len(facts.atomic_numbers)
    for bond in facts.semantic_bonds:
        if bond.kind != "covalent" or bond.order is None:
            return None
        begin, end = bond.atom_indices
        values[begin] += bond.order
        values[end] += bond.order

    for index, atomic_number in enumerate(facts.atomic_numbers):
        if atomic_number == 1:
            continue
        explicit_hydrogens = facts.hydrogen_inventory.explicit_counts[index]
        implicit_hydrogens = facts.hydrogen_inventory.implicit_counts[index]
        if explicit_hydrogens == 0 and implicit_hydrogens is not None:
            values[index] += implicit_hydrogens

    if any(abs(value - round(value)) > 1.0e-8 for value in values):
        return None
    return tuple(int(round(value)) for value in values)


def _atom_charge_candidates(
    atomic_number: int,
    bond_valence: int,
) -> Tuple[Tuple[int, int, int, int], ...]:
    valence_electrons, preferred_shells = _MAIN_GROUP_ELECTRON_MODELS[
        atomic_number
    ]
    candidates = []
    for charge in range(-4, 5):
        nonbonding_electrons = valence_electrons - bond_valence - charge
        if nonbonding_electrons < 0:
            continue
        shell_electrons = nonbonding_electrons + 2 * bond_valence
        shell_distance, shell_rank = min(
            (abs(shell_electrons - shell), rank)
            for rank, shell in enumerate(preferred_shells)
        )
        radical_electrons = nonbonding_electrons % 2
        score = (
            100 * shell_distance
            + 20 * radical_electrons
            + 4 * abs(charge)
            + shell_rank
        )
        candidates.append(
            (charge, nonbonding_electrons, radical_electrons, score)
        )
    return tuple(candidates)


def _enumerate_assignments(
    facts: MolecularFacts,
    bond_valences: Tuple[int, ...],
    target_charge: int,
    limit: int,
) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], ...]:
    states: Dict[
        int,
        List[Tuple[int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]],
    ] = {0: [(0, (), (), ())]}
    for atomic_number, bond_valence in zip(facts.atomic_numbers, bond_valences):
        next_states: Dict[
            int,
            List[Tuple[int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]],
        ] = {}
        for accumulated_charge, assignments in states.items():
            for score, charges, nonbonding, radicals in assignments:
                for atom_charge, atom_nonbonding, atom_radicals, atom_score in (
                    _atom_charge_candidates(atomic_number, bond_valence)
                ):
                    charge = accumulated_charge + atom_charge
                    next_states.setdefault(charge, []).append(
                        (
                            score + atom_score,
                            charges + (atom_charge,),
                            nonbonding + (atom_nonbonding,),
                            radicals + (atom_radicals,),
                        )
                    )
        states = {
            charge: sorted(assignments)[:limit]
            for charge, assignments in next_states.items()
        }

    target_assignments = states.get(target_charge, [])
    if not target_assignments:
        return ()
    minimum_score = target_assignments[0][0]
    return tuple(
        (charges, nonbonding, radicals)
        for score, charges, nonbonding, radicals in target_assignments
        if score == minimum_score
    )


def _component_charges(
    facts: MolecularFacts,
    atom_charges: Tuple[int, ...],
) -> Tuple[Tuple[str, int], ...]:
    return tuple(
        (
            component.identity,
            sum(atom_charges[index] for index in component.atom_indices),
        )
        for component in facts.covalent_components
    )


def _unresolved_result(
    facts: MolecularFacts,
    net_candidate: NetChargeCandidateEvidence,
    code: str,
    message: str,
) -> LewisFormalChargeResult:
    identity = _digest(
        {
            "provider": _PROVIDER_ID,
            "version": _PROVIDER_VERSION,
            "facts": facts.identity.chemical_identity_digest,
            "net_charge_candidate": net_candidate.provenance.identity,
            "status": code,
        }
    )
    return LewisFormalChargeResult(
        identity=identity,
        facts_identity=facts.identity,
        candidates=(),
        admissibility=ScientificAdmissibility.NOT_EVALUATED,
        resolution=ResolutionStatus.OUT_OF_DOMAIN,
        confidence=ChemicalConfidence.UNRESOLVED,
        findings=(ScientificFinding(code=code, message=message),),
    )


def _solve_candidate(
    facts: MolecularFacts,
    net_candidate: NetChargeCandidateEvidence,
    candidate_budget: CandidateBudget,
) -> LewisFormalChargeResult:
    unsupported = tuple(
        sorted(
            {
                atomic_number
                for atomic_number in facts.atomic_numbers
                if atomic_number not in _MAIN_GROUP_ELECTRON_MODELS
            }
        )
    )
    if unsupported:
        return _unresolved_result(
            facts,
            net_candidate,
            "PROVIDER-001.UNSUPPORTED_ELEMENT",
            f"Main-group Lewis provider does not cover atomic numbers {unsupported}.",
        )

    bond_valences = _bond_valences(facts)
    if bond_valences is None:
        return _unresolved_result(
            facts,
            net_candidate,
            "PROVIDER-001.UNSUPPORTED_BOND_SEMANTICS",
            "Lewis provider requires known covalent fixed bond orders.",
        )

    assignments = _enumerate_assignments(
        facts,
        bond_valences,
        net_candidate.charge,
        candidate_budget.max_materialized_values_per_factor,
    )
    if not assignments:
        return _unresolved_result(
            facts,
            net_candidate,
            "PROVIDER-001.NO_LEWIS_ALLOCATION",
            "No supported fixed-bond Lewis allocation satisfies net charge.",
        )

    bond_orders = tuple(
        float(bond.order)
        for bond in facts.semantic_bonds
        if bond.order is not None
    )
    candidates = []
    for atom_charges, nonbonding, radicals in assignments:
        candidate_identity = _digest(
            {
                "provider": _PROVIDER_ID,
                "version": _PROVIDER_VERSION,
                "facts": facts.identity.chemical_identity_digest,
                "net_charge_candidate": net_candidate.provenance.identity,
                "atom_charges": atom_charges,
                "nonbonding": nonbonding,
                "radicals": radicals,
            }
        )
        candidates.append(
            LewisFormalChargeCandidate(
                provenance=CandidateProvenance(
                    identity=candidate_identity,
                    authority=StateValueAuthority.DERIVED_CONSTRAINT,
                    source_identity=f"{_PROVIDER_ID}:{_PROVIDER_VERSION}",
                    evidence_ids=(net_candidate.provenance.identity,),
                ),
                net_charge_candidate_identity=net_candidate.provenance.identity,
                atom_formal_charges=atom_charges,
                bond_orders=bond_orders,
                nonbonding_electrons=nonbonding,
                radical_electrons=radicals,
                component_charges=_component_charges(facts, atom_charges),
            )
        )
    result_identity = _digest(
        {
            "provider": _PROVIDER_ID,
            "version": _PROVIDER_VERSION,
            "facts": facts.identity.chemical_identity_digest,
            "candidate_ids": tuple(
                candidate.provenance.identity for candidate in candidates
            ),
        }
    )
    return LewisFormalChargeResult(
        identity=result_identity,
        facts_identity=facts.identity,
        candidates=tuple(candidates),
        admissibility=ScientificAdmissibility.ADMISSIBLE,
        resolution=(
            ResolutionStatus.UNIQUE
            if len(candidates) == 1
            else ResolutionStatus.AMBIGUOUS
        ),
        confidence=ChemicalConfidence.MEDIUM,
        findings=(),
    )


# Public provider operation.


def solve_main_group_lewis(
    facts: MolecularFacts,
    net_charge: NetChargeResult,
    *,
    candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
) -> Tuple[LewisFormalChargeResult, ...]:
    """Solve fixed-bond Lewis allocations for hard-valid charge candidates."""
    if net_charge.facts_identity != facts.identity:
        return tuple(
            _unresolved_result(
                facts,
                candidate,
                "HARD-001.FACTS_IDENTITY_MISMATCH",
                "Net-charge result is bound to different molecular facts.",
            )
            for candidate in net_charge.candidates
        )
    return tuple(
        _solve_candidate(facts, candidate, candidate_budget)
        for candidate in net_charge.candidates
    )
