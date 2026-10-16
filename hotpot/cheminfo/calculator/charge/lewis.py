"""Deterministic main-group Lewis allocation for immutable molecular facts."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from typing import Dict, List, Optional, Tuple

from .. import legality
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
_PROVIDER_VERSION = "fixed-bond-main-group-v2"


_LewisAssignment = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


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
        if bond.kind == "coordination":
            continue
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
) -> Tuple[Tuple[int, int, int], ...]:
    valence_electrons, preferred_shells = _MAIN_GROUP_ELECTRON_MODELS[
        atomic_number
    ]
    candidates = []
    for charge in range(-4, 5):
        nonbonding_electrons = valence_electrons - bond_valence - charge
        if nonbonding_electrons < 0:
            continue
        shell_electrons = nonbonding_electrons + 2 * bond_valence
        if shell_electrons not in preferred_shells:
            continue
        radical_electrons = nonbonding_electrons % 2
        candidates.append((charge, nonbonding_electrons, radical_electrons))
    return tuple(candidates)


def _retain_bounded_assignment(
    assignments: List[_LewisAssignment],
    assignment: _LewisAssignment,
    limit: int,
) -> bool:
    """Insert one sorted unique assignment and report whether one was omitted."""
    position = bisect_left(assignments, assignment)
    if position < len(assignments) and assignments[position] == assignment:
        return False
    assignments.insert(position, assignment)
    if len(assignments) <= limit:
        return False
    assignments.pop()
    return True


def _enumerate_assignments(
    facts: MolecularFacts,
    bond_valences: Tuple[int, ...],
    target_charge: int,
    limit: int,
) -> Tuple[
    Tuple[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], ...],
    bool,
]:
    states: Dict[int, List[_LewisAssignment]] = {0: [((), (), ())]}
    truncated = False
    for atomic_number, bond_valence in zip(facts.atomic_numbers, bond_valences):
        next_states: Dict[int, List[_LewisAssignment]] = {}
        atom_candidates = _atom_charge_candidates(
            atomic_number,
            bond_valence,
        )
        for accumulated_charge, assignments in states.items():
            for charges, nonbonding, radicals in assignments:
                for (
                    atom_charge,
                    atom_nonbonding,
                    atom_radicals,
                ) in atom_candidates:
                    charge = accumulated_charge + atom_charge
                    truncated = _retain_bounded_assignment(
                        next_states.setdefault(charge, []),
                        (
                            charges + (atom_charge,),
                            nonbonding + (atom_nonbonding,),
                            radicals + (atom_radicals,),
                        ),
                        limit,
                    ) or truncated
        states = next_states

    target_assignments = tuple(states.get(target_charge, ()))
    return target_assignments, truncated


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


def _empty_result(
    facts: MolecularFacts,
    code: str,
    message: str,
    *,
    related_identities: Tuple[str, ...] = (),
    admissibility: ScientificAdmissibility = ScientificAdmissibility.NOT_EVALUATED,
    resolution: ResolutionStatus = ResolutionStatus.OUT_OF_DOMAIN,
) -> LewisFormalChargeResult:
    identity = _digest(
        {
            "provider": _PROVIDER_ID,
            "version": _PROVIDER_VERSION,
            "facts": facts.identity.chemical_identity_digest,
            "status": code,
            "related": related_identities,
        }
    )
    return LewisFormalChargeResult(
        identity=identity,
        facts_identity=facts.identity,
        candidates=(),
        admissibility=admissibility,
        resolution=resolution,
        confidence=ChemicalConfidence.UNRESOLVED,
        findings=(
            ScientificFinding(
                code=code,
                message=message,
                related_identities=related_identities,
            ),
        ),
    )


def _unresolved_result(
    facts: MolecularFacts,
    net_candidate: NetChargeCandidateEvidence,
    code: str,
    message: str,
) -> LewisFormalChargeResult:
    return _empty_result(
        facts,
        code,
        message,
        related_identities=(net_candidate.provenance.identity,),
    )


def _solve_validated_candidate(
    facts: MolecularFacts,
    candidate: NetChargeCandidateEvidence,
    candidate_budget: CandidateBudget,
) -> LewisFormalChargeResult:
    assessment = legality._validate_charge_conservation(facts, candidate)
    if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE:
        return _solve_candidate(facts, candidate, candidate_budget)
    finding = assessment.findings[0]
    return _empty_result(
        facts,
        finding.code,
        finding.message,
        related_identities=finding.related_identities,
        admissibility=assessment.admissibility,
        resolution=assessment.terminal_status or ResolutionStatus.UNRESOLVED,
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

    assignments, truncated = _enumerate_assignments(
        facts,
        bond_valences,
        net_candidate.charge,
        candidate_budget.max_materialized_values_per_factor,
    )
    if not assignments:
        if truncated:
            return _unresolved_result(
                facts,
                net_candidate,
                "PROVIDER-001.RESOURCE_TRUNCATED",
                (
                    "Lewis search budget was exhausted before a satisfying "
                    "allocation was retained."
                ),
            )
        return _unresolved_result(
            facts,
            net_candidate,
            "PROVIDER-001.NO_LEWIS_ALLOCATION",
            "No supported fixed-bond Lewis allocation satisfies net charge.",
        )

    bond_orders = tuple(
        float(bond.order)
        for bond in facts.semantic_bonds
        if bond.kind == "covalent" and bond.order is not None
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
            "truncated": truncated,
        }
    )
    return LewisFormalChargeResult(
        identity=result_identity,
        facts_identity=facts.identity,
        candidates=tuple(candidates),
        admissibility=ScientificAdmissibility.ADMISSIBLE,
        resolution=(
            ResolutionStatus.UNIQUE
            if len(candidates) == 1 and not truncated
            else ResolutionStatus.AMBIGUOUS
        ),
        confidence=ChemicalConfidence.MEDIUM,
        findings=(
            (
                ScientificFinding(
                    code="PROVIDER-001.RESOURCE_TRUNCATED",
                    message=(
                        "Lewis candidate materialization reached the configured "
                        "per-factor budget; scientific uniqueness is not claimed."
                    ),
                    related_identities=tuple(
                        candidate.provenance.identity for candidate in candidates
                    ),
                ),
            )
            if truncated
            else ()
        ),
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
        return (
            _empty_result(
                facts,
                "HARD-001.FACTS_IDENTITY_MISMATCH",
                "Net-charge result is bound to different molecular facts.",
                related_identities=(net_charge.identity,),
                admissibility=ScientificAdmissibility.INADMISSIBLE,
                resolution=ResolutionStatus.CONFLICT,
            ),
        )
    if net_charge.admissibility is not ScientificAdmissibility.ADMISSIBLE:
        return (
            _empty_result(
                facts,
                "PROVIDER-001.NET_CHARGE_RESULT_UNAVAILABLE",
                "Lewis solving requires a hard-valid net-charge result.",
                related_identities=(net_charge.identity,),
                admissibility=ScientificAdmissibility.INADMISSIBLE,
                resolution=(
                    ResolutionStatus.CONFLICT
                    if net_charge.resolution is ResolutionStatus.CONFLICT
                    else ResolutionStatus.UNRESOLVED
                ),
            ),
        )
    if not net_charge.candidates:
        return (
            _empty_result(
                facts,
                "PROVIDER-001.NET_CHARGE_CANDIDATE_MISSING",
                "Lewis solving requires at least one net-charge candidate.",
            ),
        )
    return tuple(
        _solve_validated_candidate(facts, candidate, candidate_budget)
        for candidate in net_charge.candidates
    )
