"""Hard-equation derivation within one selected scientific path."""

from __future__ import annotations

import hashlib
import json
from typing import Optional, Tuple

from ..contracts import (
    CandidateProvenance,
    CoverageStatus,
    MolecularFacts,
    ScientificQuantity,
    StateValueAuthority,
)
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    CountingConvention,
    OxidationStateCandidateEvidence,
)
from .contracts import DerivedStateValue

__all__ = []


_COMPLETE_OS_EQUATION = "HARD-004.complete-oxidation-sum-v1"
_IONIC_COMPONENT_EQUATION = "HARD-002.ionic-component-sum-v1"


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derived_charge(
    facts: MolecularFacts,
    value: int,
    equation_id: str,
    parent_candidate_ids: Tuple[str, ...],
    convention: CountingConvention,
) -> DerivedStateValue:
    identity = _digest(
        {
            "facts": facts.identity.chemical_identity_digest,
            "quantity": ScientificQuantity.NET_CHARGE.value,
            "value": value,
            "equation": equation_id,
            "parents": parent_candidate_ids,
            "convention": convention.value,
        }
    )
    return DerivedStateValue(
        provenance=CandidateProvenance(
            identity=identity,
            authority=StateValueAuthority.DERIVED_CONSTRAINT,
            source_identity=equation_id,
            evidence_ids=parent_candidate_ids,
        ),
        quantity=ScientificQuantity.NET_CHARGE,
        value=value,
        equation_id=equation_id,
        parent_candidate_ids=parent_candidate_ids,
        counting_convention=convention,
    )


def _has_implicit_hydrogen(
    facts: MolecularFacts,
    atom_indices: Tuple[int, ...],
) -> bool:
    return any(
        facts.hydrogen_inventory.explicit_counts[index] == 0
        and facts.hydrogen_inventory.implicit_counts[index] not in (None, 0)
        for index in atom_indices
    )


def _charge_from_complete_oxidation_state(
    facts: MolecularFacts,
    candidate: OxidationStateCandidateEvidence,
) -> Optional[DerivedStateValue]:
    if (
        candidate.coverage is not CoverageStatus.COMPLETE
        or any(state is None for state in candidate.atom_states)
        or _has_implicit_hydrogen(
            facts,
            tuple(range(len(facts.atomic_numbers))),
        )
    ):
        return None
    value = sum(state for state in candidate.atom_states if state is not None)
    return _derived_charge(
        facts,
        value,
        _COMPLETE_OS_EQUATION,
        (candidate.provenance.identity,),
        candidate.convention,
    )


def _ionic_component_charge(
    facts: MolecularFacts,
    oxidation_state: OxidationStateCandidateEvidence,
    counting: CoordinationCountingCandidate,
) -> Optional[DerivedStateValue]:
    if (
        oxidation_state.convention is not CountingConvention.IONIC
        or counting.convention is not CountingConvention.IONIC
        or (
            oxidation_state.coordination_counting_candidate_identity
            is not None
            and oxidation_state.coordination_counting_candidate_identity
            != counting.provenance.identity
        )
    ):
        return None
    fragments = {
        fragment.component_identity: fragment
        for fragment in counting.fragments
    }
    if len(fragments) != len(counting.fragments):
        return None
    component_charge = 0
    for component in facts.covalent_components:
        states = tuple(
            oxidation_state.atom_states[index]
            for index in component.atom_indices
        )
        if all(state is not None for state in states):
            if _has_implicit_hydrogen(facts, component.atom_indices):
                return None
            component_charge += sum(
                state for state in states if state is not None
            )
            continue
        if any(state is not None for state in states):
            return None
        fragment = fragments.get(component.identity)
        if fragment is None or fragment.ionic_bookkeeping_charge is None:
            return None
        component_charge += fragment.ionic_bookkeeping_charge
    parents = tuple(
        sorted(
            (
                oxidation_state.provenance.identity,
                counting.provenance.identity,
            )
        )
    )
    return _derived_charge(
        facts,
        component_charge,
        _IONIC_COMPONENT_EQUATION,
        parents,
        CountingConvention.IONIC,
    )


def _derive_charge_values(
    facts: MolecularFacts,
    counting_candidates: Tuple[CoordinationCountingCandidate, ...],
    oxidation_state: Optional[OxidationStateCandidateEvidence],
) -> Tuple[DerivedStateValue, ...]:
    if oxidation_state is None:
        return ()
    complete_value = _charge_from_complete_oxidation_state(
        facts,
        oxidation_state,
    )
    if complete_value is not None:
        return (complete_value,)
    values = tuple(
        value
        for counting in counting_candidates
        for value in (
            _ionic_component_charge(facts, oxidation_state, counting),
        )
        if value is not None
    )
    values_by_identity = {
        value.provenance.identity: value for value in values
    }
    return tuple(
        values_by_identity[identity]
        for identity in sorted(values_by_identity)
    )
