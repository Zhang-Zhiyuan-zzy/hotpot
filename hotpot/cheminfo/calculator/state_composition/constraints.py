"""Hard compatibility checks for one coupled scientific-state path."""

from __future__ import annotations

import hashlib
import json
from typing import Optional, Tuple

from .. import legality
from ..charge.contracts import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
    NetChargeResult,
)
from ..contracts import (
    MolecularFacts,
    ScientificAdmissibility,
    ScientificQuantity,
    StateValueAuthority,
)
from ..electronic_state.contracts import (
    ElectronicStateCandidate,
    ElectronicStateInferenceResult,
)
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    OxidationStateCandidateEvidence,
    OxidationStateInferenceResult,
    OxidationStateOverride,
)
from .contracts import CandidateConstraint, FactorKey

__all__ = []


_AUTHORITATIVE_VALUES = frozenset(
    {
        StateValueAuthority.VERIFIED_EXPLICIT,
        StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
        StateValueAuthority.EXACT_IDENTITY_TRUSTED_SOURCE,
    }
)
_CONSTRAINT_SCHEMA = "candidate-constraint-v1"


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _molecular_factor_key(
    facts: MolecularFacts,
    quantity: ScientificQuantity,
) -> FactorKey:
    return FactorKey(
        quantity,
        _molecular_subject_identity(facts),
    )


def _molecular_subject_identity(facts: MolecularFacts) -> str:
    return f"molecule:{facts.identity.chemical_identity_digest}"


def _exclusion_constraint(
    facts: MolecularFacts,
    quantity: ScientificQuantity,
    candidate_identity: str,
) -> CandidateConstraint:
    return CandidateConstraint(
        identity=_digest(
            {
                "facts": facts.identity.chemical_identity_digest,
                "quantity": quantity.value,
                "excluded_candidate": candidate_identity,
            }
        ),
        constraint_type="excludes_all",
        schema_version=_CONSTRAINT_SCHEMA,
        factor_keys=(_molecular_factor_key(facts, quantity),),
        parameters=(("candidate_ids", (candidate_identity,)),),
    )


def _authoritative_charge_domain(
    result: NetChargeResult,
) -> frozenset[int]:
    if result.admissibility is not ScientificAdmissibility.ADMISSIBLE:
        return frozenset()
    return frozenset(
        candidate.charge
        for candidate in result.candidates
        if candidate.provenance.authority in _AUTHORITATIVE_VALUES
    )


def _electronic_value(
    candidate: ElectronicStateCandidate,
) -> Tuple[int, Optional[int], bool]:
    return candidate.two_s, candidate.multiplicity, candidate.pure_spin


def _authoritative_electronic_domain(
    result: ElectronicStateInferenceResult,
) -> frozenset[Tuple[int, Optional[int], bool]]:
    if result.admissibility is not ScientificAdmissibility.ADMISSIBLE:
        return frozenset()
    return frozenset(
        _electronic_value(candidate)
        for candidate in result.candidates
        if candidate.provenance.authority in _AUTHORITATIVE_VALUES
    )


def _oxidation_compatible(
    first: OxidationStateCandidateEvidence,
    second: OxidationStateCandidateEvidence,
) -> bool:
    return first.convention is second.convention and all(
        left is None or right is None or left == right
        for left, right in zip(first.atom_states, second.atom_states)
    )


def _matches_overrides(
    candidate: OxidationStateCandidateEvidence,
    overrides: Tuple[OxidationStateOverride, ...],
) -> bool:
    return all(
        0 <= override.atom_index < len(candidate.atom_states)
        and (
            candidate.atom_states[override.atom_index] is None
            or candidate.atom_states[override.atom_index]
            == override.oxidation_state
        )
        for override in overrides
    )


def _reconcile_authoritative_results(
    facts: MolecularFacts,
    net_charge_results: Tuple[NetChargeResult, ...],
    oxidation_state_results: Tuple[OxidationStateInferenceResult, ...],
    electronic_state_results: Tuple[ElectronicStateInferenceResult, ...],
    oxidation_overrides: Tuple[OxidationStateOverride, ...] = (),
) -> Tuple[Tuple[CandidateConstraint, ...], bool]:
    constraints = []
    conflict = False

    charge_domains = tuple(
        domain
        for result in net_charge_results
        for domain in (_authoritative_charge_domain(result),)
        if domain
    )
    if len(charge_domains) > 1:
        allowed_charges = frozenset.intersection(*charge_domains)
        conflict = conflict or not allowed_charges
        constraints.extend(
            _exclusion_constraint(
                facts,
                ScientificQuantity.NET_CHARGE,
                candidate.provenance.identity,
            )
            for result in net_charge_results
            for candidate in result.candidates
            if candidate.charge not in allowed_charges
        )

    electronic_domains = tuple(
        domain
        for result in electronic_state_results
        for domain in (_authoritative_electronic_domain(result),)
        if domain
    )
    if len(electronic_domains) > 1:
        allowed_electronic = frozenset.intersection(*electronic_domains)
        conflict = conflict or not allowed_electronic
        constraints.extend(
            _exclusion_constraint(
                facts,
                ScientificQuantity.ELECTRONIC_STATE,
                candidate.provenance.identity,
            )
            for result in electronic_state_results
            for candidate in result.candidates
            if _electronic_value(candidate) not in allowed_electronic
        )

    oxidation_domains = tuple(
        tuple(
            candidate
            for candidate in result.candidates
            if candidate.provenance.authority in _AUTHORITATIVE_VALUES
        )
        for result in oxidation_state_results
        if result.admissibility is ScientificAdmissibility.ADMISSIBLE
        and any(
            candidate.provenance.authority in _AUTHORITATIVE_VALUES
            for candidate in result.candidates
        )
    )
    oxidation_candidates = tuple(
        candidate
        for result in oxidation_state_results
        for candidate in result.candidates
    )
    retained_oxidation_ids = frozenset(
        candidate.provenance.identity
        for candidate in oxidation_candidates
        if _matches_overrides(candidate, oxidation_overrides)
        and all(
            any(
                _oxidation_compatible(candidate, domain_candidate)
                for domain_candidate in domain
            )
            for domain in oxidation_domains
        )
    )
    if oxidation_candidates and (
        oxidation_overrides or len(oxidation_domains) > 1
    ):
        conflict = conflict or not retained_oxidation_ids
        constraints.extend(
            _exclusion_constraint(
                facts,
                ScientificQuantity.OXIDATION_STATE,
                candidate.provenance.identity,
            )
            for candidate in oxidation_candidates
            if candidate.provenance.identity not in retained_oxidation_ids
        )

    constraints_by_identity = {
        constraint.identity: constraint for constraint in constraints
    }
    return (
        tuple(
            constraints_by_identity[identity]
            for identity in sorted(constraints_by_identity)
        ),
        conflict,
    )


def _validate_coupled_candidates(
    facts: MolecularFacts,
    net_charge: NetChargeCandidateEvidence,
    *,
    lewis: Optional[LewisFormalChargeCandidate] = None,
    counting: Optional[CoordinationCountingCandidate] = None,
    oxidation_state: Optional[OxidationStateCandidateEvidence] = None,
    electronic_state: Optional[ElectronicStateCandidate] = None,
) -> legality._HardRuleAssessment:
    return _validate_coupled_candidate_sets(
        facts,
        net_charge,
        lewis_candidates=(lewis,) if lewis is not None else (),
        counting_candidates=(counting,) if counting is not None else (),
        oxidation_state=oxidation_state,
        electronic_state=electronic_state,
    )


def _validate_coupled_candidate_sets(
    facts: MolecularFacts,
    net_charge: NetChargeCandidateEvidence,
    *,
    lewis_candidates: Tuple[LewisFormalChargeCandidate, ...] = (),
    counting_candidates: Tuple[CoordinationCountingCandidate, ...] = (),
    oxidation_state: Optional[OxidationStateCandidateEvidence] = None,
    electronic_state: Optional[ElectronicStateCandidate] = None,
) -> legality._HardRuleAssessment:
    assessments = [legality._validate_charge_conservation(facts, net_charge)]
    assessments.extend(
        legality._validate_lewis_candidate(facts, net_charge, candidate)
        for candidate in lewis_candidates
    )
    assessments.extend(
        legality._validate_counting_candidate_for_facts(facts, candidate)
        for candidate in counting_candidates
    )
    if len({candidate.convention for candidate in counting_candidates}) > 1:
        assessments.append(
            legality._hard_failure(
                "HARD-005.COUNTING_CONVENTION_MISMATCH",
                "One coupled path cannot mix coordination-counting conventions.",
                related_identities=tuple(
                    candidate.provenance.identity
                    for candidate in counting_candidates
                ),
            )
        )
    if oxidation_state is not None and any(
        candidate.convention is not oxidation_state.convention
        for candidate in counting_candidates
    ):
        assessments.append(
            legality._hard_failure(
                "HARD-005.COUNTING_CONVENTION_MISMATCH",
                (
                    "Oxidation-state and coordination-counting candidates in "
                    "one path must use the same convention."
                ),
                related_identities=(
                    oxidation_state.provenance.identity,
                    *tuple(
                        candidate.provenance.identity
                        for candidate in counting_candidates
                    ),
                ),
            )
        )
    if oxidation_state is not None:
        assessments.append(
            legality._validate_oxidation_state_conservation(
                facts,
                net_charge,
                oxidation_state,
            )
        )
        assessments.append(
            legality._validate_counting_reference(
                oxidation_state,
                counting_candidates,
            )
        )
    if electronic_state is not None:
        assessments.append(
            legality._validate_electronic_state_candidate(
                facts,
                net_charge,
                electronic_state,
            )
        )
        assessments.append(
            legality._validate_electronic_oxidation_reference(
                electronic_state,
                (oxidation_state,) if oxidation_state is not None else (),
            )
        )
    return legality._merge_hard_rule_assessments(tuple(assessments))
