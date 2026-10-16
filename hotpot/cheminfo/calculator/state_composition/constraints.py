"""Hard compatibility checks for one coupled scientific-state path."""

from __future__ import annotations

from typing import Optional, Tuple

from .. import legality
from ..charge.contracts import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
)
from ..contracts import MolecularFacts
from ..electronic_state.contracts import ElectronicStateCandidate
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    OxidationStateCandidateEvidence,
)

__all__ = []


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
