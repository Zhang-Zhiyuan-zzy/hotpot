"""Hard compatibility checks for one coupled scientific-state path."""

from __future__ import annotations

from typing import Optional

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
    assessments = [legality._validate_charge_conservation(facts, net_charge)]
    if lewis is not None:
        assessments.append(
            legality._validate_lewis_candidate(facts, net_charge, lewis)
        )
    if counting is not None:
        assessments.append(
            legality._validate_counting_candidate_for_facts(facts, counting)
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
                (counting,) if counting is not None else (),
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
