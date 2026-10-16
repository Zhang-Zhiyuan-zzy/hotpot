"""Private JSON codec for complete optional-rule traces."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..contracts import StateValueAuthority
from ..state_composition._candidate_space_serialization import (
    _candidate_proposal_from_payload,
    _candidate_proposal_to_payload,
)
from ..state_composition._serialization import (
    _boolean,
    _mapping,
    _number,
    _require_keys,
    _sequence,
    _string,
    _strings,
)
from .contracts import (
    EvidenceItem,
    RankAdjustment,
    RuleApplicability,
    RuleOutcome,
    RulePhase,
    RuleTrace,
)

__all__ = []


def _optional_string(value: object) -> Optional[str]:
    return None if value is None else _string(value)


def _evidence_to_payload(evidence: EvidenceItem) -> Dict[str, object]:
    return {
        "identity": evidence.identity,
        "authority": evidence.authority.value,
        "source_identity": evidence.source_identity,
        "claim": evidence.claim,
        "locator": evidence.locator,
    }


def _evidence_from_payload(value: object) -> EvidenceItem:
    payload = _mapping(value)
    _require_keys(
        payload,
        ("identity", "authority", "source_identity", "claim", "locator"),
    )
    return EvidenceItem(
        identity=_string(payload["identity"]),
        authority=StateValueAuthority(_string(payload["authority"])),
        source_identity=_string(payload["source_identity"]),
        claim=_string(payload["claim"]),
        locator=_optional_string(payload["locator"]),
    )


def _rank_adjustment_to_payload(
    adjustment: RankAdjustment,
) -> Dict[str, object]:
    return {
        "target_identity": adjustment.target_identity,
        "rank_dimension": adjustment.rank_dimension,
        "value": _number(adjustment.value),
        "reason": adjustment.reason,
        "assumption_ids": list(adjustment.assumption_ids),
    }


def _rank_adjustment_from_payload(value: object) -> RankAdjustment:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "target_identity",
            "rank_dimension",
            "value",
            "reason",
            "assumption_ids",
        ),
    )
    return RankAdjustment(
        target_identity=_string(payload["target_identity"]),
        rank_dimension=_string(payload["rank_dimension"]),
        value=_number(payload["value"]),
        reason=_string(payload["reason"]),
        assumption_ids=_strings(payload["assumption_ids"]),
    )


def _applicability_to_payload(
    applicability: RuleApplicability,
) -> Dict[str, object]:
    return {
        "matches": applicability.matches,
        "reason_codes": list(applicability.reason_codes),
    }


def _applicability_from_payload(value: object) -> RuleApplicability:
    payload = _mapping(value)
    _require_keys(payload, ("matches", "reason_codes"))
    return RuleApplicability(
        _boolean(payload["matches"]),
        _strings(payload["reason_codes"]),
    )


def _rule_outcome_to_payload(outcome: RuleOutcome) -> Dict[str, object]:
    return {
        "rule_id": outcome.rule_id,
        "rule_version": outcome.rule_version,
        "applicability": _applicability_to_payload(outcome.applicability),
        "candidate_proposal": (
            None
            if outcome.candidate_proposal is None
            else _candidate_proposal_to_payload(outcome.candidate_proposal)
        ),
        "evidence": [
            _evidence_to_payload(evidence) for evidence in outcome.evidence
        ],
        "rank_adjustments": [
            _rank_adjustment_to_payload(adjustment)
            for adjustment in outcome.rank_adjustments
        ],
        "assumptions": list(outcome.assumptions),
        "warnings": list(outcome.warnings),
    }


def _rule_outcome_from_payload(value: object) -> RuleOutcome:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "rule_id",
            "rule_version",
            "applicability",
            "candidate_proposal",
            "evidence",
            "rank_adjustments",
            "assumptions",
            "warnings",
        ),
    )
    proposal = payload["candidate_proposal"]
    return RuleOutcome(
        rule_id=_string(payload["rule_id"]),
        rule_version=_string(payload["rule_version"]),
        applicability=_applicability_from_payload(payload["applicability"]),
        candidate_proposal=(
            None
            if proposal is None
            else _candidate_proposal_from_payload(proposal)
        ),
        evidence=tuple(
            _evidence_from_payload(item)
            for item in _sequence(payload["evidence"])
        ),
        rank_adjustments=tuple(
            _rank_adjustment_from_payload(item)
            for item in _sequence(payload["rank_adjustments"])
        ),
        assumptions=_strings(payload["assumptions"]),
        warnings=_strings(payload["warnings"]),
    )


def _rule_trace_to_payload(trace: RuleTrace) -> Dict[str, object]:
    return {
        "phase": trace.phase.value,
        "outcome": _rule_outcome_to_payload(trace.outcome),
    }


def _rule_trace_from_payload(value: object) -> RuleTrace:
    payload = _mapping(value)
    _require_keys(payload, ("phase", "outcome"))
    return RuleTrace(
        RulePhase(_string(payload["phase"])),
        _rule_outcome_from_payload(payload["outcome"]),
    )


def _rule_traces_to_payload(
    traces: Tuple[RuleTrace, ...],
) -> list[object]:
    return [_rule_trace_to_payload(trace) for trace in traces]


def _rule_traces_from_payload(value: object) -> Tuple[RuleTrace, ...]:
    return tuple(
        _rule_trace_from_payload(item) for item in _sequence(value)
    )
