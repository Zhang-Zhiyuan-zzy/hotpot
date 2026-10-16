"""Private versioned snapshot of candidate-space decision evidence."""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

from ..contracts import ResolutionStatus
from ..state_selection import (
    DecisionCertifiability,
    DecisionCertificationStatus,
    DecisionReason,
    ExecutionStateDecision,
    StateDecisionAction,
    StateDecisionSource,
    StaticEligibilityFinding,
    StaticProfileEligibility,
)
from ..state_rules._serialization import (
    _rule_traces_from_payload,
    _rule_traces_to_payload,
)
from ..state_rules.contracts import RulePhase, RuleTrace
from ._candidate_space_serialization import (
    _candidate_space_from_payload,
    _candidate_space_to_payload,
)
from ._serialization import (
    _boolean,
    _mapping,
    _require_keys,
    _sequence,
    _string,
    _strings,
)
from .contracts import (
    ScientificResolutionResult,
    ScientificStateCandidateSpace,
    ScientificStateIdentity,
)
from .resolution import _resolve_candidate_space

__all__ = []


_DECISION_SNAPSHOT_SCHEMA_VERSION = "scientific-state-decision-snapshot-v2"


def _optional_identity(value: object) -> Optional[ScientificStateIdentity]:
    return (
        None
        if value is None
        else ScientificStateIdentity(_string(value))
    )


def _resolution_to_payload(
    resolution: ScientificResolutionResult,
) -> Dict[str, object]:
    return {
        "status": resolution.status.value,
        "exhaustive": resolution.exhaustive,
        "equivalent_state_identities": [
            identity.digest
            for identity in resolution.equivalent_state_identities
        ],
    }


def _resolution_from_payload(value: object) -> ScientificResolutionResult:
    payload = _mapping(value)
    _require_keys(
        payload,
        ("status", "exhaustive", "equivalent_state_identities"),
    )
    return ScientificResolutionResult(
        ResolutionStatus(_string(payload["status"])),
        _boolean(payload["exhaustive"]),
        tuple(
            ScientificStateIdentity(identity)
            for identity in _strings(payload["equivalent_state_identities"])
        ),
    )


def _eligibility_finding_to_payload(
    finding: StaticEligibilityFinding,
) -> Dict[str, object]:
    return {"code": finding.code, "message": finding.message}


def _eligibility_finding_from_payload(
    value: object,
) -> StaticEligibilityFinding:
    payload = _mapping(value)
    _require_keys(payload, ("code", "message"))
    return StaticEligibilityFinding(
        _string(payload["code"]),
        _string(payload["message"]),
    )


def _static_eligibility_to_payload(
    eligibility: StaticProfileEligibility,
) -> Dict[str, object]:
    return {
        "profile_id": eligibility.profile_id,
        "effective_profile_identity": eligibility.effective_profile_identity,
        "in_domain": eligibility.in_domain,
        "findings": [
            _eligibility_finding_to_payload(finding)
            for finding in eligibility.findings
        ],
    }


def _static_eligibility_from_payload(
    value: object,
) -> StaticProfileEligibility:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "profile_id",
            "effective_profile_identity",
            "in_domain",
            "findings",
        ),
    )
    return StaticProfileEligibility(
        profile_id=_string(payload["profile_id"]),
        effective_profile_identity=_string(
            payload["effective_profile_identity"]
        ),
        in_domain=_boolean(payload["in_domain"]),
        findings=tuple(
            _eligibility_finding_from_payload(item)
            for item in _sequence(payload["findings"])
        ),
    )


def _decision_reason_to_payload(reason: DecisionReason) -> Dict[str, object]:
    return {
        "code": reason.code,
        "message": reason.message,
        "related_identities": list(reason.related_identities),
    }


def _decision_reason_from_payload(value: object) -> DecisionReason:
    payload = _mapping(value)
    _require_keys(payload, ("code", "message", "related_identities"))
    return DecisionReason(
        _string(payload["code"]),
        _string(payload["message"]),
        _strings(payload["related_identities"]),
    )


def _certifiability_to_payload(
    certifiability: DecisionCertifiability,
) -> Dict[str, object]:
    return {
        "status": certifiability.status.value,
        "findings": [
            _decision_reason_to_payload(finding)
            for finding in certifiability.findings
        ],
    }


def _certifiability_from_payload(value: object) -> DecisionCertifiability:
    payload = _mapping(value)
    _require_keys(payload, ("status", "findings"))
    return DecisionCertifiability(
        DecisionCertificationStatus(_string(payload["status"])),
        tuple(
            _decision_reason_from_payload(item)
            for item in _sequence(payload["findings"])
        ),
    )


def _execution_decision_to_payload(
    decision: ExecutionStateDecision,
) -> Dict[str, object]:
    return {
        "selected_state_identity": (
            None
            if decision.selected_state_identity is None
            else decision.selected_state_identity.digest
        ),
        "action": decision.action.value,
        "source": decision.source.value,
        "static_profile_eligibility": _static_eligibility_to_payload(
            decision.static_profile_eligibility
        ),
        "decision_certifiability": _certifiability_to_payload(
            decision.decision_certifiability
        ),
        "rationale": [
            _decision_reason_to_payload(reason)
            for reason in decision.rationale
        ],
    }


def _execution_decision_from_payload(value: object) -> ExecutionStateDecision:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "selected_state_identity",
            "action",
            "source",
            "static_profile_eligibility",
            "decision_certifiability",
            "rationale",
        ),
    )
    return ExecutionStateDecision(
        selected_state_identity=_optional_identity(
            payload["selected_state_identity"]
        ),
        action=StateDecisionAction(_string(payload["action"])),
        source=StateDecisionSource(_string(payload["source"])),
        static_profile_eligibility=_static_eligibility_from_payload(
            payload["static_profile_eligibility"]
        ),
        decision_certifiability=_certifiability_from_payload(
            payload["decision_certifiability"]
        ),
        rationale=tuple(
            _decision_reason_from_payload(item)
            for item in _sequence(payload["rationale"])
        ),
    )


def _validate_decision_snapshot(
    candidate_space: ScientificStateCandidateSpace,
    resolution: ScientificResolutionResult,
    decision: ExecutionStateDecision,
    rule_trace: Tuple[RuleTrace, ...],
) -> None:
    if resolution != _resolve_candidate_space(candidate_space):
        raise ValueError(
            "Scientific resolution does not match the candidate space"
        )
    selected_identity = decision.selected_state_identity
    path_identities = frozenset(
        path.identity for path in candidate_space.expanded_paths
    )
    if (
        selected_identity is not None
        and selected_identity not in path_identities
    ):
        raise ValueError(
            "Selected scientific-state identity is absent from candidate space"
        )
    trace_ids = tuple(trace.outcome.rule_id for trace in rule_trace)
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("Rule trace contains duplicate rule identities")
    phase_index = {phase: index for index, phase in enumerate(RulePhase)}
    phase_order = tuple(phase_index[trace.phase] for trace in rule_trace)
    if phase_order != tuple(sorted(phase_order)):
        raise ValueError("Rule trace phases are not ordered")
    contribution_ids = frozenset(
        contribution.proposal_id
        for factor in candidate_space.factors
        for contribution in factor.domain_contributions
    )
    constraint_ids = frozenset(
        constraint.identity for constraint in candidate_space.constraints
    )
    for trace in rule_trace:
        proposal = trace.outcome.candidate_proposal
        if proposal is None:
            continue
        if proposal.factors and proposal.proposal_id not in contribution_ids:
            raise ValueError("Rule proposal is absent from candidate factors")
        proposed_constraint_ids = frozenset(
            guarded.constraint.identity for guarded in proposal.constraints
        )
        if not proposed_constraint_ids.issubset(constraint_ids):
            raise ValueError("Rule proposal constraints are absent from space")


def _decision_snapshot_to_payload(
    candidate_space: ScientificStateCandidateSpace,
    resolution: ScientificResolutionResult,
    decision: ExecutionStateDecision,
    rule_trace: Tuple[RuleTrace, ...],
) -> Dict[str, object]:
    _validate_decision_snapshot(
        candidate_space,
        resolution,
        decision,
        rule_trace,
    )
    return {
        "schema_version": _DECISION_SNAPSHOT_SCHEMA_VERSION,
        "candidate_space": _candidate_space_to_payload(candidate_space),
        "scientific_resolution": _resolution_to_payload(resolution),
        "execution_decision": _execution_decision_to_payload(decision),
        "rule_trace": _rule_traces_to_payload(rule_trace),
    }


def _decision_snapshot_from_payload(
    value: object,
) -> Tuple[
    ScientificStateCandidateSpace,
    ScientificResolutionResult,
    ExecutionStateDecision,
    Tuple[RuleTrace, ...],
]:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "schema_version",
            "candidate_space",
            "scientific_resolution",
            "execution_decision",
            "rule_trace",
        ),
    )
    schema_version = _string(payload["schema_version"])
    if schema_version != _DECISION_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported decision-snapshot schema version {schema_version!r}"
        )
    candidate_space = _candidate_space_from_payload(
        payload["candidate_space"]
    )
    resolution = _resolution_from_payload(payload["scientific_resolution"])
    decision = _execution_decision_from_payload(
        payload["execution_decision"]
    )
    rule_trace = _rule_traces_from_payload(payload["rule_trace"])
    _validate_decision_snapshot(
        candidate_space,
        resolution,
        decision,
        rule_trace,
    )
    return candidate_space, resolution, decision, rule_trace


def _decision_snapshot_to_json(
    candidate_space: ScientificStateCandidateSpace,
    resolution: ScientificResolutionResult,
    decision: ExecutionStateDecision,
    rule_trace: Tuple[RuleTrace, ...],
) -> str:
    return json.dumps(
        _decision_snapshot_to_payload(
            candidate_space,
            resolution,
            decision,
            rule_trace,
        ),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decision_snapshot_from_json(
    value: str,
) -> Tuple[
    ScientificStateCandidateSpace,
    ScientificResolutionResult,
    ExecutionStateDecision,
    Tuple[RuleTrace, ...],
]:
    return _decision_snapshot_from_payload(json.loads(value))
