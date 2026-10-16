"""Private versioned codec for factorized scientific-state spaces."""

from __future__ import annotations

import json
from typing import Dict, Mapping, Optional, Tuple

from ..contracts import CandidateBudget, ScientificQuantity
from ._artifacts import _validate_candidate_space_artifacts
from ._serialization import (
    _boolean,
    _integer,
    _mapping,
    _number,
    _optional_integer,
    _require_keys,
    _sequence,
    _state_candidate_from_payload,
    _state_candidate_to_payload,
    _string,
    _strings,
)
from .contracts import (
    BranchAssignment,
    CandidateConstraint,
    CandidateDomainScope,
    CandidateDomainSummary,
    CandidateFactor,
    CandidateSearchStatus,
    ConstraintValue,
    CoupledScientificStateCandidate,
    DerivedStateValue,
    FactorCandidateValue,
    FactorDomainContribution,
    FactorKey,
    FrontierBound,
    MaterializationStatus,
    RejectedPathSummary,
    ScientificStateCandidateSpace,
    ScientificStateIdentity,
    StateValueOrigin,
    StateValueRef,
)

__all__ = []


_CANDIDATE_SPACE_SCHEMA_VERSION = "scientific-state-candidate-space-v1"


def _optional_string(value: object) -> Optional[str]:
    return None if value is None else _string(value)


def _factor_key_to_payload(factor_key: FactorKey) -> Dict[str, object]:
    return {
        "quantity": factor_key.quantity.value,
        "subject_identity": factor_key.subject_identity,
    }


def _factor_key_from_payload(value: object) -> FactorKey:
    payload = _mapping(value)
    _require_keys(payload, ("quantity", "subject_identity"))
    return FactorKey(
        ScientificQuantity(_string(payload["quantity"])),
        _string(payload["subject_identity"]),
    )


def _constraint_value_to_payload(value: ConstraintValue) -> Dict[str, object]:
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        return {"type": "float", "value": _number(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_constraint_value_to_payload(item) for item in value],
        }
    raise TypeError("Unsupported candidate-constraint value type")


def _constraint_value_from_payload(value: object) -> ConstraintValue:
    payload = _mapping(value)
    if "type" not in payload:
        raise ValueError("Serialized constraint value has no type")
    value_type = _string(payload["type"])
    if value_type == "none":
        _require_keys(payload, ("type",))
        return None
    if value_type == "bool":
        _require_keys(payload, ("type", "value"))
        return _boolean(payload["value"])
    if value_type == "int":
        _require_keys(payload, ("type", "value"))
        return _integer(payload["value"])
    if value_type == "float":
        _require_keys(payload, ("type", "value"))
        return _number(payload["value"])
    if value_type == "str":
        _require_keys(payload, ("type", "value"))
        return _string(payload["value"])
    if value_type == "tuple":
        _require_keys(payload, ("type", "items"))
        return tuple(
            _constraint_value_from_payload(item)
            for item in _sequence(payload["items"])
        )
    raise ValueError(f"Unknown candidate-constraint value type {value_type!r}")


def _frontier_bound_to_payload(bound: FrontierBound) -> Dict[str, object]:
    return {
        "branch_identity": bound.branch_identity,
        "schema_version": bound.schema_version,
        "optimistic_key": list(bound.optimistic_key),
    }


def _frontier_bound_from_payload(value: object) -> FrontierBound:
    payload = _mapping(value)
    _require_keys(
        payload,
        ("branch_identity", "schema_version", "optimistic_key"),
    )
    return FrontierBound(
        _string(payload["branch_identity"]),
        _string(payload["schema_version"]),
        _strings(payload["optimistic_key"]),
    )


def _domain_summary_to_payload(
    summary: CandidateDomainSummary,
) -> Dict[str, object]:
    return {
        "remaining_count_upper_bound": summary.remaining_count_upper_bound,
        "continuation_identity": summary.continuation_identity,
        "frontier_bound": (
            None
            if summary.frontier_bound is None
            else _frontier_bound_to_payload(summary.frontier_bound)
        ),
    }


def _domain_summary_from_payload(value: object) -> CandidateDomainSummary:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "remaining_count_upper_bound",
            "continuation_identity",
            "frontier_bound",
        ),
    )
    frontier_bound = payload["frontier_bound"]
    return CandidateDomainSummary(
        remaining_count_upper_bound=_optional_integer(
            payload["remaining_count_upper_bound"]
        ),
        continuation_identity=_optional_string(
            payload["continuation_identity"]
        ),
        frontier_bound=(
            None
            if frontier_bound is None
            else _frontier_bound_from_payload(frontier_bound)
        ),
    )


def _branch_assignment_to_payload(
    assignment: BranchAssignment,
) -> Dict[str, object]:
    return {
        "group_id": assignment.group_id,
        "option_id": assignment.option_id,
    }


def _branch_assignment_from_payload(value: object) -> BranchAssignment:
    payload = _mapping(value)
    _require_keys(payload, ("group_id", "option_id"))
    return BranchAssignment(
        _string(payload["group_id"]),
        _string(payload["option_id"]),
    )


def _factor_candidate_to_payload(
    value: FactorCandidateValue,
) -> Dict[str, object]:
    return {
        "candidate": _state_candidate_to_payload(value.candidate),
        "source_proposal_ids": list(value.source_proposal_ids),
        "branch_assignments": [
            _branch_assignment_to_payload(assignment)
            for assignment in value.branch_assignments
        ],
    }


def _factor_candidate_from_payload(value: object) -> FactorCandidateValue:
    payload = _mapping(value)
    _require_keys(
        payload,
        ("candidate", "source_proposal_ids", "branch_assignments"),
    )
    return FactorCandidateValue(
        candidate=_state_candidate_from_payload(payload["candidate"]),
        source_proposal_ids=_strings(payload["source_proposal_ids"]),
        branch_assignments=tuple(
            _branch_assignment_from_payload(item)
            for item in _sequence(payload["branch_assignments"])
        ),
    )


def _domain_contribution_to_payload(
    contribution: FactorDomainContribution,
) -> Dict[str, object]:
    return {
        "proposal_id": contribution.proposal_id,
        "domain_scope": contribution.domain_scope.value,
        "materialization_status": contribution.materialization_status.value,
        "unexpanded_domain_summary": (
            None
            if contribution.unexpanded_domain_summary is None
            else _domain_summary_to_payload(
                contribution.unexpanded_domain_summary
            )
        ),
    }


def _domain_contribution_from_payload(
    value: object,
) -> FactorDomainContribution:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "proposal_id",
            "domain_scope",
            "materialization_status",
            "unexpanded_domain_summary",
        ),
    )
    summary = payload["unexpanded_domain_summary"]
    return FactorDomainContribution(
        proposal_id=_string(payload["proposal_id"]),
        domain_scope=CandidateDomainScope(_string(payload["domain_scope"])),
        materialization_status=MaterializationStatus(
            _string(payload["materialization_status"])
        ),
        unexpanded_domain_summary=(
            None if summary is None else _domain_summary_from_payload(summary)
        ),
    )


def _factor_to_payload(factor: CandidateFactor) -> Dict[str, object]:
    return {
        "identity": factor.identity,
        "factor_key": _factor_key_to_payload(factor.factor_key),
        "domain_definition_id": factor.domain_definition_id,
        "values": [
            _factor_candidate_to_payload(value) for value in factor.values
        ],
        "domain_contributions": [
            _domain_contribution_to_payload(contribution)
            for contribution in factor.domain_contributions
        ],
        "unexpanded_domain_summary": (
            None
            if factor.unexpanded_domain_summary is None
            else _domain_summary_to_payload(factor.unexpanded_domain_summary)
        ),
        "domain_scope": factor.domain_scope.value,
        "materialization_status": factor.materialization_status.value,
    }


def _factor_from_payload(value: object) -> CandidateFactor:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "identity",
            "factor_key",
            "domain_definition_id",
            "values",
            "domain_contributions",
            "unexpanded_domain_summary",
            "domain_scope",
            "materialization_status",
        ),
    )
    summary = payload["unexpanded_domain_summary"]
    return CandidateFactor(
        identity=_string(payload["identity"]),
        factor_key=_factor_key_from_payload(payload["factor_key"]),
        domain_definition_id=_string(payload["domain_definition_id"]),
        values=tuple(
            _factor_candidate_from_payload(item)
            for item in _sequence(payload["values"])
        ),
        domain_contributions=tuple(
            _domain_contribution_from_payload(item)
            for item in _sequence(payload["domain_contributions"])
        ),
        unexpanded_domain_summary=(
            None if summary is None else _domain_summary_from_payload(summary)
        ),
        domain_scope=CandidateDomainScope(_string(payload["domain_scope"])),
        materialization_status=MaterializationStatus(
            _string(payload["materialization_status"])
        ),
    )


def _constraint_to_payload(
    constraint: CandidateConstraint,
) -> Dict[str, object]:
    return {
        "identity": constraint.identity,
        "constraint_type": constraint.constraint_type,
        "schema_version": constraint.schema_version,
        "factor_keys": [
            _factor_key_to_payload(key) for key in constraint.factor_keys
        ],
        "parameters": [
            {
                "name": name,
                "value": _constraint_value_to_payload(value),
            }
            for name, value in constraint.parameters
        ],
    }


def _constraint_from_payload(value: object) -> CandidateConstraint:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "identity",
            "constraint_type",
            "schema_version",
            "factor_keys",
            "parameters",
        ),
    )
    parameters = []
    for item in _sequence(payload["parameters"]):
        parameter = _mapping(item)
        _require_keys(parameter, ("name", "value"))
        parameters.append(
            (
                _string(parameter["name"]),
                _constraint_value_from_payload(parameter["value"]),
            )
        )
    return CandidateConstraint(
        identity=_string(payload["identity"]),
        constraint_type=_string(payload["constraint_type"]),
        schema_version=_string(payload["schema_version"]),
        factor_keys=tuple(
            _factor_key_from_payload(item)
            for item in _sequence(payload["factor_keys"])
        ),
        parameters=tuple(parameters),
    )


def _state_value_ref_to_payload(value: StateValueRef) -> Dict[str, object]:
    return {
        "quantity": value.quantity.value,
        "value": value.value,
        "origin": value.origin.value,
        "origin_identity": value.origin_identity,
    }


def _state_value_ref_from_payload(value: object) -> StateValueRef:
    payload = _mapping(value)
    _require_keys(
        payload,
        ("quantity", "value", "origin", "origin_identity"),
    )
    return StateValueRef(
        quantity=ScientificQuantity(_string(payload["quantity"])),
        value=_integer(payload["value"]),
        origin=StateValueOrigin(_string(payload["origin"])),
        origin_identity=_string(payload["origin_identity"]),
    )


def _coupled_path_to_payload(
    path: CoupledScientificStateCandidate,
) -> Dict[str, object]:
    return {
        "identity": path.identity.digest,
        "net_charge": _state_value_ref_to_payload(path.net_charge),
        "component_assignment_ids": list(path.component_assignment_ids),
        "coordination_assignment_ids": list(
            path.coordination_assignment_ids
        ),
        "oxidation_state_assignment_id": path.oxidation_state_assignment_id,
        "electronic_state_result_id": path.electronic_state_result_id,
        "electronic_state_candidate_identity": (
            path.electronic_state_candidate_identity
        ),
        "derived_values": [
            _state_candidate_to_payload(value) for value in path.derived_values
        ],
        "evidence_ids": list(path.evidence_ids),
        "assumption_ids": list(path.assumption_ids),
    }


def _derived_value_from_payload(value: object) -> DerivedStateValue:
    candidate = _state_candidate_from_payload(value)
    if not isinstance(candidate, DerivedStateValue):
        raise ValueError("Coupled-path derived value has the wrong type")
    return candidate


def _coupled_path_from_payload(
    value: object,
) -> CoupledScientificStateCandidate:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "identity",
            "net_charge",
            "component_assignment_ids",
            "coordination_assignment_ids",
            "oxidation_state_assignment_id",
            "electronic_state_result_id",
            "electronic_state_candidate_identity",
            "derived_values",
            "evidence_ids",
            "assumption_ids",
        ),
    )
    return CoupledScientificStateCandidate(
        identity=ScientificStateIdentity(_string(payload["identity"])),
        net_charge=_state_value_ref_from_payload(payload["net_charge"]),
        component_assignment_ids=_strings(
            payload["component_assignment_ids"]
        ),
        coordination_assignment_ids=_strings(
            payload["coordination_assignment_ids"]
        ),
        oxidation_state_assignment_id=_optional_string(
            payload["oxidation_state_assignment_id"]
        ),
        electronic_state_result_id=_string(
            payload["electronic_state_result_id"]
        ),
        electronic_state_candidate_identity=_optional_string(
            payload["electronic_state_candidate_identity"]
        ),
        derived_values=tuple(
            _derived_value_from_payload(item)
            for item in _sequence(payload["derived_values"])
        ),
        evidence_ids=_strings(payload["evidence_ids"]),
        assumption_ids=_strings(payload["assumption_ids"]),
    )


def _rejected_path_to_payload(
    summary: RejectedPathSummary,
) -> Dict[str, object]:
    return {
        "candidate_identities": list(summary.candidate_identities),
        "constraint_identity": summary.constraint_identity,
        "reason_codes": list(summary.reason_codes),
    }


def _rejected_path_from_payload(value: object) -> RejectedPathSummary:
    payload = _mapping(value)
    _require_keys(
        payload,
        ("candidate_identities", "constraint_identity", "reason_codes"),
    )
    return RejectedPathSummary(
        _strings(payload["candidate_identities"]),
        _string(payload["constraint_identity"]),
        _strings(payload["reason_codes"]),
    )


def _candidate_budget_to_payload(
    candidate_budget: CandidateBudget,
) -> Dict[str, object]:
    return {
        "max_materialized_values_per_factor": (
            candidate_budget.max_materialized_values_per_factor
        ),
        "max_expanded_paths": candidate_budget.max_expanded_paths,
        "max_frontier_paths": candidate_budget.max_frontier_paths,
    }


def _candidate_budget_from_payload(value: object) -> CandidateBudget:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "max_materialized_values_per_factor",
            "max_expanded_paths",
            "max_frontier_paths",
        ),
    )
    return CandidateBudget(
        _integer(payload["max_materialized_values_per_factor"]),
        _integer(payload["max_expanded_paths"]),
        _integer(payload["max_frontier_paths"]),
    )


def _candidate_space_to_payload(
    candidate_space: ScientificStateCandidateSpace,
) -> Dict[str, object]:
    _validate_candidate_space_artifacts(candidate_space)
    return {
        "schema_version": _CANDIDATE_SPACE_SCHEMA_VERSION,
        "factors": [
            _factor_to_payload(factor) for factor in candidate_space.factors
        ],
        "constraints": [
            _constraint_to_payload(constraint)
            for constraint in candidate_space.constraints
        ],
        "candidate_budget": _candidate_budget_to_payload(
            candidate_space.candidate_budget
        ),
        "candidate_budget_artifact_id": (
            candidate_space.candidate_budget_artifact_id
        ),
        "frontier_bound_schema_version": (
            candidate_space.frontier_bound_schema_version
        ),
        "expanded_paths": [
            _coupled_path_to_payload(path)
            for path in candidate_space.expanded_paths
        ],
        "rejected_path_summaries": [
            _rejected_path_to_payload(summary)
            for summary in candidate_space.rejected_path_summaries
        ],
        "frontier": [
            _frontier_bound_to_payload(bound)
            for bound in candidate_space.frontier
        ],
        "search_status": candidate_space.search_status.value,
    }


def _candidate_space_from_payload(
    value: object,
) -> ScientificStateCandidateSpace:
    payload = _mapping(value)
    _require_keys(
        payload,
        (
            "schema_version",
            "factors",
            "constraints",
            "candidate_budget",
            "candidate_budget_artifact_id",
            "frontier_bound_schema_version",
            "expanded_paths",
            "rejected_path_summaries",
            "frontier",
            "search_status",
        ),
    )
    schema_version = _string(payload["schema_version"])
    if schema_version != _CANDIDATE_SPACE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported candidate-space schema version {schema_version!r}"
        )
    candidate_space = ScientificStateCandidateSpace(
        factors=tuple(
            _factor_from_payload(item)
            for item in _sequence(payload["factors"])
        ),
        constraints=tuple(
            _constraint_from_payload(item)
            for item in _sequence(payload["constraints"])
        ),
        candidate_budget=_candidate_budget_from_payload(
            payload["candidate_budget"]
        ),
        candidate_budget_artifact_id=_string(
            payload["candidate_budget_artifact_id"]
        ),
        frontier_bound_schema_version=_string(
            payload["frontier_bound_schema_version"]
        ),
        expanded_paths=tuple(
            _coupled_path_from_payload(item)
            for item in _sequence(payload["expanded_paths"])
        ),
        rejected_path_summaries=tuple(
            _rejected_path_from_payload(item)
            for item in _sequence(payload["rejected_path_summaries"])
        ),
        frontier=tuple(
            _frontier_bound_from_payload(item)
            for item in _sequence(payload["frontier"])
        ),
        search_status=CandidateSearchStatus(
            _string(payload["search_status"])
        ),
    )
    _validate_candidate_space_artifacts(candidate_space)
    return candidate_space


def _candidate_space_to_json(
    candidate_space: ScientificStateCandidateSpace,
) -> str:
    return json.dumps(
        _candidate_space_to_payload(candidate_space),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _candidate_space_from_json(value: str) -> ScientificStateCandidateSpace:
    return _candidate_space_from_payload(json.loads(value))
