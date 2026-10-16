"""Versioned immutable artifacts for named scientific-state profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Dict, Tuple

from ..contracts import CandidateBudget, ScientificQuantity, StatePolicy

__all__ = []


_PROFILE_SCHEMA = "hotpot-scientific-state-profile-v1"
_PROFILE_RESOURCE = (
    "data/profiles/coordination_screening_v1.json"
)


@dataclass(frozen=True)
class _ScientificStateProfileArtifact:
    profile_id: str
    profile_version: str
    required_quantities: frozenset[ScientificQuantity]
    allowed_unresolved_quantities: frozenset[ScientificQuantity]
    required_provider_ids: frozenset[str]
    required_rule_versions: Tuple[Tuple[str, str], ...]
    decision_rank_dimensions: Tuple[str, ...]
    assumption_severity: Tuple[Tuple[str, int], ...]
    candidate_budget: CandidateBudget
    frontier_schema_version: str
    tie_policy: str
    content_digest: str


def _profile_payload() -> Tuple[bytes, Dict[str, object]]:
    resource = resources.files(__package__).joinpath(_PROFILE_RESOURCE)
    encoded = resource.read_bytes()
    return encoded, json.loads(encoded.decode("utf-8"))


def _string_tuple(value: object, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"Profile field {field!r} must contain strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"Profile field {field!r} contains duplicates")
    return result


def _string_mapping(
    value: object,
    field: str,
) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        or not item
        for key, item in value.items()
    ):
        raise ValueError(f"Profile field {field!r} must map strings")
    return tuple(sorted(value.items()))


def _severity_mapping(value: object) -> Tuple[Tuple[str, int], ...]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not key
        or type(item) is not int
        or item < 0
        for key, item in value.items()
    ):
        raise ValueError("Profile assumption severities must be non-negative")
    severities = tuple(sorted(value.items()))
    if len({severity for _, severity in severities}) != len(severities):
        raise ValueError("Profile assumption severities must be distinct")
    return severities


def _candidate_budget(value: object) -> CandidateBudget:
    if not isinstance(value, dict):
        raise ValueError("Profile candidate budget must be an object")
    expected = {
        "max_materialized_values_per_factor",
        "max_expanded_paths",
        "max_frontier_paths",
    }
    if set(value) != expected:
        raise ValueError("Profile candidate budget fields do not match")
    return CandidateBudget(
        value["max_materialized_values_per_factor"],
        value["max_expanded_paths"],
        value["max_frontier_paths"],
    )


@lru_cache(maxsize=1)
def _coordination_screening_profile() -> _ScientificStateProfileArtifact:
    encoded, payload = _profile_payload()
    expected_fields = {
        "schema_version",
        "profile_id",
        "profile_version",
        "required_quantities",
        "allowed_unresolved_quantities",
        "required_provider_ids",
        "required_rule_versions",
        "decision_rank_dimensions",
        "assumption_severity",
        "candidate_budget",
        "frontier_schema_version",
        "tie_policy",
    }
    if set(payload) != expected_fields:
        raise ValueError("Scientific-state profile fields do not match schema")
    if payload["schema_version"] != _PROFILE_SCHEMA:
        raise ValueError("Unsupported scientific-state profile schema")
    if payload["profile_id"] != StatePolicy.COORDINATION_SCREENING_V1.value:
        raise ValueError("Scientific-state profile ID does not match policy")

    required = frozenset(
        ScientificQuantity(value)
        for value in _string_tuple(
            payload["required_quantities"],
            "required_quantities",
        )
    )
    allowed = frozenset(
        ScientificQuantity(value)
        for value in _string_tuple(
            payload["allowed_unresolved_quantities"],
            "allowed_unresolved_quantities",
        )
    )
    if required & allowed:
        raise ValueError("Profile required and unresolved quantities overlap")
    rank_dimensions = _string_tuple(
        payload["decision_rank_dimensions"],
        "decision_rank_dimensions",
    )
    if not rank_dimensions:
        raise ValueError("Screening profile requires decision rank dimensions")
    frontier_schema = payload["frontier_schema_version"]
    tie_policy = payload["tie_policy"]
    profile_version = payload["profile_version"]
    if any(
        not isinstance(value, str) or not value
        for value in (frontier_schema, tie_policy, profile_version)
    ):
        raise ValueError("Profile version and policy fields must be strings")

    return _ScientificStateProfileArtifact(
        profile_id=payload["profile_id"],
        profile_version=profile_version,
        required_quantities=required,
        allowed_unresolved_quantities=allowed,
        required_provider_ids=frozenset(
            _string_tuple(
                payload["required_provider_ids"],
                "required_provider_ids",
            )
        ),
        required_rule_versions=_string_mapping(
            payload["required_rule_versions"],
            "required_rule_versions",
        ),
        decision_rank_dimensions=rank_dimensions,
        assumption_severity=_severity_mapping(
            payload["assumption_severity"]
        ),
        candidate_budget=_candidate_budget(payload["candidate_budget"]),
        frontier_schema_version=frontier_schema,
        tie_policy=tie_policy,
        content_digest=hashlib.sha256(encoded).hexdigest(),
    )
