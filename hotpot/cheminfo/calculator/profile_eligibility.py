"""Static eligibility and immutable identity of scientific-state profiles."""

from __future__ import annotations

import hashlib
import json

from .contracts import (
    CandidateBudget,
    ProviderSet,
    ScientificOperationRequirements,
    StatePolicy,
)
from .state_rules import RuleSet
from .state_selection import (
    StaticEligibilityFinding,
    StaticProfileEligibility,
)

__all__ = []


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _effective_profile_identity(
    policy: StatePolicy,
    requirements: ScientificOperationRequirements,
    provider_set: ProviderSet,
    rule_set: RuleSet,
    candidate_budget: CandidateBudget,
) -> str:
    provider_configuration = tuple(
        sorted(
            (
                toggle.provider_id,
                toggle.enabled,
                toggle.configuration,
            )
            for toggle in provider_set.toggles
        )
    )
    rule_configuration = tuple(
        (
            activated.rule.specification.rule_id,
            activated.rule.specification.rule_version,
            activated.toggle.enabled,
            activated.toggle.configuration,
        )
        for layer in rule_set.layers
        for activated in layer
    )
    return _digest(
        {
            "policy": policy.value,
            "requirements": {
                "operation_id": requirements.operation_id,
                "required": tuple(
                    sorted(
                        quantity.value
                        for quantity in requirements.required_resolved_quantities
                    )
                ),
                "allowed_unresolved": tuple(
                    sorted(
                        quantity.value
                        for quantity in requirements.allowed_unresolved_quantities
                    )
                ),
            },
            "providers": provider_configuration,
            "rules": rule_configuration,
            "candidate_budget": (
                candidate_budget.max_materialized_values_per_factor,
                candidate_budget.max_expanded_paths,
                candidate_budget.max_frontier_paths,
            ),
        }
    )


def _static_profile_eligibility(
    policy: StatePolicy,
    requirements: ScientificOperationRequirements,
    provider_set: ProviderSet,
    rule_set: RuleSet,
    candidate_budget: CandidateBudget,
) -> StaticProfileEligibility:
    identity = _effective_profile_identity(
        policy,
        requirements,
        provider_set,
        rule_set,
        candidate_budget,
    )
    if policy is StatePolicy.STRICT:
        return StaticProfileEligibility(
            profile_id=policy.value,
            effective_profile_identity=identity,
            in_domain=True,
            findings=(),
        )
    return StaticProfileEligibility(
        profile_id=policy.value,
        effective_profile_identity=identity,
        in_domain=False,
        findings=(
            StaticEligibilityFinding(
                "PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE",
                (
                    "The reviewed coordination-screening applicability "
                    "artifact is not active in this resolver version."
                ),
            ),
        ),
    )
