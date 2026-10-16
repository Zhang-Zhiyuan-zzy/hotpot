"""Static eligibility and immutable identity of scientific-state profiles."""

from __future__ import annotations

import hashlib
import json

from .contracts import (
    CandidateBudget,
    MolecularFacts,
    ProviderSet,
    ScientificOperationRequirements,
    StatePolicy,
)
from .state_rules import RuleSet
from .state_rules.profiles import _coordination_screening_profile
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
    facts: MolecularFacts,
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
    profile_artifact_digest = (
        _coordination_screening_profile().content_digest
        if policy is StatePolicy.COORDINATION_SCREENING_V1
        else None
    )
    return _digest(
        {
            "facts_identity": facts.identity.chemical_identity_digest,
            "policy": policy.value,
            "profile_artifact_digest": profile_artifact_digest,
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
    facts: MolecularFacts,
    policy: StatePolicy,
    requirements: ScientificOperationRequirements,
    provider_set: ProviderSet,
    rule_set: RuleSet,
    candidate_budget: CandidateBudget,
) -> StaticProfileEligibility:
    identity = _effective_profile_identity(
        facts,
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
    profile = _coordination_screening_profile()
    findings = []
    if profile.activation_status != "active":
        findings.append(
            StaticEligibilityFinding(
                "PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE",
                "The coordination-screening profile is not approved for activation.",
            )
        )
    if not profile.required_quantities.issubset(
        requirements.required_resolved_quantities
    ):
        findings.append(
            StaticEligibilityFinding(
                "PROFILE.REQUIRED_QUANTITY_MISSING",
                "The operation omits a mandatory profile quantity.",
            )
        )
    enabled_providers = frozenset(
        toggle.provider_id
        for toggle in provider_set.toggles
        if toggle.enabled
    )
    if not profile.required_provider_ids.issubset(enabled_providers):
        findings.append(
            StaticEligibilityFinding(
                "PROFILE.REQUIRED_PROVIDER_MISSING",
                "The effective provider set omits a profile provider.",
            )
        )
    enabled_rule_versions = tuple(
        sorted(
            (
                activated.rule.specification.rule_id,
                activated.rule.specification.rule_version,
            )
            for layer in rule_set.layers
            for activated in layer
            if activated.toggle.enabled
        )
    )
    if enabled_rule_versions != profile.required_rule_versions:
        findings.append(
            StaticEligibilityFinding(
                "PROFILE.RULE_SET_MISMATCH",
                "The effective rule set differs from the reviewed profile.",
            )
        )
    if candidate_budget != profile.candidate_budget:
        findings.append(
            StaticEligibilityFinding(
                "PROFILE.CANDIDATE_BUDGET_MISMATCH",
                "The candidate budget differs from the reviewed profile.",
            )
        )
    has_coordination_center = any(
        bond.kind == "coordination"
        and any(
            facts.atomic_numbers[index]
            in profile.coordination_center_atomic_numbers
            for index in bond.atom_indices
        )
        for bond in facts.semantic_bonds
    )
    if not has_coordination_center:
        findings.append(
            StaticEligibilityFinding(
                "PROFILE.COORDINATION_BOUNDARY_REQUIRED",
                "The profile requires a semantic metal coordination bond.",
            )
        )
    return StaticProfileEligibility(
        profile_id=policy.value,
        effective_profile_identity=identity,
        in_domain=not findings,
        findings=tuple(findings),
    )
