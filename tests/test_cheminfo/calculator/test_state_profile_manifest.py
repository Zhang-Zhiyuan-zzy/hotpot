"""Exact built-in coordination-screening manifest tests."""

import hotpot

from hotpot.cheminfo.calculator.contracts import StatePolicy
from hotpot.cheminfo.calculator.operation_requirements import (
    _effective_requirements,
)
from hotpot.cheminfo.calculator.profile_eligibility import (
    _static_profile_eligibility,
)
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts
from hotpot.cheminfo.calculator.state_rules.manifest import (
    _coordination_screening_manifest,
    _coordination_screening_registry,
)
from hotpot.cheminfo.calculator.state_rules.profiles import (
    _coordination_screening_profile,
)
from hotpot.cheminfo.core import BondKind


def _coordination_facts():
    mol = hotpot.read_mol("[Eu].N", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    return extract_molecular_facts(mol)


def test_registry_matches_every_profile_rule_id_and_version() -> None:
    profile = _coordination_screening_profile()
    registry = _coordination_screening_registry()

    registered = tuple(
        sorted(
            (specification.rule_id, specification.rule_version)
            for specification in registry.available()
        )
    )

    assert registered == profile.required_rule_versions


def test_manifest_builds_explicit_provider_rule_and_budget_sets() -> None:
    profile = _coordination_screening_profile()
    manifest = _coordination_screening_manifest()

    assert manifest.policy is StatePolicy.COORDINATION_SCREENING_V1
    assert manifest.activation_status == "planned"
    assert manifest.profile_content_digest == profile.content_digest
    assert manifest.candidate_budget == profile.candidate_budget
    assert tuple(
        (toggle.provider_id, toggle.enabled)
        for toggle in manifest.provider_set.toggles
    ) == tuple(
        (provider_id, True)
        for provider_id in sorted(profile.required_provider_ids)
    )
    assert tuple(
        sorted(
            (
                activated.rule.specification.rule_id,
                activated.rule.specification.rule_version,
                activated.toggle.enabled,
            )
            for layer in manifest.rule_set.layers
            for activated in layer
        )
    ) == tuple(
        (rule_id, version, True)
        for rule_id, version in profile.required_rule_versions
    )


def test_exact_manifest_dry_run_remains_blocked_only_by_activation() -> None:
    manifest = _coordination_screening_manifest()
    requirements = _effective_requirements(manifest.policy, None)

    eligibility = _static_profile_eligibility(
        _coordination_facts(),
        manifest.policy,
        requirements,
        manifest.provider_set,
        manifest.rule_set,
        manifest.candidate_budget,
    )

    assert eligibility.in_domain is False
    assert tuple(finding.code for finding in eligibility.findings) == (
        "PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE",
    )
