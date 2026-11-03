"""Explicit runtime manifests for built-in scientific-state profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import CandidateBudget
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import (
    ProviderSet,
    ProviderToggle,
    StatePolicy,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.americium_curium_iii import (
    AmericiumCuriumIIIPriorRule,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.anionic_x_ligands import (
    AnionicXLigandRule,
)
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import RuleSet
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts import (
    RuleToggle,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.f_block_hund import (
    FBlockHundPriorRule,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.ionic_configuration_spin import (
    IonicConfigurationSpinRule,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.lanthanide_iii import (
    LanthanideIIIPriorRule,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.main_group_closed_shell import (
    MainGroupClosedShellRule,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.monoatomic_ions import (
    MonoatomicIonRule,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.neutral_l_donors import (
    NeutralLDonorRule,
)
from hotpot.cheminfo.operations.calculation.scientific_state.profiles import (
    _coordination_screening_profile,
)
from hotpot.cheminfo.operations.calculation.scientific_state.rule_registry import (
    InferenceRuleRegistry,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.thorium_iv import (
    ThoriumIVPriorRule,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.trusted_history import (
    TrustedHistoryRule,
    TrustedScientificStateRecord,
)

__all__ = []


@dataclass(frozen=True)
class _ScientificStateRuntimeManifest:
    """Exact, explicit provider/rule configuration for one profile artifact."""

    policy: StatePolicy
    profile_id: str
    profile_version: str
    activation_status: str
    profile_content_digest: str
    provider_set: ProviderSet
    rule_set: RuleSet
    candidate_budget: CandidateBudget


def _coordination_screening_registry(
    trusted_records: Tuple[TrustedScientificStateRecord, ...] = (),
) -> InferenceRuleRegistry:
    registry = InferenceRuleRegistry()
    rules = (
        MainGroupClosedShellRule(),
        MonoatomicIonRule(),
        NeutralLDonorRule(),
        AnionicXLigandRule(),
        LanthanideIIIPriorRule(),
        AmericiumCuriumIIIPriorRule(),
        ThoriumIVPriorRule(),
        IonicConfigurationSpinRule(),
        FBlockHundPriorRule(),
        TrustedHistoryRule(trusted_records),
    )
    for rule in rules:
        registry.register(rule)
    return registry


def _coordination_screening_manifest(
    trusted_records: Tuple[TrustedScientificStateRecord, ...] = (),
) -> _ScientificStateRuntimeManifest:
    """Build the exact planned profile configuration without activating it."""
    profile = _coordination_screening_profile()
    registry = _coordination_screening_registry(trusted_records)
    specifications = {
        specification.rule_id: specification
        for specification in registry.available()
    }
    required_versions = dict(profile.required_rule_versions)
    registered_versions = {
        rule_id: specification.rule_version
        for rule_id, specification in specifications.items()
    }
    if registered_versions != required_versions:
        raise ValueError(
            "Built-in rule implementations do not match the profile artifact"
        )
    toggles = tuple(
        RuleToggle(rule_id, True)
        for rule_id in sorted(required_versions)
    )
    return _ScientificStateRuntimeManifest(
        policy=StatePolicy.COORDINATION_SCREENING_V1,
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        activation_status=profile.activation_status,
        profile_content_digest=profile.content_digest,
        provider_set=ProviderSet(
            tuple(
                ProviderToggle(provider_id, True)
                for provider_id in sorted(profile.required_provider_ids)
            )
        ),
        rule_set=registry.activate(
            StatePolicy.COORDINATION_SCREENING_V1,
            toggles,
        ),
        candidate_budget=profile.candidate_budget,
    )
