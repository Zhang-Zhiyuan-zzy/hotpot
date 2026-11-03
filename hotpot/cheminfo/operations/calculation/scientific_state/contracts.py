"""Immutable scientific-state operation policy and activation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import ProviderValue
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.contracts import (
    RuleToggle,
)

__all__ = [
    'StatePolicy',
    'ProviderToggle',
    'ProviderSet',
    'ActivatedRule',
    'RuleSet',
]


class StatePolicy(Enum):
    """Named scientific-state decision profiles."""

    STRICT = "strict"
    COORDINATION_SCREENING_V1 = "coordination_screening_v1"


@dataclass(frozen=True)
class ProviderToggle:
    """Immutable activation and configuration of one deterministic provider."""

    provider_id: str
    enabled: bool
    configuration: Tuple[Tuple[str, ProviderValue], ...] = ()


@dataclass(frozen=True)
class ProviderSet:
    """Provider settings supplied to one resolution run."""

    toggles: Tuple[ProviderToggle, ...] = ()


@dataclass(frozen=True)
class ActivatedRule:
    """One registered rule paired with its per-run toggle."""

    rule: "InferenceRule"
    toggle: RuleToggle


@dataclass(frozen=True)
class RuleSet:
    """Dependency-validated topological layers of activated rules."""

    layers: Tuple[Tuple[ActivatedRule, ...], ...] = ()
