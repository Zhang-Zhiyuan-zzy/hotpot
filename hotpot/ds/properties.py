"""Extensible property-type validation registry."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from .models import Observation, PropertyType


Validator = Callable[[Observation, PropertyType], Iterable[str]]


@dataclass(frozen=True)
class PropertyPlugin:
    property_type_id: str
    validator: Validator


class PropertyRegistry:
    def __init__(self) -> None:
        self._plugins: Dict[str, PropertyPlugin] = {}

    def register(self, property_type_id: str, validator: Validator) -> None:
        self._plugins[property_type_id] = PropertyPlugin(property_type_id, validator)

    def validate(self, observation: Observation, definition: PropertyType) -> Tuple[str, ...]:
        issues = list(_validate_value_kind(observation, definition))
        plugin = self._plugins.get(definition.property_type_id)
        if plugin is not None:
            issues.extend(plugin.validator(observation, definition))
        return tuple(issues)


def _validate_value_kind(
    observation: Observation,
    definition: PropertyType,
) -> Iterable[str]:
    value = observation.value
    kind = definition.value_kind
    if kind == "integer" and (isinstance(value, bool) or not isinstance(value, Integral)):
        yield "value must be an integer"
    elif kind == "real" and (isinstance(value, bool) or not isinstance(value, Real)):
        yield "value must be a real number"
    elif kind in {"category", "text", "series_asset"} and not isinstance(value, str):
        yield f"value must be a string for {kind}"
    if definition.canonical_unit is not None and observation.unit != definition.canonical_unit:
        yield f"unit must be {definition.canonical_unit!r}"
    if definition.canonical_unit is None and observation.unit is not None:
        yield "unit must be null for a dimensionless/unitless property"


def _spin_multiplicity(observation: Observation, _: PropertyType) -> Iterable[str]:
    if isinstance(observation.value, Integral) and observation.value < 1:
        yield "spin multiplicity must be positive"


def _spin_energy_gap(observation: Observation, definition: PropertyType) -> Iterable[str]:
    yield from _conditioned_real(observation, definition)
    if not observation.method.get("sign_convention"):
        yield "spin energy gap method must declare sign_convention"


def _allowed_category(observation: Observation, definition: PropertyType) -> Iterable[str]:
    allowed = definition.metadata.get(
        "allowed_values",
        ["low_spin", "intermediate_spin", "high_spin"],
    )
    if observation.value not in allowed:
        yield f"category must be one of {allowed}"


def _conditioned_category(
    observation: Observation, definition: PropertyType
) -> Iterable[str]:
    yield from _allowed_category(observation, definition)
    yield from _conditioned_real(observation, definition)


def _conditioned_real(observation: Observation, definition: PropertyType) -> Iterable[str]:
    for condition in definition.metadata.get("required_conditions", []):
        if condition not in observation.conditions:
            yield f"required condition is missing: {condition}"


def _positive_real(observation: Observation, _: PropertyType) -> Iterable[str]:
    if isinstance(observation.value, Real) and observation.value <= 0:
        yield "value must be positive"


def _positive_conditioned_real(
    observation: Observation, definition: PropertyType
) -> Iterable[str]:
    yield from _positive_real(observation, definition)
    yield from _conditioned_real(observation, definition)


def _spectrum(observation: Observation, _: PropertyType) -> Iterable[str]:
    if "x_axis" not in observation.metadata or "y_axis" not in observation.metadata:
        yield "spectrum metadata must define x_axis and y_axis"


DEFAULT_PROPERTY_REGISTRY = PropertyRegistry()
DEFAULT_PROPERTY_REGISTRY.register("spin_multiplicity", _spin_multiplicity)
DEFAULT_PROPERTY_REGISTRY.register("spin_energy_gap", _spin_energy_gap)
DEFAULT_PROPERTY_REGISTRY.register("spin_category", _allowed_category)
DEFAULT_PROPERTY_REGISTRY.register("spin_transition_behavior", _conditioned_category)
DEFAULT_PROPERTY_REGISTRY.register("spin_transition_temperature", _positive_conditioned_real)
DEFAULT_PROPERTY_REGISTRY.register("stability_constant", _positive_conditioned_real)
DEFAULT_PROPERTY_REGISTRY.register("log_stability_constant", _conditioned_real)
DEFAULT_PROPERTY_REGISTRY.register("separation_factor", _positive_conditioned_real)
DEFAULT_PROPERTY_REGISTRY.register("distribution_ratio", _positive_conditioned_real)
DEFAULT_PROPERTY_REGISTRY.register("spectrum", _spectrum)


def builtin_property_types() -> Tuple[PropertyType, ...]:
    """Return the current core vocabulary; applications may add more types."""
    return (
        PropertyType(
            "oxidation_state", "Oxidation state", "integer", None, "current",
            {
                "subject_kind": "metal_site_or_complex",
                "typical_range_for_review": [-5, 10],
                "range_is_not_a_validity_constraint": True,
            },
        ),
        PropertyType(
            "molecular_formal_charge", "Molecular formal charge", "integer", None, "current",
            {"subject_kind": "molecular_structure"},
        ),
        PropertyType(
            "spin_multiplicity", "Spin multiplicity (2S+1)", "integer", None, "current",
            {"subject_kind": "metal_site_or_complex"},
        ),
        PropertyType(
            "spin_category", "Reported spin category", "category", None, "current",
            {
                "allowed_values": [
                    "low_spin", "intermediate_spin", "high_spin",
                ],
                "subject_kind": "metal_site_or_complex",
            },
        ),
        PropertyType(
            "spin_transition_behavior", "Spin-transition behavior", "category", None, "current",
            {
                "allowed_values": ["spin_crossover", "spin_transition", "spin_switching"],
                "condition_dependent": True,
                "required_conditions": ["stimulus", "measurement_protocol"],
            },
        ),
        PropertyType(
            "spin_state_text", "Unresolved spin-state text", "text", None, "current",
            {"claim_only_by_default": True},
        ),
        PropertyType(
            "spin_energy_gap", "Spin-state energy difference", "real", "kcal/mol", "current",
            {
                "sign_convention": "must be declared per observation",
                "required_conditions": [
                    "lower_spin_multiplicity",
                    "upper_spin_multiplicity",
                    "energy_definition",
                ],
            },
        ),
        PropertyType(
            "spin_transition_temperature", "Spin-transition temperature", "real", "K", "current",
            {
                "required_conditions": ["stimulus", "measurement_protocol", "branch"],
                "condition_dependent": True,
            },
        ),
        PropertyType(
            "stability_constant", "Stability constant", "real", None, "current",
            {
                "required_conditions": [
                    "temperature", "medium", "ionic_strength", "standard_state",
                ],
                "required_participant_roles": ["complex"],
            },
        ),
        PropertyType(
            "log_stability_constant", "Log stability constant", "real", None, "current",
            {
                "log_base": 10,
                "required_conditions": [
                    "temperature", "medium", "ionic_strength", "standard_state",
                ],
                "required_participant_roles": ["complex"],
            },
        ),
        PropertyType(
            "separation_factor", "Separation factor", "real", None, "current",
            {
                "required_conditions": [
                    "temperature", "aqueous_phase", "organic_phase", "aqueous_pH",
                ],
                "required_participant_roles": ["numerator_analyte", "denominator_analyte"],
            },
        ),
        PropertyType(
            "distribution_ratio", "Distribution ratio", "real", None, "current",
            {
                "required_conditions": [
                    "temperature", "aqueous_phase", "organic_phase", "aqueous_pH",
                ],
                "required_participant_roles": ["analyte"],
            },
        ),
        PropertyType(
            "spectrum", "Spectrum", "series_asset", None, "current",
            {"axis_metadata_required": True},
        ),
    )
