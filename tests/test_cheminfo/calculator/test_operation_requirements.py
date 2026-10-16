"""Scientific operation-requirement normalization tests."""

from hotpot.cheminfo.calculator.contracts import (
    ScientificOperationRequirements,
    ScientificQuantity,
    StatePolicy,
)
from hotpot.cheminfo.calculator.operation_requirements import (
    _effective_requirements,
)


def _requirements(
    identity: str,
    required: frozenset[ScientificQuantity],
    allowed: frozenset[ScientificQuantity],
) -> ScientificOperationRequirements:
    return ScientificOperationRequirements(identity, required, allowed)


def test_strict_default_is_analysis_only() -> None:
    effective = _effective_requirements(StatePolicy.STRICT, None)

    assert effective.required_resolved_quantities == frozenset()
    assert effective.allowed_unresolved_quantities == frozenset(
        ScientificQuantity
    )


def test_screening_profile_requires_charge_and_oxidation_state() -> None:
    effective = _effective_requirements(
        StatePolicy.COORDINATION_SCREENING_V1,
        None,
    )

    assert effective.required_resolved_quantities == frozenset(
        {
            ScientificQuantity.NET_CHARGE,
            ScientificQuantity.OXIDATION_STATE,
        }
    )
    assert ScientificQuantity.ELECTRONIC_STATE in (
        effective.allowed_unresolved_quantities
    )


def test_required_quantity_overrides_permission_to_remain_unresolved() -> None:
    caller = _requirements(
        "caller",
        frozenset({ScientificQuantity.ELECTRONIC_STATE}),
        frozenset({ScientificQuantity.NET_CHARGE}),
    )
    backend = _requirements(
        "backend",
        frozenset({ScientificQuantity.NET_CHARGE}),
        frozenset({ScientificQuantity.ELECTRONIC_STATE}),
    )

    effective = _effective_requirements(StatePolicy.STRICT, caller, backend)

    assert effective.required_resolved_quantities == frozenset(
        {
            ScientificQuantity.NET_CHARGE,
            ScientificQuantity.ELECTRONIC_STATE,
        }
    )
    assert not (
        effective.required_resolved_quantities
        & effective.allowed_unresolved_quantities
    )


def test_normalization_is_deterministic() -> None:
    caller = _requirements(
        "caller",
        frozenset({ScientificQuantity.NET_CHARGE}),
        frozenset(),
    )

    first = _effective_requirements(StatePolicy.STRICT, caller)
    second = _effective_requirements(StatePolicy.STRICT, caller)

    assert first == second
