"""Cross-language fence for canonical force-field workflow defaults."""

from dataclasses import fields

from hotpot.cheminfo.forcefields.native import (
    ComplexOptimizationOptions,
    CoordinationStageOptions,
    MetalPlacementOptions,
    OptimizationStoppingOptions,
    RingScreeningOptions,
)
from hotpot.cheminfo.obWrappers.native import _native_module
from hotpot.cheminfo.obWrappers.settings import (
    TORSION_REPAIR_ANGLE_RADIANS,
    TORSION_SINGULARITY_THRESHOLD,
)


def _assert_scalar_fields_match(native: object, python: object) -> None:
    for field in fields(python):
        expected = getattr(python, field.name)
        if field.name in {"geometry_settings", "placement", "ring_screening"}:
            continue
        actual = getattr(native, field.name)
        if hasattr(expected, "name"):
            assert actual.name == expected.name
        else:
            assert actual == expected


def _assert_geometry_fields_match(native: object, python: object) -> None:
    settings = python.geometry_settings
    for field in fields(settings.tolerance):
        assert getattr(native, f"geometry_{field.name}") == getattr(
            settings.tolerance, field.name
        )
    for field in fields(settings.surface):
        assert getattr(native, f"surface_{field.name}") == getattr(
            settings.surface, field.name
        )


def test_native_placement_defaults_match_python_policy() -> None:
    native_module = _native_module()
    native = native_module._DEFAULT_METAL_PLACEMENT_OPTIONS
    python = MetalPlacementOptions()

    _assert_scalar_fields_match(native, python)
    _assert_geometry_fields_match(native, python)


def test_native_stopping_defaults_match_python_policy() -> None:
    native_module = _native_module()
    _assert_scalar_fields_match(
        native_module._DEFAULT_OPTIMIZATION_STOPPING_OPTIONS,
        OptimizationStoppingOptions(),
    )


def test_native_ring_screening_defaults_match_python_policy() -> None:
    native_module = _native_module()
    native = native_module._DEFAULT_RING_SCREENING_OPTIONS
    python = RingScreeningOptions()

    _assert_scalar_fields_match(native, python)
    _assert_geometry_fields_match(native, python)

    # Eight limits exhaustive nonplanar-surface enumeration. Sixteen limits
    # force-field ring opening; these are intentionally different policies.
    assert native.surface_maximum_cycle_vertices == 8
    assert native.maximum_actionable_ring_size == 16
    assert native.maximum_relevant_cycle_count == 10000


def test_native_coordination_stage_defaults_match_python_policy() -> None:
    native_module = _native_module()
    native = native_module._DEFAULT_COORDINATION_STAGE_OPTIONS
    python = CoordinationStageOptions()

    _assert_scalar_fields_match(native, python)
    _assert_scalar_fields_match(native.placement, python.placement)
    _assert_geometry_fields_match(native.placement, python.placement)


def test_native_optimization_stage_defaults_match_python_policy() -> None:
    native_module = _native_module()
    native = native_module._DEFAULT_COMPLEX_OPTIMIZATION_OPTIONS
    python = ComplexOptimizationOptions()

    _assert_scalar_fields_match(native, python)
    _assert_scalar_fields_match(native.ring_screening, python.ring_screening)
    _assert_geometry_fields_match(native.ring_screening, python.ring_screening)


def test_native_torsion_defaults_match_obwrapper_policy() -> None:
    native = _native_module()._default_torsion_settings_snapshot()

    assert native == {
        "singularity_threshold": TORSION_SINGULARITY_THRESHOLD,
        "repair_angle_radians": TORSION_REPAIR_ANGLE_RADIANS,
    }
