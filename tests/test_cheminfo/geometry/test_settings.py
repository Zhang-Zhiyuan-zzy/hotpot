from dataclasses import FrozenInstanceError

import pytest

from hotpot.cheminfo.geometry.settings import (
    DEFAULT_GEOMETRY_SETTINGS,
    GeometrySettings,
    NumericToleranceSettings,
    SurfaceEnumerationSettings,
)


def test_default_numeric_tolerances_match_relation_contract():
    tolerance = NumericToleranceSettings()

    assert tolerance.absolute_length == 1.0e-8
    assert tolerance.relative_length == 1.0e-10
    assert tolerance.parameter == 1.0e-10
    assert tolerance.machine_epsilon_factor == 64.0
    assert tolerance.predicate_guard_factor == 4.0
    assert tolerance.planarity_factor == 1.0
    assert tolerance.winding_residual == 1.0e-10
    assert tolerance.intersection_merge_factor == 4.0
    assert tolerance.aabb_padding_factor == 4.0


def test_default_surface_budgets_match_relation_contract():
    surface = SurfaceEnumerationSettings()

    assert surface.maximum_cycle_vertices == 8
    assert surface.maximum_surface_count == 132
    assert surface.maximum_segment_triangle_tests == 792
    assert surface.maximum_triangle_pair_tests == 1980


def test_default_geometry_settings_use_independent_nested_values():
    first = GeometrySettings()
    second = GeometrySettings()

    assert first == DEFAULT_GEOMETRY_SETTINGS
    assert first.tolerance is not second.tolerance
    assert first.surface is not second.surface


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("absolute_length", 0.0),
        ("relative_length", -1.0),
        ("parameter", 0.0),
        ("parameter", 0.125),
        ("machine_epsilon_factor", 0.5),
        ("predicate_guard_factor", 1.0),
        ("planarity_factor", 0.0),
        ("winding_residual", 0.0),
        ("winding_residual", 0.5),
        ("intersection_merge_factor", 0.5),
        ("absolute_length", float("nan")),
    ],
)
def test_numeric_tolerance_rejects_values_outside_contract(keyword, value):
    with pytest.raises(ValueError):
        NumericToleranceSettings(**{keyword: value})


def test_numeric_tolerance_requires_aabb_guard_to_cover_predicate_guard():
    with pytest.raises(ValueError, match="aabb_padding_factor"):
        NumericToleranceSettings(
            predicate_guard_factor=5.0,
            aabb_padding_factor=4.0,
        )


@pytest.mark.parametrize(
    "keyword",
    [
        "maximum_surface_count",
        "maximum_segment_triangle_tests",
        "maximum_triangle_pair_tests",
    ],
)
def test_surface_enumeration_requires_positive_budgets(keyword):
    with pytest.raises(ValueError):
        SurfaceEnumerationSettings(**{keyword: 0})


def test_surface_enumeration_requires_a_cycle_capacity_of_three():
    with pytest.raises(ValueError):
        SurfaceEnumerationSettings(maximum_cycle_vertices=2)


def test_settings_are_immutable():
    with pytest.raises(FrozenInstanceError):
        DEFAULT_GEOMETRY_SETTINGS.tolerance.absolute_length = 1.0
