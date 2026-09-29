"""Differential fence for native nonplanar-surface preparation."""

from dataclasses import replace
from typing import Iterable

import numpy as np
import pytest

from hotpot.cheminfo.geometry import _geometry_native, native, relation
from hotpot.cheminfo.geometry.object import Cycle, Segment
from hotpot.cheminfo.geometry.relation import (
    PiercingState,
    SegmentCycleIndeterminacy,
)
from hotpot.cheminfo.geometry.settings import (
    DEFAULT_GEOMETRY_SETTINGS,
    GeometrySettings,
)


WARPED_SQUARE = Cycle(
    ((0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0))
)
PROVEN_NONEMBEDDED = Cycle(
    (
        (0.6, -1.2, 1.2),
        (0.3, -0.9, 1.9),
        (-2.5, -0.2, 1.2),
        (0.5, -0.4, -0.6),
        (-2.5, 1.1, -0.5),
    )
)
CONSTRUCTION_UNDETERMINED = Cycle(
    ((0, 0, 0), (2, 0, 0), (2, 2, 1), (2, 0, 0), (0, 2, 0))
)


def _settings(**surface_changes) -> GeometrySettings:
    return replace(
        DEFAULT_GEOMETRY_SETTINGS,
        surface=replace(
            DEFAULT_GEOMETRY_SETTINGS.surface,
            **surface_changes,
        ),
    )


def _native_summary(cycle: Cycle, settings: GeometrySettings) -> tuple:
    prepared = native._prepare_nonplanar_surface_family(cycle, settings)
    return (
        prepared.enumeration_complete,
        prepared.enumerated_surface_count,
        prepared.embedded_surface_count,
        prepared.proven_non_embedded_surface_count,
        prepared.construction_undetermined_count,
        prepared.triangle_pair_tests_used,
        frozenset(cause.name for cause in prepared.causes),
    )


def _cycle(coordinates: Iterable[Iterable[float]]) -> Cycle:
    return Cycle(tuple(tuple(point) for point in coordinates))


@pytest.mark.parametrize(
    ("cycle", "states"),
    (
        (WARPED_SQUARE, ("EMBEDDED", "EMBEDDED")),
        (
            PROVEN_NONEMBEDDED,
            (
                "EMBEDDED",
                "PROVEN_NON_EMBEDDED",
                "EMBEDDED",
                "PROVEN_NON_EMBEDDED",
                "EMBEDDED",
            ),
        ),
        (
            CONSTRUCTION_UNDETERMINED,
            (
                "CONSTRUCTION_UNDETERMINED",
                "PROVEN_NON_EMBEDDED",
                "PROVEN_NON_EMBEDDED",
                "PROVEN_NON_EMBEDDED",
                "CONSTRUCTION_UNDETERMINED",
            ),
        ),
    ),
)
def test_native_surface_categories_match_expected(cycle, states):
    settings = DEFAULT_GEOMETRY_SETTINGS

    prepared = native._prepare_nonplanar_surface_family(cycle, settings)

    assert tuple(state.name for state in prepared.surface_states) == states


@pytest.mark.parametrize(
    "settings",
    (
        _settings(maximum_triangle_pair_tests=1),
        _settings(maximum_triangle_pair_tests=2),
        _settings(maximum_surface_count=1),
        _settings(maximum_cycle_vertices=3),
    ),
)
def test_native_surface_budget_evidence_is_self_consistent(settings):
    prepared = native._prepare_nonplanar_surface_family(
        WARPED_SQUARE,
        settings,
    )

    assert prepared.enumerated_surface_count == len(prepared.surface_states)
    assert prepared.enumerated_surface_count == (
        prepared.embedded_surface_count
        + prepared.proven_non_embedded_surface_count
        + prepared.construction_undetermined_count
    )
    assert prepared.enumeration_complete == (
        "INCOMPLETE_SURFACE_FAMILY"
        not in {cause.name for cause in prepared.causes}
    )


def test_triangle_pair_budget_is_shared_across_surfaces():
    prepared = native._prepare_nonplanar_surface_family(
        WARPED_SQUARE,
        _settings(maximum_triangle_pair_tests=1),
    )

    assert not prepared.enumeration_complete
    assert prepared.enumerated_surface_count == 2
    assert prepared.embedded_surface_count == 1
    assert prepared.construction_undetermined_count == 1
    assert prepared.triangle_pair_tests_used == 1
    assert tuple(state.name for state in prepared.surface_states) == (
        "EMBEDDED",
        "CONSTRUCTION_UNDETERMINED",
    )
    assert {cause.name for cause in prepared.causes} == {
        "INCOMPLETE_SURFACE_FAMILY",
        "SURFACE_CONSTRUCTION",
    }


def test_native_random_generic_cycle_surface_counts_are_self_consistent():
    random = np.random.default_rng(20260929)
    for vertex_count in range(3, 8):
        for _ in range(6):
            coordinates = random.normal(size=(vertex_count, 3))
            cycle = _cycle(coordinates)
            prepared = native._prepare_nonplanar_surface_family(
                cycle,
                DEFAULT_GEOMETRY_SETTINGS,
            )
            assert prepared.enumerated_surface_count == len(
                prepared.surface_states
            )
            assert prepared.enumerated_surface_count == (
                prepared.embedded_surface_count
                + prepared.proven_non_embedded_surface_count
                + prepared.construction_undetermined_count
            )


@pytest.mark.parametrize(
    "cycle",
    (
        Cycle(((0, 0, 0), (1, 0, 0), (0, np.nan, 0))),
        Cycle(((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))),
    ),
)
def test_nonfinite_and_tiny_preparation_is_incomplete(cycle):
    settings = DEFAULT_GEOMETRY_SETTINGS
    prepared = native._prepare_nonplanar_surface_family(cycle, settings)

    assert not prepared.enumeration_complete
    assert prepared.embedded_surface_count == 0


@pytest.mark.parametrize(
    ("cycle", "cause"),
    (
        (
            Cycle(((0, 0, 0), (1, 0, 0), (0, np.nan, 0))),
            SegmentCycleIndeterminacy.NONFINITE_INPUT,
        ),
        (
            Cycle(((0, 0, 0), (1, 0, 0), (2, 0, 0))),
            SegmentCycleIndeterminacy.DEGENERATE_CYCLE,
        ),
    ),
)
def test_public_dispatch_rejects_invalid_cycle_before_surface_preparation(
    cycle,
    cause,
):
    result = relation.determine_segment_cycle_relation(
        Segment((0, 0, -1), (0, 0, 1)),
        cycle,
    )

    assert result.state is PiercingState.UNDETERMINED
    assert cause in result.indeterminacy_causes


def test_native_summary_is_rigid_motion_and_scale_invariant():
    coordinates = np.asarray(
        tuple(point.coordinates for point in PROVEN_NONEMBEDDED.vertices),
        dtype=np.float64,
    )
    rotation = np.asarray(
        ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    )
    translation = np.asarray((7.0, -3.0, 2.0))
    baseline = native._prepare_nonplanar_surface_family(
        PROVEN_NONEMBEDDED,
        DEFAULT_GEOMETRY_SETTINGS,
    )
    baseline_summary = _native_summary(
        PROVEN_NONEMBEDDED, DEFAULT_GEOMETRY_SETTINGS
    )
    baseline_states = tuple(state.name for state in baseline.surface_states)

    transformed = _cycle(coordinates @ rotation.T + translation)
    transformed_prepared = native._prepare_nonplanar_surface_family(
        transformed,
        DEFAULT_GEOMETRY_SETTINGS,
    )
    assert _native_summary(
        transformed, DEFAULT_GEOMETRY_SETTINGS
    ) == baseline_summary
    assert tuple(
        state.name for state in transformed_prepared.surface_states
    ) == baseline_states

    scale = 1.0e4
    scaled_settings = replace(
        DEFAULT_GEOMETRY_SETTINGS,
        tolerance=replace(
            DEFAULT_GEOMETRY_SETTINGS.tolerance,
            absolute_length=(
                DEFAULT_GEOMETRY_SETTINGS.tolerance.absolute_length * scale
            ),
        ),
    )
    scaled = _cycle(coordinates * scale)
    scaled_prepared = native._prepare_nonplanar_surface_family(
        scaled, scaled_settings
    )
    assert _native_summary(scaled, scaled_settings) == baseline_summary
    assert tuple(state.name for state in scaled_prepared.surface_states) == (
        baseline_states
    )


def test_raw_preparation_owns_coordinates_tolerances_and_limits():
    coordinates = np.asarray(
        tuple(point.coordinates for point in WARPED_SQUARE.vertices),
        dtype=np.float64,
    )
    tolerance = DEFAULT_GEOMETRY_SETTINGS.tolerance
    raw_tolerances = _geometry_native.NumericTolerances(
        tolerance.absolute_length,
        tolerance.relative_length,
        tolerance.parameter,
        tolerance.machine_epsilon_factor,
        tolerance.predicate_guard_factor,
        tolerance.planarity_factor,
        tolerance.winding_residual,
        tolerance.intersection_merge_factor,
        tolerance.aabb_padding_factor,
    )
    raw_limits = _geometry_native.SurfaceEnumerationLimits(9, 17, 23, 29)
    prepared = _geometry_native.prepare_nonplanar_surface_family(
        coordinates,
        raw_tolerances,
        raw_limits,
    )
    coordinates[:] = 99.0

    assert prepared.coordinates == [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.4),
        (0.0, 2.0, 0.0),
    ]
    assert prepared.tolerances.absolute_length == tolerance.absolute_length
    assert prepared.limits.maximum_cycle_vertices == 9
    assert prepared.limits.maximum_surface_count == 17
    assert prepared.limits.maximum_segment_triangle_tests == 23
    assert prepared.limits.maximum_triangle_pair_tests == 29


def test_native_preparation_entry_remains_internal():
    assert "SurfaceEnumerationLimits" not in native.__all__
    assert "PreparedNonplanarSurfaceFamily" not in native.__all__
    assert "_prepare_nonplanar_surface_family" not in native.__all__

    with pytest.raises(TypeError):
        _geometry_native.prepare_nonplanar_surface_family(
            np.asarray(
                tuple(point.coordinates for point in WARPED_SQUARE.vertices),
                dtype=np.float64,
            ),
            native._native_tolerances(DEFAULT_GEOMETRY_SETTINGS),
        )
