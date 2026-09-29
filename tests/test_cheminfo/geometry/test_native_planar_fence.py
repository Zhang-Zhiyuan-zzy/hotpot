"""Behavior fence for moving planar cycle predicates to the native backend.

The cases in this module exercise only public Python APIs.  They deliberately
pin the guard-band boundaries and the facts that must survive the backend
cutover, rather than retesting implementation helpers.
"""

from dataclasses import replace
import math

import numpy as np
import pytest

from hotpot.cheminfo.geometry.object import Cycle, Plane, Point, Segment
from hotpot.cheminfo.geometry.relation import (
    PiercingState,
    PlanarityKind,
    PointCycleLocation,
    SegmentCycleFeature,
    SegmentCycleIndeterminacy,
    closest_cycle_edge,
    determine_segment_cycle_relation,
    iter_segment_cycle_screenings,
    locate_point_in_planar_cycle,
    measure_planarity,
)
from hotpot.cheminfo.geometry.settings import DEFAULT_GEOMETRY_SETTINGS


SQUARE = Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
XY_PLANE = Plane(Point((0, 0, 0)), (0, 0, 1))


def _settings(*, absolute_length: float, parameter: float = 1.0e-6):
    tolerance = replace(
        DEFAULT_GEOMETRY_SETTINGS.tolerance,
        absolute_length=absolute_length,
        relative_length=0.0,
        parameter=parameter,
    )
    return replace(DEFAULT_GEOMETRY_SETTINGS, tolerance=tolerance)


@pytest.mark.parametrize(
    ("out_of_plane_height", "expected"),
    (
        (0.003, PlanarityKind.PLANAR),
        (0.008, PlanarityKind.UNDETERMINED),
        (0.020, PlanarityKind.NONPLANAR),
    ),
)
def test_planarity_guard_bands_are_explicit(out_of_plane_height, expected):
    cycle = Cycle(
        (
            (0, 0, 0),
            (2, 0, 0),
            (2, 2, out_of_plane_height),
            (0, 2, 0),
        )
    )

    measurement = measure_planarity(
        cycle,
        _settings(absolute_length=1.0e-3),
    )

    assert measurement.kind is expected


@pytest.mark.parametrize(
    ("edge_distance", "expected"),
    (
        (0.0005, PointCycleLocation.BOUNDARY),
        (0.0020, PointCycleLocation.UNDETERMINED),
        (0.0100, PointCycleLocation.INTERIOR),
    ),
)
def test_planar_point_boundary_guard_bands_are_explicit(edge_distance, expected):
    location = locate_point_in_planar_cycle(
        Point((edge_distance, 1, 0)),
        SQUARE,
        XY_PLANE,
        _settings(absolute_length=1.0e-3),
    )

    assert location is expected


@pytest.mark.parametrize(
    ("plane_parameter", "expected_state", "expected_feature"),
    (
        (-0.10, PiercingState.DOES_NOT_PIERCE,
         SegmentCycleFeature.LINE_EXTENSION_INTERIOR),
        (-0.02, PiercingState.UNDETERMINED, None),
        (-0.005, PiercingState.DOES_NOT_PIERCE,
         SegmentCycleFeature.SEGMENT_ENDPOINT_CONTACT),
        (0.02, PiercingState.UNDETERMINED, None),
        (0.05, PiercingState.PIERCES,
         SegmentCycleFeature.TRANSVERSE_INTERIOR),
    ),
)
def test_segment_parameter_guard_separates_contacts_bands_and_extensions(
    plane_parameter,
    expected_state,
    expected_feature,
):
    relation = determine_segment_cycle_relation(
        Segment(
            (1, 1, plane_parameter),
            (1, 1, plane_parameter - 1),
        ),
        SQUARE,
        _settings(absolute_length=1.0e-8, parameter=0.01),
    )

    assert relation.state is expected_state
    if expected_feature is None:
        assert SegmentCycleIndeterminacy.NUMERIC_BAND in (
            relation.indeterminacy_causes
        )
    else:
        assert expected_feature in relation.features


def test_invalid_effective_parameter_domain_is_reported_not_classified():
    scale = 1.0e-7
    cycle = Cycle(
        ((0, 0, 0), (scale, 0, 0), (scale, scale, 0), (0, scale, 0))
    )
    relation = determine_segment_cycle_relation(
        Segment(
            (scale / 2, scale / 2, -scale),
            (scale / 2, scale / 2, scale),
        ),
        cycle,
        _settings(absolute_length=1.0e-8, parameter=0.12),
    )

    assert relation.state is PiercingState.UNDETERMINED
    assert relation.indeterminacy_causes == frozenset(
        {SegmentCycleIndeterminacy.TOLERANCE_DOMAIN}
    )


@pytest.mark.parametrize(
    ("point", "expected_location", "expected_state"),
    (
        ((1.5, 0.5, 0), PointCycleLocation.INTERIOR, PiercingState.PIERCES),
        ((1.5, 1.5, 0), PointCycleLocation.EXTERIOR,
         PiercingState.DOES_NOT_PIERCE),
    ),
)
def test_concave_cycle_preserves_inside_and_notch_outside(
    point,
    expected_location,
    expected_state,
):
    cycle = Cycle(
        ((0, 0, 0), (3, 0, 0), (3, 3, 0), (1.5, 1, 0), (0, 3, 0))
    )

    assert locate_point_in_planar_cycle(
        Point(point), cycle, XY_PLANE
    ) is expected_location
    assert determine_segment_cycle_relation(
        Segment((point[0], point[1], -1), (point[0], point[1], 1)),
        cycle,
    ).state is expected_state


def test_bow_tie_cycle_cannot_yield_a_definite_piercing():
    cycle = Cycle(((0, 0, 0), (2, 2, 0), (0, 2, 0), (2, 0, 0)))

    location = locate_point_in_planar_cycle(Point((0.5, 1, 0)), cycle, XY_PLANE)
    relation = determine_segment_cycle_relation(
        Segment((0.5, 1, -1), (0.5, 1, 1)), cycle
    )

    assert location is PointCycleLocation.UNDETERMINED
    assert relation.state is PiercingState.UNDETERMINED
    assert relation.indeterminacy_causes == frozenset(
        {SegmentCycleIndeterminacy.SELF_INTERSECTION}
    )


@pytest.mark.parametrize(
    ("segment", "state", "feature", "cause"),
    (
        (
            Segment((0, 0, -1), (0, 0, 1)),
            PiercingState.DOES_NOT_PIERCE,
            SegmentCycleFeature.CYCLE_VERTEX_CONTACT,
            None,
        ),
        (
            Segment((1, 1, 1), (1, 1, 2)),
            PiercingState.DOES_NOT_PIERCE,
            SegmentCycleFeature.LINE_EXTENSION_INTERIOR,
            None,
        ),
        (
            Segment((1, 1, math.nan), (1, 1, 1)),
            PiercingState.UNDETERMINED,
            None,
            SegmentCycleIndeterminacy.NONFINITE_INPUT,
        ),
        (
            Segment((1, 1, 1), (1, 1, 1)),
            PiercingState.UNDETERMINED,
            None,
            SegmentCycleIndeterminacy.DEGENERATE_SEGMENT,
        ),
    ),
)
def test_planar_contact_extension_and_invalid_input_facts(
    segment,
    state,
    feature,
    cause,
):
    relation = determine_segment_cycle_relation(segment, SQUARE)

    assert relation.state is state
    if feature is not None:
        assert feature in relation.features
    if cause is not None:
        assert cause in relation.indeterminacy_causes


def test_aabb_separation_is_strict_at_the_padded_boundary():
    settings = _settings(absolute_length=0.125)
    touching_guard = Segment((2.5, 0, -1), (2.5, 0, 1))
    beyond_guard = Segment((2.50001, 0, -1), (2.50001, 0, 1))

    screenings = tuple(
        iter_segment_cycle_screenings(
            (touching_guard, beyond_guard),
            SQUARE,
            settings,
        )
    )

    assert not screenings[0].aabb_separated
    assert screenings[0].relation is not None
    assert screenings[1].aabb_separated
    assert screenings[1].relation is None


def test_closest_edge_tolerance_tie_uses_lowest_cycle_edge_index():
    tied = closest_cycle_edge(
        SQUARE,
        Segment((0.5 - 5.0e-9, 0.5, -1), (0.5 - 5.0e-9, 0.5, 1)),
    )
    resolved = closest_cycle_edge(
        SQUARE,
        Segment((0.5 - 2.0e-8, 0.5, -1), (0.5 - 2.0e-8, 0.5, 1)),
    )

    assert tied is not None and tied.edge_index == 0
    assert resolved is not None and resolved.edge_index == 3


def test_planar_relation_is_invariant_to_reflection_translation_and_cycle_shift():
    cycle_coordinates = np.asarray(
        ((0, 0, 0), (3, 0, 0), (3, 3, 0), (1.5, 1, 0), (0, 3, 0)),
        dtype=float,
    )
    segment_coordinates = np.asarray(((1.5, 0.5, -1), (1.5, 0.5, 1)))
    reflection = np.diag((-1.0, 1.0, 1.0))
    translation = np.asarray((7.0, -3.0, 2.0))

    baseline = determine_segment_cycle_relation(
        Segment(*segment_coordinates), Cycle(cycle_coordinates)
    )
    transformed_cycle = cycle_coordinates @ reflection.T + translation
    transformed_segment = segment_coordinates @ reflection.T + translation
    transformed_cycle = np.roll(transformed_cycle, 2, axis=0)
    transformed = determine_segment_cycle_relation(
        Segment(*transformed_segment), Cycle(transformed_cycle)
    )

    assert transformed.state is baseline.state
    assert transformed.features == baseline.features
    assert transformed.indeterminacy_causes == baseline.indeterminacy_causes

