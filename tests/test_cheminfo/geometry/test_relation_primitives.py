import math

import pytest

from hotpot.cheminfo.geometry.object import Cycle, Line, Plane, Point, Segment
from hotpot.cheminfo.geometry.relation import (
    LineRelationKind,
    PlanarityKind,
    PointCycleLocation,
    determine_line_relation,
    find_point_pairs_below_distance,
    line_distance,
    locate_point_in_planar_cycle,
    measure_planarity,
    point_pair_distances,
    point_segment_distance,
    segment_segment_distance,
)


def test_planarity_reports_fitted_plane_and_nonplanar_residuals():
    planar = Cycle([(0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)])
    planar_measurement = measure_planarity(planar)

    assert planar_measurement.kind is PlanarityKind.PLANAR
    assert planar_measurement.maximum_deviation == pytest.approx(0.0)
    assert planar_measurement.rms_deviation == pytest.approx(0.0)
    assert planar_measurement.normal is not None
    assert abs(planar_measurement.normal[2]) == pytest.approx(1.0)

    nonplanar = Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0.3), (0, 2, 0)])
    nonplanar_measurement = measure_planarity(nonplanar)

    assert nonplanar_measurement.kind is PlanarityKind.NONPLANAR
    assert nonplanar_measurement.maximum_deviation > 0.0
    assert nonplanar_measurement.normal is not None


def test_planarity_reports_degenerate_and_nonfinite_inputs_without_a_normal():
    collinear = Cycle([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    nonfinite = Cycle([(0, 0, 0), (1, 0, 0), (0, math.nan, 0)])

    degenerate = measure_planarity(collinear)
    undetermined = measure_planarity(nonfinite)

    assert degenerate.kind is PlanarityKind.DEGENERATE
    assert degenerate.normal is None
    assert undetermined.kind is PlanarityKind.UNDETERMINED
    assert undetermined.normal is None
    assert math.isnan(undetermined.length_scale)


@pytest.mark.parametrize(
    ("first", "second", "expected_kind", "expected_distance"),
    [
        (
            Line((0, 0, 0), (1, 0, 0)),
            Line((0, 1, 0), (0, -1, 0)),
            LineRelationKind.INTERSECTING,
            0.0,
        ),
        (
            Line((0, 0, 0), (1, 0, 0)),
            Line((0, 1, 0), (1, 0, 0)),
            LineRelationKind.PARALLEL,
            1.0,
        ),
        (
            Line((0, 0, 0), (1, 0, 0)),
            Line((3, 0, 0), (-4, 0, 0)),
            LineRelationKind.COINCIDENT,
            0.0,
        ),
        (
            Line((0, 0, 0), (1, 0, 0)),
            Line((0, 1, 1), (0, 1, 0)),
            LineRelationKind.SKEW,
            1.0,
        ),
    ],
)
def test_line_relation_classifies_infinite_lines(
    first, second, expected_kind, expected_distance
):
    relation = determine_line_relation(first, second)

    assert relation.kind is expected_kind
    assert relation.distance == pytest.approx(expected_distance)
    assert line_distance(first, second) == pytest.approx(expected_distance)


def test_line_relation_is_invariant_to_direction_scale_and_origin_shift():
    baseline = determine_line_relation(
        Line((0, 0, 0), (1, 0, 0)),
        Line((0, 1, 1), (0, 1, 0)),
    )
    reparameterized = determine_line_relation(
        Line((17, 0, 0), (-23, 0, 0)),
        Line((0, -12, 1), (0, 31, 0)),
    )

    assert reparameterized.kind is baseline.kind
    assert reparameterized.distance == pytest.approx(baseline.distance)
    assert reparameterized.parallel_measure == pytest.approx(
        baseline.parallel_measure
    )


def test_line_relation_preserves_tiny_nonzero_angular_measure():
    relation = determine_line_relation(
        Line((0, 0, 0), (1, 0, 0)),
        Line((0, 1, 0), (1, 1.0e-200, 0)),
    )

    assert relation.kind is LineRelationKind.UNDETERMINED
    assert relation.parallel_measure == pytest.approx(1.0e-200)


def test_degenerate_and_nonfinite_lines_have_no_defined_distance():
    degenerate = determine_line_relation(
        Line((0, 0, 0), (0, 0, 0)),
        Line((0, 1, 0), (1, 0, 0)),
    )
    nonfinite = determine_line_relation(
        Line((math.nan, 0, 0), (1, 0, 0)),
        Line((0, 1, 0), (1, 0, 0)),
    )

    assert degenerate.kind is LineRelationKind.DEGENERATE
    assert math.isnan(
        line_distance(
            Line((0, 0, 0), (0, 0, 0)),
            Line((0, 1, 0), (1, 0, 0)),
        )
    )
    assert nonfinite.kind is LineRelationKind.UNDETERMINED
    assert nonfinite.distance is None


def test_finite_segment_distance_measurements_include_degenerate_segments():
    point = Point((1, 1, 0))

    assert point_segment_distance(point, Segment((0, 0, 0), (2, 0, 0))) == pytest.approx(1.0)
    assert point_segment_distance(point, Segment((0, 0, 0), (0, 0, 0))) == pytest.approx(
        math.sqrt(2.0)
    )
    assert segment_segment_distance(
        Segment((0, 0, 0), (1, 0, 0)),
        Segment((0.5, -1, 0), (0.5, 1, 0)),
    ) == pytest.approx(0.0)
    assert segment_segment_distance(
        Segment((0, 0, 0), (0, 0, 0)),
        Segment((2, 0, 0), (3, 0, 0)),
    ) == pytest.approx(2.0)


def test_point_pair_measurements_and_strict_caller_threshold():
    points = (Point((0, 0, 0)), Point((1, 0, 0)), Point((3, 0, 0)))

    all_distances = point_pair_distances(points)
    below = find_point_pairs_below_distance(points, 3.0)

    assert [(item.first_index, item.second_index, item.distance) for item in all_distances] == [
        (0, 1, 1.0),
        (0, 2, 3.0),
        (1, 2, 2.0),
    ]
    assert [(item.first_index, item.second_index) for item in below] == [(0, 1), (1, 2)]


def test_planar_point_location_distinguishes_interior_boundary_and_exterior():
    cycle = Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])
    plane = Plane((0, 0, 0), (0, 0, 1))

    assert locate_point_in_planar_cycle(Point((1, 1, 0)), cycle, plane) is PointCycleLocation.INTERIOR
    assert locate_point_in_planar_cycle(Point((2, 1, 0)), cycle, plane) is PointCycleLocation.BOUNDARY
    assert locate_point_in_planar_cycle(Point((3, 1, 0)), cycle, plane) is PointCycleLocation.EXTERIOR


def test_self_intersecting_planar_boundary_has_no_defined_interior():
    bow_tie = Cycle([(0, 0, 0), (2, 2, 0), (0, 2, 0), (2, 0, 0)])
    plane = Plane((0, 0, 0), (0, 0, 1))

    assert (
        locate_point_in_planar_cycle(Point((0.5, 1, 0)), bow_tie, plane)
        is PointCycleLocation.UNDETERMINED
    )
