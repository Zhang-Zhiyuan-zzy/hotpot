from dataclasses import FrozenInstanceError
from math import inf

import pytest

from hotpot.cheminfo.geometry.object import (
    Cycle,
    Line,
    Plane,
    Point,
    Segment,
    Triangle,
)


def test_point_canonicalizes_coordinates_to_immutable_tuple():
    point = Point.from_coordinates([1, 2, 3])

    assert point.coordinates == (1.0, 2.0, 3.0)
    assert tuple(point) == point.coordinates
    assert (point.x, point.y, point.z) == point.coordinates
    assert hash(point) == hash(Point((1.0, 2.0, 3.0)))


def test_point_requires_exactly_three_coordinates():
    with pytest.raises(ValueError):
        Point((1.0, 2.0))


def test_point_preserves_nonfinite_input_for_relation_classification():
    point = Point((inf, 0.0, 0.0))

    assert point.x == inf


def test_line_has_distinct_origin_and_direction_semantics():
    line = Line((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

    assert line.origin == Point((1.0, 2.0, 3.0))
    assert line.direction == (4.0, 5.0, 6.0)


def test_line_from_points_computes_direction_without_reinterpreting_constructor():
    line = Line.from_points((1.0, 2.0, 3.0), (5.0, 7.0, 9.0))

    assert line == Line((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


def test_line_can_represent_zero_direction_for_degenerate_relation_result():
    line = Line((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    assert line.direction == (0.0, 0.0, 0.0)


def test_segment_retains_order_and_exposes_uncontroversial_derived_values():
    segment = Segment((0.0, 0.0, 0.0), (0.0, 3.0, 4.0))

    assert segment.start == Point((0.0, 0.0, 0.0))
    assert segment.end == Point((0.0, 3.0, 4.0))
    assert segment.direction == (0.0, 3.0, 4.0)
    assert segment.length == 5.0


def test_segment_can_represent_zero_length_for_relation_classification():
    segment = Segment((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))

    assert segment.length == 0.0


def test_plane_normalizes_a_finite_normal_vector():
    plane = Plane((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))

    assert plane.point == Point((0.0, 0.0, 0.0))
    assert plane.normal == (0.0, 0.0, 1.0)


def test_plane_rejects_zero_normal_vector():
    with pytest.raises(ValueError):
        Plane((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_triangle_retains_ordered_vertices_and_edges():
    triangle = Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0))

    assert triangle.vertices == (
        Point((0, 0, 0)),
        Point((1, 0, 0)),
        Point((0, 1, 0)),
    )
    assert triangle.edges == (
        Segment(triangle.first, triangle.second),
        Segment(triangle.second, triangle.third),
        Segment(triangle.third, triangle.first),
    )


def test_triangle_can_represent_collinear_vertices():
    triangle = Triangle((0, 0, 0), (1, 0, 0), (2, 0, 0))

    assert len(triangle.vertices) == 3


def test_cycle_is_only_an_ordered_closed_boundary():
    cycle = Cycle(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))

    assert len(cycle) == 4
    assert tuple(cycle) == cycle.vertices
    assert cycle.edges[-1] == Segment(cycle.vertices[-1], cycle.vertices[0])


def test_cycle_accepts_repeated_and_nonplanar_vertices_for_later_classification():
    cycle = Cycle(((0, 0, 0), (1, 0, 1), (1, 0, 1), (0, 1, 0)))

    assert cycle.vertices[1] == cycle.vertices[2]


def test_cycle_requires_three_vertices_to_express_a_closed_boundary():
    with pytest.raises(ValueError):
        Cycle(((0, 0, 0), (1, 0, 0)))


@pytest.mark.parametrize(
    "geometry_object",
    [
        Point((0, 0, 0)),
        Line((0, 0, 0), (1, 0, 0)),
        Segment((0, 0, 0), (1, 0, 0)),
        Plane((0, 0, 0), (0, 0, 1)),
        Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        Cycle(((0, 0, 0), (1, 0, 0), (0, 1, 0))),
    ],
)
def test_geometry_objects_are_immutable(geometry_object):
    with pytest.raises(FrozenInstanceError):
        geometry_object.extra = True
