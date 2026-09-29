"""Focused equivalence tests for the Phase-1 native geometry kernels."""

import math

import numpy as np
import pytest

from hotpot.cheminfo.geometry import _geometry_native
from hotpot.cheminfo.geometry import native
from hotpot.cheminfo.geometry.object import Line, Point, Segment
from hotpot.cheminfo.geometry.relation import (
    LineRelationKind,
    determine_line_relation as python_line_relation,
    find_point_pairs_below_distance as python_pairs_below,
    line_distance as python_line_distance,
    point_pair_distances as python_pair_distances,
    point_segment_distance as python_point_segment_distance,
    segment_segment_distance as python_segment_segment_distance,
)
from hotpot.cheminfo.geometry.settings import (
    GeometrySettings,
    NumericToleranceSettings,
)


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
def test_native_line_relations_match_the_characterized_python_backend(
    first,
    second,
    expected_kind,
    expected_distance,
):
    expected = python_line_relation(first, second)
    actual = native.determine_line_relation(first, second)

    assert actual.kind is expected_kind is expected.kind
    assert actual.distance == pytest.approx(expected_distance)
    assert actual.distance == pytest.approx(expected.distance)
    assert actual.parallel_measure == pytest.approx(expected.parallel_measure)
    assert native.line_distance(first, second) == pytest.approx(
        python_line_distance(first, second)
    )


def test_native_line_relation_preserves_degenerate_nonfinite_and_tiny_angle_states():
    degenerate_first = Line((0, 0, 0), (0, 0, 0))
    finite_second = Line((0, 1, 0), (1, 0, 0))
    nonfinite_first = Line((math.nan, 0, 0), (1, 0, 0))
    tiny_second = Line((0, 1, 0), (1, 1.0e-200, 0))

    degenerate = native.determine_line_relation(degenerate_first, finite_second)
    nonfinite = native.determine_line_relation(nonfinite_first, finite_second)
    tiny = native.determine_line_relation(
        Line((0, 0, 0), (1, 0, 0)),
        tiny_second,
    )

    assert degenerate.kind is LineRelationKind.DEGENERATE
    assert degenerate.distance is None
    assert nonfinite.kind is LineRelationKind.UNDETERMINED
    assert nonfinite.distance is None
    assert tiny.kind is LineRelationKind.UNDETERMINED
    assert tiny.parallel_measure == pytest.approx(1.0e-200)
    assert math.isnan(native.line_distance(degenerate_first, finite_second))


def test_native_point_segment_measurement_and_distance_share_one_kernel():
    point = Point((1, 1, 0))
    segment = Segment((0, 0, 0), (2, 0, 0))

    measurement = native.point_segment_measurement(point, segment)

    assert measurement.distance == pytest.approx(1.0)
    assert measurement.closest_point == pytest.approx((1.0, 0.0, 0.0))
    assert measurement.parameter == pytest.approx(0.5)
    assert not measurement.segment_degenerate
    assert native.point_segment_distance(point, segment) == pytest.approx(
        python_point_segment_distance(point, segment)
    )


def test_native_distance_kernels_honor_explicit_degeneracy_tolerance():
    point = Point((0.4, 1.0, 0.0))
    segment = Segment((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))
    settings = GeometrySettings(
        tolerance=NumericToleranceSettings(absolute_length=1.0)
    )

    measurement = native.point_segment_measurement(point, segment, settings)

    assert measurement.segment_degenerate
    assert measurement.parameter == 0.0
    assert measurement.distance == pytest.approx(math.sqrt(1.16))
    assert measurement.distance == pytest.approx(
        python_point_segment_distance(point, segment, settings)
    )


def test_native_segment_segment_measurement_matches_scalar_public_semantics():
    first = Segment((0, 0, 0), (1, 0, 0))
    second = Segment((0.5, -1, 0), (0.5, 1, 0))
    degenerate = Segment((2, 0, 0), (2, 0, 0))

    crossing = native.segment_segment_measurement(first, second)
    separated = native.segment_segment_measurement(degenerate, first)

    assert crossing.distance == pytest.approx(0.0)
    assert crossing.first_closest_point == pytest.approx((0.5, 0.0, 0.0))
    assert crossing.second_closest_point == pytest.approx((0.5, 0.0, 0.0))
    assert crossing.first_parameter == pytest.approx(0.5)
    assert crossing.second_parameter == pytest.approx(0.5)
    assert separated.first_segment_degenerate
    assert native.segment_segment_distance(first, second) == pytest.approx(
        python_segment_segment_distance(first, second)
    )
    assert native.segment_segment_distance(degenerate, first) == pytest.approx(
        python_segment_segment_distance(degenerate, first)
    )


def test_native_point_pair_batches_preserve_order_nan_and_strict_threshold():
    points = (
        Point((0, 0, 0)),
        Point((1, 0, 0)),
        Point((3, 0, 0)),
        Point((math.nan, 0, 0)),
    )

    expected_all = python_pair_distances(points)
    actual_all = native.point_pair_distances(points)
    expected_below = python_pairs_below(points, 3.0)
    actual_below = native.find_point_pairs_below_distance(points, 3.0)

    assert [
        (item.first_index, item.second_index) for item in actual_all
    ] == [
        (item.first_index, item.second_index) for item in expected_all
    ]
    for expected, actual in zip(expected_all, actual_all):
        if math.isnan(expected.distance):
            assert math.isnan(actual.distance)
        else:
            assert actual.distance == pytest.approx(expected.distance)
    assert actual_below == expected_below


def test_raw_native_pair_subset_uses_typed_batch_contract():
    coordinates = np.ascontiguousarray(
        ((0, 0, 0), (1, 0, 0), (3, 0, 0)),
        dtype=np.float64,
    )
    pair_indices = np.ascontiguousarray(((2, 0), (1, 2)), dtype=np.int64)

    indices, distances = _geometry_native.point_pair_distances(
        coordinates,
        pair_indices,
    )

    assert indices.tolist() == [[2, 0], [1, 2]]
    assert distances.tolist() == pytest.approx([3.0, 2.0])


def test_aabb_batch_predicates_preserve_strict_guard_boundaries():
    first = np.asarray(
        [
            [[0, 0, 0], [1, 1, 1]],
            [[0, 0, 0], [1, 1, 1]],
            [[0, 0, 0], [1, 1, 1]],
        ],
        dtype=np.float64,
    )
    second = np.asarray(
        [
            [[2, 0, 0], [3, 1, 1]],
            [[1, 0, 0], [2, 1, 1]],
            [[1.25, 0, 0], [2, 1, 1]],
        ],
        dtype=np.float64,
    )

    separated = native.aabb_separation_mask(first, second, [0.0, 0.0, 0.25])

    assert separated.tolist() == [True, False, False]
    assert native.aabb_bounds(((3, -1, 5), (0, 1, 2))).tolist() == [
        [0.0, -1.0, 2.0],
        [3.0, 1.0, 5.0],
    ]
    assert np.isnan(native.aabb_bounds(((0, 0, 0), (math.nan, 1, 1)))[:, 0]).all()


def test_aabb_candidate_and_segment_batches_are_stable_and_explicit():
    first = np.asarray(
        [
            [[0, 0, 0], [1, 1, 1]],
            [[4, 0, 0], [5, 1, 1]],
        ],
        dtype=np.float64,
    )
    second = np.asarray(
        [
            [[2, 0, 0], [3, 1, 1]],
            [[1, 0, 0], [2, 1, 1]],
            [[4.5, 0, 0], [6, 1, 1]],
        ],
        dtype=np.float64,
    )
    segments = (
        Segment((0, 0, 0), (1, 0, 0)),
        Segment((3, 0, 0), (4, 0, 0)),
        Segment((1.25, 0, 0), (2, 0, 0)),
    )
    target = np.asarray(((0, -1, -1), (1, 1, 1)), dtype=np.float64)

    candidates = native.aabb_candidate_pairs(first, second, 0.0)
    separated = native.segment_aabb_separation_mask(
        segments,
        target,
        [0.0, 0.0, 0.25],
    )

    assert candidates.tolist() == [[0, 1], [1, 2]]
    assert separated.tolist() == [False, True, False]


def test_raw_native_contract_rejects_implicit_dtype_and_invalid_tolerances():
    with pytest.raises(TypeError):
        _geometry_native.NumericTolerances()

    tolerances = _geometry_native.NumericTolerances(
        1.0e-8,
        1.0e-10,
        1.0e-10,
        64.0,
        4.0,
        1.0,
        1.0e-10,
        4.0,
        4.0,
    )
    point = np.asarray((0, 0, 0), dtype=np.float32)

    with pytest.raises(TypeError, match="unexpected dtype"):
        _geometry_native.point_segment_distance(
            point,
            point,
            point,
            tolerances,
        )
    with pytest.raises(ValueError, match="absolute_length"):
        _geometry_native.NumericTolerances(
            0.0,
            1.0e-10,
            1.0e-10,
            64.0,
            4.0,
            1.0,
            1.0e-10,
            4.0,
            4.0,
        )
