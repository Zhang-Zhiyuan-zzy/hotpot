"""Differential fence for the native planar-cycle geometry kernels."""

from dataclasses import replace
import math
from typing import Iterable, Optional, Tuple

import numpy as np
import pytest

from hotpot.cheminfo.geometry import _geometry_native, native
from hotpot.cheminfo.geometry.object import Cycle, Plane, Point, Segment
from hotpot.cheminfo.geometry.relation import (
    ClosestCycleEdge,
    PlanarityMeasurement,
    SegmentCycleRelation,
    SegmentCycleScreening,
    closest_cycle_edge,
    determine_segment_cycle_relation,
    iter_segment_cycle_relations,
    iter_segment_cycle_screenings,
    locate_point_in_planar_cycle,
    measure_planarity,
)
from hotpot.cheminfo.geometry.settings import (
    DEFAULT_GEOMETRY_SETTINGS,
    GeometrySettings,
)


SQUARE = Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
XY_PLANE = Plane((0, 0, 0), (0, 0, 1))


def _settings(
    *,
    absolute_length: float,
    parameter: float = 1.0e-6,
) -> GeometrySettings:
    tolerance = replace(
        DEFAULT_GEOMETRY_SETTINGS.tolerance,
        absolute_length=absolute_length,
        relative_length=0.0,
        parameter=parameter,
    )
    return replace(DEFAULT_GEOMETRY_SETTINGS, tolerance=tolerance)


def _raw_tolerances(settings: GeometrySettings):
    tolerance = settings.tolerance
    return _geometry_native.NumericTolerances(
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


def _assert_float_equal(
    expected: float,
    actual: float,
    *,
    scale: float = 1.0,
) -> None:
    if math.isnan(expected):
        assert math.isnan(actual)
        return
    if math.isinf(expected):
        assert actual == expected
        return
    assert actual == pytest.approx(
        expected,
        rel=5.0e-11,
        abs=5.0e-11 * max(1.0, scale),
    )


def _assert_planarity_equal(
    expected: PlanarityMeasurement,
    actual: PlanarityMeasurement,
) -> None:
    assert actual.kind is expected.kind
    scale = expected.length_scale if math.isfinite(expected.length_scale) else 1.0
    np.testing.assert_allclose(
        actual.centroid.coordinates,
        expected.centroid.coordinates,
        rtol=5.0e-11,
        atol=5.0e-11 * max(1.0, scale),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        actual.singular_values,
        expected.singular_values,
        rtol=5.0e-11,
        atol=5.0e-11 * max(1.0, scale),
        equal_nan=True,
    )
    _assert_float_equal(
        expected.maximum_deviation,
        actual.maximum_deviation,
        scale=scale,
    )
    _assert_float_equal(
        expected.rms_deviation,
        actual.rms_deviation,
        scale=scale,
    )
    _assert_float_equal(expected.length_scale, actual.length_scale, scale=scale)
    _assert_float_equal(
        expected.length_tolerance,
        actual.length_tolerance,
        scale=scale,
    )
    assert (actual.normal is None) is (expected.normal is None)
    if expected.normal is not None and actual.normal is not None:
        alignment = abs(float(np.dot(expected.normal, actual.normal)))
        assert alignment == pytest.approx(1.0, abs=5.0e-12)


def _assert_closest_edge_equal(
    expected: Optional[ClosestCycleEdge],
    actual: Optional[ClosestCycleEdge],
    *,
    scale: float,
) -> None:
    assert (actual is None) is (expected is None)
    if expected is None or actual is None:
        return
    assert actual.edge_index == expected.edge_index
    assert actual.edge == expected.edge
    _assert_float_equal(expected.distance, actual.distance, scale=scale)


def _assert_relation_equal(
    expected: SegmentCycleRelation,
    actual: SegmentCycleRelation,
) -> None:
    assert actual.state is expected.state
    assert actual.features == expected.features
    assert actual.indeterminacy_causes == expected.indeterminacy_causes
    assert actual.surface_model is expected.surface_model
    assert actual.surface_evidence == expected.surface_evidence
    assert actual.settings == expected.settings
    scale = max(
        (
            abs(coordinate)
            for point in expected.intersection_points
            for coordinate in point.coordinates
        ),
        default=1.0,
    )
    assert len(actual.intersection_points) == len(expected.intersection_points)
    for expected_point, actual_point in zip(
        expected.intersection_points,
        actual.intersection_points,
    ):
        np.testing.assert_allclose(
            actual_point.coordinates,
            expected_point.coordinates,
            rtol=5.0e-11,
            atol=5.0e-11 * max(1.0, scale),
        )
    _assert_closest_edge_equal(
        expected.closest_boundary_edge,
        actual.closest_boundary_edge,
        scale=scale,
    )


def _assert_screening_equal(
    expected: SegmentCycleScreening,
    actual: SegmentCycleScreening,
) -> None:
    assert actual.state is expected.state
    assert actual.aabb_separated is expected.aabb_separated
    assert actual.surface_complete is expected.surface_complete
    assert (actual.relation is None) is (expected.relation is None)
    if expected.relation is not None and actual.relation is not None:
        _assert_relation_equal(expected.relation, actual.relation)


def _transform_points(
    points: Iterable[Iterable[float]],
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: float,
) -> np.ndarray:
    coordinates = np.asarray(tuple(points), dtype=np.float64)
    return coordinates @ rotation.T * scale + translation


def _random_planar_cases(
    seed: int = 20260929,
    count: int = 16,
) -> Tuple[Tuple[Cycle, Plane, Tuple[Segment, ...]], ...]:
    rng = np.random.default_rng(seed)
    cases = []
    for case_index in range(count):
        vertex_count = int(rng.integers(3, 10))
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, vertex_count))
        radii = rng.uniform(0.5, 2.5, vertex_count)
        local_cycle = np.column_stack(
            (
                radii * np.cos(angles),
                radii * np.sin(angles),
                np.zeros(vertex_count),
            )
        )
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        scale = float(10.0 ** rng.uniform(-2.0, 2.0))
        translation = rng.normal(size=3) * scale
        cycle_coordinates = _transform_points(
            local_cycle,
            rotation,
            translation,
            scale,
        )
        if case_index % 5 == 0 and vertex_count >= 4:
            cycle_coordinates[[1, 2]] = cycle_coordinates[[2, 1]]
        cycle = Cycle(cycle_coordinates)
        plane = Plane(translation, rotation[:, 2])

        extent = float(np.max(radii))
        local_segments = [
            ((0, 0, -2), (0, 0, 2)),
            ((4 * extent, 0, -2), (4 * extent, 0, 2)),
            ((-4 * extent, 0, 0), (4 * extent, 0, 0)),
            ((0, 0, 0), (0, 0, 2)),
        ]
        local_segments.extend(
            tuple(rng.normal(size=(2, 3)))
            for _ in range(6)
        )
        segments = tuple(
            Segment(*_transform_points(points, rotation, translation, scale))
            for points in local_segments
        )
        cases.append((cycle, plane, segments))
    return tuple(cases)


@pytest.mark.parametrize(
    ("cycle", "settings"),
    (
        (SQUARE, DEFAULT_GEOMETRY_SETTINGS),
        (
            Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0.008), (0, 2, 0))),
            _settings(absolute_length=1.0e-3),
        ),
        (
            Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0.3), (0, 2, 0))),
            DEFAULT_GEOMETRY_SETTINGS,
        ),
        (
            Cycle(((0, 0, 0), (1, 0, 0), (2, 0, 0))),
            DEFAULT_GEOMETRY_SETTINGS,
        ),
        (
            Cycle(((0, 0, 0), (1, 0, 0), (0, math.nan, 0))),
            DEFAULT_GEOMETRY_SETTINGS,
        ),
    ),
)
def test_native_planarity_matches_python_all_evidence(cycle, settings):
    _assert_planarity_equal(
        measure_planarity(cycle, settings),
        native.measure_planarity(cycle, settings),
    )


@pytest.mark.parametrize("vertex_count", (0, 1, 2))
@pytest.mark.parametrize(
    "operation",
    (
        _geometry_native.measure_planarity,
        _geometry_native.prepare_planar_cycle,
    ),
)
def test_raw_native_cycle_entry_rejects_fewer_than_three_vertices(
    vertex_count,
    operation,
):
    cycle = np.empty((vertex_count, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="at least three vertices"):
        operation(cycle, _raw_tolerances(DEFAULT_GEOMETRY_SETTINGS))


def test_prepared_cycle_owns_query_tolerances():
    loose_settings = _settings(absolute_length=1.0e-3, parameter=1.0e-3)
    tight_settings = _settings(absolute_length=1.0e-8, parameter=1.0e-10)
    cycle = np.asarray(
        tuple(point.coordinates for point in SQUARE.vertices),
        dtype=np.float64,
    )
    loose_cycle = _geometry_native.prepare_planar_cycle(
        cycle,
        _raw_tolerances(loose_settings),
    )
    tight_cycle = _geometry_native.prepare_planar_cycle(
        cycle,
        _raw_tolerances(tight_settings),
    )
    near_boundary = np.asarray((2.0 + 5.0e-4, 1.0, 0.0))
    plane_origin = np.asarray((0.0, 0.0, 0.0))
    plane_normal = np.asarray((0.0, 0.0, 1.0))

    assert _geometry_native.locate_point_in_planar_cycle(
        near_boundary,
        loose_cycle,
        plane_origin,
        plane_normal,
    ) == _geometry_native.PointCycleLocation.BOUNDARY
    assert _geometry_native.locate_point_in_planar_cycle(
        near_boundary,
        tight_cycle,
        plane_origin,
        plane_normal,
    ) == _geometry_native.PointCycleLocation.EXTERIOR

    segment_start = np.asarray((1.0, 1.0, 1.0e-4))
    segment_end = np.asarray((1.0, 1.0, -1.0))
    loose_relation = _geometry_native.determine_planar_segment_cycle_relation(
        segment_start,
        segment_end,
        loose_cycle,
    )
    tight_relation = _geometry_native.determine_planar_segment_cycle_relation(
        segment_start,
        segment_end,
        tight_cycle,
    )
    assert (
        loose_relation.state
        == _geometry_native.PiercingState.DOES_NOT_PIERCE
    )
    assert tight_relation.state == _geometry_native.PiercingState.PIERCES

    with pytest.raises(TypeError):
        _geometry_native.determine_planar_segment_cycle_relation(
            segment_start,
            segment_end,
            loose_cycle,
            _raw_tolerances(tight_settings),
        )

    _assert_relation_equal(
        determine_segment_cycle_relation(
            Segment(segment_start, segment_end),
            SQUARE,
            loose_settings,
        ),
        native.determine_planar_segment_cycle_relation(
            Segment(segment_start, segment_end),
            SQUARE,
            loose_settings,
        ),
    )


@pytest.mark.parametrize(
    "point",
    (
        Point((1, 1, 0)),
        Point((2, 1, 0)),
        Point((2 + 2.0e-8, 1, 0)),
        Point((3, 1, 0)),
        Point((1, 1, 4)),
    ),
)
def test_native_planar_point_location_matches_python(point):
    assert native.locate_point_in_planar_cycle(point, SQUARE, XY_PLANE) is (
        locate_point_in_planar_cycle(point, SQUARE, XY_PLANE)
    )


def test_native_planar_scalar_relations_and_closest_edges_match_python():
    segments = (
        Segment((1, 1, -1), (1, 1, 1)),
        Segment((1, 1, 1), (1, 1, 2)),
        Segment((0, 0, -1), (0, 0, 1)),
        Segment((1, 1, 0), (1, 1, 1)),
        Segment((-1, 1, 0), (3, 1, 0)),
        Segment((4, 4, -1), (4, 4, 1)),
        Segment((1, 1, 1), (1, 1, 1)),
        Segment((1, 1, math.nan), (1, 1, 1)),
    )
    for segment in segments:
        _assert_relation_equal(
            determine_segment_cycle_relation(segment, SQUARE),
            native.determine_planar_segment_cycle_relation(segment, SQUARE),
        )
        _assert_closest_edge_equal(
            closest_cycle_edge(SQUARE, segment),
            native.closest_cycle_edge(SQUARE, segment),
            scale=4.0,
        )


def test_native_planar_batches_match_python_over_curated_and_randomized_cases():
    bow_tie = Cycle(((0, 0, 0), (2, 2, 0), (0, 2, 0), (2, 0, 0)))
    curated_segments = (
        Segment((0.5, 1, -1), (0.5, 1, 1)),
        Segment((5, 5, -1), (5, 5, 1)),
    )
    cases = ((bow_tie, XY_PLANE, curated_segments),) + _random_planar_cases()

    for cycle, plane, segments in cases:
        _assert_planarity_equal(
            measure_planarity(cycle),
            native.measure_planarity(cycle),
        )
        for point in (cycle.vertices[0], plane.point):
            assert native.locate_point_in_planar_cycle(
                point,
                cycle,
                plane,
            ) is locate_point_in_planar_cycle(point, cycle, plane)

        expected_relations = tuple(iter_segment_cycle_relations(segments, cycle))
        actual_relations = native.planar_segment_cycle_relations(segments, cycle)
        assert len(actual_relations) == len(expected_relations)
        for expected, actual in zip(expected_relations, actual_relations):
            _assert_relation_equal(expected, actual)

        expected_screenings = tuple(
            iter_segment_cycle_screenings(segments, cycle)
        )
        actual_screenings = native.planar_segment_cycle_screenings(
            segments,
            cycle,
        )
        assert len(actual_screenings) == len(expected_screenings)
        for expected, actual in zip(expected_screenings, actual_screenings):
            _assert_screening_equal(expected, actual)
