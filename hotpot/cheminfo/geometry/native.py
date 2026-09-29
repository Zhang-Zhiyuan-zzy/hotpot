"""Typed Python adapters for the canonical C++ geometry kernels.

This module is the direct Python entry to the native geometry API during the
staged migration. It intentionally does not provide a Python numerical
fallback. The stable package facade is switched to these adapters only after
each migration phase passes its characterization fence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence, Tuple, Union

import numpy as np

try:
    from . import _geometry_native as _native
except ImportError as exc:
    raise ImportError(
        "the hotpot.cheminfo.geometry native extension is unavailable; "
        "install a compatible Hotpot wheel or rebuild Hotpot from source"
    ) from exc

from .object import Line, Point, Segment
from .settings import DEFAULT_GEOMETRY_SETTINGS, GeometrySettings

if TYPE_CHECKING:
    from .relation import LineRelation, PointPairDistance


__all__ = [
    "NumericTolerances",
    "PointSegmentMeasurement",
    "SegmentSegmentMeasurement",
    "determine_line_relation",
    "line_distance",
    "point_segment_measurement",
    "point_segment_distance",
    "segment_segment_measurement",
    "segment_segment_distance",
    "point_pair_distances",
    "find_point_pairs_below_distance",
    "aabb_bounds",
    "aabb_separation_mask",
    "segment_aabb_separation_mask",
    "aabb_candidate_pairs",
]


NumericTolerances = _native.NumericTolerances
PointSegmentMeasurement = _native.PointSegmentMeasurement
SegmentSegmentMeasurement = _native.SegmentSegmentMeasurement

Coordinates = Union[Sequence[float], Point]
BoundsInput = Union[np.ndarray, Sequence[Sequence[float]]]


def _point_array(point: Coordinates) -> np.ndarray:
    coordinates = point.coordinates if isinstance(point, Point) else point
    return np.ascontiguousarray(coordinates, dtype=np.float64)


def _point_matrix(points: Iterable[Coordinates]) -> np.ndarray:
    return np.ascontiguousarray(
        [
            point.coordinates if isinstance(point, Point) else point
            for point in points
        ],
        dtype=np.float64,
    ).reshape((-1, 3))


def _segment_batch(segments: Iterable[Segment]) -> np.ndarray:
    return np.ascontiguousarray(
        [
            (segment.start.coordinates, segment.end.coordinates)
            for segment in segments
        ],
        dtype=np.float64,
    ).reshape((-1, 2, 3))


def _bounds_batch(bounds: BoundsInput) -> np.ndarray:
    return np.ascontiguousarray(bounds, dtype=np.float64).reshape((-1, 2, 3))


def _paddings(
    padding: Union[float, Sequence[float], np.ndarray],
    count: int,
) -> np.ndarray:
    values = np.asarray(padding, dtype=np.float64)
    if values.ndim == 0:
        return np.full(count, float(values), dtype=np.float64)
    return np.ascontiguousarray(values, dtype=np.float64).reshape((-1,))


def _native_tolerances(settings: GeometrySettings) -> NumericTolerances:
    tolerance = settings.tolerance
    return NumericTolerances(
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


def determine_line_relation(
    first: Line,
    second: Line,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> LineRelation:
    """Classify two infinite lines through the canonical C++ kernel."""

    from .relation import LineRelation, LineRelationKind

    result = _native.determine_line_relation(
        _point_array(first.origin),
        _point_array(first.direction),
        _point_array(second.origin),
        _point_array(second.direction),
        _native_tolerances(settings),
    )
    return LineRelation(
        LineRelationKind[result.kind.name],
        result.distance,
        result.parallel_measure,
    )


def line_distance(
    first: Line,
    second: Line,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float:
    """Return the native infinite-line distance."""

    return _native.line_distance(
        _point_array(first.origin),
        _point_array(first.direction),
        _point_array(second.origin),
        _point_array(second.direction),
        _native_tolerances(settings),
    )


def point_segment_measurement(
    point: Point,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PointSegmentMeasurement:
    """Return native distance, closest-point and parameter evidence."""

    return _native.point_segment_measurement(
        _point_array(point),
        _point_array(segment.start),
        _point_array(segment.end),
        _native_tolerances(settings),
    )


def point_segment_distance(
    point: Point,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float:
    """Measure the Euclidean distance from a point to a finite segment."""

    return _native.point_segment_distance(
        _point_array(point),
        _point_array(segment.start),
        _point_array(segment.end),
        _native_tolerances(settings),
    )


def segment_segment_measurement(
    first: Segment,
    second: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> SegmentSegmentMeasurement:
    """Return native distance, closest-point and parameter evidence."""

    return _native.segment_segment_measurement(
        _point_array(first.start),
        _point_array(first.end),
        _point_array(second.start),
        _point_array(second.end),
        _native_tolerances(settings),
    )


def segment_segment_distance(
    first: Segment,
    second: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float:
    """Measure the minimum Euclidean distance between finite segments."""

    return _native.segment_segment_distance(
        _point_array(first.start),
        _point_array(first.end),
        _point_array(second.start),
        _point_array(second.end),
        _native_tolerances(settings),
    )


def _point_pair_results(
    arrays: Tuple[np.ndarray, np.ndarray],
) -> Tuple[PointPairDistance, ...]:
    from .relation import PointPairDistance

    indices, distances = arrays
    return tuple(
        PointPairDistance(int(pair[0]), int(pair[1]), float(distance))
        for pair, distance in zip(indices, distances)
    )


def point_pair_distances(
    points: Sequence[Point],
) -> Tuple[PointPairDistance, ...]:
    """Return every unordered point-pair distance in stable index order."""

    return _point_pair_results(
        _native.point_pair_distances(_point_matrix(points), None)
    )


def find_point_pairs_below_distance(
    points: Sequence[Point],
    threshold: float,
) -> Tuple[PointPairDistance, ...]:
    """Return point pairs strictly below an explicit distance threshold."""

    return _point_pair_results(
        _native.find_point_pairs_below_distance(
            _point_matrix(points),
            threshold,
        )
    )


def aabb_bounds(coordinates: Iterable[Coordinates]) -> np.ndarray:
    """Return ``[minimum, maximum]`` bounds for three-dimensional points."""

    return _native.aabb_bounds(_point_matrix(coordinates))


def aabb_separation_mask(
    first_bounds: BoundsInput,
    second_bounds: BoundsInput,
    padding: Union[float, Sequence[float], np.ndarray],
) -> np.ndarray:
    """Return strict guarded-separation facts for paired AABB batches."""

    first = _bounds_batch(first_bounds)
    second = _bounds_batch(second_bounds)
    return _native.aabb_separation_mask(
        first,
        second,
        _paddings(padding, len(first)),
    )


def segment_aabb_separation_mask(
    segments: Iterable[Segment],
    target_bounds: BoundsInput,
    padding: Union[float, Sequence[float], np.ndarray],
) -> np.ndarray:
    """Return strict guarded separation of each segment from one AABB."""

    native_segments = _segment_batch(segments)
    target = np.ascontiguousarray(target_bounds, dtype=np.float64).reshape((2, 3))
    return _native.segment_aabb_separation_mask(
        native_segments,
        target,
        _paddings(padding, len(native_segments)),
    )


def aabb_candidate_pairs(
    first_bounds: BoundsInput,
    second_bounds: BoundsInput,
    padding: float,
) -> np.ndarray:
    """Return lexicographically ordered AABB pairs not proven separated."""

    return _native.aabb_candidate_pairs(
        _bounds_batch(first_bounds),
        _bounds_batch(second_bounds),
        padding,
    )
