"""Dimension-aware factual relations between immutable geometry objects.

This module contains no chemical or force-field policy.  The public
classifiers report only geometric facts and explicit numerical uncertainty.
Their mathematical contract and public API are documented in the adjacent
``README.md``; ``README.zh.md`` is its Chinese mirror.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from math import atan2, comb, pi, sqrt
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Tuple, cast

import numpy as np

from .object import Cycle, Line, Plane, Point, Segment, Triangle
from .settings import DEFAULT_GEOMETRY_SETTINGS, GeometrySettings


__all__ = [
    "PlanarityKind",
    "LineRelationKind",
    "PointCycleLocation",
    "SurfaceEmbeddingState",
    "SurfaceSegmentState",
    "PiercingState",
    "SegmentCycleFeature",
    "SegmentCycleIndeterminacy",
    "CycleSurfaceModel",
    "PlanarityMeasurement",
    "LineRelation",
    "PointPairDistance",
    "ClosestCycleEdge",
    "SurfaceFamilyEvidence",
    "SegmentCycleRelation",
    "SegmentCycleScreening",
    "measure_planarity",
    "determine_line_relation",
    "line_distance",
    "point_segment_distance",
    "segment_segment_distance",
    "point_pair_distances",
    "find_point_pairs_below_distance",
    "locate_point_in_planar_cycle",
    "iter_segment_cycle_relations",
    "iter_segment_cycle_screenings",
    "determine_segment_cycle_relation",
    "closest_cycle_edge",
]


# Public relation vocabulary and immutable evidence records.


class PlanarityKind(Enum):
    PLANAR = "planar"
    NONPLANAR = "nonplanar"
    DEGENERATE = "degenerate"
    UNDETERMINED = "undetermined"


class LineRelationKind(Enum):
    INTERSECTING = "intersecting"
    PARALLEL = "parallel"
    COINCIDENT = "coincident"
    SKEW = "skew"
    DEGENERATE = "degenerate"
    UNDETERMINED = "undetermined"


class PointCycleLocation(Enum):
    INTERIOR = "interior"
    BOUNDARY = "boundary"
    EXTERIOR = "exterior"
    UNDETERMINED = "undetermined"


class SurfaceEmbeddingState(Enum):
    EMBEDDED = "embedded"
    PROVEN_NON_EMBEDDED = "proven_non_embedded"
    CONSTRUCTION_UNDETERMINED = "construction_undetermined"


class SurfaceSegmentState(Enum):
    INTERSECTING = "intersecting"
    NON_PIERCING = "non_piercing"
    EVALUATION_UNDETERMINED = "evaluation_undetermined"


class PiercingState(Enum):
    PIERCES = "pierces"
    DOES_NOT_PIERCE = "does_not_pierce"
    UNDETERMINED = "undetermined"


class SegmentCycleFeature(Enum):
    TRANSVERSE_INTERIOR = "transverse_interior"
    LINE_EXTENSION_INTERIOR = "line_extension_interior"
    CYCLE_EDGE_CONTACT = "cycle_edge_contact"
    CYCLE_VERTEX_CONTACT = "cycle_vertex_contact"
    SEGMENT_ENDPOINT_CONTACT = "segment_endpoint_contact"
    COPLANAR_CONTACT = "coplanar_contact"


class SegmentCycleIndeterminacy(Enum):
    NONFINITE_INPUT = "nonfinite_input"
    NUMERIC_BAND = "numeric_band"
    TOLERANCE_DOMAIN = "tolerance_domain"
    DEGENERATE_CYCLE = "degenerate_cycle"
    DEGENERATE_SEGMENT = "degenerate_segment"
    DEGENERATE_TRIANGLE = "degenerate_triangle"
    SELF_INTERSECTION = "self_intersection"
    SURFACE_DISAGREEMENT = "surface_disagreement"
    INCOMPLETE_SURFACE_FAMILY = "incomplete_surface_family"
    SURFACE_CONSTRUCTION = "surface_construction"


class CycleSurfaceModel(Enum):
    PLANAR_POLYGON = "planar_polygon"
    VERTEX_TRIANGULATION_FAMILY = "vertex_triangulation_family"


@dataclass(frozen=True)
class PlanarityMeasurement:
    kind: PlanarityKind
    centroid: Point
    normal: Optional[Tuple[float, float, float]]
    singular_values: Tuple[float, float, float]
    maximum_deviation: float
    rms_deviation: float
    length_scale: float
    length_tolerance: float


@dataclass(frozen=True)
class LineRelation:
    kind: LineRelationKind
    distance: Optional[float]
    parallel_measure: float


@dataclass(frozen=True)
class PointPairDistance:
    first_index: int
    second_index: int
    distance: float


@dataclass(frozen=True)
class ClosestCycleEdge:
    edge_index: int
    edge: Segment
    distance: float


@dataclass(frozen=True)
class SurfaceFamilyEvidence:
    enumeration_complete: bool
    enumerated_surface_count: int
    embedded_surface_count: int
    proven_non_embedded_surface_count: int
    construction_undetermined_count: int
    intersecting_surface_count: int
    non_piercing_surface_count: int
    evaluation_undetermined_count: int
    segment_triangle_tests_used: int
    triangle_pair_tests_used: int


@dataclass(frozen=True)
class SegmentCycleRelation:
    state: PiercingState
    features: FrozenSet[SegmentCycleFeature]
    indeterminacy_causes: FrozenSet[SegmentCycleIndeterminacy]
    surface_model: Optional[CycleSurfaceModel]
    intersection_points: Tuple[Point, ...]
    closest_boundary_edge: Optional[ClosestCycleEdge]
    surface_evidence: SurfaceFamilyEvidence
    settings: GeometrySettings


@dataclass(frozen=True)
class SegmentCycleScreening:
    """State-only screening result for one finite segment and one cycle.

    ``relation`` is omitted only when strict AABB separation proves that the
    finite segment cannot meet any valid surface represented by the cycle.
    """

    state: PiercingState
    relation: Optional[SegmentCycleRelation]
    aabb_separated: bool
    surface_complete: bool


# Private numerical records shared by the primitive and surface kernels.


@dataclass(frozen=True)
class _PredicateTolerances:
    length_scale: float
    length: float
    parameter: float
    area: float
    volume: float
    aabb: float
    merge: float


class _PolygonSimplicity(Enum):
    SIMPLE = "simple"
    SELF_INTERSECTING = "self_intersecting"
    UNDETERMINED = "undetermined"


class _TriangleHitKind(Enum):
    STRICT_INTERIOR = "strict_interior"
    TRIANGLE_BOUNDARY = "triangle_boundary"
    SEGMENT_ENDPOINT = "segment_endpoint"
    LINE_EXTENSION_INTERIOR = "line_extension_interior"
    COPLANAR = "coplanar"
    SEPARATED = "separated"
    DEGENERATE = "degenerate"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class _TriangleHit:
    kind: _TriangleHitKind
    point: Optional[np.ndarray]
    barycentric: Optional[Tuple[float, float, float]]


@dataclass(frozen=True)
class _SurfaceSegmentResult:
    state: SurfaceSegmentState
    features: FrozenSet[SegmentCycleFeature]
    causes: FrozenSet[SegmentCycleIndeterminacy]
    points: Tuple[Point, ...]
    tests_used: int


@dataclass
class _SurfaceCounters:
    segment_triangle_tests: int = 0
    triangle_pair_tests: int = 0
    segment_triangle_budget_exhausted: bool = False
    triangle_pair_budget_exhausted: bool = False


_TriangleIndices = Tuple[int, int, int]
_SurfaceIndices = Tuple[_TriangleIndices, ...]
_EdgeIndices = Tuple[int, int]


@dataclass(frozen=True)
class _CycleTopologyTemplate:
    """Immutable, coordinate-free topology shared by equal-sized cycles."""

    triangulations: Tuple[_SurfaceIndices, ...]
    internal_edges: Tuple[Tuple[_EdgeIndices, ...], ...]
    triangle_pairs: Tuple[Tuple[Tuple[int, int], ...], ...]
    shared_simplices: Tuple[Tuple[Tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class _PreparedNonplanarSurfaceFamily:
    enumeration_complete: bool
    enumerated_surface_count: int
    embedded_surfaces: Tuple[Tuple[Tuple[int, int, int], ...], ...]
    embedded_internal_edges: Tuple[Tuple[_EdgeIndices, ...], ...]
    proven_non_embedded_surface_count: int
    construction_undetermined_count: int
    triangle_pair_tests_used: int
    causes: FrozenSet[SegmentCycleIndeterminacy]


@dataclass(frozen=True)
class _PreparedCycleGeometry:
    """Coordinate-dependent facts shared by segment--cycle predicates."""

    cycle: Cycle
    coordinates: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    planarity: PlanarityMeasurement
    planar_simplicity: Optional[_PolygonSimplicity]
    nonplanar_surface_family: Optional[_PreparedNonplanarSurfaceFamily]


# Dimension-aware numerical helpers.


def _point_array(point: Point) -> np.ndarray:
    return np.asarray(point.coordinates, dtype=np.float64)


def _segment_arrays(segment: Segment) -> Tuple[np.ndarray, np.ndarray]:
    return _point_array(segment.start), _point_array(segment.end)


def _all_finite(arrays: Iterable[np.ndarray]) -> bool:
    return all(bool(np.all(np.isfinite(array))) for array in arrays)


def _diameter(coordinates: np.ndarray) -> float:
    if len(coordinates) < 2:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        differences = (
            coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        )
        diameter = float(np.max(np.linalg.norm(differences, axis=2)))
    return diameter


def _local_length_scale(
    points: Sequence[Point],
    segments: Sequence[Segment] = (),
    cycle: Optional[Cycle] = None,
) -> float:
    coordinates = np.asarray([point.coordinates for point in points], dtype=np.float64)
    if not _all_finite((coordinates,)):
        return float("nan")
    candidates: List[float] = [_diameter(coordinates)]
    with np.errstate(over="ignore", invalid="ignore"):
        candidates.extend(segment.length for segment in segments)
    if cycle is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            edge_lengths = np.asarray(
                [edge.length for edge in cycle.edges], dtype=np.float64
            )
        if not bool(np.all(np.isfinite(edge_lengths))):
            return float("nan")
        candidates.append(float(np.median(edge_lengths)))
    return max(candidates)


def _predicate_tolerances(
    length_scale: float,
    settings: GeometrySettings,
) -> _PredicateTolerances:
    tolerance = settings.tolerance
    relative = max(
        tolerance.relative_length,
        tolerance.machine_epsilon_factor * np.finfo(np.float64).eps,
    )
    length = tolerance.absolute_length + relative * length_scale
    parameter = tolerance.parameter + length / length_scale
    return _PredicateTolerances(
        length_scale=length_scale,
        length=length,
        parameter=parameter,
        area=length * length_scale,
        volume=length * length_scale * length_scale,
        aabb=tolerance.aabb_padding_factor * length,
        merge=tolerance.intersection_merge_factor * length,
    )


def _empty_evidence(enumeration_complete: bool = False) -> SurfaceFamilyEvidence:
    return SurfaceFamilyEvidence(
        enumeration_complete,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def _nan_planarity(kind: PlanarityKind) -> PlanarityMeasurement:
    nan = float("nan")
    return PlanarityMeasurement(
        kind,
        Point((nan, nan, nan)),
        None,
        (nan, nan, nan),
        nan,
        nan,
        nan,
        nan,
    )


def _parameter_domain_is_valid(
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
) -> bool:
    return (
        settings.tolerance.predicate_guard_factor * tolerances.parameter < 0.5
    )


def _fit_plane_svd(
    cycle: Cycle,
    settings: GeometrySettings,
) -> PlanarityMeasurement:
    coordinates = np.asarray(
        [point.coordinates for point in cycle.vertices], dtype=np.float64
    )
    if not bool(np.all(np.isfinite(coordinates))):
        return _nan_planarity(PlanarityKind.UNDETERMINED)

    length_scale = _local_length_scale(cycle.vertices, cycle=cycle)
    if not np.isfinite(length_scale):
        return _nan_planarity(PlanarityKind.UNDETERMINED)
    tolerance = settings.tolerance
    length_tolerance = tolerance.absolute_length
    if length_scale > 0.0:
        length_tolerance = _predicate_tolerances(length_scale, settings).length

    centroid_coordinates = np.mean(coordinates, axis=0)
    centroid = Point(centroid_coordinates)
    centered = coordinates - centroid_coordinates
    _, singular_values_array, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    padded = np.zeros(3, dtype=np.float64)
    padded[: len(singular_values_array)] = singular_values_array
    singular_values = tuple(float(value) for value in padded)
    guard = tolerance.predicate_guard_factor

    if length_scale <= tolerance.absolute_length:
        return PlanarityMeasurement(
            PlanarityKind.DEGENERATE,
            centroid,
            None,
            singular_values,
            float("nan"),
            float("nan"),
            length_scale,
            length_tolerance,
        )

    tolerances = _predicate_tolerances(length_scale, settings)
    first_singular, second_singular, _ = singular_values
    if first_singular <= tolerances.length:
        kind = PlanarityKind.DEGENERATE
    elif first_singular <= guard * tolerances.length:
        kind = PlanarityKind.UNDETERMINED
    else:
        rank_measure = second_singular / first_singular
        if rank_measure <= tolerances.parameter:
            kind = PlanarityKind.DEGENERATE
        elif rank_measure <= guard * tolerances.parameter:
            kind = PlanarityKind.UNDETERMINED
        else:
            normal_array = right_vectors[-1]
            deviations = np.abs(centered @ normal_array)
            maximum_deviation = float(np.max(deviations))
            rms_deviation = float(sqrt(float(np.mean(deviations * deviations))))
            plane_tolerance = tolerance.planarity_factor * tolerances.length
            if maximum_deviation <= plane_tolerance:
                kind = PlanarityKind.PLANAR
            elif maximum_deviation <= guard * plane_tolerance:
                kind = PlanarityKind.UNDETERMINED
            else:
                kind = PlanarityKind.NONPLANAR
            return PlanarityMeasurement(
                kind,
                centroid,
                tuple(float(value) for value in normal_array),
                singular_values,
                maximum_deviation,
                rms_deviation,
                length_scale,
                tolerances.length,
            )

    return PlanarityMeasurement(
        kind,
        centroid,
        None,
        singular_values,
        float("nan"),
        float("nan"),
        length_scale,
        tolerances.length,
    )


def _normalized_direction(direction: Sequence[float]) -> Optional[np.ndarray]:
    array = np.asarray(direction, dtype=np.float64)
    if not bool(np.all(np.isfinite(array))):
        return None
    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return None
    scaled = array / maximum
    return scaled / np.linalg.norm(scaled)


def _scale_safe_norm(vector: np.ndarray) -> float:
    maximum = float(np.max(np.abs(vector)))
    if maximum == 0.0:
        return 0.0
    if not np.isfinite(maximum):
        return float("nan")
    return maximum * float(np.linalg.norm(vector / maximum))


def _point_segment_distance_arrays(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    direction = end - start
    squared_length = float(np.dot(direction, direction))
    if squared_length == 0.0:
        return float(np.linalg.norm(point - start))
    parameter = float(np.dot(point - start, direction) / squared_length)
    parameter = min(1.0, max(0.0, parameter))
    return float(np.linalg.norm(point - (start + parameter * direction)))


def _segment_segment_distance_arrays(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    squared_length_tolerance: float,
) -> float:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    offset = first_start - second_start
    first_squared = float(np.dot(first_direction, first_direction))
    second_squared = float(np.dot(second_direction, second_direction))
    if first_squared <= squared_length_tolerance:
        return _point_segment_distance_arrays(
            first_start, second_start, second_end
        )
    if second_squared <= squared_length_tolerance:
        return _point_segment_distance_arrays(
            second_start, first_start, first_end
        )

    direction_dot = float(np.dot(first_direction, second_direction))
    first_offset = float(np.dot(first_direction, offset))
    second_offset = float(np.dot(second_direction, offset))
    denominator = first_squared * second_squared - direction_dot * direction_dot
    first_parameter = 0.0
    if denominator != 0.0:
        first_parameter = min(
            1.0,
            max(
                0.0,
                (direction_dot * second_offset - first_offset * second_squared)
                / denominator,
            ),
        )
    second_parameter = (
        direction_dot * first_parameter + second_offset
    ) / second_squared
    if second_parameter < 0.0:
        second_parameter = 0.0
        first_parameter = min(1.0, max(0.0, -first_offset / first_squared))
    elif second_parameter > 1.0:
        second_parameter = 1.0
        first_parameter = min(
            1.0,
            max(0.0, (direction_dot - first_offset) / first_squared),
        )
    first_closest = first_start + first_parameter * first_direction
    second_closest = second_start + second_parameter * second_direction
    return float(np.linalg.norm(first_closest - second_closest))


# Planar projection and polygon predicates.


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis_index = int(np.argmin(np.abs(normal)))
    axis = np.zeros(3, dtype=np.float64)
    axis[axis_index] = 1.0
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def _project_to_plane(
    coordinates: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    first, second = _plane_basis(normal)
    centered = coordinates - origin
    return np.column_stack((centered @ first, centered @ second))


def _orient2d(first: np.ndarray, second: np.ndarray, point: np.ndarray) -> float:
    direction = second - first
    relative = point - first
    return float(direction[0] * relative[1] - direction[1] * relative[0])


def _aabb_stably_separated(
    first: np.ndarray,
    second: np.ndarray,
    padding: float,
) -> bool:
    return bool(
        np.any(
            (np.max(first, axis=0) + padding < np.min(second, axis=0))
            | (np.max(second, axis=0) + padding < np.min(first, axis=0))
        )
    )


def _aabb_bounds(coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.min(coordinates, axis=0), np.max(coordinates, axis=0)


def _segment_cycle_aabbs_stably_separated(
    segment: Segment,
    cycle_bounds: Tuple[np.ndarray, np.ndarray],
    padding: float,
) -> bool:
    """Return whether guarded finite-segment and cycle bounds are disjoint."""
    segment_coordinates = np.asarray(
        (segment.start.coordinates, segment.end.coordinates),
        dtype=np.float64,
    )
    segment_minimum = np.min(segment_coordinates, axis=0)
    segment_maximum = np.max(segment_coordinates, axis=0)
    cycle_minimum, cycle_maximum = cycle_bounds
    return bool(
        np.any(
            (segment_maximum + padding < cycle_minimum)
            | (cycle_maximum + padding < segment_minimum)
        )
    )


def _projected_polygon_simplicity(
    polygon: np.ndarray,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
) -> _PolygonSimplicity:
    count = len(polygon)
    guard = settings.tolerance.predicate_guard_factor
    for first_index in range(count):
        first_next = (first_index + 1) % count
        first_edge = np.asarray(
            [polygon[first_index], polygon[first_next]], dtype=np.float64
        )
        for second_index in range(first_index + 1, count):
            second_next = (second_index + 1) % count
            if (
                first_index == second_index
                or first_next == second_index
                or second_next == first_index
            ):
                continue
            second_edge = np.asarray(
                [polygon[second_index], polygon[second_next]], dtype=np.float64
            )
            if _aabb_stably_separated(first_edge, second_edge, tolerances.aabb):
                continue
            orientations = (
                _orient2d(first_edge[0], first_edge[1], second_edge[0]),
                _orient2d(first_edge[0], first_edge[1], second_edge[1]),
                _orient2d(second_edge[0], second_edge[1], first_edge[0]),
                _orient2d(second_edge[0], second_edge[1], first_edge[1]),
            )
            if any(abs(value) <= guard * tolerances.area for value in orientations):
                return _PolygonSimplicity.UNDETERMINED
            if (
                np.sign(orientations[0]) != np.sign(orientations[1])
                and np.sign(orientations[2]) != np.sign(orientations[3])
            ):
                return _PolygonSimplicity.SELF_INTERSECTING
    return _PolygonSimplicity.SIMPLE


def _locate_projected_point(
    point: np.ndarray,
    polygon: np.ndarray,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
) -> PointCycleLocation:
    boundary_distance = min(
        _point_segment_distance_arrays(
            point,
            polygon[index],
            polygon[(index + 1) % len(polygon)],
        )
        for index in range(len(polygon))
    )
    guard = settings.tolerance.predicate_guard_factor
    if boundary_distance <= tolerances.length:
        return PointCycleLocation.BOUNDARY
    if boundary_distance <= guard * tolerances.length:
        return PointCycleLocation.UNDETERMINED

    angle_sum = 0.0
    for index in range(len(polygon)):
        first = polygon[index] - point
        second = polygon[(index + 1) % len(polygon)] - point
        angle_sum += atan2(
            _orient2d(np.zeros(2), first, second),
            float(np.dot(first, second)),
        )
    winding = angle_sum / (2.0 * pi)
    winding_tolerance = settings.tolerance.winding_residual
    if abs(abs(winding) - 1.0) <= winding_tolerance:
        return PointCycleLocation.INTERIOR
    if abs(winding) <= winding_tolerance:
        return PointCycleLocation.EXTERIOR
    return PointCycleLocation.UNDETERMINED


def _projected_segment_polygon_contact(
    segment: np.ndarray,
    polygon: np.ndarray,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
) -> Tuple[bool, bool]:
    locations = tuple(
        _locate_projected_point(point, polygon, tolerances, settings)
        for point in segment
    )
    if any(
        location in (PointCycleLocation.INTERIOR, PointCycleLocation.BOUNDARY)
        for location in locations
    ):
        return True, False
    if any(location is PointCycleLocation.UNDETERMINED for location in locations):
        return False, True

    squared_tolerance = tolerances.length * tolerances.length
    edge_distances = tuple(
        _segment_segment_distance_arrays(
            segment[0],
            segment[1],
            polygon[index],
            polygon[(index + 1) % len(polygon)],
            squared_tolerance,
        )
        for index in range(len(polygon))
    )
    if min(edge_distances) <= tolerances.length:
        return True, False
    if min(edge_distances) <= (
        settings.tolerance.predicate_guard_factor * tolerances.length
    ):
        return False, True
    return False, False


def _coplanar_segment_triangle_contact(
    segment: Segment,
    triangle: Triangle,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
) -> Tuple[bool, bool]:
    triangle_coordinates = np.asarray(
        [vertex.coordinates for vertex in triangle.vertices], dtype=np.float64
    )
    normal = np.cross(
        triangle_coordinates[1] - triangle_coordinates[0],
        triangle_coordinates[2] - triangle_coordinates[0],
    )
    normal /= _scale_safe_norm(normal)
    projected_triangle = _project_to_plane(
        triangle_coordinates, triangle_coordinates[0], normal
    )
    projected_segment = _project_to_plane(
        np.asarray(_segment_arrays(segment)), triangle_coordinates[0], normal
    )
    return _projected_segment_polygon_contact(
        projected_segment, projected_triangle, tolerances, settings
    )


# Segment--plane, segment--triangle, and surface construction kernels.


def _barycentric_coordinates(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    normal: np.ndarray,
) -> Tuple[float, float, float]:
    squared_normal = float(np.dot(normal, normal))
    first_weight = float(
        np.dot(np.cross(second - point, third - point), normal) / squared_normal
    )
    second_weight = float(
        np.dot(np.cross(third - point, first - point), normal) / squared_normal
    )
    return first_weight, second_weight, 1.0 - first_weight - second_weight


def _classify_barycentric(
    barycentric: Tuple[float, float, float],
    parameter_tolerance: float,
    guard: float,
) -> _TriangleHitKind:
    if all(value > guard * parameter_tolerance for value in barycentric):
        return _TriangleHitKind.STRICT_INTERIOR
    if (
        all(value >= -parameter_tolerance for value in barycentric)
        and any(abs(value) <= parameter_tolerance for value in barycentric)
    ):
        return _TriangleHitKind.TRIANGLE_BOUNDARY
    if any(value < -guard * parameter_tolerance for value in barycentric):
        return _TriangleHitKind.SEPARATED
    return _TriangleHitKind.UNDETERMINED


def _segment_triangle_relation(
    segment: Segment,
    triangle: Triangle,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
) -> _TriangleHit:
    start, end = _segment_arrays(segment)
    first, second, third = (
        _point_array(vertex) for vertex in triangle.vertices
    )
    if not _all_finite((start, end, first, second, third)):
        return _TriangleHit(_TriangleHitKind.UNDETERMINED, None, None)

    guard = settings.tolerance.predicate_guard_factor
    segment_length = float(np.linalg.norm(end - start))
    if segment_length <= tolerances.length:
        return _TriangleHit(_TriangleHitKind.DEGENERATE, None, None)
    if segment_length <= guard * tolerances.length:
        return _TriangleHit(_TriangleHitKind.UNDETERMINED, None, None)

    normal = np.cross(second - first, third - first)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= tolerances.area:
        return _TriangleHit(_TriangleHitKind.DEGENERATE, None, None)
    if normal_length <= guard * tolerances.area:
        return _TriangleHit(_TriangleHitKind.UNDETERMINED, None, None)
    unit_normal = normal / normal_length
    start_height = float(np.dot(unit_normal, start - first))
    end_height = float(np.dot(unit_normal, end - first))
    start_absolute = abs(start_height)
    end_absolute = abs(end_height)

    if (
        start_absolute <= tolerances.length
        and end_absolute <= tolerances.length
    ):
        return _TriangleHit(_TriangleHitKind.COPLANAR, None, None)
    if (
        tolerances.length < start_absolute <= guard * tolerances.length
        or tolerances.length < end_absolute <= guard * tolerances.length
    ):
        return _TriangleHit(_TriangleHitKind.UNDETERMINED, None, None)

    one_endpoint = (
        start_absolute <= tolerances.length
        and end_absolute > guard * tolerances.length
    ) or (
        end_absolute <= tolerances.length
        and start_absolute > guard * tolerances.length
    )
    height_difference = start_height - end_height
    if one_endpoint:
        point = start if start_absolute <= tolerances.length else end
        barycentric = _barycentric_coordinates(
            point, first, second, third, normal
        )
        location = _classify_barycentric(
            barycentric, tolerances.parameter, guard
        )
        if location in (
            _TriangleHitKind.STRICT_INTERIOR,
            _TriangleHitKind.TRIANGLE_BOUNDARY,
        ):
            return _TriangleHit(
                _TriangleHitKind.SEGMENT_ENDPOINT, point, barycentric
            )
        return _TriangleHit(location, point, barycentric)

    if abs(height_difference) <= tolerances.length:
        return _TriangleHit(_TriangleHitKind.SEPARATED, None, None)
    if abs(height_difference) <= guard * tolerances.length:
        return _TriangleHit(_TriangleHitKind.UNDETERMINED, None, None)
    if not _parameter_domain_is_valid(tolerances, settings):
        return _TriangleHit(_TriangleHitKind.UNDETERMINED, None, None)

    parameter = start_height / height_difference
    point = start + parameter * (end - start)
    barycentric = _barycentric_coordinates(point, first, second, third, normal)
    location = _classify_barycentric(
        barycentric, tolerances.parameter, guard
    )
    if location in (
        _TriangleHitKind.TRIANGLE_BOUNDARY,
        _TriangleHitKind.UNDETERMINED,
    ):
        return _TriangleHit(location, point, barycentric)
    if location is _TriangleHitKind.SEPARATED:
        return _TriangleHit(location, point, barycentric)

    parameter_tolerance = tolerances.parameter
    if guard * parameter_tolerance < parameter < 1.0 - guard * parameter_tolerance:
        return _TriangleHit(_TriangleHitKind.STRICT_INTERIOR, point, barycentric)
    if abs(parameter) <= parameter_tolerance or abs(1.0 - parameter) <= parameter_tolerance:
        return _TriangleHit(_TriangleHitKind.SEGMENT_ENDPOINT, point, barycentric)
    if parameter < -guard * parameter_tolerance or parameter > 1.0 + guard * parameter_tolerance:
        return _TriangleHit(
            _TriangleHitKind.LINE_EXTENSION_INTERIOR, point, barycentric
        )
    return _TriangleHit(_TriangleHitKind.UNDETERMINED, point, barycentric)


def _enumerate_cycle_triangulations(
    vertex_count: int,
) -> Tuple[_SurfaceIndices, ...]:
    cache: Dict[
        Tuple[int, int], Tuple[Tuple[Tuple[int, int, int], ...], ...]
    ] = {}

    def enumerate_interval(
        first: int,
        last: int,
    ) -> Tuple[Tuple[Tuple[int, int, int], ...], ...]:
        key = first, last
        if key in cache:
            return cache[key]
        if last - first < 2:
            result = ((),)
        else:
            surfaces: List[Tuple[Tuple[int, int, int], ...]] = []
            for middle in range(first + 1, last):
                for left in enumerate_interval(first, middle):
                    for right in enumerate_interval(middle, last):
                        surfaces.append(
                            left + right + ((first, middle, last),)
                        )
            result = tuple(surfaces)
        cache[key] = result
        return result

    return enumerate_interval(0, vertex_count - 1)


def _surface_internal_edges(
    surface: _SurfaceIndices,
    vertex_count: int,
) -> Tuple[_EdgeIndices, ...]:
    counts: Dict[_EdgeIndices, int] = {}
    for triangle_indices in surface:
        for index in range(3):
            edge = cast(
                _EdgeIndices,
                tuple(sorted((
                    triangle_indices[index],
                    triangle_indices[(index + 1) % 3],
                ))),
            )
            counts[edge] = counts.get(edge, 0) + 1
    cycle_edges = {
        tuple(sorted((index, (index + 1) % vertex_count)))
        for index in range(vertex_count)
    }
    return tuple(
        edge
        for edge, count in counts.items()
        if count == 2 and edge not in cycle_edges
    )


@lru_cache(maxsize=None)
def _cycle_topology_template(vertex_count: int) -> _CycleTopologyTemplate:
    """Return coordinate-free triangulation facts for one cycle size."""

    triangulations = _enumerate_cycle_triangulations(vertex_count)
    internal_edges = tuple(
        _surface_internal_edges(surface, vertex_count)
        for surface in triangulations
    )
    triangle_pairs = tuple(
        tuple(
            (first_index, second_index)
            for first_index in range(len(surface))
            for second_index in range(first_index + 1, len(surface))
        )
        for surface in triangulations
    )
    shared_simplices = tuple(
        tuple(
            tuple(sorted(
                set(surface[first_index]) & set(surface[second_index])
            ))
            for first_index, second_index in surface_pairs
        )
        for surface, surface_pairs in zip(triangulations, triangle_pairs)
    )
    return _CycleTopologyTemplate(
        triangulations,
        internal_edges,
        triangle_pairs,
        shared_simplices,
    )


def _triangle_from_indices(
    cycle: Cycle,
    indices: Tuple[int, int, int],
) -> Triangle:
    return Triangle(*(cycle.vertices[index] for index in indices))


def _point_to_shared_simplex_distance(
    point: np.ndarray,
    cycle: Cycle,
    shared_indices: Tuple[int, ...],
) -> float:
    if not shared_indices:
        return float("inf")
    if len(shared_indices) == 1:
        return float(
            np.linalg.norm(point - _point_array(cycle.vertices[shared_indices[0]]))
        )
    first = _point_array(cycle.vertices[shared_indices[0]])
    second = _point_array(cycle.vertices[shared_indices[1]])
    return _point_segment_distance_arrays(point, first, second)


def _coplanar_triangle_pair_state(
    first_indices: Tuple[int, int, int],
    second_indices: Tuple[int, int, int],
    cycle: Cycle,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
    shared: Tuple[int, ...],
) -> SurfaceEmbeddingState:
    first_coordinates = np.asarray(
        [cycle.vertices[index].coordinates for index in first_indices],
        dtype=np.float64,
    )
    second_coordinates = np.asarray(
        [cycle.vertices[index].coordinates for index in second_indices],
        dtype=np.float64,
    )
    normal = np.cross(
        first_coordinates[1] - first_coordinates[0],
        first_coordinates[2] - first_coordinates[0],
    )
    normal /= np.linalg.norm(normal)
    first_projected = _project_to_plane(
        first_coordinates, first_coordinates[0], normal
    )
    second_projected = _project_to_plane(
        second_coordinates, first_coordinates[0], normal
    )
    guard = settings.tolerance.predicate_guard_factor

    for first_edge_index in range(3):
        first_a_index = first_indices[first_edge_index]
        first_b_index = first_indices[(first_edge_index + 1) % 3]
        first_edge = np.asarray(
            [
                first_projected[first_edge_index],
                first_projected[(first_edge_index + 1) % 3],
            ]
        )
        for second_edge_index in range(3):
            second_a_index = second_indices[second_edge_index]
            second_b_index = second_indices[(second_edge_index + 1) % 3]
            second_edge = np.asarray(
                [
                    second_projected[second_edge_index],
                    second_projected[(second_edge_index + 1) % 3],
                ]
            )
            shared_edge_vertices = {
                first_a_index,
                first_b_index,
            } & {second_a_index, second_b_index}
            if len(shared_edge_vertices) == 2:
                continue
            if _aabb_stably_separated(first_edge, second_edge, tolerances.aabb):
                continue
            orientations = (
                _orient2d(first_edge[0], first_edge[1], second_edge[0]),
                _orient2d(first_edge[0], first_edge[1], second_edge[1]),
                _orient2d(second_edge[0], second_edge[1], first_edge[0]),
                _orient2d(second_edge[0], second_edge[1], first_edge[1]),
            )
            if any(abs(value) <= guard * tolerances.area for value in orientations):
                allowed_endpoint = bool(shared_edge_vertices)
                if not allowed_endpoint:
                    return SurfaceEmbeddingState.CONSTRUCTION_UNDETERMINED
                continue
            if (
                np.sign(orientations[0]) != np.sign(orientations[1])
                and np.sign(orientations[2]) != np.sign(orientations[3])
            ):
                return SurfaceEmbeddingState.PROVEN_NON_EMBEDDED

    for indices, projected, other_projected in (
        (first_indices, first_projected, second_projected),
        (second_indices, second_projected, first_projected),
    ):
        for local_index, global_index in enumerate(indices):
            if global_index in shared:
                continue
            location = _locate_projected_point(
                projected[local_index], other_projected, tolerances, settings
            )
            if location is PointCycleLocation.INTERIOR:
                return SurfaceEmbeddingState.PROVEN_NON_EMBEDDED
            if location in (
                PointCycleLocation.BOUNDARY,
                PointCycleLocation.UNDETERMINED,
            ):
                return SurfaceEmbeddingState.CONSTRUCTION_UNDETERMINED
    return SurfaceEmbeddingState.EMBEDDED


def _triangle_pair_state(
    first_indices: Tuple[int, int, int],
    second_indices: Tuple[int, int, int],
    cycle: Cycle,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
    shared: Tuple[int, ...],
) -> SurfaceEmbeddingState:
    first_triangle = _triangle_from_indices(cycle, first_indices)
    second_triangle = _triangle_from_indices(cycle, second_indices)
    first_coordinates = np.asarray(
        [vertex.coordinates for vertex in first_triangle.vertices], dtype=np.float64
    )
    second_coordinates = np.asarray(
        [vertex.coordinates for vertex in second_triangle.vertices], dtype=np.float64
    )
    if not shared and _aabb_stably_separated(
        first_coordinates, second_coordinates, tolerances.aabb
    ):
        return SurfaceEmbeddingState.EMBEDDED

    first_normal = np.cross(
        first_coordinates[1] - first_coordinates[0],
        first_coordinates[2] - first_coordinates[0],
    )
    second_normal = np.cross(
        second_coordinates[1] - second_coordinates[0],
        second_coordinates[2] - second_coordinates[0],
    )
    first_normal_length = float(np.linalg.norm(first_normal))
    second_normal_length = float(np.linalg.norm(second_normal))
    guard = settings.tolerance.predicate_guard_factor
    if (
        first_normal_length <= tolerances.area
        or second_normal_length <= tolerances.area
    ):
        return SurfaceEmbeddingState.PROVEN_NON_EMBEDDED
    if (
        first_normal_length <= guard * tolerances.area
        or second_normal_length <= guard * tolerances.area
    ):
        return SurfaceEmbeddingState.CONSTRUCTION_UNDETERMINED

    first_nonshared = [
        second_coordinates[index]
        for index, global_index in enumerate(second_indices)
        if global_index not in shared
    ]
    second_nonshared = [
        first_coordinates[index]
        for index, global_index in enumerate(first_indices)
        if global_index not in shared
    ]
    first_residuals = [
        abs(float(np.dot(first_normal, point - first_coordinates[0])))
        for point in first_nonshared
    ]
    second_residuals = [
        abs(float(np.dot(second_normal, point - second_coordinates[0])))
        for point in second_nonshared
    ]
    residuals = first_residuals + second_residuals
    if residuals and all(value <= tolerances.volume for value in residuals):
        return _coplanar_triangle_pair_state(
            first_indices,
            second_indices,
            cycle,
            tolerances,
            settings,
            shared,
        )
    if any(
        tolerances.volume < value <= guard * tolerances.volume
        for value in residuals
    ):
        return SurfaceEmbeddingState.CONSTRUCTION_UNDETERMINED

    for triangle_indices, triangle, other_indices, other_triangle in (
        (first_indices, first_triangle, second_indices, second_triangle),
        (second_indices, second_triangle, first_indices, first_triangle),
    ):
        for edge_index, edge in enumerate(triangle.edges):
            edge_global = {
                triangle_indices[edge_index],
                triangle_indices[(edge_index + 1) % 3],
            }
            if len(edge_global & set(other_indices)) == 2:
                continue
            hit = _segment_triangle_relation(
                edge, other_triangle, tolerances, settings
            )
            if hit.kind in (
                _TriangleHitKind.DEGENERATE,
                _TriangleHitKind.UNDETERMINED,
                _TriangleHitKind.COPLANAR,
            ):
                return SurfaceEmbeddingState.CONSTRUCTION_UNDETERMINED
            if hit.point is None or hit.kind in (
                _TriangleHitKind.SEPARATED,
                _TriangleHitKind.LINE_EXTENSION_INTERIOR,
            ):
                continue
            distance = _point_to_shared_simplex_distance(hit.point, cycle, shared)
            if distance > guard * tolerances.length:
                return SurfaceEmbeddingState.PROVEN_NON_EMBEDDED
            if distance > tolerances.length:
                return SurfaceEmbeddingState.CONSTRUCTION_UNDETERMINED
    return SurfaceEmbeddingState.EMBEDDED


def _determine_surface_embedding(
    surface: _SurfaceIndices,
    triangle_pairs: Tuple[Tuple[int, int], ...],
    shared_simplices: Tuple[Tuple[int, ...], ...],
    cycle: Cycle,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
    counters: _SurfaceCounters,
) -> SurfaceEmbeddingState:
    guard = settings.tolerance.predicate_guard_factor
    for indices in surface:
        coordinates = np.asarray(
            [cycle.vertices[index].coordinates for index in indices],
            dtype=np.float64,
        )
        area_measure = float(
            np.linalg.norm(
                np.cross(
                    coordinates[1] - coordinates[0],
                    coordinates[2] - coordinates[0],
                )
            )
        )
        if area_measure <= tolerances.area:
            return SurfaceEmbeddingState.PROVEN_NON_EMBEDDED
        if area_measure <= guard * tolerances.area:
            return SurfaceEmbeddingState.CONSTRUCTION_UNDETERMINED

    for (first_index, second_index), shared in zip(
        triangle_pairs, shared_simplices
    ):
        if (
            counters.triangle_pair_tests
            >= settings.surface.maximum_triangle_pair_tests
        ):
            counters.triangle_pair_budget_exhausted = True
            return SurfaceEmbeddingState.CONSTRUCTION_UNDETERMINED
        counters.triangle_pair_tests += 1
        pair_state = _triangle_pair_state(
            surface[first_index],
            surface[second_index],
            cycle,
            tolerances,
            settings,
            shared,
        )
        if pair_state is not SurfaceEmbeddingState.EMBEDDED:
            return pair_state
    return SurfaceEmbeddingState.EMBEDDED


def _point_near_internal_simplex(
    point: np.ndarray,
    internal_edges: Tuple[Tuple[int, int], ...],
    cycle: Cycle,
    tolerance: float,
) -> bool:
    for first_index, second_index in internal_edges:
        distance = _point_segment_distance_arrays(
            point,
            _point_array(cycle.vertices[first_index]),
            _point_array(cycle.vertices[second_index]),
        )
        if distance <= tolerance:
            return True
    return False


def _point_boundary_feature(
    point: np.ndarray,
    cycle: Cycle,
    tolerances: _PredicateTolerances,
) -> SegmentCycleFeature:
    if any(
        float(np.linalg.norm(point - _point_array(vertex))) <= tolerances.length
        for vertex in cycle.vertices
    ):
        return SegmentCycleFeature.CYCLE_VERTEX_CONTACT
    return SegmentCycleFeature.CYCLE_EDGE_CONTACT


def _merge_points(
    points: Sequence[np.ndarray],
    merge_tolerance: float,
) -> Tuple[Point, ...]:
    merged: List[np.ndarray] = []
    for point in points:
        if not any(
            float(np.linalg.norm(point - current)) <= merge_tolerance
            for current in merged
        ):
            merged.append(point)
    return tuple(Point(point) for point in merged)


def _surface_segment_relation(
    segment: Segment,
    surface: _SurfaceIndices,
    internal_edges: Tuple[_EdgeIndices, ...],
    cycle: Cycle,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
    counters: _SurfaceCounters,
) -> _SurfaceSegmentResult:
    features = set()
    causes = set()
    points: List[np.ndarray] = []
    confirmed_intersection = False
    evaluation_undetermined = False
    guard = settings.tolerance.predicate_guard_factor

    for indices in surface:
        if (
            counters.segment_triangle_tests
            >= settings.surface.maximum_segment_triangle_tests
        ):
            counters.segment_triangle_budget_exhausted = True
            causes.add(SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY)
            evaluation_undetermined = True
            break
        counters.segment_triangle_tests += 1
        triangle = _triangle_from_indices(cycle, indices)
        hit = _segment_triangle_relation(
            segment,
            triangle,
            tolerances,
            settings,
        )
        if hit.kind is _TriangleHitKind.STRICT_INTERIOR and hit.point is not None:
            if _point_near_internal_simplex(
                hit.point,
                internal_edges,
                cycle,
                guard * tolerances.length,
            ):
                evaluation_undetermined = True
                causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)
            else:
                confirmed_intersection = True
                features.add(SegmentCycleFeature.TRANSVERSE_INTERIOR)
                points.append(hit.point)
        elif hit.kind is _TriangleHitKind.LINE_EXTENSION_INTERIOR:
            features.add(SegmentCycleFeature.LINE_EXTENSION_INTERIOR)
        elif hit.kind is _TriangleHitKind.TRIANGLE_BOUNDARY and hit.point is not None:
            if _point_near_internal_simplex(
                hit.point,
                internal_edges,
                cycle,
                guard * tolerances.length,
            ):
                evaluation_undetermined = True
                causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)
            else:
                features.add(_point_boundary_feature(hit.point, cycle, tolerances))
                points.append(hit.point)
        elif hit.kind is _TriangleHitKind.SEGMENT_ENDPOINT:
            if hit.point is not None:
                on_cycle_boundary = any(
                    _point_segment_distance_arrays(
                        hit.point,
                        _point_array(edge.start),
                        _point_array(edge.end),
                    )
                    <= tolerances.length
                    for edge in cycle.edges
                )
                if on_cycle_boundary:
                    features.add(SegmentCycleFeature.SEGMENT_ENDPOINT_CONTACT)
                    features.add(
                        _point_boundary_feature(hit.point, cycle, tolerances)
                    )
                    points.append(hit.point)
                elif _point_near_internal_simplex(
                    hit.point,
                    internal_edges,
                    cycle,
                    guard * tolerances.length,
                ):
                    evaluation_undetermined = True
                    causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)
                else:
                    features.add(SegmentCycleFeature.SEGMENT_ENDPOINT_CONTACT)
                    points.append(hit.point)
        elif hit.kind is _TriangleHitKind.COPLANAR:
            has_contact, contact_undetermined = _coplanar_segment_triangle_contact(
                segment, triangle, tolerances, settings
            )
            if has_contact:
                features.add(SegmentCycleFeature.COPLANAR_CONTACT)
            elif contact_undetermined:
                evaluation_undetermined = True
                causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)
        elif hit.kind is _TriangleHitKind.DEGENERATE:
            evaluation_undetermined = True
            causes.add(SegmentCycleIndeterminacy.DEGENERATE_TRIANGLE)
        elif hit.kind is _TriangleHitKind.UNDETERMINED:
            evaluation_undetermined = True
            causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)

    if confirmed_intersection:
        state = SurfaceSegmentState.INTERSECTING
    elif evaluation_undetermined:
        state = SurfaceSegmentState.EVALUATION_UNDETERMINED
    else:
        state = SurfaceSegmentState.NON_PIERCING
    return _SurfaceSegmentResult(
        state,
        frozenset(features),
        frozenset(causes),
        _merge_points(points, tolerances.merge),
        counters.segment_triangle_tests,
    )


def _segment_cycle_base_data(
    segment: Segment,
    cycle: Cycle,
    settings: GeometrySettings,
) -> Tuple[Optional[_PredicateTolerances], FrozenSet[SegmentCycleIndeterminacy]]:
    points = (segment.start, segment.end) + cycle.vertices
    arrays = tuple(_point_array(point) for point in points)
    if not _all_finite(arrays):
        return None, frozenset({SegmentCycleIndeterminacy.NONFINITE_INPUT})
    length_scale = _local_length_scale(points, (segment,), cycle)
    if not np.isfinite(length_scale):
        return None, frozenset({SegmentCycleIndeterminacy.NUMERIC_BAND})
    if length_scale <= settings.tolerance.absolute_length:
        return None, frozenset({SegmentCycleIndeterminacy.DEGENERATE_CYCLE})
    tolerances = _predicate_tolerances(length_scale, settings)
    if not _parameter_domain_is_valid(tolerances, settings):
        return tolerances, frozenset({SegmentCycleIndeterminacy.TOLERANCE_DOMAIN})
    segment_length = segment.length
    guard = settings.tolerance.predicate_guard_factor
    if segment_length <= tolerances.length:
        return tolerances, frozenset({SegmentCycleIndeterminacy.DEGENERATE_SEGMENT})
    if segment_length <= guard * tolerances.length:
        return tolerances, frozenset({SegmentCycleIndeterminacy.NUMERIC_BAND})
    return tolerances, frozenset()


def _undetermined_segment_cycle_relation(
    segment: Segment,
    cycle: Cycle,
    model: Optional[CycleSurfaceModel],
    settings: GeometrySettings,
    causes: FrozenSet[SegmentCycleIndeterminacy],
    evidence: Optional[SurfaceFamilyEvidence] = None,
) -> SegmentCycleRelation:
    return SegmentCycleRelation(
        PiercingState.UNDETERMINED,
        frozenset(),
        causes,
        model,
        tuple(),
        closest_cycle_edge(cycle, segment, settings),
        evidence if evidence is not None else _empty_evidence(),
        settings,
    )


def _planar_segment_cycle_relation(
    segment: Segment,
    cycle: Cycle,
    planarity: PlanarityMeasurement,
    simplicity: _PolygonSimplicity,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
) -> SegmentCycleRelation:
    normal = np.asarray(planarity.normal, dtype=np.float64)
    origin = _point_array(planarity.centroid)
    polygon = _project_to_plane(
        np.asarray([vertex.coordinates for vertex in cycle.vertices]),
        origin,
        normal,
    )
    closest = closest_cycle_edge(cycle, segment, settings)
    if simplicity is _PolygonSimplicity.SELF_INTERSECTING:
        evidence = SurfaceFamilyEvidence(True, 1, 0, 1, 0, 0, 0, 0, 0, 0)
        return SegmentCycleRelation(
            PiercingState.UNDETERMINED,
            frozenset(),
            frozenset({SegmentCycleIndeterminacy.SELF_INTERSECTION}),
            CycleSurfaceModel.PLANAR_POLYGON,
            tuple(),
            closest,
            evidence,
            settings,
        )
    if simplicity is _PolygonSimplicity.UNDETERMINED:
        evidence = SurfaceFamilyEvidence(True, 1, 0, 0, 1, 0, 0, 0, 0, 0)
        return SegmentCycleRelation(
            PiercingState.UNDETERMINED,
            frozenset(),
            frozenset({SegmentCycleIndeterminacy.NUMERIC_BAND}),
            CycleSurfaceModel.PLANAR_POLYGON,
            tuple(),
            closest,
            evidence,
            settings,
        )

    start, end = _segment_arrays(segment)
    start_height = float(np.dot(normal, start - origin))
    end_height = float(np.dot(normal, end - origin))
    start_absolute = abs(start_height)
    end_absolute = abs(end_height)
    guard = settings.tolerance.predicate_guard_factor
    features = set()
    causes = set()
    points: List[np.ndarray] = []
    state = PiercingState.DOES_NOT_PIERCE

    if start_absolute <= tolerances.length and end_absolute <= tolerances.length:
        projected_segment = _project_to_plane(
            np.asarray((start, end)), origin, normal
        )
        has_contact, contact_undetermined = _projected_segment_polygon_contact(
            projected_segment, polygon, tolerances, settings
        )
        if has_contact:
            features.add(SegmentCycleFeature.COPLANAR_CONTACT)
        elif contact_undetermined:
            state = PiercingState.UNDETERMINED
            causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)
    elif (
        tolerances.length < start_absolute <= guard * tolerances.length
        or tolerances.length < end_absolute <= guard * tolerances.length
    ):
        state = PiercingState.UNDETERMINED
        causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)
    else:
        height_difference = start_height - end_height
        one_endpoint = (
            start_absolute <= tolerances.length
            and end_absolute > guard * tolerances.length
        ) or (
            end_absolute <= tolerances.length
            and start_absolute > guard * tolerances.length
        )
        if one_endpoint:
            point = start if start_absolute <= tolerances.length else end
            location = _locate_projected_point(
                _project_to_plane(point[np.newaxis, :], origin, normal)[0],
                polygon,
                tolerances,
                settings,
            )
            if location in (
                PointCycleLocation.INTERIOR,
                PointCycleLocation.BOUNDARY,
            ):
                features.add(SegmentCycleFeature.SEGMENT_ENDPOINT_CONTACT)
                points.append(point)
            if location is PointCycleLocation.BOUNDARY:
                features.add(_point_boundary_feature(point, cycle, tolerances))
            elif location is PointCycleLocation.UNDETERMINED:
                state = PiercingState.UNDETERMINED
                causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)
        elif abs(height_difference) <= tolerances.length:
            pass
        elif abs(height_difference) <= guard * tolerances.length:
            state = PiercingState.UNDETERMINED
            causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)
        else:
            parameter = start_height / height_difference
            point = start + parameter * (end - start)
            projected = _project_to_plane(
                point[np.newaxis, :], origin, normal
            )[0]
            location = _locate_projected_point(
                projected, polygon, tolerances, settings
            )
            parameter_tolerance = tolerances.parameter
            parameter_inside = (
                guard * parameter_tolerance
                < parameter
                < 1.0 - guard * parameter_tolerance
            )
            parameter_endpoint = (
                abs(parameter) <= parameter_tolerance
                or abs(1.0 - parameter) <= parameter_tolerance
            )
            parameter_outside = (
                parameter < -guard * parameter_tolerance
                or parameter > 1.0 + guard * parameter_tolerance
            )
            if location is PointCycleLocation.INTERIOR:
                if parameter_inside:
                    state = PiercingState.PIERCES
                    features.add(SegmentCycleFeature.TRANSVERSE_INTERIOR)
                    points.append(point)
                elif parameter_endpoint:
                    features.add(SegmentCycleFeature.SEGMENT_ENDPOINT_CONTACT)
                    points.append(point)
                elif parameter_outside:
                    features.add(SegmentCycleFeature.LINE_EXTENSION_INTERIOR)
                else:
                    state = PiercingState.UNDETERMINED
                    causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)
            elif location is PointCycleLocation.BOUNDARY:
                if parameter_inside or parameter_endpoint:
                    features.add(_point_boundary_feature(point, cycle, tolerances))
                    if parameter_endpoint:
                        features.add(SegmentCycleFeature.SEGMENT_ENDPOINT_CONTACT)
                    points.append(point)
                elif not parameter_outside:
                    state = PiercingState.UNDETERMINED
                    causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)
            elif location is PointCycleLocation.UNDETERMINED:
                state = PiercingState.UNDETERMINED
                causes.add(SegmentCycleIndeterminacy.NUMERIC_BAND)

    if state is PiercingState.PIERCES:
        counts = (1, 0, 0)
    elif state is PiercingState.DOES_NOT_PIERCE:
        counts = (0, 1, 0)
    else:
        counts = (0, 0, 1)
    evidence = SurfaceFamilyEvidence(
        True,
        1,
        1,
        0,
        0,
        counts[0],
        counts[1],
        counts[2],
        0,
        0,
    )
    return SegmentCycleRelation(
        state,
        frozenset(features),
        frozenset(causes),
        CycleSurfaceModel.PLANAR_POLYGON,
        _merge_points(points, tolerances.merge),
        closest,
        evidence,
        settings,
    )


def _prepare_nonplanar_surface_family(
    cycle: Cycle,
    settings: GeometrySettings,
) -> _PreparedNonplanarSurfaceFamily:
    surface_settings = settings.surface
    vertex_count = len(cycle)
    if vertex_count > surface_settings.maximum_cycle_vertices:
        return _PreparedNonplanarSurfaceFamily(
            False,
            0,
            tuple(),
            tuple(),
            0,
            0,
            0,
            frozenset({SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY}),
        )

    length_scale = _local_length_scale(cycle.vertices, cycle=cycle)
    if not np.isfinite(length_scale) or length_scale <= settings.tolerance.absolute_length:
        return _PreparedNonplanarSurfaceFamily(
            False,
            0,
            tuple(),
            tuple(),
            0,
            1,
            0,
            frozenset({SegmentCycleIndeterminacy.SURFACE_CONSTRUCTION}),
        )
    tolerances = _predicate_tolerances(length_scale, settings)

    topology = _cycle_topology_template(vertex_count)
    triangulations = topology.triangulations
    expected_count = comb(2 * vertex_count - 4, vertex_count - 2) // (
        vertex_count - 1
    )
    enumeration_complete = (
        len(triangulations) == expected_count
        and len(triangulations) <= surface_settings.maximum_surface_count
    )
    if len(triangulations) > surface_settings.maximum_surface_count:
        triangulations = triangulations[: surface_settings.maximum_surface_count]

    counters = _SurfaceCounters()
    enumerated = 0
    proven_nonembedded = 0
    construction_unknown = 0
    causes = set()
    embedded_surfaces: List[Tuple[Tuple[int, int, int], ...]] = []
    embedded_internal_edges: List[Tuple[_EdgeIndices, ...]] = []

    for surface_index, surface in enumerate(triangulations):
        if counters.triangle_pair_budget_exhausted:
            enumeration_complete = False
            break
        embedding = _determine_surface_embedding(
            surface,
            topology.triangle_pairs[surface_index],
            topology.shared_simplices[surface_index],
            cycle,
            tolerances,
            settings,
            counters,
        )
        enumerated += 1
        if embedding is SurfaceEmbeddingState.PROVEN_NON_EMBEDDED:
            proven_nonembedded += 1
            continue
        if embedding is SurfaceEmbeddingState.CONSTRUCTION_UNDETERMINED:
            construction_unknown += 1
            causes.add(SegmentCycleIndeterminacy.SURFACE_CONSTRUCTION)
            if counters.triangle_pair_budget_exhausted:
                enumeration_complete = False
                break
            continue

        embedded_surfaces.append(surface)
        embedded_internal_edges.append(topology.internal_edges[surface_index])

    if not enumeration_complete:
        causes.add(SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY)

    return _PreparedNonplanarSurfaceFamily(
        enumeration_complete,
        enumerated,
        tuple(embedded_surfaces),
        tuple(embedded_internal_edges),
        proven_nonembedded,
        construction_unknown,
        counters.triangle_pair_tests,
        frozenset(causes),
    )


def _nonplanar_segment_cycle_relation(
    segment: Segment,
    cycle: Cycle,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
    prepared: _PreparedNonplanarSurfaceFamily,
) -> SegmentCycleRelation:
    model = CycleSurfaceModel.VERTEX_TRIANGULATION_FAMILY
    enumeration_complete = prepared.enumeration_complete
    embedded_surfaces = prepared.embedded_surfaces
    embedded = len(embedded_surfaces)
    intersecting = 0
    nonpiercing = 0
    evaluation_unknown = 0
    features = set()
    causes = set(prepared.causes)
    points: List[np.ndarray] = []
    counters = _SurfaceCounters(
        triangle_pair_tests=prepared.triangle_pair_tests_used
    )

    for surface_index, (surface, internal_edges) in enumerate(zip(
        embedded_surfaces, prepared.embedded_internal_edges
    )):
        surface_result = _surface_segment_relation(
            segment,
            surface,
            internal_edges,
            cycle,
            tolerances,
            settings,
            counters,
        )
        features.update(surface_result.features)
        causes.update(surface_result.causes)
        points.extend(_point_array(point) for point in surface_result.points)
        if surface_result.state is SurfaceSegmentState.INTERSECTING:
            intersecting += 1
        elif surface_result.state is SurfaceSegmentState.NON_PIERCING:
            nonpiercing += 1
        else:
            evaluation_unknown += 1
        if (
            SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY
            in surface_result.causes
        ):
            enumeration_complete = False
            causes.add(SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY)
            evaluation_unknown += embedded - surface_index - 1
            break

    if intersecting and nonpiercing:
        causes.add(SegmentCycleIndeterminacy.SURFACE_DISAGREEMENT)

    evidence = SurfaceFamilyEvidence(
        enumeration_complete,
        prepared.enumerated_surface_count,
        embedded,
        prepared.proven_non_embedded_surface_count,
        prepared.construction_undetermined_count,
        intersecting,
        nonpiercing,
        evaluation_unknown,
        counters.segment_triangle_tests,
        counters.triangle_pair_tests,
    )
    if (
        enumeration_complete
        and prepared.construction_undetermined_count == 0
        and evaluation_unknown == 0
        and embedded > 0
        and intersecting == embedded
    ):
        state = PiercingState.PIERCES
    elif (
        enumeration_complete
        and prepared.construction_undetermined_count == 0
        and evaluation_unknown == 0
        and embedded > 0
        and nonpiercing == embedded
    ):
        state = PiercingState.DOES_NOT_PIERCE
    else:
        state = PiercingState.UNDETERMINED
        if not causes:
            causes.add(SegmentCycleIndeterminacy.SURFACE_DISAGREEMENT)

    return SegmentCycleRelation(
        state,
        frozenset(features),
        frozenset(causes),
        model,
        _merge_points(points, tolerances.merge),
        closest_cycle_edge(cycle, segment, settings),
        evidence,
        settings,
    )


def _prepare_cycle_geometry(
    cycle: Cycle,
    settings: GeometrySettings,
) -> _PreparedCycleGeometry:
    """Prepare the coordinate facts shared by a batch of segment queries."""

    coordinates = np.asarray(
        [vertex.coordinates for vertex in cycle.vertices],
        dtype=np.float64,
    )
    bounds = _aabb_bounds(coordinates)
    coordinates.setflags(write=False)
    for bound in bounds:
        bound.setflags(write=False)
    planarity = measure_planarity(cycle, settings)
    planar_simplicity: Optional[_PolygonSimplicity] = None
    if planarity.kind is PlanarityKind.PLANAR:
        length_scale = _local_length_scale(cycle.vertices, cycle=cycle)
        tolerances = _predicate_tolerances(length_scale, settings)
        normal = np.asarray(planarity.normal, dtype=np.float64)
        origin = _point_array(planarity.centroid)
        polygon = _project_to_plane(coordinates, origin, normal)
        planar_simplicity = _projected_polygon_simplicity(
            polygon,
            tolerances,
            settings,
        )
    nonplanar_surface_family = (
        _prepare_nonplanar_surface_family(cycle, settings)
        if planarity.kind is PlanarityKind.NONPLANAR
        else None
    )
    return _PreparedCycleGeometry(
        cycle=cycle,
        coordinates=coordinates,
        bounds=bounds,
        planarity=planarity,
        planar_simplicity=planar_simplicity,
        nonplanar_surface_family=nonplanar_surface_family,
    )


def _prepared_segment_cycle_relation(
    segment: Segment,
    prepared_cycle: _PreparedCycleGeometry,
    tolerances: _PredicateTolerances,
    settings: GeometrySettings,
) -> SegmentCycleRelation:
    """Apply the scalar relation kernel to one prepared cycle."""

    cycle = prepared_cycle.cycle
    planarity = prepared_cycle.planarity
    if planarity.kind is PlanarityKind.PLANAR:
        return _planar_segment_cycle_relation(
            segment,
            cycle,
            planarity,
            cast(_PolygonSimplicity, prepared_cycle.planar_simplicity),
            tolerances,
            settings,
        )
    if planarity.kind is PlanarityKind.NONPLANAR:
        return _nonplanar_segment_cycle_relation(
            segment,
            cycle,
            tolerances,
            settings,
            cast(
                _PreparedNonplanarSurfaceFamily,
                prepared_cycle.nonplanar_surface_family,
            ),
        )
    cause = (
        SegmentCycleIndeterminacy.DEGENERATE_CYCLE
        if planarity.kind is PlanarityKind.DEGENERATE
        else SegmentCycleIndeterminacy.NUMERIC_BAND
    )
    return _undetermined_segment_cycle_relation(
        segment,
        cycle,
        None,
        settings,
        frozenset({cause}),
    )


def _iter_prepared_segment_cycle_relations(
    segments: Iterable[Segment],
    prepared_cycle: _PreparedCycleGeometry,
    settings: GeometrySettings,
) -> Iterator[SegmentCycleRelation]:
    """Classify segments through the scalar kernel using one cycle preparation."""

    cycle = prepared_cycle.cycle
    for segment in segments:
        tolerances, causes = _segment_cycle_base_data(segment, cycle, settings)
        if causes:
            yield _undetermined_segment_cycle_relation(
                segment,
                cycle,
                None,
                settings,
                causes,
            )
            continue
        yield _prepared_segment_cycle_relation(
            segment,
            prepared_cycle,
            cast(_PredicateTolerances, tolerances),
            settings,
        )


def _iter_prepared_segment_cycle_screenings(
    segments: Iterable[Segment],
    prepared_cycle: _PreparedCycleGeometry,
    settings: GeometrySettings,
) -> Iterator[SegmentCycleScreening]:
    """Screen segments through AABB and scalar kernels for one prepared cycle."""

    cycle = prepared_cycle.cycle
    planarity = prepared_cycle.planarity
    nonplanar_surface_family = prepared_cycle.nonplanar_surface_family
    planar_surface_is_valid = (
        planarity.kind is PlanarityKind.PLANAR
        and prepared_cycle.planar_simplicity is _PolygonSimplicity.SIMPLE
    )
    nonplanar_surface_is_valid = (
        planarity.kind is PlanarityKind.NONPLANAR
        and nonplanar_surface_family is not None
        and nonplanar_surface_family.enumeration_complete
        and nonplanar_surface_family.construction_undetermined_count == 0
        and bool(nonplanar_surface_family.embedded_surfaces)
    )

    for segment in segments:
        tolerances, causes = _segment_cycle_base_data(segment, cycle, settings)
        if causes:
            relation = _undetermined_segment_cycle_relation(
                segment,
                cycle,
                None,
                settings,
                causes,
            )
            yield SegmentCycleScreening(
                relation.state,
                relation,
                False,
                relation.surface_evidence.enumeration_complete,
            )
            continue
        tolerances = cast(_PredicateTolerances, tolerances)

        if (
            (planar_surface_is_valid or nonplanar_surface_is_valid)
            and _segment_cycle_aabbs_stably_separated(
                segment,
                prepared_cycle.bounds,
                tolerances.aabb,
            )
        ):
            yield SegmentCycleScreening(
                PiercingState.DOES_NOT_PIERCE,
                None,
                True,
                True,
            )
            continue

        relation = _prepared_segment_cycle_relation(
            segment,
            prepared_cycle,
            tolerances,
            settings,
        )
        yield SegmentCycleScreening(
            relation.state,
            relation,
            False,
            relation.surface_evidence.enumeration_complete,
        )


# Public primitive measurements and classifiers.


def measure_planarity(
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PlanarityMeasurement:
    """Measure the ordered cycle's best-fit-plane residuals."""

    return _fit_plane_svd(cycle, settings)


def determine_line_relation(
    first: Line,
    second: Line,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> LineRelation:
    """Classify the relation between two infinite lines."""

    first_origin = _point_array(first.origin)
    second_origin = _point_array(second.origin)
    first_direction_raw = np.asarray(first.direction, dtype=np.float64)
    second_direction_raw = np.asarray(second.direction, dtype=np.float64)
    if not _all_finite(
        (
            first_origin,
            second_origin,
            first_direction_raw,
            second_direction_raw,
        )
    ):
        return LineRelation(LineRelationKind.UNDETERMINED, None, float("nan"))

    first_direction = _normalized_direction(first.direction)
    second_direction = _normalized_direction(second.direction)
    if first_direction is None or second_direction is None:
        return LineRelation(LineRelationKind.DEGENERATE, None, float("nan"))

    cross = np.cross(first_direction, second_direction)
    parallel_measure = _scale_safe_norm(cross)
    if not np.isfinite(parallel_measure):
        return LineRelation(LineRelationKind.UNDETERMINED, None, float("nan"))
    tolerance = settings.tolerance
    angular_tolerance = max(
        tolerance.parameter,
        tolerance.machine_epsilon_factor * np.finfo(np.float64).eps,
    )
    guard = tolerance.predicate_guard_factor
    separation = second_origin - first_origin
    if parallel_measure == 0.0:
        distance = float(np.linalg.norm(np.cross(separation, first_direction)))
        length_tolerance = tolerance.absolute_length + max(
            tolerance.relative_length,
            tolerance.machine_epsilon_factor * np.finfo(np.float64).eps,
        ) * distance
        if distance <= length_tolerance:
            kind = LineRelationKind.COINCIDENT
        elif distance > guard * length_tolerance:
            kind = LineRelationKind.PARALLEL
        else:
            kind = LineRelationKind.UNDETERMINED
            return LineRelation(kind, None, parallel_measure)
        return LineRelation(kind, distance, parallel_measure)
    if parallel_measure <= guard * angular_tolerance:
        return LineRelation(LineRelationKind.UNDETERMINED, None, parallel_measure)

    distance = abs(float(np.dot(separation, cross))) / parallel_measure
    length_tolerance = tolerance.absolute_length + max(
        tolerance.relative_length,
        tolerance.machine_epsilon_factor * np.finfo(np.float64).eps,
    ) * distance
    if distance <= length_tolerance:
        kind = LineRelationKind.INTERSECTING
    elif distance > guard * length_tolerance:
        kind = LineRelationKind.SKEW
    else:
        return LineRelation(LineRelationKind.UNDETERMINED, None, parallel_measure)
    return LineRelation(kind, distance, parallel_measure)


def line_distance(
    first: Line,
    second: Line,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float:
    """Return the line distance, or NaN when the relation is undefined."""

    relation = determine_line_relation(first, second, settings)
    return float("nan") if relation.distance is None else relation.distance


def point_segment_distance(
    point: Point,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float:
    """Measure the Euclidean distance from a point to a finite segment."""

    point_array = _point_array(point)
    start, end = _segment_arrays(segment)
    if not _all_finite((point_array, start, end)):
        return float("nan")
    length_scale = _local_length_scale(
        (point, segment.start, segment.end), (segment,)
    )
    length_tolerance = settings.tolerance.absolute_length
    if length_scale > 0.0:
        length_tolerance = _predicate_tolerances(length_scale, settings).length
    if float(np.linalg.norm(end - start)) <= length_tolerance:
        return float(np.linalg.norm(point_array - start))
    return _point_segment_distance_arrays(point_array, start, end)


def segment_segment_distance(
    first: Segment,
    second: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float:
    """Measure the minimum Euclidean distance between finite segments."""

    first_start, first_end = _segment_arrays(first)
    second_start, second_end = _segment_arrays(second)
    if not _all_finite(
        (first_start, first_end, second_start, second_end)
    ):
        return float("nan")
    length_scale = _local_length_scale(
        (first.start, first.end, second.start, second.end), (first, second)
    )
    length_tolerance = settings.tolerance.absolute_length
    if length_scale > 0.0:
        length_tolerance = _predicate_tolerances(length_scale, settings).length
    return _segment_segment_distance_arrays(
        first_start,
        first_end,
        second_start,
        second_end,
        length_tolerance * length_tolerance,
    )


def point_pair_distances(
    points: Sequence[Point],
) -> Tuple[PointPairDistance, ...]:
    """Return every unordered point-pair distance in stable index order."""

    results = []
    for first_index in range(len(points)):
        first = _point_array(points[first_index])
        for second_index in range(first_index + 1, len(points)):
            second = _point_array(points[second_index])
            distance = (
                float(np.linalg.norm(first - second))
                if _all_finite((first, second))
                else float("nan")
            )
            results.append(
                PointPairDistance(first_index, second_index, distance)
            )
    return tuple(results)


def find_point_pairs_below_distance(
    points: Sequence[Point],
    threshold: float,
) -> Tuple[PointPairDistance, ...]:
    """Return pairs whose measured distance is strictly below a caller threshold."""

    return tuple(
        measurement
        for measurement in point_pair_distances(points)
        if measurement.distance < threshold
    )


def locate_point_in_planar_cycle(
    point: Point,
    cycle: Cycle,
    plane: Plane,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PointCycleLocation:
    """Locate a point in the projection of a proven planar simple cycle."""

    point_array = _point_array(point)
    origin = _point_array(plane.point)
    normal = np.asarray(plane.normal, dtype=np.float64)
    cycle_coordinates = np.asarray(
        [vertex.coordinates for vertex in cycle.vertices], dtype=np.float64
    )
    if not _all_finite((point_array, origin, normal, cycle_coordinates)):
        return PointCycleLocation.UNDETERMINED
    length_scale = _local_length_scale(
        (point,) + cycle.vertices, cycle=cycle
    )
    if length_scale <= settings.tolerance.absolute_length:
        return PointCycleLocation.UNDETERMINED
    tolerances = _predicate_tolerances(length_scale, settings)
    polygon = _project_to_plane(cycle_coordinates, origin, normal)
    if (
        _projected_polygon_simplicity(polygon, tolerances, settings)
        is not _PolygonSimplicity.SIMPLE
    ):
        return PointCycleLocation.UNDETERMINED
    projected_point = _project_to_plane(
        point_array[np.newaxis, :], origin, normal
    )[0]
    return _locate_projected_point(
        projected_point, polygon, tolerances, settings
    )


# Public composite cycle relations.


def iter_segment_cycle_screenings(
    segments: Iterable[Segment],
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Iterator[SegmentCycleScreening]:
    """Classify finite segments with a strict AABB broad phase.

    The cycle's planar polygon or nonplanar surface family is validated before
    AABB separation may prove ``DOES_NOT_PIERCE``.  Boundary and guard-band
    cases continue through the complete relation kernel.
    """

    yield from _iter_prepared_segment_cycle_screenings(
        segments,
        _prepare_cycle_geometry(cycle, settings),
        settings,
    )


def iter_segment_cycle_relations(
    segments: Iterable[Segment],
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Iterator[SegmentCycleRelation]:
    """Classify segments while preparing the shared cycle surface only once."""

    yield from _iter_prepared_segment_cycle_relations(
        segments,
        _prepare_cycle_geometry(cycle, settings),
        settings,
    )


def determine_segment_cycle_relation(
    segment: Segment,
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> SegmentCycleRelation:
    """Classify finite-segment piercing against an ordered cycle boundary."""

    return next(iter_segment_cycle_relations((segment,), cycle, settings))


def closest_cycle_edge(
    cycle: Cycle,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Optional[ClosestCycleEdge]:
    """Return the nearest boundary edge with deterministic tolerance ties."""

    arrays = tuple(_point_array(point) for point in cycle.vertices) + tuple(
        _segment_arrays(segment)
    )
    if not _all_finite(arrays):
        return None
    length_scale = _local_length_scale(
        cycle.vertices + (segment.start, segment.end), (segment,), cycle
    )
    if (
        not np.isfinite(length_scale)
        or length_scale <= settings.tolerance.absolute_length
    ):
        return None
    length_tolerance = _predicate_tolerances(length_scale, settings).length
    distances = tuple(
        segment_segment_distance(edge, segment, settings)
        for edge in cycle.edges
    )
    finite_distances = tuple(
        distance for distance in distances if np.isfinite(distance)
    )
    if not finite_distances:
        return None
    minimum = min(finite_distances)
    edge_index = min(
        index
        for index, distance in enumerate(distances)
        if np.isfinite(distance) and distance <= minimum + length_tolerance
    )
    return ClosestCycleEdge(edge_index, cycle.edges[edge_index], distances[edge_index])
