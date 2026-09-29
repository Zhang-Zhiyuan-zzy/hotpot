"""Typed Python adapters for the canonical C++ geometry kernels.

This module is the direct Python entry to the native geometry API during the
staged migration. It intentionally does not provide a Python numerical
fallback. The stable package facade is switched to these adapters only after
each migration phase passes its characterization fence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from . import _geometry_native as _native
except ImportError as exc:
    raise ImportError(
        "the hotpot.cheminfo.geometry native extension is unavailable; "
        "install a compatible Hotpot wheel or rebuild Hotpot from source"
    ) from exc

from .object import Cycle, Line, Plane, Point, Segment
from .settings import DEFAULT_GEOMETRY_SETTINGS, GeometrySettings

if TYPE_CHECKING:
    from .relation import (
        ClosestCycleEdge,
        LineRelation,
        PlanarityMeasurement,
        PointCycleLocation,
        PointPairDistance,
        SegmentCycleRelation,
        SegmentCycleScreening,
    )


__all__ = [
    "NumericTolerances",
    "PointSegmentMeasurement",
    "SegmentSegmentMeasurement",
    "measure_planarity",
    "locate_point_in_planar_cycle",
    "closest_cycle_edge",
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
PreparedPlanarCycle = _native.PreparedPlanarCycle
SurfaceEnumerationLimits = _native.SurfaceEnumerationLimits
PreparedNonplanarSurfaceFamily = _native.PreparedNonplanarSurfaceFamily

Coordinates = Union[Sequence[float], Point]
BoundsInput = Union[np.ndarray, Sequence[Sequence[float]]]
ArrayValues = Union[
    np.ndarray,
    Sequence[float],
    Sequence[Sequence[float]],
    Sequence[Sequence[Sequence[float]]],
]


def _aligned_array(values: ArrayValues, shape: Tuple[int, ...]) -> np.ndarray:
    return np.require(
        values,
        dtype=np.float64,
        requirements=("C", "A"),
    ).reshape(shape)


def _point_array(point: Coordinates) -> np.ndarray:
    coordinates = point.coordinates if isinstance(point, Point) else point
    return _aligned_array(coordinates, (3,))


def _point_matrix(points: Iterable[Coordinates]) -> np.ndarray:
    return _aligned_array(
        [
            point.coordinates if isinstance(point, Point) else point
            for point in points
        ],
        (-1, 3),
    )


def _segment_batch(segments: Iterable[Segment]) -> np.ndarray:
    return _aligned_array(
        [
            (segment.start.coordinates, segment.end.coordinates)
            for segment in segments
        ],
        (-1, 2, 3),
    )


def _cycle_matrix(cycle: Cycle) -> np.ndarray:
    return _point_matrix(cycle.vertices)


def _bounds_batch(bounds: BoundsInput) -> np.ndarray:
    return _aligned_array(bounds, (-1, 2, 3))


def _paddings(
    padding: Union[float, Sequence[float], np.ndarray],
    count: int,
) -> np.ndarray:
    values = np.asarray(padding, dtype=np.float64)
    if values.ndim == 0:
        return np.full(count, float(values), dtype=np.float64)
    return _aligned_array(values, (-1,))


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


def _surface_enumeration_limits(
    settings: GeometrySettings,
) -> SurfaceEnumerationLimits:
    surface = settings.surface
    return SurfaceEnumerationLimits(
        surface.maximum_cycle_vertices,
        surface.maximum_surface_count,
        surface.maximum_segment_triangle_tests,
        surface.maximum_triangle_pair_tests,
    )


def _planarity_result(
    result: _native.PlanarityMeasurement,
) -> PlanarityMeasurement:
    from .relation import PlanarityKind, PlanarityMeasurement

    normal = None if result.normal is None else tuple(result.normal)
    return PlanarityMeasurement(
        PlanarityKind[result.kind.name],
        Point(result.centroid),
        normal,
        tuple(result.singular_values),
        result.maximum_deviation,
        result.rms_deviation,
        result.length_scale,
        result.length_tolerance,
    )


def measure_planarity(
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PlanarityMeasurement:
    """Measure cycle planarity through the canonical C++ kernel."""

    return _planarity_result(
        _native.measure_planarity(
            _cycle_matrix(cycle),
            _native_tolerances(settings),
        )
    )


def prepare_planar_cycle(
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PreparedPlanarCycle:
    """Prepare reusable planar-cycle facts for native segment batches."""

    return _native.prepare_planar_cycle(
        _cycle_matrix(cycle),
        _native_tolerances(settings),
    )


def _prepare_nonplanar_surface_family(
    cycle: Cycle,
    settings: GeometrySettings,
) -> PreparedNonplanarSurfaceFamily:
    """Prepare internal nonplanar surface evidence through the C++ kernel."""

    return _native.prepare_nonplanar_surface_family(
        _cycle_matrix(cycle),
        _native_tolerances(settings),
        _surface_enumeration_limits(settings),
    )


def locate_point_in_planar_cycle(
    point: Point,
    cycle: Cycle,
    plane: Plane,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PointCycleLocation:
    """Locate a point in a planar cycle through the canonical C++ kernel."""

    from .relation import PointCycleLocation

    prepared_cycle = prepare_planar_cycle(cycle, settings)
    result = _native.locate_point_in_planar_cycle(
        _point_array(point),
        prepared_cycle,
        _point_array(plane.point),
        _point_array(plane.normal),
    )
    return PointCycleLocation[result.name]


def _closest_cycle_edge_result(
    result: Optional[_native.ClosestCycleEdge],
    cycle: Cycle,
) -> Optional[ClosestCycleEdge]:
    if result is None:
        return None
    from .relation import ClosestCycleEdge

    edge_index = int(result.edge_index)
    return ClosestCycleEdge(
        edge_index,
        cycle.edges[edge_index],
        float(result.distance),
    )


def closest_cycle_edge(
    cycle: Cycle,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Optional[ClosestCycleEdge]:
    """Return nearest cycle-boundary evidence from the C++ kernel."""

    return _closest_cycle_edge_result(
        _native.closest_cycle_edge(
            prepare_planar_cycle(cycle, settings),
            _point_array(segment.start),
            _point_array(segment.end),
        ),
        cycle,
    )


def _segment_cycle_relation_result(
    result: _native.SegmentCycleRelation,
    cycle: Cycle,
    settings: GeometrySettings,
) -> SegmentCycleRelation:
    from .relation import (
        CycleSurfaceModel,
        PiercingState,
        SegmentCycleFeature,
        SegmentCycleIndeterminacy,
        SegmentCycleRelation,
        SurfaceFamilyEvidence,
    )

    evidence = result.surface_evidence
    return SegmentCycleRelation(
        PiercingState[result.state.name],
        frozenset(SegmentCycleFeature[item.name] for item in result.features),
        frozenset(
            SegmentCycleIndeterminacy[item.name]
            for item in result.indeterminacy_causes
        ),
        (
            None
            if result.surface_model is None
            else CycleSurfaceModel[result.surface_model.name]
        ),
        tuple(Point(point) for point in result.intersection_points),
        _closest_cycle_edge_result(result.closest_boundary_edge, cycle),
        SurfaceFamilyEvidence(
            evidence.enumeration_complete,
            evidence.enumerated_surface_count,
            evidence.embedded_surface_count,
            evidence.proven_non_embedded_surface_count,
            evidence.construction_undetermined_count,
            evidence.intersecting_surface_count,
            evidence.non_piercing_surface_count,
            evidence.evaluation_undetermined_count,
            evidence.segment_triangle_tests_used,
            evidence.triangle_pair_tests_used,
        ),
        settings,
    )


def determine_planar_segment_cycle_relation(
    segment: Segment,
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> SegmentCycleRelation:
    """Classify one segment against a planar cycle using native evidence."""

    prepared_cycle = prepare_planar_cycle(cycle, settings)
    result = _native.determine_planar_segment_cycle_relation(
        _point_array(segment.start),
        _point_array(segment.end),
        prepared_cycle,
    )
    return _segment_cycle_relation_result(result, cycle, settings)


def planar_segment_cycle_relations(
    segments: Iterable[Segment],
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Tuple[SegmentCycleRelation, ...]:
    """Classify a segment batch while preparing the planar cycle once."""

    segment_tuple = tuple(segments)
    results = _native.planar_segment_cycle_relations(
        _segment_batch(segment_tuple),
        prepare_planar_cycle(cycle, settings),
    )
    return tuple(
        _segment_cycle_relation_result(result, cycle, settings)
        for result in results
    )


def planar_segment_cycle_screenings(
    segments: Iterable[Segment],
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Tuple[SegmentCycleScreening, ...]:
    """Screen a planar segment batch with strict native AABB exclusion."""

    from .relation import PiercingState, SegmentCycleScreening

    segment_tuple = tuple(segments)
    results = _native.planar_segment_cycle_screenings(
        _segment_batch(segment_tuple),
        prepare_planar_cycle(cycle, settings),
    )
    return tuple(
        SegmentCycleScreening(
            PiercingState[result.state.name],
            (
                None
                if result.relation is None
                else _segment_cycle_relation_result(
                    result.relation,
                    cycle,
                    settings,
                )
            ),
            result.aabb_separated,
            result.surface_complete,
        )
        for result in results
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
    target = _aligned_array(target_bounds, (2, 3))
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
