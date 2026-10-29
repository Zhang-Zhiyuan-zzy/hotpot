"""Factual geometry relations exposed through the canonical native backend.

This module owns the stable Python relation vocabulary and immutable result
records. Numerical predicates live exclusively in the C++ geometry backend;
the functions below only adapt native results to these public contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Iterable, Iterator, Optional, Sequence, Tuple

from . import native as _native
from .object import Cycle, Line, Plane, Point, Segment
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


# Public primitive measurements and classifiers.


def measure_planarity(
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PlanarityMeasurement:
    """Measure the ordered cycle's best-fit-plane residuals."""

    return _native.measure_planarity(cycle, settings)


def determine_line_relation(
    first: Line,
    second: Line,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> LineRelation:
    """Classify the relation between two infinite lines."""

    return _native.determine_line_relation(first, second, settings)


def line_distance(
    first: Line,
    second: Line,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float:
    """Return the line distance, or NaN when the relation is undefined."""

    return _native.line_distance(first, second, settings)


def point_segment_distance(
    point: Point,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float:
    """Measure the Euclidean distance from a point to a finite segment."""

    return _native.point_segment_distance(point, segment, settings)


def segment_segment_distance(
    first: Segment,
    second: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> float:
    """Measure the minimum Euclidean distance between finite segments."""

    return _native.segment_segment_distance(first, second, settings)


def point_pair_distances(
    points: Sequence[Point],
) -> Tuple[PointPairDistance, ...]:
    """Return every unordered point-pair distance in stable index order."""

    return _native.point_pair_distances(points)


def find_point_pairs_below_distance(
    points: Sequence[Point],
    threshold: float,
) -> Tuple[PointPairDistance, ...]:
    """Return pairs whose measured distance is below an explicit threshold."""

    return _native.find_point_pairs_below_distance(points, threshold)


def locate_point_in_planar_cycle(
    point: Point,
    cycle: Cycle,
    plane: Plane,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PointCycleLocation:
    """Locate a point in the projection of a proven planar simple cycle."""

    return _native.locate_point_in_planar_cycle(point, cycle, plane, settings)


# Public composite cycle relations.


def iter_segment_cycle_screenings(
    segments: Iterable[Segment],
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Iterator[SegmentCycleScreening]:
    """Classify finite segments with a strict AABB broad phase."""

    yield from _native._iter_segment_cycle_screenings(segments, cycle, settings)


def iter_segment_cycle_relations(
    segments: Iterable[Segment],
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Iterator[SegmentCycleRelation]:
    """Classify segments while preparing the shared cycle surface once."""

    yield from _native._iter_segment_cycle_relations(segments, cycle, settings)


def determine_segment_cycle_relation(
    segment: Segment,
    cycle: Cycle,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> SegmentCycleRelation:
    """Classify finite-segment piercing against an ordered cycle boundary."""

    return _native._determine_segment_cycle_relation(segment, cycle, settings)


def closest_cycle_edge(
    cycle: Cycle,
    segment: Segment,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Optional[ClosestCycleEdge]:
    """Return the nearest boundary edge with deterministic tolerance ties."""

    return _native.closest_cycle_edge(cycle, segment, settings)
