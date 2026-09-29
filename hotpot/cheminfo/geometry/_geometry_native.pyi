from typing import FrozenSet, List, Optional, Tuple

import numpy as np


class LineRelationKind:
    INTERSECTING: "LineRelationKind"
    PARALLEL: "LineRelationKind"
    COINCIDENT: "LineRelationKind"
    SKEW: "LineRelationKind"
    DEGENERATE: "LineRelationKind"
    UNDETERMINED: "LineRelationKind"


class PlanarityKind:
    PLANAR: "PlanarityKind"
    NONPLANAR: "PlanarityKind"
    DEGENERATE: "PlanarityKind"
    UNDETERMINED: "PlanarityKind"


class PolygonSimplicity:
    SIMPLE: "PolygonSimplicity"
    SELF_INTERSECTING: "PolygonSimplicity"
    UNDETERMINED: "PolygonSimplicity"


class PointCycleLocation:
    INTERIOR: "PointCycleLocation"
    BOUNDARY: "PointCycleLocation"
    EXTERIOR: "PointCycleLocation"
    UNDETERMINED: "PointCycleLocation"


class SurfaceEmbeddingState:
    EMBEDDED: "SurfaceEmbeddingState"
    PROVEN_NON_EMBEDDED: "SurfaceEmbeddingState"
    CONSTRUCTION_UNDETERMINED: "SurfaceEmbeddingState"


class NonplanarSurfaceCause:
    INCOMPLETE_SURFACE_FAMILY: "NonplanarSurfaceCause"
    SURFACE_CONSTRUCTION: "NonplanarSurfaceCause"


class PiercingState:
    PIERCES: "PiercingState"
    DOES_NOT_PIERCE: "PiercingState"
    UNDETERMINED: "PiercingState"


class SegmentCycleFeature:
    TRANSVERSE_INTERIOR: "SegmentCycleFeature"
    LINE_EXTENSION_INTERIOR: "SegmentCycleFeature"
    CYCLE_EDGE_CONTACT: "SegmentCycleFeature"
    CYCLE_VERTEX_CONTACT: "SegmentCycleFeature"
    SEGMENT_ENDPOINT_CONTACT: "SegmentCycleFeature"
    COPLANAR_CONTACT: "SegmentCycleFeature"


class SegmentCycleIndeterminacy:
    NONFINITE_INPUT: "SegmentCycleIndeterminacy"
    NUMERIC_BAND: "SegmentCycleIndeterminacy"
    TOLERANCE_DOMAIN: "SegmentCycleIndeterminacy"
    DEGENERATE_CYCLE: "SegmentCycleIndeterminacy"
    DEGENERATE_SEGMENT: "SegmentCycleIndeterminacy"
    DEGENERATE_TRIANGLE: "SegmentCycleIndeterminacy"
    SELF_INTERSECTION: "SegmentCycleIndeterminacy"
    SURFACE_DISAGREEMENT: "SegmentCycleIndeterminacy"
    INCOMPLETE_SURFACE_FAMILY: "SegmentCycleIndeterminacy"
    SURFACE_CONSTRUCTION: "SegmentCycleIndeterminacy"


class CycleSurfaceModel:
    PLANAR_POLYGON: "CycleSurfaceModel"
    VERTEX_TRIANGULATION_FAMILY: "CycleSurfaceModel"


class NumericTolerances:
    absolute_length: float
    relative_length: float
    parameter: float
    machine_epsilon_factor: float
    predicate_guard_factor: float
    planarity_factor: float
    winding_residual: float
    intersection_merge_factor: float
    aabb_padding_factor: float

    def __init__(
        self,
        absolute_length: float,
        relative_length: float,
        parameter: float,
        machine_epsilon_factor: float,
        predicate_guard_factor: float,
        planarity_factor: float,
        winding_residual: float,
        intersection_merge_factor: float,
        aabb_padding_factor: float,
    ) -> None: ...


class LineRelation:
    kind: LineRelationKind
    distance: Optional[float]
    parallel_measure: float


class PointSegmentMeasurement:
    distance: float
    closest_point: Tuple[float, float, float]
    parameter: float
    segment_degenerate: bool


class SegmentSegmentMeasurement:
    distance: float
    first_closest_point: Tuple[float, float, float]
    second_closest_point: Tuple[float, float, float]
    first_parameter: float
    second_parameter: float
    first_segment_degenerate: bool
    second_segment_degenerate: bool


class PlanarityMeasurement:
    kind: PlanarityKind
    centroid: Tuple[float, float, float]
    normal: Optional[Tuple[float, float, float]]
    singular_values: Tuple[float, float, float]
    maximum_deviation: float
    rms_deviation: float
    length_scale: float
    length_tolerance: float


class PreparedPlanarCycle:
    coordinates: List[Tuple[float, float, float]]
    planarity: PlanarityMeasurement
    tolerances: NumericTolerances
    projection: List[Tuple[float, float]]
    simplicity: PolygonSimplicity
    has_planar_surface: bool
    has_simple_planar_surface: bool


class SurfaceEnumerationLimits:
    maximum_cycle_vertices: int
    maximum_surface_count: int
    maximum_segment_triangle_tests: int
    maximum_triangle_pair_tests: int

    def __init__(
        self,
        maximum_cycle_vertices: int,
        maximum_surface_count: int,
        maximum_segment_triangle_tests: int,
        maximum_triangle_pair_tests: int,
    ) -> None: ...


class PreparedNonplanarSurfaceFamily:
    coordinates: List[Tuple[float, float, float]]
    tolerances: NumericTolerances
    limits: SurfaceEnumerationLimits
    enumeration_complete: bool
    enumerated_surface_count: int
    embedded_surface_count: int
    proven_non_embedded_surface_count: int
    construction_undetermined_count: int
    triangle_pair_tests_used: int
    causes: FrozenSet[NonplanarSurfaceCause]
    surface_states: List[SurfaceEmbeddingState]


class ClosestCycleEdge:
    edge_index: int
    distance: float


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


class SegmentCycleRelation:
    state: PiercingState
    features: List[SegmentCycleFeature]
    indeterminacy_causes: List[SegmentCycleIndeterminacy]
    surface_model: Optional[CycleSurfaceModel]
    intersection_points: List[Tuple[float, float, float]]
    closest_boundary_edge: Optional[ClosestCycleEdge]
    surface_evidence: SurfaceFamilyEvidence


class SegmentCycleScreening:
    state: PiercingState
    relation: Optional[SegmentCycleRelation]
    aabb_separated: bool
    surface_complete: bool


def measure_planarity(
    cycle: np.ndarray,
    tolerances: NumericTolerances,
) -> PlanarityMeasurement: ...


def prepare_planar_cycle(
    cycle: np.ndarray,
    tolerances: NumericTolerances,
) -> PreparedPlanarCycle: ...


def prepare_nonplanar_surface_family(
    cycle: np.ndarray,
    tolerances: NumericTolerances,
    limits: SurfaceEnumerationLimits,
) -> PreparedNonplanarSurfaceFamily: ...


def locate_point_in_planar_cycle(
    point: np.ndarray,
    cycle: PreparedPlanarCycle,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> PointCycleLocation: ...


def closest_cycle_edge(
    cycle: PreparedPlanarCycle,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> Optional[ClosestCycleEdge]: ...


def determine_planar_segment_cycle_relation(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    cycle: PreparedPlanarCycle,
) -> SegmentCycleRelation: ...


def planar_segment_cycle_relations(
    segments: np.ndarray,
    cycle: PreparedPlanarCycle,
) -> List[SegmentCycleRelation]: ...


def planar_segment_cycle_screenings(
    segments: np.ndarray,
    cycle: PreparedPlanarCycle,
) -> List[SegmentCycleScreening]: ...


def determine_line_relation(
    first_origin: np.ndarray,
    first_direction: np.ndarray,
    second_origin: np.ndarray,
    second_direction: np.ndarray,
    tolerances: NumericTolerances,
) -> LineRelation: ...


def line_distance(
    first_origin: np.ndarray,
    first_direction: np.ndarray,
    second_origin: np.ndarray,
    second_direction: np.ndarray,
    tolerances: NumericTolerances,
) -> float: ...


def point_segment_measurement(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    tolerances: NumericTolerances,
) -> PointSegmentMeasurement: ...


def point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    tolerances: NumericTolerances,
) -> float: ...


def segment_segment_measurement(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerances: NumericTolerances,
) -> SegmentSegmentMeasurement: ...


def segment_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerances: NumericTolerances,
) -> float: ...


def point_pair_distances(
    coordinates: np.ndarray,
    pair_indices: Optional[np.ndarray] = ...,
) -> Tuple[np.ndarray, np.ndarray]: ...


def find_point_pairs_below_distance(
    coordinates: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]: ...


def aabb_bounds(coordinates: np.ndarray) -> np.ndarray: ...


def aabb_separation_mask(
    first_bounds: np.ndarray,
    second_bounds: np.ndarray,
    paddings: np.ndarray,
) -> np.ndarray: ...


def segment_aabb_separation_mask(
    segments: np.ndarray,
    target_bounds: np.ndarray,
    paddings: np.ndarray,
) -> np.ndarray: ...


def aabb_candidate_pairs(
    first_bounds: np.ndarray,
    second_bounds: np.ndarray,
    padding: float,
) -> np.ndarray: ...
