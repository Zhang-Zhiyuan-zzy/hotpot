from typing import FrozenSet, List, Optional, Tuple, Union

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


class DetailLevel:
    STATE_ONLY: "DetailLevel"
    ACTIONABLE: "DetailLevel"
    FULL: "DetailLevel"


class SegmentCyclePair:
    segment_index: int
    cycle_index: int

    def __init__(self, segment_index: int, cycle_index: int) -> None: ...


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
    @property
    def coordinates(self) -> List[Tuple[float, float, float]]: ...
    @property
    def planarity(self) -> PlanarityMeasurement: ...
    @property
    def tolerances(self) -> NumericTolerances: ...
    @property
    def projection(self) -> List[Tuple[float, float]]: ...
    @property
    def simplicity(self) -> PolygonSimplicity: ...
    @property
    def has_planar_surface(self) -> bool: ...
    @property
    def has_simple_planar_surface(self) -> bool: ...


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
    @property
    def coordinates(self) -> List[Tuple[float, float, float]]: ...
    @property
    def tolerances(self) -> NumericTolerances: ...
    @property
    def limits(self) -> SurfaceEnumerationLimits: ...
    @property
    def enumeration_complete(self) -> bool: ...
    @property
    def enumerated_surface_count(self) -> int: ...
    @property
    def embedded_surface_count(self) -> int: ...
    @property
    def proven_non_embedded_surface_count(self) -> int: ...
    @property
    def construction_undetermined_count(self) -> int: ...
    @property
    def triangle_pair_tests_used(self) -> int: ...
    @property
    def causes(self) -> FrozenSet[NonplanarSurfaceCause]: ...
    @property
    def surface_states(self) -> List[SurfaceEmbeddingState]: ...


class PreparedCycle:
    @property
    def coordinates(self) -> List[Tuple[float, float, float]]: ...
    @property
    def bounds(self) -> np.ndarray: ...
    @property
    def planarity(self) -> PlanarityMeasurement: ...
    @property
    def tolerances(self) -> NumericTolerances: ...
    @property
    def limits(self) -> SurfaceEnumerationLimits: ...
    @property
    def uses_nonplanar_surface_family(self) -> bool: ...


class PreparedCycleBatch:
    @property
    def coordinate_count(self) -> int: ...
    @property
    def cycle_count(self) -> int: ...
    @property
    def coordinates(self) -> List[Tuple[float, float, float]]: ...
    @property
    def cycle_indices(self) -> List[int]: ...
    @property
    def cycle_offsets(self) -> List[int]: ...
    @property
    def cycle_bounds(self) -> np.ndarray: ...
    def cycle(self, index: int) -> PreparedCycle: ...


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


class SegmentCycleBatch:
    @property
    def detail(self) -> DetailLevel: ...
    @property
    def requested_pair_count(self) -> int: ...
    @property
    def evaluated_pair_count(self) -> int: ...
    @property
    def aabb_separated_pair_count(self) -> int: ...
    @property
    def exact_pair_count(self) -> int: ...
    @property
    def piercing_pair_count(self) -> int: ...
    @property
    def does_not_pierce_pair_count(self) -> int: ...
    @property
    def undetermined_pair_count(self) -> int: ...
    @property
    def scan_complete(self) -> bool: ...
    @property
    def states(self) -> List[PiercingState]: ...
    @property
    def aabb_separated(self) -> np.ndarray: ...
    @property
    def surface_complete(self) -> np.ndarray: ...
    @property
    def relation_positions(self) -> List[int]: ...
    @property
    def relations(self) -> List[SegmentCycleRelation]: ...


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


def prepare_cycle(
    cycle: np.ndarray,
    tolerances: NumericTolerances,
    limits: SurfaceEnumerationLimits,
) -> PreparedCycle: ...


def prepare_cycles(
    coordinates: np.ndarray,
    cycle_indices: np.ndarray,
    cycle_offsets: np.ndarray,
    tolerances: NumericTolerances,
    limits: SurfaceEnumerationLimits,
) -> PreparedCycleBatch: ...


def determine_segment_cycle_relations(
    cycles: PreparedCycleBatch,
    segments: np.ndarray,
    candidate_pairs: np.ndarray,
) -> List[SegmentCycleRelation]: ...


def screen_segments(
    cycles: PreparedCycleBatch,
    segments: np.ndarray,
    candidate_pairs: np.ndarray,
    detail: DetailLevel,
    stop_after_confirmed: bool = False,
) -> SegmentCycleBatch: ...


def locate_point_in_planar_cycle(
    point: np.ndarray,
    cycle: PreparedPlanarCycle,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> PointCycleLocation: ...


def closest_cycle_edge(
    cycle: Union[PreparedPlanarCycle, PreparedCycle],
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


def determine_nonplanar_segment_cycle_relation(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    family: PreparedNonplanarSurfaceFamily,
) -> SegmentCycleRelation: ...


def nonplanar_segment_cycle_relations(
    segments: np.ndarray,
    family: PreparedNonplanarSurfaceFamily,
) -> List[SegmentCycleRelation]: ...


def nonplanar_segment_cycle_screenings(
    segments: np.ndarray,
    family: PreparedNonplanarSurfaceFamily,
) -> List[SegmentCycleScreening]: ...


def determine_segment_cycle_relation(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    cycle: PreparedCycle,
) -> SegmentCycleRelation: ...


def segment_cycle_relations(
    segments: np.ndarray,
    cycle: PreparedCycle,
) -> List[SegmentCycleRelation]: ...


def segment_cycle_screenings(
    segments: np.ndarray,
    cycle: PreparedCycle,
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
