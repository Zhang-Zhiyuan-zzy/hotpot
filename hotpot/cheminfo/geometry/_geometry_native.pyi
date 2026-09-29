from typing import Optional, Tuple

import numpy as np


class LineRelationKind:
    INTERSECTING: "LineRelationKind"
    PARALLEL: "LineRelationKind"
    COINCIDENT: "LineRelationKind"
    SKEW: "LineRelationKind"
    DEGENERATE: "LineRelationKind"
    UNDETERMINED: "LineRelationKind"


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
