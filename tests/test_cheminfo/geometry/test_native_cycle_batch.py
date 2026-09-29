"""Behavioral fence for packed native segment--cycle batches."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hotpot.cheminfo.geometry import _geometry_native, native
from hotpot.cheminfo.geometry.object import Cycle, Segment
from hotpot.cheminfo.geometry.settings import DEFAULT_GEOMETRY_SETTINGS


SQUARE = Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
SHIFTED_SQUARE = Cycle(
    ((10, 0, 0), (12, 0, 0), (12, 2, 0), (10, 2, 0))
)
WARPED_SQUARE = Cycle(
    ((0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0))
)


def _pack_cycles(*cycles: Cycle):
    coordinates: list[tuple[float, float, float]] = []
    indices: list[int] = []
    offsets = [0]
    for cycle in cycles:
        start = len(coordinates)
        coordinates.extend(point.coordinates for point in cycle)
        indices.extend(range(start, len(coordinates)))
        offsets.append(len(indices))
    return native._prepare_cycles(coordinates, indices, offsets)


def _relation_signature(relation) -> tuple:
    evidence = relation.surface_evidence
    closest = relation.closest_boundary_edge
    return (
        relation.state.name,
        tuple(item.name for item in relation.features),
        tuple(item.name for item in relation.indeterminacy_causes),
        None if relation.surface_model is None else relation.surface_model.name,
        tuple(tuple(point) for point in relation.intersection_points),
        (
            None
            if closest is None
            else (closest.edge_index, closest.distance)
        ),
        (
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
    )


def test_packed_batch_matches_individual_planar_and_nonplanar_dispatch() -> None:
    cycles = (SQUARE, WARPED_SQUARE)
    prepared = _pack_cycles(*cycles)
    segments = (
        Segment((1, 1, -1), (1, 1, 1)),
        Segment((5, 5, -1), (5, 5, 1)),
        Segment((0.5, 0.5, 0), (1.5, 0.5, 0)),
        Segment((1, 1, 0), (1, 1, 0)),
    )
    pairs = tuple(
        (segment_index, cycle_index)
        for cycle_index in range(len(cycles))
        for segment_index in range(len(segments))
    )

    dense = native._determine_segment_cycle_relations(
        prepared,
        segments,
        pairs,
    )
    expected = tuple(
        native._determine_segment_cycle_relation(
            segments[segment_index],
            cycles[cycle_index],
        )
        for segment_index, cycle_index in pairs
    )
    assert tuple(map(_relation_signature, dense)) == tuple(
        map(_relation_signature, expected)
    )

    screened = native._screen_segments(
        prepared,
        segments,
        pairs,
        native.DetailLevel.FULL,
    )
    expected_screenings = tuple(
        native._segment_cycle_screenings(
            (segments[segment_index],),
            cycles[cycle_index],
        )[0]
        for segment_index, cycle_index in pairs
    )
    assert [state.name for state in screened.states] == [
        result.state.name for result in expected_screenings
    ]
    assert screened.aabb_separated.tolist() == [
        result.aabb_separated for result in expected_screenings
    ]
    expected_positions = [
        position
        for position, result in enumerate(expected_screenings)
        if result.relation is not None
    ]
    assert screened.relation_positions == expected_positions
    assert tuple(map(_relation_signature, screened.relations)) == tuple(
        _relation_signature(expected_screenings[position].relation)
        for position in expected_positions
    )


def test_detail_levels_counters_sparse_relations_and_early_stop() -> None:
    prepared = _pack_cycles(SQUARE, SHIFTED_SQUARE)
    segments = (
        Segment((100, 100, -1), (100, 100, 1)),
        Segment((1, 1, -1), (1, 1, 1)),
        Segment((11, 1, -1), (11, 1, 1)),
        Segment((0.5, 0.5, 0), (1.5, 0.5, 0)),
        Segment((1, 1, 0), (1, 1, 0)),
    )
    pairs = ((0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0))

    state_only = native._screen_segments(
        prepared,
        segments,
        pairs,
        native.DetailLevel.STATE_ONLY,
    )
    assert state_only.requested_pair_count == 7
    assert state_only.evaluated_pair_count == 7
    assert state_only.aabb_separated_pair_count == 3
    assert state_only.exact_pair_count == 4
    assert state_only.piercing_pair_count == 2
    assert state_only.does_not_pierce_pair_count == 4
    assert state_only.undetermined_pair_count == 1
    assert state_only.scan_complete is True
    assert state_only.relation_positions == []
    assert state_only.relations == []

    actionable = native._screen_segments(
        prepared,
        segments,
        pairs,
        native.DetailLevel.ACTIONABLE,
    )
    assert actionable.relation_positions == [1, 3, 6]
    assert [relation.state.name for relation in actionable.relations] == [
        "PIERCES",
        "PIERCES",
        "UNDETERMINED",
    ]

    full = native._screen_segments(
        prepared,
        segments,
        pairs,
        native.DetailLevel.FULL,
    )
    assert full.relation_positions == [1, 3, 5, 6]
    assert len(full.relations) == full.exact_pair_count

    stopped = native._screen_segments(
        prepared,
        segments,
        pairs,
        native.DetailLevel.STATE_ONLY,
        stop_after_confirmed=True,
    )
    assert stopped.requested_pair_count == 7
    assert stopped.evaluated_pair_count == 2
    assert stopped.scan_complete is False
    assert [state.name for state in stopped.states] == [
        "DOES_NOT_PIERCE",
        "PIERCES",
    ]

    stopped_at_end = native._screen_segments(
        prepared,
        segments,
        ((0, 0), (3, 0), (1, 0)),
        native.DetailLevel.STATE_ONLY,
        stop_after_confirmed=True,
    )
    assert stopped_at_end.evaluated_pair_count == 3
    assert stopped_at_end.scan_complete is True


def test_prepared_cycle_batch_is_copied_read_only_and_caches_bounds() -> None:
    coordinates = np.array(
        [point.coordinates for point in SQUARE],
        dtype=np.float64,
    )
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    offsets = np.array([0, 4], dtype=np.int64)
    prepared = native._prepare_cycles(coordinates, indices, offsets)
    coordinates[:] = 100.0
    indices[:] = 0
    offsets[:] = 0

    assert prepared.coordinate_count == 4
    assert prepared.cycle_count == 1
    assert prepared.cycle_indices == [0, 1, 2, 3]
    assert prepared.cycle_offsets == [0, 4]
    np.testing.assert_allclose(
        prepared.cycle_bounds,
        np.array([[[0, 0, 0], [2, 2, 0]]], dtype=np.float64),
    )
    with pytest.raises(AttributeError):
        prepared.cycle_count = 3
    with pytest.raises(AttributeError):
        prepared.cycle(0).coordinates = []


def test_nonplanar_segment_budget_resets_for_each_candidate_pair() -> None:
    settings = replace(
        DEFAULT_GEOMETRY_SETTINGS,
        surface=replace(
            DEFAULT_GEOMETRY_SETTINGS.surface,
            maximum_segment_triangle_tests=1,
        ),
    )
    prepared = native._prepare_cycles(
        [point.coordinates for point in WARPED_SQUARE],
        (0, 1, 2, 3),
        (0, 4),
        settings,
    )
    segment = Segment((1, 1, -1), (1, 1, 1))
    relations = native._determine_segment_cycle_relations(
        prepared,
        (segment,),
        ((0, 0), (0, 0)),
    )

    assert _relation_signature(relations[0]) == _relation_signature(relations[1])
    assert relations[0].surface_evidence.segment_triangle_tests_used == 1


@pytest.mark.parametrize(
    ("indices", "offsets"),
    (
        ([0, 1, 2], []),
        ([0, 1, 2], [1, 3]),
        ([0, 1, 2], [0, 2]),
        ([0, 1, 2], [0, 3, 2]),
        ([0, 1, 4], [0, 3]),
    ),
)
def test_packed_cycle_csr_validation(indices, offsets) -> None:
    coordinates = [point.coordinates for point in SQUARE]
    with pytest.raises(ValueError):
        native._prepare_cycles(coordinates, indices, offsets)


def test_candidate_pairs_are_validated_before_scanning() -> None:
    prepared = _pack_cycles(SQUARE)
    segments = (Segment((1, 1, -1), (1, 1, 1)),)
    with pytest.raises(ValueError):
        native._screen_segments(
            prepared,
            segments,
            ((0, 0), (1, 0)),
            native.DetailLevel.FULL,
        )
    with pytest.raises(ValueError):
        native._determine_segment_cycle_relations(
            prepared,
            segments,
            ((0, 0), (0, 1)),
        )
    with pytest.raises(ValueError):
        _geometry_native.screen_segments(
            prepared,
            np.zeros((1, 2, 3), dtype=np.float64),
            np.array([[0, -1]], dtype=np.int64),
            _geometry_native.DetailLevel.STATE_ONLY,
        )
