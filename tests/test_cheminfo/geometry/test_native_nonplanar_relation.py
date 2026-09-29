"""Differential fence for native nonplanar segment--cycle relations."""

import math
from dataclasses import replace

import numpy as np
import pytest

from hotpot.cheminfo.geometry import _geometry_native, native
from hotpot.cheminfo.geometry.object import Cycle, Segment
from hotpot.cheminfo.geometry.relation import (
    PlanarityKind,
    SegmentCycleRelation,
    SegmentCycleScreening,
    determine_segment_cycle_relation,
    iter_segment_cycle_relations,
    iter_segment_cycle_screenings,
    measure_planarity,
)
from hotpot.cheminfo.geometry.settings import (
    DEFAULT_GEOMETRY_SETTINGS,
    GeometrySettings,
)


WARPED_SQUARE = Cycle(
    ((0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0))
)
CURATED_SEGMENTS = (
    Segment((0.6, 0.8, -1), (0.6, 0.8, 1)),
    Segment((0.8, 0.8, -1), (0.8, 0.8, 1)),
    Segment((0, 0, 0), (-1, -1, -1)),
    Segment((0.6, 0.8, 1), (0.6, 0.8, 2)),
    Segment((0, 1, -1), (0, 1, 1)),
    Segment((0.2, 0, 0), (1.5, 0, 0)),
    Segment((10, 10, 4), (11, 10, 4)),
    Segment((math.nan, 0, 0), (0, 0, 1)),
)


def _surface_settings(**changes) -> GeometrySettings:
    return replace(
        DEFAULT_GEOMETRY_SETTINGS,
        surface=replace(DEFAULT_GEOMETRY_SETTINGS.surface, **changes),
    )


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
    assert len(actual.intersection_points) == len(expected.intersection_points)
    for expected_point, actual_point in zip(
        expected.intersection_points,
        actual.intersection_points,
    ):
        np.testing.assert_allclose(
            actual_point.coordinates,
            expected_point.coordinates,
            rtol=5.0e-11,
            atol=5.0e-11,
        )
    assert (actual.closest_boundary_edge is None) is (
        expected.closest_boundary_edge is None
    )
    if expected.closest_boundary_edge is not None:
        assert actual.closest_boundary_edge is not None
        assert (
            actual.closest_boundary_edge.edge_index
            == expected.closest_boundary_edge.edge_index
        )
        assert actual.closest_boundary_edge.edge == expected.closest_boundary_edge.edge
        assert actual.closest_boundary_edge.distance == pytest.approx(
            expected.closest_boundary_edge.distance,
            rel=5.0e-11,
            abs=5.0e-11,
        )


def _assert_screening_equal(
    expected: SegmentCycleScreening,
    actual: SegmentCycleScreening,
) -> None:
    assert actual.state is expected.state
    assert actual.aabb_separated is expected.aabb_separated
    assert actual.surface_complete is expected.surface_complete
    assert (actual.relation is None) is (expected.relation is None)
    if expected.relation is not None:
        assert actual.relation is not None
        _assert_relation_equal(expected.relation, actual.relation)


@pytest.mark.parametrize(
    "settings",
    (
        DEFAULT_GEOMETRY_SETTINGS,
        _surface_settings(maximum_segment_triangle_tests=1),
        _surface_settings(maximum_triangle_pair_tests=1),
        _surface_settings(maximum_surface_count=1),
        _surface_settings(maximum_cycle_vertices=3),
    ),
)
def test_native_nonplanar_scalar_relations_match_all_python_fields(settings):
    for segment in CURATED_SEGMENTS:
        _assert_relation_equal(
            determine_segment_cycle_relation(segment, WARPED_SQUARE, settings),
            native._determine_nonplanar_segment_cycle_relation(
                segment,
                WARPED_SQUARE,
                settings,
            ),
        )


def test_native_nonplanar_family_categories_match_all_python_fields():
    cases = (
        (
            Cycle(
                (
                    (0.6, -1.2, 1.2),
                    (0.3, -0.9, 1.9),
                    (-2.5, -0.2, 1.2),
                    (0.5, -0.4, -0.6),
                    (-2.5, 1.1, -0.5),
                )
            ),
            Segment((0, 0, -3), (0, 0, 3)),
        ),
        (
            Cycle(
                ((0, 0, 0), (2, 0, 0), (2, 2, 1), (2, 0, 0), (0, 2, 0))
            ),
            Segment((0.5, 0.5, -2), (0.5, 0.5, 2)),
        ),
        (
            Cycle(((0, 0, 0), (2, 2, 0), (0, 2, 1), (2, 0, 0))),
            Segment((0.5, 0.5, -2), (0.5, 0.5, 2)),
        ),
    )
    for cycle, segment in cases:
        _assert_relation_equal(
            determine_segment_cycle_relation(segment, cycle),
            native._determine_nonplanar_segment_cycle_relation(segment, cycle),
        )


def test_native_nonplanar_batch_relations_and_screenings_match_python():
    expected_relations = tuple(
        iter_segment_cycle_relations(CURATED_SEGMENTS, WARPED_SQUARE)
    )
    actual_relations = native._nonplanar_segment_cycle_relations(
        CURATED_SEGMENTS,
        WARPED_SQUARE,
    )
    assert len(actual_relations) == len(expected_relations)
    for expected, actual in zip(expected_relations, actual_relations):
        _assert_relation_equal(expected, actual)

    expected_screenings = tuple(
        iter_segment_cycle_screenings(CURATED_SEGMENTS, WARPED_SQUARE)
    )
    actual_screenings = native._nonplanar_segment_cycle_screenings(
        CURATED_SEGMENTS,
        WARPED_SQUARE,
    )
    assert len(actual_screenings) == len(expected_screenings)
    for expected, actual in zip(expected_screenings, actual_screenings):
        _assert_screening_equal(expected, actual)


def test_native_nonplanar_batch_budget_resets_for_each_segment():
    settings = _surface_settings(maximum_segment_triangle_tests=1)
    expected = tuple(
        iter_segment_cycle_relations(
            CURATED_SEGMENTS[:3],
            WARPED_SQUARE,
            settings,
        )
    )
    actual = native._nonplanar_segment_cycle_relations(
        CURATED_SEGMENTS[:3],
        WARPED_SQUARE,
        settings,
    )
    for expected_relation, actual_relation in zip(expected, actual):
        _assert_relation_equal(expected_relation, actual_relation)
        assert actual_relation.surface_evidence.segment_triangle_tests_used == 1


def test_triangle_boundary_on_line_extension_is_not_finite_segment_contact():
    segment = Segment((0, 1, 1), (0, 1, 2))

    expected = determine_segment_cycle_relation(segment, WARPED_SQUARE)
    actual = native._determine_nonplanar_segment_cycle_relation(
        segment,
        WARPED_SQUARE,
    )

    _assert_relation_equal(expected, actual)
    assert expected.state.name == "DOES_NOT_PIERCE"
    assert expected.features == frozenset()
    assert expected.intersection_points == ()


def test_raw_nonplanar_result_order_and_owned_settings_are_stable():
    family = native._prepare_nonplanar_surface_family(
        WARPED_SQUARE,
        DEFAULT_GEOMETRY_SETTINGS,
    )
    endpoint_segment = CURATED_SEGMENTS[2]
    start = np.require(
        endpoint_segment.start.coordinates,
        dtype=np.float64,
        requirements=("C", "A"),
    )
    end = np.require(
        endpoint_segment.end.coordinates,
        dtype=np.float64,
        requirements=("C", "A"),
    )
    result = _geometry_native.determine_nonplanar_segment_cycle_relation(
        start,
        end,
        family,
    )

    assert [feature.name for feature in result.features] == [
        "SEGMENT_ENDPOINT_CONTACT",
        "CYCLE_VERTEX_CONTACT",
    ]
    assert result.indeterminacy_causes == []
    assert result.intersection_points == [(0.0, 0.0, 0.0)]
    with pytest.raises(TypeError):
        _geometry_native.determine_nonplanar_segment_cycle_relation(
            start,
            end,
            family,
            native._native_tolerances(DEFAULT_GEOMETRY_SETTINGS),
        )


def test_native_nonplanar_randomized_relations_match_all_python_fields():
    random = np.random.default_rng(20260929)
    checked = 0
    for vertex_count in range(4, 8):
        for _ in range(3):
            cycle = Cycle(random.normal(size=(vertex_count, 3)))
            if measure_planarity(cycle).kind is not PlanarityKind.NONPLANAR:
                continue
            segments = tuple(
                Segment(*random.normal(size=(2, 3)))
                for _ in range(5)
            )
            expected = tuple(iter_segment_cycle_relations(segments, cycle))
            actual = native._nonplanar_segment_cycle_relations(segments, cycle)
            for expected_relation, actual_relation in zip(expected, actual):
                _assert_relation_equal(expected_relation, actual_relation)
            expected_screenings = tuple(
                iter_segment_cycle_screenings(segments, cycle)
            )
            actual_screenings = native._nonplanar_segment_cycle_screenings(
                segments,
                cycle,
            )
            for expected_screening, actual_screening in zip(
                expected_screenings,
                actual_screenings,
            ):
                _assert_screening_equal(expected_screening, actual_screening)
            checked += len(segments)
    assert checked == 60


def test_phase3b_adapters_remain_internal():
    assert "_determine_nonplanar_segment_cycle_relation" not in native.__all__
    assert "_nonplanar_segment_cycle_relations" not in native.__all__
    assert "_nonplanar_segment_cycle_screenings" not in native.__all__
