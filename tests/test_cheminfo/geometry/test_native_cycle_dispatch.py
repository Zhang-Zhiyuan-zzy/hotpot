"""Differential fence for the canonical native prepared-cycle dispatcher."""

from __future__ import annotations

from dataclasses import replace
import math

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


SQUARE = Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
WARPED_SQUARE = Cycle(
    ((0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0))
)
COLLINEAR = Cycle(((0, 0, 0), (1, 0, 0), (2, 0, 0)))
NONFINITE = Cycle(
    ((0, 0, 0), (2, 0, 0), (math.nan, 2, 0), (0, 2, 0))
)
SELF_INTERSECTING = Cycle(
    ((0, 0, 0), (2, 2, 0), (0, 2, 0), (2, 0, 0))
)
SEGMENTS = (
    Segment((1, 1, -1), (1, 1, 1)),
    Segment((0, 0, 0), (-1, -1, -1)),
    Segment((0.2, 0, 0), (1.5, 0, 0)),
    Segment((10, 10, 4), (11, 10, 4)),
    Segment((0, 0, 0), (0, 0, 0)),
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
            equal_nan=True,
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
    ("cycle", "kind", "uses_nonplanar"),
    (
        (SQUARE, PlanarityKind.PLANAR, False),
        (SELF_INTERSECTING, PlanarityKind.PLANAR, False),
        (WARPED_SQUARE, PlanarityKind.NONPLANAR, True),
        (COLLINEAR, PlanarityKind.DEGENERATE, False),
        (NONFINITE, PlanarityKind.UNDETERMINED, False),
    ),
)
@pytest.mark.parametrize(
    "settings",
    (
        DEFAULT_GEOMETRY_SETTINGS,
        _surface_settings(maximum_segment_triangle_tests=1),
        _surface_settings(maximum_surface_count=1),
    ),
)
def test_general_native_dispatch_matches_all_legacy_fields(
    cycle,
    kind,
    uses_nonplanar,
    settings,
):
    prepared = native._prepare_cycle(cycle, settings)
    assert prepared.planarity.kind.name == kind.name
    assert prepared.uses_nonplanar_surface_family is uses_nonplanar
    np.testing.assert_allclose(
        prepared.coordinates,
        [point.coordinates for point in cycle.vertices],
        equal_nan=True,
    )

    expected_relations = tuple(
        iter_segment_cycle_relations(SEGMENTS, cycle, settings)
    )
    actual_relations = native._segment_cycle_relations(
        SEGMENTS,
        cycle,
        settings,
    )
    for expected, actual in zip(expected_relations, actual_relations):
        _assert_relation_equal(expected, actual)

    expected_screenings = tuple(
        iter_segment_cycle_screenings(SEGMENTS, cycle, settings)
    )
    actual_screenings = native._segment_cycle_screenings(
        SEGMENTS,
        cycle,
        settings,
    )
    for expected, actual in zip(expected_screenings, actual_screenings):
        _assert_screening_equal(expected, actual)

    for segment in SEGMENTS:
        _assert_relation_equal(
            determine_segment_cycle_relation(segment, cycle, settings),
            native._determine_segment_cycle_relation(
                segment,
                cycle,
                settings,
            ),
        )


def test_general_dispatch_randomized_planar_and_nonplanar_batches():
    random = np.random.default_rng(20260929)
    checked = 0
    for vertex_count in range(3, 8):
        for planar in (True, False):
            for _ in range(2):
                coordinates = random.normal(size=(vertex_count, 3))
                if planar:
                    coordinates[:, 2] = 0.0
                cycle = Cycle(coordinates)
                if measure_planarity(cycle).kind not in (
                    PlanarityKind.PLANAR,
                    PlanarityKind.NONPLANAR,
                ):
                    continue
                segments = tuple(
                    Segment(*random.normal(size=(2, 3)))
                    for _ in range(4)
                )
                expected = tuple(iter_segment_cycle_relations(segments, cycle))
                actual = native._segment_cycle_relations(segments, cycle)
                for expected_relation, actual_relation in zip(expected, actual):
                    _assert_relation_equal(expected_relation, actual_relation)
                expected_screenings = tuple(
                    iter_segment_cycle_screenings(segments, cycle)
                )
                actual_screenings = native._segment_cycle_screenings(
                    segments,
                    cycle,
                )
                for expected_screening, actual_screening in zip(
                    expected_screenings,
                    actual_screenings,
                ):
                    _assert_screening_equal(
                        expected_screening,
                        actual_screening,
                    )
                checked += len(segments)
    assert checked >= 64


def test_general_prepared_cycle_is_factory_only_and_settings_owned():
    with pytest.raises(TypeError):
        _geometry_native.PreparedCycle()

    prepared = native._prepare_cycle(WARPED_SQUARE)
    with pytest.raises(AttributeError):
        prepared.coordinates = []
    with pytest.raises(AttributeError):
        prepared.planarity = None

    segment = SEGMENTS[0]
    start = np.require(
        segment.start.coordinates,
        dtype=np.float64,
        requirements=("C", "A"),
    )
    end = np.require(
        segment.end.coordinates,
        dtype=np.float64,
        requirements=("C", "A"),
    )
    with pytest.raises(TypeError):
        _geometry_native.determine_segment_cycle_relation(
            start,
            end,
            prepared,
            native._native_tolerances(DEFAULT_GEOMETRY_SETTINGS),
        )


def test_general_dispatch_adapters_remain_internal():
    assert "PreparedCycle" not in native.__all__
    assert "_prepare_cycle" not in native.__all__
    assert "_determine_segment_cycle_relation" not in native.__all__
    assert "_segment_cycle_relations" not in native.__all__
    assert "_segment_cycle_screenings" not in native.__all__
