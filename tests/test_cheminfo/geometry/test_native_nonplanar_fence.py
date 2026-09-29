"""Public behavior fence for native nonplanar cycle relations.

These tests characterize the current Python backend without reaching into its
private triangulation or predicate helpers.  Public evidence counters and
ordered intersection points expose the traversal contracts that the native
backend must preserve.
"""

from dataclasses import replace
from math import cos, pi, sin

import numpy as np
import pytest

from hotpot.cheminfo.geometry.object import Cycle, Segment
from hotpot.cheminfo.geometry.relation import (
    CycleSurfaceModel,
    PiercingState,
    SegmentCycleFeature,
    SegmentCycleIndeterminacy,
    SurfaceFamilyEvidence,
    determine_segment_cycle_relation,
    iter_segment_cycle_relations,
    iter_segment_cycle_screenings,
)
from hotpot.cheminfo.geometry.settings import DEFAULT_GEOMETRY_SETTINGS


WARPED_SQUARE = Cycle(
    ((0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0))
)
PIERCING_SEGMENT = Segment((0.6, 0.8, -1), (0.6, 0.8, 1))


def _surface_settings(**changes):
    return replace(
        DEFAULT_GEOMETRY_SETTINGS,
        surface=replace(DEFAULT_GEOMETRY_SETTINGS.surface, **changes),
    )


@pytest.mark.parametrize(
    ("vertex_count", "catalan_count"),
    ((4, 2), (5, 5), (6, 14), (7, 42), (8, 132)),
)
def test_public_evidence_exposes_complete_catalan_surface_counts(
    vertex_count,
    catalan_count,
):
    cycle = Cycle(
        (
            (
                2.0 * cos(index * 2.0 * pi / vertex_count),
                2.0 * sin(index * 2.0 * pi / vertex_count),
                0.2 if index % 2 else -0.15,
            )
            for index in range(vertex_count)
        )
    )

    relation = determine_segment_cycle_relation(
        Segment((20, 20, -5), (20, 20, 5)), cycle
    )

    assert relation.surface_model is CycleSurfaceModel.VERTEX_TRIANGULATION_FAMILY
    assert relation.surface_evidence.enumeration_complete
    assert relation.surface_evidence.enumerated_surface_count == catalan_count


def test_quad_surface_order_and_exact_consensus_counters_are_stable():
    relation = determine_segment_cycle_relation(PIERCING_SEGMENT, WARPED_SQUARE)

    assert relation.state is PiercingState.PIERCES
    assert relation.surface_model is CycleSurfaceModel.VERTEX_TRIANGULATION_FAMILY
    assert relation.features == frozenset(
        {SegmentCycleFeature.TRANSVERSE_INTERIOR}
    )
    assert relation.indeterminacy_causes == frozenset()
    assert tuple(point.coordinates for point in relation.intersection_points) == (
        pytest.approx((0.6, 0.8, 0.0)),
        pytest.approx((0.6, 0.8, 0.12)),
    )
    assert relation.surface_evidence == SurfaceFamilyEvidence(
        enumeration_complete=True,
        enumerated_surface_count=2,
        embedded_surface_count=2,
        proven_non_embedded_surface_count=0,
        construction_undetermined_count=0,
        intersecting_surface_count=2,
        non_piercing_surface_count=0,
        evaluation_undetermined_count=0,
        segment_triangle_tests_used=4,
        triangle_pair_tests_used=2,
    )


@pytest.mark.parametrize(
    (
        "segment",
        "state",
        "features",
        "causes",
        "points",
        "intersecting_count",
        "non_piercing_count",
        "undetermined_count",
    ),
    (
        (
            Segment((0.8, 0.8, -1), (0.8, 0.8, 1)),
            PiercingState.UNDETERMINED,
            frozenset({SegmentCycleFeature.TRANSVERSE_INTERIOR}),
            frozenset({SegmentCycleIndeterminacy.NUMERIC_BAND}),
            ((0.8, 0.8, 0.0),),
            1,
            0,
            1,
        ),
        (
            Segment((0, 0, 0), (-1, -1, -1)),
            PiercingState.DOES_NOT_PIERCE,
            frozenset(
                {
                    SegmentCycleFeature.CYCLE_VERTEX_CONTACT,
                    SegmentCycleFeature.SEGMENT_ENDPOINT_CONTACT,
                }
            ),
            frozenset(),
            ((0.0, 0.0, 0.0),),
            0,
            2,
            0,
        ),
        (
            Segment((0.6, 0.8, 1), (0.6, 0.8, 2)),
            PiercingState.DOES_NOT_PIERCE,
            frozenset({SegmentCycleFeature.LINE_EXTENSION_INTERIOR}),
            frozenset(),
            (),
            0,
            2,
            0,
        ),
        (
            Segment((0, 1, -1), (0, 1, 1)),
            PiercingState.DOES_NOT_PIERCE,
            frozenset({SegmentCycleFeature.CYCLE_EDGE_CONTACT}),
            frozenset(),
            ((0.0, 1.0, 0.0),),
            0,
            2,
            0,
        ),
        (
            Segment((0.2, 0, 0), (1.5, 0, 0)),
            PiercingState.UNDETERMINED,
            frozenset({SegmentCycleFeature.COPLANAR_CONTACT}),
            frozenset({SegmentCycleIndeterminacy.NUMERIC_BAND}),
            (),
            0,
            0,
            2,
        ),
    ),
)
def test_nonplanar_features_causes_and_surface_outcomes_are_exact(
    segment,
    state,
    features,
    causes,
    points,
    intersecting_count,
    non_piercing_count,
    undetermined_count,
):
    relation = determine_segment_cycle_relation(segment, WARPED_SQUARE)

    assert relation.state is state
    assert relation.features == features
    assert relation.indeterminacy_causes == causes
    assert len(relation.intersection_points) == len(points)
    for actual, expected in zip(relation.intersection_points, points):
        assert actual.coordinates == pytest.approx(expected)
    evidence = relation.surface_evidence
    assert evidence.enumeration_complete
    assert evidence.enumerated_surface_count == 2
    assert evidence.embedded_surface_count == 2
    assert evidence.proven_non_embedded_surface_count == 0
    assert evidence.construction_undetermined_count == 0
    assert evidence.intersecting_surface_count == intersecting_count
    assert evidence.non_piercing_surface_count == non_piercing_count
    assert evidence.evaluation_undetermined_count == undetermined_count
    assert evidence.segment_triangle_tests_used == 4
    assert evidence.triangle_pair_tests_used == 2


def test_embedded_surfaces_must_agree_before_a_definite_state_is_returned():
    cycle = Cycle(
        ((0, 0, 0), (2, 2, 0), (0, 2, 1), (2, 0, 0))
    )

    relation = determine_segment_cycle_relation(
        Segment((0.5, 0.5, -2), (0.5, 0.5, 2)), cycle
    )

    assert relation.state is PiercingState.UNDETERMINED
    assert relation.features == frozenset(
        {
            SegmentCycleFeature.TRANSVERSE_INTERIOR,
            SegmentCycleFeature.CYCLE_EDGE_CONTACT,
        }
    )
    assert relation.indeterminacy_causes == frozenset(
        {SegmentCycleIndeterminacy.SURFACE_DISAGREEMENT}
    )
    assert tuple(point.coordinates for point in relation.intersection_points) == (
        pytest.approx((0.5, 0.5, 0.0)),
        pytest.approx((0.5, 0.5, 0.25)),
    )
    assert relation.surface_evidence == SurfaceFamilyEvidence(
        enumeration_complete=True,
        enumerated_surface_count=2,
        embedded_surface_count=2,
        proven_non_embedded_surface_count=0,
        construction_undetermined_count=0,
        intersecting_surface_count=1,
        non_piercing_surface_count=1,
        evaluation_undetermined_count=0,
        segment_triangle_tests_used=4,
        triangle_pair_tests_used=2,
    )


@pytest.mark.parametrize(
    ("cycle", "segment", "state", "causes", "evidence"),
    (
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
            PiercingState.DOES_NOT_PIERCE,
            frozenset(),
            SurfaceFamilyEvidence(True, 5, 3, 2, 0, 0, 3, 0, 9, 13),
        ),
        (
            Cycle(
                ((0, 0, 0), (2, 0, 0), (2, 2, 1), (2, 0, 0), (0, 2, 0))
            ),
            Segment((0.5, 0.5, -2), (0.5, 0.5, 2)),
            PiercingState.UNDETERMINED,
            frozenset({SegmentCycleIndeterminacy.SURFACE_CONSTRUCTION}),
            SurfaceFamilyEvidence(True, 5, 0, 3, 2, 0, 0, 0, 0, 3),
        ),
    ),
)
def test_surface_embedding_categories_and_counters_are_preserved(
    cycle,
    segment,
    state,
    causes,
    evidence,
):
    relation = determine_segment_cycle_relation(segment, cycle)

    assert relation.state is state
    assert relation.indeterminacy_causes == causes
    assert relation.surface_evidence == evidence
    assert evidence.enumerated_surface_count == (
        evidence.embedded_surface_count
        + evidence.proven_non_embedded_surface_count
        + evidence.construction_undetermined_count
    )


@pytest.mark.parametrize(
    ("setting_change", "features", "causes", "evidence"),
    (
        (
            {"maximum_triangle_pair_tests": 1},
            frozenset({SegmentCycleFeature.TRANSVERSE_INTERIOR}),
            frozenset(
                {
                    SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY,
                    SegmentCycleIndeterminacy.SURFACE_CONSTRUCTION,
                }
            ),
            SurfaceFamilyEvidence(False, 2, 1, 0, 1, 1, 0, 0, 2, 1),
        ),
        (
            {"maximum_segment_triangle_tests": 1},
            frozenset(),
            frozenset(
                {SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY}
            ),
            SurfaceFamilyEvidence(False, 2, 2, 0, 0, 0, 0, 2, 1, 2),
        ),
        (
            {"maximum_surface_count": 1},
            frozenset({SegmentCycleFeature.TRANSVERSE_INTERIOR}),
            frozenset(
                {SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY}
            ),
            SurfaceFamilyEvidence(False, 1, 1, 0, 0, 1, 0, 0, 2, 1),
        ),
        (
            {"maximum_cycle_vertices": 3},
            frozenset(),
            frozenset(
                {SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY}
            ),
            SurfaceFamilyEvidence(False, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ),
    ),
)
def test_each_surface_budget_fails_closed_with_exact_accounting(
    setting_change,
    features,
    causes,
    evidence,
):
    relation = determine_segment_cycle_relation(
        PIERCING_SEGMENT,
        WARPED_SQUARE,
        _surface_settings(**setting_change),
    )

    assert relation.state is PiercingState.UNDETERMINED
    assert relation.features == features
    assert relation.indeterminacy_causes == causes
    assert relation.surface_evidence == evidence


def test_segment_triangle_budget_resets_for_each_public_batch_item():
    settings = _surface_settings(maximum_segment_triangle_tests=1)
    relations = tuple(
        iter_segment_cycle_relations(
            (
                PIERCING_SEGMENT,
                Segment((0.7, 0.9, -1), (0.7, 0.9, 1)),
            ),
            WARPED_SQUARE,
            settings,
        )
    )

    segment_test_counts = [
        item.surface_evidence.segment_triangle_tests_used for item in relations
    ]
    assert segment_test_counts == [
        1,
        1,
    ]
    assert [item.surface_evidence.triangle_pair_tests_used for item in relations] == [
        2,
        2,
    ]
    assert all(item.state is PiercingState.UNDETERMINED for item in relations)


def test_aabb_screening_requires_a_complete_usable_surface_family():
    far_segment = Segment((10, 10, 4), (11, 10, 4))
    complete = next(
        iter_segment_cycle_screenings((far_segment,), WARPED_SQUARE)
    )
    incomplete = next(
        iter_segment_cycle_screenings(
            (far_segment,),
            WARPED_SQUARE,
            _surface_settings(maximum_surface_count=1),
        )
    )
    construction_cycle = Cycle(
        ((0, 0, 0), (2, 0, 0), (2, 2, 1), (2, 0, 0), (0, 2, 0))
    )
    construction = next(
        iter_segment_cycle_screenings((far_segment,), construction_cycle)
    )

    assert complete.state is PiercingState.DOES_NOT_PIERCE
    assert complete.aabb_separated
    assert complete.surface_complete
    assert complete.relation is None

    assert incomplete.state is PiercingState.UNDETERMINED
    assert not incomplete.aabb_separated
    assert not incomplete.surface_complete
    assert incomplete.relation is not None
    assert SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY in (
        incomplete.relation.indeterminacy_causes
    )

    assert construction.state is PiercingState.UNDETERMINED
    assert not construction.aabb_separated
    assert construction.surface_complete
    assert construction.relation is not None
    assert SegmentCycleIndeterminacy.SURFACE_CONSTRUCTION in (
        construction.relation.indeterminacy_causes
    )


def test_nonplanar_relation_is_rigid_motion_and_cycle_shift_invariant():
    cycle_coordinates = np.asarray(
        tuple(point.coordinates for point in WARPED_SQUARE.vertices)
    )
    segment_coordinates = np.asarray(
        (PIERCING_SEGMENT.start.coordinates, PIERCING_SEGMENT.end.coordinates)
    )
    transform = np.asarray(((0, -1, 0), (-1, 0, 0), (0, 0, -1)), dtype=float)
    translation = np.asarray((7.0, -3.0, 2.0))

    baseline = determine_segment_cycle_relation(PIERCING_SEGMENT, WARPED_SQUARE)
    transformed_cycle = cycle_coordinates @ transform.T + translation
    transformed_segment = segment_coordinates @ transform.T + translation
    transformed = determine_segment_cycle_relation(
        Segment(*transformed_segment),
        Cycle(np.roll(transformed_cycle, 2, axis=0)),
    )

    assert transformed.state is baseline.state
    assert transformed.features == baseline.features
    assert transformed.indeterminacy_causes == baseline.indeterminacy_causes
    assert transformed.surface_evidence == baseline.surface_evidence
    expected_points = tuple(
        np.asarray(point.coordinates) @ transform.T + translation
        for point in baseline.intersection_points
    )
    for actual, expected in zip(transformed.intersection_points, expected_points):
        assert actual.coordinates == pytest.approx(expected)


@pytest.mark.parametrize("scale", (1.0e-4, 1.0e4))
def test_nonplanar_relation_is_scale_invariant_when_length_tolerance_scales(scale):
    cycle = Cycle(
        tuple(
            tuple(scale * coordinate for coordinate in point.coordinates)
            for point in WARPED_SQUARE.vertices
        )
    )
    segment = Segment(
        tuple(scale * coordinate for coordinate in PIERCING_SEGMENT.start.coordinates),
        tuple(scale * coordinate for coordinate in PIERCING_SEGMENT.end.coordinates),
    )
    settings = replace(
        DEFAULT_GEOMETRY_SETTINGS,
        tolerance=replace(
            DEFAULT_GEOMETRY_SETTINGS.tolerance,
            absolute_length=(
                DEFAULT_GEOMETRY_SETTINGS.tolerance.absolute_length * scale
            ),
        ),
    )

    baseline = determine_segment_cycle_relation(PIERCING_SEGMENT, WARPED_SQUARE)
    scaled = determine_segment_cycle_relation(segment, cycle, settings)

    assert scaled.state is baseline.state
    assert scaled.features == baseline.features
    assert scaled.indeterminacy_causes == baseline.indeterminacy_causes
    assert scaled.surface_evidence == baseline.surface_evidence
    for actual, expected in zip(
        scaled.intersection_points, baseline.intersection_points
    ):
        assert actual.coordinates == pytest.approx(
            tuple(scale * coordinate for coordinate in expected.coordinates)
        )
