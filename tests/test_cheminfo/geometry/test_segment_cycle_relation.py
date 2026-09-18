import math
from dataclasses import replace

import pytest

from hotpot.cheminfo.geometry.object import Cycle, Segment
from hotpot.cheminfo.geometry.relation import (
    CycleSurfaceModel,
    PiercingState,
    SegmentCycleFeature,
    SegmentCycleIndeterminacy,
    closest_cycle_edge,
    determine_segment_cycle_relation,
    iter_segment_cycle_relations,
)
from hotpot.cheminfo.geometry.settings import (
    DEFAULT_GEOMETRY_SETTINGS,
    GeometrySettings,
    SurfaceEnumerationSettings,
)


@pytest.fixture
def square():
    return Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])


def test_planar_strict_interior_crossing_is_a_piercing(square):
    relation = determine_segment_cycle_relation(
        Segment((1, 1, -1), (1, 1, 1)), square
    )

    assert relation.state is PiercingState.PIERCES
    assert relation.surface_model is CycleSurfaceModel.PLANAR_POLYGON
    assert relation.features == frozenset(
        {SegmentCycleFeature.TRANSVERSE_INTERIOR}
    )
    assert len(relation.intersection_points) == 1
    assert relation.intersection_points[0].coordinates == pytest.approx((1, 1, 0))
    assert relation.surface_evidence.enumeration_complete
    assert relation.surface_evidence.intersecting_surface_count == 1
    assert relation.surface_evidence.segment_triangle_tests_used == 0
    assert relation.surface_evidence.triangle_pair_tests_used == 0


def test_planar_extension_hit_does_not_count_as_finite_segment_piercing(square):
    relation = determine_segment_cycle_relation(
        Segment((1, 1, 1), (1, 1, 2)), square
    )

    assert relation.state is PiercingState.DOES_NOT_PIERCE
    assert SegmentCycleFeature.LINE_EXTENSION_INTERIOR in relation.features
    assert SegmentCycleFeature.TRANSVERSE_INTERIOR not in relation.features


def test_planar_edge_and_endpoint_contacts_are_not_strict_piercings(square):
    edge_contact = determine_segment_cycle_relation(
        Segment((0, 1, -1), (0, 1, 1)), square
    )
    endpoint_contact = determine_segment_cycle_relation(
        Segment((1, 1, 0), (1, 1, 1)), square
    )

    assert edge_contact.state is PiercingState.DOES_NOT_PIERCE
    assert SegmentCycleFeature.CYCLE_EDGE_CONTACT in edge_contact.features
    assert endpoint_contact.state is PiercingState.DOES_NOT_PIERCE
    assert SegmentCycleFeature.SEGMENT_ENDPOINT_CONTACT in endpoint_contact.features


def test_planar_coplanar_segment_is_contact_not_piercing(square):
    relation = determine_segment_cycle_relation(
        Segment((0.5, 1, 0), (1.5, 1, 0)), square
    )

    assert relation.state is PiercingState.DOES_NOT_PIERCE
    assert SegmentCycleFeature.COPLANAR_CONTACT in relation.features


def test_planar_contacts_are_not_reported_outside_the_cycle(square):
    endpoint = determine_segment_cycle_relation(
        Segment((10, 10, 0), (10, 10, 1)), square
    )
    coplanar = determine_segment_cycle_relation(
        Segment((10, 10, 0), (11, 10, 0)), square
    )

    assert endpoint.state is PiercingState.DOES_NOT_PIERCE
    assert endpoint.features == frozenset()
    assert coplanar.state is PiercingState.DOES_NOT_PIERCE
    assert coplanar.features == frozenset()


def test_coplanar_segment_crossing_cycle_has_contact(square):
    relation = determine_segment_cycle_relation(
        Segment((-1, 1, 0), (3, 1, 0)), square
    )

    assert SegmentCycleFeature.COPLANAR_CONTACT in relation.features


def test_self_intersecting_planar_cycle_is_explicitly_undetermined():
    cycle = Cycle([(0, 0, 0), (2, 2, 0), (0, 2, 0), (2, 0, 0)])
    relation = determine_segment_cycle_relation(
        Segment((0.5, 1, -1), (0.5, 1, 1)), cycle
    )

    assert relation.state is PiercingState.UNDETERMINED
    assert SegmentCycleIndeterminacy.SELF_INTERSECTION in relation.indeterminacy_causes
    assert relation.surface_evidence.proven_non_embedded_surface_count == 1


def test_cycle_construction_evidence_does_not_depend_on_segment_length():
    cycle = Cycle([(0, 0, 0), (2, 2, 0), (0, 2, 0), (2, 0, 0)])

    relation = determine_segment_cycle_relation(
        Segment((1, 1, -1.0e9), (1, 1, 1.0e9)), cycle
    )

    assert relation.state is PiercingState.UNDETERMINED
    assert SegmentCycleIndeterminacy.SELF_INTERSECTION in relation.indeterminacy_causes


def test_nonplanar_consensus_requires_every_embedded_surface_to_agree():
    cycle = Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)])
    relation = determine_segment_cycle_relation(
        Segment((0.6, 0.8, -1), (0.6, 0.8, 1)), cycle
    )

    assert relation.state is PiercingState.PIERCES
    assert relation.surface_model is CycleSurfaceModel.VERTEX_TRIANGULATION_FAMILY
    evidence = relation.surface_evidence
    assert evidence.enumeration_complete
    assert evidence.enumerated_surface_count == 2
    assert evidence.embedded_surface_count == 2
    assert evidence.intersecting_surface_count == 2
    assert evidence.embedded_surface_count == (
        evidence.intersecting_surface_count
        + evidence.non_piercing_surface_count
        + evidence.evaluation_undetermined_count
    )
    assert evidence.enumerated_surface_count == (
        evidence.embedded_surface_count
        + evidence.proven_non_embedded_surface_count
        + evidence.construction_undetermined_count
    )


def test_internal_triangulation_diagonal_contact_is_conservatively_undetermined():
    cycle = Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)])
    relation = determine_segment_cycle_relation(
        Segment((0.8, 0.8, -1), (0.8, 0.8, 1)), cycle
    )

    assert relation.state is PiercingState.UNDETERMINED
    assert relation.surface_evidence.evaluation_undetermined_count >= 1
    assert SegmentCycleIndeterminacy.NUMERIC_BAND in relation.indeterminacy_causes


def test_internal_diagonal_endpoint_contact_is_conservatively_undetermined():
    cycle = Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)])
    relation = determine_segment_cycle_relation(
        Segment((0.8, 0.8, 0.16), (0.8, 0.8, 1)), cycle
    )

    assert relation.state is PiercingState.UNDETERMINED
    assert relation.surface_evidence.evaluation_undetermined_count >= 1
    assert SegmentCycleIndeterminacy.NUMERIC_BAND in relation.indeterminacy_causes


def test_nonplanar_cycle_vertex_endpoint_is_a_stable_boundary_contact():
    cycle = Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)])

    relation = determine_segment_cycle_relation(
        Segment((0, 0, 0), (-1, -1, -1)), cycle
    )

    assert relation.state is PiercingState.DOES_NOT_PIERCE
    assert SegmentCycleFeature.SEGMENT_ENDPOINT_CONTACT in relation.features
    assert SegmentCycleFeature.CYCLE_VERTEX_CONTACT in relation.features
    assert relation.surface_evidence.evaluation_undetermined_count == 0


def test_cycle_vertex_contact_does_not_hide_a_later_surface_crossing():
    cycle = Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 2), (0, 2, 0)])

    relation = determine_segment_cycle_relation(
        Segment((0, 0, 0), (2, 2, 1)), cycle
    )

    assert relation.state is not PiercingState.DOES_NOT_PIERCE
    assert SegmentCycleFeature.SEGMENT_ENDPOINT_CONTACT in relation.features
    assert SegmentCycleFeature.CYCLE_VERTEX_CONTACT in relation.features
    assert SegmentCycleFeature.TRANSVERSE_INTERIOR in relation.features
    assert relation.surface_evidence.intersecting_surface_count >= 1


def test_nonplanar_triangle_coplanarity_requires_finite_overlap():
    cycle = Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)])

    relation = determine_segment_cycle_relation(
        Segment((10, 10, 0), (11, 10, 0)), cycle
    )

    assert SegmentCycleFeature.COPLANAR_CONTACT not in relation.features


def test_surface_budget_exhaustion_never_forms_a_partial_consensus():
    cycle = Cycle(
        [
            (0, 0, 0),
            (2, 0, 0),
            (3, 1, 0.1),
            (2, 2, 0),
            (0, 2, -0.1),
        ]
    )
    settings = GeometrySettings(
        surface=SurfaceEnumerationSettings(
            maximum_cycle_vertices=8,
            maximum_surface_count=1,
            maximum_segment_triangle_tests=792,
            maximum_triangle_pair_tests=1980,
        )
    )
    relation = determine_segment_cycle_relation(
        Segment((1, 1, -1), (1, 1, 1)), cycle, settings
    )

    assert relation.state is PiercingState.UNDETERMINED
    assert not relation.surface_evidence.enumeration_complete
    assert relation.surface_evidence.enumerated_surface_count <= 1
    assert SegmentCycleIndeterminacy.INCOMPLETE_SURFACE_FAMILY in relation.indeterminacy_causes


def test_segment_budget_accounts_for_every_embedded_surface():
    cycle = Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)])
    settings = replace(
        DEFAULT_GEOMETRY_SETTINGS,
        surface=replace(
            DEFAULT_GEOMETRY_SETTINGS.surface,
            maximum_segment_triangle_tests=1,
        ),
    )

    relation = determine_segment_cycle_relation(
        Segment((0.6, 0.8, -1), (0.6, 0.8, 1)), cycle, settings
    )
    evidence = relation.surface_evidence

    assert relation.state is PiercingState.UNDETERMINED
    assert evidence.embedded_surface_count == (
        evidence.intersecting_surface_count
        + evidence.non_piercing_surface_count
        + evidence.evaluation_undetermined_count
    )


def test_batch_relations_prepare_nonplanar_surface_once(monkeypatch):
    import hotpot.cheminfo.geometry.relation as relation_module

    cycle = Cycle([(0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)])
    segments = (
        Segment((0.6, 0.8, -1), (0.6, 0.8, 1)),
        Segment((0.6, 0.8, 1), (0.6, 0.8, 2)),
    )
    expected = tuple(
        determine_segment_cycle_relation(segment, cycle) for segment in segments
    )
    prepare = relation_module._prepare_nonplanar_surface_family
    calls = 0

    def counted_prepare(cycle, settings):
        nonlocal calls
        calls += 1
        return prepare(cycle, settings)

    monkeypatch.setattr(
        relation_module, "_prepare_nonplanar_surface_family", counted_prepare
    )

    actual = tuple(iter_segment_cycle_relations(segments, cycle))

    assert actual == expected
    assert calls == 1


def test_huge_finite_coordinates_are_numerically_undetermined():
    cycle = Cycle(
        [
            (-1.0e308, 0, 0),
            (0, 1.0e308, 0),
            (1.0e308, 0, 0),
            (0, -1.0e308, 1),
        ]
    )

    relation = determine_segment_cycle_relation(
        Segment((0, 0, -1), (0, 0, 1)), cycle
    )

    assert relation.state is PiercingState.UNDETERMINED
    assert SegmentCycleIndeterminacy.NUMERIC_BAND in relation.indeterminacy_causes


def test_nonfinite_and_degenerate_inputs_are_not_silently_nonpiercing(square):
    nonfinite = determine_segment_cycle_relation(
        Segment((1, 1, math.nan), (1, 1, 1)), square
    )
    degenerate = determine_segment_cycle_relation(
        Segment((1, 1, 1), (1, 1, 1)), square
    )

    assert nonfinite.state is PiercingState.UNDETERMINED
    assert nonfinite.surface_model is None
    assert SegmentCycleIndeterminacy.NONFINITE_INPUT in nonfinite.indeterminacy_causes
    assert degenerate.state is PiercingState.UNDETERMINED
    assert degenerate.surface_model is None
    assert SegmentCycleIndeterminacy.DEGENERATE_SEGMENT in degenerate.indeterminacy_causes


def test_closest_boundary_edge_uses_smallest_index_for_tolerance_ties(square):
    result = closest_cycle_edge(square, Segment((-1, -1, -1), (-1, -1, 1)))

    assert result is not None
    assert result.edge_index == 0
    assert result.distance == pytest.approx(math.sqrt(2.0))
