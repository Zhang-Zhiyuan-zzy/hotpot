"""Behavioral tests for the private scalar prepared-cycle seam."""

from typing import Callable, Iterator, Union

import pytest

from hotpot.cheminfo.geometry import relation
from hotpot.cheminfo.geometry.object import Cycle, Segment
from hotpot.cheminfo.geometry.relation import (
    PiercingState,
    SegmentCycleRelation,
    SegmentCycleScreening,
)
from hotpot.cheminfo.geometry.settings import GeometrySettings


def _segments() -> tuple[Segment, ...]:
    return (
        Segment((0.6, 0.8, -1.0), (0.6, 0.8, 1.0)),
        Segment((10.0, 10.0, 4.0), (11.0, 10.0, 4.0)),
        Segment((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )


@pytest.mark.parametrize(
    "cycle",
    (
        Cycle(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 2.0, 0.0),
                (0.0, 2.0, 0.0),
            )
        ),
        Cycle(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 2.0, 0.4),
                (0.0, 2.0, 0.0),
            )
        ),
    ),
    ids=("planar", "nonplanar"),
)
def test_prepared_scalar_paths_preserve_relation_and_screening_results(
    cycle: Cycle,
) -> None:
    segments = _segments()
    prepared_cycle = relation._prepare_cycle_geometry(
        cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    )

    prepared_relations = tuple(relation._iter_prepared_segment_cycle_relations(
        segments,
        prepared_cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    ))
    public_relations = tuple(relation.iter_segment_cycle_relations(segments, cycle))
    prepared_screenings = tuple(relation._iter_prepared_segment_cycle_screenings(
        segments,
        prepared_cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    ))
    public_screenings = tuple(relation.iter_segment_cycle_screenings(segments, cycle))

    assert prepared_relations == public_relations
    assert prepared_screenings == public_screenings
    assert tuple(item.state for item in prepared_relations) == (
        PiercingState.PIERCES,
        PiercingState.DOES_NOT_PIERCE,
        PiercingState.UNDETERMINED,
    )
    assert tuple(item.aabb_separated for item in prepared_screenings) == (
        False,
        True,
        False,
    )


@pytest.mark.parametrize(
    "iterator",
    (
        relation.iter_segment_cycle_relations,
        relation.iter_segment_cycle_screenings,
    ),
    ids=("relations", "screenings"),
)
def test_public_batch_prepares_cycle_once(
    iterator: Callable[
        [tuple[Segment, ...], Cycle],
        Union[
            Iterator[SegmentCycleRelation],
            Iterator[SegmentCycleScreening],
        ],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle = Cycle(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.4), (0.0, 2.0, 0.0))
    )
    original_prepare = relation._prepare_cycle_geometry
    prepared_cycles = []

    def counted_prepare(cycle: Cycle, settings: GeometrySettings):
        prepared_cycles.append(cycle)
        return original_prepare(cycle, settings)

    monkeypatch.setattr(relation, "_prepare_cycle_geometry", counted_prepare)

    assert len(tuple(iterator(_segments(), cycle))) == 3
    assert prepared_cycles == [cycle]


def test_prepared_cycle_owns_read_only_coordinate_facts() -> None:
    cycle = Cycle(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )

    prepared_cycle = relation._prepare_cycle_geometry(
        cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    )

    assert not prepared_cycle.coordinates.flags.writeable
    assert all(not bound.flags.writeable for bound in prepared_cycle.bounds)
    with pytest.raises(ValueError):
        prepared_cycle.coordinates[0, 0] = 1.0


def test_prepared_aabb_does_not_hide_an_invalid_cycle() -> None:
    self_intersecting_cycle = Cycle(
        ((0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0), (2.0, 0.0, 0.0))
    )
    prepared_cycle = relation._prepare_cycle_geometry(
        self_intersecting_cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    )

    screening = next(relation._iter_prepared_segment_cycle_screenings(
        (Segment((10.0, 10.0, 4.0), (11.0, 10.0, 4.0)),),
        prepared_cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    ))

    assert screening.state is PiercingState.UNDETERMINED
    assert not screening.aabb_separated
    assert screening.relation is not None


def test_cycle_topology_template_is_reused_by_vertex_count() -> None:
    relation._cycle_topology_template.cache_clear()

    first = relation._cycle_topology_template(4)
    second = relation._cycle_topology_template(4)
    different_size = relation._cycle_topology_template(5)

    assert first is second
    assert first is not different_size
    assert relation._cycle_topology_template.cache_info().hits == 1
    assert relation._cycle_topology_template.cache_info().misses == 2
    assert all(isinstance(surface, tuple) for surface in first.triangulations)
    assert all(isinstance(edges, tuple) for edges in first.internal_edges)
    assert all(isinstance(pairs, tuple) for pairs in first.triangle_pairs)
    assert all(
        isinstance(simplices, tuple)
        for simplices in first.shared_simplices
    )


def test_topology_cache_never_reuses_coordinate_facts() -> None:
    relation._cycle_topology_template.cache_clear()
    first_cycle = Cycle(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.4), (0.0, 2.0, 0.0))
    )
    shifted_cycle = Cycle(tuple(
        (point.x + 10.0, point.y, point.z)
        for point in first_cycle.vertices
    ))

    first = relation._prepare_cycle_geometry(
        first_cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    )
    shifted = relation._prepare_cycle_geometry(
        shifted_cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    )
    segment = (Segment((0.6, 0.8, -1.0), (0.6, 0.8, 1.0)),)
    first_result = tuple(relation._iter_prepared_segment_cycle_screenings(
        segment,
        first,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    ))
    shifted_result = tuple(relation._iter_prepared_segment_cycle_screenings(
        segment,
        shifted,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    ))

    assert relation._cycle_topology_template.cache_info().misses == 1
    assert relation._cycle_topology_template.cache_info().hits == 1
    assert first is not shifted
    assert first.nonplanar_surface_family is not shifted.nonplanar_surface_family
    assert not (first.coordinates == shifted.coordinates).all()
    assert first_result[0].state is PiercingState.PIERCES
    assert shifted_result[0].state is PiercingState.DOES_NOT_PIERCE
    assert shifted_result[0].aabb_separated
