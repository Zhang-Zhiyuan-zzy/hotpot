"""Behavioral tests for the private scalar prepared-cycle seam."""

from typing import Callable, Iterator, Optional, Union

import numpy as np
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
    assert not prepared_cycle.edge_coordinates.flags.writeable
    assert prepared_cycle.planar_projection is not None
    assert not prepared_cycle.planar_projection.flags.writeable
    with pytest.raises(ValueError):
        prepared_cycle.coordinates[0, 0] = 1.0


def test_nonplanar_preparation_reuses_one_cycle_coordinate_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle = Cycle(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.4), (0.0, 2.0, 0.0))
    )
    original_prepare = relation._prepare_nonplanar_surface_family
    coordinate_snapshots = []

    def recorded_prepare(
        cycle: Cycle,
        settings: GeometrySettings,
        coordinates: Optional[np.ndarray] = None,
    ):
        coordinate_snapshots.append(coordinates)
        return original_prepare(cycle, settings, coordinates)

    monkeypatch.setattr(
        relation,
        "_prepare_nonplanar_surface_family",
        recorded_prepare,
    )

    prepared_cycle = relation._prepare_cycle_geometry(
        cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    )

    assert len(coordinate_snapshots) == 1
    assert coordinate_snapshots[0] is prepared_cycle.coordinates


def test_unique_triangle_facts_are_reused_and_read_only() -> None:
    cycle = Cycle(tuple(
        (
            2.0 * np.cos(index * np.pi / 4.0),
            2.0 * np.sin(index * np.pi / 4.0),
            0.2 if index % 2 else -0.15,
        )
        for index in range(8)
    ))

    prepared_cycle = relation._prepare_cycle_geometry(
        cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    )
    family = prepared_cycle.nonplanar_surface_family

    assert family is not None
    assert len(family.unique_triangles) == 56
    unique_ids = {id(triangle) for triangle in family.unique_triangles}
    assert all(
        id(triangle) in unique_ids
        for surface in family.embedded_surfaces
        for triangle in surface.triangles
    )
    for triangle in family.unique_triangles:
        assert not triangle.coordinates.flags.writeable
        assert not triangle.normal.flags.writeable
        assert all(not bound.flags.writeable for bound in triangle.bounds)
        assert all(not edge.coordinates.flags.writeable for edge in triangle.edges)


def test_batched_aabb_mask_matches_scalar_bounds_predicate() -> None:
    rng = np.random.default_rng(147)
    cycle = Cycle(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    prepared_cycle = relation._prepare_cycle_geometry(
        cycle,
        relation.DEFAULT_GEOMETRY_SETTINGS,
    )
    segments = tuple(
        Segment(rng.normal(size=3), rng.normal(size=3))
        for _ in range(64)
    )
    queries = tuple(
        relation._prepare_segment_cycle_query(
            segment,
            cycle,
            relation.DEFAULT_GEOMETRY_SETTINGS,
        )
        for segment in segments
    )

    batch = relation._segment_cycle_aabb_separation_mask(
        queries,
        prepared_cycle.bounds,
    )
    scalar = np.asarray([
        relation._bounds_stably_separated(
            query.geometry.bounds,
            prepared_cycle.bounds,
            query.tolerances.aabb,
        )
        if not query.causes and query.tolerances is not None
        else False
        for query in queries
    ])

    assert np.array_equal(batch, scalar)


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
