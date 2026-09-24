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
