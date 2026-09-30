"""Public cut-over fence for the native prepared-cycle backend."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Callable, Iterator, Union

import pytest

from hotpot.cheminfo.geometry import native, relation
from hotpot.cheminfo.geometry.object import Cycle, Segment
from hotpot.cheminfo.geometry.relation import (
    PiercingState,
    SegmentCycleRelation,
    SegmentCycleScreening,
)


def _segments() -> tuple[Segment, ...]:
    return (
        Segment((0.6, 0.8, -1.0), (0.6, 0.8, 1.0)),
        Segment((10.0, 10.0, 4.0), (11.0, 10.0, 4.0)),
        Segment((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )


@pytest.mark.parametrize(
    "cycle",
    (
        Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))),
        Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0))),
    ),
    ids=("planar", "nonplanar"),
)
def test_public_iterators_preserve_scalar_results(cycle: Cycle) -> None:
    segments = _segments()

    relations = tuple(relation.iter_segment_cycle_relations(segments, cycle))
    screenings = tuple(relation.iter_segment_cycle_screenings(segments, cycle))

    assert relations == tuple(
        relation.determine_segment_cycle_relation(segment, cycle)
        for segment in segments
    )
    assert tuple(item.state for item in relations) == (
        PiercingState.PIERCES,
        PiercingState.DOES_NOT_PIERCE,
        PiercingState.UNDETERMINED,
    )
    assert tuple(item.aabb_separated for item in screenings) == (
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
def test_public_iterators_are_lazy_and_prepare_the_cycle_once(
    iterator: Callable[
        [Iterator[Segment], Cycle],
        Union[Iterator[SegmentCycleRelation], Iterator[SegmentCycleScreening]],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle = Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)))
    consumed: list[Segment] = []
    prepared: list[Cycle] = []
    original_prepare = native._prepare_cycle

    def segment_source() -> Iterator[Segment]:
        for segment in _segments():
            consumed.append(segment)
            yield segment

    def counted_prepare(cycle: Cycle, settings=relation.DEFAULT_GEOMETRY_SETTINGS):
        prepared.append(cycle)
        return original_prepare(cycle, settings)

    monkeypatch.setattr(native, "_prepare_cycle", counted_prepare)
    monkeypatch.setattr(native, "_SEGMENT_CYCLE_BATCH_SIZE", 2)
    results = iterator(segment_source(), cycle)

    assert consumed == []
    assert prepared == []
    next(results)
    assert consumed == list(_segments()[:2])
    assert prepared == [cycle]
    tuple(results)
    assert consumed == list(_segments())
    assert prepared == [cycle]


@pytest.mark.parametrize(
    ("iterator", "native_batch_name"),
    (
        (relation.iter_segment_cycle_relations, "segment_cycle_relations"),
        (relation.iter_segment_cycle_screenings, "segment_cycle_screenings"),
    ),
    ids=("relations", "screenings"),
)
def test_public_iterators_batch_one_shot_inputs_in_order(
    iterator: Callable[
        [Iterator[Segment], Cycle],
        Union[Iterator[SegmentCycleRelation], Iterator[SegmentCycleScreening]],
    ],
    native_batch_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle = Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
    segment_count = native._SEGMENT_CYCLE_BATCH_SIZE * 2 + 1
    segments = tuple(
        Segment((1, 1, -1), (1, 1, 1))
        if index % 2 == 0
        else Segment((10 + index, 10, -1), (10 + index, 10, 1))
        for index in range(segment_count)
    )
    iteration_count = 0
    yielded_segments: list[Segment] = []
    batch_sizes: list[int] = []

    class OneShotSegments:
        def __iter__(self) -> Iterator[Segment]:
            nonlocal iteration_count
            iteration_count += 1
            assert iteration_count == 1
            for segment in segments:
                yielded_segments.append(segment)
                yield segment

    native_batch = getattr(native._native, native_batch_name)

    def counted_batch(segment_array, prepared_cycle):
        batch_sizes.append(len(segment_array))
        return native_batch(segment_array, prepared_cycle)

    monkeypatch.setattr(native._native, native_batch_name, counted_batch)

    results = iterator(OneShotSegments(), cycle)
    assert yielded_segments == []

    states = tuple(result.state for result in results)

    assert iteration_count == 1
    assert yielded_segments == list(segments)
    assert batch_sizes == [
        native._SEGMENT_CYCLE_BATCH_SIZE,
        native._SEGMENT_CYCLE_BATCH_SIZE,
        1,
    ]
    assert states == tuple(
        PiercingState.PIERCES
        if index % 2 == 0
        else PiercingState.DOES_NOT_PIERCE
        for index in range(segment_count)
    )


@pytest.mark.parametrize(
    ("iterator", "native_batch_name"),
    (
        (relation.iter_segment_cycle_relations, "segment_cycle_relations"),
        (relation.iter_segment_cycle_screenings, "segment_cycle_screenings"),
    ),
    ids=("relations", "screenings"),
)
def test_public_iterators_skip_native_work_for_empty_inputs(
    iterator: Callable[
        [Iterator[Segment], Cycle],
        Union[Iterator[SegmentCycleRelation], Iterator[SegmentCycleScreening]],
    ],
    native_batch_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle = Cycle(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))

    def unexpected_call(*args, **kwargs):
        raise AssertionError("empty input must not prepare or enter native code")

    monkeypatch.setattr(native, "_prepare_cycle", unexpected_call)
    monkeypatch.setattr(native._native, native_batch_name, unexpected_call)

    assert tuple(iterator(iter(()), cycle)) == ()


def test_public_screening_does_not_hide_invalid_cycles_behind_aabb() -> None:
    cycle = Cycle(((0, 0, 0), (2, 2, 0), (0, 2, 0), (2, 0, 0)))
    screening = next(relation.iter_segment_cycle_screenings(
        (Segment((10, 10, 4), (11, 10, 4)),),
        cycle,
    ))

    assert screening.state is PiercingState.UNDETERMINED
    assert not screening.aabb_separated
    assert screening.relation is not None


def test_relation_module_contains_no_python_numerical_backend_or_fallback() -> None:
    source = Path(relation.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    private_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("_")
    }

    assert "numpy" not in imported_roots
    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))
    assert private_functions == set()


def test_cold_import_reports_a_missing_native_extension() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockGeometryNative(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "hotpot.cheminfo.geometry._geometry_native":
                    raise ImportError("blocked native extension")
                return None

        sys.meta_path.insert(0, BlockGeometryNative())
        try:
            import hotpot.cheminfo.geometry
        except ImportError as error:
            assert "native extension is unavailable" in str(error)
        else:
            raise AssertionError("geometry import silently bypassed its native backend")
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
