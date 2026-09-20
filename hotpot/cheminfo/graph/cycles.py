"""Relevant-cycle perception for integer-labelled simple graphs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from importlib import import_module
from operator import index
from types import ModuleType
from typing import Optional


__all__ = (
    "DEFAULT_RELEVANT_CYCLE_LIMIT",
    "RelevantCycleLimitExceeded",
    "relevant_cycles",
)


NodeIndex = int
EdgeInput = Sequence[int]
DenseEdge = tuple[int, int]
NodeCycle = tuple[NodeIndex, ...]

DEFAULT_RELEVANT_CYCLE_LIMIT = 10_000


class RelevantCycleLimitExceeded(RuntimeError):
    """Raised instead of returning an incomplete Relevant Cycle collection."""


def _native_module() -> ModuleType:
    try:
        return import_module("hotpot.cheminfo.graph._relevant_cycles")
    except ImportError as exc:
        raise ImportError(
            "the hotpot.cheminfo.graph native extension is unavailable; "
            "install a compatible Hotpot wheel or rebuild Hotpot from source"
        ) from exc


def _optional_nonnegative_integer(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or None")
    try:
        normalized = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer or None") from exc
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _normalize_edges(
    edges: Iterable[EdgeInput],
) -> tuple[tuple[DenseEdge, ...], tuple[NodeIndex, ...]]:
    normalized_edges: list[DenseEdge] = []
    seen_edges: set[DenseEdge] = set()
    node_indices: set[NodeIndex] = set()

    for edge in edges:
        if len(edge) != 2:
            raise ValueError("each edge must contain exactly two node indices")
        first, second = edge
        if isinstance(first, bool) or isinstance(second, bool):
            raise TypeError("node indices must be integers")
        try:
            first_index = index(first)
            second_index = index(second)
        except TypeError as exc:
            raise TypeError("node indices must be integers") from exc
        if first_index < 0 or second_index < 0:
            raise ValueError("node indices must be nonnegative")
        if first_index == second_index:
            raise ValueError("self-loops are not supported")

        normalized_edge = tuple(sorted((first_index, second_index)))
        if normalized_edge in seen_edges:
            raise ValueError(f"duplicate undirected edge: {normalized_edge!r}")
        seen_edges.add(normalized_edge)
        normalized_edges.append(normalized_edge)
        node_indices.update(normalized_edge)

    ordered_nodes = tuple(sorted(node_indices))
    dense_index = {node_index: position for position, node_index in enumerate(ordered_nodes)}
    dense_edges = tuple(
        sorted(
            (dense_index[first], dense_index[second])
            for first, second in normalized_edges
        )
    )
    return dense_edges, ordered_nodes


def _canonical_cycle(cycle: Sequence[NodeIndex]) -> NodeCycle:
    node_cycle = tuple(cycle)
    rotations = tuple(
        orientation[offset:] + orientation[:offset]
        for orientation in (node_cycle, tuple(reversed(node_cycle)))
        for offset in range(len(node_cycle))
    )
    return min(rotations)


def _cycle_vertices_from_edge_ids(
    cycle_edge_ids: Sequence[int],
    dense_edges: Sequence[DenseEdge],
) -> tuple[int, ...]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge_id in cycle_edge_ids:
        first, second = dense_edges[edge_id]
        adjacency[first].append(second)
        adjacency[second].append(first)

    if len(adjacency) < 3 or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise RuntimeError("native Relevant Cycle output is not a simple cycle")

    start = min(adjacency)
    previous: Optional[int] = None
    current = start
    ordered_vertices: list[int] = []
    while True:
        ordered_vertices.append(current)
        next_vertices = [node for node in adjacency[current] if node != previous]
        next_vertex = min(next_vertices) if previous is None else next_vertices[0]
        previous, current = current, next_vertex
        if current == start:
            break
        if len(ordered_vertices) > len(adjacency):
            raise RuntimeError("native Relevant Cycle output does not form one cycle")

    if len(ordered_vertices) != len(adjacency):
        raise RuntimeError("native Relevant Cycle output is disconnected")
    return tuple(ordered_vertices)


def relevant_cycles(
    edges: Iterable[EdgeInput],
    *,
    max_size: Optional[int] = None,
    max_cycles: Optional[int] = DEFAULT_RELEVANT_CYCLE_LIMIT,
) -> tuple[NodeCycle, ...]:
    """Return all requested Relevant Cycles of an undirected unweighted graph."""
    normalized_max_size = _optional_nonnegative_integer(max_size, "max_size")
    normalized_max_cycles = _optional_nonnegative_integer(max_cycles, "max_cycles")
    if normalized_max_size is not None and normalized_max_size < 3:
        raise ValueError("max_size must be at least 3 or None")

    dense_edges, node_indices = _normalize_edges(edges)
    if not dense_edges:
        return ()

    native = _native_module()
    try:
        cycle_edge_sets = native.relevant_cycles(
            dense_edges,
            normalized_max_size,
            normalized_max_cycles,
        )
    except native.RelevantCycleLimitExceeded as exc:
        raise RelevantCycleLimitExceeded(str(exc)) from exc

    canonical_cycles = tuple(
        _canonical_cycle(
            tuple(
                node_indices[dense_node]
                for dense_node in _cycle_vertices_from_edge_ids(
                    cycle_edge_ids,
                    dense_edges,
                )
            )
        )
        for cycle_edge_ids in cycle_edge_sets
    )
    if len(set(canonical_cycles)) != len(canonical_cycles):
        raise RuntimeError("native Relevant Cycle output contains duplicate cycles")
    if normalized_max_size is not None and any(
        len(cycle) > normalized_max_size for cycle in canonical_cycles
    ):
        raise RuntimeError("native Relevant Cycle output violates max_size")
    return tuple(sorted(canonical_cycles, key=lambda cycle: (len(cycle), cycle)))
