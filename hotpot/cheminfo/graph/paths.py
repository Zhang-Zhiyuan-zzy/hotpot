"""Deterministic path algorithms over explicit integer topology."""

from __future__ import annotations

from collections import deque
from collections.abc import Container, Iterable


def _ordered_adjacency(
    node_count: int,
    edges: Iterable[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(node_count)]
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    return tuple(tuple(sorted(neighbors)) for neighbors in adjacency)


def shortest_path(
    node_count: int,
    edges: Iterable[tuple[int, int]],
    source: int,
    target: int,
) -> list[int]:
    """Return the lexicographically first shortest path, or ``[]``."""
    if source == target:
        return [source]
    adjacency = _ordered_adjacency(node_count, edges)
    predecessors = {source: -1}
    pending = deque([source])
    while pending:
        node = pending.popleft()
        for neighbor in adjacency[node]:
            if neighbor in predecessors:
                continue
            predecessors[neighbor] = node
            if neighbor == target:
                path = [target]
                while path[-1] != source:
                    path.append(predecessors[path[-1]])
                path.reverse()
                return path
            pending.append(neighbor)
    return []


def all_simple_paths(
    node_count: int,
    edges: Iterable[tuple[int, int]],
    source: int,
    target: int,
    cutoff: int | None = None,
) -> list[list[int]]:
    """Return simple paths in deterministic ascending-neighbor DFS order."""
    if source == target:
        return []
    adjacency = _ordered_adjacency(node_count, edges)
    maximum_edges = node_count - 1 if cutoff is None else cutoff
    paths: list[list[int]] = []
    path = [source]
    visited = {source}

    def visit(node: int) -> None:
        if len(path) - 1 >= maximum_edges:
            return
        for neighbor in adjacency[node]:
            if neighbor in visited:
                continue
            path.append(neighbor)
            if neighbor == target:
                paths.append(path.copy())
            else:
                visited.add(neighbor)
                visit(neighbor)
                visited.remove(neighbor)
            path.pop()

    visit(source)
    return paths


def longest_simple_path(
    node_count: int,
    edges: Iterable[tuple[int, int]],
    source: int | None = None,
    target: int | None = None,
) -> list[int]:
    """Return the first longest path under ascending pair/neighbor order."""
    if node_count == 0:
        return []
    adjacency = _ordered_adjacency(node_count, edges)

    def longest_between(first: int, second: int) -> list[int] | None:
        best = None
        path = [first]
        visited = {first}

        def visit(node: int) -> None:
            nonlocal best
            if node == second:
                if best is None or len(path) > len(best):
                    best = path.copy()
                return
            for neighbor in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                path.append(neighbor)
                visit(neighbor)
                path.pop()
                visited.remove(neighbor)

        visit(first)
        return best

    if source is not None and target is not None:
        path = longest_between(source, target)
        return path if path is not None else [source, target]

    starts = (source,) if source is not None else range(node_count)
    ends = (target,) if target is not None else range(node_count)
    best = None
    for first in starts:
        for second in ends:
            if first == second:
                candidate = [first]
            elif source is None and target is None and first > second:
                continue
            else:
                candidate = longest_between(first, second)
            if candidate is not None and (
                best is None or len(candidate) > len(best)
            ):
                best = candidate
    return best


def depth_first_path(
    node_count: int,
    edges: Iterable[tuple[int, int]],
    start_node: int = 0,
    scope_nodes: Container[int] | None = None,
    min_depth: int | None = None,
    max_depth: int | None = None,
) -> list[int] | None:
    """Preserve the legacy one-branch DFS events over explicit topology."""
    adjacency = _ordered_adjacency(node_count, edges)

    def visit(node: int, visited: list[int]) -> list[int] | None:
        visited.append(node)
        if max_depth and len(visited) >= max_depth:
            return visited
        for child in adjacency[node]:
            if child not in visited and scope_nodes and child in scope_nodes:
                return visit(child, visited)
        if min_depth and len(visited) >= min_depth:
            return visited
        return None

    return visit(start_node, [])


__all__ = (
    "all_simple_paths",
    "depth_first_path",
    "longest_simple_path",
    "shortest_path",
)
