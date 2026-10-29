"""Deterministic connected components over explicit integer edges."""

from __future__ import annotations

from collections.abc import Iterable


def connected_components(
    node_count: int,
    edges: Iterable[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    """Return ascending components for nodes ``range(node_count)``."""
    parents = list(range(node_count))
    ranks = [0] * node_count

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for first, second in edges:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            continue
        if ranks[first_root] < ranks[second_root]:
            first_root, second_root = second_root, first_root
        parents[second_root] = first_root
        if ranks[first_root] == ranks[second_root]:
            ranks[first_root] += 1

    grouped: dict[int, list[int]] = {}
    for node in range(node_count):
        grouped.setdefault(find(node), []).append(node)
    return tuple(
        sorted(
            (tuple(nodes) for nodes in grouped.values()),
            key=lambda component: component[0],
        )
    )


__all__ = ("connected_components",)
