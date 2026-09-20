"""Deterministic graph fixtures for Relevant Cycle tests."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx


Edge = tuple[int, int]
GraphCase = tuple[str, tuple[Edge, ...]]


def normalized_edges(edges: Sequence[Sequence[int]]) -> tuple[Edge, ...]:
    return tuple(sorted(tuple(sorted(edge)) for edge in edges))


def reference_graphs() -> tuple[GraphCase, ...]:
    return (
        ("tree", ((0, 1), (1, 2), (1, 3))),
        ("triangle", ((0, 1), (1, 2), (0, 2))),
        (
            "square_with_diagonal",
            ((0, 1), (1, 2), (2, 3), (0, 3), (0, 2)),
        ),
        (
            "theta",
            ((0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)),
        ),
        (
            "naphthalene_topology",
            (
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5),
                (4, 6), (6, 7), (7, 8), (8, 9), (5, 9),
            ),
        ),
        (
            "anthracene_topology",
            (
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5),
                (3, 6), (6, 7), (7, 8), (8, 9), (4, 9),
                (7, 10), (10, 11), (11, 12), (12, 13), (8, 13),
            ),
        ),
        ("complete_4", normalized_edges(tuple(nx.complete_graph(4).edges))),
        ("cubane", normalized_edges(tuple(nx.cubical_graph().edges))),
        (
            "disconnected_cycles",
            ((0, 1), (1, 2), (0, 2), (10, 11), (11, 12), (10, 12)),
        ),
    )
