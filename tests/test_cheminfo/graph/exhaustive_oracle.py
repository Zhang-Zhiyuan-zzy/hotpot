"""Independent exhaustive Relevant Cycle oracle for small test graphs only."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import networkx as nx


NodeCycle = tuple[int, ...]


def canonical_cycle(cycle: Sequence[int]) -> NodeCycle:
    """Return the least rotation across both traversal directions."""
    node_cycle = tuple(cycle)
    return min(
        orientation[offset:] + orientation[:offset]
        for orientation in (node_cycle, tuple(reversed(node_cycle)))
        for offset in range(len(node_cycle))
    )


def cycle_edges(cycle: Sequence[int]) -> frozenset[tuple[int, int]]:
    """Return the undirected edge set of an ordered cycle."""
    return frozenset(
        tuple(sorted((cycle[index], cycle[(index + 1) % len(cycle)])))
        for index in range(len(cycle))
    )


def _insert_vector(basis: dict[int, int], vector: int) -> None:
    while vector:
        pivot = vector.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = vector
            return
        vector ^= basis[pivot]


def _is_in_span(basis: dict[int, int], vector: int) -> bool:
    while vector:
        pivot = vector.bit_length() - 1
        if pivot not in basis:
            return False
        vector ^= basis[pivot]
    return True


def exhaustive_relevant_cycles(
    edges: Iterable[Sequence[int]],
) -> tuple[NodeCycle, ...]:
    """Enumerate and classify all simple cycles; suitable only for small graphs."""
    graph = nx.Graph(tuple(tuple(edge) for edge in edges))
    canonical_cycles = {
        canonical_cycle(cycle)
        for cycle in nx.simple_cycles(graph)
        if len(cycle) >= 3
    }
    ordered_edges = tuple(sorted(tuple(sorted(edge)) for edge in graph.edges))
    edge_index = {edge: position for position, edge in enumerate(ordered_edges)}

    cycle_vectors = {
        cycle: sum(1 << edge_index[edge] for edge in cycle_edges(cycle))
        for cycle in canonical_cycles
    }
    shorter_cycle_basis: dict[int, int] = {}
    relevant = []
    cycle_lengths = sorted({len(cycle) for cycle in canonical_cycles})
    for cycle_length in cycle_lengths:
        same_length = sorted(
            cycle for cycle in canonical_cycles if len(cycle) == cycle_length
        )
        relevant.extend(
            cycle
            for cycle in same_length
            if not _is_in_span(shorter_cycle_basis, cycle_vectors[cycle])
        )
        for cycle in same_length:
            _insert_vector(shorter_cycle_basis, cycle_vectors[cycle])

    return tuple(sorted(relevant, key=lambda cycle: (len(cycle), cycle)))
