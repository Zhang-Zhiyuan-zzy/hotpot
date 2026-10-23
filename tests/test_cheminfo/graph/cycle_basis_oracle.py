"""Independent exhaustive oracle for Plan 19 cycle-basis validation.

This module deliberately uses only Python integer/set operations.  It does not
import Hotpot's graph implementation or NetworkX, so future native results can
be checked against a separately implemented definition.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

Node = int
Edge = tuple[Node, Node]
Cycle = tuple[Node, ...]
Basis = tuple[Cycle, ...]


def canonical_cycle(cycle: Sequence[Node]) -> Cycle:
    """Return the rotation/reversal-invariant key for one simple cycle."""
    values = tuple(cycle)
    minimum_position = values.index(min(values))
    forward = values[minimum_position:] + values[:minimum_position]
    reversed_values = tuple(reversed(values))
    reversed_position = reversed_values.index(min(values))
    backward = (
        reversed_values[reversed_position:]
        + reversed_values[:reversed_position]
    )
    return min(forward, backward)


def normalized_graph(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
) -> tuple[tuple[Node, ...], tuple[Edge, ...], dict[Node, tuple[Node, ...]]]:
    """Build the oracle's deterministic simple undirected graph inputs."""
    ordered_nodes = tuple(sorted(nodes))
    normalized_edges = tuple(
        sorted((min(first, second), max(first, second)) for first, second in edges)
    )
    adjacency = {node: [] for node in ordered_nodes}
    for first, second in normalized_edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    return (
        ordered_nodes,
        normalized_edges,
        {node: tuple(sorted(neighbors)) for node, neighbors in adjacency.items()},
    )


def simple_cycles(nodes: Iterable[Node], edges: Iterable[Edge]) -> tuple[Cycle, ...]:
    """Enumerate every undirected simple cycle exactly once."""
    ordered_nodes, _, adjacency = normalized_graph(nodes, edges)
    cycles: set[Cycle] = set()

    def visit(start: Node, current: Node, path: list[Node], seen: set[Node]) -> None:
        for neighbor in adjacency[current]:
            if neighbor == start:
                if len(path) >= 3:
                    cycles.add(canonical_cycle(path))
            elif neighbor > start and neighbor not in seen:
                seen.add(neighbor)
                path.append(neighbor)
                visit(start, neighbor, path, seen)
                path.pop()
                seen.remove(neighbor)

    for start in ordered_nodes:
        visit(start, start, [start], {start})

    return tuple(sorted(cycles, key=lambda cycle: (len(cycle), cycle)))


def cycle_edge_mask(cycle: Sequence[Node], edges: Sequence[Edge]) -> int:
    """Encode a cycle as a GF(2) bit vector over canonical edge order."""
    edge_positions = {edge: position for position, edge in enumerate(edges)}
    mask = 0
    for position, first in enumerate(cycle):
        second = cycle[(position + 1) % len(cycle)]
        edge = (min(first, second), max(first, second))
        mask ^= 1 << edge_positions[edge]
    return mask


def gf2_rank(vectors: Iterable[int]) -> int:
    """Return the rank of integer bit vectors over GF(2)."""
    pivots: dict[int, int] = {}
    for value in vectors:
        reduced = value
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = reduced
                break
            reduced ^= pivots[pivot]
    return len(pivots)


def cycle_rank(nodes: Iterable[Node], edges: Iterable[Edge]) -> int:
    """Return ``|E| - |V| + components`` for a simple undirected graph."""
    ordered_nodes, normalized_edges, adjacency = normalized_graph(nodes, edges)
    remaining = set(ordered_nodes)
    component_count = 0
    while remaining:
        component_count += 1
        stack = [max(remaining)]
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            stack.extend(adjacency[node])
    return len(normalized_edges) - len(ordered_nodes) + component_count


def all_simple_cycle_bases(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
) -> tuple[Basis, ...]:
    """Exhaustively return every basis consisting only of simple cycles."""
    ordered_nodes, ordered_edges, _ = normalized_graph(nodes, edges)
    candidates = simple_cycles(ordered_nodes, ordered_edges)
    rank = cycle_rank(ordered_nodes, ordered_edges)
    if rank == 0:
        return ((),)

    masks = tuple(cycle_edge_mask(cycle, ordered_edges) for cycle in candidates)
    bases = []
    for positions in combinations(range(len(candidates)), rank):
        if gf2_rank(masks[position] for position in positions) == rank:
            bases.append(tuple(candidates[position] for position in positions))
    return tuple(
        sorted(
            bases,
            key=lambda basis: tuple((len(cycle), cycle) for cycle in basis),
        )
    )


def deterministic_paton_basis(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
) -> Basis:
    """Freeze Plan 19's maximum-root, ascending-neighbor Paton contract."""
    ordered_nodes, _, adjacency = normalized_graph(nodes, edges)
    remaining = set(ordered_nodes)
    cycles: list[Cycle] = []

    while remaining:
        root = max(remaining)
        stack = [root]
        predecessor = {root: root}
        used = {root: set()}
        while stack:
            node = stack.pop()
            node_used = used[node]
            for neighbor in adjacency[node]:
                if neighbor not in used:
                    predecessor[neighbor] = node
                    stack.append(neighbor)
                    used[neighbor] = {node}
                elif neighbor not in node_used:
                    neighbor_used = used[neighbor]
                    cycle = [neighbor, node]
                    predecessor_node = predecessor[node]
                    while predecessor_node not in neighbor_used:
                        cycle.append(predecessor_node)
                        predecessor_node = predecessor[predecessor_node]
                    cycle.append(predecessor_node)
                    cycles.append(tuple(cycle))
                    used[neighbor].add(node)
        remaining.difference_update(predecessor)

    return tuple(cycles)
