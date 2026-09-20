"""Small ctypes adapter for the pinned RingDecomposerLib test oracle.

This module is test infrastructure, not a production fallback.  It exposes only
the Relevant Cycle iterator needed for differential validation.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

RDL_LIBRARY_ENV = "HOTPOT_RDL_ORACLE_LIBRARY"

NodeCycle = Tuple[int, ...]


class _RDLGraph(ctypes.Structure):
    pass


class _RDLData(ctypes.Structure):
    pass


class _RDLCycleIterator(ctypes.Structure):
    pass


_RDLNode = ctypes.c_uint
_RDLEdge = _RDLNode * 2


class _RDLCycle(ctypes.Structure):
    _fields_ = (
        ("edges", ctypes.POINTER(_RDLEdge)),
        ("weight", ctypes.c_uint),
        ("urf", ctypes.c_uint),
        ("rcf", ctypes.c_uint),
    )


def canonical_cycle(cycle: Sequence[int]) -> NodeCycle:
    """Return the lexicographically smallest rotation in either direction."""
    nodes = tuple(cycle)
    rotations = tuple(
        nodes[offset:] + nodes[:offset] for offset in range(len(nodes))
    )
    reversed_nodes = tuple(reversed(nodes))
    reverse_rotations = tuple(
        reversed_nodes[offset:] + reversed_nodes[:offset]
        for offset in range(len(reversed_nodes))
    )
    return min(rotations + reverse_rotations)


def _ordered_cycle_from_edges(edges: Sequence[Tuple[int, int]]) -> NodeCycle:
    adjacency = {}
    for node1, node2 in edges:
        adjacency.setdefault(node1, []).append(node2)
        adjacency.setdefault(node2, []).append(node1)

    start = min(adjacency)
    previous = None
    current = start
    ordered = []
    while True:
        ordered.append(current)
        candidates = adjacency[current]
        next_node = candidates[0] if candidates[0] != previous else candidates[1]
        previous, current = current, next_node
        if current == start:
            break
    return canonical_cycle(ordered)


class RDLOracle:
    """Relevant Cycle oracle backed by a locally compiled RDL shared library."""

    def __init__(self, library_path: Path):
        self.library_path = Path(library_path).resolve()
        self._library = ctypes.CDLL(str(self.library_path))
        self._configure_api()

    def _configure_api(self) -> None:
        library = self._library
        library.RDL_initNewGraph.argtypes = (ctypes.c_uint,)
        library.RDL_initNewGraph.restype = ctypes.POINTER(_RDLGraph)

        library.RDL_deleteGraph.argtypes = (ctypes.POINTER(_RDLGraph),)
        library.RDL_deleteGraph.restype = None

        library.RDL_addUEdge.argtypes = (
            ctypes.POINTER(_RDLGraph),
            ctypes.c_uint,
            ctypes.c_uint,
        )
        library.RDL_addUEdge.restype = ctypes.c_uint

        library.RDL_calculate.argtypes = (ctypes.POINTER(_RDLGraph),)
        library.RDL_calculate.restype = ctypes.POINTER(_RDLData)

        library.RDL_deleteData.argtypes = (ctypes.POINTER(_RDLData),)
        library.RDL_deleteData.restype = None

        library.RDL_getRCyclesIterator.argtypes = (ctypes.POINTER(_RDLData),)
        library.RDL_getRCyclesIterator.restype = ctypes.POINTER(_RDLCycleIterator)

        library.RDL_cycleIteratorAtEnd.argtypes = (
            ctypes.POINTER(_RDLCycleIterator),
        )
        library.RDL_cycleIteratorAtEnd.restype = ctypes.c_int

        library.RDL_cycleIteratorGetCycle.argtypes = (
            ctypes.POINTER(_RDLCycleIterator),
        )
        library.RDL_cycleIteratorGetCycle.restype = ctypes.POINTER(_RDLCycle)

        library.RDL_cycleIteratorNext.argtypes = (
            ctypes.POINTER(_RDLCycleIterator),
        )
        library.RDL_cycleIteratorNext.restype = ctypes.POINTER(_RDLCycleIterator)

        library.RDL_deleteCycle.argtypes = (ctypes.POINTER(_RDLCycle),)
        library.RDL_deleteCycle.restype = None

        library.RDL_deleteCycleIterator.argtypes = (
            ctypes.POINTER(_RDLCycleIterator),
        )
        library.RDL_deleteCycleIterator.restype = None

    def relevant_cycles(
        self,
        edges: Iterable[Sequence[int]],
        *,
        max_size: Optional[int] = None,
        max_cycles: Optional[int] = 100_000,
    ) -> Tuple[NodeCycle, ...]:
        """Return exact Relevant Cycles for a simple undirected graph."""
        original_edges = tuple((int(edge[0]), int(edge[1])) for edge in edges)
        if not original_edges:
            return ()

        original_nodes = sorted({node for edge in original_edges for node in edge})
        dense_node = {node: index for index, node in enumerate(original_nodes)}
        dense_edges = tuple(
            (dense_node[node1], dense_node[node2])
            for node1, node2 in original_edges
        )

        graph = self._library.RDL_initNewGraph(len(original_nodes))
        if not graph:
            raise RuntimeError("RingDecomposerLib could not allocate a graph")

        data = None
        try:
            for node1, node2 in dense_edges:
                edge_id = self._library.RDL_addUEdge(graph, node1, node2)
                if edge_id >= 2**32 - 2:
                    raise RuntimeError("RingDecomposerLib rejected an input edge")

            data = self._library.RDL_calculate(graph)
            if not data:
                raise RuntimeError("RingDecomposerLib calculation failed")
            graph = None
            dense_cycles = self._read_cycles(
                data,
                max_size=max_size,
                max_cycles=max_cycles,
            )
        finally:
            if data:
                self._library.RDL_deleteData(data)
            elif graph:
                self._library.RDL_deleteGraph(graph)

        normalized_cycles = tuple(
            canonical_cycle(tuple(original_nodes[node] for node in cycle))
            for cycle in dense_cycles
        )
        if len(set(normalized_cycles)) != len(normalized_cycles):
            raise RuntimeError("RingDecomposerLib returned duplicate Relevant Cycles")
        return tuple(
            sorted(normalized_cycles, key=lambda cycle: (len(cycle), cycle))
        )

    def _read_cycles(
        self,
        data: ctypes.POINTER(_RDLData),
        *,
        max_size: Optional[int],
        max_cycles: Optional[int],
    ) -> Tuple[NodeCycle, ...]:
        iterator = self._library.RDL_getRCyclesIterator(data)
        if not iterator:
            raise RuntimeError("RingDecomposerLib could not create a cycle iterator")

        cycles = []
        try:
            while not self._library.RDL_cycleIteratorAtEnd(iterator):
                cycle_pointer = self._library.RDL_cycleIteratorGetCycle(iterator)
                if not cycle_pointer:
                    raise RuntimeError("RingDecomposerLib returned an invalid cycle")
                try:
                    cycle = cycle_pointer.contents
                    cycle_edges = tuple(
                        (cycle.edges[index][0], cycle.edges[index][1])
                        for index in range(cycle.weight)
                    )
                    if max_size is None or cycle.weight <= max_size:
                        cycles.append(_ordered_cycle_from_edges(cycle_edges))
                        if max_cycles is not None and len(cycles) > max_cycles:
                            raise RuntimeError(
                                "RingDecomposerLib oracle exceeded max_cycles"
                            )
                finally:
                    self._library.RDL_deleteCycle(cycle_pointer)
                self._library.RDL_cycleIteratorNext(iterator)
        finally:
            self._library.RDL_deleteCycleIterator(iterator)
        return tuple(cycles)


def load_oracle_from_environment() -> Optional[RDLOracle]:
    """Load the optional oracle selected by ``HOTPOT_RDL_ORACLE_LIBRARY``."""
    library_path = os.environ.get(RDL_LIBRARY_ENV)
    return RDLOracle(Path(library_path)) if library_path else None
