"""Benchmark final Plan 19 direct-topology algorithms on fixed inputs."""

from __future__ import annotations

import gc
import itertools
import json
import platform
import statistics
import subprocess
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter_ns

import networkx as nx

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.graph.components import connected_components
from hotpot.cheminfo.graph.paths import all_simple_paths, longest_simple_path


SAMPLES = 21
NODE_COUNT = 512
SHORTEST_SOURCE = 7
SHORTEST_TARGET = 503
COMPLETE_NODE_COUNT = 10
COMPLETE_SOURCE = 0
COMPLETE_TARGET = 1


def _chain(atom_count: int) -> Molecule:
    molecule = Molecule()
    for _ in range(atom_count):
        molecule._create_atom(atomic_number=6)
    for index in range(atom_count - 1):
        molecule._add_bond(index, index + 1, bond_order=1.0)
    molecule._update_topology(clear_conformers=False)
    return molecule


def _sparse_edges() -> tuple[tuple[int, int], ...]:
    edges = set()
    for component in range(8):
        first = component * 64
        edges.update((node, node + 1) for node in range(first, first + 63))
        edges.update(
            (first + offset, first + offset + 5)
            for offset in range(0, 59, 7)
        )
    return tuple(sorted(edges))


def _normalized_networkx_components(graph: nx.Graph) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            (tuple(sorted(component)) for component in nx.connected_components(graph)),
            key=lambda component: component[0],
        )
    )


def _task_local_networkx_components(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(edges)
    return _normalized_networkx_components(graph)


def _materializing_longest_pair(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    source: int,
    target: int,
) -> list[int]:
    """Reproduce the pre-fix list-materializing fixed-pair implementation."""
    paths = all_simple_paths(node_count, edges, source, target)
    return max(paths, key=len) if paths else [source, target]


def _median_us(function, *, samples: int = SAMPLES, loops: int = 1) -> float:
    function()
    values = []
    gc.disable()
    try:
        for _ in range(samples):
            start = perf_counter_ns()
            for _ in range(loops):
                function()
            values.append((perf_counter_ns() - start) / loops / 1_000.0)
    finally:
        gc.enable()
    return statistics.median(values)


def _peak_tracemalloc_bytes(function) -> tuple[int, object]:
    function()
    gc.collect()
    tracemalloc.start()
    try:
        result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, result


def main() -> None:
    sparse_edges = _sparse_edges()
    sparse_graph = nx.Graph()
    sparse_graph.add_nodes_from(range(NODE_COUNT))
    sparse_graph.add_edges_from(sparse_edges)
    direct_components = connected_components(NODE_COUNT, sparse_edges)
    oracle_components = _normalized_networkx_components(sparse_graph)
    assert direct_components == oracle_components

    chain = _chain(NODE_COUNT)
    chain_edges = tuple((index, index + 1) for index in range(NODE_COUNT - 1))
    chain_graph = nx.Graph()
    chain_graph.add_nodes_from(range(NODE_COUNT))
    chain_graph.add_edges_from(chain_edges)
    source_atom = chain.atoms[SHORTEST_SOURCE]
    target_atom = chain.atoms[SHORTEST_TARGET]
    direct_shortest = chain.shortest_path_index(source_atom, target_atom)
    oracle_shortest = nx.shortest_path(
        chain_graph,
        SHORTEST_SOURCE,
        SHORTEST_TARGET,
    )
    assert direct_shortest == oracle_shortest

    complete_edges = tuple(itertools.combinations(range(COMPLETE_NODE_COUNT), 2))
    streaming = lambda: longest_simple_path(
        COMPLETE_NODE_COUNT,
        complete_edges,
        COMPLETE_SOURCE,
        COMPLETE_TARGET,
    )
    materializing = lambda: _materializing_longest_pair(
        COMPLETE_NODE_COUNT,
        complete_edges,
        COMPLETE_SOURCE,
        COMPLETE_TARGET,
    )
    streaming_result = streaming()
    materializing_result = materializing()
    assert streaming_result == materializing_result
    assert len(streaming_result) == COMPLETE_NODE_COUNT

    streaming_peak, traced_streaming_result = _peak_tracemalloc_bytes(streaming)
    materializing_peak, traced_materializing_result = _peak_tracemalloc_bytes(
        materializing
    )
    assert traced_streaming_result == traced_materializing_result == streaming_result

    median_us = {
        "connected_components_direct_512_sparse": _median_us(
            lambda: connected_components(NODE_COUNT, sparse_edges),
            loops=100,
        ),
        "connected_components_networkx_512_sparse": _median_us(
            lambda: _normalized_networkx_components(sparse_graph),
            loops=100,
        ),
        "connected_components_task_local_networkx_512_sparse": _median_us(
            lambda: _task_local_networkx_components(NODE_COUNT, sparse_edges),
            loops=100,
        ),
        "longest_simple_path_materializing_reference_k10_pair": _median_us(
            materializing
        ),
        "longest_simple_path_streaming_k10_pair": _median_us(streaming),
        "molecule_shortest_path_index_512_chain": _median_us(
            lambda: chain.shortest_path_index(source_atom, target_atom),
            loops=100,
        ),
        "networkx_shortest_path_512_chain_reused_graph": _median_us(
            lambda: nx.shortest_path(
                chain_graph,
                SHORTEST_SOURCE,
                SHORTEST_TARGET,
            ),
            loops=100,
        ),
    }
    results = {
        "correctness": {
            "components_equal": True,
            "longest_path_equal": True,
            "shortest_path_equal": True,
        },
        "median_us": median_us,
        "metadata": {
            "complete_graph_fixed_pair": [
                COMPLETE_NODE_COUNT,
                COMPLETE_SOURCE,
                COMPLETE_TARGET,
            ],
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "python": sys.version,
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "samples": SAMPLES,
            "shortest_path_chain_query": [
                NODE_COUNT,
                SHORTEST_SOURCE,
                SHORTEST_TARGET,
            ],
            "sparse_component_graph": {
                "components": len(direct_components),
                "edges": len(sparse_edges),
                "nodes": NODE_COUNT,
            },
            "worktree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
        },
        "peak_tracemalloc_bytes": {
            "longest_simple_path_materializing_reference_k10_pair": materializing_peak,
            "longest_simple_path_streaming_k10_pair": streaming_peak,
        },
        "ratios": {
            "components_direct_over_networkx_time": (
                median_us["connected_components_direct_512_sparse"]
                / median_us["connected_components_networkx_512_sparse"]
            ),
            "components_direct_over_task_local_networkx_time": (
                median_us["connected_components_direct_512_sparse"]
                / median_us[
                    "connected_components_task_local_networkx_512_sparse"
                ]
            ),
            "longest_streaming_over_materializing_peak_memory": (
                streaming_peak / materializing_peak
            ),
            "longest_streaming_over_materializing_time": (
                median_us["longest_simple_path_streaming_k10_pair"]
                / median_us[
                    "longest_simple_path_materializing_reference_k10_pair"
                ]
            ),
            "shortest_direct_over_networkx_time": (
                median_us["molecule_shortest_path_index_512_chain"]
                / median_us[
                    "networkx_shortest_path_512_chain_reused_graph"
                ]
            ),
        },
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
