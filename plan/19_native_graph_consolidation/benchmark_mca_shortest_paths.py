"""Benchmark the Plan 19 MCA all-pairs shortest-path cutover."""

from __future__ import annotations

import gc
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hotpot.cheminfo.AImodels.mca.featurizer import (
    _SPARSE_SHORTEST_PATH_THRESHOLD,
    _shortest_paths,
)


SIZES = (64, 128, 255, 256, 512)
SAMPLES = 21


def _molecule_like_adjacency(atom_count: int) -> np.ndarray:
    """Return one deterministic sparse connected unit-weight topology."""
    adjacency = np.zeros((atom_count, atom_count), dtype=np.int32)
    edges = [(node, (node - 1) // 2) for node in range(1, atom_count)]
    edges.extend(
        (node, node + 5)
        for node in range(0, atom_count - 5, 11)
    )
    for first, second in edges:
        adjacency[first, second] = 1
        adjacency[second, first] = 1
    return adjacency


def _dense_reference(adjacency: np.ndarray) -> np.ndarray:
    """Retain the exact pre-cutover NumPy Floyd-Warshall implementation."""
    distances = np.where(adjacency == 0, 510, adjacency).astype(np.int32)
    np.fill_diagonal(distances, 0)
    for intermediate in range(distances.shape[0]):
        distances = np.minimum(
            distances,
            distances[:, intermediate, None]
            + distances[None, intermediate, :],
        )
    distances[distances >= 510] = 510
    return distances


def _median_us(function) -> float:
    function()
    elapsed = []
    gc.disable()
    try:
        for _ in range(SAMPLES):
            start = perf_counter_ns()
            function()
            elapsed.append((perf_counter_ns() - start) / 1_000.0)
    finally:
        gc.enable()
    return statistics.median(elapsed)


def main() -> None:
    median_us = {}
    exact_parity = {}
    speedup = {}
    for atom_count in SIZES:
        adjacency = _molecule_like_adjacency(atom_count)
        reference = _dense_reference(adjacency)
        result = _shortest_paths(adjacency)
        exact = np.array_equal(reference, result)
        exact_parity[str(atom_count)] = exact
        if not exact:
            raise RuntimeError(
                f"shortest-path parity failed for {atom_count} atoms"
            )

        dense_us = _median_us(lambda: _dense_reference(adjacency))
        final_us = _median_us(lambda: _shortest_paths(adjacency))
        median_us[f"dense_reference_{atom_count}"] = dense_us
        median_us[f"final_{atom_count}"] = final_us
        speedup[str(atom_count)] = dense_us / final_us

    results = {
        "metadata": {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "samples": SAMPLES,
            "sparse_threshold": _SPARSE_SHORTEST_PATH_THRESHOLD,
            "worktree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
            "graph": "deterministic sparse binary tree with periodic chords",
        },
        "exact_parity": exact_parity,
        "median_us": median_us,
        "dense_over_final_speedup": speedup,
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
