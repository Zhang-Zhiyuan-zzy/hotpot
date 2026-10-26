"""Measure the Plan 19 Molecule topology index on fixed inputs."""

from __future__ import annotations

import gc
import json
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns

from hotpot.cheminfo.core import Molecule


def _chain(atom_count: int) -> Molecule:
    molecule = Molecule()
    for _ in range(atom_count):
        molecule._create_atom(atomic_number=6)
    for index in range(atom_count - 1):
        molecule._add_bond(index, index + 1, bond_order=1.0)
    molecule._update_topology(clear_conformers=False)
    return molecule


def _median_us(function, *, samples: int = 21, loops: int = 1) -> float:
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


def _deep_size(value) -> int:
    seen: set[int] = set()

    def visit(current) -> int:
        identity = id(current)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(current)
        if isinstance(current, dict):
            return size + sum(visit(key) + visit(item) for key, item in current.items())
        if isinstance(current, (list, tuple, set, frozenset)):
            return size + sum(visit(item) for item in current)
        return size

    return visit(value)


def main() -> None:
    chain = _chain(512)
    results = {
        "metadata": {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "python": sys.version,
            "platform": platform.platform(),
            "samples": 21,
            "worktree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
        },
        "median_us": {
            "atom_idx_512": _median_us(lambda: chain.atoms[400].idx, loops=1000),
            "bond_lookup_512": _median_us(
                lambda: chain.bond(400, 401), loops=1000
            ),
            "connectivity_512": _median_us(
                lambda: chain.atoms[400].connectivity, loops=1000
            ),
            "copy_chain_512": _median_us(chain.copy),
            "neighbors_512": _median_us(
                lambda: chain.atoms[400].neighbours, loops=1000
            ),
            "refresh_chain_512": _median_us(
                lambda: chain._update_topology(clear_conformers=False)
            ),
        },
        "retained_topology_index_bytes_chain_512": _deep_size(
            chain._topology_index
        ),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
