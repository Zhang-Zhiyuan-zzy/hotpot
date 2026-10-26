"""Measure the pre-Plan-19 topology paths on fixed deterministic inputs."""

from __future__ import annotations

import gc
import json
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns

import networkx as nx

import hotpot as hp
from hotpot.cheminfo.AImodels.mca.site_detection import find_nucleophilic_sites
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.graph import relevant_cycles
from hotpot.cheminfo.search import QueryAtom, Searcher, Substructure


def _molecule(edges: list[tuple[int, int]], atom_count: int) -> Molecule:
    molecule = Molecule()
    for _ in range(atom_count):
        molecule._create_atom(atomic_number=6)
    for first, second in edges:
        molecule._add_bond(first, second, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    return molecule


def _chain(atom_count: int) -> Molecule:
    return _molecule(
        [(index, index + 1) for index in range(atom_count - 1)],
        atom_count,
    )


def _carbon_pair_searcher() -> Searcher:
    query = Substructure()
    query.add_atom(QueryAtom(atomic_number={6}))
    query.add_atom(QueryAtom(atomic_number={6}))
    query.add_bond(0, 1)
    return Searcher(query)


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
    cube_edges = list(nx.cubical_graph().edges)
    cube = _molecule(cube_edges, 8)
    search_target = hp.read_mol("CCCCCCCCCCCC", fmt="smi")
    searcher = _carbon_pair_searcher()
    mca_target = hp.read_mol("NCCCN(C)", fmt="smi")

    duplicate_state = [
        chain._graph.__dict__,
        [atom._neighbours for atom in chain.atoms],
        [atom._bonds for atom in chain.atoms],
    ]
    results = {
        "metadata": {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "python": sys.version,
            "platform": platform.platform(),
            "samples": 21,
        },
        "median_us": {
            "atom_idx_512": _median_us(lambda: chain.atoms[400].idx, loops=1000),
            "connectivity_512": _median_us(
                lambda: chain.atoms[400].connectivity,
                loops=1000,
            ),
            "neighbors_512": _median_us(
                lambda: chain.atoms[400].neighbours,
                loops=1000,
            ),
            "bond_lookup_512": _median_us(
                lambda: chain.bond(400, 401),
                loops=1000,
            ),
            "refresh_chain_512": _median_us(
                lambda: chain._update_graph(clear_conformers=False)
            ),
            "copy_chain_512": _median_us(chain.copy),
            "legacy_cycle_basis_cube": _median_us(
                lambda: cube._uncached_cycle_basis_rings()
            ),
            "relevant_cycles_cube": _median_us(
                lambda: relevant_cycles(cube_edges)
            ),
            "search_12_carbon_pair": _median_us(
                lambda: searcher.search(search_target)
            ),
            "mca_sites": _median_us(lambda: find_nucleophilic_sites(mca_target)),
        },
        "retained_duplicate_topology_bytes_chain_512": _deep_size(
            duplicate_state
        ),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
