"""Deterministic graph corpora for Relevant Cycle differential validation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import networkx as nx

Edge = Tuple[int, int]
GraphCase = Tuple[str, Tuple[Edge, ...]]


@dataclass(frozen=True)
class PubChemCase:
    """A pinned PubChem connectivity record used only to derive an edge graph."""

    cid: int
    title: str
    connectivity_smiles: str


# Retrieved from PubChem PUG REST on 2026-09-20. Connectivity SMILES deliberately
# omit stereochemistry because Relevant Cycles depend only on graph connectivity.
PUBCHEM_CASES = (
    PubChemCase(
        2519,
        "caffeine",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ),
    PubChemCase(
        5997,
        "cholesterol",
        "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    ),
    PubChemCase(
        12560,
        "erythromycin",
        "CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)"
        "OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)O)(C)O",
    ),
    PubChemCase(
        36314,
        "paclitaxel",
        "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)"
        "NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
    ),
    PubChemCase(
        14969,
        "vancomycin",
        "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=C4C=C5C=C3OC6=C(C=C(C=C6)C(C(C(=O)"
        "NC(C(=O)NC5C(=O)NC7C8=CC(=C(C=C8)O)C9=C(C=C(C=C9O)O)C(NC(=O)C(C(C1="
        "CC(=C(O4)C=C1)Cl)O)NC7=O)C(=O)O)CC(=O)N)NC(=O)C(CC(C)C)NC)O)Cl)CO)"
        "O)O)(C)N)O",
    ),
)


def handcrafted_graph_cases() -> Tuple[GraphCase, ...]:
    """Return small graphs that isolate important Relevant Cycle semantics."""
    return (
        ("triangle", ((0, 1), (1, 2), (2, 0))),
        (
            "square_with_diagonal",
            ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)),
        ),
        (
            "equal_path_theta",
            ((0, 1), (1, 4), (0, 2), (2, 4), (0, 3), (3, 4)),
        ),
        (
            "naphthalene_topology",
            (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 0),
                (4, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 5),
            ),
        ),
        ("complete_4", _integer_edges(nx.complete_graph(4))),
        (
            "disconnected_triangles",
            ((0, 1), (1, 2), (2, 0), (4, 5), (5, 6), (6, 4)),
        ),
        (
            "figure_eight",
            ((0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)),
        ),
    )


def _integer_edges(graph: nx.Graph) -> Tuple[Edge, ...]:
    integer_graph = nx.convert_node_labels_to_integers(
        graph,
        ordering="sorted",
    )
    return tuple(
        sorted(
            (min(node1, node2), max(node1, node2))
            for node1, node2 in integer_graph.edges
        )
    )


def named_graph_cases() -> Tuple[GraphCase, ...]:
    """Return deterministic named graphs with varied symmetric ring systems."""
    graphs = {
        "cubical": nx.cubical_graph(),
        "dodecahedral": nx.dodecahedral_graph(),
        "icosahedral": nx.icosahedral_graph(),
        "desargues": nx.desargues_graph(),
        "heawood": nx.heawood_graph(),
        "petersen": nx.petersen_graph(),
        "complete_8": nx.complete_graph(8),
        "complete_bipartite_5_5": nx.complete_bipartite_graph(5, 5),
        "hypercube_4": nx.hypercube_graph(4),
        "grid_6_6": nx.grid_2d_graph(6, 6),
        "circular_ladder_20": nx.circular_ladder_graph(20),
    }
    return tuple((name, _integer_edges(graph)) for name, graph in graphs.items())


def pubchem_graph_cases() -> Tuple[GraphCase, ...]:
    """Return pinned PubChem structures as integer edge lists."""
    from rdkit import Chem

    graph_cases = []
    for case in PUBCHEM_CASES:
        molecule = Chem.MolFromSmiles(case.connectivity_smiles)
        edges = tuple(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in molecule.GetBonds()
        )
        graph_cases.append((f"pubchem_{case.cid}_{case.title}", edges))
    return tuple(graph_cases)


def graph_atlas_cases() -> Iterator[GraphCase]:
    """Yield every cyclic simple graph in NetworkX's Graph Atlas."""
    for index, graph in enumerate(nx.graph_atlas_g()):
        cycle_rank = (
            graph.number_of_edges()
            - graph.number_of_nodes()
            + nx.number_connected_components(graph)
        )
        if cycle_rank <= 0:
            continue
        yield f"graph_atlas_{index:04d}", _integer_edges(graph)


def deterministic_random_graph_cases() -> Iterator[GraphCase]:
    """Yield 1,000 fixed G(n, p) graphs beyond Graph Atlas sizes."""
    random_generator = random.Random(20260920)
    for node_count in range(3, 13):
        upper_probability = min(0.55, 4.5 / (node_count - 1))
        for sample in range(100):
            edge_probability = random_generator.uniform(0.08, upper_probability)
            graph_seed = random_generator.randrange(1 << 30)
            graph = nx.gnp_random_graph(
                node_count,
                edge_probability,
                seed=graph_seed,
            )
            edges = list(graph.edges)
            random_generator.shuffle(edges)
            oriented_edges = tuple(
                edge if random_generator.randrange(2) else tuple(reversed(edge))
                for edge in edges
            )
            yield f"random_{node_count:02d}_{sample:03d}", oriented_edges


def golden_graph_cases() -> Dict[str, Tuple[Edge, ...]]:
    """Return the stable corpus represented by the checked-in RDL golden file."""
    return dict(
        handcrafted_graph_cases() + named_graph_cases() + pubchem_graph_cases()
    )
