"""Broad small-graph differential validation against an exhaustive oracle."""

import networkx as nx

from hotpot.cheminfo.graph import relevant_cycles

from .exhaustive_oracle import exhaustive_relevant_cycles
from .fixtures import normalized_edges


def test_all_cyclic_graph_atlas_graphs_match_exhaustive_definition():
    compared = 0
    for graph in nx.graph_atlas_g():
        cycle_rank = (
            graph.number_of_edges()
            - graph.number_of_nodes()
            + nx.number_connected_components(graph)
        )
        if cycle_rank <= 0:
            continue
        edges = normalized_edges(tuple(graph.edges))
        assert relevant_cycles(edges, max_cycles=None) == exhaustive_relevant_cycles(edges)
        compared += 1

    assert compared == 1_173
