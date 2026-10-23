"""Emit the frozen pre-Plan-19 graph-semantics snapshot as JSON.

This script intentionally exercises interfaces that Plan 19 removes.  It is
historical baseline evidence, not a compatibility example for the final API.
"""

from __future__ import annotations

import json

import networkx as nx

import hotpot as hp
from hotpot.cheminfo.AImodels.data_extract import extract_ring_attrs
from hotpot.cheminfo.AImodels.mca.site_detection import find_nucleophilic_sites
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.graph import graph_dfs_path, relevant_cycles
from hotpot.cheminfo.search import QueryAtom, Searcher, Substructure


def _molecule(edges: list[tuple[int, int]], atom_count: int) -> Molecule:
    molecule = Molecule()
    for _ in range(atom_count):
        molecule._create_atom(atomic_number=6)
    for first, second in edges:
        molecule._add_bond(first, second, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    return molecule


def _carbon_pair() -> Substructure:
    query = Substructure()
    query.add_atom(QueryAtom(atomic_number={6}, map_number=1))
    query.add_atom(QueryAtom(atomic_number={6}, map_number=2))
    query.add_bond(0, 1)
    return query


def main() -> None:
    cubane_edges = list(nx.cubical_graph().edges)
    cubane = _molecule(cubane_edges, 8)
    ring_data = extract_ring_attrs(cubane)
    path = _molecule([(0, 1), (1, 2), (1, 3)], 5)
    propane = hp.read_mol("CCC", fmt="smi")
    hits = Searcher(_carbon_pair()).search(propane)

    snapshot = {
        "cubane": {
            "cycle_basis": [
                list(ring.atoms_indices) for ring in cubane.cycle_basis_rings
            ],
            "relevant_cycles": relevant_cycles(cubane_edges),
            "ring_tensor": {
                key: value.tolist()
                for key, value in ring_data.items()
                if key
                in {
                    "mol_rings_nums",
                    "rings_node_index",
                    "rings_node_nums",
                    "mol_rings_node_nums",
                }
            },
        },
        "direct_topology": {
            "neighbors": [atom.idx for atom in path.atoms[1].neighbours],
            "incident_bonds": [
                [bond.a1idx, bond.a2idx] for bond in path.atoms[1].bonds
            ],
            "connectivity": path.atoms[1].connectivity,
            "shortest_path_0_3": [
                atom.idx
                for atom in path.shortest_path(path.atoms[0], path.atoms[3])
            ],
            "components": [
                [len(component.atoms), len(component.bonds)]
                for component in path.components
            ],
            "dfs_path": graph_dfs_path(
                path.graph,
                start_node=0,
                scope_nodes={1, 2, 3},
                max_deep=3,
            ),
        },
        "search": [
            {
                "atoms": sorted(hit.atom_indices),
                "mappings": [
                    list(mapping.items()) for mapping in hit.mappings
                ],
            }
            for hit in hits
        ],
        "mca_sites": {
            smiles: [
                [site.atom_index, site.site_type]
                for site in find_nucleophilic_sites(
                    hp.read_mol(smiles, fmt="smi")
                )
            ]
            for smiles in (
                "COC",
                "CC(=O)N",
                "NCCN",
                "NCCCN(C)",
                "[13CH3]N.CN",
            )
        },
    }
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
