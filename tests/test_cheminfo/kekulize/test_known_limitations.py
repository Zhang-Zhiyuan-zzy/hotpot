"""Executable records of known legacy aromaticity limitations."""

from __future__ import annotations

import networkx as nx
import pytest
from dataclasses import replace

import hotpot as hp
from hotpot.cheminfo.operations.chemistry import _bonding_snapshot
from hotpot.cheminfo.rules.chemistry.bonding.kekule import _calculate_aromatic_assignment
from tests.test_cheminfo._stage5_fixtures import supplied_aromatic_flags


def _planar_theta_graph() -> hp.Molecule:
    paths = ((0, 2, 1), (0, 3, 4, 1), (0, 5, 6, 7, 1))
    edges = [edge for path in paths for edge in zip(path, path[1:])]
    graph = nx.Graph(edges)
    coordinates = nx.planar_layout(graph)
    mol = hp.Molecule()
    for atom_index in range(8):
        mol._create_atom(
            atomic_number=6,
            implicit_hydrogens=0 if graph.degree[atom_index] == 3 else 1,
            coordinates=(*coordinates[atom_index], 0.0),
        )
    for first, second in edges:
        mol._add_bond(first, second, bond_order=1.0)
    mol._update_topology(clear_conformers=False)
    return mol


def _apply_in_order(mol: hp.Molecule, reverse: bool) -> frozenset[int]:
    snapshot = _bonding_snapshot(mol)
    rings = list(snapshot.rings)
    if reverse:
        rings.reverse()
    assignment = _calculate_aromatic_assignment(replace(snapshot, rings=tuple(rings)), kekule=False)
    return frozenset(row for row, flag in enumerate(assignment.atom_aromatic) if flag)


@pytest.mark.xfail(
    strict=True,
    reason="the retained per-ring assignment makes aromaticity depend on ring order",
)
def test_aromaticity_is_independent_of_relevant_cycle_order() -> None:
    forward = _planar_theta_graph()
    reverse = _planar_theta_graph()

    assert _apply_in_order(forward, False) == _apply_in_order(reverse, True)


@pytest.mark.xfail(
    strict=True,
    reason="legacy per-ring Huckel counting does not perceive fused azulene globally",
)
def test_azulene_is_perceived_without_imported_aromatic_flags() -> None:
    mol = hp.read_mol("C1=CC=C2C=CC=C2C=C1", "smi")
    graph = nx.Graph()
    graph.add_nodes_from(range(len(mol.atoms)))
    graph.add_edges_from((bond.a1idx, bond.a2idx) for bond in mol.bonds)
    coordinates = nx.planar_layout(graph)
    for atom in mol.atoms:
        atom.coordinates = (*coordinates[atom.idx], 0.0)
    snapshot = _bonding_snapshot(mol)
    assignment = _calculate_aromatic_assignment(
        replace(snapshot, aromatic=(False,) * len(mol.atoms)), kekule=False)
    assert all(assignment.atom_aromatic)


@pytest.mark.xfail(
    strict=True,
    reason="legacy joint-ring grouping rejects rings sharing multiple edges",
)
def test_aromatic_joint_rings_accept_relevant_cycles_with_shared_paths() -> None:
    mol = _planar_theta_graph()
    mol = supplied_aromatic_flags(mol, (True,) * len(mol.atoms))
    assert mol.aromatic_joint_rings
