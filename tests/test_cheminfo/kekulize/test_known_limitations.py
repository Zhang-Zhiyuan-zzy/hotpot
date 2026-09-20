"""Executable records of known legacy aromaticity limitations."""

from __future__ import annotations

import networkx as nx
import pytest

import hotpot as hp
from hotpot.cheminfo import kekulize


def _planar_theta_graph() -> hp.Molecule:
    paths = ((0, 2, 1), (0, 3, 4, 1), (0, 5, 6, 7, 1))
    edges = [edge for path in paths for edge in zip(path, path[1:])]
    graph = nx.Graph(edges)
    coordinates = nx.planar_layout(graph)
    mol = hp.Molecule()
    for atom_index in range(8):
        mol._create_atom(
            atomic_number=6,
            is_aromatic=False,
            implicit_hydrogens=0 if graph.degree[atom_index] == 3 else 1,
            coordinates=(*coordinates[atom_index], 0.0),
        )
    for first, second in edges:
        mol._add_bond(first, second, bond_order=1.0)
    mol._update_graph(clear_conformers=False)
    return mol


def _apply_in_order(mol: hp.Molecule, reverse: bool) -> frozenset[int]:
    rings = list(mol.rings)
    if reverse:
        rings.reverse()
    for ring in rings:
        kekulize.determine_ring_aromaticity(ring, inplace=True)
    return frozenset(atom.idx for atom in mol.atoms if atom.is_aromatic)


@pytest.mark.xfail(
    strict=True,
    reason="legacy per-ring writes make aromaticity depend on ring order",
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
    coordinates = nx.planar_layout(mol.graph)
    for atom in mol.atoms:
        atom.coordinates = (*coordinates[atom.idx], 0.0)
        atom.is_aromatic = False

    for ring in mol.rings:
        kekulize.determine_ring_aromaticity(ring, inplace=True)

    assert all(atom.is_aromatic for atom in mol.atoms)


@pytest.mark.xfail(
    strict=True,
    reason="legacy joint-ring grouping rejects rings sharing multiple edges",
)
def test_aromatic_joint_rings_accept_relevant_cycles_with_shared_paths() -> None:
    mol = _planar_theta_graph()
    for atom in mol.atoms:
        atom.is_aromatic = True

    assert mol.aromatic_joint_rings
