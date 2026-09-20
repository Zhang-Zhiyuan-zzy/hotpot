"""SMARTS ring primitives backed by the primary Relevant Cycle family."""

from __future__ import annotations

import random

import networkx as nx
import pytest
from rdkit import Chem

import hotpot as hp


pytestmark = pytest.mark.smarts_core


def _cubane(edge_seed: int) -> hp.Molecule:
    edges = list(nx.cubical_graph().edges)
    random.Random(edge_seed).shuffle(edges)
    mol = hp.Molecule()
    for _ in range(8):
        mol._create_atom(atomic_number=6)
    for first, second in edges:
        mol._add_bond(first, second, bond_order=1.0)
    mol._update_graph(clear_conformers=False)
    return mol


def _matching_indices(mol: hp.Molecule, smarts: str) -> set[int]:
    query = hp.Substructure.from_smarts(smarts)
    return {
        next(iter(hit.atom_indices))
        for hit in hp.Searcher(query).search(mol)
    }


def test_cubane_numeric_ring_primitives_use_relevant_cycles() -> None:
    mol = _cubane(0)

    assert _matching_indices(mol, "[C;R]") == set(range(8))
    assert _matching_indices(mol, "[C;R0]") == set()
    assert _matching_indices(mol, "[C;R2]") == set()
    assert _matching_indices(mol, "[C;R3]") == set(range(8))
    assert _matching_indices(mol, "[C;r4]") == set(range(8))
    assert _matching_indices(mol, "[C;r6]") == set()


def test_numeric_ring_primitives_are_invariant_to_edge_order() -> None:
    first = _cubane(0)
    second = _cubane(31)

    for smarts in ("[C;R]", "[C;R0]", "[C;R2]", "[C;R3]", "[C;r4]", "[C;r6]"):
        assert _matching_indices(first, smarts) == _matching_indices(second, smarts)


def test_cubane_numeric_ring_primitives_match_rdkit() -> None:
    hotpot_mol = _cubane(0)
    rdkit_mol = Chem.MolFromSmiles("C12C3C4C1C1C2C3C41")

    for smarts in ("[C;R]", "[C;R0]", "[C;R2]", "[C;R3]", "[C;r4]", "[C;r6]"):
        query = Chem.MolFromSmarts(smarts)
        rdkit_indices = {
            atom_index
            for match in rdkit_mol.GetSubstructMatches(query, uniquify=True)
            for atom_index in match
        }
        assert _matching_indices(hotpot_mol, smarts) == rdkit_indices
