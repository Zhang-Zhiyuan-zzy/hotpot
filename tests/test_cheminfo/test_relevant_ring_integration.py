"""Core integration tests for Relevant Cycles as the primary ring family."""

from __future__ import annotations

import random

import networkx as nx
import pytest

from hotpot.cheminfo import forcefields
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.AImodels.data_extract import extract_ring_attrs
from hotpot.cheminfo.graph import RelevantCycleLimitExceeded


def _molecule_from_edges(
        edges: list[tuple[int, int]],
        atom_count: int,
) -> Molecule:
    mol = Molecule()
    for _ in range(atom_count):
        mol._create_atom(atomic_number=6)
    for first, second in edges:
        mol._add_bond(first, second, bond_order=1.0)
    mol._update_graph(clear_conformers=False)
    return mol


def _ring_keys(mol: Molecule, attribute: str) -> tuple[frozenset[int], ...]:
    rings = getattr(mol, attribute)
    return tuple(sorted(
        (frozenset(atom.idx for atom in ring.atoms) for ring in rings),
        key=lambda ring: (len(ring), tuple(sorted(ring))),
    ))


def _cubane_edges() -> list[tuple[int, int]]:
    return list(nx.cubical_graph().edges)


def test_primary_and_legacy_ring_families_are_explicit() -> None:
    mol = _molecule_from_edges(_cubane_edges(), atom_count=8)

    assert len(mol.rings) == 6
    assert {len(ring) for ring in mol.rings} == {4}
    assert len(mol.cycle_basis_rings) == 5


def test_relevant_rings_are_invariant_to_edge_insertion_order() -> None:
    edges = _cubane_edges()
    shuffled_edges = list(edges)
    random.Random(13).shuffle(shuffled_edges)

    first = _molecule_from_edges(edges, atom_count=8)
    second = _molecule_from_edges(shuffled_edges, atom_count=8)

    assert _ring_keys(first, "rings") == _ring_keys(second, "rings")


def test_relevant_ring_scope_forwards_native_limits() -> None:
    mol = _molecule_from_edges(_cubane_edges(), atom_count=8)

    assert mol.rings_for_scope("full_graph", max_size=3) == []
    assert len(mol.rings_for_scope("full_graph", max_size=4)) == 6
    with pytest.raises(RelevantCycleLimitExceeded):
        mol.rings_for_scope("full_graph", max_cycles=5)


def test_relevant_ring_cache_is_invalidated_by_topology_update() -> None:
    mol = _molecule_from_edges([(0, 1), (1, 2), (2, 0)], atom_count=4)

    assert len(mol.rings_for_scope("full_graph")) == 1
    assert mol._relevant_cycle_indices_cache

    mol.add_bond(2, 3, bond_order=1.0)

    assert mol._relevant_cycle_indices_cache == {}
    assert len(mol.rings_for_scope("full_graph")) == 1


def test_existing_ml_ring_features_keep_the_legacy_cycle_basis() -> None:
    mol = _molecule_from_edges(_cubane_edges(), atom_count=8)

    data = extract_ring_attrs(mol)

    assert len(mol.rings) == 6
    assert data["mol_rings_nums"].tolist() == [5]
    assert data["rings_node_nums"].tolist() == [4, 4, 6, 4, 4]


def test_forcefield_ring_opening_ignores_the_legacy_cycle_basis(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    edges = [
        (0, 1),
        (0, 2), (2, 3), (3, 1),
        (0, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 1),
        (11, 12),
    ]
    coordinates = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
        (0.0, -2.0, 0.0), (-1.0, -3.0, 0.0),
        (-2.0, -3.0, 0.0), (-3.0, -2.0, 0.0),
        (-3.0, -1.0, 0.0), (-2.0, -0.5, 0.0),
        (-1.0, -0.5, 0.0),
        (0.4, -0.1, 0.0), (0.6, -0.1, 0.0),
    ]
    mol = Molecule()
    for atom_coordinates in coordinates:
        mol._create_atom(
            atomic_number=6,
            coordinates=atom_coordinates,
        )
    for first, second in edges:
        mol._add_bond(first, second, bond_order=1.0)
    mol._update_graph(clear_conformers=False)

    small_ring = next(ring for ring in mol.rings if len(ring) == 4)
    shared_edge = mol.bond(0, 1)

    def reject_legacy_cycle_basis_for_scope(
            _mol: Molecule,
            _ring_scope: str,
    ) -> None:
        raise AssertionError("force-field ring opening read the legacy cycle basis")

    def reject_legacy_cycle_basis_property(_mol: Molecule) -> None:
        raise AssertionError("force-field ring opening read the legacy cycle basis")

    monkeypatch.setattr(
        Molecule,
        "cycle_basis_rings_for_scope",
        reject_legacy_cycle_basis_for_scope,
    )
    monkeypatch.setattr(
        Molecule,
        "cycle_basis_rings",
        property(reject_legacy_cycle_basis_property),
    )

    selected_edge = forcefields._select_ring_opening_edge(
        mol,
        small_ring,
        mol.bond(11, 12),
        ring_scope="full_graph",
    )

    assert {len(ring) for ring in mol.rings} == {4, 9}
    assert selected_edge is not None
    assert {selected_edge.a1idx, selected_edge.a2idx} != {
        shared_edge.a1idx,
        shared_edge.a2idx,
    }
