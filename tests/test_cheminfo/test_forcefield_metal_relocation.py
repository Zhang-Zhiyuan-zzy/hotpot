from __future__ import annotations

import numpy as np

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.forcefields import coordination
from hotpot.cheminfo.forcefields.settings import (
    _METAL_RELOCATION_MAX_CANDIDATES,
)


def _unbound_metal_molecule():
    mol = Molecule()
    mol.create_atom(atomic_number=63, coordinates=(0.0, 0.0, 0.0))
    mol.create_atom(atomic_number=8, coordinates=(2.0, 0.0, 0.0))
    mol.create_atom(atomic_number=6, coordinates=(3.3, 0.0, 0.0))
    coordination_bond = mol.add_bond(0, 1, bond_order=1.0)
    mol.add_bond(1, 2, bond_order=1.0)
    mol.hide_bonds(coordination_bond, clear_conformers=False)
    return mol, mol.atoms[0], coordination_bond


def test_relocate_unbound_metal_is_deterministic_and_moves_only_metal():
    mol, metal, pending_bond = _unbound_metal_molecule()
    ligand_coordinates = mol.coordinates[1:].copy()

    first = coordination._relocate_unbound_metal(mol, metal, (pending_bond,))
    first_coordinates = np.asarray(metal.coordinates, dtype=float).copy()
    metal.coordinates = first.original_coordinates
    second = coordination._relocate_unbound_metal(mol, metal, (pending_bond,))

    assert first.status == "relocated"
    assert first.moved
    assert first.safe_donor_indices == (1,)
    assert first.selected_coordinates == second.selected_coordinates
    assert first.candidates_evaluated == second.candidates_evaluated
    assert np.array_equal(mol.coordinates[1:], ligand_coordinates)
    assert np.allclose(first_coordinates, second.selected_coordinates)


def test_relocate_unbound_metal_reports_infeasible_without_mutation(monkeypatch):
    mol, metal, pending_bond = _unbound_metal_molecule()
    initial_coordinates = mol.coordinates.copy()
    monkeypatch.setattr(
        coordination,
        "_evaluate_metal_candidate",
        lambda *args: None,
    )

    result = coordination._relocate_unbound_metal(mol, metal, (pending_bond,))

    assert result.status == "infeasible"
    assert not result.moved
    assert result.selected_coordinates is None
    assert 0 < result.candidates_evaluated <= _METAL_RELOCATION_MAX_CANDIDATES
    assert np.array_equal(mol.coordinates, initial_coordinates)


def test_relocate_unbound_metal_candidate_search_is_bounded(monkeypatch):
    mol, metal, pending_bond = _unbound_metal_molecule()
    evaluated_coordinates = []

    def reject_candidate(*args):
        evaluated_coordinates.append(tuple(float(value) for value in args[1]))
        return None

    monkeypatch.setattr(
        coordination,
        "_evaluate_metal_candidate",
        reject_candidate,
    )

    coordination._relocate_unbound_metal(mol, metal, (pending_bond,))

    assert 0 < len(evaluated_coordinates) <= _METAL_RELOCATION_MAX_CANDIDATES
    assert len(set(evaluated_coordinates)) == len(evaluated_coordinates)


def test_relocate_unbound_metal_uses_declared_candidate_order(monkeypatch):
    mol, metal, pending_bond = _unbound_metal_molecule()
    candidate_coordinates = tuple(
        np.array((float(index), 1.0, 0.0)) for index in range(4)
    )
    candidates = (
        coordination._MetalPlacementCandidate(
            coordinates=tuple(candidate_coordinates[0]),
            safe_donor_indices=(1,),
            minimum_normalized_clearance=9.0,
            coordination_distance_deviation=0.0,
            sequence_index=0,
        ),
        coordination._MetalPlacementCandidate(
            coordinates=tuple(candidate_coordinates[1]),
            safe_donor_indices=(1, 2),
            minimum_normalized_clearance=1.0,
            coordination_distance_deviation=0.0,
            sequence_index=1,
        ),
        coordination._MetalPlacementCandidate(
            coordinates=tuple(candidate_coordinates[2]),
            safe_donor_indices=(1, 2),
            minimum_normalized_clearance=2.0,
            coordination_distance_deviation=0.3,
            sequence_index=2,
        ),
        coordination._MetalPlacementCandidate(
            coordinates=tuple(candidate_coordinates[3]),
            safe_donor_indices=(1, 2),
            minimum_normalized_clearance=2.0,
            coordination_distance_deviation=0.1,
            sequence_index=3,
        ),
    )
    monkeypatch.setattr(
        coordination,
        "_iter_metal_candidate_coordinates",
        lambda *args: iter(candidate_coordinates),
    )
    monkeypatch.setattr(
        coordination,
        "_evaluate_metal_candidate",
        lambda context, coordinates, sequence_index: candidates[sequence_index],
    )

    result = coordination._relocate_unbound_metal(mol, metal, (pending_bond,))

    assert result.selected_coordinates == tuple(candidate_coordinates[3])
    assert tuple(metal.coordinates) == tuple(candidate_coordinates[3])


def test_relocate_unbound_metal_requires_one_geometrically_safe_donor(monkeypatch):
    mol, metal, pending_bond = _unbound_metal_molecule()
    initial_coordinates = tuple(metal.coordinates)
    monkeypatch.setattr(
        coordination,
        "_segment_pierces_any_cycle",
        lambda *args: True,
    )

    result = coordination._relocate_unbound_metal(mol, metal, (pending_bond,))

    assert result.status == "infeasible"
    assert tuple(metal.coordinates) == initial_coordinates
