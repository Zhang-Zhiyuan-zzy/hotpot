"""Regression tests for the isolated legacy aromaticity implementation."""

from __future__ import annotations

import networkx as nx

import hotpot as hp
from hotpot.cheminfo import kekulize


def _planar_benzene() -> hp.Molecule:
    mol = hp.read_mol("c1ccccc1", "smi")
    coordinates = nx.planar_layout(mol.graph)
    for atom in mol.atoms:
        atom.coordinates = (*coordinates[atom.idx], 0.0)
        atom.is_aromatic = False
    for bond in mol.bonds:
        bond.bond_order = 1.0
    return mol


def _bond_orders(mol: hp.Molecule) -> tuple[float, ...]:
    return tuple(float(bond.bond_order) for bond in mol.bonds)


def _planar_molecule(smiles: str) -> hp.Molecule:
    mol = hp.read_mol(smiles, "smi")
    coordinates = nx.planar_layout(mol.graph)
    for atom in mol.atoms:
        atom.coordinates = (*coordinates[atom.idx], 0.0)
        atom.is_aromatic = False
    return mol


def test_ring_facade_matches_isolated_aromaticity_function() -> None:
    direct = _planar_benzene()
    facade = _planar_benzene()

    direct_result = kekulize.determine_ring_aromaticity(
        direct.rings[0],
        inplace=True,
    )
    facade_result = facade.rings[0].determine_aromatic(inplace=True)

    assert direct_result is facade_result is True
    assert [atom.is_aromatic for atom in direct.atoms] == [
        atom.is_aromatic for atom in facade.atoms
    ]


def test_ring_facade_matches_isolated_legacy_assignment() -> None:
    direct = _planar_benzene()
    facade = _planar_benzene()

    kekulize.kekulize_ring(direct.rings[0])
    facade.rings[0].kekulize()

    assert _bond_orders(direct) == _bond_orders(facade)
    assert sum(order == 2.0 for order in _bond_orders(direct)) == 3


def test_molecule_facade_matches_isolated_legacy_workflow() -> None:
    direct = _planar_benzene()
    facade = _planar_benzene()

    kekulize.kekulize_molecule_rings(direct)
    facade.determine_rings_aromatic()

    assert _bond_orders(direct) == _bond_orders(facade)
    assert [atom.is_aromatic for atom in direct.atoms] == [
        atom.is_aromatic for atom in facade.atoms
    ]


def test_legacy_aromaticity_golden_cases() -> None:
    expected = {
        "C1=CC=CC=C1": True,
        "C1CCC1": False,
        "C1=COC=C1": True,
        "C1=CC=NC=C1": True,
    }

    for smiles, expected_aromaticity in expected.items():
        mol = _planar_molecule(smiles)
        result = kekulize.determine_ring_aromaticity(
            mol.rings[0],
            inplace=True,
        )
        assert result is expected_aromaticity
        assert all(atom.is_aromatic is expected_aromaticity for atom in mol.atoms)


def test_legacy_metal_ring_is_not_aromatic() -> None:
    mol = hp.Molecule()
    for atomic_number, coordinates in (
        (6, (0.0, 0.0, 0.0)),
        (6, (1.0, 0.0, 0.0)),
        (26, (0.5, 1.0, 0.0)),
    ):
        mol.create_atom(
            atomic_number=atomic_number,
            coordinates=coordinates,
            is_aromatic=False,
        )
    mol.add_bond(0, 1, bond_order=1.0)
    mol.add_bond(1, 2, bond_order=1.0)
    mol.add_bond(2, 0, bond_order=1.0)

    assert kekulize.determine_ring_aromaticity(
        mol.rings[0],
        inplace=True,
    ) is False
    assert not any(atom.is_aromatic for atom in mol.atoms)


def test_valence_aromaticity_gate_is_owned_by_kekulize_package() -> None:
    skipped = _planar_benzene()
    perceived = _planar_benzene()

    kekulize.perceive_ligand_ring_aromaticity(skipped, force=False)
    kekulize.perceive_ligand_ring_aromaticity(perceived, force=True)

    assert not any(atom.is_aromatic for atom in skipped.atoms)
    assert all(atom.is_aromatic for atom in perceived.atoms)
