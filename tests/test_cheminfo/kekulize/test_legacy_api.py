"""Characterized chemistry through pure rules and complete Molecule operations.

Historical test names remain the frozen regression selection identities.
"""

from __future__ import annotations

import math

import networkx as nx

import hotpot as hp
from hotpot.cheminfo.operations.chemistry import _bonding_snapshot
from hotpot.cheminfo.rules.chemistry.bonding.aromaticity import _determine_ring_aromaticity
from hotpot.cheminfo.rules.chemistry.bonding.kekule import _calculate_aromatic_assignment
from hotpot.cheminfo.structure._commit import _commit_aromatic_assignment
from tests.test_cheminfo._stage5_fixtures import factual_state, supplied_aromatic_flags


def _layout(mol: hp.Molecule):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(mol.atoms)))
    graph.add_edges_from((bond.a1idx, bond.a2idx) for bond in mol.bonds)
    return nx.planar_layout(graph)


def _planar_benzene() -> hp.Molecule:
    mol = hp.Molecule()
    for row in range(6):
        mol.create_atom(atomic_number=6,
            coordinates=(math.cos(row * math.pi / 3), math.sin(row * math.pi / 3), 0.))
    for row in range(6):
        mol.add_bond(row, (row + 1) % 6, 1.)
    # Match the old all-single pre-assignment input: inferred H2, not a
    # fresh aromatic parse carrying a protected supplied H1 declaration.
    mol.calc_implicit_hydrogens()
    return mol


def _bond_orders(mol: hp.Molecule) -> tuple[float, ...]:
    return tuple(float(bond.bond_order) for bond in mol.bonds)


def _planar_molecule(smiles: str) -> hp.Molecule:
    mol = hp.read_mol(smiles, 'smi')
    coordinates = _layout(mol)
    for atom in mol.atoms:
        atom.coordinates = (*coordinates[atom.idx], 0.)
    return supplied_aromatic_flags(mol, (False,) * len(mol.atoms))


def test_ring_facade_matches_isolated_aromaticity_function() -> None:
    direct, facade = _planar_benzene(), _planar_benzene()
    before = factual_state(direct)
    snapshot = _bonding_snapshot(direct)
    result = _determine_ring_aromaticity(snapshot, snapshot.rings[0], True, snapshot.aromatic)
    facade.calc_atom_valence(assign_aromatic=True)
    assert result is True
    assert factual_state(direct) == before
    assert all(atom.is_aromatic for atom in facade.atoms)


def test_ring_facade_matches_isolated_legacy_assignment() -> None:
    direct, facade = _planar_benzene(), _planar_benzene()
    assert [atom.implicit_hydrogens for atom in direct.atoms] == [2] * 6
    before = factual_state(direct)
    proposal = _calculate_aromatic_assignment(_bonding_snapshot(direct), kekule=True)
    assert factual_state(direct) == before
    facade.determine_rings_aromatic()
    assert proposal.bond_orders == _bond_orders(facade)
    assert sum(order == 2. for order in proposal.bond_orders) == 3
    assert [atom.implicit_hydrogens for atom in facade.atoms] == [1] * 6


def test_molecule_facade_matches_isolated_legacy_workflow() -> None:
    direct, facade = _planar_benzene(), _planar_benzene()
    snapshot = _bonding_snapshot(direct)
    _commit_aromatic_assignment(direct, _calculate_aromatic_assignment(snapshot, kekule=True), snapshot=snapshot)
    facade.determine_rings_aromatic()
    assert _bond_orders(direct) == _bond_orders(facade)
    assert [atom.is_aromatic for atom in direct.atoms] == [atom.is_aromatic for atom in facade.atoms]


def test_legacy_aromaticity_golden_cases() -> None:
    for smiles, expected in {'C1=CC=CC=C1': True, 'C1CCC1': False,
                             'C1=COC=C1': True, 'C1=CC=NC=C1': True}.items():
        mol = _planar_molecule(smiles)
        snapshot = _bonding_snapshot(mol)
        assert _determine_ring_aromaticity(snapshot, snapshot.rings[0], True, snapshot.aromatic) is expected
        mol.calc_atom_valence(assign_aromatic=True)
        assert all(atom.is_aromatic is expected for atom in mol.atoms)


def test_legacy_metal_ring_is_not_aromatic() -> None:
    mol = hp.Molecule()
    for number, coordinates in ((6, (0., 0., 0.)), (6, (1., 0., 0.)), (26, (.5, 1., 0.))):
        mol.create_atom(atomic_number=number, coordinates=coordinates)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        mol.add_bond(first, second, 1.)
    snapshot = _bonding_snapshot(mol, ligand_skeleton=False)
    assert _determine_ring_aromaticity(snapshot, snapshot.rings[0], True, snapshot.aromatic) is False
    mol.determine_rings_aromatic()
    assert not any(atom.is_aromatic for atom in mol.atoms)


def test_valence_aromaticity_gate_is_owned_by_kekulize_package() -> None:
    skipped, perceived = _planar_benzene(), _planar_benzene()
    skipped.calc_atom_valence(assign_aromatic=False)
    perceived.calc_atom_valence(assign_aromatic=True)
    assert not any(atom.is_aromatic for atom in skipped.atoms)
    assert all(atom.is_aromatic for atom in perceived.atoms)
