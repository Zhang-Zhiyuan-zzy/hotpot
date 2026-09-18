"""Real-molecule integration coverage for the factual geometry package."""

from __future__ import annotations

import numpy as np

from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.core import Molecule


def _molecule(coordinates, bonds=(), atomic_numbers=None):
    mol = Molecule()
    if atomic_numbers is None:
        atomic_numbers = [6] * len(coordinates)
    for atomic_number, position in zip(atomic_numbers, coordinates):
        mol.create_atom(
            atomic_number=atomic_number,
            coordinates=position,
        )
    for first, second in bonds:
        mol.add_bond(first, second, bond_order=1.0)
    mol.refresh_atom_id()
    return mol


def test_ligand_scope_excludes_a_chelate_cycle_without_mutating_ring_caches():
    mol = _molecule(
        (
            (0.0, -1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 1.5, 0.0),
            (1.0, 1.5, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, -1.0),
            (0.0, 0.5, 1.0),
        ),
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6)),
        (30, 7, 6, 6, 7, 8, 8),
    )
    rings_cache = mol._rings
    ligand_rings_cache = mol._ligand_rings
    ligand_rings_signature = mol._ligand_rings_signature

    full_graph = geo.scan_bond_ring_relations(
        mol,
        ring_scope="full_graph",
        max_ring_size=8,
    )
    ligand_skeleton = geo.scan_bond_ring_relations(
        mol,
        ring_scope="ligand_skeleton",
        max_ring_size=8,
    )

    assert full_graph.piercing_pair_count == 1
    assert ligand_skeleton.selected_ring_count == 0
    assert ligand_skeleton.piercing_pair_count == 0
    assert mol._rings is rings_cache
    assert mol._ligand_rings is ligand_rings_cache
    assert mol._ligand_rings_signature is ligand_rings_signature


def test_metal_ligand_bond_can_pierce_a_real_ligand_ring():
    ring_points = tuple(
        (
            2.0 * np.cos(index * np.pi / 3.0),
            2.0 * np.sin(index * np.pi / 3.0),
            0.0,
        )
        for index in range(6)
    )
    mol = _molecule(
        ring_points + ((0.0, 0.0, -2.0), (0.0, 0.0, 2.0)),
        tuple((index, (index + 1) % 6) for index in range(6))
        + ((6, 7),),
        (6, 6, 6, 6, 6, 6, 30, 7),
    )
    metal_ligand_bond = mol.bond(6, 7)

    report = geo.scan_bond_ring_relations(
        mol,
        ring_scope="ligand_skeleton",
        max_ring_size=8,
    )

    assert report.piercing_pair_count == 1
    (finding,) = report.piercings
    assert finding.target.bond.source is metal_ligand_bond
    assert finding.relation.state is geo.PiercingState.PIERCES
