"""Full-graph and ligand-skeleton ring behavior for coordination compounds."""

import pytest

import hotpot as hp

from .coordination_cases import (
    REPOSITORY_INPUTS,
    ethylenediamine_chelate,
    matching_indices,
    metal_bound_ligand_ring,
)


pytestmark = pytest.mark.smarts_core


def ring_atom_sets(rings):
    return {frozenset(atom.idx for atom in ring.atoms) for ring in rings}


def metal_ligand_pairs(molecule):
    return {
        frozenset((bond.a1idx, bond.a2idx))
        for bond in molecule.bonds
        if bond.is_metal_ligand_bond
    }


def test_chelate_is_a_full_graph_ring_but_not_a_ligand_ring():
    molecule = ethylenediamine_chelate()

    assert ring_atom_sets(molecule.rings) == {frozenset(range(5))}
    assert molecule.rings[0].has_metal
    assert matching_indices(molecule, "[C;r5]") == {2, 3}
    assert matching_indices(molecule, "[N;r5]") == {1, 4}
    assert matching_indices(molecule, "[M;r5]") == {0}

    original_bonds = metal_ligand_pairs(molecule)
    assert molecule.ligand_rings == []
    assert metal_ligand_pairs(molecule) == original_bonds

    # ``ligand_rings`` restores the graph; the public SMARTS profile therefore
    # continues to expose the full-graph chelate cycle.
    assert matching_indices(molecule, "[M;r5]") == {0}


def test_ligand_rings_preserve_a_covalent_ring_bound_to_a_metal():
    molecule = metal_bound_ligand_ring()
    covalent_ring = frozenset(range(1, 7))

    assert ring_atom_sets(molecule.rings) == {covalent_ring}
    assert ring_atom_sets(molecule.ligand_rings) == {covalent_ring}
    assert matching_indices(molecule, "[N;r6]") == {1}
    assert matching_indices(molecule, "[C;r6]") == set(range(2, 7))
    assert matching_indices(molecule, "[M;R]") == set()


def test_real_afoqui_full_graph_has_metal_cycles_and_ligand_view_does_not():
    molecule = hp.read_mol(REPOSITORY_INPUTS / "AFOQUI_clean.cif")
    original_bonds = metal_ligand_pairs(molecule)
    full_rings = molecule.rings
    ring_metals = {
        atom.idx
        for ring in full_rings
        if ring.has_metal
        for atom in ring.atoms
        if atom.is_metal
    }

    assert original_bonds
    assert ring_metals
    assert matching_indices(molecule, "[M;R]") == ring_metals
    assert all(not ring.has_metal for ring in molecule.ligand_rings)
    assert metal_ligand_pairs(molecule) == original_bonds
