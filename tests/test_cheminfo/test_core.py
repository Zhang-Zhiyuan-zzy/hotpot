"""Focused contracts retained from the former monolithic core smoke script."""

from copy import copy

import numpy as np
import pytest

import hotpot as hp


def test_molecule_has_3d_requires_distinct_atom_coordinates() -> None:
    molecule = hp.read_mol("CC", "smi")

    assert not molecule.has_3d

    molecule.coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        dtype=float,
    )

    assert molecule.has_3d


def test_molecule_copy_preserves_graph_without_sharing_atoms() -> None:
    molecule = hp.read_mol("CCN.CCO", "smi")
    cloned = copy(molecule)

    assert cloned is not molecule
    np.testing.assert_array_equal(cloned.link_matrix, molecule.link_matrix)
    np.testing.assert_array_equal(
        cloned.atom_matrix,
        molecule.atom_matrix,
    )
    assert all(
        cloned_atom is not source_atom
        for source_atom, cloned_atom in zip(molecule.atoms, cloned.atoms)
    )
    assert tuple(len(component.atoms) for component in cloned.components) == (3, 3)


def test_molecule_similarity_is_symmetric_and_identity_is_one() -> None:
    first = hp.read_mol("CCOP(=O)(OCC)OCC", "smi")
    second = hp.read_mol("c1ccccc1N", "smi")
    identical = hp.read_mol("CCOP(=O)(OCC)OCC", "smi")

    assert first.similarity(identical) == pytest.approx(1.0, abs=1.0e-12)
    assert first.similarity(second) == pytest.approx(
        second.similarity(first),
        abs=1.0e-12,
    )


def test_add_hydrogens_retains_expected_methanol_valence() -> None:
    molecule = hp.read_mol("CO", "smi")

    molecule.add_hydrogens(
        rm_polar_hs=False,
        rng=np.random.default_rng(17),
    )

    assert len(molecule.hydrogens) == 4
    assert molecule.atoms[1].explicit_hydrogens == 1
