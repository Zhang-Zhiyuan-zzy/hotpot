"""The approved complete numeric projection has one public property."""

import numpy as np
import pytest

from hotpot import Atom, Molecule


def test_complete_matrix_preserves_raw_order_and_always_returns_float64():
    mol = Molecule()
    rows = np.arange(38, dtype=np.int32).reshape(2, 19)
    for row in rows:
        Atom(mol, attrs_array=row, update_electron_config=False)
    matrix = mol.atom_matrix
    assert matrix.shape == (2, 19) and matrix.dtype == np.float64
    np.testing.assert_array_equal(matrix, rows)
    assert all(not np.shares_memory(matrix, atom.attrs) for atom in mol.atoms)
    matrix[:] = -1
    np.testing.assert_array_equal(mol.atom_matrix, rows)


def test_empty_matrix_has_the_declared_complete_shape_and_property_permissions():
    mol = Molecule()
    assert mol.atom_matrix.shape == (0, 19)
    assert mol.atom_matrix.dtype == np.float64
    with pytest.raises(AttributeError):
        mol.atom_matrix = np.zeros((0, 19))
    assert isinstance(Molecule.atom_matrix, property)
    assert not hasattr(Molecule, 'atom_attr_matrix')


def test_matrix_tracks_membership_order_without_exporting_private_uids():
    mol = Molecule()
    first = mol.create_atom(atomic_number=6, id=91)
    second = mol.create_atom(atomic_number=8, id=95)
    historical = mol.atom_matrix
    mol.remove_atom(first)
    second.id = 200
    current = mol.atom_matrix
    assert current.shape == (1, 19)
    assert current[0, 0] == 8 and current[0, 15] == 200
    assert historical[:, 15].tolist() == [91, 95]
