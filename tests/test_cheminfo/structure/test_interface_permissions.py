"""Approved public batch permissions, selection and complete failure semantics."""

import numpy as np
import pytest

from hotpot import Atom, Molecule
from hotpot.cheminfo.structure._dependencies import _dependency_state


def molecule():
    mol = Molecule()
    for number in (6, 7, 8):
        mol.create_atom(atomic_number=number)
    mol.coordinates = np.arange(9.0).reshape(3, 3)
    return mol


def state(mol):
    matrix = mol.atom_matrix
    status, seeds = _dependency_state(mol._atom_table)
    return (tuple(mol.atoms), tuple(atom._uid for atom in mol.atoms),
            matrix.tobytes(), mol._atom_table.presence().tobytes(),
            mol._atom_table.sources().tobytes(), status.tobytes(), seeds)


def unchanged(mol, before):
    assert state(mol) == before


def test_batch_all_atoms_and_scalar_broadcast():
    mol = molecule()
    assert mol.updata_atoms(formal_charge=-2, id=[101, 102, 103]) is None
    assert [atom.formal_charge for atom in mol.atoms] == [-2, -2, -2]
    assert [atom.id for atom in mol.atoms] == [101, 102, 103]


def test_batch_preserves_handle_selection_order():
    mol = molecule()
    first, untouched, last = mol.atoms
    assert mol.updata_atoms(atoms=[last, first], formal_charge=[2, -1], id=[31, 11]) is None
    assert (first.formal_charge, untouched.formal_charge, last.formal_charge) == (-1, 0, 2)
    assert (first.id, last.id) == (11, 31)


def test_batch_coordinates_broadcast_and_agreeing_aliases():
    mol = molecule()
    assert mol.updata_atoms(coordinates=(1, 2, 3), x=1) is None
    np.testing.assert_array_equal(mol.coordinates, [[1, 2, 3]] * 3)
    assert mol.updata_atoms(atoms=mol.atoms[::-1], coordinates=np.arange(9.0).reshape(3, 3)) is None
    np.testing.assert_array_equal(mol.coordinates, np.arange(9.0).reshape(3, 3)[::-1])


def test_batch_symbol_and_number_aliases_agree():
    mol = molecule()
    assert mol.updata_atoms(atoms=[mol.atoms[2], mol.atoms[0]], symbol=['N', 'O'], atomic_number=[7, 8]) is None
    assert [atom.atomic_number for atom in mol.atoms] == [8, 7, 7]


@pytest.mark.parametrize('columns', [
    {'formal_charge': [0, 1]},
    {'formal_charge': 2, 'coordinates': np.zeros((2, 3))},
    {'formal_charge': 2, 'coordinates': (1, 2, 3), 'x': [1, 1, 9]},
    {'symbol': 'C', 'atomic_number': 7},
    {'id': [1, object(), 3], 'formal_charge': 2},
])
def test_invalid_complete_batch_leaves_every_fact_unchanged(columns):
    mol = molecule()
    before = state(mol)
    with pytest.raises((TypeError, ValueError)):
        mol.updata_atoms(**columns)
    unchanged(mol, before)


@pytest.mark.parametrize('field', ['n', 'valence', 'partial_charge', 'is_aromatic', 'constraint'])
def test_batch_rejects_nonwritable_fields(field):
    mol = molecule()
    before = state(mol)
    update = mol.updata_atoms
    with pytest.raises(AttributeError):
        update(**{field: 1})
    unchanged(mol, before)


def test_unknown_field_is_not_an_arbitrary_setattr():
    mol = molecule()
    before = state(mol)
    with pytest.raises(TypeError):
        mol.updata_atoms(unknown_fact=1)
    unchanged(mol, before)
    assert not hasattr(mol.atoms[0], 'unknown_fact')


@pytest.mark.parametrize('selection', ['foreign', 'detached', 'duplicate'])
def test_invalid_member_selection_is_rejected_without_partial_changes(selection):
    mol = molecule()
    foreign = Atom(atomic_number=6)
    removed = mol.create_atom(atomic_number=6)
    mol.remove_atom(removed)
    selected = {'foreign': [mol.atoms[0], foreign], 'detached': [mol.atoms[0], removed],
                'duplicate': [mol.atoms[0], mol.atoms[0]]}[selection]
    before = state(mol)
    with pytest.raises(Exception) as caught:
        mol.updata_atoms(atoms=selected, formal_charge=[1, 2])
    if selection == 'duplicate':
        assert type(caught.value) is ValueError
    else:
        assert type(caught.value).__name__ in ('ObjNotInMolecule', 'NotInSameMolecule')
    unchanged(mol, before)
    assert foreign.mol is not mol and removed.mol is mol


def test_empty_and_no_column_writes_return_none():
    mol = molecule()
    before = state(mol)
    assert mol.updata_atoms() is None
    assert mol.updata_atoms(atoms=[]) is None
    assert mol.updata_atoms(atoms=[], formal_charge=2, coordinates=(1, 2, 3)) is None
    assert Molecule().updata_atoms(formal_charge=0) is None
    unchanged(mol, before)

