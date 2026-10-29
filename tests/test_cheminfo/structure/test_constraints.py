"""Persisted sequence constraints survive fresh views, remap and rollback."""

from copy import copy, deepcopy
import pickle

import pytest

from hotpot import Molecule
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._identity import _capture_source, _check_source


def molecule():
    mol = Molecule()
    removed = mol.create_atom(atomic_number=6)
    mol.remove_atom(removed)
    for row in range(4):
        mol.create_atom(atomic_number=6, coordinates=(row, row % 2, row // 2))
    for row in range(3):
        mol.add_bond(row, row + 1)
    return mol


def test_constraints_project_one_owner_and_do_not_create_view_storage():
    mol = molecule()
    angle, torsion = mol.angles[0], mol.torsions[0]
    source = _capture_source(mol, ('constraints',))
    geometry = _capture_source(mol, ('geometry',))
    angle.constraint = True
    torsion.constraint = True
    assert 'constraint' not in angle.__dict__ and 'constraint' not in torsion.__dict__
    assert mol.angles[0].constraint and mol.torsions[0].constraint
    assert mol._constraints.angles == (tuple(atom._uid for atom in angle.atoms),)
    with pytest.raises(RuntimeError, match='constraints'):
        _check_source(mol, source)
    _check_source(mol, geometry)


@pytest.mark.parametrize('clone', [copy, deepcopy, lambda mol: pickle.loads(pickle.dumps(mol))])
def test_copy_and_current_pickle_remap_constraint_members_with_uid_gaps(clone):
    mol = molecule()
    mol.angles[0].constraint = True
    mol.torsions[0].constraint = True
    other = clone(mol)
    assert other.angles[0].constraint and other.torsions[0].constraint
    assert other._constraints.angles == (tuple(atom._uid for atom in other.angles[0].atoms),)
    other.clear_constraints()
    assert mol.angles[0].constraint and mol.torsions[0].constraint


def test_member_row_compaction_preserves_constraint_subjects_and_rejects_old_view():
    mol = molecule()
    angle = mol.angles[-1]
    angle.constraint = True
    uids = tuple(atom._uid for atom in angle.atoms)
    mol.remove_atom(mol.atoms[0])
    assert mol._constraints.angles == (uids,)
    assert mol.angles[0].constraint
    assert tuple(atom._uid for atom in mol.angles[0].atoms) == uids
    with pytest.raises(RuntimeError):
        _ = angle.degrees


def test_constraint_topology_and_coordinate_failure_restores_original_owner():
    mol = molecule()
    angle = mol.angles[0]
    owner = mol._constraints
    coordinates = mol.coordinates.copy()
    original = MemoryError('interrupted complete edit')
    with pytest.raises(MemoryError) as caught:
        with _FactTransaction(mol, producer=1):
            angle.constraint = True
            mol.atoms[0].x = 5.
            mol.remove_atom(mol.atoms[-1])
            raise original
    assert caught.value is original
    assert mol._constraints is owner and not angle.constraint
    assert (mol.coordinates == coordinates).all()


def test_component_and_merge_remap_constraints_without_mutable_sharing():
    source = molecule()
    source.angles[0].constraint = True
    source.torsions[0].constraint = True
    destination = Molecule()
    destination.create_atom(atomic_number=8)
    destination.add_component(source)
    component = next(part for part in destination.components if len(part.atoms) == 4)
    assert component.angles[0].constraint and component.torsions[0].constraint
    component.clear_constraints()
    assert source.angles[0].constraint and destination.angles[0].constraint
