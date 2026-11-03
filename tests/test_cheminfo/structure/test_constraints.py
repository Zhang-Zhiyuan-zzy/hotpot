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


def test_sequence_owner_replacement_shares_only_the_canonical_axis_and_bond_maps():
    mol = molecule()
    owner = mol._constraints
    axes, bonds = owner.atom_axes, owner.bond_values
    mol.atoms[0].x_constraint = True
    mol.bonds[0].constraint = True
    mol.angles[0].constraint = True
    assert mol._constraints is not owner
    assert mol._constraints.atom_axes is axes and mol._constraints.bond_values is bonds
    assert all(atom._constraint_values is axes for atom in mol.atoms)
    assert all(bond._constraint_values is bonds for bond in mol.bonds)
    assert set(axes) == {atom._uid for atom in mol.atoms}
    assert set(bonds) == {bond._uid for bond in mol.bonds}


def test_scalar_constraint_paths_keep_existing_maps_and_avoid_exports_or_hashes(monkeypatch):
    from hotpot import Bond
    from hotpot.cheminfo.structure import _identity
    from hotpot.cheminfo.structure._atom_table import _AtomFactTable
    from hotpot.cheminfo.structure.constraints import _ConstraintSet

    mol = molecule()
    atom, bond = mol.atoms[0], mol.bonds[0]
    other = Bond(mol.atoms[0], mol.atoms[-1], constraint=True, id=87)
    axes, bonds = mol._constraints.atom_axes, mol._constraints.bond_values
    untouched = axes[mol.atoms[-1]._uid]

    def forbidden(*args, **kwargs):
        raise AssertionError('scalar constraint path exported, hashed or rebuilt an owner')

    monkeypatch.setattr(_identity, '_digest', forbidden)
    for name in ('row', 'matrix', 'select'):
        monkeypatch.setattr(_AtomFactTable, name, forbidden)
    monkeypatch.setattr(_ConstraintSet, '__init__', forbidden)
    monkeypatch.setattr(_ConstraintSet, 'subset', forbidden)
    atom.x_constraint = True
    assert atom.x_constraint
    atom.constraint = True
    assert atom.constraint
    bond.constraint = True
    assert bond.constraint
    other.constraint = False
    assert not other.constraint and other.id == 87
    assert mol._constraints.atom_axes is axes and mol._constraints.bond_values is bonds
    assert axes[mol.atoms[-1]._uid] is untouched
    assert other._constraint_values is not bonds
    assert set(other._constraint_values) == {other._uid}


def test_isolated_atom_registration_enrolls_only_new_key_without_rebuilding_owner(monkeypatch):
    from hotpot.cheminfo.structure.constraints import _ConstraintSet

    mol = molecule()
    owner = mol._constraints
    axes, bonds = owner.atom_axes, owner.bond_values
    original = dict(axes)

    def forbidden(*args, **kwargs):
        raise AssertionError('isolated registration rebuilt or filtered an existing owner')

    monkeypatch.setattr(_ConstraintSet, '__init__', forbidden)
    monkeypatch.setattr(_ConstraintSet, 'subset', forbidden)
    monkeypatch.setattr('hotpot.cheminfo.core._filter_constraint_members', forbidden)
    atom = mol.create_atom(atomic_number=6, x_constraint=True)
    assert mol._constraints is owner and owner.atom_axes is axes and owner.bond_values is bonds
    assert set(axes) == set(original) | {atom._uid}
    assert all(axes[uid] is value for uid, value in original.items())
    assert atom._constraint_values is axes and atom.x_constraint
