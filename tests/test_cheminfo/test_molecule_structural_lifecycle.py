from copy import copy

import numpy as np
import pytest

from hotpot.cheminfo.core import BondKind, Molecule
from hotpot.cheminfo.operations.topology_queries import _components
from hotpot.cheminfo.structure._selection import _capture_topology_selection


def _molecule_with_complete_bonds():
    molecule = Molecule()
    for number in (30, 7, 6, 8, 6, 6):
        molecule.create_atom(atomic_number=number)
    molecule.add_bond(0, 1, bond_order=1., id=11, bond_kind=BondKind.DATIVE,
                      bond_direction='atom1_to_atom2', bond_source='lifecycle-test',
                      bond_source_metadata={'role': 'metal'})
    molecule.add_bond(1, 2, bond_order=1., id=12)
    molecule.add_bond(2, 3, bond_order=2., id=13)
    molecule.add_bond(4, 5, bond_order=1., id=14)
    return molecule


def test_copy_clones_complete_bonds_with_stable_order_and_no_source_alias():
    molecule = _molecule_with_complete_bonds()
    clone = copy(molecule)
    assert [bond.id for bond in clone.bonds] == [11, 12, 13, 14]
    assert all(left is not right for left, right in zip(clone.bonds, molecule.bonds))
    assert all(bond.atom1.mol is clone and bond.atom2.mol is clone for bond in clone.bonds)
    assert all(not np.shares_memory(left.attrs, right.attrs)
               for left, right in zip(molecule.atoms, clone.atoms))
    assert clone.bonds[0].bond_kind is BondKind.DATIVE
    assert clone.bonds[0].bond_direction == 'atom1_to_atom2'
    clone.bonds[0]._bond_source_metadata['role'] = 'edited'
    clone.remove_bond(clone.bonds[1])
    assert [bond.id for bond in clone.bonds] == [11, 13, 14]
    assert [bond.id for bond in molecule.bonds] == [11, 12, 13, 14]
    assert molecule.bonds[0].bond_source_metadata == {'role': 'metal'}


def test_add_component_clones_all_bond_classes_in_destination_then_source_order():
    target = Molecule()
    target.create_atom(atomic_number=6)
    target.create_atom(atomic_number=8)
    original = target.add_bond(0, 1, bond_order=1., id=10)
    component = _molecule_with_complete_bonds()
    source_atoms, source_bonds = tuple(component.atoms), tuple(component.bonds)
    target.add_component(component)
    assert [bond.id for bond in target.bonds] == [10, 11, 12, 13, 14]
    assert target.bonds[0] is original
    assert all(bond.atom1.mol is target and bond.atom2.mol is target for bond in target.bonds)
    assert all(atom is not source for atom in target.atoms[2:] for source in source_atoms)
    assert all(bond is not source for bond in target.bonds[1:] for source in source_bonds)
    assert tuple(component.atoms) == source_atoms and tuple(component.bonds) == source_bonds


def test_components_own_independent_atom_attribute_arrays():
    molecule = Molecule()
    molecule.create_atom(atomic_number=6, formal_charge=-1, coordinates=(1., 2., 3.), id=21)
    molecule.create_atom(atomic_number=8, id=22)
    molecule.create_atom(atomic_number=7, formal_charge=1, id=23)
    molecule.add_bond(0, 1, bond_order=1.)
    first, second = molecule.components
    assert [atom.id for atom in first.atoms] == [21, 22]
    assert [atom.id for atom in second.atoms] == [23]
    assert not np.shares_memory(molecule.atoms[0].attrs, first.atoms[0].attrs)
    assert not np.shares_memory(molecule.atoms[2].attrs, second.atoms[0].attrs)
    first.atoms[0].formal_charge = 2
    first.atoms[0].coordinates = (9., 8., 7.)
    second.atoms[0].formal_charge = -2
    assert molecule.atoms[0].formal_charge == -1
    assert np.array_equal(molecule.atoms[0].coordinates, (1., 2., 3.))
    assert molecule.atoms[2].formal_charge == 1


def test_public_components_use_complete_edges_while_operation_components_filter_exact_rows():
    molecule = Molecule()
    for number in (30, 7, 6, 6, 6, 8):
        molecule.create_atom(atomic_number=number)
    for first, second, label in ((0, 2, 31), (2, 1, 32), (2, 5, 33), (3, 4, 34),
                                 (0, 1, 35), (1, 5, 36), (1, 3, 37)):
        molecule.add_bond(first, second, bond_order=1., id=label)
    source = tuple(molecule.bonds)
    assert len(molecule.components) == 1
    selection = _capture_topology_selection(molecule, excluded_bonds=(source[-1],))
    first, second = _components(molecule, Molecule, selection=selection)
    assert [bond.id for bond in first.bonds] == [31, 32, 33, 35, 36]
    assert [bond.id for bond in second.bonds] == [34]
    assert all(bond.atom1.mol is first and bond.atom2.mol is first for bond in first.bonds)
    assert all(bond is not original for bond in first.bonds for original in source)
    assert tuple(molecule.bonds) == source


def _molecule_for_atom_removal():
    molecule = Molecule()
    for number in (30, 7, 6, 8, 6, 6):
        molecule.create_atom(atomic_number=number)
    for first, second, label in ((1, 2, 21), (4, 5, 22), (0, 1, 23),
                                 (1, 3, 24), (0, 3, 25), (2, 4, 26)):
        molecule.add_bond(first, second, bond_order=1., id=label)
    return molecule


@pytest.mark.parametrize('batch', [False, True])
def test_atom_removal_clears_all_incident_bonds_and_preserves_survivor_order(batch):
    molecule = _molecule_for_atom_removal()
    originals = tuple(molecule.bonds)
    removed_atoms = [molecule.atoms[1]]
    if batch:
        removed_atoms.append(molecule.atoms[2])
        molecule.remove_atoms(removed_atoms)
    else:
        molecule.remove_atom(removed_atoms[0])
    assert [bond.id for bond in molecule.bonds] == ([22, 25] if batch else [22, 25, 26])
    assert tuple(molecule.bonds) == tuple(bond for bond in originals
                                         if not any(atom in bond.atoms for atom in removed_atoms))
    retained = {id(atom) for atom in molecule.atoms}
    assert all(id(endpoint) in retained for bond in molecule.bonds for endpoint in bond.atoms)
    assert all(id(endpoint) in retained for atom in molecule.atoms for bond in atom.bonds for endpoint in bond.atoms)
    assert all(atom.bonds == [] and atom.neighbours == [] for atom in removed_atoms)
