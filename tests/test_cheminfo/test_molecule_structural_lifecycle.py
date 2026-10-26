from copy import copy

import numpy as np
import pytest

from hotpot.cheminfo.core import BondKind, Molecule


def _molecule_with_hidden_bonds():
    molecule = Molecule()
    for atomic_number in (30, 7, 6, 8, 6, 6):
        molecule._create_atom(atomic_number=atomic_number)

    hidden_metal = molecule._add_bond(
        0,
        1,
        bond_order=1.0,
        id=11,
        bond_kind=BondKind.DATIVE,
        bond_direction="atom1_to_atom2",
        bond_source="lifecycle-test",
        bond_source_metadata={"role": "metal"},
    )
    hidden_covalent = molecule._add_bond(1, 2, bond_order=1.0, id=12)
    molecule._add_bond(2, 3, bond_order=2.0, id=13)
    molecule._add_bond(4, 5, bond_order=1.0, id=14)
    molecule._update_topology(clear_conformers=False)
    molecule.hide_bonds(
        hidden_metal,
        hidden_covalent,
        clear_conformers=False,
    )
    return molecule


def _all_structural_bonds(molecule):
    return (
        molecule.bonds
        + molecule._hided_metal_bonds
        + molecule._hided_covalent_bonds
    )


def test_copy_clones_active_and_hidden_bonds_with_stable_order():
    molecule = _molecule_with_hidden_bonds()

    clone = copy(molecule)

    assert [bond.id for bond in clone.bonds] == [13, 14]
    assert [bond.id for bond in clone._hided_metal_bonds] == [11]
    assert [bond.id for bond in clone._hided_covalent_bonds] == [12]
    assert all(
        clone_bond is not source_bond
        for clone_bond, source_bond in zip(
            _all_structural_bonds(clone),
            _all_structural_bonds(molecule),
        )
    )
    assert all(
        bond.atom1.mol is clone and bond.atom2.mol is clone
        for bond in _all_structural_bonds(clone)
    )
    assert all(
        not np.shares_memory(source.attrs, cloned.attrs)
        for source, cloned in zip(molecule.atoms, clone.atoms)
    )

    clone.recover_hided_metal_ligand_bonds(clear_conformers=False)
    clone.recover_hided_covalent_bonds(clear_conformers=False)

    assert [bond.id for bond in clone.bonds] == [13, 14, 11, 12]
    assert clone._hided_metal_bonds == []
    assert clone._hided_covalent_bonds == []
    assert [bond.id for bond in molecule.bonds] == [13, 14]
    assert [bond.id for bond in molecule._hided_metal_bonds] == [11]
    assert [bond.id for bond in molecule._hided_covalent_bonds] == [12]


def test_add_component_clones_all_bond_classes_and_rebinds_endpoints():
    target = Molecule()
    target._create_atom(atomic_number=6)
    target._create_atom(atomic_number=8)
    target._add_bond(0, 1, bond_order=1.0, id=10)
    target._update_topology(clear_conformers=False)
    component = _molecule_with_hidden_bonds()
    component_atoms = tuple(component.atoms)
    component_bonds = tuple(_all_structural_bonds(component))

    target.add_component(component)

    assert [bond.id for bond in target.bonds] == [10, 13, 14]
    assert [bond.id for bond in target._hided_metal_bonds] == [11]
    assert [bond.id for bond in target._hided_covalent_bonds] == [12]
    assert all(
        bond.atom1.mol is target and bond.atom2.mol is target
        for bond in _all_structural_bonds(target)
    )
    assert all(
        target_atom is not component_atom
        for target_atom in target.atoms[2:]
        for component_atom in component_atoms
    )
    assert all(
        target_bond is not component_bond
        for target_bond in _all_structural_bonds(target)[1:]
        for component_bond in component_bonds
    )
    assert [bond.id for bond in component.bonds] == [13, 14]
    assert [bond.id for bond in component._hided_metal_bonds] == [11]
    assert [bond.id for bond in component._hided_covalent_bonds] == [12]


def test_components_own_independent_atom_attribute_arrays():
    molecule = Molecule()
    molecule._create_atom(
        atomic_number=6,
        formal_charge=-1,
        coordinates=(1.0, 2.0, 3.0),
        id=21,
    )
    molecule._create_atom(atomic_number=8, id=22)
    molecule._create_atom(atomic_number=7, formal_charge=1, id=23)
    molecule._add_bond(0, 1, bond_order=1.0)
    molecule._update_topology(clear_conformers=False)

    first, second = molecule.components

    assert [atom.id for atom in first.atoms] == [21, 22]
    assert [atom.id for atom in second.atoms] == [23]
    assert not np.shares_memory(molecule.atoms[0].attrs, first.atoms[0].attrs)
    assert not np.shares_memory(molecule.atoms[2].attrs, second.atoms[0].attrs)

    first.atoms[0].formal_charge = 2
    first.atoms[0].coordinates = (9.0, 8.0, 7.0)
    second.atoms[0].formal_charge = -2

    assert molecule.atoms[0].formal_charge == -1
    assert np.array_equal(molecule.atoms[0].coordinates, (1.0, 2.0, 3.0))
    assert molecule.atoms[2].formal_charge == 1


def test_components_clone_only_hidden_bonds_with_both_endpoints_selected():
    molecule = Molecule()
    for atomic_number in (30, 7, 6, 6, 6, 8):
        molecule._create_atom(atomic_number=atomic_number)

    molecule._add_bond(0, 2, bond_order=1.0, id=31)
    molecule._add_bond(2, 1, bond_order=1.0, id=32)
    molecule._add_bond(2, 5, bond_order=1.0, id=33)
    molecule._add_bond(3, 4, bond_order=1.0, id=34)
    hidden_metal_internal = molecule._add_bond(0, 1, bond_order=1.0, id=35)
    hidden_covalent_internal = molecule._add_bond(1, 5, bond_order=1.0, id=36)
    hidden_covalent_cross_component = molecule._add_bond(
        1,
        3,
        bond_order=1.0,
        id=37,
    )
    molecule._update_topology(clear_conformers=False)
    molecule.hide_bonds(
        hidden_metal_internal,
        hidden_covalent_internal,
        hidden_covalent_cross_component,
        clear_conformers=False,
    )

    first, second = molecule.components

    assert [bond.id for bond in first._hided_metal_bonds] == [35]
    assert [bond.id for bond in first._hided_covalent_bonds] == [36]
    assert first._hided_metal_bonds[0] is not hidden_metal_internal
    assert first._hided_covalent_bonds[0] is not hidden_covalent_internal
    assert all(
        bond.atom1.mol is first and bond.atom2.mol is first
        for bond in _all_structural_bonds(first)
    )
    assert second._hided_metal_bonds == []
    assert second._hided_covalent_bonds == []
    assert [bond.id for bond in molecule._hided_metal_bonds] == [35]
    assert [bond.id for bond in molecule._hided_covalent_bonds] == [36, 37]


def _molecule_for_atom_removal():
    molecule = Molecule()
    for atomic_number in (30, 7, 6, 8, 6, 6):
        molecule._create_atom(atomic_number=atomic_number)

    molecule._add_bond(1, 2, bond_order=1.0, id=21)
    molecule._add_bond(4, 5, bond_order=1.0, id=22)
    hidden_metal_incident = molecule._add_bond(0, 1, bond_order=1.0, id=23)
    hidden_covalent_incident = molecule._add_bond(1, 3, bond_order=1.0, id=24)
    hidden_metal_safe = molecule._add_bond(0, 3, bond_order=1.0, id=25)
    hidden_covalent_safe = molecule._add_bond(2, 4, bond_order=1.0, id=26)
    molecule._update_topology(clear_conformers=False)
    molecule.hide_bonds(
        hidden_metal_incident,
        hidden_covalent_incident,
        hidden_metal_safe,
        hidden_covalent_safe,
        clear_conformers=False,
    )
    return molecule


@pytest.mark.parametrize("batch", [False, True])
def test_atom_removal_clears_incident_active_and_hidden_bonds(batch):
    molecule = _molecule_for_atom_removal()
    removed_atoms = [molecule.atoms[1]]
    if batch:
        removed_atoms.append(molecule.atoms[2])
        molecule.remove_atoms(removed_atoms)
    else:
        molecule.remove_atom(removed_atoms[0])

    expected_hidden_covalent_ids = [] if batch else [26]
    assert [bond.id for bond in molecule.bonds] == [22]
    assert [bond.id for bond in molecule._hided_metal_bonds] == [25]
    assert [bond.id for bond in molecule._hided_covalent_bonds] == (
        expected_hidden_covalent_ids
    )

    retained_atom_ids = {id(atom) for atom in molecule.atoms}
    assert all(
        id(endpoint) in retained_atom_ids
        for bond in _all_structural_bonds(molecule)
        for endpoint in bond.atoms
    )
    assert all(
        id(endpoint) in retained_atom_ids
        for atom in molecule.atoms
        for bond in atom.bonds
        for endpoint in bond.atoms
    )
    assert all(atom.bonds == [] for atom in removed_atoms)
    assert all(atom.neighbours == [] for atom in removed_atoms)
