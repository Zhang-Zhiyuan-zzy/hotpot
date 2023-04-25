"""
python v3.7.9
@Project: hotpot
@File   : cheminfo.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/14
@Time   : 4:09
"""
import copy
import os
import re
from io import IOBase
from os import PathLike
from pathlib import Path
from abc import ABC, abstractmethod
import json
from typing import *
import numpy as np
from openbabel import openbabel as ob, pybel as pb
from src._io import retrieve_format, Dumper, Parser
from src.tanks.quantum import Gaussian
import src.tanks.lmp as lmp

dir_root = os.path.join(os.path.dirname(__file__))
periodic_table = json.load(open(f'{dir_root}/../data/periodic_table.json', encoding='utf-8'))['elements']
periodic_table = {d['symbol']: d for d in periodic_table}

_bond_type = {
    'Unknown': 0,
    'Single': 1,
    'Double': 2,
    'Triple': 3,
    'Aromatic': 5,
}

_type_bond = {
    0: 'Unknown',
    1: 'Single',
    2: 'Double',
    3: 'Triple',
    5: 'Aromatic'
}


class Wrapper(ABC):
    """
    A wrapper for openbabel object.
    The _set_attrs method is used to set any keyword attribute, the attribute names, in the wrapper context, are defined
    by the keys from returned dict of _attr_setters; the values of the returned dict of _attr_setters are a collection
    of specific private method to wrapper and call openbabel method to set the attributes in openbabel object.
    """

    def _set_attrs(self, **kwargs):
        """    Set any atomic attributes by name    """
        attr_setters = self._attr_setters
        for name, value in kwargs.items():
            setter = attr_setters.get(name)

            if setter:  # if the attribute is exist in the object.
                assert isinstance(setter, Callable)
                setter(value)

    @property
    @abstractmethod
    def _attr_setters(self) -> Dict[str, Callable]:
        raise NotImplemented

    def kwargs_setters(self):
        list_setters = [f'{k}: {s.__doc__}' for k, s in self._attr_setters.items()]
        print("\n".join(list_setters))

    @property
    def setter_keys(self):
        return list(self._attr_setters.keys())


class Molecule(Wrapper, ABC):
    """"""
    # All Molecule attribute's items relating to molecule conformers
    conformer_items = (
        # the items are ranked by the number of values for each atom, for example:
        #   - the all_atom_charges and atom_spin_densities have 1 value for each atom, so they are placed in the second
        #     item (with the index 1)
        #   - the coordinates have 3 values for each atom, i.e., [x, y, z], so it is placed in the forth
        #     item (with the index 3）
        # For the molecular attributes, which have only one value for each conformer and represent the attribute
        # of whole molecule, they are place in the first item (with the index 0)
        ('all_energy',),
        ('all_atom_charges', 'all_atom_spin_densities'),
        (),
        ('all_coordinates', 'all_forces')
    )

    def __init__(self, ob_mol: ob.OBMol = None, _data: dict = None,**kwargs):
        if _data:
            self._data = _data
        else:
            self._data = {
                'ob_mol': ob_mol if ob_mol else ob.OBMol()
            }
        self._set_attrs(**kwargs)

    def __repr__(self):
        return f'Mol({self.ob_mol.GetFormula()})'

    def __add__(self, other: ['Molecule']):
        """
        Two Molecule objects could add to a new one to merge all of their conformers.
        All of information about the conformer will be merged to one.
        the other information will reserve the one in the left item
        Args:
            other: the right item

        Returns:
            Molecule
        """
        # When other obj is a Molecule or a child of Molecule
        if isinstance(other, Molecule):
            # If this one is compatible to the other
            if self.iadd_accessible(other):
                clone = self.copy()
                clone += other

                return clone

            # If the other one is compatible to this one
            if other.iadd_accessible(self):
                clone = other.copy()
                clone += self

                return clone

            # If they are compatible, but are Molecule or child of Molecule
            return MolBundle([self, other])

        # if isinstance(other, MixSameAtomMol):
        #     return self.to_mix_mol() + other

        # When other obj is a MolBundle
        if isinstance(other, MolBundle):
            return MolBundle([self] + other.mols)

        else:
            raise TypeError('the Molecule only add with Molecule or MolBundle')

    def __iadd__(self, other):
        """
        Self add with other Molecule object with consist atoms list,
        The attributes or information about conformers will merge with the other,
        other attributes or information will be reserved
        Args:
            other: the merged Molecule object

        Returns:
            None
        """
        if not isinstance(other, Molecule):
            raise TypeError('the Molecule object is only allowed to add with other Molecule')

        # Check whether left and right Molecules have consist atom list
        if not self.iadd_accessible(other):
            raise AttributeError(
                'the self addition cannot be performed among molecules with different atoms list!'
            )

        return self._merge_conformer_attr(other)

    def _merge_conformer_attr(self, other: 'Molecule'):
        """ Merge attributes, relate to molecule conformer, in other Molecule into this Molecule """

        def merge_attr(attr_name: str):
            """ Merge single conformer attr """
            left_attr = getattr(self, attr_name)
            right_attr = getattr(other, attr_name)

            if isinstance(left_attr, np.ndarray) and isinstance(right_attr, np.ndarray):
                self._data[attr_name] = np.concatenate([left_attr, right_attr])
            elif not (  # If the left and right values are not both empty, raise Attributes error.
                (left_attr is None) or (isinstance(left_attr, np.ndarray) and (not left_attr.all())) and
                (right_attr is None) or (isinstance(right_attr, np.ndarray) and (not right_attr.all()))
            ):
                raise AttributeError(
                    f'the configure relational attribute {attr_name} is different in:\n'
                    f'  - {self}_identifier: {self.identifier}\n'
                    f'  - {other}_identifier: {other.identifier}'
                    'they cannot to perform addition operation'
                )

        for i, items in enumerate(self.conformer_items):
            for item in items:
                merge_attr(item)

        return self

    @property
    def _ob_atom_indices(self):
        """ Get the indices for all OBAtom """
        indices = []

        try:
            num_ob_atoms = self.ob_mol.NumAtoms()
        # If there is none of atoms in the OBMol, raise the TypeError.
        except TypeError:
            num_ob_atoms = 0

        idx = 0
        while len(indices) < num_ob_atoms:
            ob_atom = self.ob_mol.GetAtomById(idx)

            # if get a OBAtom
            if ob_atom:
                assert idx == ob_atom.GetId()
                indices.append(idx)

            idx += 1

        return indices

    @property
    def ob_mol(self):
        return self._data['ob_mol']

    @staticmethod
    def _assign_coordinates(the_mol: 'Molecule', coordinates: np.ndarray):
        """ Assign coordinates for all atoms in the Molecule """
        if len(the_mol.atoms) != coordinates.shape[-2]:
            raise AttributeError('the coordinate matrix do not match the number of atoms')

        for new_mol_atom, new_atom_coord in zip(the_mol.atoms, coordinates):
            new_mol_atom.coordinates = new_atom_coord

    @property
    def _attr_setters(self) -> Dict[str, Callable]:
        return {
            'atoms.partial_charge': self._set_atoms_partial_charge,
            "identifier": self._set_identifier,
            "energy": self._set_energy,
            'all_energy': self._set_all_energy,
            'charge': self._set_mol_charge,
            'all_atom_charges': self._set_all_atom_charges,
            'all_atom_spin_densities': self._set_all_atom_spin_densities,
            'spin': self._set_spin_multiplicity,
            'atoms': self._set_atoms,
            'mol_orbital_energies': self._set_mol_orbital_energies,
            'all_coordinates': self._set_all_coordinates,
            'all_forces': self._set_all_forces,
            'forces': self._set_forces
        }

    def _pert_mol_generate(self, coordinates: np.ndarray):
        """
        Generate new molecule obj according to new given coordinate
        Args:
            coordinates: New coordinates matrix

        Returns:
            Molecule, copy of this molecule with new coordinates
        """
        clone_mol = self.copy()
        self._assign_coordinates(clone_mol, coordinates)
        return clone_mol
    #
    # def _remake_atoms(self, atomic_numbers: np.ndarray):
    #     """ Remake all atoms in the molecule according to given atoms_numbers """
    #     ob_mol = ob.OBMol()
    #     self._data['ob_mol'] = ob_mol
    #     atomic_numbers = atomic_numbers.flatten()
    #
    #     for an in atomic_numbers:
    #         ob_mol.AddAtom(an)
    #
    #     print(f'Remade molecule with {len(self.atoms)} atoms')

    def _reorganize_atom_indices(self):
        """ reorganize or rearrange the indices for all atoms """
        for i, ob_atom in enumerate(ob.OBMolAtomIter(self.ob_mol)):
            ob_atom.SetId(i)

    def _set_atoms(self, atoms_kwargs: List[Dict[str, Any]]):
        """ add a list of atoms by a list atoms attributes dict """
        for atom_kwarg in atoms_kwargs:
            a = Atom(**atom_kwarg)
            self.add_atom(a)

    def _set_atoms_partial_charge(self, partial_charges: [np.ndarray, Sequence[float]]):
        """ Set partial charges for all atoms in the molecule """
        if not isinstance(partial_charges, (np.ndarray, Sequence[int])):
            raise TypeError(
                f'the `partial_charges` should be np.ndarray or Sequence of float, not {type(partial_charges)}'
            )

        if self.atom_num != len(partial_charges):
            raise ValueError('the given partial charges should have same numbers with the number of atoms')

        for atom, partial_charge in zip(self.atoms, partial_charges):
            atom.partial_charge = partial_charge

    def _set_all_coordinates(self, all_coordinates: np.ndarray):
        """
        Assign the coordinates collection directly
        Args:
            all_coordinates: numpy array with the shape (M, N, 3), where the M is the number of coordinates
            in the collection, the N is the number of atoms of the molecule.

        Returns:
            None
        """
        if not isinstance(all_coordinates, np.ndarray):
            raise ValueError(f'the given all_coordinates must be a numpy.ndarray class, instead of {type(all_coordinates)}')

        if all_coordinates.shape[-1] != 3:
            raise ValueError(f'the coordinate must be 3 dimension, instead of {all_coordinates.shape[-1]}')

        if len(all_coordinates.shape) == 2:
            # if only give a group of coordinates
            all_coordinates = all_coordinates.reshape((-1, all_coordinates.shape[-2], 3))
        elif len(all_coordinates.shape) != 3:
            raise ValueError(
                f'the shape of given all_coordinates should with length 2 or 3, now is {len(all_coordinates.shape)}'
            )

        self._data['all_coordinates'] = all_coordinates

    def _set_atom_charges(self, charge: Union[Sequence, np.ndarray]):
        """ Set partial charge for each atoms in the mol """
        if not isinstance(charge, (Sequence, np.ndarray)):
            raise TypeError(f'the charge should be a sequence or np.ndarray, got {type(charge)}')

        if isinstance(charge, np.ndarray):
            charge = charge.flatten()

        if len(charge) != self.atom_num:
            raise ValueError('the number of charges do not match with the atom charge')

        for atom, ch in zip(self.atoms, charge):
            atom.partial_charge = ch

    def _set_all_atom_charges(self, charges: np.ndarray):
        """
        set groups of charges for each atoms in the mol, and each group of charges are corresponding to a
        conformer of the mol
        Args:
            charges: group of atoms with the shape of (C, N), where the C is the number of the conformer
             and the N is the number of the atom in the molecule
        """
        if not isinstance(charges, np.ndarray):
            raise TypeError('the arg charges should be np.ndarray')

        if len(charges.shape) != 2 and charges.shape[1] != self.atom_num:
            raise ValueError('the shape of the arg: charge should be (number_of_conformer, number_of_atoms),'
                             f'got the value with shape {charges.shape}')

        self._data['all_atom_charges'] = charges

    def _set_all_atom_spin_densities(self, group_spd: np.ndarray) -> None:
        """
        assign groups of spin densities for all atom in molecule, each group is corresponding to a conformer
        Args:
            group_spd(np.ndarray): group of spin densities, the numpy array with the (C, N) shape,
             where the C is the number of conformer, the N is the number of atoms
        """
        if not isinstance(group_spd, np.ndarray):
            raise TypeError('the arg group_spd should be np.ndarray')

        if len(group_spd.shape) != 2 and group_spd.shape[1] != self.atom_num:
            raise ValueError('the shape of the arg: group_spd should be (number_of_conformer, number_of_atoms),'
                             f'got the value with shape {group_spd.shape}')

        self._data['all_atom_spin_densities'] = group_spd

    def _set_atom_spin_densities(self, spd: Union[Sequence, np.ndarray]):
        """ assign the spin density for each of atoms in the mol """
        if not isinstance(spd, (Sequence, np.ndarray)):
            raise TypeError(f'the charge should be a sequence or np.ndarray, got {type(spd)}')

        if isinstance(spd, np.ndarray):
            spd = spd.flatten()

        if len(spd) != self.atom_num:
            raise ValueError('the number of charges do not match with the atom charge')

        for atom, sp in zip(self.atoms, spd):
            atom.spin_density = sp

    def _set_forces(self, forces: np.ndarray):
        """ Set the force vectors for each atoms in the molecule """
        if not isinstance(forces, np.ndarray):
            raise TypeError('the forces should be np.ndarray')

        if len(forces.shape) != 2:
            raise ValueError('the length of shape of forces should be 2')

        if forces.shape[-2] != self.atom_num:
            raise ValueError('the give forces do not match to the number of atoms')

        for atom, force_vector in zip(self.atoms, forces):
            atom.force_vector = force_vector

    def _set_all_forces(self, all_forces: np.ndarray):
        """ Store the force matrix into the attribute dict """
        if not isinstance(all_forces, np.ndarray):
            raise TypeError('the all_forces should be np.ndarray')

        if len(all_forces.shape) != 3:
            raise ValueError('the length of shape of all_forces should be 3')

        if all_forces.shape[-2] != self.atom_num:
            raise ValueError('the give all_forces do not match to the number of atoms')

        self._data['all_forces'] = all_forces

    def _set_mol_charge(self, charge: int):
        self.ob_mol.SetTotalCharge(charge)

    def _set_mol_orbital_energies(self, orbital_energies: list[np.ndarray]):
        self._data['mol_orbital_energies'] = orbital_energies[0]

    def _set_energy(self, energy: float):
        """ set the energy """
        self.ob_mol.SetEnergy(energy)

    def _set_all_energy(self, all_energy: Union[float, np.ndarray], config_index: Optional[int] = None):
        """ set the energy for all configures """
        if isinstance(all_energy, float):
            self._data['all_energy'] = np.array([all_energy])
        else:
            all_energy = all_energy.flatten()
            self._data['all_energy'] = all_energy

    def _set_identifier(self, identifier):
        self.ob_mol.SetTitle(identifier)

    def _set_spin_multiplicity(self, spin):
        self.ob_mol.SetTotalSpinMultiplicity(spin)

    def add_atom(self, atom: Union["Atom", str, int]):
        """
        Add a new atom out of the molecule into the molecule.
        Args:
            atom:

        Returns:

        """
        if isinstance(atom, str):
            atom = Atom(symbol=atom)
        elif isinstance(atom, int):
            atom = Atom(atomic_number=atom)

        # Avoid add a existed atom into the molecule
        if atom.molecule and (atom.molecule is self or atom.molecule.ob_mol is self.ob_mol):
            raise AttributeError("This atom have exist in the molecule")

        success = self.ob_mol.AddAtom(atom.ob_atom)
        if success:

            # the OBAtom have stored in the OBMol and self
            # get atom by idx enumerate from 1, instead of 0.
            ob_atom_in_ob_mol = self.ob_mol.GetAtom(self.ob_mol.NumAtoms())

            # Get the attribute dict and replace the 'OBAtom' and '_mol' items
            # The 'OBAtom' is the OBAtom has been stored in the self(Molecule)
            # The '_mol' is the self(Molecule)
            new_atom_data = atom._data
            new_atom_data['OBAtom'] = ob_atom_in_ob_mol
            new_atom_data['_mol'] = self

            # replace the attr data dict
            atom = Atom()
            atom._replace_attr_data(new_atom_data)

            # add the new atom into atoms list directly
            atoms = self._data.setdefault('atoms', [])
            atoms.append(atom)

            return atom

        else:
            print(RuntimeWarning("Fail to add atom"))

    def add_bond(
            self,
            atom1: Union[str, int, 'Atom'],
            atom2: Union[str, int, 'Atom'],
            bond_type: Union[str, int],
    ):
        """ Add a new bond into the molecule """
        atoms = (atom1, atom2)
        atom_idx = []
        for a in atoms:
            if isinstance(a, int):
                atom_idx.append(a)
            if isinstance(a, Atom):
                atom_idx.append(a.idx)
            if isinstance(a, str):
                atom_idx.append(self.atom(a).idx)

        # Represent the bond type by int, refer to _bond_type dict
        bond_type = bond_type if isinstance(bond_type, int) else _bond_type[bond_type]

        # Try to add new OBMol
        # 'openbabel' has an odd behave that `index` of the `OBAtom` with various origin in the `OBMol`.
        # the `Id` of `OBAtom` from 0; but the `Idx` of `OBAtom` from 1.
        # To meet the convention, the `Id` is selected to be the unique `index` to specify `Atom`.
        # However, when try to add a `OBBond` to link each two `OBAtoms`, the `Idx` is the only method
        # to specify the atoms, so our `index` in `Atom` are added 1 to match the 'Idx'
        success = self.ob_mol.AddBond(atom_idx[0] + 1, atom_idx[1] + 1, bond_type)

        if success:
            new_bond_idx = self.ob_mol.NumBonds() - 1
            new_ob_bond = self.ob_mol.GetBondById(new_bond_idx)
            bond = Bond(new_ob_bond, self)

            # Add new bond into Molecule
            bonds = self._data.setdefault('bonds', [])
            bonds.append(bond)

        elif atom_idx[0] not in self.atom_indices:
            raise KeyError("the start atom1 doesn't exist in molecule")

        elif atom_idx[1] not in self.atom_indices:
            raise KeyError("the end atom2 doesn't exist in molecule")

        else:
            raise RuntimeError('add bond not successful!')

        # Return the bond have add into the molecule
        return bond

    def assign_bond_types(self):
        self.ob_mol.PerceiveBondOrders()

    def atom(self, idx_label: Union[int, str]) -> 'Atom':
        """ get atom by label or idx """
        if not self.is_labels_unique:
            print(AttributeError('the molecule atoms labels are not unique!'))
            return

        if isinstance(idx_label, str):
            return self.atoms[self.labels.index(idx_label)]
        elif isinstance(idx_label, int):
            return self.atoms[idx_label]
        else:
            raise TypeError(f'the given idx_label is expected to be int or string, but given {type(idx_label)}')
        
    @property
    def atom_num(self):
        return self.ob_mol.NumAtoms()

    @property
    def atoms(self):
        """
        If the list of atoms doesn't exist, generate atoms list and then return
        If the list of atoms have existed, without consistency to the list indices of OBAtom in the OBMol,
        resort the atoms list, create new atoms and delete nonexistent atoms.
        If the list have existed with correct sort, return directly.
        Returns:
            list of Atoms
        """
        atoms = self._data.get('atoms')  # existed atoms list
        atoms_indices = [a.idx for a in atoms] if atoms else []  # the indices list of existed atoms
        ob_atom_indices = self._ob_atom_indices  # the sorted indices list for OBAtom in OBMol

        if not atoms:
            atoms = self._data['atoms'] = [
                Atom(ob_atom=self.ob_mol.GetAtomById(idx), _mol=self)
                for idx in ob_atom_indices
            ]

        # If the order of atom index by atoms is non-match with the order by OBAtoms
        # the atom indices and OBAtom indices are NOT required to be complete, but MUST be consistent.
        # for example, the atom indices and the OBAtom indices might be [1, 2, 4, 6], simultaneously,
        # however, the situation that the atom indices are [1, 2] and the OBAtom indices are [1, 2, 3] is not allowed!!
        elif len(atoms) != len(ob_atom_indices) or any(ai != Ai for ai, Ai in zip(atoms_indices, ob_atom_indices)):
            new_atoms = []
            for OBAtom_idx in ob_atom_indices:

                try:  # If the atom has existed in the old atom list, append it into new list with same order of OBAtoms
                    atom_idx = atoms_indices.index(OBAtom_idx)
                    new_atoms.append(atoms[atom_idx])

                # If the atom doesn't exist in the old atom, create a new atoms and append to new list
                except ValueError:
                    new_atoms.append(Atom(self.ob_mol.GetAtomById(OBAtom_idx), _mol=self))

            # Update the atoms list
            atoms = self._data['atoms'] = new_atoms

        return copy.copy(atoms)  # Copy the atoms list

    @property
    def atom_charges(self) -> np.ndarray:
        """ Return all atoms charges as a numpy array """
        return np.array([a.partial_charge for a in self.atoms])

    @property
    def all_atom_charges(self) -> np.ndarray:
        """ Return all atoms charges as a numpy array for every conformers """
        all_atom_charges = self._data.get('all_atom_charges')
        if isinstance(all_atom_charges, np.ndarray):
            return all_atom_charges
        return self.atom_charges.reshape((-1, self.atom_num))

    @property
    def atom_indices(self) -> list[int]:
        return [a.idx for a in self.atoms]

    @property
    def atom_labels(self) -> list[str]:
        return [a.label for a in self.atoms]

    @property
    def atom_spin_densities(self) -> np.ndarray:
        return np.array([a.spin_density for a in self.atoms])

    @property
    def all_atom_spin_densities(self):
        all_atom_spin_densities = self._data.get('all_atom_spin_densities')
        if all_atom_spin_densities is not None:
            return all_atom_spin_densities
        return self.atom_spin_densities.reshape((-1, self.atom_num))

    @property
    def atomic_numbers(self):
        return tuple(a.atomic_number for a in self.atoms)

    def bond(self, atom1: Union[int, str], atom2: Union[int, str], miss_raise: bool = False) -> 'Bond':
        """
        Return the Bond by given atom index labels in the bond ends
        if the bond is missing in the molecule, return None if given miss_raise is False else raise a KeyError
        Args:
            atom1(int|str): index or label of atom in one of the bond end
            atom2(int|str): index or label of atom in the other end of the bond
            miss_raise(bool): Whether to raise error when can't find the bond

        Returns:
            Bond

        Raises:
            KeyError: when can't find the bond, and the miss_raise passing True

        """
        atom1: Atom = self.atom(atom1)
        atom2: Atom = self.atom(atom2)
        ob_bond = self.ob_mol.GetBond(atom1.ob_atom, atom2.ob_atom)

        if ob_bond:
            return Bond(ob_bond, self)
        else:
            return None

    @property
    def bonds(self):
        return [Bond(OBBond, _mol=self) for OBBond in ob.OBMolBondIter(self.ob_mol)]

    def build_bonds(self):
        self.ob_mol.ConnectTheDots()

    @property
    def charge(self):
        return self.ob_mol.GetTotalCharge()

    @charge.setter
    def charge(self, charge):
        self._set_mol_charge(charge)

    def clean_bonds(self):
        """ Remove all bonds """
        # Iterate directly will fail.
        ob_bonds = [OBBond for OBBond in ob.OBMolBondIter(self.ob_mol)]
        for OBBond in ob_bonds:
            self.ob_mol.DeleteBond(OBBond)

    @property
    def components(self):
        """ get all fragments don't link each by any bonds """
        separated_obmol = self.ob_mol.Separate()
        return [Molecule(obmol) for obmol in separated_obmol]

    @property
    def configure_number(self):
        all_coordinates = self._data.get('all_coordinates')
        if isinstance(all_coordinates, np.ndarray):
            return all_coordinates.shape[0]
        else:
            return 1

    def configure_select(self, config_idx: int):
        """ select specific configure by index """

        def assign_numpy_attrs(attrs_name: str, setter: Callable):
            attrs = self._data.get(attrs_name)
            if isinstance(attrs, np.ndarray):
                try:
                    attr = attrs[config_idx]
                except IndexError:
                    attr = None
            else:
                attr = None

            if isinstance(attr, np.ndarray) or attr:
                setter(attr)

        all_coordinates = self._data.get('all_coordinates')
        if all_coordinates is None and config_idx:
            raise IndexError('Only one configure here!')

        # assign the coordinates for the molecule
        coordinates = all_coordinates[config_idx]
        self._assign_coordinates(self, coordinates)

        assign_numpy_attrs('all_energy', self._set_energy)

        assign_numpy_attrs('all_atom_charges', self._set_atom_charges)

        assign_numpy_attrs('all_atom_spin_densities', self._set_atom_spin_densities)

        assign_numpy_attrs('all_forces', self._set_forces)

    @property
    def coordinates(self) -> np.ndarray:
        """
        Get the matrix of all atoms coordinates,
        where the row index point to the atom index;
        the column index point to the (x, y, z)
        """
        return np.array([atom.coordinates for atom in self.atoms], dtype=np.float64)

    @property
    def all_coordinates(self) -> np.ndarray:
        """
        Get the collections of the matrix of all atoms coordinates,
        each matrix represents a configure.
        The return array with shape of (C, N, 3),
        where the C is the number of conformers, the N is the number of atoms
        """
        all_coordinates = self._data.get('all_coordinates')
        if isinstance(all_coordinates, np.ndarray):
            return all_coordinates
        else:
            return self.coordinates.reshape((-1, self.atom_num, 3))

    def copy(self) -> 'Molecule':
        """ Copy the Molecule """
        # Create the new data sheet
        new_data = {
            'ob_mol': ob.OBMol(),
            'atoms': [],
            'bonds': []
        }

        # Copy all attribute except fro 'OBMol', 'atoms' and 'bonds'
        for name, value in self._data.items():
            if name not in new_data:
                new_data[name] = copy.copy(value)

        # Create new Molecule
        clone_mol = Molecule()
        clone_mol._data = new_data
        clone_mol.identifier = f"{self.identifier}_clone"

        # Clone the old UnitCell data into new
        cell_data = self.ob_mol.GetData(12)  # the UnitCell of OBmol save with idx 12
        if cell_data:
            clone_mol.ob_mol.CloneData(cell_data)

        # copy, in turn, each Atom in this molecule and add them into new Molecule
        for atom in self.atoms:
            clone_atom = atom.copy()
            clone_atom_in_new_mol = clone_mol.add_atom(clone_atom)

        # generate Bonds in new Molecule with same graph pattern in this Molecule
        for bond in self.bonds:
            new_bond = clone_mol.add_bond(*bond.atoms, bond_type=bond.type)

        return clone_mol

    def copy_data(self):
        return copy.copy(self._data)

    def clean_configures(self, pop: bool = False):
        """ clean all config save inside the molecule """
        try:
            all_coordinates = self._data.pop('all_coordinates')
        except KeyError:
            all_coordinates = None

        if pop:
            return all_coordinates

    def create_atom(self, symbol: str, **kwargs):
        """
        Discarded !!!
        Create a new atom into the molecule
        Args:
            symbol: the atomic symbol
            **kwargs: any attribute for the atom

        Returns:
            the created atom in the molecule
        """
        OBAtom: ob.OBAtom = self.ob_mol.NewAtom()
        atomic_number = periodic_table[symbol]['number']
        OBAtom.SetAtomicNum(atomic_number)
        atom = Atom(OBAtom, mol=self, **kwargs)

        return atom

    def crystal(self):
        """ Get the Crystal containing the Molecule """
        cell_index = ob.UnitCell  # Get the index the UnitCell data save
        cell_data = self.ob_mol.GetData(cell_index)

        if cell_data:
            ob_unit_cell = ob.toUnitCell(cell_data)
            return Crystal(ob_unit_cell, molecule=self)
        else:
            return None

    def dump(self, fmt: str, *args, **kwargs) -> Union[str, bytes]:
        """"""
        dumper = Dumper(fmt=fmt, source=self, *args, **kwargs)
        return dumper()

    @property
    def elements(self) -> list[str]:
        return re.findall(r'[A-Z][a-z]*', self.formula)

    @property
    def all_energy(self):
        return self._data.get('all_energy')

    @property
    def energy(self):
        """ Return energy with kcal/mol as default """
        return self.ob_mol.GetEnergy()

    def feature_matrix(self, feature_names):
        n = self.atom_num
        m = len(feature_names)
        np.set_printoptions(suppress=True, precision=6)
        feature_matrix = np.zeros((n, m+3))

        for i, atom in enumerate(self.atoms):
            atom_features = atom.element_features(output_type='dict', *feature_names)
            for j, feature_name in enumerate(feature_names):
                if feature_name == 'atom_orbital_feature':
                    ao_features = atom_features[feature_name]
                    feature_matrix[i, j] = ao_features['s']
                    feature_matrix[i, j + 1] = ao_features['p']
                    feature_matrix[i, j + 2] = ao_features['d']
                    feature_matrix[i, j + 3] = ao_features['f']
                    j += 3
                else:
                    feature_matrix[i, j] = atom_features[feature_name]
        return feature_matrix

    @property
    def forces(self):
        """ return the all force vectors for all atoms in the molecule """
        return np.vstack((atom.force_vector for atom in self.atoms))

    @property
    def all_forces(self):
        """ the force matrix for all configure """
        force_matrix = self._data.get("all_forces")
        if isinstance(force_matrix, np.ndarray):
            return force_matrix
        return self.forces

    @property
    def formula(self) -> str:
        return self.ob_mol.GetFormula()

    def generate_compact_lattice(self, inplace=False):
        """"""
        ob_unit_cell = ob.OBUnitCell()
        lattice_params = np.concatenate((self.xyz_diff, [90., 90., 90.]))
        ob_unit_cell.SetData(*lattice_params)
        ob_unit_cell.SetSpaceGroup('P1')

        mol = self if inplace else self.copy()
        mol.ob_mol.CloneData(ob_unit_cell)

        return mol

    def gaussian(
            self,
            g16root: Union[str, PathLike],
            link0: Union[str, List[str]],
            route: Union[str, List[str]],
            path_log_file: Union[str, PathLike] = None,
            path_err_file: Union[str, PathLike] = None,
            inplace_attrs: bool = False,
            *args, **kwargs
    ) -> (Union[None, str], str):
        """
        calculation by gaussion.
        for running the method normally, MAKE SURE THE Gaussian16 HAVE BEEN INSTALLED AND ALL ENV VAR SET RITHT !!
        Args:
            g16root:
            link0:
            route:
            path_log_file: the path to save the out.log file
            path_err_file: the path to save the error log file
            inplace_attrs: Whether to inplace self attribute according to the results from attributes
            *args:
            **kwargs:

        Returns:

        """

        # Make the input gjf script
        script = self.dump('gjf', *args, link0=link0, route=route, **kwargs)

        # Run Gaussian16
        gaussian = Gaussian(g16root)
        stdout, stderr = gaussian.run(script, args, **kwargs)

        # save the calculate result into the molecule data dict
        self._data['gaussian_output'] = stdout
        self._data['gaussian_parse_data'] = gaussian.data

        # Inplace the self attribute according to the result from gaussian
        if inplace_attrs:
            self._set_attrs(**gaussian.molecule_setter_dict)

        # Save log file
        if path_log_file:
            with open(path_log_file, 'w') as writer:
                writer.write(stdout)

        # Save error file
        if path_err_file:
            with open(path_err_file, 'w') as writer:
                writer.write(stderr)

        # return results and error info
        return stdout, stderr

    @property
    def identifier(self):
        return self.ob_mol.GetTitle()

    @identifier.setter
    def identifier(self, value):
        self.ob_mol.SetTitle(value)

    def iadd_accessible(self, other):
        if self.atomic_numbers == self.atomic_numbers:
            return True
        return False


    @property
    def is_labels_unique(self):
        """ Determine whether all atom labels are unique """
        labels = set(self.labels)
        if len(labels) == self.atom_num:
            return True
        return False

    @property
    def labels(self):
        return [a.label for a in self.atoms]

    @property
    def lmp(self):
        """ handle to operate the Lammps object """
        return self._data.get('lmp')

    def lmp_close(self):
        pop_lmp = self._data.pop('lmp')
        pop_lmp.close()

    def lmp_setup(self, *args, **kwargs):
        self._data['lmp'] = lmp.HpLammps(self, *args, **kwargs)

    @property
    def link_matrix(self):
        return np.array([[b.atom1_idx, b.atom2_idx] for b in self.bonds]).T

    @property
    def mol_orbital_energies(self):
        energies = self._data.get('mol_orbital_energies')
        if energies:
            return energies
        else:
            return None

    def normalize_labels(self):
        """ Reorder the atoms labels in the molecule """
        element_counts = {}
        for atom in self.atoms:
            count = element_counts.get(atom.symbol, 0)
            count += 1
            element_counts[atom.symbol] = count
            atom.label = f'{atom.symbol}{count}'

    def perturb_mol_lattice(
            self,
            random_style='uniform',
            mol_distance=0.5,
            lattice_fraction=0.05,
            freeze_dim: Sequence[int] = (),
            max_generate_num: int = 10,
            inplace: bool = False
    ) -> Generator["Molecule", None, None]:
        """
        Perturb the coordinate of atom in the mol or the lattice parameters
        generate new mol
        Args:
            random_style: how to sample, 'uniform' or 'normal'
            mol_distance: the max distance of perturbation in 'uniform'; the square variance in 'normal'
            lattice_fraction: the percentage of the lattice perturbation
            freeze_dim: tuple of int or str, 0 = x, 1 = y, 2 = z
            max_generate_num: the maximum of generated molecule
            inplace

        Returns:
            Generator of perturbed molecule
        """
        dim_transform = {'x': 0, 'y': 1, 'z': 2}

        coordinates_shape = (self.atom_num, 3)  # the shape of coordinates matrix (atom counts, 3 dimension)
        origin_coordinates = self.coordinates

        def coordinates_generator():
            """ Generating """
            for _ in range(max_generate_num):
                if random_style == 'uniform':
                    perturb_matrix = np.float64(np.random.uniform(-mol_distance, mol_distance, coordinates_shape))
                elif random_style == 'normal':
                    perturb_matrix = np.float64(np.random.normal(0, mol_distance, coordinates_shape))
                else:
                    raise ValueError('the perturb style is not defined!')

                if freeze_dim:
                    dim = [
                        i if (isinstance(i, int) and 0 <= i <= 3) else dim_transform[i]
                        for i in freeze_dim
                    ]

                    perturb_matrix[:, dim] = 0.

                new_coord = origin_coordinates + perturb_matrix

                yield new_coord

        def lattice_generator():
            """ TODO: this function is prepare to generate the new lattice """

        if inplace:
            origin_all_coordinates = self._data.get('all_coordinates')
            new_all_coordinates = np.array([c for c in coordinates_generator()])

            # TODO: test changes
            if origin_all_coordinates is not None:
                self._data['all_coordinates'] = np.concatenate([origin_all_coordinates, new_all_coordinates])
            else:
                self._data['all_coordinates'] = np.concatenate(
                    [np.reshape(origin_coordinates, (1,) + origin_coordinates.shape), new_all_coordinates]
                )

        else:
            return (self._pert_mol_generate(c) for c in coordinates_generator())

    @classmethod
    def readfile(cls, path_file: Union[str, PathLike], fmt=None, *args, **kwargs):
        """
        Construct Molecule by read file.
        This will in turn to try several method to Construct from several packages:
            1) `openbabel.pybel` module
            2) 'cclib' module
            3) custom mothod define by method
        Args:
            path_file:
            fmt:
            **kwargs:

        Returns:

        """
        if not fmt:
            if isinstance(path_file, str):
                path_file = Path(path_file)

            fmt = path_file.suffix.strip('.')

        # Try to read file by `openbabel`
        try:
            ob_mol = next(pb.readfile(fmt, str(path_file), **kwargs)).OBMol
            return cls(ob_mol, **kwargs)

        except StopIteration:
            # in the case, got Nothing from pybel.readfile.
            return None

        except ValueError:
            """ Fail to read file by 'pybel' module """

        # TODO:Try to read file by 'cclib'

        # Try to read file by custom reader
        custom_reader = retrieve_format(fmt)()
        mol_kwargs = custom_reader.read(path_file, *args, **kwargs)
        return cls(**mol_kwargs)

    @classmethod
    def read_from(cls, source: Union[str, PathLike, IOBase], fmt=None, *args, **kwargs) -> 'Molecule':
        """
        read source to the Molecule obj by call _io.Parser class
        Args:
            source(str, PathLike, IOBase): the formatted source
            fmt:
            *args:
            **kwargs:

        Returns:

        """
        if not fmt:
            if isinstance(source, str):
                source = Path(source)

            if isinstance(source, Path):
                fmt = source.suffix.strip('.')
            else:
                raise ValueError(f'the arguments should be specified for {type(source)} source')

        parser = Parser(fmt, source, *args, **kwargs)
        return parser()

    def remove_atoms(self, *atoms: Union[int, str, 'Atom']) -> None:
        """
        Remove atom according to given atom index, label or the atoms self.
        Args:
            atoms(int|str|Atom): the index, label or self of Removed atom

        Returns:
            None
        """
        for atom in atoms:

            # Check and locate the atom
            if isinstance(atom, int):
                atom = self.atoms[atom]
            elif isinstance(atom, str):
                atom = self.atoms[self.atom_labels.index(atom)]
            elif isinstance(atom, Atom):
                if not (atom.molecule is self):
                    raise AttributeError('the given atom not in the molecule')
            else:
                raise TypeError('the given atom should be int, str or Atom')

            # Removing atom
            self.ob_mol.DeleteAtom(atom.ob_atom)
            atom._data['_mol'] = None

    @property
    def rotatable_bonds_number(self):
        return self.ob_mol.NumRotors()

    def set(self, **kwargs):
        """ Set the attributes directly """
        self._set_attrs(**kwargs)

    def set_label(self, idx: int, label: str):
        self.atoms[idx].label = label

    @property
    def smiles(self):
        return self.dump('smi').strip().split('\t')[0]

    @property
    def spin(self):
        return self.ob_mol.GetTotalSpinMultiplicity()

    @spin.setter
    def spin(self, spin: int):
        self._set_spin_multiplicity(spin)

    def to_deepmd_train_data(
            self, path_save: Union[str, PathLike],
            valid_set_size: Union[int, float] = 0.2,
            is_test_set: bool = False,
            _call_by_bundle: bool = False
    ):
        """"""

    def to_mix_mol(self):
        return MixSameAtomMol(_data=self._data)

    def to_mol(self):
        return Molecule(self.copy_data())

    @property
    def weight(self):
        return self.ob_mol.GetExactMass()

    def writefile(self, fmt: str, path_file, *args, **kwargs):
        """Write the Molecule Info into a file with specific format(fmt)"""
        script = self.dump(fmt=fmt, *args, **kwargs)
        if isinstance(script, str):
            mode = 'w'
        elif isinstance(script, bytes):
            mode = 'wb'
        else:
            raise IOError(f'the {type(script)} type for script is not supported to write into file')

        with open(path_file, mode) as writer:
            writer.write(script)

    @property
    def xyz_min(self) -> np.ndarray:
        """ Return the minimum of x coordinates wreathing all atoms """
        return self.coordinates.min(axis=0)

    @property
    def xyz_max(self) -> np.ndarray:
        return self.coordinates.max(axis=0)

    @property
    def xyz_diff(self):
        return self.xyz_max - self.xyz_min


class MixSameAtomMol(Molecule):
    """ the only difference to the Molecule class is the method of their addition  """

    def iadd_accessible(self, other):
        if self.atom_num == other.atom_num:
            return True
        return False


class Atom(Wrapper, ABC):
    """ The Atom wrapper for OBAtom class in openbabel """

    def __init__(
            self,
            ob_atom: ob.OBAtom = None,
            **kwargs
    ):
        # Contain all data to reappear this Atom
        self._data = {
            'OBAtom': ob_atom if ob_atom else ob.OBAtom(),
            # '_mol': _mol,
        }

        self._set_attrs(**kwargs)

    @property
    def ob_atom(self):
        return self._data['OBAtom']

    @property
    def _mol(self) -> Molecule:
        return self._data.get('_mol')

    def __repr__(self):
        return f"Atom({self.label if self.label else self.symbol})"

    @property
    def _attr_setters(self) -> Dict[str, Callable]:
        return {
            "_mol": self._set_molecule,
            "mol": self._set_molecule,
            'molecule': self._set_molecule,
            "atomic_number": self._set_atomic_number,
            'symbol': self._set_atomic_symbol,
            "coordinates": self._set_coordinate,
            'partial_charge': self._set_partial_charge,
            'label': self._set_label,
            'idx': self._set_idx,
            'spin_density': self._set_spin_density
        }

    def _replace_attr_data(self, data: Dict):
        """ Replace the core data dict directly """
        self._data = data

    def _set_atomic_number(self, atomic_number: int):
        self.ob_atom.SetAtomicNum(int(atomic_number))

    def _set_atomic_symbol(self, symbol):
        atomic_number = periodic_table[symbol]['number']
        self.ob_atom.SetAtomicNum(atomic_number)

    def _set_coordinate(self, coordinates):
        self.ob_atom.SetVector(*coordinates)

    def _set_force_vector(self, force_vector: Union[Sequence, np.ndarray]):
        if isinstance(force_vector, Sequence):
            if all(isinstance(f, float) for f in force_vector):
                force_vector = np.array(force_vector)
            else:
                ValueError('the give force_vector must float vector with dimension 3')
        elif isinstance(force_vector, np.ndarray):
            force_vector = force_vector.flatten()
        else:
            raise TypeError('the force vector should be Sequence or np.ndarray')

        self._data['force_vector'] = force_vector

    def _set_idx(self, idx):
        self.ob_atom.SetId(idx)

    def _set_label(self, label):
        self._data['label'] = label

    def _set_molecule(self, molecule: Molecule):
        self._data['_mol'] = molecule

    def _set_partial_charge(self, charge):
        self.ob_atom.SetPartialCharge(charge)

    def _set_spin_density(self, spin_density: float):
        self._data['spin_density'] = spin_density

    @property
    def atom_type(self):
        """ Some atom have specific type, such as Carbon with sp1, sp2 and sp3, marked as C1, C2 and C3 """
        return self.ob_atom.GetType()

    @property
    def atomic_number(self):
        return self.ob_atom.GetAtomicNum()

    @property
    def coordinates(self):
        return self.ob_atom.GetX(), self.ob_atom.GetY(), self.ob_atom.GetZ()

    @coordinates.setter
    def coordinates(self, value):
        self._set_coordinate(value)

    def copy(self):
        """ Make a copy of self """
        # Extract old data
        new_attrs = {
            "atomic_number": self.atomic_number,
            "coordinates": self.coordinates,
            'partial_charge': self.partial_charge,
            'label': self.label,
            'idx': self.idx
        }

        return Atom(**new_attrs)

    def element_features(self, *feature_names, output_type='dict'):
        atom_feature = periodic_table.get(self.symbol)
        features = {}
        for name in feature_names:
            if name == "atom_orbital_feature":
                features['atom_orbital_feature'] = self._atomic_orbital_feature()
            else:
                features[name] = atom_feature[name]
        if output_type == 'list':
            features = [features[name] for name in feature_names]

        return features

    def _atomic_orbital_feature(self, outermost_layer=True, nonexistent_orbit=0):
        """    Calculating the feature about atomic orbital structures    """
        _atomic_orbital_structure_max = {
            "1s": 2,
            "2s": 2, "2p": 6,
            "3s": 2, "3p": 6,
            "4s": 2, "3d": 10, "4p": 6,
            "5s": 2, "4d": 10, "5p": 6,
            "6s": 2, "4f": 14, "5d": 10, "6p": 6,
            "7s": 2, "5f": 14, "6d": 10, "7p": 6
        }
        atomic_orbital_structure = {
            "1s": 0,
            "2s": 0, "2p": 0,
            "3s": 0, "3p": 0,
            "4s": 0, "3d": 0, "4p": 0,
            "5s": 0, "4d": 0, "5p": 0,
            "6s": 0, "4f": 0, "5d": 0, "6p": 0,
            "7s": 0, "5f": 0, "6d": 0, "7p": 0
        }

        # Calculating atomic orbital structure
        residual_electron = self.atomic_number
        n_osl = 0  # Principal quantum number (n) of open shell layers (osl)
        for orbital_name, men in _atomic_orbital_structure_max.items():  # max electron number (men)

            # Update Principal quantum number (n)
            if orbital_name[1] == "s":
                n_osl = int(orbital_name[0])

            # Filled atomic orbital
            if residual_electron - men >= 0:
                residual_electron = residual_electron - men
                atomic_orbital_structure[orbital_name] = men
            else:
                atomic_orbital_structure[orbital_name] = residual_electron
                break

        # Readout and return outermost electron structure
        atom_orbital_feature = {"atomic_number": self.atomic_number, "n_osl": n_osl}
        if outermost_layer:
            diff_max_n = {"s": 0, "p": 0, "d": -1, "f": -2}
            for layer, diff in diff_max_n.items():  # Angular momentum quantum number (l)
                electron_number = atomic_orbital_structure.get(f"{n_osl + diff}{layer}", nonexistent_orbit)
                atom_orbital_feature[layer] = electron_number
        else:
            atom_orbital_feature.update(atomic_orbital_structure)

        # return whole electron structure directly
        return atom_orbital_feature

    # def _rings_feature(self):
    #     """    Calculating feature about the rings    """
    #     atom_rings_feature = {
    #         "is_aromatic": 0,  # Whether the atoms on aromatic rings
    #         3: 0,  # How many 3-members rings is the atom on
    #         4: 0,  # How many 4-members rings is the atom on
    #         5: 0,  # How many 5-members rings is the atom on
    #         6: 0,  # How many 6-members rings is the atom on
    #         7: 0,  # How many 7-members rings is the atom on
    #         8: 0  # How many 8-members rings is the atom on
    #     }
    #     if self.is_cyclic:
    #         rings = self.rings
    #         if any(ring.is_aromatic for ring in rings):
    #             atom_rings_feature["is_aromatic"] = 1
    #         list_ring_member = [len(ring.bonds) for ring in rings]
    #         for rmb in list_ring_member:  # for each ring members
    #             if rmb in [3, 4, 5, 6, 7, 8]:
    #                 atom_rings_feature[rmb] += 1
    #
    #     return atom_rings_feature
    @property
    def force_vector(self):
        return self._data.get('force_vector', np.zeros(3, dtype=float))

    @force_vector.setter
    def force_vector(self, force_vector: Union[Sequence, np.ndarray]):
        self._set_force_vector(force_vector)

    @property
    def kwargs_attributes(self):
        return tuple(self._attr_setters.keys())

    @property
    def idx(self):
        return self.ob_atom.GetId()

    @property
    def is_aromatic(self):
        return self.ob_atom.IsAromatic()

    @property
    def is_metal(self):
        return self.ob_atom.IsMetal()

    @property
    def is_chiral(self):
        return self.ob_atom.IsChiral()

    @property
    def label(self):
        return self._data.get('label', self.symbol)

    @label.setter
    def label(self, value):
        self._set_label(value)

    @property
    def mass(self):
        return self.ob_atom.GetAtomicMass()

    @property
    def molecule(self) -> Molecule:
        return self._mol

    @property
    def neighbours(self):
        """ Get all atoms bond with this atom in same molecule """
        if self._mol:
            return [self._mol.atoms[OBAtom.GetId()] for OBAtom in ob.OBAtomAtomIter(self.ob_atom)]
        else:
            return []

    @property
    def partial_charge(self):
        return self.ob_atom.GetPartialCharge()

    @partial_charge.setter
    def partial_charge(self, value: float):
        # This is necessary to take effect to the assignment.
        # the reason is unknown
        self.ob_atom.GetPartialCharge()
        self._set_partial_charge(value)

    @property
    def spin_density(self):
        return self._data.get('spin_density', 0.0)

    @spin_density.setter
    def spin_density(self, spin_density: float):
        self._set_spin_density(spin_density)

    @property
    def symbol(self):
        return list(periodic_table.keys())[self.atomic_number - 1]


class Bond(Wrapper, ABC):
    """"""

    def __init__(self, ob_bond: ob.OBBond, _mol: Molecule):
        self._data = {
            "OBBond": ob_bond,
            'mol': _mol
        }

    def __repr__(self):
        return f"Bond({self.atoms[0].label}, {self.atoms[1].label}, {self.type_name})"

    @property
    def _OBBond(self):
        return self._data['OBBond']

    @property
    def _attr_setters(self) -> Dict[str, Callable]:
        return {
        }

    @property
    def atom1(self):
        return self.atoms[0]

    @property
    def atom2(self):
        return self.atoms[1]

    @property
    def atom1_idx(self):
        return self.atom1.idx

    @property
    def atom2_idx(self):
        return self.atom2.idx

    @property
    def atoms(self):
        atoms = self.molecule.atoms
        begin_idx, end_idx = self.begin_end_idx
        return atoms[begin_idx], atoms[end_idx]

    @property
    def begin_end_idx(self):
        return self._OBBond.GetBeginAtomIdx() - 1, self._OBBond.GetEndAtomIdx() - 1

    @property
    def ideal_length(self):
        return self._OBBond.GetEquibLength()

    @property
    def length(self):
        return self._OBBond.GetLength()

    @property
    def molecule(self):
        return self._data['mol']

    @property
    def type_name(self):
        return _type_bond[self.type]

    @property
    def type(self):
        return self._OBBond.GetBondOrder()


class Crystal(Wrapper, ABC):
    """"""
    _lattice_type = (
        'Undefined', 'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Rhombohedral', 'Hexagonal', 'Cubic'
    )

    def __init__(self, ob_unitcell: ob.OBUnitCell = None, **kwargs):

        self._data = {
            'OBUnitCell': ob_unitcell if ob_unitcell else ob.OBUnitCell(),
        }

        self._set_attrs(**kwargs)

    def __repr__(self):
        return f'Crystal({self.lattice_type}, {self.space_group}, {self.molecule})'

    @property
    def _OBUnitCell(self) -> ob.OBUnitCell:
        return self._data.get('OBUnitCell')

    def _set_molecule(self, molecule: Molecule):
        if molecule.crystal and isinstance(molecule.crystal, Crystal):
            print(AttributeError("the Molecule have been stored in a Crystal, "
                                 "can't save the same Molecule into two Crystals"))
        else:
            self._data['mol'] = molecule

    def _set_space_group(self, space_group: str):
        self._OBUnitCell.SetSpaceGroup(space_group)

    @property
    def _attr_setters(self) -> Dict[str, Callable]:
        return {
            'mol': self._set_molecule,
            'molecule': self._set_molecule,
            'space_group': self._set_space_group
        }

    @property
    def lattice_type(self) -> str:
        return self._lattice_type[self._OBUnitCell.GetLatticeType()]

    @property
    def lattice_params(self) -> np.ndarray[2, 3]:
        a = self._OBUnitCell.GetA()
        b = self._OBUnitCell.GetB()
        c = self._OBUnitCell.GetC()
        alpha = self._OBUnitCell.GetAlpha()
        beta = self._OBUnitCell.GetBeta()
        gamma = self._OBUnitCell.GetGamma()
        return np.array([[a, b, c], [alpha, beta, gamma]])

    @property
    def molecule(self) -> Molecule:
        return self._data.get('mol')

    @property
    def pack_molecule(self) -> Molecule:
        mol = self.molecule  # Get the contained Molecule

        if not mol:  # If get None
            print(RuntimeWarning("the crystal doesn't contain any Molecule!"))
            return None

        pack_mol = mol.copy()
        self._OBUnitCell.FillUnitCell(pack_mol.ob_mol)  # Full the crystal
        pack_mol._reorganize_atom_indices()  # Rearrange the atom indices.

        return pack_mol

    def set_lattice(
            self,
            a: float, b: float, c: float,
            alpha: float, beta: float, gamma: float
    ):
        self._OBUnitCell.SetData(a, b, c, alpha, beta, gamma)

    @property
    def space_group(self):
        space_group = self._OBUnitCell.GetSpaceGroup()
        if space_group:
            return space_group.GetHMName()
        else:
            return None

    @space_group.setter
    def space_group(self, value: str):
        self._set_space_group(value)

    @property
    def volume(self):
        return self._OBUnitCell.GetCellVolume()

    @property
    def vector(self):
        v1, v2, v3 = self._OBUnitCell.GetCellVectors()
        return np.array([
            [v1.GetX(), v1.GetY(), v1.GetZ()],
            [v2.GetX(), v2.GetY(), v2.GetZ()],
            [v3.GetX(), v3.GetY(), v3.GetZ()]
        ])

    def zeo_plus_plus(self):
        """ TODO: complete the method after define the Crystal and ZeoPlusPlus tank """


from src.bundle import MolBundle
