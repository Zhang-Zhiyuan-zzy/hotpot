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

dir_root = os.path.join(os.path.dirname(__file__))
_elements = json.load(open(f'{dir_root}/../data/periodic_table.json', encoding='utf-8'))['elements']
_elements = {d['symbol']: d for d in _elements}

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
    def __init__(self, ob_mol: ob.OBMol = None, **kwargs):
        self._data = {
            'ob_mol': ob_mol if ob_mol else ob.OBMol()
        }
        self._set_attrs(**kwargs)

    def __repr__(self):
        return f'Mol({self.ob_mol.GetFormula()})'

    @property
    def _OBAtom_indices(self):
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
    def _assign_atom_coords(the_mol: 'Molecule', coord_matrix: np.ndarray):
        """ Assign coordinates for all atoms in the Molecule """
        if len(the_mol.atoms) != coord_matrix.shape[-2]:
            raise AttributeError('the coordinate matrix do not match the number of atoms')

        for new_mol_atom, new_atom_coord in zip(the_mol.atoms, coord_matrix):
            new_mol_atom.coordinates = new_atom_coord

    @property
    def _attr_setters(self) -> Dict[str, Callable]:
        return {
            'atoms.partial_charge': self._set_atoms_partial_charge,
            "identifier": self._set_identifier,
            "energy": self._set_mol_energy,
            'energies': self._set_mol_energies,
            'charge': self._set_mol_charge,
            'spin': self._set_spin_multiplicity,
            'atoms': self._set_atoms,
            'mol_orbital_energies': self._set_mol_orbital_energies,
            'coord_collect': self._set_coord_collect,
        }

    @property
    def _coord_collect(self):
        coord_collect = self._data.get('_coord_collect')
        if isinstance(coord_collect, np.ndarray):
            return coord_collect
        else:
            return self.coord_matrix

    def _pert_mol_generate(self, coord_matrix: np.ndarray):
        """
        Generate new molecule obj according to new given coordinate
        Args:
            coord_matrix: New coordinates matrix

        Returns:
            Molecule, copy of this molecule with new coordinates
        """
        clone_mol = self.copy()
        self._assign_atom_coords(clone_mol, coord_matrix)
        return clone_mol

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

        if len(self.atoms) != len(partial_charges):
            raise ValueError('the given partial charges should have same numbers with the number of atoms')

        for atom, partial_charge in zip(self.atoms, partial_charges):
            atom.partial_charge = partial_charge

    def _set_coord_collect(self, coord_collect: np.ndarray):
        """
        Assign the coordinates collection directly
        Args:
            coord_collect: numpy array with the shape (M, N, 3), where the M is the number of coordinates
            in the collection, the N is the number of atoms of the molecule.

        Returns:
            None
        """
        if not isinstance(coord_collect, np.ndarray):
            raise ValueError(f'the given coord_collect must be a numpy.ndarray class, instead of {type(coord_collect)}')

        if coord_collect.shape[-1] != 3:
            raise ValueError(f'the coordinate must be 3 dimension, instead of {coord_collect.shape[-1]}')

        if len(coord_collect.shape) == 2:
            # if only give a group of coordinates
            coord_collect = coord_collect.reshape((-1, coord_collect.shape[-2], 3))
        elif len(coord_collect.shape) != 3:
            raise ValueError(
                f'the shape of given coord_collect should with length 2 or 3, now is {len(coord_collect.shape)}'
            )

        self._data['_coord_collect'] = coord_collect

    def _set_mol_charge(self, charge: int):
        self.ob_mol.SetTotalCharge(charge)

    def _set_mol_orbital_energies(self, orbital_energies: list[np.ndarray]):
        self._data['mol_orbital_energies'] = orbital_energies[0]

    def _set_mol_energy(self, energy: float):
        """ set the energy """
        self.ob_mol.SetEnergy(energy)

    def _set_mol_energies(self, energies: Union[float, np.ndarray], config_index: Optional[int] = None):
        """ set the energies vector """
        if isinstance(energies, float):
            self._data['energies'] = np.array([energies])
        else:
            energies = energies.flatten()
            self._data['energies'] = energies

        if isinstance(config_index, int):
            try:
                energy = energies[config_index]
            except IndexError:
                energy = 0.0

            self._set_mol_energy(energy)

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

        elif atom_idx[0] not in self.atoms_indices:
            raise KeyError("the start atom1 doesn't exist in molecule")

        elif atom_idx[1] not in self.atoms_indices:
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
        ob_atom_indices = self._OBAtom_indices  # the sorted indices list for OBAtom in OBMol

        if not atoms:
            atoms = self._data['atoms'] = [
                Atom(OBAtom=self.ob_mol.GetAtomById(idx), _mol=self)
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
    def atoms_indices(self) -> list[int]:
        return [a.idx for a in self.atoms]

    @property
    def atom_labels(self):
        return [a.label for a in self.atoms]

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
        coordinate_matrix_collection = self._data.get('_coord_collect')
        if isinstance(coordinate_matrix_collection, np.ndarray):
            return coordinate_matrix_collection.shape[0]
        else:
            return 1

    def configure_select(self, config_idx: int):
        """ select specific configure by index """
        coordinate_matrix_collection = self._data.get('_coord_collect')
        if coordinate_matrix_collection is None and config_idx:
            raise IndexError('Only one configure here!')

        # assign the coordinates for the molecule
        config_coord_matrix = coordinate_matrix_collection[config_idx]
        self._assign_atom_coords(self, config_coord_matrix)

        energies = self._data.get('energies')
        if isinstance(energies, np.ndarray):
            try:
                energy = energies[config_idx]
            except IndexError:
                # if can't find corresponding energy
                energy = 0.0

            self._set_mol_energy(energy)

        else:
            self._set_mol_energy(0.0)

    @property
    def coord_matrix(self) -> np.ndarray:
        """
        Get the matrix of all atoms coordinates,
        where the row index point to the atom index;
        the column index point to the (x, y, z)
        """
        return np.array([atom.coordinates for atom in self.atoms], dtype=np.float64)

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
            coord_collect = self._data.pop('_coord_collect')
        except KeyError:
            coord_collect = None

        if pop:
            return coord_collect

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
        atomic_number = _elements[symbol]['number']
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
    def energies_vector(self):
        return self._data.get('energies')

    @property
    def energy(self):
        """ Return energy with kcal/mol as default """
        return self.ob_mol.GetEnergy()

    def feature_matrix(self, *feature_names):
        """"""

    @property
    def formula(self) -> str:
        return self.ob_mol.GetFormula()

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

    @property
    def is_labels_unique(self):
        """ Determine whether all atom labels are unique """
        labels = set(self.labels)
        if len(labels) == len(self.atoms):
            return True
        return False

    @property
    def labels(self):
        return [a.label for a in self.atoms]

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

        coord_matrix_shape = (len(self.atoms), 3)  # the shape of coordinates matrix (atom counts, 3 dimension)
        origin_coord_matrix = self.coord_matrix

        def coordinates_generator():
            """ Generating """
            for _ in range(max_generate_num):
                if random_style == 'uniform':
                    perturb_matrix = np.float64(np.random.uniform(-mol_distance, mol_distance, coord_matrix_shape))
                elif random_style == 'normal':
                    perturb_matrix = np.float64(np.random.normal(0, mol_distance, coord_matrix_shape))
                else:
                    raise ValueError('the perturb style is not defined!')

                if freeze_dim:
                    dim = [
                        i if (isinstance(i, int) and 0 <= i <= 3) else dim_transform[i]
                        for i in freeze_dim
                    ]

                    perturb_matrix[:, dim] = 0.

                new_coord = origin_coord_matrix + perturb_matrix

                yield new_coord

        def lattice_generator():
            """ TODO: this function is prepare to generate the new lattice """

        if inplace:
            origin_coord_collect = self._data.get('_coord_collect')
            new_coord_collect = np.array([c for c in coordinates_generator()])

            # TODO: test changes
            if origin_coord_collect is not None:
                self._data['_coord_collect'] = np.concatenate([origin_coord_collect, new_coord_collect])
            else:
                self._data['_coord_collect'] = np.concatenate(
                    [np.reshape(origin_coord_matrix, (1,) + origin_coord_matrix.shape), new_coord_collect]
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
    def read_from(cls, source: Union[str, PathLike, IOBase], fmt=None, *args, **kwargs):
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
            atom(int|str|Atom): the index, label or self of Removed atom

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
                if not(atom.molecule is self):
                    raise AttributeError('the given atom not in the molecule')
            else:
                raise TypeError('the given atom should be int, str or Atom')

            # Removing atom
            self.ob_mol.DeleteAtom(atom.ob_atom)
            atom._data['_mol'] = None

    @property
    def rotatable_bonds_number(self):
        return self.ob_mol.NumRotors()

    def save_coord_matrix_to_npy(
            self, save_path: Union[str, PathLike],
            which: Literal['present', 'all'] = 'present',
            **kwargs
    ):
        """"""
        if which == 'present':
            np.save(save_path, self.coord_matrix, **kwargs)
        elif which == 'all':
            np.save(save_path, self._coord_collect, **kwargs)
        else:
            return ValueError(f"the which should be `present` or `all`, instead of `{which}`")

    def set(self, **kwargs):
        """ Set the attributes directly """
        self._set_attrs(**kwargs)

    def set_label(self, idx: int, label: str):
        self.atoms[idx].label = label

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


class Atom(Wrapper, ABC):
    """ The Atom wrapper for OBAtom class in openbabel """
    def __init__(
        self,
        OBAtom: ob.OBAtom = None,
        **kwargs
    ):
        # Contain all data to reappear this Atom
        self._data = {
            'OBAtom': OBAtom if OBAtom else ob.OBAtom(),
            # '_mol': _mol,
        }

        self._set_attrs(**kwargs)

    @property
    def ob_atom(self):
        return self._data['OBAtom']

    @property
    def _mol(self):
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
        atomic_number = _elements[symbol]['number']
        self.ob_atom.SetAtomicNum(atomic_number)

    def _set_coordinate(self, coordinates):
        self.ob_atom.SetVector(*coordinates)

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

    def element_feature(self, *feature_name: str):
        """输入以列特征名称， 返回这些特征值的向量"""

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
        self._set_partial_charge(value)

    @property
    def spin_density(self):
        return self._data.get('spin_density')

    @spin_density.setter
    def spin_density(self, spin_density: float):
        self._set_spin_density(spin_density)

    @property
    def symbol(self):
        return list(_elements.keys())[self.atomic_number - 1]


class Bond(Wrapper, ABC):
    """"""
    def __init__(self, _OBBond: ob.OBBond, _mol: Molecule):
        self._data = {
            "OBBond": _OBBond,
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
        return self._OBBond.GetBeginAtomIdx()-1, self._OBBond.GetEndAtomIdx()-1

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

    def __init__(self, _OBUnitCell: ob.OBUnitCell = None, **kwargs):

        self._data = {
            'OBUnitCell': _OBUnitCell if _OBUnitCell else ob.OBUnitCell(),
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
        return self._OBUnitCell.GetSpaceGroup().GetHMName()

    @space_group.setter
    def space_group(self, value: str):
        self._set_space_group(value)

    @property
    def volume(self):
        return self._OBUnitCell.GetCellVolume()

    def zeo_plus_plus(self):
        """ TODO: complete the method after define the Crystal and ZeoPlusPlus tank """
