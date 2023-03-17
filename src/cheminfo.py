"""
python v3.7.9
@Project: hotpot
@File   : cheminfo.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/14
@Time   : 4:09
"""
import os
from os import PathLike
from pathlib import Path
from abc import ABC, abstractmethod
import json
from typing import *
import numpy as np
from openbabel import openbabel as ob, pybel as pb
from src._io import retrieve_format, Dumper
import subprocess

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


class Molecule(Wrapper, ABC):
    """"""
    def __init__(self, OBMol: ob.OBMol = None, **kwargs):
        self._data = {
            'OBMol': OBMol if OBMol else ob.OBMol()
        }
        self._set_attrs(**kwargs)

    def __repr__(self):
        return f'Mol({self._OBMol.GetFormula()})'

    @property
    def _OBAtom_indices(self):
        """ Get the indices for all OBAtom """
        indices = []

        try:
            num_OBAtoms = self._OBMol.NumAtoms()
        # If there is none of atoms in the OBMol, raise the TypeError.
        except TypeError:
            num_OBAtoms = 0

        idx = 0
        while len(indices) < num_OBAtoms:
            OBAtom = self._OBMol.GetAtomById(idx)

            # if get a OBAtom
            if OBAtom:
                assert idx == OBAtom.GetId()
                indices.append(idx)

            idx += 1

        return indices

    @property
    def _OBMol(self):
        return self._data['OBMol']

    @property
    def _attr_setters(self) -> Dict[str, Callable]:
        return {
            "identifier": self._set_identifier,
            "energy": self._set_mol_energy,
            'charge': self._set_mol_charge,
            'spin': self._set_spin_multiplicity,
            'atoms': self._set_atoms
        }

    def _set_atoms(self, atoms_kwargs: List[Dict[str, Any]]):
        """ add a list of atoms by a list atoms attributes dict """
        for atom_kwarg in atoms_kwargs:
            a = Atom(**atom_kwarg)
            self.add_atom(a)

    def _set_mol_charge(self, charge: int):
        self._OBMol.SetTotalCharge(charge)

    def _set_mol_energy(self, energy: float):
        self._OBMol.SetEnergy(energy)

    def _set_identifier(self, identifier):
        self._OBMol.SetTitle(identifier)

    def _set_spin_multiplicity(self, spin):
        self._OBMol.SetTotalSpinMultiplicity(spin)

    def add_atom(self, atom: "Atom"):
        """
        Add a new atom out of the molecule into the molecule.
        Args:
            atom:

        Returns:

        """
        # Avoid add a existed atom into the molecule
        if atom.molecule and (atom.molecule is self or atom.molecule._OBMol is self._OBMol):
            raise AttributeError("This atom have exist in the molecule")

        success = self._OBMol.AddAtom(atom._OBAtom)
        if success:

            # the OBAtom have stored in the OBMol and self
            # get atom by idx enumerate from 1, instead of 0.
            OBAtom_in_OBMol = self._OBMol.GetAtom(self._OBMol.NumAtoms())

            # Get the attribute dict and replace the 'OBAtom' and '_mol' items
            # The 'OBAtom' is the OBAtom has been stored in the self(Molecule)
            # The '_mol' is the self(Molecule)
            new_atom_data = atom._data
            new_atom_data['OBAtom'] = OBAtom_in_OBMol
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
            **kwargs
    ):
        """"""
        atoms = (atom1, atom2)
        atom_idx = []
        for a in atoms:
            if isinstance(a, int):
                atom_idx.append(a)
            if isinstance(a, Atom):
                atom_idx.append(a.idx)
            if isinstance(a, str):
                atom_idx.append(self.atom_from_label(a).idx)

        # Represent the bond type by int, refer to _bond_type dict
        bond_type = bond_type if isinstance(bond_type, int) else _bond_type[bond_type]

        # Try to add new OBMol
        # 'openbabel' has an odd behave that `index` of the `OBAtom` with various origin in the `OBMol`.
        # the `Id` of `OBAtom` from 0; but the `Idx` of `OBAtom` from 1.
        # To meet the convention, the `Id` is selected to be the unique `index` to specify `Atom`.
        # However, when try to add a `OBBond` to link each two `OBAtoms`, the `Idx` is the only method
        # to specify the atoms, so our `index` in `Atom` are added 1 to match the 'Idx'
        success = self._OBMol.AddBond(atom_idx[0]+1, atom_idx[1]+1, bond_type)

        if success:
            new_bond_idx = self._OBMol.NumBonds() - 1
            new_OBBond = self._OBMol.GetBondById(new_bond_idx)
            bond = Bond(new_OBBond, self)

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

    def atom_from_label(self, label: str) -> 'Atom':
        """ get atom by label """
        if not self.is_labels_unique:
            print(AttributeError('the molecule atoms labels are not unique!'))
            return

        return self.atoms[self.labels.index(label)]

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
        OBAtom_indices = self._OBAtom_indices  # the sorted indices list for OBAtom in OBMol

        if not atoms:
            atoms = self._data['atoms'] = [
                Atom(OBAtom=self._OBMol.GetAtomById(idx), _mol=self)
                for idx in OBAtom_indices
            ]

        # If the order of atom index by atoms is non-match with the order by OBAtoms
        elif len(atoms) != len(OBAtom_indices) or any(ai != Ai for ai, Ai in zip(atoms_indices, OBAtom_indices)):
            new_atoms = []
            for OBAtom_idx in OBAtom_indices:

                try:  # If the atom has existed in the old atom list, append it into new list with same order of OBAtoms
                    atom_idx = atoms_indices.index(OBAtom_idx)
                    new_atoms.append(atoms[atom_idx])

                # If the atom doesn't exist in the old atom, create a new atoms and append to new list
                except ImportError:
                    new_atoms.append(Atom(self._OBMol.GetAtomById(OBAtom_idx), _mol=self))

            # Update the atoms list
            atoms = self._data['atoms'] = new_atoms

        return atoms

    @property
    def atoms_indices(self) -> list[int]:
        return [a.idx for a in self.atoms]

    @property
    def bonds(self):
        bonds = self._data.get('bonds')
        if not bonds:
            bonds = self._data['bonds'] = [
                Bond(self._OBMol.GetBondById(i), _mol=self)
                for i in range(self._OBMol.NumBonds())
            ]

        return bonds

    @property
    def atom_labels(self):
        return [a.label for a in self.atoms]

    @property
    def charge(self):
        return self._OBMol.GetTotalCharge()

    @charge.setter
    def charge(self, charge):
        self._set_mol_charge(charge)

    @property
    def coord_matrix(self) -> np.ndarray:
        """
        Get the matrix of all atoms coordinates,
        where the row index point to the atom index;
        the column index point to the (x, y, z)
        """
        return np.array([atom.coordinates for atom in self.atoms], dtype=np.float64)

    def copy(self):
        """ Copy the Molecule """
        # Create the new data sheet
        new_data = {
            'OBMol': ob.OBMol(),
            'atoms': [],
            'bonds': []
        }

        # Copy all attribute out the
        for name, value in self._data.items():
            if name not in new_data:
                new_data[name] = value

        # Create new Molecule
        clone_mol = Molecule()
        clone_mol._data = new_data
        clone_mol.identifier = f"{self.identifier}_clone"

        # copy, in turn, each Atom in this molecule and add them into new Molecule
        for atom in self.atoms:
            clone_atom = atom.copy()
            clone_atom_in_new_mol = clone_mol.add_atom(clone_atom)

        # generate Bonds in new Molecule with same graph pattern in this Molecule
        for bond in self.bonds:
            new_bond = clone_mol.add_bond(*bond.atoms, bond_type=bond.type)

        return clone_mol

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
        OBAtom: ob.OBAtom = self._OBMol.NewAtom()
        atomic_number = _elements[symbol]['number']
        OBAtom.SetAtomicNum(atomic_number)
        atom = Atom(OBAtom, mol=self, **kwargs)

        return atom

    def dump(self, fmt: str, *args, **kwargs) -> Union[str, bytes]:
        """"""
        dumper = Dumper(fmt=fmt, mol=self, *args, **kwargs)
        return dumper.dump()

    @property
    def energy(self):
        """ Return energy with kcal/mol as default """
        return self._OBMol.GetEnergy()

    def gaussian(
            self,
            link0: str,
            route: str,
            path_log_file: Union[str, PathLike] = None,
            *args, **kwargs
    ) -> (Union[None, str], str):
        """
        calculation by gaussion.
        for running the method normally, MAKE SURE THE Gaussian16 HAVE BEEN INSTALLED AND ALL ENV VAR SET RITHT !!
        Args:
            link0:
            route:
            path_log_file:
            *args:
            **kwargs:

        Returns:

        """

        script = self.dump('gjf', *args, link0=link0, route=route, **kwargs)

        # Run Gaussian using subprocess
        g16_process = subprocess.Popen(['g16'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       universal_newlines=True)
        stdout, stderr = g16_process.communicate(script)

        if path_log_file:
            with open(path_log_file, 'w') as writer:
                writer.write(stdout)
        else:
            return stdout, stderr

    @property
    def identifier(self):
        return self._OBMol.GetTitle()

    @identifier.setter
    def identifier(self, value):
        self._OBMol.SetTitle(value)

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
            max_generate_num: int = 1000
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

        Returns:
            Generator of perturbed molecule
        """
        dim_transform = {'x': 0, 'y': 1, 'z': 2}

        coord_matrix_shape = (len(self.atoms), 3)  # the shape of coordinates matrix (atom counts, 3 dimension)
        origin_coord_matrix = self.coord_matrix

        def perturbed_mol(nc: np.ndarray):
            """
            Generate new
            Args:
                nc: New coordinates

            Returns:

            """
            clone_mol = self.copy()

            for new_mol_atom, new_atom_coord in zip(clone_mol.atoms, nc):
                new_mol_atom.coordinates = new_atom_coord

            return clone_mol

        # Generating
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

            yield perturbed_mol(new_coord)

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
            OBMol = next(pb.readfile(fmt, str(path_file), **kwargs)).OBMol
            return cls(OBMol, **kwargs)

        except ValueError:
            """ Fail to read file by 'pybel' module """

        # TODO:Try to read file by 'cclib'

        # Try to read file by custom reader
        custom_reader = retrieve_format(fmt)()
        mol_kwargs = custom_reader.read(path_file, *args, **kwargs)
        return cls(**mol_kwargs)

    def set_label(self, idx: int, label: str):
        self.atoms[idx].label = label

    @property
    def spin(self):
        return self._OBMol.GetTotalSpinMultiplicity()

    @spin.setter
    def spin(self, spin: int):
        self._set_spin_multiplicity(spin)

    @property
    def weight(self):
        return self._OBMol.GetExactMass()

    def writefile(self, fmt: str, path_file, *args, **kwargs):
        """Write the Molecule Info into a file with specific format(fmt)"""
        script = self.dump(fmt=fmt, *args, **kwargs)
        if isinstance(script, str):
            mode = 'w'
        elif isinstance(script, bytes):
            mode = 'b'
        else:
            raise IOError(f'the {type(script)} type for script is not supported to write into file')

        with open(path_file, 'w') as writer:
            writer.write(script)


class Atom(Wrapper, ABC):
    """ The Atom wrapper for OBAtom class in openbabel """
    def __init__(
        self,
        OBAtom: ob.OBAtom = None,
        _mol: Molecule = None,
        **kwargs
    ):
        # Contain all data to reappear this Atom
        self._data = {
            'OBAtom': OBAtom if OBAtom else ob.OBAtom(),
            '_mol': _mol,
        }

        self._set_attrs(**kwargs)

    @property
    def _OBAtom(self):
        return self._data['OBAtom']

    @property
    def _mol(self):
        return self._data['_mol']

    def __repr__(self):
        return f"Atom({self.label if self.label else self.symbol})"

    @property
    def _attr_setters(self) -> Dict[str, Callable]:
        return {
            "atomic_number": self._set_atomic_number,
            'symbol': self._set_atomic_symbol,
            "coordinates": self._set_coordinate,
            'partial_charge': self._set_partial_charge,
            'label': self._set_label,
            'idx': self._set_idx
        }

    def _replace_attr_data(self, data: Dict):
        """ Replace the core data dict directly """
        self._data = data

    def _set_atomic_number(self, atomic_number: int):
        self._OBAtom.SetAtomicNum(atomic_number)

    def _set_coordinate(self, coordinates):
        self._OBAtom.SetVector(*coordinates)

    def _set_idx(self, idx):
        self._OBAtom.SetId(idx)

    def _set_label(self, label):
        self._data['label'] = label

    def _set_molecule(self, molecule: Molecule):
        atom_property = getattr(self._OBAtom, 'property')
        atom_property['molecule'] = molecule

    def _set_partial_charge(self, charge):
        self._OBAtom.SetPartialCharge(charge)

    def _set_atomic_symbol(self, symbol):
        atomic_number = _elements[symbol]['number']
        self._OBAtom.SetAtomicNum(atomic_number)

    @property
    def atom_type(self):
        """ Some atom have specific type, such as Carbon with sp1, sp2 and sp3, marked as C1, C2 and C3 """
        return self._OBAtom.GetType()

    @property
    def atomic_number(self):
        return self._OBAtom.GetAtomicNum()

    @property
    def coordinates(self):
        return self._OBAtom.GetX(), self._OBAtom.GetY(), self._OBAtom.GetZ()

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

    @property
    def kwargs_attributes(self):
        return tuple(self._attr_setters.keys())

    @property
    def neighbours(self):
        """ Get all atoms bond with this atom in same molecule """
        if self._mol:
            return [self._mol.atoms[OBAtom.GetId()] for OBAtom in ob.OBAtomAtomIter(self._OBAtom)]
        else:
            return []

    @property
    def idx(self):
        return self._OBAtom.GetId()

    @property
    def label(self):
        return self._data.get('label', self.symbol)

    @label.setter
    def label(self, value):
        self._set_label(value)

    @property
    def mass(self):
        return self._OBAtom.GetAtomicMass()

    @property
    def molecule(self):
        return self._mol

    @property
    def partial_charge(self):
        return self._OBAtom.GetPartialCharge()

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
        return f"Bond({self.atoms[0].label}, {self.atoms[1].label}, {self.type_name})" \

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
        return _bond_type[self.type]

    @property
    def type(self):
        return self._OBBond.GetBondOrder()

