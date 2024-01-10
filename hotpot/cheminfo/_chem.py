"""
python v3.9.0
@Project: hotpot
@File   : molecule
@Auther : Zhiyuan Zhang
@Data   : 2023/10/14
@Time   : 16:25
"""
import os
import weakref
import logging
from os import PathLike
from abc import ABC
from typing import *
from pathlib import Path
from itertools import combinations, product

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from openbabel import openbabel as ob, pybel as pb

from hotpot.plugins import lmp

from ._base import Wrapper
from ._io import Parser, Dumper
from ._cryst import Crystal
from ._thermo import Thermo


_molecule_dict = weakref.WeakValueDictionary()

_stable_charges = {
    "H": 1,  "He": 0,
    "Li": 1, "Be": 2, "B": 3,  "C": 4,  "N": -3,  "O": -2,  "F": -1,  "Ne": 0,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": -3,  "S": -2,  "Cl": -1, "Ar": 0,
    "K": 1,  "Ca": 2, "Ga": 3, "Ge": 4, "As": -3, "Se": -2, "Br": -1, "Kr": 0,
    "Rb": 1, "Sr": 2, "In": 3, "Sn": 2, "Sb": -3, "Te": -2, "I": -1,  "Xe": 0,
    "Cs": 1, "Ba": 2, "Tl": 3, "Pb": 2, "Bi": 3,  "Po": -2, "At": -1, "Rn": 0,
    "Fr": 1, "Ra": 2, "Nh": 8, "Fl": 8, "Mc": 8,  "Lv": 8,  "Ts": 8,  "Og": 8,

    "Sc": 3, "Ti": 4, "V": 5,  "Cr": 3, "Mn": 2,  "Fe": 3,  "Co": 3,  "Ni": 2, "Cu": 2, "Zn": 2,
    "Y": 3,  "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7,  "Ru": 4,  "Rh": 3,  "Pd": 2, "Ag": 1, "Cd": 2,
    "Lu": 3, "Hf": 4, "Ta": 5, "W": 6,  "Re": 7,  "Os": 4,  "Ir": 3,  "Pt": 2, "Au": 1, "Hg": 2,
    "Lr": 3, "Rf": 4, "Db": 5, "Sg": 6, "Bh": 7,  "Hs": 8,  "Mt": 8,  "Ds": 8, "Rg": 8, "Cn": 8,

    "La": 3, "Ce": 4, "Pr": 3, "Nd": 3, "Pm": 3,  "Sm": 3,  "Eu": 3,  "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3,
    "Ac": 3, "Th": 4, "Pa": 5, "U": 6,  "Np": 6,  "Pu": 6,  "Am": 3,  "Cm": 6, "Bk": 6, "Cf": 6, "Es": 6, "Fm": 6, "Md": 6, "No": 6,
}


def _refcode_getter(ob_mol: ob.OBMol) -> Union[int, None]:
    """ retrieve refcode from given OBMol object """
    return ob_mol and (obdata := ob.toCommentData(ob_mol.GetData("refcode"))) and int(obdata.GetData())


class Molecule(Wrapper, ABC):
    """Represent an intuitive molecule"""
    def __new__(cls, ob_mol=None):
        return _molecule_dict.get(_refcode_getter(ob_mol), super(Molecule, cls).__new__(cls))

    def __init__(self, ob_mol=None):
        super().__init__(ob_mol or ob.OBMol())
        self._set_refcode()

    def __repr__(self):
        return f"Mol({self.ob_mol.GetFormula()})"

    def __eq__(self, other):
        """ if two molecule with 1.0 similarity in 2FP fingerprint they are identical """
        if self.similarity(other) == 1.0:
            return True
        return False

    def __hash__(self):
        return hash(f"hotpot.Molecule(refcode={self.refcode})")

    def __add__(self, other: 'Molecule'):
        mol, other = self.copy(), other.copy()
        mol.remove_hydrogens()
        other.remove_hydrogens()

        mol.add_component(other)
        mol.add_hydrogens()
        return mol

    def _set_refcode(self):
        """ put an int value into the OBMol as the refcode """
        if self.refcode is None:
            self._set_ob_int_data('refcode', 0 if not _molecule_dict else max(_molecule_dict.keys()) + 1)
            _molecule_dict[self.refcode] = self

    def add_atom(self, atom: Union["Atom", str, int]) -> 'Atom':
        """
        Add a new atom out of the molecule into the molecule.
        Args:
            atom(Atom|str|int):

        atom_kwargs(kwargs for this added atom):
            atomic_number(int): set atomic number
            symbol(str): set atomic symbol
            coordinates(Sequence, numpy.ndarray): coordinates of the atom
            partial_charge:
            label:
            spin_density:

        Returns:
            the copy of atom in the molecule
        """
        oba = ob.OBAtom()  # Initialize a new OBAtom
        if isinstance(atom, Atom):
            oba.Duplicate(atom.ob_atom)
        elif isinstance(atom, str):
            oba.SetAtomicNum(ob.GetAtomicNum(atom))
        elif isinstance(atom, int):
            oba.SetAtomicNum(atom)
        else:
            raise TypeError('the argument `atom` should be str, int or Atom')

        success = self.ob_mol.AddAtom(oba)  # add OBAtom to the OBMol
        if success:
            return self.atoms[-1]
        raise RuntimeError('fail to add atom!')

    def add_bond(self, atom1: 'Atom', atom2: 'Atom', bond_type: int):
        """ Add a new bond into the molecule """
        # Try to add new OBMol
        # 'openbabel' has an odd behave that `index` of the `OBAtom` with various origin in the `OBMol`.
        # the `Id` of `OBAtom` from 0; but the `Idx` of `OBAtom` from 1.
        # To meet the convention, the `Id` is selected to be the unique `index` to specify `Atom`.
        # However, when try to add a `OBBond` to link each two `OBAtoms`, the `Idx` is the only method
        # to specify the atoms, so our `index` in `Atom` are added 1 to match the 'Idx'
        assert atom1.molecule is self
        assert atom2.molecule is self
        success = self.ob_mol.AddBond(atom1.idx, atom2.idx, bond_type)

        if success:
            return self.bonds[-1]  # the new atoms should place in the terminal of the bond list

        else:
            raise RuntimeError('add bond not successful!')

    def add_component(self, component: "Molecule") -> (dict[int, int], dict[int, int]):
        """
        add a Molecule object to be a new component into the zone of this molecule
        Args:
            component: the Molecule object added into this Molecule as a new component

        Returns:
            dict: {AtomID_old: AtomID_new}, dict: {BondID_old: BondID_new}.
            Where:
                1) AtomID_old: the ob_id of added atoms in the original Molecule
                2) AtomID_new: the ob_id of added atoms on the added component of this Molecule
                 after the adding operation
                3) BondID_old: the ob_id of added bonds in the original Molecule
                4) BondID_old: the ob_id of added bonds in the added component of this Molecule
                 after the adding operation
        """
        atoms_mapping, bonds_mapping = {}, {}
        for atom in component.atoms:
            added_atom = self.add_atom(atom)
            atoms_mapping[atom.idx] = added_atom.idx

        for bond in component.bonds:
            old_oba1_id, old_oba2_id = bond.atom1.idx, bond.atom2.idx
            new_oba1_id, new_oba2_id = atoms_mapping[old_oba1_id], atoms_mapping[old_oba2_id]
            added_bond = self.add_bond(self.atom(new_oba1_id), self.atom(new_oba2_id), bond.type)

            bonds_mapping[bond.idx] = added_bond.idx

        return atoms_mapping, bonds_mapping

    def add_hydrogens(
            self,
            polar_only: bool = False,
            correct_for_ph: bool = False,
            ph: float = 1.0,
            balance_hydrogen: bool = True,
    ):
        """
        add hydrogens for the molecule
        Args:
            ph: add hydrogen in which PH environment
            polar_only: Whether to add hydrogens only to polar atoms (i.e., not to C atoms)
            correct_for_ph: Correct for pH by applying the OpenBabel::OBPhModel transformations
            balance_hydrogen: whether to balance the bond valance of heavy atom to their valence
        """
        is_aromatic = {atom.idx: atom.is_aromatic for atom in self.atoms}
        self.ob_mol.AddHydrogens(polar_only, correct_for_ph, ph)

        if balance_hydrogen:
            for atom in self.atoms:
                atom.balance_hydrogen()

        # for atom in self.atoms:
        #     if is_aromatic.get(atom.idx):
        #         atom.set_aromatic()

    @property
    def angles(self):
        angles = []
        for vertex in self.atoms:
            angles.extend([Angle(vertex, *non_vertex) for non_vertex in combinations(vertex.neighbours, 2)])

        return angles
    
    def assign_atoms_formal_charge(self):
        """ Assign the formal charges for all atoms in the molecule """
        self.add_hydrogens(balance_hydrogen=False)

        for atom in self.atoms:
            if atom.is_polar_hydrogen:
                atom.formal_charge = 1
            elif atom.is_hydrogen or atom.is_carbon:
                atom.formal_charge = 0
            elif atom.is_metal:
                atom.formal_charge = _stable_charges[atom.symbol]
            elif [a for a in atom.neighbours if a.is_polar_hydrogen]:
                atom.formal_charge = -(len([a for a in atom.neighbours if a.is_polar_hydrogen]))
            elif atom.symbol in ['S', 'P']:
                if not [a for a in atom.neighbours if a.symbol == 'O']:
                    atom.formal_charge = atom.covalent_valence - (2 if atom.symbol == 'S' else 3)
                else:
                    atom.formal_charge = 0
            else:
                atom.formal_charge = atom.covalent_valence - atom.stable_valence

    def assign_bond_types(self):
        self.ob_mol.PerceiveBondOrders()

    def atom(self, idx_label_atom: Union[int, str, "Atom"]) -> 'Atom':
        """ get atom by label or idx """
        if isinstance(idx_label_atom, Atom):
            if idx_label_atom.molecule is self:
                return idx_label_atom
            else:
                raise AttributeError('the given atom is not in this molecule')

        elif isinstance(idx_label_atom, str):

            if not self.is_labels_unique:
                raise AttributeError(
                    'the label is not unique, cannot get atom by label. try to get atom by ob_id '
                    'or normalize the label before'
                )

            for atom in self.atoms:
                if atom.label == idx_label_atom:
                    return atom
            raise KeyError(f'No atom with label {idx_label_atom}')

        elif isinstance(idx_label_atom, int):
            return Atom(self.ob_mol.GetAtom(idx_label_atom))
        else:
            raise TypeError(f'the given idx_label is expected to be int or string, but given {type(idx_label_atom)}')

    @property
    def atoms(self) -> list["Atom"]:
        return [Atom(oba) for oba in ob.OBMolAtomIter(self.ob_mol)]

    @property
    def atoms_dist_matrix(self) -> np.ndarray:
        """ The distance matrix between each of atoms pairs """
        return cdist(self.coordinates, self.coordinates)

    def balance_hydrogens(self):
        """ Add or remove hydrogens for make or heave atom to achieve the stable valence """
        for a in self.heavy_atoms:
            a.balance_hydrogen()

    def bond(self, atom1: Union[int, str, "Atom"], atom2: Union[int, str, "Atom"]) -> 'Bond':
        """
        Return the Bond by given atom index labels in the bond ends
        if the bond is missing in the molecule, return None if given miss_raise is False else raise a KeyError
        Args:
            atom1(int|str): index or label of atom in one of the bond end
            atom2(int|str): index or label of atom in the other end of the bond

        Returns:
            Bond or None
        """
        return (obb := self.ob_mol.GetBond(self.atom(atom1).idx, self.atom(atom2).idx)) and Bond(obb)

    @property
    def bonds(self) -> list["Bond"]:
        return [Bond(obb) for obb in ob.OBMolBondIter(self.ob_mol)]

    def build_2d(self):
        """ build 2d conformer """
        pb.Molecule(self.ob_mol).make2D()

    def build_3d(self, force_field: str = 'UFF', steps: int = 500, balance_hydrogen=False):
        """ build 3D coordinates for the molecule """
        pb.Molecule(self.ob_mol).make3D(force_field, steps)
        if balance_hydrogen:
            self.balance_hydrogens()

    def build_rd3d(self):
        """ build 3d by rdkit method """
        self.add_hydrogens()
        self.normalize_labels()
        self.build_2d()

        rdmol = Chem.MolFromMolBlock(self.dump('mol'))
        rdmol = Chem.AddHs(rdmol)
        assert len(self.atoms) == len(rdmol.GetAtoms())
        for a, ra in zip(self.atoms, rdmol.GetAtoms()):
            assert a.symbol == ra.GetSymbol()
            ra.SetProp('label', a.label)

        AllChem.EmbedMolecule(rdmol)
        AllChem.UFFOptimizeMolecule(rdmol)

        for ra, rc in zip(rdmol.GetAtoms(), rdmol.GetConformer().GetPositions()):
            self.atom(ra.GetProp('label')).coordinate = rc

        self.remove_hydrogens()
        self.add_hydrogens()

    def build_bonds(self):
        self.ob_mol.ConnectTheDots()

    @property
    def capacity(self) -> float:
        return self._get_ob_float_data('capacity')

    @capacity.setter
    def capacity(self, value: float):
        self._set_ob_float_data('capacity', value)

    @property
    def centroid_geom(self):
        return self.coordinates.mean(axis=0)

    @property
    def centroid_mass(self):
        masses = np.array([a.mass for a in self.atoms])
        return (masses * self.coordinates.T).T.sum(axis=0) / masses.sum()

    @property
    def charge(self):
        return self.ob_mol.GetTotalCharge()

    @charge.setter
    def charge(self, charge):
        self.ob_mol.SetTotalCharge(charge)

    @property
    def components(self):
        """ get all fragments don't link each by any bonds """
        return [self.__class__(obc) for obc in self.ob_mol.Separate()]

    @property
    def coordinates(self) -> np.ndarray:
        """
        Get the matrix of all atoms coordinates,
        where the row index point to the atom index;
        the column index point to the (x, y, z)
        """
        return np.array([atom.coordinate for atom in self.atoms], dtype=np.float64)

    def copy(self) -> "Molecule":
        """ get a clone for this Molecule """
        clone: ob.OBMol = ob.OBMol(self.ob_mol)
        clone.DeleteData('refcode')
        return self.__class__(clone)

    def compact_crystal(self, inplace=False):
        """
        Create a compact crystal for this molecule
        Args:
            inplace: Whether inplace the crystal of the Molecule

        Returns:
            compact Crystal object
        """
        mol = self if inplace else self.copy()

        xyz_diff = self.coordinates.max(axis=0) - self.coordinates.min(axis=0)
        mol.translate(*(xyz_diff/2 - mol.centroid_geom))

        lattice_params = np.concatenate((xyz_diff, [90., 90., 90.]))
        crystal = mol.make_crystal(*lattice_params)
        crystal.ob_unit_cell.SetSpaceGroup('P1')

        return crystal

    @classmethod
    def create_aCryst_by_mq(
            cls, elements: Dict[str, float], force_field: Union[str, os.PathLike],
            density: float = 1.0, a: float = 25., b: float = 25., c: float = 25.,
            alpha: float = 90., beta: float = 90., gamma: float = 90., time_step: float = 0.0001,
            origin_temp: float = 298.15, melt_temp: float = 4000., highest_temp: float = 10000.,
            ff_args: Sequence = (), path_writefile: Optional[str] = None, path_dump_to: Optional[str] = None,
            dump_every: int = 100
    ) -> "Molecule":
        """
        Create a Amorphous crystal materials by Melt-Quench process.
        This process is performed by LAMMPS package, make sure the LAMMPS is accessible.
        A suitable force field is required for the process are performed correctly.
        Args:
            elements(dict[str, float]): Dict of elements and their composition ratio
            force_field(str, os.PathLike): The name of force filed or the path to load a force filed. The name
             of the force filed is refer to the relative path to the 'hotpot_root/data/force_field'.
            density: the demand density for the created amorphous crystal
            a: the length of a vector in the crystal
            b: the length of b vector in the crystal
            c: the length of c vector in the crystal
            alpha: alpha angle of crystal param.
            beta: beta angle of crystal param.
            gamma: gamma angle of crystal param
            time_step: time interval between path integrals when performing melt-quench
            origin_temp: the initial temperature before melt
            melt_temp: the round melting point to the materials
            highest_temp: the highest temperature to liquefy the materials
            ff_args: the arguments the force file requried, refering the LAMMPS pair_coeff:
             "pair_coeff I J args" url: https://docs.lammps.org/pair_coeff.html
            path_writefile: the path to write the final material (screenshot) to file, if not specify, not save.
            path_dump_to:  the path to save the trajectory of the melt-quench process, if not specify not save.
            dump_every: the step interval between each dump operations

        Returns:
            Molecule, a created amorphous material
        """
        return lmp.AmorphousMaker(elements, force_field, density, a, b, c, alpha, beta, gamma).melt_quench(
            *ff_args, path_writefile=path_writefile, path_dump_to=path_dump_to, origin_temp=origin_temp,
            melt_temp=melt_temp, highest_temp=highest_temp, time_step=time_step, dump_every=dump_every)

    @property
    def crystal(self) -> "Crystal":
        """ Retrieve the Crystal that this Molecule is built in """
        return (cell_data := self.ob_mol.GetData(ob.UnitCell)) and Crystal(ob.toUnitCell(cell_data), self)

    @property
    def disorder_bonds(self) -> list["Bond"]:
        """ Get all disorder bonds in the Molecule """
        return [b for b in self.bonds if not (0.85 < b.length/b.ideal_length < 1.15)]

    @overload
    def dump(self, fmt: str):
        """ general input arguments """

    @overload
    def dump(self, fmt: Literal['gjf'], *, link0: Union[list[str], str], route: Union[list[str], str], addition):
        """ args to dump Gaussian16 gjf file """

    def dump(self, fmt: str, *args, **kwargs) -> Union[str, bytes, dict]:
        return Dumper.get_io(fmt)(self, *args, **kwargs)

    @property
    def energy(self):
        """ Return energy with kcal/mol as default """
        return self.ob_mol.GetEnergy()

    @energy.setter
    def energy(self, value: float):
        self.ob_mol.SetEnergy(value)

    @property
    def enthalpy(self):
        return self._get_ob_float_data('enthalpy')

    @enthalpy.setter
    def enthalpy(self, value: float):
        self._set_ob_float_data('enthalpy', value)

    @property
    def entropy(self):
        return self._get_ob_float_data('entropy')

    @entropy.setter
    def entropy(self, value: float):
        self._set_ob_float_data('entropy', value)

    def fingerprint(self, fptype: Literal['FP2', 'FP3', 'FP4', 'MACCS'] = 'FP2'):
        """
        Calculate the molecular fingerprint for this molecule, the supporting fingerprint include:

        1. "FP2": The FP2 fingerprint is a path-based fingerprint that encodes the presence of linear
        fragments up to 7 atoms long. It is a 1024-bit fingerprint and is commonly used for substructure
        searches and similarity calculations.

        2. "FP3": The FP3 fingerprint is designed for searching 3D conformations, such as those found
        in protein-ligand complexes. It encodes the presence of particular pharmacophoric features,
        such as hydrogen bond donors, acceptors, and hydrophobic regions.

        3. "FP4": The FP4 fingerprint is a circular fingerprint based on the Morgan algorithm. It
        captures information about the local environment of each atom in the molecule, up to a certain
        radius. It is useful for similarity calculations and machine learning plugins.

        4. "MACCS": The MACCS fingerprint is a 166-bit structural key-based fingerprint. It represents
        the presence of specific substructures or functional groups defined by the MACCS keys. It is
        commonly used for similarity calculations and substructure searches.

        Return:
            the Fingerprint object in pybel module
        """
        return pb.Molecule(self.ob_mol).calcfp(fptype)

    @property
    def forces(self) -> np.ndarray:
        """ Get the forces matrix for all atoms in the Molecule """
        return np.stack([a.force for a in self.atoms])

    @property
    def formula(self) -> str:
        return self.ob_mol.GetFormula()

    @property
    def frac_coordinates(self) -> np.ndarray:
        """ get the fractional coordinate relate to the crystal of the Molecule place """
        return self.crystal and np.dot(np.linalg.inv(self.crystal.matrix), self.coordinates.T).T

    @property
    def free_energy(self):
        return self._get_ob_float_data('free_energy')

    @free_energy.setter
    def free_energy(self, value: float):
        self._set_ob_float_data('free_energy', value)

    def gaussian(
            self,
            link0: Union[str, List[str]],
            route: Union[str, List[str]],
            path_log_file: Union[str, PathLike] = None,
            path_err_file: Union[str, PathLike] = None,
            path_chk_file: Union[str, PathLike] = None,
            path_rwf_file: Union[str, PathLike] = None,
            inplace_attrs: bool = False,
            output_in_running: bool = True,
            g16root: Union[str, PathLike] = None,
            *args, **kwargs
    ) -> (Union[None, str], str):
        """
        calculation by Gaussian.
        for running the method normally, MAKE SURE THE Gaussian16 HAVE BEEN INSTALLED AND ALL ENV VAR SET RITHT !!
        Args:
            g16root: the dir Gaussian16 software installed
            link0: the link0 command in gjf script
            route: the route command in gjf script
            path_log_file: Optional, the path to save the out.log file. If not given, the logfile would be written
             to the work dir
            path_err_file: Optional, the path to save the error log file. If not given, the err file would be written
             to the work dir
            path_chk_file: Optional, the path to the checkpoint file. If not given the chk file would be written
             to the work dir
            path_rwf_file: Optional, the path to the read-write file. If not given the rwf file would be written
             to the work dir
            inplace_attrs: Whether to inplace self attribute according to the results from attributes.
            debugger: define the method to handle the Gaussian error, like l9999, l103 or l502 ...,
             the default method is the 'auto', which just to handle some common error case. More elaborate
             debugger could be handmade by inherit from `Debugger` abstract class. For detail, seeing
             the documentation.
            output_in_running: Whether write the output file when the Gaussian process is running
            *args:
            **kwargs:

        Returns:
            the standard output of g16 log file(string), the standard output of g16 err file(string)
        """
        # TODO: refactoring

    def gcmc(
            self, *guest: 'Molecule', force_field: Union[str, os.PathLike] = None,
            work_dir: Union[str, os.PathLike] = None, T: float = 298.15, P: float = 1.0, **kwargs
    ):
        """
        Run gcmc to determine the adsorption of guest,
        Args:
            self: the framework as the sorbent of guest molecule
            guest(Molecule): the guest molecule to be adsorbed into the framework
            force_field(str|PathLike): the path to force field file or the self-existent force file contained
             in force field directory (in the case, a str should be given as a relative path from the root of
             force field root to the specified self-existent force filed). By default, the force field is UFF
             which in the relative path 'UFF/LJ.json' for the force field path.
            work_dir: the user-specified dir to store the result of GCMC and log file.
            T: the environmental temperature (default, 298.15 K)
            P: the relative pressure related to the saturation vapor in the environmental temperature.
        """
        # TODO: refactoring

    @overload
    def get_thermo(self, *, T: float, P: float):
        """
        If certain substance don't retrieve information from current database, some required thermodynamical
        parameters should pass into equation_of_state to initialization
        Keyword Args:
            T: the ambient temperature for thermodynamical system
            P: the ambient pressure for thermodynamical system
        Returns:
            Thermo class
        """

    @overload
    def get_thermo(self, *, P: float, V: float):
        """
        If certain substance don't retrieve information from current database, some required thermodynamical
        parameters should pass into equation_of_state to initialization
        Keyword Args:
            P: the ambient pressure for thermodynamical system
            V: the volume of thermodynamical system
        Returns:
            Thermo class
        """

    @overload
    def get_thermo(self, *, T: float, V: float):
        """
        If certain substance don't retrieve information from current database, some required thermodynamical
        parameters should pass into equation_of_state to initialization
        Keyword Args:
            T: the ambient temperature for thermodynamical system
            V: the volume of thermodynamical system
        Returns:
            Thermo class
        """

    @overload
    def get_thermo(self, *, Tc: float, Pc: float, omega: float):
        """
        If certain substance don't retrieve information from current database, some required thermodynamical
        parameters should pass into equation_of_state to initialization
        Keyword Args:
            Tc: the critical temperature of the molecule
            Pc: the critical pressure of the molecule
            omega: acentric factor of the molecule
        Returns:
            Thermo class
        """

    def get_thermo(self, **kwargs) -> Thermo:
        """
        If certain substance don't retrieve information from current database, some required thermodynamical
        parameters should pass into equation_of_state to initialization
        Keyword Args:
            T: the ambient temperature for thermodynamical system
            P: the ambient pressure for thermodynamical system
            V: the volume of thermodynamical system
            Tc: the critical temperature of the molecule
            Pc: the critical pressure of the molecule
            omega: acentric factor of the molecule
        Returns:
            Thermo class
        """
        return Thermo(self, **kwargs)

    @property
    def heavy_atoms(self):
        """ Get the atoms except for hydrogens """
        return [a for a in self.atoms if a.is_heavy]

    @property
    def has_3d(self):
        """ Whether atoms in the molecule have 3d coordinates """
        return self.ob_mol.Has3D()

    @property
    def has_hydrogen_added(self):
        """ Have hydrogens been added to the molecule by call Molecule.add_hydrogen()? """
        return self.ob_mol.HasHydrogensAdded()

    @property
    def has_nan_coordinates(self) -> bool:
        """ Check whether any nan coordinates exists """
        return np.any(np.isnan(self.coordinates))

    @property
    def identifier(self):
        return self.ob_mol.GetTitle()

    @identifier.setter
    def identifier(self, value):
        self.ob_mol.SetTitle(value)

    @property
    def inchi(self) -> str:
        return self.dump('inchi').strip()

    @property
    def is_labels_unique(self):
        """ Determine whether all atom labels are unique """
        return len({a.label for a in self.atoms}) == len(self.atoms)

    @property
    def is_organic(self):
        """ Whether the Molecule is an organic compounds """
        return all(not a.is_metal for a in self.atoms) and any(a.is_carbon for a in self.atoms)

    @property
    def ob_mol(self) -> ob.OBMol:
        return self._obj

    @property
    def link_matrix(self):
        """
        Returns: numpy.Array with a shape of [2, 2*Nb], where the Nb is the number of bonds.

        the indices in the first raw is the id of source atoms, the indices in the second raw is
        the id of target atoms. Because of the molecules are always regard as an undirected graph,
        representing the link relation of a bond needs two link edges, the number of columns is thus
        2 * Nb.

        the columns of link matrix, or edges, are arranged as the order of bonds. the i-th bonds
        are referred by i-th and 2i-th edges.
        """
        # TODO: Refactoring by networkx

    @property
    def lssr(self) -> list["Ring"]:
        return [Ring(obr) for obr in self.ob_mol.GetLSSR()]

    def localed_optimize(
            self,
            force_field: str = 'UFF',
            steps: int = 100,
            balance_hydrogen: bool = False,
            to_optimal: bool = False,
            tolerable_displacement: float = 1e-1,
            max_iter: int = 10
    ):
        """
        Locally optimize the coordinates. referring openbabel.pybel package
        Args:
            force_field: all accessible force field in openbabel.pybel
            steps: the optimization steps in each iteration
            balance_hydrogen: if true, to set the number of hydrogen of heave atoms to its imply number.
            to_optimal: perform a loop optimization, until the equilibrium achieves or the max iteration
             is exceeded.
            tolerable_displacement: the lower displacement is regard as equilibrium status
            max_iter: if the number of iteration exceeds this value, break out the optimization loop.

        Returns:

        """
        displacement = 0
        equilibrium_counts = 0
        last_atom_coord = 0
        num_iter = 0

        while (not isinstance(last_atom_coord, np.ndarray)) or (
                to_optimal and equilibrium_counts < 3 and num_iter < max_iter
        ):
            # reload hydrogens
            self.remove_hydrogens()
            self.add_hydrogens(balance_hydrogen=balance_hydrogen)

            pymol = pb.Molecule(self.ob_mol)
            pymol.localopt(force_field, steps)

            self.remove_hydrogens()
            displacement = np.max(self.coordinates - last_atom_coord)

            if displacement < tolerable_displacement:
                equilibrium_counts += 1
            else:
                equilibrium_counts = 0

            last_atom_coord = self.coordinates
            num_iter += 1

            logging.info(f"displacement: {displacement}")

        if displacement > tolerable_displacement:
            logging.warning(f"the optimal structure does not achieve !!!")

        self.add_hydrogens()

    @overload
    def make_crystal(self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> 'Crystal':
        """"""
    @overload
    def make_crystal(self, matrix: np.ndarray) -> "Crystal":
        """"""

    def make_crystal(self, *args) -> 'Crystal':
        """ Put this molecule into the specified crystal """
        self.ob_mol.CloneData(ob.OBUnitCell())

        if len(args) == 6 and all(isinstance(arg, float) for arg in args):
            self.crystal.ob_unit_cell.SetData(*args)
        elif len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].shape == (3, 3):
            self.crystal.ob_unit_cell.SetData(*Crystal.matrix_to_params(args[0]))
        else:
            raise ValueError('the given args is wrong')

        return self.crystal

    @property
    def metals(self) -> List['Atom']:
        return [a for a in self.atoms if a.is_metal]

    @property
    def orbital_energies(self) -> np.ndarray:
        """ the orbital energies calculated from quantum mechanics tools """
        return np.ndarray(self._get_ob_list_data('orbital_energies'))

    @orbital_energies.setter
    def orbital_energies(self, value: Union[Sequence, np.ndarray]):
        self._set_ob_list_data('orbital_energies', list(value))

    @property
    def nx_graph(self) -> nx.Graph:
        """ Return networkx Graph of the Molecule """
        graph = nx.Graph()
        graph.add_edges_from([(b.atom1.idx, b.atom2.idx) for b in self.bonds])
        return graph

    def normalize_labels(self):
        """ Reorder the atoms labels in the molecule """
        element_counts = {}
        for atom in self.atoms:
            element_counts[atom.symbol] = count = element_counts.get(atom.symbol, 0) + 1
            atom.label = f'{atom.symbol}{count}'

    @classmethod
    def read_from(cls, src: Union[str, PathLike], fmt=None, *args, **kwargs) -> 'Molecule':
        if not fmt and isinstance(src, str):
            src = Path(src)
            fmt = src.suffix.strip('.')

        return Parser.get_io(fmt)(src, *args, **kwargs)

    @property
    def refcode(self) -> int:
        """ get the refcode of this molecule in the molecular WeakValueDictionary """
        return self._get_ob_int_data('refcode')

    def remove_atoms(self, *atoms: 'Atom') -> None:
        """
        Remove atom according to given atom index, label or the atoms self.
        Args:
            atoms(int|str|Atom): the index, label or self of Removed atom
        """
        for atom in atoms:
            # remove connecting hydrogens
            # for nh in atom.neighbours_hydrogen:
            #     if len(nh.neighbours) == 1:
            #         self.ob_mol.DeleteAtom(nh.ob_atom, False)

            # Removing the atom
            self.ob_mol.DeleteAtom(atom.ob_atom, False)

    def remove_bonds(self, *bonds: 'Bond'):
        """ Remove the bonds in the molecule """
        for bond in bonds:
            successful = self.ob_mol.DeleteBond(bond.ob_bond, False)
            if not successful:
                raise RuntimeError(f'Fail to remove {bonds}')

    def remove_hydrogens(self):
        is_aromatic = {atom.idx: atom.is_aromatic for atom in self.atoms}
        self.ob_mol.DeleteHydrogens()

        # for atom in self.atoms:
        #     if is_aromatic.get(atom.idx):
        #         atom.set_aromatic()

    def reorder_ob_ids(self):
        """ Reorder the ob_ids of all atoms and all bonds in the Molecule """
        for i, oba in enumerate(ob.OBMolAtomIter(self.ob_mol)):
            oba.SetId(i)
        for i, obb in enumerate(ob.OBMolBondIter(self.ob_mol)):
            obb.SetId(i)

    @property
    def rotatable_bonds_number(self):
        return self.ob_mol.NumRotors()

    def save_2d_img(self, file_path: Union[str, os.PathLike], **kwargs):
        """
        Export 2d image to file
        Args:
            file_path:
            **kwargs: other keywords arguments for 2d image make and save

        Keyword Args:

        """
        self.to_2d_img(**kwargs).save(file_path)

    def shortest_paths(self, src_atom: 'Atom', des_atom: 'Atom') -> list[list['Atom']]:
        """
        retrieve the shortest path from the source atom to the destination atom
        Args:
            src_atom: source atom, or its index or label
            des_atom: destination atom, or its index or label

        Returns:
            list of list of atoms
        """
        atoms_dict = {a.idx: a for a in self.atoms}
        paths = nx.all_shortest_paths(self.nx_graph, src_atom.idx, des_atom.idx)
        return [[atoms_dict[i] for i in path] for path in paths]

    def similarity(self, other: 'Molecule', fptype: Literal['FP2', 'FP3', 'FP4', 'MACCS'] = 'MACCS') -> int:
        """
        Compare the similarity with other molecule, based on specified fingerprint
        Args:
            other(Molecule): the other Molecule
            fptype(str): the fingerprint type to perform comparison of similarity

        Return:
            the similarity(int)
        """
        return self.fingerprint(fptype) | other.fingerprint(fptype)

    @property
    def smiles(self):
        """ Get the canonical smiles """
        return self.dump('can').split()[0]

    @property
    def spin(self):
        return self.ob_mol.GetTotalSpinMultiplicity()

    @property
    def thermal_energy(self) -> float:
        return self._get_ob_float_data('thermal_energy')

    @thermal_energy.setter
    def thermal_energy(self, value: float):
        self._set_ob_float_data('thermal_energy', value)

    @property
    def torsions(self) -> list["Torsion"]:
        torsions = []
        for axis_bond in self.bonds:
            atom1_neigh = [a for a in axis_bond.atom1.neighbours if a not in axis_bond]
            atom2_neigh = [a for a in axis_bond.atom2.neighbours if a not in axis_bond]
            for a, d in product(atom1_neigh, atom2_neigh):
                torsions.append(Torsion(a, axis_bond.atom1, axis_bond.atom2, d))

        return torsions

    def to_2d_img(self, **kwargs):
        """
        Get a 2D image objects for the molecule
        Keyword Args:
            kekulize: whether to applying Kekulize style to aromatical rings

        Returns:

        """
        clone = self.copy()
        clone.build_2d()
        return Draw.MolToImage(clone.to_rdmol(), **kwargs)

    def to_rdmol(self):
        """ convert hotpot Molecule object to RdKit mol object """
        return Chem.MolFromMol2Block(self.dump('mol2'), sanitize=False)

    def translate(self, x: float, y: float, z: float):
        """ translate the coordinates of the atoms according to given translating vector (x, y, z) """
        translated_coordinates = self.coordinates + np.array([x, y, z])
        for atom, c in zip(self.atoms, translated_coordinates):
            atom.coordinate = c

    @property
    def weight(self):
        return self.ob_mol.GetExactMass()

    def writefile(self, fmt: str, path_file, *args, **kwargs):
        """Write the Molecule Info into a file with specific format(fmt)"""
        with open(path_file, 'w') as writer:
            writer.write(self.dump(fmt=fmt, *args, **kwargs))

    @property
    def zero_point(self) -> float:
        return self._get_ob_float_data('zero_point')

    @zero_point.setter
    def zero_point(self, value):
        self._set_ob_float_data('zero_point', value)


#######################################################################################################################
#######################################################################################################################


class MolBuildUnit(Wrapper, ABC):
    """ defining the base methods for Molecule Units ,like Atom and Bonds """
    @property
    def molecule(self) -> "Molecule":
        """ the parent molecule """
        return _molecule_dict[_refcode_getter(self._obj.GetParent())]

    @property
    def idx(self):
        return self._obj.GetIdx()


class Atom(MolBuildUnit):
    """Represent an intuitive Atom"""
    def __init__(self, ob_atom: ob.OBAtom = None, *, symbol: str = None, atomic_number: int = None):
        if not ob_atom:
            ob_atom = ob.OBAtom()
            if symbol:
                ob_atom.SetAtomicNum(ob.GetAtomicNum(symbol))
            elif atomic_number:
                ob_atom.SetAtomicNum(atomic_number)

        super().__init__(ob_atom)

    def __repr__(self):
        return f"Atom({self.label})"

    def __eq__(self, other):
        if isinstance(other, Atom):
            return self.ob_atom == other.ob_atom

    def __hash__(self):
        return hash(f'Atom({self.atomic_number})')

    def add_atom(self, symbol: str, bond_type=1) -> "Atom":
        """ add atom to link with this atom """
        new_atom = self.molecule.add_atom(symbol)
        self.molecule.add_bond(self, new_atom, bond_type)
        return new_atom

    def add_hydrogen(self) -> "Atom":
        """ add hydrogen to the atom """
        return self.add_atom('H')
    
    @property
    def atomic_number(self) -> int:
        return self.ob_atom.GetAtomicNum()

    def balance_hydrogen(self):
        """ Remove or add hydrogens link with this atom, if the bond valence is not equal to the atomic valence """
        if self.is_heavy and not self.is_metal:  # Do not add or remove hydrogens to the metal, H or inert elements
            while self.bond_orders > self.stable_valence and self.neighbours_hydrogen:
                self.molecule.remove_atoms(self.neighbours_hydrogen[0])

            # add hydrogen, if the bond valence less than the atomic valence
            while self.bond_orders < self.stable_valence:
                self.add_hydrogen()

    @property
    def bond_orders(self) -> int:
        """ Return the sum of bond order linking with this atom """
        return sum(b.type if b.type else 0 if b.is_covalent else 1 for b in self.bonds)

    @property
    def bonds(self):
        """ Get all bonds link with the atoms """
        return [Bond(obb) for obb in ob.OBAtomBondIter(self.ob_atom)]

    @property
    def coordinate(self) -> (float, float, float):
        return self.ob_atom.GetX(), self.ob_atom.GetY(), self.ob_atom.GetZ()

    @coordinate.setter
    def coordinate(self, value: Union[Sequence, np.ndarray]):
        self.ob_atom.SetVector(*value)

    @property
    def covalent_valence(self):
        """ the number of covalent electrons for this atoms """
        return sum(b.type if b.is_covalent else 0 for b in self.bonds)

    @property
    def electronegativity(self):
        return ob.GetElectroNeg(self.atomic_number)

    @property
    def force(self) -> np.ndarray:
        """ Get the force vector of this atom """
        return np.array(self._get_ob_list_data('force'))

    @force.setter
    def force(self, value: Union[Sequence, np.ndarray]):
        assert (isinstance(value, Sequence) and len(value) == 3) or \
               (isinstance(value, np.ndarray) and value.shape == (3,))

        self._set_ob_list_data('force', list(value))

    @property
    def formal_charge(self) -> float:
        return self.ob_atom.GetFormalCharge()

    @formal_charge.setter
    def formal_charge(self, value: float):
        self.ob_atom.SetFormalCharge(value)

    @property
    def hybridization(self):
        """ The hybridization of this atom:
        1 for sp, 2 for sp2, 3 for sp3, 4 for sq. planar, 5 for trig. bipy, 6 for octahedral """
        return self.ob_atom.GetHyb()

    @property
    def generations(self) -> int:
        return self._get_ob_int_data("generations") or 0

    @generations.setter
    def generations(self, value: int):
        self._set_ob_int_data("generations", value)

    @staticmethod
    def given_atomic_number_is_metal(atomic_number: int) -> bool:
        """ Whether the atomic number is the metal elements """
        oba = ob.OBAtom()
        oba.SetAtomicNum(atomic_number)
        return oba.IsMetal()
    
    @property
    def is_aromatic(self):
        return self.ob_atom.IsAromatic()
    
    @property
    def is_carbon(self):
        return self.atomic_number == 6

    @property
    def is_chiral(self):
        return self.ob_atom.IsChiral()

    @property
    def is_heavy(self):
        """ Whether the atom is heavy atom """
        return not self.is_hydrogen

    @property
    def is_hydrogen(self):
        return self.ob_atom.GetAtomicNum() == 1

    @property
    def is_in_ring(self):
        """ Whether the atom is in rings """
        return self.ob_atom.IsInRing()

    @property
    def is_metal(self):
        return self.ob_atom.IsMetal()
    
    @property
    def is_polar_hydrogen(self) -> bool:
        """ Is this atom a hydrogen connected to a polar atom """
        return self.ob_atom.IsPolarHydrogen()

    @property
    def ob_atom(self) -> ob.OBAtom:
        return self._obj

    @property
    def label(self) -> str:
        return self._get_ob_comment_data('label') or self.symbol

    @label.setter
    def label(self, value: str):
        self._set_ob_comment_data('label', value)

    @property
    def link_degree(self) -> int:
        """ the degree of the atom in their parent molecule """
        return self.ob_atom.GetTotalDegree()

    @property
    def mass(self):
        return self.ob_atom.GetAtomicMass()

    @property
    def member_of_ring_count(self) -> int:
        """ How many rings this atom is on """
        return self.ob_atom.MemberOfRingCount()

    @property
    def neighbours(self) -> List['Atom']:
        """ Get all atoms bond with this atom in same molecule """
        return [self.__class__(oba) for oba in ob.OBAtomAtomIter(self.ob_atom)]

    @property
    def neighbours_hydrogen(self) -> List['Atom']:
        """ return all neigh hydrogen atoms """
        return [a for a in self.neighbours if a.is_hydrogen]

    @property
    def partial_charge(self):
        return self.ob_atom.GetPartialCharge()

    @partial_charge.setter
    def partial_charge(self, value: float):
        # This is necessary to take effect to the assignment.
        # the reason is unknown
        self.ob_atom.GetPartialCharge()
        self.ob_atom.SetPartialCharge(value)

    @property
    def rings(self) -> list['Ring']:
        """ the lssr rings this atom is on """
        return [ring for ring in self.molecule.lssr if self.idx in ring.atoms_ids]

    def set_aromatic(self):
        """ Set this atom to be aromatic """
        self.ob_atom.IsAromatic()
        self.ob_atom.SetAromatic()

    @property
    def spin_density(self):
        return self._get_ob_float_data('spin_density')

    @spin_density.setter
    def spin_density(self, value: float):
        self._set_ob_float_data('spin_density', value)

    @property
    def stable_charge(self) -> int:
        if self.symbol == 'S':
            if not [a for a in self.neighbours if a.symbol == 'O']:
                return 2
            elif self.covalent_valence <= 2:
                return 2
            elif self.covalent_valence <= 4:
                return 4
            else:
                return 6
        elif self.symbol == 'P':
            if not [a for a in self.neighbours if a.symbol == 'O']:
                return 3
            elif self.covalent_valence == 0:
                return 0
            elif self.covalent_valence == 1:
                return 1
            elif self.covalent_valence <= 3:
                return 3
            else:
                return 5
        else:
            return _stable_charges[self.symbol]

    @property
    def stable_valence(self) -> int:
        if self.is_metal:
            return 0
        elif self.symbol == 'S':
            if not [a for a in self.neighbours if a.symbol == 'O']:
                return 2
            elif self.covalent_valence <= 2:
                return 2
            elif self.covalent_valence <= 4:
                return 4
            else:
                return 6
        elif self.symbol == 'P':
            if not [a for a in self.neighbours if a.symbol == 'O']:
                return 3
            elif self.covalent_valence == 0:
                return 0
            elif self.covalent_valence == 1:
                return 1
            elif self.covalent_valence <= 3:
                return 3
            else:
                return 5
        else:
            return abs(_stable_charges[self.symbol])

    @property
    def symbol(self) -> str:
        return ob.GetSymbol(self.ob_atom.GetAtomicNum())

    @symbol.setter
    def symbol(self, value: str):
        self.ob_atom.SetAtomicNum(ob.GetAtomicNum(value))


class Bond(MolBuildUnit):
    """ Represent an intuitive Bond """
    _type_name = {
        0: 'Unknown',
        1: 'Single',
        2: 'Double',
        3: 'Triple',
        5: 'Aromatic'
    }

    def __repr__(self):
        return f"Bond({self.atom1.label}, {self.atom2.label}, {self.type_name})"

    def __contains__(self, atom: Atom):
        return atom == self.atom1 or atom == self.atom2

    @property
    def atom1(self) -> Atom:
        return Atom(self.ob_bond.GetBeginAtom())

    @property
    def atom2(self) -> Atom:
        return Atom(self.ob_bond.GetEndAtom())

    @property
    def ideal_length(self):
        return self.ob_bond.GetEquibLength()

    @property
    def is_aromatic(self):
        return self.ob_bond.IsAromatic()

    @property
    def is_covalent(self) -> bool:
        return not self.atom1.is_metal and not self.atom2.is_metal

    @property
    def is_rigid(self) -> bool:
        """ Is a rigid bond """
        return self._get_ob_bool_data("is_rigid")

    @is_rigid.setter
    def is_rigid(self, value: bool):
        self._set_ob_bool_data("is_rigid", value)

    @property
    def length(self):
        return self.ob_bond.GetLength()

    @property
    def ob_bond(self) -> ob.OBBond:
        return self._obj

    @property
    def type(self) -> int:
        return self.ob_bond.GetBondOrder()

    @type.setter
    def type(self, bond_order: int):
        self.ob_bond.SetBondOrder(bond_order)

    @property
    def type_name(self):
        return self._type_name[self.type]


class Angle:
    def __init__(self, vertex: Atom, atom1: Atom, atom2: Atom):
        assert vertex.molecule is atom1.molecule is atom2.molecule
        self.vertex = vertex
        self.atom1 = atom1
        self.atom2 = atom2

    def __repr__(self):
        return f"{self.__class__.__name__}({self.vertex.label}, " \
               f"{self.atom1.label}, {self.atom2.label}, {round(self.degree, 3)})"

    @property
    def degree(self) -> float:
        return self.molecule.ob_mol.GetAngle(self.atom1.ob_atom, self.vertex.ob_atom, self.atom2.ob_atom)

    @property
    def molecule(self) -> Molecule:
        return self.vertex.molecule


class Ring(MolBuildUnit, ABC):
    """ Representing a ring structure in a Molecule """
    def _joint_rings(self, expand: bool = True, aromatic: bool = False) -> set["Ring"]:
        """get rings joint with this ring"""
        rings = {self} if not aromatic or self.is_aromatic else {}

        while len(rings) != len(rings := {
            ar for ring in rings for atom in ring.atoms for ar in atom.rings if not aromatic or ar.is_aromatic}):
            if not expand:
                break

        return rings

    @property
    def atoms(self) -> list[Atom]:
        return [atom for atom in self.molecule.atoms if self.ob_ring.IsMember(atom.ob_atom)]

    @property
    def atoms_ids(self):
        return [atom.idx for atom in self.molecule.atoms if self.ob_ring.IsMember(atom.ob_atom)]

    @property
    def bonds(self) -> list[Bond]:
        return [bond for bond in self.molecule.bonds if self.ob_ring.IsMember(bond.ob_bond)]

    @property
    def centroid(self) -> np.ndarray:
        """ The geometric center """
        return self.coordinates.mean(axis=0)

    @property
    def coordinates(self) -> np.ndarray:
        """ the coordinates matrix for all atoms """
        return np.array([atom.coordinate for atom in self.atoms])

    def intersection_atoms(self, other: "Ring"):
        """ Get intersecting atom among this ring and other """
        return [self.molecule.atom(ob_id) for ob_id in set(self.atoms_ids) & set(other.atoms_ids)]

    @property
    def is_aromatic(self) -> bool:
        return self.ob_ring.IsAromatic()

    @property
    def joint_rings(self) -> list["Ring"]:
        """get rings joint with this ring"""
        return [ring for ring in self._joint_rings(False) if ring is not self]

    @property
    def ob_ring(self) -> ob.OBRing:
        return self._obj

    def neigh_atoms(self, atom: Atom) -> (Atom, Atom):
        if atom not in self.atoms:
            raise ValueError('the given atom not in the ring!')

        return [a for a in atom.neighbours if a in self.atoms]

    @property
    def normal_vector(self) -> np.ndarray:
        """
        normal vector of ring's plane, we calculate it by the averaging the normal vectors of
        all places determined by arbitrary three-atoms combinations on the ring.
        """
        vectors = []
        for c1, c2, c0 in combinations(self.coordinates, 3):
            v1 = c1 - c0
            v2 = c2 - c0

            normal_vector = np.cross(v1, v2)
            if np.any(normal_vector):
                normal_vector = normal_vector / np.linalg.norm(normal_vector)

                if not vectors or np.dot(normal_vector, vectors[0]) > 0:
                    vectors.append(normal_vector)
                else:
                    # If the included angle between current vector and
                    # the first normal vector is larger than 90 degree,
                    # the arrow (direction) of current vector should be
                    # reverse.
                    vectors.append(-normal_vector)

        mean_vector = np.mean(vectors, axis=0)

        return mean_vector / np.linalg.norm(mean_vector)

    @property
    def size(self):
        return self.ob_ring.Size()


class Torsion:
    def __init__(self, *atoms: Atom):
        assert len(atoms) == 4 and all(a.molecule is atoms[0].molecule for a in atoms)
        self.atoms = atoms

    def __repr__(self):
        return f"{self.__class__.__name__}{self.atoms}={self.degress}°"

    @property
    def degress(self) -> float:
        return self.molecule.ob_mol.GetTorsion(*[a.ob_atom for a in self.atoms])

    @property
    def molecule(self):
        return self.atoms[0].molecule


