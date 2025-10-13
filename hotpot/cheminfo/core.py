"""
python v3.9.0
@Project: hotpot
@File   : core__
@Auther : Zhiyuan Zhang
@Data   : 2024/12/5
@Time   : 18:30
"""
import logging
import re
import sys
import time
import json
import operator
import os.path as osp
from typing import Union, Literal, Iterable, Optional, Callable
from copy import copy
from collections import Counter
from itertools import combinations, product

import cython
import numpy as np
import networkx as nx
from openbabel import pybel as pb, openbabel as ob
from scipy.spatial.distance import pdist, squareform

from hotpot.utils import types, chem as hpchem
import hotpot.cheminfo.obconvert as obc
from .rdconvert import to_rdmol
from . import graph, forcefields as ff
from . import geometry, crystal as cryst
from .pubchem import smi_to_cid, smi_to_name, smi_to_cas

if sys.modules.get('hotpot.cheminfo._io', None) is None:
    from . import _io
else:
    _io = sys.modules['hotpot.cheminfo._io']

def _metal_valence(atom):
    return 0


# Exceptions
class NotInSameMolecule(Exception):
    """ The exception is toggled when two objects are expected in a same Molecule, but not """
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2
        self.message = f"The {obj1} and {obj2} are expected in a same Molecule, but not"
        super().__init__(self.message)
        
class ObjNotInMolecule(Exception):
    """ The exception is toggled when the given object not in a specific Molecule. """
    def __init__(self, obj, mol):
        self.obj = obj
        self.mol = mol
        self.message = f"The {obj} not in {mol} molecule"
        super().__init__(self.message)


class Molecule:
    """
    Represents a molecular structure and provides methods for manipulating 
    and analyzing geometry, atoms, bonds, and other properties of a molecule.

    This class is designed to represent a molecular structure, providing attributes 
    such as atoms, bonds, rings, torsions, conformers, and graph representations. 
    It supports operations like adding atoms and bonds, updating molecular graphs, 
    and calculating molecular properties like charge and paths. The molecule class 
    integrates with other internal structures such as Atom, Bond, and Conformers, 
    forming the basis for chemical computations and geometry manipulations.
    """
    def __init__(self):
        self._atoms = []
        self._bonds = []
        self._conformers = Conformers()
        self._conformers_index = 0

        self._atom_pairs = AtomPairs(mol=self)
        self._angles = []
        self._torsions = []
        self._rings = []
        self._graph = nx.Graph()
        self._obmol = None
        self._row2idx = None
        self._crystal = None

        self._hided_metal_bonds = []
        self._hided_covalent_bonds = []
        self.properties = {}  # To store any mol properties

        self.charge = 0

    def __dir__(self):
        return list(Conformers._attrs) + list(super(Molecule, self).__dir__())

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError as e:
            # Retrieve attributes from conformers
            if "_conformers" in self.__dict__ and item in Conformers._attrs:
                return self._conformers.index_attr(item, self._conformers_index)
            else:
                raise e

    def __repr__(self):
        return f"{self.__class__.__name__}({self.formula})"

    def __add__(self, other):
        clone = copy(self)
        clone.add_component(other)
        return clone

    def __copy__(self):
        return self.copy()

    @property
    def _hided_bonds(self):
        return self._hided_metal_bonds + self._hided_covalent_bonds

    def copy(self):
        clone = Molecule()
        for atom in self._atoms:
            # clone._create_atom_from_array(atom.attrs)
            clone._create_atom(**atom.attr_dict)
        for bond in self._bonds:
            clone._add_bond(bond.a1idx, bond.a2idx, **bond.attr_dict)

        clone._update_graph()

        return clone

    @property
    def crystal(self):
        return self._crystal

    def create_crystal(self, a, b, c, alpha, beta, gamma):
        self._crystal = cryst.Crystal(a, b, c, alpha, beta, gamma)
        return self._crystal

    def setattr(self, **attrs):
        """
        Assigns multiple attributes to an instance from provided keyword arguments.

        Each keyword argument represents an attribute name and its corresponding 
        value to be set on the instance's namespace.

        Parameters:
            **attrs: dict
                The keyword arguments where each key is the attribute name and the 
                associated value is the value to be assigned to that attribute.

        Returns:
            None
        """
        for name, value in attrs.items():
            setattr(self, name, value)

    @staticmethod
    def read_one(src, fmt=None, **kwargs):
        """ Just read the first Molecule instance in the src """
        return next(_io.MolReader(src, fmt, **kwargs))

    @property
    def hydrogens(self) -> list["Atom"]:
        """
        Returns a list of hydrogen atoms in the molecule.

        This property iterates through all atoms in the molecule's internal atom
        list and filters out atoms with an atomic number of 1. It provides
        a convenient way to access all hydrogen atoms in the molecule.

        @return list of Atom
        """
        return [a for a in self._atoms if a.atomic_number == 1]

    @property
    def sum_explicit_hydrogens(self) -> int:
        """
        Calculate the total number of explicit hydrogen atoms.

        Explicit hydrogens are those directly associated with this object rather 
        than implicit hydrogens inferred from its structure or valency.

        Returns
        -------
        int
            The count of explicit hydrogen atoms.
        """
        return len(self.hydrogens)

    @property
    def sum_implicit_hydrogens(self) -> int:
        return sum(a.implicit_hydrogens for a in self.heavy_atoms)

    @property
    def has_hydrogens(self) -> bool:
        """
        Indicates whether the object contains any hydrogens.
        @return: Returns True if there are hydrogens, False otherwise.
        """
        return len(self.hydrogens) > 0

    def calc_mol_default_charge(self):
        """
        Calculates the default molecular charge of a molecule.

        The method determines the molecular charge by calculating the explicit
        and implicit hydrogens for organic molecules and applies specific rules
        for non-organic molecules. The calculation may include adding hydrogens
        or hiding metal-ligand bonds for appropriate evaluation.

        Raises
        ------
        None

        Returns
        -------
        int
            The calculated molecular charge.
        """
        self.calc_atom_valence()

        clone = copy(self)
        if not clone.has_hydrogens:
            clone.add_hydrogens()

        if clone.is_organic:
            return clone.sum_explicit_hydrogens - clone.sum_implicit_hydrogens
        else:
            # clone = copy(self)
            clone.hide_metal_ligand_bonds()

            charge = 0
            for c in clone.components:
                if c.is_organic:
                    charge += c.sum_explicit_hydrogens - c.sum_implicit_hydrogens
                elif len(c.atoms) == 1:
                    charge += c.atoms[0].get_formal_charge()
                else:
                    pass

            return charge

    def _after_hide_metal_ligand_bonds(self, func: Callable, atom_attrs: list, bond_attrs: list):
        self.refresh_atom_id()
        self.hide_metal_ligand_bonds()

        for c in self.components:
            func(c)
            atom_attrs_dict = {a.id: {a_attr: getattr(a, a_attr) for a_attr in atom_attrs} for a in c.atoms}
            bond_attrs_dict = {b.id: {b_attr: getattr(b, b_attr)} for b_attr in bond_attrs for b in c.bonds}
            self.update_atoms_attrs_from_id_dict(atom_attrs_dict)
            self.update_bonds_attrs_from_id_dict(bond_attrs_dict)

    def _retrieve_torsions(self):
        """
        Retrieve all torsions associated with the bonds of a molecule.

        This method calculates torsions for each bond in the molecule by 
        iterating over neighboring atoms of both bonded atoms. A torsion 
        is defined by four atoms: two bonded atoms and their respective 
        neighbors. This information is used to create `Torsion` objects.

        Raises:
            None

        Returns:
            list: A list of `Torsion` objects representing all the torsions 
            in the molecule.
        """
        torsion = []
        t1 = time.time()
        for bond in self.bonds:
            a1_neigh = list(bond.atom1.neighbours)
            a2_neigh = list(bond.atom2.neighbours)
            a1_neigh.remove(bond.atom2)
            a2_neigh.remove(bond.atom1)

            for a, d in product(a1_neigh, a2_neigh):
                torsion.append(Torsion(a, bond.atom1, bond.atom2, d))
        t2 = time.time()
        logging.info(f'Torsion calculation took {t2-t1} seconds')

        return torsion

    def simple_paths(self, source=None, target=None, cutoff=None):
        """
        Return all simple paths in a graph.

        This method computes all simple paths between specified source and target nodes
        within the provided graph. If no source and target are specified, it computes all
        simple paths between all node pairs in the graph. The result includes all paths
        up to a specified maximum "cutoff" length.

        Parameters:
            source (int, optional): The starting node for paths. If None, paths from all nodes
                                     are considered. Defaults to None.
            target (int, optional): The ending node for paths. If None, paths to all nodes
                                     are considered. Defaults to None.
            cutoff (int, optional): The maximum path length to compute. If None, no length
                                     limit is applied. Defaults to None.

        Returns:
            list: A list of all simple paths in the graph, where each path is represented
                  as a sequence of nodes.
        """
        if source is None and target is None:
            st = combinations(self.graph.nodes, 2)
        elif isinstance(source, int) and isinstance(target, int):
            st = [(source, target)]
        elif isinstance(source, int):
            st = [(source, t) for t in self.graph.nodes if t!=source]
        else:
            st = [(s, target) for s in self.graph.nodes if s!=target]

        return [p for s, t in st for p in nx.all_simple_paths(self.graph, s, t, cutoff=cutoff)]

    def _set_coordinates(self, coords: types.ArrayLike):
        coords = np.array(coords)
        if coords.shape != (len(self._atoms), 3):
            raise ValueError(f"The shape of coordinates should be {(len(self._atoms), 3)}, but got {coords.shape}")

        for atom, coord in zip(self._atoms, coords):
            atom.coordinates = coord

    def _update_atom_neighbours_bonds(self):
        for atom in self.atoms:
            atom._neighbours = []
            atom._bonds = []

        for bond in self.bonds:
            bond.atom1._neighbours.append(bond.atom2)
            bond.atom2._neighbours.append(bond.atom1)
            bond.atom1._bonds.append(bond)
            bond.atom2._bonds.append(bond)

    def _update_graph(self, clear_conformers=True):
        """
        Updates the molecular graph representation along with clearing conformers and 
        related atom sequences if specified. The method initializes a new networkx 
        Graph for the molecule, attaches updated nodes and edges with their respective 
        attributes, and resets properties such as angles, torsions, and rings.

        Parameters:
        clear_conformers: bool
            If True, clears all existing molecular conformers.

        Raises:
        None
        """
        self._graph = nx.Graph()
        self._graph.add_edges_from(self._edge_with_attrs())
        self._graph.add_nodes_from(self._node_with_attrs())

        # update neighbours and bonds of atoms
        self._update_atom_neighbours_bonds()

        # clear the older AtomSeq
        self._angles = []
        self._torsions = []
        self._rings = []
        self._obmol = None

        if clear_conformers:
            self.conformers.clear()

    def _add_atom(self, atom):
        """
        Adds an atom to the molecule and associates it with the molecule instance.

        This method appends an atom to the molecule's internal list of atoms and sets
        the molecule attribute of the atom to the current molecule instance. The
        method is used to establish a bidirectional association between the molecule
        and its atoms.

        Parameters:
            atom: Atom
                The Atom instance to add to the molecule.

        Returns:
            Atom
                The Atom instance that was added.
        """
        self._atoms.append(atom)
        atom.mol = self

        return atom

    def _add_bond(self, atom1: Union[int, "Atom"], atom2: Union[int, "Atom"], bond_order=1., **kwargs):
        """
        Adds a bond between two atoms of a molecule with the specified bond order.

        A bond is created between two atoms, which must belong to the same
        molecule. The function verifies if the specified atoms are part of
        the molecule instance and ensures that the bond does not already
        exist between these atoms. If validation passes, the bond is created
        and added to the molecule's internal storage.

        Parameters:
            atom1 (Union[int, Atom]): An `Atom` instance or index of the atom
                in the molecule to form one end of the bond.
            atom2 (Union[int, Atom]): An `Atom` instance or index of the atom
                in the molecule to form the other end of the bond.
            bond_order (float): The bond order defining the multiplicity of
                the bond (default is 1.0).
            **kwargs: Additional keyword arguments for bond initialization.

        Returns:
            Bond: The created `Bond` instance representing the connection
            between the two specified atoms.

        Raises:
            ValueError: If any of `atom1` or `atom2` do not belong to the
            molecule, or a bond between the specified atoms already exists.
        """
        atom1 = atom1 if isinstance(atom1, Atom) else self._atoms[atom1]
        atom2 = atom2 if isinstance(atom2, Atom) else self._atoms[atom2]

        if atom1.mol is not self or atom2.mol is not self:
            raise ValueError("at least one of atom1 and atom2 not on the molecule!")

        if (atom1.idx, atom2.idx) in self.graph.edges:
            raise ValueError(f"the bond between {atom1.idx} and {atom2.idx} is already exist!!")

        kwargs['bond_order'] = bond_order
        bond = Bond(atom1, atom2, **kwargs)
        self._bonds.append(bond)

        return bond

    def add_atom(self, atom):
        """
        add_atom(self, atom)

        Adds an atom to the internal graph representation and updates
        the graph structure accordingly. This method ensures the graph
        remains up-to-date after a new atom is added, allowing for
        further operations or analysis on the updated graph.

        Parameters:
            atom: The atom to be added to the graph.

        Returns:
            The atom object added to the graph.
        """
        atom = self._add_atom(atom)
        self._update_graph()
        return atom

    def _replace_atom(
            self,
            original_atom_idx: int,
            new_atom: Union[int, str, "Atom"],
    ):
        if isinstance(new_atom, Atom):
            pass
        elif isinstance(new_atom, int):
            new_atom = Atom(atomic_number=new_atom)
        elif isinstance(new_atom, str):
            new_atom = Atom(symbol=new_atom)
        else:
            raise ValueError(f"Unsupported atom type {type(new_atom)}")
        new_atom.mol = self

        ori_atom = self._atoms[original_atom_idx]
        self._atoms[original_atom_idx] = new_atom

        for bond in ori_atom.bonds:
            Bond._replace_atom(bond, ori_atom, new_atom)

        return ori_atom

    def auto_pair_metal(self, metal, threshold=0., greedy=True):
        from .AImodels.cbond.apply import auto_build_cbond
        return auto_build_cbond(self, metal, threshold, greedy)

    def replace_atom(
            self,
            original_atom_idx: int,
            new_atom: Union[int, str, "Atom"],
    ):
        """ Replaces an atom in the molecule to the other external, while inherit the environment of original one."""
        ori_atom = self._replace_atom(original_atom_idx, new_atom)
        self._update_graph()
        self.calc_implicit_hydrogens()
        return ori_atom

    def add_bond(self, atom1: Union[int, "Atom"], atom2: Union[int, "Atom"], bond_order=1., **kwargs):
        """
        Adds a bond between two atoms in the molecule with the specified bond order and updates
        the molecular graph representation. This method ensures that the bond is created and
        reflected in the graph structure.

        Parameters:
        atom1: Union[int, Atom]
            The first atom participating in the bond. Can be provided as an integer indicating
            the atom's index or as an Atom object.
        atom2: Union[int, Atom]
            The second atom participating in the bond. Can be provided as an integer indicating
            the atom's index or as an Atom object.
        bond_order: float, optional
            The order of the bond to be created. Defaults to 1.0.
        **kwargs
            Additional keyword arguments that may be passed to customize the bond creation.

        Returns:
        Bond
            The Bond object representing the created bond.
        """
        bond = self._add_bond(atom1, atom2, bond_order, **kwargs)
        self._update_graph()
        return bond

    def add_bonds(self, idx_order: Iterable[tuple[int, int, float]], list_kw: Optional[list[dict]] = None) -> list["Bond"]:
        if list_kw:
            list_kw = list(list_kw)
        else:
            list_kw = [{} for _ in range(len(idx_order))]
        assert len(idx_order) == len(list_kw)

        bonds = []
        for ido, kw in zip(idx_order, list_kw):
            bonds.append(self._add_bond(*ido, **kw))

        return bonds

    def add_component(self, component: "Molecule"):
        """
        Adds a molecule component to the current structure by integrating its atoms and bonds.

        Parameters
        ----------
        component: Molecule
            The molecule component to be added to the current structure. It
            will be copied, and its atoms and bonds will be integrated into
            the structure.

        Notes
        -----
        This method incorporates all atoms from the provided molecule
        component into the current structure. Bonds present in the molecule
        component will also be included. The graph representing the molecule
        structure is updated after the addition.

        """
        component = copy(component)
        for atom in component.atoms:
            self._add_atom(atom)
        self._bonds.extend(component.bonds)
        del component
        self._update_graph()

    def add_hydrogens(self, rm_polar_hs: bool = True):
        """
        Adds or removes hydrogen atoms to/from the molecule.

        This method iterates over all atoms in the molecule and determines whether
        hydrogen atoms need to be added or removed based on the polarity settings.
        If any modifications are made to the molecular structure, the graph is updated
        accordingly.

        Parameters
        ----------
        rm_polar_hs : bool, optional
            Determines whether polar hydrogens should be removed. Defaults to True.

        Returns
        -------
        None
        """
        # Add hydrogens
        modified = False
        for atom in self.atoms:
            if not (atom.is_hydrogen or atom.is_metal):
                add_or_rm, hs = atom._add_hydrogens(rm_polar_hs=rm_polar_hs)
                if add_or_rm:
                    modified = True

        if modified:
            self._update_graph()

    @property
    def cid(self) -> Optional[int]:
        return smi_to_cid(self.smiles)

    @property
    def name(self) -> str:
        return smi_to_name(self.smiles)

    @property
    def cas(self):
        return smi_to_cas(self.smiles)

    def clear_constraints(self) -> None:
        """ clear all set constraints """
        for atom in self.atoms:
            atom.constraint = False
        for bond in self.bonds:
            bond.constraint = False
        for angle in self.angles:
            angle.constraint = False
        for torsion in self.torsions:
            torsion.constraint = False

    def clear_metal_ligand_bonds(self) -> None:
        """
        Clears all existing metal-ligand bonds in the current instance. This involves hiding 
        the visual representation of the bonds and resetting the data structure responsible 
        for tracking the broken bonds.

        Methods  
        -------
        hide_metal_ligand_bonds()
            Called to hide the visual representation of the metal-ligand bonds.
        """
        self.hide_metal_ligand_bonds()
        self._hided_metal_bonds = []

    def clear_hided_covalent_bonds(self) -> None:
        self._hided_covalent_bonds = []

    def clear_hided_bonds(self) -> None:
        self._hided_metal_bonds = []
        self._hided_covalent_bonds = []

    def recover_hided_metal_ligand_bonds(self, clear_conformers: bool = False) -> None:
        """
        Restores metal-ligand bonds that were previously broken and updates the internal graph representation.

        This function recovers all previously broken metal-ligand bonds by adding them back 
        to the existing bond list. It ensures no duplicate bonds exist by using a set operation. 
        Additionally, it updates the internal graph representation to reflect the recovered bonds 
        and provides the option to clear conformer data. After recovery, the list of broken metal bonds 
        is reset.

        Arguments:
            clear_conformers (bool, optional): Determines whether existing conformer data should 
                                               be cleared during the graph update. Defaults to False.

        Returns:
            None
        """
        if self._hided_metal_bonds:
            self._bonds = list(set(self._bonds + self._hided_metal_bonds))
            self._update_graph(clear_conformers)
            logging.info(f"[green]Recover {len(self._hided_metal_bonds)} hided metal-ligand bonds[/]")
            self._hided_metal_bonds = []

    def recover_hided_covalent_bonds(self, clear_conformers: bool = False) -> None:
        if self._hided_covalent_bonds:
            self._bonds = list(set(self._bonds + self._hided_covalent_bonds))
            self._update_graph(clear_conformers)
            logging.info(f"[green]Recover {len(self._hided_covalent_bonds)} hided covalent bonds[/]")
            self._hided_covalent_bonds = []

    @property
    def atom_pairs(self) -> "AtomPairs":
        """
        Provides access to atom pair information stored in the object.

        This property retrieves the `_atom_pairs` attribute of the object, which 
        encapsulates atom pair information in the form of an `AtomPairs` object. 
        The returned `AtomPairs` object provides detailed data and functionalities 
        pertaining to atom pair relationships.

        Returns:
            AtomPairs: The atom pair information associated with the object.
        """
        return self._atom_pairs

    @property
    def angles(self) -> list["Angle"]:
        """
        Gets the angles property which is a list of Angle objects representing
        all possible angles formed by the given atoms and their neighbors. The
        property is generated lazily and cached for future access.

        Returns:
            list[Angle]: A list of Angle objects, where each angle is formed by
            a combination of two neighbors of an atom present in the `atoms`.

        """
        if not self._angles:
            self._angles = [Angle(n1, a, n2) for a in self.atoms for n1, n2 in combinations(a.neighbours, 2)]
        return copy(self._angles)

    @property
    def torsions(self) -> list["Torsion"]:
        """
            Retrieves and returns a copy of the torsions associated with the object.
            The torsions are computed once and cached for future retrieval.

            Returns:
                list["Torsion"]: A list of torsion objects. A copy of the cached
                list is always returned to ensure the original list is immutable
                to external changes.
        """
        if not self._torsions:
            self._torsions = self._retrieve_torsions()
        return copy(self._torsions)

    def assign_aromatic(self):
        pass

    def assign_bond_order(self):
        """
        Assigns bond orders to the molecule and updates its structure accordingly.

        This function is designed to manage bond order assignment in a way that temporarily 
        hides metal-ligand bonds, performs the bond order assignment, and then restores the 
        hidden bonds. It ensures the molecule's structural integrity during this process.

        Raises:
            Exception: If any issues occur during bond order assignment or if the molecular 
            structure is invalid.
        """
        self.hide_metal_ligand_bonds()
        obc.assign_bond_order(self)
        self.recover_hided_metal_ligand_bonds()

    @property
    def atom_attr_matrix(self) -> np.ndarray:
        """
        Returns the attribute matrix of atoms.

        This property compiles the attributes of all atoms in the 
        object into a matrix format for further processing or analysis.

        @rtype: np.ndarray
        @return: A 2D numpy array where each row represents the attributes 
                 of an atom.
        """
        return np.array([a.attrs for a in self._atoms])

    @property
    def atoms(self):
        """
        Provides an interface to access the atoms of the molecule. This property 
        returns a copy of the internal _atoms list to ensure the encapsulation of 
        the original data. Modification of the returned list does not affect the 
        internal state of the object.

        Returns:
            list: A copy of the internal _atoms list that can be freely modified 
            without impacting the internal state.
        """
        return copy(self._atoms)

    @property
    def c_bonds(self) -> list["Bond"]:
        """ Get all coordination bonds in the Molecule """
        return [b for b in self.bonds if b.is_metal_ligand_bond]

    @property
    def bonds(self) -> list["Bond"]:
        """
        Retrieves a copy of the bonds.

        Returns a duplicate of the stored bonds data, ensuring that the original 
        bonds information remains unmodified during external operations. This 
        ensures data integrity while allowing external access to the data.

        Returns:
            list: A copy of the stored bonds data.
        """
        return copy(self._bonds)

    def bond(self, a1idx: int, a2idx: int):
        """
        Gets the bond information between two atoms in the graph.

        This method retrieves the bond data from the graph structure given
        the indices of the two atoms. The user must supply the indices of
        the atoms as integers, and the bond data will be returned as
        stored in the graph's edge attributes.

        Parameters:
        a1idx : int
            The index of the first atom.
        a2idx : int
            The index of the second atom.

        Returns:
        The bond information stored in the graph's edge attributes between
        the two atoms.
        """
        return self.graph.edges[a1idx, a2idx]['bond']

    def build3d(
            self,
            forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']] = 'UFF',
            steps: int = 500,
            sophisticated: bool = True,
            **kwargs
    ):
        """
        Builds a 3D structure for the molecule using the specified forcefield, optimization
        steps, and sophistication mode. This method ensures that either a simple or
        complex building and optimization procedure is applied depending on the 
        sophistication flag and molecule type.

        :param forcefield: Specifies the forcefield to be used for optimization.
                           Default is 'UFF'. Supported forcefields include 'UFF', 
                           'MMFF94', 'MMFF94s', 'GAFF', and 'Ghemical'.
        :type forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']]
        :param steps: Number of optimization steps for 3D structure generation.
        :type steps: int
        :param sophisticated: If True, applies a sophisticated approach for building 
                               and optimizing the 3D structure for organic molecules.
        :type sophisticated: bool
        :param kwargs: Additional parameters passed to the building or optimization 
                       methods.
        :type kwargs: dict
        """
        if sophisticated and not self.is_organic:
            ff.complexes_build(self, **kwargs)

        else:
            ff.ob_build(self)
            ff.ob_optimize(self, forcefield, steps)

    def update_mol_charge(self):
        """
        Updates the molecular charge by summing the charges of all constituent
        atoms. This method assumes the charges of individual atoms have been 
        correctly assigned beforehand.

        """
        self.charge = self.sum_atoms_charge

    @property
    def sum_atoms_charge(self) -> int:
        """
        Calculates the sum of formal charges of all atoms in the molecule.

        This property iterates through all atoms and sums their formal charges, providing
        the total charge value of the molecule.

        @return: An integer representing the total sum of formal charges of the atoms.
        """
        return sum(a.formal_charge for a in self._atoms)

    @property
    def default_unpaired_electrons(self):
        """
        Returns the default number of unpaired electrons in a molecular system.

        The calculation is based on the total sum of atomic numbers of the 
        atoms in the system, adjusted for the system's charge. The result 
        is taken modulo 2 to determine the default number of unpaired 
        electrons.

        @return int: Default number of unpaired electrons in the system.
        """
        return (sum(a.atomic_number for a in self.atoms) - self.charge) % 2

    @property
    def default_spin_mult(self) -> int:
        """
        This method calculates and returns the default spin multiplicity based on the 
        current atomic configuration and charge. The spin multiplicity is determined 
        by the formula: (sum of atomic numbers of all atoms - total charge) % 2 + 1.

        Returns
        -------
        int
            The default spin multiplicity as an integer value.
        """
        return (sum(a.atomic_number for a in self.atoms) - self.charge) % 2 + 1

    def determine_rings_aromatic(self):
        """
        Determines whether the rings in a molecule are aromatic, applying specific procedures such as
        hiding metal-ligand bonds and kekulizing the rings. The method involves iterating through 
        predefined rings of the molecule and addressing individual ring properties.

        Methods
        -------
        determine_rings_aromatic:
            Processes the rings in the molecule to determine their aromaticity, temporarily modifying
            connectivity before restoring any changes.

        Parameters
        ----------
        This method does not take any parameters.

        Raises
        ------
        This method does not explicitly raise any exceptions.

        Returns
        -------
        This method does not return any value.
        """
        self.hide_metal_ligand_bonds()
        for ring in self.rings:
            # ring.determine_aromatic(inplace=True)
            ring.kekulize()

        self.recover_hided_metal_ligand_bonds()

    @property
    def conformers(self) -> "Conformers":
        """
        Returns the conformers object associated with the current instance.

        This property provides access to the Conformers object which is used
        to store and manipulate the relevant conformers related to the current
        context.

        @return: Conformers object representing the conformers for the instance.
        """
        return self._conformers

    def conformer_load(self, i: int):
        """ Load specific conformer to be current conformer """
        self.coordinates = self.conformers.index_attr('coordinates', i)
        self._conformers_index = i

    @property
    def conformers_number(self) -> int:
        """
        Returns the number of conformers in the current object. This property provides access to the count of all conformers stored 
        in the object, which can be useful for iteration or data validation processes.

        @return:
            The total count of conformers currently stored in the object.

        @rtype:
            int
        """
        return len(self._conformers)

    def conformer_add(
            self, coords: Optional[types.ArrayLike] = None,
            energy: Optional[Union[types.ArrayLike, float]] = None
    ):
        """
        Add a new conformer to the collection.

        This function allows adding a conformer with optional coordinates and energy
        values. If no coordinates are provided, the current coordinates of the object
        are used. The energy value can be assigned optionally as well.

        Parameters:
        coords : Optional[types.ArrayLike]
            The coordinates of the conformer. If None, the current coordinates
            will be used.
        energy : Optional[Union[types.ArrayLike, float]]
            The energy associated with the conformer. This value is optional.

        Returns:
        None
        """
        if coords is None:
            self._conformers.add(self.coordinates, energy)
        else:
            self._conformers.add(coords, energy)

    def conformer_clear(self):
        """
        Clears all stored conformers from the internal conformer storage.

        This method is used to empty the internal list of conformers,
        removing all previously stored data. This can be useful when resetting
        or preparing the object for storing a new set of conformers.

        Raises:
            No exceptions are raised by this method.
        """
        self._conformers.clear()

    def conformer_get(self, idx: Union[int, slice]) -> dict:
        """ Get specific conformer coordinates """
        return self._conformers[idx]

    def optimize(
            self,
            forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']] = None,
            algorithm: Literal["steepest", "conjugate"] = "conjugate",
            steps: Optional[int] = 100,
            step_size: int = 100,
            equilibrium: bool = True,
            equi_check_steps: int = 5,
            equi_max_displace: float = 1e-4,
            equi_max_energy: float = 1e-4,
            perturb_steps: Optional[int] = None,
            perturb_sigma: float = 0.5,
            save_screenshot: bool = False,
            increasing_Vdw: bool = False,
            Vdw_cutoff_start: float = 0.0,
            Vdw_cutoff_end: float = 12.5,
            print_energy: Optional[int] = None
    ):
        """
            Optimize the atomistic model structure using a specified force field and algorithm. 
            The optimization process includes features for equilibrium checks, perturbation, 
            and van der Waals cutoff techniques to fine-tune the molecular geometry.

            Parameters:
                forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']]
                    The force field to be used for optimization.
                    Defaults to 'UFF' if the model contains metal atoms, 
                    otherwise defaults to 'MMFF94s'.
                algorithm: Literal["steepest", "conjugate"]
                    The optimization algorithm to use. Options are "steepest" 
                    for Steepest Descent and "conjugate" for Conjugate Gradient.
                    Default is "conjugate".
                steps: Optional[int]
                    The number of optimization steps to perform. Default is 100.
                step_size: int
                    Size of each optimization step. Default is 100.
                equilibrium: bool
                    Indicates whether to perform equilibrium checks. Default is True.
                equi_check_steps: int
                    Number of steps at which equilibrium checks are performed. 
                    Default is 5.
                equi_max_displace: float
                    Maximum allowed displacement per step for equilibrium detection.
                    Default is 1e-4.
                equi_max_energy: float
                    Maximum allowed energy change for equilibrium detection. 
                    Default is 1e-4.
                perturb_steps: Optional[int]
                    Number of steps for applying a random perturbation. Default is None.
                perturb_sigma: float
                    Standard deviation for the random displacement applied during
                    perturbation. Default is 0.5.
                save_screenshot: bool
                    Specifies if screenshots should be saved during optimization.
                    Default is False.
                increasing_Vdw: bool
                    Indicates whether vdW (van der Waals) cutoff should increase during 
                    the iteration. Default is False.
                Vdw_cutoff_start: float
                    The starting cutoff distance for vdW interactions. Default is 0.0.
                Vdw_cutoff_end: float
                    The maximum cutoff distance for vdW interactions. Default is 12.5.
                print_energy: Optional[int]
                    Print the energy every specific number of steps. If None, energy
                    printing is disabled. Default is None.

            Raises:
                ValueError: Raised if provided values are invalid for specific 
                options or configurations within the parameters.

            Returns:
                None
        """
        arguments = copy(locals())
        del arguments["self"]
        del arguments["forcefield"]

        if forcefield is None:
            if self.has_metal:
                arguments['ff'] = 'UFF'
            else:
                arguments['ff'] = 'MMFF94s'
        else:
            arguments['ff'] = forcefield

        ff.OBFF(**arguments).optimize(self)

    def optimize_complexes(
            self,
            algorithm: Literal["steepest", "conjugate"] = "steepest",
            steps: Optional[int] = None,
            equilibrium: bool = True,
            equi_threshold: float = 1e-4,
            max_iter: int = 100,
            save_screenshot: bool = False
    ):
        """
        Optimize the geometry of metal-ligand complexes using a specified force field.

        This method is specifically tailored for optimizing metal-ligand complexes. 
        If the system does not contain metals, it will redirect to the standard `optimize()` method,
        which is faster and more suitable for organic compounds.

        Parameters:
            algorithm (Literal["steepest", "conjugate"], optional): 
                Optimization algorithm to use. Options are:
                - "steepest": Steepest descent algorithm.
                - "conjugate": Conjugate gradient algorithm.
                Defaults to "steepest".

            steps (Optional[int], optional): 
                Number of optimization steps to perform. If None, a default value will be used.

            equilibrium (bool, optional): 
                Whether to enforce equilibrium conditions during optimization. Defaults to True.

            equi_threshold (float, optional): 
                Threshold for equilibrium convergence. Defaults to 1e-4.

            max_iter (int, optional): 
                Maximum number of iterations for the optimizer. Defaults to 100.

            save_screenshot (bool, optional): 
                Whether to save a snapshot of the structure after optimization. Defaults to False.

        Notes:
            - For metal-ligand complexes, uses the Universal Force Field (UFF) for optimization, 
              with special handling of metal-ligand interactions.
            - Organic subcomponents are optimized separately, and constraints are applied to 
              non-metal atoms before proceeding with metal-ligand system optimization.
            - If no metal is detected in the structure, the slower metal-specific workflow 
              is bypassed in favor of the more efficient default optimization routine 
              (`optimize()`).

        Warnings:
            - When applied to organic compounds without metals, this method is slower 
              than the recommended `optimize()` method.

        """
        if not self.has_metal:
            print(UserWarning(
                "The `optimize_complexes()` is specified for metal-ligand complexes, \n"
                "it's much slower than `optimize()` method. For organic compounds, the \n"
                "`optimize() is more recommended."
            ))
            self.optimize(
                'MMFF94s',
                algorithm, steps, equilibrium,
            )
            return


        self.refresh_atom_id()

        # Initialize optimizer
        obff = ff.OBFF(
            ff='UFF',
            algorithm=algorithm,
            steps=steps,
            equilibrium=equilibrium,
            equi_threshold=equi_threshold,
            max_iter=max_iter,
            save_screenshot=False
        )

        clone = copy(self)
        clone.hide_metal_ligand_bonds()

        for component in clone.components:
            if component.is_organic:
                obff.optimize(component)
                clone.update_atoms_attrs_from_id_dict({a.id: {'coordinates': a.coordinates} for a in component.atoms})

        clone.recover_hided_metal_ligand_bonds()
        # clone.constraint_bonds_angles()
        for a in clone.atoms:
            if not a.is_metal:
                a.constraint = True

        obff.ff = ob.OBForceField.FindType('UFF')
        obff.save_screenshot = save_screenshot
        # obff.optimize(clone)

        obff.equilibrium = True
        obff.perturb_steps = 30
        obff.perturb_sigma = 0.5
        self.coordinates = clone.coordinates
        self._conformers = clone.conformers
        obff.optimize(self)

    def complexes_build_optimize_(
            self,
            algorithm: Literal["steepest", "conjugate"] = "conjugate",
            steps: Optional[int] = 500,
            step_size: int = 100,
            equilibrium: bool = False,
            equi_check_steps: int = 5,
            equi_max_displace: float = 1e-4,
            equi_max_energy: float = 1e-4,
            perturb_steps: Optional[int] = 50,
            perturb_sigma: float = 0.5,
            save_screenshot: bool = True,
            increasing_Vdw: bool = False,
            Vdw_cutoff_start: float = 0.0,
            Vdw_cutoff_end: float = 12.5,
            print_energy: Optional[int] = 100,
            # parameter for complexes build
            build_times: int = 5,
            init_opt_steps: int = 500,
            second_opt_steps: int = 1000,
            min_energy_opt_steps: int = 3000,
            rm_polar_hs: bool = True
    ):
        """
        Optimize and optionally build metal-ligand complexes using specified parameters.

        This method is designed for optimization of molecular structures, particularly 
        metal-ligand complexes. It supports building complexes from scratch and refining 
        them using energy minimization algorithms. For organic compounds without metals, 
        this method falls back to a faster alternative `optimize()`.

        Parameters:
            algorithm (Literal["steepest", "conjugate"]): The optimization algorithm to 
                use. Defaults to "conjugate".
            steps (Optional[int]): The maximum number of optimization steps. Defaults 
                to 500.
            step_size (int): The size of a single optimization step. Defaults to 100.
            equilibrium (bool): Whether to check for equilibrium conditions during 
                optimization. Defaults to False.
            equi_check_steps (int): How frequently to check for equilibrium during 
                optimization, measured in the number of steps. Defaults to 5.
            equi_max_displace (float): The maximum allowed displacement to reach 
                equilibrium. Defaults to 1e-4.
            equi_max_energy (float): The maximum allowed energy change to reach 
                equilibrium. Defaults to 1e-4.
            perturb_steps (Optional[int]): The number of steps for structure 
                perturbation. Defaults to 50.
            perturb_sigma (float): The magnitude of perturbation applied during 
                perturbation phases. Defaults to 0.5.
            save_screenshot (bool): Whether to save graphical screenshots after key 
                stages of optimization. Defaults to True.
            increasing_Vdw (bool): If true, the Van der Waals cutoff distance is 
                increased incrementally during optimization stages. Defaults to False.
            Vdw_cutoff_start (float): Initial Van der Waals cutoff distance. Defaults 
                to 0.0.
            Vdw_cutoff_end (float): Final Van der Waals cutoff distance. Defaults to 
                12.5.
            print_energy (Optional[int]): Frequency of energy reporting during 
                optimization. Defaults to 100.
            build_times (int): The number of attempts allowed for building the 
                complex. Defaults to 5.
            init_opt_steps (int): Number of optimization steps in the initial stage. 
                Defaults to 500.
            second_opt_steps (int): Number of optimization steps in the second stage. 
                Defaults to 1000.
            min_energy_opt_steps (int): Number of optimization steps in the final stage 
                targeting minimal energy. Defaults to 3000.
            rm_polar_hs (bool): Removes polar hydrogens before generating the complex 
                if set to True. Defaults to True.

        Returns:
            None
        """
        arguments = copy(locals())
        arguments.pop('self')

        # For organic compound
        if not self.has_metal:
            print(UserWarning(
                "The `optimize_complexes()` is specified for metal-ligand complexes, \n"
                "it's much slower than `optimize()` method. For organic compounds, the \n"
                "`optimize() is more recommended."
            ))
            arguments['ff'] = 'MMFF94s'
            self.optimize_(**arguments)
            return

        # build complex
        ff.complexes_build(
            self,
            build_times,
            init_opt_steps,
            second_opt_steps,
            min_energy_opt_steps,
            rm_polar_hs=rm_polar_hs
        )

        # Initialize optimizer
        arguments['ff'] = 'UFF'
        obff = ff.OBFF_(**arguments)
        obff.ff.SetVDWCutOff(12.5)
        obff.optimize(self)

    def calc_atom_valence(self, assign_aromatic: Optional[bool] = None):
        """
        Calculate the valence and number of implicit hydrogens for each atom.

        The method iterates through all the atoms in the `atoms` list, determines the valence
        of each atom using the `get_valence` method, and calculates the implicit hydrogens
        using the `calc_implicit_hydrogens` method. The results are assigned to the respective
        attributes of each atom.

        """
        if assign_aromatic is not False:
            rings = self.ligand_rings
            if assign_aromatic or (rings and not any(r.is_aromatic for r in rings)):
                for r in rings:
                    r.determine_aromatic(inplace=True)

        for atom in self.atoms:
            atom.valence = atom.get_valence()
            atom.calc_implicit_hydrogens()

    def calc_implicit_hydrogens(self):
        """
        Calculates the implicit hydrogens for each atom in the molecule.

        This method iterates over all atoms in the molecule and calculates the 
        number of implicit hydrogens for each atom. This is useful for analyzing 
        structures where explicit hydrogen atoms are not represented.

        """
        for a in self.atoms:
            a.calc_implicit_hydrogens()

    @property
    def components(self) -> list['Molecule']:
        """
        Fetches and returns the connected components of the molecule as a list of Molecule objects. 
        Each component is a distinct connected substructure derived from the molecular graph.

        Returns:
            list[Molecule]: A list of Molecule objects, each representing one connected component 
                            of the molecular graph.

        Raises:
            None
        """
        graph = self.graph

        components = []
        for c_node_idx in nx.connected_components(graph):
            c_node_idx = tuple(c_node_idx)
            subgraph = graph.subgraph(c_node_idx)

            component = Molecule()

            for node_idx in c_node_idx:
                component._create_atom(**graph.nodes[node_idx])

            for edge_begin_idx, edge_end_index in subgraph.edges:
                component._add_bond(
                    c_node_idx.index(edge_begin_idx),
                    c_node_idx.index(edge_end_index),
                    **subgraph.edges[edge_begin_idx, edge_end_index]['bond'].attr_dict
                )

            component._update_graph()
            components.append(component)

        return components

    def constraint_bonds_angles(self, exclude_metal_bonds: bool = True):
        """
        Constrains bonds and angles based on the provided conditions.

        This method updates the angles first, then applies constraints to the bonds
        and angles depending on whether they involve metals. Bonds and angles with
        metals are excluded from constraints if the flag `exclude_metal_bonds` is set
        to True.

        Args:
            exclude_metal_bonds (bool): A flag to indicate whether bonds and angles 
                involving metals should be excluded from constraints. Default is True.
        """
        self.update_angles()
        if exclude_metal_bonds:
            for bond in self.bonds:
                if not bond.has_metal:
                    bond.constraint = True
            for angle in self.angles:
                if not angle.has_metal:
                    angle.constraint = True

    @property
    def coordinates(self) -> np.ndarray:
        """
        Returns the coordinates of all atoms in the structure.

        This property computes and returns the coordinates of all the atoms 
        contained in the `_atoms` class attribute. The coordinates are organized
        as a NumPy array, with each atom's coordinates represented as an individual
        element in the array.

        Returns:
            np.ndarray: A NumPy array containing the coordinates of atoms.

        """
        return np.array([a.coordinates for a in self._atoms])

    @coordinates.setter
    def coordinates(self, value: np.ndarray) -> None:
        """
        Sets the coordinates of all atoms in the system.

        This method ensures that the given coordinates match the expected dimensions
        based on the number of atoms in the system. It iterates through each atom and
        assigns the corresponding coordinate values to it.

        Args:
            value (np.ndarray): A 2D numpy array containing the coordinates of all 
                atoms. The array should have a shape of (len(self.atoms), 3) where 
                each row corresponds to the x, y, z coordinates of an atom.

        """
        assert value.shape == (len(self.atoms), 3)
        for a, row in zip(self.atoms, value):
            a.coordinates = row

    def _create_atom_from_array(self, attrs_array: np.ndarray) -> "Atom":
        """
        Creates an Atom instance using the provided attributes array.

        This method is used to create an Atom object by passing an array of attributes.
        The returned Atom instance is associated with the current molecule.

        Args:
            attrs_array (np.ndarray): An array containing attributes necessary to
            initialize the Atom object.

        Returns:
            Atom: A new Atom instance initialized with the provided attributes array.
        """
        return Atom(self, attrs_array=attrs_array, update_electron_config=False)

    def _create_atom(self, **kwargs):
        """
        Creates and returns an instance of the Atom class associated with the 
        current object. This method acts primarily as a helper for initializing
        an Atom object with the provided arguments. The returned Atom instance 
        inherits attributes or properties derived from the associated parent 
        instance.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments passed to initialize an Atom 
            object. The arguments can include properties or configurations 
            required by the Atom class. Refer to the Atom constructor for valid 
            parameters.

        Returns
        -------
        Atom
            An instance of the Atom class connected or associated to the 
            current object.
        """
        return Atom(self, **kwargs)

    def create_atom(self, **kwargs) -> 'Atom':
        """
        Creates and returns a new Atom object with the specified properties. 

        This method uses the provided keyword arguments to create a new Atom using 
        an internal helper method. After the creation, it updates the internal graph 
        representation to reflect the addition of the new Atom.

        Parameters:
            **kwargs: Arbitrary keyword arguments used for defining properties of the 
                Atom during the creation process.

        Returns:
            Atom: The newly created Atom object.
        """
        atom = self._create_atom(**kwargs)
        self._update_graph()
        return atom

    @property
    def atom_pairwise_index(self) -> np.ndarray:
        """
        Return an array containing all unique atom pairwise indices within a molecule.

        This property computes all unique combinations of pairs of indices from 
        the atom list within the molecule and returns them as a NumPy array.

        @return: The resulting array is of shape (n, 2), where n is the number of 
        unique pairs based on the length of the atom list in the molecule. Each row 
        represents a pair of atom indices.

        @rtype: np.ndarray
        """
        return np.array(list(combinations(range(len(self.atoms)), 2)))

    def get_partial_charge(
            self,
            model: Literal["eem", "mmff94", "gasteiger", "qeq", "qtpie",
                            "eem2015ha", "eem2015hm", "eem2015hn",
                            "eem2015ba", "eem2015bm", "eem2015bn"] = "qeq"
    ):
        ob_charge_model = ob.OBChargeModel.FindType(model)
        ob_charge_model.ComputeCharges(self.to_obmol())
        return ob_charge_model.GetPartialCharges()

    def assign_partial_charge(
            self,
            model: Literal["eem", "mmff94", "gasteiger", "qeq", "qtpie",
            "eem2015ha", "eem2015hm", "eem2015hn",
            "eem2015ba", "eem2015bm", "eem2015bn"] = "qeq"
    ):
        for a, c in zip(self.atoms, self.get_partial_charge(model = "qtpie")):
            a.partial_charge = c

    @property
    def pair_dist(self) -> np.ndarray:
        """
        Computes the pairwise distance between coordinates.

        This property uses the scipy.spatial.distance.pdist function to calculate
        the pairwise distances between points in the provided coordinates.

        Returns:
            np.ndarray: A 1D array containing the pairwise distances between points
            in the coordinates.
        """
        return pdist(self.coordinates)

    @property
    def dist_matrix(self) -> np.ndarray:
        """
        Gets the distance matrix for a set of coordinates.

        Calculates the pairwise distances between each set of points
        in the given coordinates. The distance calculation is based on
        the Euclidean distance.

        @return: A 2D numpy array representing the pairwise distance
        matrix. Each element [i, j] in the matrix represents the distance
        between point i and point j.
        @rtype: numpy.ndarray
        """
        return squareform(pdist(self.coordinates))

    @property
    def element_counts(self):
        """
        Returns the count of each element in the list of atoms by their symbol.

        Summary:
        Computes the frequency of each element within the atoms of the object 
        by using their symbols. Utilizes a Python Counter for efficient calculation.

        Returns:
            dict: A dictionary mapping element symbols (str) to their counts (int).
        """
        return Counter([a.symbol for a in self.atoms])

    @property
    def simple_graph(self) -> nx.Graph:
        """
        Return a networkx Graph without nodes and edges attrs

        This property generates an undirected graph using the networkx library based on 
        the link matrix attribute of the object. The resulting graph consists of nodes 
        and edges defined by the link matrix, where each entry implies a connection between 
        nodes.

        @return: An undirected graph object built from the link matrix.
        """
        graph = nx.Graph()
        graph.add_edges_from(self.link_matrix)
        return graph

    def _node_with_attrs(self):
        """
        Returns a list of tuples representing nodes with their attributes.

        Each tuple contains the index of the node and a dictionary of its attributes.
        This is useful for constructing graph representations where nodes 
        have associated attributes.

        Returns:
            list[tuple[int, dict]]: A list of tuples, where each tuple consists of an
            integer representing the node index and a dictionary holding the node's 
            attributes.
        """
        # return [(a.idx, {n:getattr(a, n) for n in a.attrs_enumerator}) for a in self._atoms]
        return [(a.idx, {'attrs': a.attrs}) for a in self._atoms]

    def _edge_with_attrs(self):
        """
        Generates a list of tuples representing edges with attributes from the bonds.

        This method processes the list of bonds available in the object and constructs
        a list of tuples where each tuple represents an edge. Each edge contains its
        two endpoint indices and a dictionary of attributes associated with the bond.

        Returns:
            list: A list of tuples where each tuple is of the form 
            (a1idx, a2idx, {'bond': bond_object}).
        """
        # attrs = ('idx',) + Bond._attrs_enumerator
        return [(b.a1idx, b.a2idx, {'bond': b}) for b in self._bonds]

    def _hide_bond(self, bond: "Bond"):
        if bond not in self.bonds:
            raise ObjNotInMolecule(bond, self)

        if bond.has_metal:
            self._hided_metal_bonds.append(bond)
        else:
            self._hided_covalent_bonds.append(bond)

        self._bonds.remove(bond)

    def hide_bonds(self, *bond: "Bond", clear_conformers: bool = False):
        for b in bond:
            self._hide_bond(b)
        self._update_graph(clear_conformers)

    def hide_metal_ligand_bonds(self, clear_conformers: bool = False) -> None:
        """ break all bonds link with metals """
        metal_bonds = [b for b in self.bonds if b.is_metal_ligand_bond]
        self._hided_metal_bonds.extend(metal_bonds)

        for b in metal_bonds:
            self._bonds.remove(b)

        self._update_graph(clear_conformers)

    @property
    def graph(self):
        """ Return networkx graph with nodes and edges attrs """
        return self._graph

    @property
    def atom_bond_graph(self):
        _graph = nx.Graph()
        _graph.add_nodes_from([(a.idx, {'atom': a}) for a in self._atoms])
        _graph.add_edges_from([(b.a1idx, b.a2idx, {'bond': b}) for b in self._bonds])
        return _graph

    def graph_spectral(self, norm: Literal['infinite', 'min', 'l1', 'l2'] = 'l2'):
        """ Return graph spectral matrix """
        clone = copy(self)
        clone.add_hydrogens()
        adj = graph.linkmat2adj(len(clone.atoms), clone.link_matrix)
        return graph.GraphSpectrum.from_adj_atoms(adj, np.array([a.atomic_number for a in clone.atoms]), norm=norm)

    @property
    def formula(self) -> str:
        """ Formula of the molecule """
        formula = ''
        for ele, count in self.element_counts.items():
            formula += f'{ele}{count}'

        return formula

    @property
    def is_disorder(self):
        """
        Check the disorder state based on pairwise distances.

        This property evaluates if there are any pairwise distances in the 
        `pair_dist` attribute that are less than 0.5. If such distances are found, 
        it indicates a disorder state.

        @property
            Returns
            -------
            bool
                True if any distance in `pair_dist` is less than 0.5, False otherwise.
        """
        return np.any(self.pair_dist < 0.5)

    @property
    def has_3d(self):
        """
        Check if the molecular structure has 3D coordinates.

        This property determines if any of the atoms in the molecular structure 
        has 3D coordinates by comparing each atom's coordinates with the 
        coordinates of the first atom in the list.

        @return: Whether the molecular structure has 3D coordinates.
        @rtype: bool
        """
        return any(a.coordinates != self.atoms[0] for a in self.atoms)

    @property
    def has_metal(self) -> bool:
        """
        Check if the object contains any metallic atoms.

        The `has_metal` property determines whether any atom within the object is 
        classified as a metallic atom.

        @property
        @return: bool
            Returns True if at least one atom in the object is metallic, otherwise False.
        """
        return any(a.is_metal for a in self.atoms)

    @property
    def has_bond_ring_intersection(self) -> bool:
        """
        Checks if any bond intersects with any ring in the structure.

        This method evaluates all combinations of rings and bonds in the structure
        to determine if there is an intersection between any ring and any bond. It
        utilizes `is_bond_intersect_the_ring` for individual intersection checks.

        Returns:
            bool: True if there is at least one bond that intersects with a ring;
            False otherwise.
        """
        return any(r.is_bond_intersect_the_ring(b) for r, b in product(self.rings_small, self.bonds))

    @property
    def intersection_bonds_rings(self) -> list[tuple['Ring', 'Bond']]:
        return [(r, b) for r, b in product(self.rings_small, self.bonds) if r.is_bond_intersect_the_ring(b)]

    @property
    def heavy_atoms(self) -> list["Atom"]:
        """
        Gets the heavy atoms from the list of atoms in the molecule.

        The heavy atoms are defined as atoms with an atomic number different
        from 1 (not hydrogen). This property filters the internal list of
        atoms and returns only the heavy atoms.

        @return list["Atom"]: List of heavy atoms present in the molecule.
        """
        return [a for a in self._atoms if a.atomic_number != 1]

    @property
    def is_organic(self) -> bool:
        """
        Determines whether a molecule is organic based on its atomic composition. 

        Organic molecules are defined as containing no metal atoms, at least one
        hydrogen atom (explicit or implicit), and at least one carbon atom.

        @return: True if the molecule meets the criteria for being organic; otherwise, False
        @rtype: bool
        """
        return (
            all(not a.is_metal for a in self.atoms) and
            any(a.is_hydrogen or a.implicit_hydrogens != 0 for a in self._atoms) and
            any(a.atomic_number == 6 for a in self._atoms)
        )

    @property
    def is_full_halogenated(self) -> bool:
        """
        Determines whether the molecule is derived from an organic mol by replace all hydrogen
        to halogenated.
        """
        if self.is_organic:
            return False
        else:
            clone = self.copy()
            for atom in clone.atoms:
                if atom.is_halogens:
                    atom.atomic_number = 1

            return clone.is_organic

    def link_atoms(self, assign_bond_order: bool = True):
        """
        Links all atoms of the molecule together and updates the molecular graph. 
        This method can also assign bond orders between atoms if specified.

        Args:
            assign_bond_order (bool): Indicates whether to assign bond orders. 
                Defaults to True.
        """
        obc.link_atoms(self)
        # conformers = copy(self._conformers)

        self._update_graph(clear_conformers=False)

        if assign_bond_order:
            self.assign_bond_order()

        # self._conformers = conformers

    @property
    def link_matrix(self) -> np.ndarray:
        """
        Returns a numpy array representation of the link matrix for the bonds.

        This property generates a matrix where each row represents a bond between
        two atoms. The first column identifies the index of the first atom, and
        the second column identifies the index of the second atom. The matrix is
        constructed based on the bonds associated with the object.

        @return: A 2D numpy array of shape (n, 2), where each row contains the
        indices of the two atoms forming a bond. The dtype of the array is `int`.
        """
        return np.array([[b.atom1.idx, b.atom2.idx] for b in self.bonds], dtype=int)

    @property
    def metals(self) -> list['Atom']:
        """
        Represents a property that retrieves a list of metal atoms.

        This property filters through the internal list of atoms and returns
        a new list containing only those atoms that are metals. It utilizes
        the `is_metal` attribute of each atom to determine whether an atom
        is considered a metal.

        Returns
        -------
        list['Atom']
            List of atoms that are metals.
        """
        return [a for a in self._atoms if a.is_metal]

    def refresh_atom_id(self):
        """
        Updates the ID of each atom in the atom list to reflect its index position. Each atom's ID
        is replaced with its respective index in the list, starting from 0.
        """
        for i, atom in enumerate(self.atoms):
            atom.id = i

    @property
    def atom_id_dict(self):
        """
        Represents a property to generate a dictionary mapping atom IDs to atom objects
        from the atoms attribute of an object.

        The resulting dictionary uses atom IDs as keys and their corresponding atom
        objects as values. This method assumes that the attribute `atoms` contains
        iterable objects with an `id` attribute.

        @return: A dictionary mapping atom IDs to their respective atom objects.
        """
        return {a.id: a for a in self.atoms}

    @property
    def bond_id_dict(self):
        """
        Returns a dictionary mapping bond IDs to bond objects within the bonds
        attribute.

        The property generates a dictionary where each bond's unique ID serves
        as the key, and the bond object itself is the corresponding value. This
        provides a convenient way to access bonds by their ID.

        Returns:
            dict: A dictionary where keys are bond IDs and values are bond objects.
        """
        return {b.id: b for b in self.bonds}

    def _rm_atom(self, atom: "Atom"):
        """
        Removes a specified atom and its associated bonds from the structure.

        This method removes an atom, specified either by its index or as
        an Atom object, from the internal structure. All bonds connected
        to the specified atom are also removed. The atom and its bonds
        are deleted from the structure's internal atom and bond lists.

        Parameters:
            atom (Atom or int): The atom to remove, identified either by its

        Returns:
            None
        """
        if isinstance(atom, int):
            atom = self._atoms[atom]

        rm_bonds = atom.bonds
        for rmb in rm_bonds:
            self._bonds.remove(rmb)
        self._atoms.remove(atom)

    def _rm_atoms(self, atoms: Iterable[Union["Atom", int]]):
        """
        Removes specified atoms and their associated bonds from the structure.

        The method takes an iterable of atoms or their indices, identifies all related
        bonds connected to those atoms, and removes both the bonds and the specified
        atoms from the internal structure. This is particularly useful for modifying the
        structural representation by cleaning up unwanted atoms and their bonds.

        Args:
            atoms: An iterable containing Atom objects or indices of Atom objects to 
                   be removed from the structure.

        Raises:
            AttributeError: Raised if elements in the iterable are not valid Atom objects 
                            or indices.

        """
        atoms = [a if isinstance(a, Atom) else self._atoms[a] for a in atoms]

        rm_bonds = {b for a in atoms for b in a.bonds}
        for rmb in rm_bonds:
            self._bonds.remove(rmb)
        for rma in atoms:
            self._atoms.remove(rma)

    def remove_atom(self, atom: Union[int, "Atom"]) -> None:
        """
        Removes an atom from the underlying data structure and updates the graph.

        This method removes a specified atom and ensures the underlying graph
        representation is updated accordingly. It accepts either an integer
        representing the atom ID or an Atom object. The function performs
        the removal operation and updates the internal structure.

        Arguments:
            atom: Union[int, Atom]
                An integer ID of the atom to be removed or an instance of
                the Atom class representing the atom to be removed.

        Returns:
            None
        """
        self._rm_atom(atom)
        self._update_graph()

    def remove_atoms(self, atoms: Iterable["Atom"]) -> None:
        """
        Removes specified atoms from the structure and updates the associated graph.

        This method is used to remove a set of atoms from the structure. Once the 
        atoms are removed, it ensures that the graph representation of the structure 
        is updated accordingly. The input should be an iterable containing the atoms 
        to be removed.

        Args:
            atoms (Iterable[Atom]): An iterable containing Atom objects that need to 
            be removed from the structure.

        Returns:
            None
        """
        self._rm_atoms(atoms)
        self._update_graph()

    def _rm_bond(self, bond: "Bond"):
        """
        Removes a specific bond from the list of bonds associated with the object.

        This method is used to delete a bond instance from the internal collection
        of bonds.

        Parameters:
            bond (Bond): The bond instance to be removed.

        """
        self._bonds.remove(bond)

    def _rm_bonds(self, bonds: Iterable["Bond"]):
        """
        Removes the specified bonds from the internal bonds list.

        This method iterates through the given iterable of `Bond` objects
        and removes each bond from the '_bonds' list of the current instance.

        Args:
            bonds (Iterable["Bond"]): An iterable containing Bond objects
            to be removed.

        """
        for bond in bonds:
            self._bonds.remove(bond)

    def remove_bond(self, bond: "Bond") -> None:
        """
        Removes a bond from the collection and updates the associated graph structure.

        Parameters:
            bond (Bond): The bond object to be removed.

        Returns:
            None
        """
        self._bonds.remove(bond)
        self._update_graph()

    def remove_bonds(self, bonds: Iterable["Bond"]) -> None:
        """
        Removes bonds from the internal graph representation and updates the
        graph. Used to manage and maintain connectivity by removing specified
        bonds. The method ensures the internal graph is consistent after
        the removal operation.

        Args:
            bonds (Iterable[Bond]): An iterable of Bond objects to be removed.

        Returns:
            None
        """
        self._rm_bonds(bonds)
        self._update_graph()

    def remove_hydrogens(self):
        """
        Removes all hydrogen atoms from the current object.

        This function iterates through the list of atoms stored in the "_atoms"
        attribute of the object, identifies the hydrogen atoms, and removes them
        using the `remove_atoms` method. Hydrogen atoms are determined based on
        the "is_hydrogen" property of each atom object.

        Raises:
            None
        """
        self.remove_atoms([a for a in self._atoms if a.is_hydrogen])

    def remove_metals(self):
        """
        Removes metal atoms from the structure by calling the remove_atoms method
        on the list of metal atoms.

        Returns:
            None
        """
        self.remove_atoms(self.metals)

    def set_default_valence(self):
        """
        Sets the default valence for each atom in the internal list of atoms.

        This method iterates over all atoms in the `_atoms` attribute of the class
        and calls the `set_default_valence` method for each atom. It is typically
        used to initialize or reset the default valence for every atom in the
        collection.

        Raises:
            AttributeError: If the `_atoms` attribute or `set_default_valence`
                method on any atom is not properly defined.

        """
        for atom in self._atoms:
            atom.set_default_valence()

    def similarity(
            self,
            other: "Molecule",
            method: str = "spectrum",
            norm: Literal['infinite', 'min', 'l1', 'l2'] = 'l2'
    ):
        """
        Calculates the similarity between the current molecule and another molecule
        using the specified method. The default method is "spectrum".

        Arguments:
            other ("Molecule"): The molecule to compare the current molecule against.
            method (str): The method used to calculate the similarity. Defaults to
                          "spectrum".
            norm (Literal['infinite', 'min', 'l1', 'l2']): The type of norm to use for
                          spectral comparison. Defaults to 'l2'.

        Returns:
            float: The similarity score rounded to 15 decimals.

        Raises:
            NotImplementedError: If the specified method is not implemented.
        """
        if method == "spectrum":
            return round(self.graph_spectral(norm) | other.graph_spectral(norm), 15)
        else:
            raise NotImplementedError

    @staticmethod
    def _longest_path(graph: nx.Graph, _start_node: int = None, _end_node: int = None):
        """
            Finds the longest simple path in a graph.

            This method calculates the longest simple path in a given graph by 
            considering various start and end node configurations. A simple
            path is a path in which no node repeats. The implementation uses
            NetworkX to explore all simple paths between nodes, and selects 
            the longest path based on its length. If no valid path is found 
            under certain configurations, a default fallback path is returned.

            Parameters:
            graph: nx.Graph
                The graph on which the longest path calculation will be performed.
            _start_node: int or None, optional
                The starting node of the path. If None, considers all possible 
                nodes as start nodes. Default is None.
            _end_node: int or None, optional
                The ending node of the path. If None, considers all possible 
                nodes as end nodes. Default is None.

            Returns:
            list
                A list of nodes representing the longest path found in the graph. 
                If no valid path exists, an empty or fallback path is returned.
        """
        longest_path = []
        try:
            if isinstance(_start_node, int) and isinstance(_end_node, int):
                return max(nx.all_simple_paths(graph, _start_node, _end_node), key=len)

            elif isinstance(_start_node, int) and _end_node is None:
                for end_node in graph.nodes:
                    if end_node == _start_node:
                        longest_path.append([end_node])
                    else:
                        longest_path.append(max(nx.all_simple_paths(graph, _start_node, end_node), key=len))

            elif _start_node is None and isinstance(_end_node, int):
                for start_node in graph.nodes:
                    if start_node == _end_node:
                        longest_path.append([_end_node])
                    else:
                        longest_path.append(max(nx.all_simple_paths(graph, start_node, _end_node), key=len))

            else:
                for start_node, end_node in combinations(graph.nodes, 2):
                    longest_path.append(max(nx.all_simple_paths(graph, start_node, end_node), key=len))

        except ValueError as e:
            if e.args[0] == "max() arg is an empty sequence":
                path = []
                if isinstance(_start_node, int):
                    path.append(_start_node)
                if isinstance(_end_node, int):
                    path.append(_end_node)
                longest_path.append(path)

        return max(longest_path, key=len)

    def longest_path(self):
        """
        Finds the longest path in a graph.

        This method calculates the longest path within a graph 
        using internal recursive logic. It assumes that the graph 
        is represented as an adjacency list. The longest path is 
        determined and returned to the caller.

        Returns
        -------
        Any
            The longest path within the graph. The exact type is 
            dependent on the graph data structure used and the path
            representation format.
        """
        return self._longest_path(self.graph)

    def canonical_tree(self):
        """
        Generate a canonical tree representation based on the longest path of the graph.

        The function constructs a nested structure representing the tree where subtrees 
        are organized according to their branching paths. Starting from the longest path 
        on the graph, it recursively traverses and splits the graph while building the 
        hierarchical nested representation.

        Parameters:
            self (CanonicalTree): The instance of the class that includes this method.

        Returns:
            tuple: A tuple containing the longest path of the entire graph and a 
            nested dictionary representing the hierarchical tree structure.
        """
        def _dfs(
                _graph: nx.Graph,
                _longest_path: tuple[int],
                _nested: dict
        ):
            branch_start = [(n, p) for p in _longest_path for n in _graph.neighbors(p) if n not in _longest_path]
            if branch_start:
                clone = _graph.copy()
                clone.remove_edges_from(branch_start)
            else:
                return

            for connected_nodes in nx.connected_components(clone):

                # Exclude the trunk path (parent longest path)
                if not connected_nodes - set(_longest_path):
                    continue

                sub_graph = clone.subgraph(connected_nodes)
                start_node, parent_node = next((n, p) for n, p in branch_start if n in connected_nodes)

                longest_path = self._longest_path(sub_graph, start_node)

                list_nested = _nested.setdefault(parent_node, [])
                _sub_nested = {}
                list_nested.append((longest_path, _sub_nested))

                _dfs(sub_graph, longest_path, _sub_nested)

        graph = self.graph
        nested_tree = (self._longest_path(graph), {})
        _dfs(graph, nested_tree[0], nested_tree[1])

        return nested_tree

    def super_acyclic_graph(self):
        raise NotImplementedError

    @property
    def InChi(self) -> str:
        """
        Provides access to the InChI (IUPAC International Chemical Identifier) string representation
        of the molecule. The InChI format is a standardized textual identifier for chemical substances,
        designed to provide a standard and human-readable representation of molecules.

        Returns:
            str: The InChI representation of the molecule. The method converts the internal
            object representation to an Open Babel molecule, and then outputs its InChI string.
        """
        return pb.Molecule(self.to_obmol()).write('inchi')

    def shortest_path_index(self, atom1, atom2) -> list[int]:
        """
        Calculates the shortest path index between two atoms in a molecular graph.

        This method determines the shortest path in terms of index connections 
        between two specified atoms within the molecular graph. If there is no 
        path between the two atoms, it returns an empty list.

        Parameters:
        atom1: Any
            The first atom for which the shortest path index is to be calculated.
        atom2: Any
            The second atom for which the shortest path index is to be calculated.

        Returns:
        list
            A list containing the indices of the shortest path between the two atoms.
            Returns an empty list if no path exists.
        """
        try:
            return nx.shortest_path(self.graph, self._atoms.index(atom1), self._atoms.index(atom2))
        except nx.NetworkXNoPath:
            return []

    def shortest_path(self, atom1, atom2) -> list['Atom']:
        """
        Find the shortest path between two atoms in the molecular structure.

        This method calculates the shortest path between two specified atoms in the 
        molecular structure and returns a list of the atoms that constitute this path.

        Parameters:
        atom1 : int
            Index of the first atom in the molecular structure.
        atom2 : int
            Index of the second atom in the molecular structure.

        Returns:
        list
            A list of atoms representing the shortest path between the specified 
            atoms in the molecular structure.
        """
        return [self._atoms[i] for i in self.shortest_path_index(atom1, atom2)]

    def shortest_paths_indices(self, atom1, atom2):
        """
        Finds the shortest paths indices between two atoms in a graph.

        This method calculates the shortest path between two specified atoms in a molecular
        graph, represented internally by a NetworkX graph object. It identifies the path
        indices by locating the position of the given atoms in the list of atoms and computing
        the shortest route based on the internal graph structure.

        Parameters:
        atom1: The first atom for which the shortest path is to be calculated.
        atom2: The second atom for which the shortest path is to be calculated.

        Returns:
        list:
            A list of integers representing the indices of the shortest path between
            the given atoms in the graph, as determined by NetworkX.
        """
        return nx.shortest_path(self.graph, self._atoms.index(atom1), self._atoms.index(atom2))

    @property
    def smiles(self) -> str:
        """ Return smiles string. """
        # return pb.readstring('smi', pb.Molecule(mol2obmol(self)[0]).write().strip()).write('can').strip()
        # return pb.Molecule(self.to_obmol()).write('can')
        return pb.readstring('mol2', pb.Molecule(self.to_obmol()).write('mol2')).write('can').split()[0]
        # return pb.readstring('smi', pb.Molecule(mol2obmol(self)[0]).write().strip()).write('can').strip()
        # return pb.Molecule(mol2obmol(self)[0]).write().strip()

    @property
    def rings(self) -> list["Ring"]:
        """
        Determines and returns a list of rings present in the graph represented by the object.

        The method calculates the ring structures in the graph, if they are not already computed,
        by identifying cycles using a cycle basis algorithm applied to the internal graph representation. 
        The rings are then constructed as instances of the Ring class, based on the atoms corresponding 
        to the graph cycles. The result is cached for reuse during subsequent accesses.

        @return: List of Ring objects computed from the cycles found in the graph.

        Attributes:
            _rings (Optional[list["Ring"]]): Cached list of Ring objects. If not previously computed, 
                this attribute is populated by the method.
            _atoms (list[Any]): Internal representation of the atoms in the molecule, used to instantiate 
                the Ring objects.
            graph (nx.Graph): Graph representation of the molecular structure to identify cycle bases.

        Returns:
            list["Ring"]: A list of Ring objects, each representing a detected cyclic structure 
            in the graph.
        """
        if not self._rings:
            self._rings = [Ring(*(self._atoms[i] for i in cycle)) for cycle in nx.cycle_basis(self.graph)]

        return copy(self._rings)

    @property
    def aromatic_joint_rings(self) -> list["JointRing"]:
        """
        Returns a list of joint aromatic rings formed by the combination of individual
        rings that share common elements. Joint aromatic rings are constructed by
        iteratively merging overlapping rings.

        Returns
        -------
        list["JointRing"]
            A list of JointRing objects representing joint aromatic rings.
        """
        rings = self.rings
        joint_rings = []
        while rings:
            ring = rings.pop()
            joint_ring = ring.joint_ring()

            if joint_ring:
                to_remove = []
                for r in rings:
                    if r in joint_ring:
                        to_remove.append(r)

                for r in to_remove:
                    rings.remove(r)

                joint_rings.append(joint_ring)

        return joint_rings

    @property
    def rings_small(self) -> list["Ring"]:
        """
        Returns a list of small rings.

        Small rings are defined as rings with a length of 8 or less. This property
        filters through the rings associated with the object and includes only
        those that meet the condition.

        @rtype: list["Ring"]
        @return: A list of small rings, where each ring has a length of 8 or less.
        """
        return [r for r in self.rings if len(r) <= 8]

    @property
    def ligand_rings(self) -> list["Ring"]:
        """
        Retrieve the list of ligand rings associated with the object.

        This property accesses the list of rings, specifically those that
        are part of metal-ligand bonds. It temporarily hides metal-ligand
        bonds to facilitate the computation of rings, and then restores them
        before returning the result.

        @property
            Returns:
                list[Ring]: A list of Ring objects representing ligand
                rings associated with the object.
        """
        self.hide_metal_ligand_bonds()
        rings = self.rings
        self.recover_hided_metal_ligand_bonds()
        return rings

    def to_obmol(self) -> ob.OBMol:
        """
        Converts the current molecule representation to an Open Babel OBMol object.

        This method checks if the internal OBMol object representation exists. If it does, the
        stored OBMol is returned. If not, the molecule is converted to an OBMol representation
        using the appropriate conversion function and then stored for future use.

        Returns:
            ob.OBMol: The Open Babel molecule object representation of the current molecule.
        """
        if not self._obmol:
            self._obmol, self._row2idx = obc.mol2obmol(self)
        return self._obmol

    def to_rdmol(self):
        """
        Converts the current molecule representation to an RDKit Mol object.

        This method facilitates the transformation of the molecule's internal 
        representation into an RDKit molecule object, allowing for compatibility 
        with RDKit's cheminformatics functionalities. RDKit's Mol object provides 
        extensive features for visualization, property computation, and molecular 
        analysis.

        Returns:
            rdkit.Chem.rdchem.Mol: The RDKit Mol object representing the current 
            molecule.
        """
        return to_rdmol(self)

    def to_pybel_mol(self) -> pb.Molecule:
        """
        Represents a method to convert a molecular representation into a Pybel Molecule object.

        This method is intended to transform the molecular data managed by the class into
        a Pybel-compatible Molecule object using the conversion utility provided. It relies
        on the `mol2obmol` method to perform the required conversion and returns the first
        resulting object.

        Raises:
            No raised exceptions are documented for this method.

        Returns:
            pb.Molecule: A Pybel Molecule object generated from the molecular representation.
        """
        return pb.Molecule(obc.mol2obmol(self)[0])

    def translation(self, vector: types.ArrayLike):
        """
        Update the coordinates of the object by applying a translation vector. The translation
        vector is added to the current coordinates to produce the updated values.

        Args:
            vector (types.ArrayLike): A 1-dimensional array-like object representing
            the translation vector. Must have exactly 3 elements.

        Returns:
            None
        """
        vector = np.array(vector).flatten()
        assert len(vector) == 3

        coordinates = self.coordinates
        self.coordinates = coordinates + vector

    def update_atoms_attrs_from_id_dict(self, id_dict: dict[int, dict]):
        """
        Updates the attributes of atoms in the atom_id_dict based on the provided id_dict.

        This function iterates through the given id_dict, matches the atom IDs in the
        atom_id_dict, and updates their attributes using the key-value pairs provided
        in the id_dict. Each atom's attributes are updated with the specified values.

        Parameters:
            id_dict (dict[int, dict]): A dictionary where keys are atom IDs (int) and
            values are dictionaries containing attribute names as keys and their values.

        Returns:
            None
        """
        id_atoms = self.atom_id_dict
        for i, attr in id_dict.items():
            id_atoms[i].setattr(**attr)

    def update_bonds_attrs_from_id_dict(self, id_dict: dict[int, dict]):
        """
        Updates attributes of bonds in the bond_id_dict using values from the given dictionary.

        This function iterates over the provided id_dict and for each key-value pair, 
        updates the corresponding bond instance's attributes in the bond_id_dict. 
        The attributes are updated using the setattr method, passing the dictionary 
        values as keyword arguments.

        Parameters:
            id_dict (dict[int, dict]): A dictionary where each key corresponds to a bond ID, 
            and the value is another dictionary specifying attributes to be updated for the 
            corresponding bond.

        Returns:
            None
        """
        id_bonds = self.bond_id_dict
        for i, attr in id_dict.items():
            id_bonds[i].setattr(**attr)

    def update_angles(self):
        """
        Updates the list of angles based on the current state of atoms and their neighbors.

        In this method, all possible angle configurations are recalculated and updated 
        in the '_angles' attribute. The computation is performed by iterating over each 
        atom in the 'atoms' attribute, then generating all unique combinations of neighboring 
        atoms for forming an angle. Each angle is represented as an instance of the Angle class.

        """
        self._angles = [Angle(n1, a, n2) for a in self.atoms for n1, n2 in combinations(a.neighbours, 2)]

    def update_torsions(self):
        """
        Updates torsion angles by retrieving appropriate data.

        The method refreshes the torsion angles of an object by calling an
        internal function `_retrieve_torsions`. The updated values are then
        stored in the internal variable `_torsions`.

        Raises:
            RuntimeError: If the torsions retrieval fails.
        """
        self._torsions = self._retrieve_torsions()

    @property
    def weight(self):
        """
        Returns the total weight of the structure by summing the mass of all atoms.

        This property calculates the weight based on the individual masses of 
        all atoms contained within the `_atoms` attribute.
        """
        return sum(a.mass for a in self._atoms)

    def write(
            self,
            filename=None,
            fmt: Optional[str] = None,
            overwrite=False,
            write_single: bool = False,
            ob_opt: dict = None,
            **kwargs
    ):
        """
        Write the molecular data to a file with the specified format and options.

        Parameters:
        filename : str, optional
            Path to the output file. If not provided, a default name may be used.
        fmt : str, optional
            Desired format of the output file. If not provided, a default format may
            be assumed based on the file extension.
        overwrite : bool, default is False
            Determines whether to overwrite the existing file with the same name. If
            False and the file exists, an error may occur.
        write_single : bool, default is False
            If True, specifies that only a single entity (e.g., molecule) should be 
            written to the file.
        ob_opt : dict, optional
            A dictionary of additional options for the write operation, which may be 
            passed to the underlying Open Babel functionalities.
        **kwargs
            Additional keyword arguments passed to the internal file writer.

        Returns:
        bool
            True if the file was successfully written, False otherwise.
        """
        # write_by_pybel(self, fmt, str(filename), overwrite, opt)
        writer = _io.MolWriter(filename, fmt, overwrite=overwrite, **kwargs)
        return writer.write(self, write_single=write_single)


class MolBlock:
    """
    Represents an abstraction of a molecular block.

    The MolBlock class provides an abstraction to represent and manage molecular 
    blocks, typically used in chemical or molecular modeling. It provides methods 
    and properties to interact with molecular attributes, rings, and other 
    chemical properties. The class supports controlled attribute setting, 
    retrieval, and enumeration while limiting certain operations like copying.

    Attributes:
        _attrs_dict (dict): Internal mapping of molecular attributes to their 
            respective processing functions.
        _attrs_setter (dict): Internal mapping of molecular attributes to their 
            respective setter functions.
        _default_attrs (dict): Default molecular attribute values.
        _attrs_enumerator (tuple): A tuple enumerating all attribute names in 
            _attrs_dict.

    Raises:
        PermissionError: When attempting to copy an instance of MolBlock.

    Methods:
        __repr__: Returns a string representation of the MolBlock instance.
        __copy__: Restricts copying of the MolBlock instance; raises PermissionError.
        __dir__: Modifies the directory of attributes to include the molecular 
            attributes in _attrs_enumerator.
        __getattr__: Retrieves a molecular attribute's value based on its processor 
            function or delegates to the parent class for non-existent attributes.
        __setattr__: Sets an attribute using a custom setter function; defaults to 
            a custom handler for key-value assignment.
        _default_attr_setter: Default attribute setter function for numerical values.
        attrs_enumerator: Property to get the tuple of molecular attribute names.
        attr_dict: Property to get a dictionary mapping attribute names to values.
        in_ring: Property to check if the molecular block is part of any ring.
        in_organic: Property to check if the molecular block belongs to an 
            organic molecule.
        rings: Property to get a list of rings involving the molecular block.
        setattr: Method for batch attribute setting, optionally including 
            default values.
    """
    _attrs_dict = {}
    _attrs_setter = {}
    _default_attrs = {}
    _attrs_enumerator = tuple(_attrs_dict.keys())

    def __repr__(self):
        return f"{self.__class__.__name__}({self.label})"

    def __copy__(self):
        raise PermissionError(f"The {self.__class__.__name__} not allow to copy")

    def __dir__(self):
        return list(self._attrs_enumerator) + list(super(MolBlock, self).__dir__())

    def __getattr__(self, item):
        try:
            attr_idx = self._attrs_enumerator.index(item)
            return self._attrs_dict[item](self.attrs[attr_idx])
        except ValueError:
            # raise AttributeError(f"{item} is not an attribute of the {self.__class__.__name__}")
            return super().__getattribute__(item)

    def __setattr__(self, key, value):
        try:
            setter = self._attrs_setter.get(key, self._default_attr_setter)
            setter(self, key, value)
            # attr_idx = self._attrs_enumerator.index(key)
            # self.attrs[attr_idx] = float(value)
        except ValueError:
            super().__setattr__(key, value)
        except Exception as e:
            print(key, value)
            raise e

    @staticmethod
    def _default_attr_setter(self, key, value):
        attr_idx = self._attrs_enumerator.index(key)
        self.attrs[attr_idx] = float(value)

    @property
    def attrs_enumerator(self) -> tuple:
        return self._attrs_enumerator

    @property
    def attr_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.attrs_enumerator}

    @property
    def in_ring(self):
        """
        Determine if the atom or group is part of one or more rings in the molecule.

        Check each ring in the molecule associated with the object and confirm 
        if the object exists within any of those rings.

        @return: Boolean indicating whether the object is part of any ring

        """
        return any(self in r for r in self.mol.rings)

    @property
    def in_organic(self) -> bool:
        return self.mol.is_organic

    @property
    def rings(self):
        """
        Represents a property that computes and returns a list of rings containing the instance.

        This property iterates over the rings of a molecule and checks if the object 
        invoking the property is a member of each ring. If the object is part of any 
        rings, they are included in the returned list.

        @return: A list of rings containing the invoking instance
        """
        return [r for r in self.mol.rings if self in r]

    def setattr(self, *, add_defaults=False, **kwargs):
        """
        Sets attributes of the current object based on the passed keyword arguments. 
        An optional flag allows to start with default attributes before applying 
        the provided updates.

        Parameters
        ----------
        add_defaults : bool, optional
            If True, initializes attributes with the default set before applying 
            updates from kwargs. Default is False.
        **kwargs : Any
            Arbitrary keyword arguments representing attribute names and their 
            associated values to be applied as updates.

        TODO: Warning!
        TODO: This methods is recommended to be invoked after, the molecular graph has
        TODO: been built. If the method is invoked in the obj-create stage. Some unexpected
        TODO: error might raise.
        """
        _attrs = copy(self._default_attrs) if add_defaults else {}
        _attrs.update(kwargs)
        for name, value in _attrs.items():
            setattr(self, name, value)


#######################################################################
#######################################################################
# Define attributes setters
def _atomic_number_setter(self: "Atom", key, atomic_number):
    """
    Sets the atomic number for the atom object and updates its electronic configuration.

    This method enforces the atomic property by accepting a key specific 
    to "atomic_number". Upon setting the atomic number, the elements' 
    attributes, including its electronic configuration, are recalculated 
    and updated accordingly based on the chemical properties.

    Parameters:
        key: str
            Key name, expected to be "atomic_number".
        atomic_number: int
            The atomic number of the element to set and use for recalculations.
    """
    assert key == "atomic_number"
    self.attrs[0] = atomic_number
    self.attrs[Atom._ELECTRON_N_CONFIG: Atom._ELECTRON_G_CONFIG+1] = Atom.elements.electron_configs[atomic_number]
    # Adding in 2025/7/19
    self.attrs[Atom._attrs_enumerator.index("valence")] = self.get_valence()
    self.attrs[Atom._attrs_enumerator.index("implicit_hydrogens")] = self._calc_implicit_hydrogens()

# ---------------------------------------------------------------------

class Atom(MolBlock):
    """
    Represents a single atom with detailed attributes and properties.

    The class provides atomic data and models an atomic entity, including 
    attributes such as position coordinates (x, y, z), atomic number, 
    valence, charge details, and its chemical symbol. It also encapsulates 
    essential chemical information (e.g., atomic orbitals, periodic table 
    data, and default valencies) useful for computational chemistry and 
    molecular modeling tasks.

    Attributes
    ----------
    atomic_number : int
        Atomic number of the element.
    formal_charge : int
        Formal charge associated with the atom.
    partial_charge : float
        Partial charge of the atom derived from calculations or estimations.
    x : float
        X-coordinate of the atom in 3D Cartesian space.
    y : float
        Y-coordinate of the atom in 3D Cartesian space.
    z : float
        Z-coordinate of the atom in 3D Cartesian space.
    valence : int
        Valence electrons or bonding capacity of the atom.
    id : int
        Unique identifier for the atom instance.
    symbol : str
        Elemental symbol representing the atom (e.g., "H" for Hydrogen).
    idx : int
        Index of the atom in molecular structures or calculations.

    """
    # Cython define
    atomic_number: int
    formal_charge: int
    partial_charge: float
    x: float
    y: float
    z: float
    valence: int
    id: int
    symbol: str
    idx: int
    is_aromatic: bool
    implicit_hydrogens: int

    _coord_getter = operator.attrgetter('x', 'y', 'z')

    _attrs_dict = {
        # Name: datatype
        'atomic_number': int,
        'n': int,
        's': int,
        'p': int,
        'd': int,
        'f': int,
        'g': int,
        'formal_charge': int,
        'partial_charge': float,
        'is_aromatic': bool,
        'x': float,
        'y': float,
        'z': float,
        'valence': int,
        'implicit_hydrogens': int,
        'id': int,
        'x_constraint': bool,
        'y_constraint': bool,
        'z_constraint': bool,
        # 'explicit_hydrogens'
    }

    _attrs_setter = {
        'atomic_number': _atomic_number_setter
    }

    _default_attrs = {
        "atomic_number": 0,
        "is_aromatic": False,
        "formal_charge": 0,
        "partial_charge": 0.,
        "coordinates": (0., 0., 0.),
        "x": 0.,
        "y": 0.,
        "z": 0.,
        "valence": 0,
        "implicit_hydrogens": 0,
    }
    _attrs_enumerator = tuple(_attrs_dict.keys())

    # Index Check
    _ELECTRON_N_CONFIG = _attrs_enumerator.index('n')
    _ELECTRON_G_CONFIG = _attrs_enumerator.index('g')

    from .elements import elements

    def __init__(
            self,
            mol: Molecule = None,
            *,
            attrs_array: np.ndarray = None,
            update_electron_config: bool = True,
            **kwargs
    ):
        """
        Initializes an instance of the class, allowing the creation of an object associated with a given
        molecular structure and attributes. The method provides options for defining default behavior
        through keyword arguments or specifying attributes explicitly with an array.

        Attributes:
        mol (Molecule): Represents the molecular structure associated with the object.
        attrs_array (np.ndarray): A numpy array of attributes with a fixed size equivalent to
                                  the enumerated attributes of the class.

        Methods:
        __init__: Constructor to initialize the molecule and its attributes.  

        Parameters:
        mol: Optional; Defaults to None. If provided, specifies the molecular structure
             to associate the object with; otherwise, the object is associated with a default Molecule instance.

        attrs_array: The given numpy array specifying custom attribute values.  
                    Must be one-dimensional. 

         behaviors/ Rules ;proper Error Inflect 
               ** detail ensure behaviors s """
        self.mol = mol or Molecule()
        self._bonds = []
        self._neighbours = []
        getattr(self.mol, '_atoms').append(self)

        if isinstance(attrs_array, np.ndarray):
            assert attrs_array.ndim == 1
            if len(self._attrs_enumerator) == len(attrs_array):
                self.attrs = attrs_array
            else:
                raise ValueError(
                    f'Given attrs_array should be length {len(self._attrs_enumerator)},\n'
                    f'The attrs arrange like the following: \n'
                    f'\t {self._attrs_enumerator}'
                )

        else:
            # When create a new atoms, bypassing the attr_setter interface to avoiding
            # unexpected invoke. For example:
            # - Calculate the bond order and implicit hydrogen before the bonds and molecular
            #   graph is built.
            if 'symbol' in kwargs:
                kwargs['atomic_number'] = ob.GetAtomicNum(kwargs.pop('symbol'))
            if 'coordinates' in kwargs:
                kwargs['x'], kwargs['y'], kwargs['z'] = kwargs.pop('coordinates')

            self.attrs = np.array([kwargs.pop(a, 0.) for a in self._attrs_enumerator])

            # TODO: Warning! It's not recommended to create a new atom obj.
            self.setattr(**kwargs)  # Warning: This operation might cause some Error or Bug.

        if update_electron_config:
            self.electron_configuration = self.elements.electron_configs[self.atomic_number]

    @classmethod
    def _get_atom_attr_dict(cls, atomic_number: int) -> dict:
        """
        Provides an internal utility method to retrieve a dictionary
        containing various attributes of an atom based on its atomic number.
        This dynamic method initializes an Open Babel OBAtom object, sets the 
        atomic number, and fetches the relevant chemical properties associated
        with the atom like atomic number, formal charge, partial charge, valence, 
        and implicit hydrogens.

        Parameters:
            atomic_number (int): The atomic number of the element being queried.

        Returns:
            dict: A dictionary containing properties of the atom, such as:
                - atomic_number 
                - formal_charge 
                - partial_charge 
                - valence 
                - implicit_hydrogens
        """
        oba = ob.OBAtom()
        oba.SetAtomicNum(atomic_number)
        return dict(
            atomic_number=oba.GetAtomicNum(),
            formal_charge=oba.GetFormalCharge(),
            partial_charge=oba.GetPartialCharge(),
            valence=oba.GetTotalValence(),
            implicit_hydrogens=oba.GetImplicitHCount()
        )

    def _add_atom(
            self,
            atom: Union[str, int, "Atom"] = 1,
            bond_order: int = 1,
            atom_attrs: dict = None,
            bond_attrs: dict = None
    ):
        """
        Adds an atom to the molecular structure with the specified properties and bond order.

        The method allows adding atoms by different formats (e.g., string, integer, or Atom object)
        and automatically resolves the appropriate atomic properties. Bond-related attributes and
        additional atomic attributes can also be specified. The atom is added with its specified
        properties and linked with the molecule structure according to the given bond order.

        Args:
            atom: Union of string, integer, or Atom instance representing the atom to be added.
                  Defaults to 1.
            bond_order: Integer specifying the bond order of the bond between the given atom and 
                        the molecule. Defaults to 1.
            atom_attrs: Dictionary holding additional attributes for the atom (optional).
            bond_attrs: Dictionary holding additional attributes for the bond (optional).

        Returns:
            Atom: The newly created atom instance within the molecule structure.

        Raises:
            This method does not include explicit error raising information. Ensure valid input
            types and values.
        """
        if isinstance(atom, str):
            atom = ob.GetAtomicNum(atom)

        if isinstance(atom, int):
            atom_attrs_ = self._get_atom_attr_dict(atom)
        else:
            atom_attrs_ = atom.attr_dict

        atom_attrs_.update(atom_attrs or {})
        atom = self.mol._create_atom(**atom_attrs_)
        getattr(self.mol, '_add_bond')(self, atom, bond_order, **(bond_attrs or {}))

        return atom

    def add_atom(
            self,
            atom: Union[str, int, "Atom"] = 1,
            bond_order: int = 1,
            atom_attrs: dict = None,
            bond_attrs: dict = None
    ):
        """
        Add an atom to a molecular graph with optional attributes and bond details.

        This function facilitates the addition of a single atom to the molecular
        graph. The user can specify the atom type, the bond order to the existing
        structure, and additional optional attributes for both the atom and the
        bond. The resulting atom is internally processed and added to the graph,
        and the molecular graph representation is updated afterward.

        Parameters:
            atom: Union[str, int, "Atom"]
                The atom to add. Can be a string (symbol), an integer (atomic number),
                or an Atom object. Defaults to integer 1.
            bond_order: int
                The bond order for connecting the new atom to the existing structure.
                Defaults to 1.
            atom_attrs: dict
                Additional attributes to assign to the newly added atom. Defaults
                to None.
            bond_attrs: dict
                Additional attributes to assign to the bond connecting the new atom.
                Defaults to None.

        Returns:
            atom
                The newly created and added atom object linked to the molecular graph.
        """
        atom = self._add_atom(atom, bond_order, atom_attrs, bond_attrs)
        getattr(self.mol, '_update_graph')()
        return atom

    @staticmethod
    def random_point_on_sphere(radius: float = 1.):
        """
        Generates a random point on the surface of a sphere with a given radius.

        The function uses uniformly distributed random numbers to compute a point
        on the sphere by generating the polar angle (theta) and azimuthal angle
        (phi) in spherical coordinates, and converting them into Cartesian
        coordinates (x, y, z). The generated point lies on the sphere's surface.

        Args:
            radius (float): The radius of the sphere. Default is 1.0.

        Returns:
            tuple[float, float, float]: Cartesian coordinates (x, y, z) of the
            point on the sphere's surface.
        """
        # 随机生成极角 theta 和方位角 phi
        theta = np.arccos(2 * np.random.rand() - 1)  # 0 到 pi
        phi = 2 * np.pi * np.random.rand()  # 0 到 2pi

        # convert to Cartesian coordination
        x = np.sin(theta) * np.cos(phi) * radius
        y = np.sin(theta) * np.sin(phi) * radius
        z = np.cos(theta) * radius

        return x, y, z

    @property
    def polar_hydrogen_site(self) -> bool:
        """
        Checks if the atomic site is a polar hydrogen site.

        A polar hydrogen site is identified by its atomic number and is determined to
        be polar if the atomic number equals 8 (oxygen) or, in the case of nitrogen 
        (atomic number 7), an additional check ensures it is aromatic.

        Returns:
            bool: True if the atomic site is a polar hydrogen site, False otherwise.
        """
        return self.atomic_number == 8 or (self.atomic_number == 7 and self.is_aromatic)

    def _add_hydrogens(self, num: int = None, rm_polar_hs: bool = True) -> (int, list["Atom"]):
        """
        Adds or removes hydrogen atoms to achieve the correct count based on the molecule's implicit hydrogens.

        This method adjusts the number of hydrogen atoms attached to a given site. It accounts
        for polar hydrogen sites, metal-ligand bonds, and optionally removes polar hydrogens if needed.
        The number of hydrogens to be added or removed is calculated based on the implicit hydrogen
        count and existing hydrogens present. If adjustments are necessary, the function either
        adds new hydrogen atoms or removes an excess based on the specified criteria. Returns
        information about the adjustment performed.

        Parameters:
            num: int or None
                The number of hydrogens to add or remove. If None, the value is determined 
                based on the implicit hydrogen count and the current state of the molecule.
            rm_polar_hs: bool
                A flag to indicate whether polar hydrogens should be removed when reducing
                the hydrogen count in case of excess.

        Returns:
            tuple[int, list["Atom"]]
                A tuple where the first element indicates the type of adjustment performed:
                - 1 indicates addition of hydrogens,
                - -1 indicates removal,
                - 0 indicates no adjustment.
                The second element is a list of hydrogen atoms added or removed.

        """
        neighbours = self.neighbours
        hydrogens = [a for a in neighbours if a.atomic_number == 1]

        if num is None:
            num = self.implicit_hydrogens - len(hydrogens)
            if self.polar_hydrogen_site:
                num -= len([a for a in neighbours if a.is_metal])  # minus metal-ligand bonds

        if num > 0:
            return 1, [
                self._add_atom(atom_attrs={
                    'coordinates': np.array(self.coordinates) + self.random_point_on_sphere(1.05)
                }) for _ in range(num)
            ]

        elif num < 0 and self.polar_hydrogen_site and hydrogens and rm_polar_hs:
            self.mol._rm_atoms(hydrogens[:abs(num)])
            return -1, hydrogens[:abs(num)]

        else:
            return 0, []

    def add_hydrogen(self, num: int = None) -> list["Atom"]:
        """
        Adds hydrogen atoms to the molecule.

        This method adds hydrogen atoms to a molecule. The number of hydrogens to add 
        can be specified, or the method may add them based on default behavior. After 
        adding the hydrogens, the molecule graph is updated to reflect the changes.

        Args:
            num (int, optional): The number of hydrogen atoms to add. If None, the 
            method will determine the number of hydrogens automatically.

        Returns:
            list["Atom"]: A list of the hydrogen atoms added to the molecule.
        """
        add_or_rm, hydrogens = self._add_hydrogens(num)
        getattr(self.mol, '_update_graph')()
        return hydrogens

    @property
    def bonds(self) -> list["Bond"]:
        """
        Returns the bonds associated with the current atomic structure. It utilizes the
        molecular graph's edge viewer to retrieve bond details for the specified atom based
        on its index. If the atom is isolated, an empty list is returned.

        Returns:
            list[Bond]: A list of Bond objects associated with the atom's index. If no
            bonds are associated or the atom is isolated, an empty list is returned.

        Raises:
            NetworkXError: If accessing the molecular graph edges fails for the
            specified atom index (handled internally).
        """
        # # return [self.mol.bonds[i] for i in self.bonds_idx]
        # edge_viewer = self.mol.graph.edges
        #
        # try:
        #     return [edge_viewer[u, v]['bond'] for u, v in edge_viewer(self.idx)]
        # except nx.NetworkXError:
        #     return []  # if the atom is an isolate atom
        return copy(self._bonds)

    def _calc_implicit_hydrogens(self) -> int:
        # TODO: Implement by C++
        if self.is_metal:
            return 0
        elif self.atomic_number == 1:
            if self.neighbours[0].atomic_number == 1:
                return 1
            else:
                return 0

        elif self.is_aromatic:
            num = len([a for a in self.neighbours if a.atomic_number != 1 and (not a.is_metal)])

            if self.atomic_number in [6, 14]:
                if num == 3:
                    return 0
                else:
                    return 1
            elif self.atomic_number in [7, 15, 33]:
                if num == 3 or self.sum_heavy_cov_orders > 2:
                    return 0
                else:
                    return 1
            elif self.atomic_number in [8, 16, 34]:
                return 0
            elif self.atomic_number == 5:
                return 1
            elif self.atomic_number == 32:  # Ge
                return 0
            else:
                raise AttributeError(f"Get an incorrect atom!!， {self.symbol}")
        else:
            return max(self.valence - self.sum_heavy_cov_orders, 0)


    def calc_implicit_hydrogens(self):
        """
        Calculates the implicit hydrogen count for an atom based on its atomic properties.

        This function determines the number of implicit hydrogens an atom should possess 
        depending on its atomic number, aromaticity, and covalent bonding context. The 
        calculation is influenced by whether the atom is metallic, aromatic, part of specific 
        groups (e.g., carbon, nitrogen), or based on its valence and bonding characteristics.

        Raises
        ------
        AttributeError
            If the atom's properties do not align with known rules (e.g., incorrect type or 
            unrecognized atomic number).

        See Also
        --------
        calc_implicit_hydrogens : This function should ideally be implemented in C++ for 
            performance improvements.
        """
        self.implicit_hydrogens = self._calc_implicit_hydrogens()

    @property
    def constraint(self) -> bool:
        """
        Returns a boolean indicating whether all constraints (x_constraint, 
        y_constraint, z_constraint) are satisfied.

        The method evaluates multiple constraints and determines if they are 
        all True. This is commonly used to validate that all conditions 
        required by the system are met.

        @return: Boolean value indicating whether all constraints are satisfied
        """
        return all([self.x_constraint, self.y_constraint, self.z_constraint])

    @constraint.setter
    def constraint(self, value: bool):
        """
        Setter method for the `constraint` property that sets the same value 
        for `x_constraint`, `y_constraint`, and `z_constraint` attributes of the instance.

        Args:
            value (bool): The value to be set for the constraints.
        """
        self.x_constraint = self.y_constraint = self.z_constraint = value

    @property
    def coordinates(self):
        """
        Coordinates of the atom.

        This property allows access to the 3D coordinates of an atom. It retrieves
        the coordinates dynamically via an internal getter method.

        Returns
        -------
        list of float
            A list representing the x, y, and z coordinates of the atom.
        """
        return Atom._coord_getter(self)

    @coordinates.setter
    def coordinates(self, value: types.ArrayLike):
        """
        Sets the coordinates of the object.

        This setter method allows assigning new values to the 
        x, y, and z attributes of the object. The provided value 
        must be an array-like object containing exactly three 
        elements, which correspond to the coordinates x, y, and 
        z in the given order.

        Arguments:
            value (types.ArrayLike): An array-like object that 
            contains exactly three elements representing the x, 
            y, and z coordinates. The input must conform to the 
            ArrayLike type.

        """
        self.x, self.y, self.z = value

    @property
    def exact_mass(self):
        """
        Returns the exact mass of the element based on its atomic number.

        The exact mass refers to the precise mass of the most abundant 
        isotope of the element, calculated directly. The value is derived 
        from the atomic number associated with the element.

        Returns
        -------
        float
            The exact mass of the element based on its atomic number.
        """
        return ob.GetExactMass(self.atomic_number)

    @property
    def electronegativity(self) -> Optional[float]:
        """
        Return the electronegativity of the element.

        The electronegativity of an element is a measure of its tendency to attract
        and bond with electrons. This property is dependent on the element's atomic 
        number and is based on predefined values from the elements data.

        @return: The electronegativity value of the element or None if the value
        is not defined.

        @rtype: Optional[float]
        """
        return self.elements.electronegativity[self.atomic_number]

    @property
    def explicit_hydrogens(self) -> int:
        """
        Calculate the number of explicit hydrogen atoms attached.

        Returns the count of hydrogen atoms explicitly bonded as neighbors 
        to the current atom in the structure. Only neighbors with an atomic 
        number of 1 are considered hydrogen atoms.

        Returns:
            int: The count of explicit hydrogen atoms.
        """
        return len([a for a in self.neighbours if a.atomic_number == 1])

    @property
    def covalent_radius(self) -> float:
        """
        This property retrieves the covalent radius of an atom based on its 
        atomic number, utilizing pre-stored data for elements' covalent radii.

        Returns
        -------
        float
            The covalent radius of the atom in appropriate units.
        """
        return self.elements.covalent_radii[self.atomic_number]

    @property
    def density(self) -> float:
        """
        Calculate and return the density of an element based on its atomic number 
        and the associated density values stored in the elements object.

        Attrs:
            density (float): The density of the element retrieved using the
            element's atomic number. This property fetches the corresponding
            density value from the elements object's density mapping.
        """
        return self.elements.density[self.atomic_number]

    @property
    def hyb(self) -> int:
        """
        Determines the hybridization state of an atom based on its atomic number, bond orders, and 
        other associated properties. This computation is specific to elements such as Carbon, Silicon,
        Nitrogen, Phosphorus, and Hydrogen, accounting for aromaticity, double bonds, triple bonds, or 
        other bonding characteristics.

        @return: Returns the hybridization state of the atom as an integer. 
                 0 indicates sp3 hybridization, 1 indicates sp hybridization, 
                 2 indicates sp2 hybridization, and 3 indicates an undefined or other state.

        @rtype: int
        """
        # TODO: Debug
        if self.atomic_number == 1:
            return 0
        elif self.atomic_number in {6, 14}:  # C Si
            if self.is_aromatic or any(b.bond_order == 2 for b in self.bonds):
                return 2
            elif any(b.bond_order == 3 for b in self.bonds):
                return 1
            else:
                return 3
        elif self.atomic_number == 7:  # N
            if self.sum_heavy_cov_orders == 3:
                return 2
            else:
                return 3
        elif self.atomic_number == 15:  # P
            if self.sum_heavy_cov_orders == 3:
                return 2
            else:
                return 3
        else:
            return 0


    @property
    def idx(self) -> int:
        """
        Property that calculates and returns the index of the current atom 
        within the molecule's atoms list. It leverages the `atoms.index` 
        method from the `mol` object to determine the atom's position.

        @return int: Index of the current atom in the molecule's atoms list
        """
        return self.mol.atoms.index(self)

    @property
    def is_error_electron_configure(self) -> bool:
        """
        Checks if the electron configuration of an element has errors, considering
        its metallicity and the number of missing electrons.

        Returns:
            bool: Indicates if the electron configuration is erroneous.

        """
        if self.is_metal:
           return False

        if self.missing_electrons_element != 0:
            return True
        return False

    @property
    def is_hydrogen(self) -> bool:
        """
        Determines if the element is hydrogen based on its atomic number.

        Checks whether the atomic number of the element is equal to 1, 
        which is the atomic number of hydrogen.

        Returns:
            bool: True if the atomic number is 1; otherwise, False.
        """
        return self.atomic_number == 1

    @property
    def is_polar_hydrogen(self) -> bool:
        """
        Returns a boolean value indicating whether the atom is a polar 
        hydrogen. A polar hydrogen is defined as a hydrogen atom that is 
        bonded to a polar hydrogen site.

        Raises:
            ImportError: If there are issues accessing the attributes.

        Returns:
            bool: True if the atom is a polar hydrogen, False otherwise.
        """
        try:
            return self.is_hydrogen and self.neighbours[0].polar_hydrogen_site
        except ImportError:
            return False

    @property
    def is_noble_gases(self):
        """
        A property that checks if the current atom is a noble gas.

        This property evaluates the atomic number of the atom and determines whether
        it belongs to the set of noble gases.

        Returns:
            bool: True if the atomic number corresponds to a noble gas, False otherwise.
        """
        return self.atomic_number in Atom.elements.noble_gases

    @property
    def is_halogens(self):
        """
        Check if the atom is a halogen element based on its atomic number.

        Atoms with atomic numbers found in the predefined halogens list will return
        True for this property. Halogens are a group of elements with specific
        chemical properties.

        Returns:
            bool: True if the atom is a halogen, otherwise False.
        """
        return self.atomic_number in Atom.elements.halogens

    @property
    def is_metal(self):
        """
        Determine if the element is a metal based on its atomic number.

        This property checks whether the element, characterized by its atomic
        number, belongs to the set of elements classified as metals.

        Returns:
            bool: True if the element is a metal, otherwise False.
        """
        # return _clib.is_metal(self.atomic_number)
        return self.atomic_number in self.elements.metal

    @property
    def label(self) -> str:
        """
        Constructs and returns a label combining the symbol and the index.

        The label is generated by concatenating the symbol and string
        representation of the index. It is intended to uniquely identify an
        entity based on these two components.

        Returns:
             str: The concatenated label consisting of symbol followed by
                  the string representation of idx.
        """
        return self.symbol + str(self.idx)

    def link_with(self, other: "Atom"):
        """
        Links the current atom with another atom from a different molecule.

        This method assumes that the two atoms belong to different molecules and
        connects them by creating a bond between the atom in the current molecule
        and a cloned copy of the other atom. The molecule of the other atom is
        integrated into the molecule of the current atom as a separate component.

        Args:
            other ("Atom"): The atom from a different molecule to link with.

        Raises:
            AssertionError: If the two atoms belong to the same molecule.
        """
        assert self.mol is not other.mol, (
            "The `link_with` method should only implement in two atoms from different molecules, "
            "this atom and other atom come from a same molecule")

        other_clone_idx = other.idx + len(self.mol.atoms)
        self.mol.add_component(other.mol)

        other_clone = self.mol.atoms[other_clone_idx]
        self.mol.add_bond(self, other_clone)

    @property
    def mass(self):
        """
        Gets the mass of an object based on its atomic number.

        @property
            A read-only attribute that calculates and returns the mass
            of the current object using the provided atomic number.

        Raises:
            ValueError: If the atomic number is invalid or out of range.

        Returns:
            float: The calculated mass of the object.
        """
        return ob.GetMass(self.atomic_number)

    @property
    def molecule(self):
        """
        Property to access the molecule object.

        This property allows retrieving the molecule object associated with the
        current instance. The molecule object represents the data structure
        used to store and manipulate molecular information.

        Returns
        -------
        Any
            The molecule object associated with the instance.
        """
        return self.mol

    @property
    def neigh_idx(self) -> np.ndarray:
        """
        Provides the functionality to fetch the neighboring indices of a node 
        in a molecular graph, represented as a NetworkX graph. The method 
        ensures that the neighbors are returned as a NumPy array of integer 
        type. If the node is invalid or an error in the graph structure occurs, 
        an empty NumPy array is returned instead.

        Returns
        -------
        np.ndarray
            A NumPy array of integers representing the indices of neighboring 
            nodes in the graph. If an error occurs, an empty NumPy array is 
            returned.

        Raises
        ------
        nx.NetworkXError
            Raised if an error occurs in accessing the neighbors of the 
            specified node index due to the graph structure.
        """
        # try:
        #     return np.array(list(self.mol.graph.neighbors(self.idx)), dtype=int)
        # except nx.NetworkXError:
        #     return np.array([])
        return np.array([a.idx for a in self._neighbours])

    @property
    def hydrogens(self):
        """
        Property to retrieve the list of hydrogen atom neighbors.

        The hydrogens property filters and returns neighbors of the current atom that 
        are hydrogen atoms.

        Returns:
            list: A list containing atom objects of all neighbors that are identified 
            as hydrogen atoms.
        """
        return [a for a in self.neighbours if a.is_hydrogen]

    @property
    def neighbours(self) -> list['Atom']:
        """
        Access the list of neighboring atoms for the current atom in a molecule.

        @return: A list of neighboring Atom objects. If the operation encounters an
            issue, such as an invalid network structure, an empty list is returned.
        """
        return copy(self._neighbours)
        # try:
        #     return np.take(self.mol.atoms, self.neigh_idx).tolist()
        # except nx.NetworkXError:
        #     return []

    @property
    def heavy_neighbours(self) -> list['Atom']:
        """
        Determines and returns a list of all heavy atom neighbors of the
        current atom. Heavy atoms are defined as atoms with an atomic
        number not equal to 1.

        Returns:
            list[Atom]: A list of heavy atom neighbors for the current atom.
        """
        return [a for a in self.neighbours if a.atomic_number != 1]

    @property
    def missing_electrons_element(self):
        """
        Calculates and returns the number of missing electrons for the element
        based on its electron configuration. The method uses the atomic orbital
        data and calculates the difference between the maximum possible number 
        of electrons for the orbital and the current configuration.

        Returns:
            int: The number of missing electrons for the element.
        """
        conf = self.electron_configuration
        return sum(self.elements.atomic_orbital[conf[0]]) - sum(conf[1:])

    @property
    def missing_electrons(self):
        """ missing electrons in open shell for this atom. """
        return self.missing_electrons_element - self.sum_bond_orders

    @property
    def open_shell_electrons(self) -> int:
        """
        Calculate and return the number of open-shell electrons in the system.

        The method computes the electron configuration using the 
        `calc_electron_config` method and calculates the total number
        of electrons in the open shells from the configuration.

        Returns:
            int: The number of open-shell electrons.
        """
        n, l, conf = self.calc_electron_config()
        return sum(conf)

    @property
    def electron_configuration(self) -> np.ndarray[int]:
        return np.int_(self.attrs[Atom._ELECTRON_N_CONFIG: Atom._ELECTRON_G_CONFIG + 1])

    @electron_configuration.setter
    def electron_configuration(self, value) -> None:
        value = np.array(value)
        assert len(value) == 6
        self.attrs[Atom._ELECTRON_N_CONFIG: Atom._ELECTRON_G_CONFIG + 1] = value

    @property
    def l(self) -> int:
        """
        Gets the angular momentum quantum number (l) from the calculated
        electron configuration.

        The property calculates the electron configuration using the
        `calc_electron_config` method and retrieves only the `l` value
        from the returned configuration data.

        Returns:
            int: The angular momentum quantum number (l) derived from
            the electron configuration calculation.
        """
        n, l, conf = self.calc_electron_config()
        return l

    @property
    def n(self) -> int:
        """
        Returns the principal quantum number (n) which determines the energy
        level of the electron in an atom. This is calculated based on the 
        electron configuration.

        Returns:
            int: The principal quantum number of the electron.
        """
        n, l, conf = self.calc_electron_config()
        return n

    def calc_electron_config(self) -> (int, int, list[int]):
        """
        Calculates the electron configuration of an element based on its atomic number.

        This function returns the electron configuration for a given element as a
        tuple containing the total number of electrons, valence electrons, and
        a list of electrons in each energy level shell.

        Returns:
            tuple[int, int, list[int]]: A tuple representing the electron
            configuration, including the total number of electrons, the number
            of valence electrons, and a list of electrons in each shell.
        """
        return hpchem.calc_electron_config(self.atomic_number)

    @property
    def oxidation_state(self) -> int:
        """
        Determines the oxidation state of an atom based on its properties and bonded environment.

        The oxidation state is calculated using the formal charge, atomic number, bonds,
        electronegativity, and implicit hydrogens of the atom. Specific rules are implemented 
        for certain atomic numbers and for atoms that are metals, ensuring a consistent 
        computation for a wide range of chemical environments.

        Returns:
            int: The calculated oxidation state of the atom.

        Raises:
            AttributeError: If required properties are missing in the atom or its bonds.
        """
        if self.is_metal:
            return self.formal_charge
        elif self.atomic_number == 6:
            return 4
        elif self.atomic_number == 7:
            return 3
        elif self.atomic_number == 8:
            return 2
        elif self.atomic_number in [1, 9, 17]:  # H, F, Cl
            return 1

        _state = 0
        _temp_state = 0
        for bond in self.bonds:
            other_atom = bond.another_end(self)
            if other_atom.is_metal or other_atom.atomic_number == 1:
                continue

            if other_atom.atomic_number == 6:
                _temp_state += 1

            elif self.electronegativity > other_atom.electronegativity:
                _state -= int(bond.bond_order)
            # elif self.electronegativity < other_atom.electronegativity:
            else:
                _state += int(bond.bond_order)

        _state += self.implicit_hydrogens

        if _state > 0:
            _state += _temp_state
        elif _state < 0:
            _state -= _temp_state

        return _state

    def get_formal_charge(self) -> cython.int:
        """
        Determines the formal charge of the atom based on its atomic number, neighbors,
        and other attributes. Specific rules and conditions for different elements 
        are implemented to calculate the formal charge.

        Raises:
            No exceptions are explicitly raised by this function. However, it assumes 
            that the relevant attributes such as `atomic_number`, `is_metal`, 
            `neighbours`, and `sum_covalent_orders` are correctly defined and accessible.

        Returns:
            cython.int: Corresponds to the formal charge of the atom as determined 
            by its atomic properties and interaction with neighbors.
        """
        if self.is_metal or self.atomic_number in self.elements.metalloid_2nd:
            return Atom.elements.default_valence[self.atomic_number]
        elif self.atomic_number in [6, 14]:  # C, Si
            return 4
        elif self.atomic_number == 8:  # O
            return -2
        elif self.atomic_number == 7:  # N
            if all(na.atomic_number != 8 for na in self.neighbours):
                return -3
            else:
                return max(5, 2 * len([na for na in self.neighbours if na.atomic_number == 8]) + 1)
        elif self.atomic_number == 15: # P
            if all(na.atomic_number != 8 for na in self.neighbours):
                return 3
            else:
                return max(5, 2 * len([na for na in self.neighbours if na.atomic_number == 8]) + 1)
        elif self.atomic_number == 16: # S
            if all(na.atomic_number != 8 for na in self.neighbours):
                return -2
            elif self.sum_covalent_orders <= 4:
                return 4
            else:
                return 6
        elif self.atomic_number == 5: # B
            return 3
        elif self.atomic_number == 33: # As, TODO: more precise adjustments are needed.
            return 3
        elif self.atomic_number == 1:
            return 1
        elif self.is_halogens:
            return -1
        else:
            raise NotImplementedError(f'The formal charge of {self.symbol} is not implemented.')

    def get_valence(self) -> cython.int:
        """
        Returns the valence for an atom based on its atomic number or specific properties.

        The function calculates the valence of the current atom instance by considering 
        its atomic number, types of neighboring atoms, sum of covalent orders, and other 
        specific attributes (such as whether the atom is a metal, halogen, or noble gas). 
        Different rules apply depending on the atomic number and bonding configuration. 

        Raises an exception if the atomic number or class attributes do not provide sufficient 
        information to determine the valence.

        Returns:
            cython.int: The determined valence of the atom.

        Raises:
            AttributeError: If required attributes are missing from the object.
            KeyError: If the atomic number is not present in the default valence lookup.
        """
        # C Implementation
        # neigh_atomic_number = [a.atomic_number for a in self.neighbours]
        # _clib.get_valence(self.atomic_number, neigh_atomic_number, len(neigh_atomic_number))

        if self.atomic_number in [6, 14]:  # C, Si
            return 4
        elif self.atomic_number == 8:  # O
            return 2
        elif self.atomic_number == 7:  # N
            if all(na.atomic_number != 8 for na in self.neighbours):
                return 3
            else:
                return max(5, 2 * len([na for na in self.neighbours if na.atomic_number == 8]) + 1)
        elif self.atomic_number == 15: # P
            if all(na.atomic_number != 8 for na in self.neighbours):
                return 3
            else:
                return max(5, 2 * len([na for na in self.neighbours if na.atomic_number == 8]) + 1)
        elif self.atomic_number == 16: # S
            if all(na.atomic_number != 8 for na in self.neighbours):
                return 2
            elif self.sum_covalent_orders <= 4:
                return 4
            else:
                return 6
        elif self.atomic_number == 5: # B
            return 3
        elif self.atomic_number == 1 or self.is_halogens:
            return 1
        elif self.is_metal:
            return Atom.elements.default_valence[self.atomic_number]
        elif self.is_noble_gases:
            return 0
        else:
            return Atom.elements.default_valence[self.atomic_number]

    def set_valence_to_default(self):
        """
        Sets the valence of the atom to its default value.

        This method resets the valence of the atom using the default settings
        based on its atomic number. The default valence calculation can be 
        custom or provided by another function depending on the implementation.

        Raises:
            KeyError: If the atomic number is not recognized or is missing 
                from the valence configuration.
        """
        # self.valence = Atom._default_valence[self.atomic_number]
        self.valence = self.get_valence()

    def setattr(self, *, add_defaults=False, **kwargs):
        """
        Sets attributes on an object with the option to translate specific keys 
        into alternative metadata formats. Particularly useful for custom handling of 
        coordinate and symbol-to-atomic-number translation, while supporting additional 
        keyword arguments.

        Parameters:
            add_defaults (bool): Indicates whether default attributes should be added.
            kwargs: Additional keyword arguments to set as attributes.

        Keyword Args:
            coordinates (Any): Optional key representing 3D coordinates, expected to 
                be an iterable with three elements (e.g., [x, y, z]) that will be 
                translated into x, y, z attributes.
            symbol (str): Optional key representing an atomic symbol, which will be 
                converted into an 'atomic_number' attribute based on internal symbol 
                indexing.

        Raises:
            Any exceptions raised during the setting of attributes in the parent class's
            setattr method.

        This method allows dynamic translation of given attributes and delegates all 
        remaining attribute assignments to the parent's setattr implementation. Handles 
        keys intelligently, such as extracting individual axes from a coordinates tuple 
        or substituting atomic symbols with their corresponding atomic numbers.

        TODO: Warning!
        TODO: This methods is recommended to be invoked after the molecular graph has
        TODO: been built. If the method is invoked in the obj-create stage. Some unexpected
        TODO: error might raise.
        """
        coords = kwargs.get("coordinates", None)
        symbol = kwargs.get("symbol", None)

        if coords is not None:
            del kwargs["coordinates"]
            kwargs.update({'x': coords[0], 'y': coords[1], 'z': coords[2]})
        if isinstance(symbol, str):
            del kwargs["symbol"]
            kwargs['atomic_number'] = Atom.elements.symbols.index(symbol)

        super().setattr(add_defaults=add_defaults, **kwargs)

    @property
    def sum_bond_orders(self) -> int:
        """
        Represents the calculation of the total bond orders for a set of bonds. This
        property iterates through all bonds in the `bonds` attribute and sums up their
        individual bond orders.

        @property

        Returns:
            int: The sum of all bond orders within the `bonds` attribute.
        """
        return int(sum(b.bond_order for b in self.bonds))

    @property
    def sum_heavy_cov_orders(self) -> int:
        """
        Calculate the sum of heavy covalent bond orders for all bonds in the object.

        The method iterates over all bonds, filtering for bonds marked as heavy covalent,
        and computes the sum of their bond orders. It provides a concise total of
        bond orders exclusively for heavy covalent bonds.

        Returns: 
            int: The sum of bond orders for heavy covalent bonds in the object.
        """
        return int(sum(b.bond_order for b in self.bonds if b.is_heavy_covalent))

    @property
    def sum_covalent_orders(self) -> int:
        """
        Calculate the total of covalent bond orders for the bonds associated
        with the instance.

        @return: The sum of covalent bond orders
        @rtype: int
        """
        return int(sum(b.bond_order for b in self.bonds if b.is_covalent))

    @property
    def symbol(self) -> str:
        """
        The `symbol` property retrieves the chemical symbol of an element based on its atomic number
        from the `_symbols` attribute. This property is used to access the chemical symbol as a string
        representation.

        @property
            Gets the chemical symbol corresponding to the atomic number.

        Returns:
            str: The chemical symbol associated with the element's atomic number.
        """
        return self.elements.symbols[self.atomic_number]

    @symbol.setter
    def symbol(self, value: str):
        """
            Updates the element's atomic number using its chemical symbol.

            The `symbol` property allows setting the chemical symbol of the element,
            and automatically updates the corresponding atomic number.

            Raises:
                ValueError: If the provided symbol is not in the predefined list
                of symbols.

            Args:
                value: A string representing the chemical symbol of the element.
        """
        self.atomic_number = self.elements.symbols.index(value)

    @property
    def vector(self) -> np.ndarray:
        """
        Returns the vector representation of the object's coordinates as a numpy array.

        The property retrieves the coordinates of the object and converts them into a 
        one-dimensional numpy array, which can be used for mathematical operations or 
        further processing.

        Returns:
            np.ndarray: A numpy array representation of the coordinates.
        """
        return np.array([self.coordinates])


class AtomSeq:
    """
    Represent instances assembled by a sequence of atoms, like Bond, Angles, Torison, and Rings.
    This class represents a sequence of atoms and provides various utilities for working
    with atoms, their indices, bonds, and molecular structures related to the sequence.

    The AtomSeq class is designed to manage a sequence of atoms that belong to the same
    molecule. It validates the consistency of provided atoms, such as ensuring all atoms
    belong to the same molecule and verifying the expected number of atoms. The class provides
    methods and attributes for accessing atoms, their indices, associated bonds, and other
    structural information. It also supports conversion of the represented sequence to
    a new Molecule instance.

    Attributes:
        _length (int or None): Expected fixed length for atoms in sub-classes or None if no constraint exists.
        _atoms (tuple[Atom]): A tuple containing the sequence of Atom objects.
        _bonds (list[Bond]): A list containing Bond objects derived from the atom sequence.
    """
    _length = None

    def __init__(self, *atoms: Atom):
        self._check_is_same_mol(*atoms)
        self._check_atom_number(*atoms)
        self._atoms = atoms

        if isinstance(self, Bond):
            self._bonds = [self]
        else:
            self._bonds = [self.mol.bond(atoms[i].idx, atoms[i+1].idx) for i in range(len(atoms) - 1)]

    def __repr__(self):
        return f"{self.__class__.__name__}" + '(' + ''.join(a.symbol for a in self.atoms) + ')'

    def __getattr__(self, item):
        if re.match(r"atom\d+", item):
            idx = int(item[4:]) - 1
            return self._atoms[idx]

        elif re.match(r"bond\d+", item):
            idx = int(item[4:]) - 1
            atom_start = self._atoms[idx]
            atom_end = self._atoms[idx+1]
            return list(set(atom_start.bonds) & set(atom_end.bonds))[0]

        elif re.match(r"a\d+idx", item):
            len_idx = len(item) - 4
            idx = int(item[1:1+len_idx]) - 1
            return self._atoms[idx].idx

        else:
            super().__getattribute__(item)

    def __len__(self):
        return len(self._atoms)

    def __contains__(self, item: Union[Atom, "Bond"]):
        if isinstance(item, Atom):
            return item in self._atoms
        elif isinstance(item, Bond):
            return item in self._bonds

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if len(self.atoms) != len(other.atoms) or len(self.bonds) != len(other.bonds):
            return False

        return all(a in self._atoms for a in other.atoms) and all(b in self._bonds for b in other.bonds)

    def _replace_atom(self, old: Union[int, Atom], new: Atom):
        assert isinstance(new, Atom) and new not in self
        if isinstance(old, int):
            if 0 <= old < len(self._atoms):
                raise ValueError(f'The atom index in Bond must be greater than or equal to 0 and less than {len(self)}')
            self._atoms = tuple(a if i != old else new for i, a in enumerate(self._atoms))
        elif isinstance(old, Atom):
            assert old in self
            self._atoms = tuple(a if a is not old else new for a in self._atoms)
        else:
            raise TypeError(f"The old atom should be given by int or Atom, instead of a {type(old)}")

    @staticmethod
    def _check_is_same_mol(*atoms):
        if any(atoms[0].mol is not a.mol for a in atoms[1:]):
            raise ValueError('All atoms must belong to same mol.')

    def _check_atom_number(self, *atoms):
        _length = getattr(self, '_length', None)
        if _length and len(atoms) != _length:
            raise ValueError(
                f"The the atom counts of {self.__class__.__name__} is {_length}, but {len(self.atoms)} are given.")

    @property
    def atoms(self) -> Iterable[Atom]:
        return copy(self._atoms)

    @property
    def atoms_indices(self):
        return [a.idx for a in self.atoms]

    @property
    def bonds(self) -> list["Bond"]:
        return copy(self._bonds)

    @property
    def mol(self) -> Molecule:
        return self._atoms[0].mol

    @property
    def has_metal(self):
        return any(a.is_metal for a in self._atoms)

    def to_mol(self):
        mol = Molecule()
        for atom in self._atoms:
            mol.create_atom(**atom.attr_dict)

        mol_atoms = mol.atoms
        for bond in self._bonds:
            a1 = mol_atoms[self._atoms.index(bond.atom1)]
            a2 = mol_atoms[self._atoms.index(bond.atom2)]
            mol.add_bond(a1, a2, **bond.attr_dict)

        return mol


class AtomPairKey:
    def __init__(self, atom1: Atom, atom2: Atom):
        assert isinstance(atom1, Atom) and isinstance(atom2, Atom)
        self.atom1, self.atom2 = (atom1, atom2) if atom1.idx <= atom2.idx else (atom2, atom1)

    def __hash__(self):
        return hash((self.atom1, self.atom2)) + hash((self.atom2, self.atom1))

    def __repr__(self):
        return f"AtomPairKey({self.atom1}, {self.atom2})"

    def __lt__(self, other):
        return (self.atom1.idx, self.atom2.idx) < (other.atom1.idx, other.atom2.idx)

    def __eq__(self, other):
        return (other.atom1 == self.atom1 and other.atom2 == self.atom2) or (other.atom1 == self.atom2 and other.atom2 == self.atom1)

    def __contains__(self, check_atom):
        return check_atom is self.atom1 or check_atom is self.atom2


class AtomPair:
    """
    Represents a pair of atoms within a molecule.

    Defines a data structure for managing pairs of atoms in a molecule, 
    while offering attributes and methods that allow for the calculation and 
    retrieval of information such as geometric distance, bond order, and 
    shortest path length. The two atoms in the pair are required to belong 
    to the same molecule.

    Attributes:
        atom1 (Atom): The first atom of the pair.
        atom2 (Atom): The second atom of the pair.
        mol (Molecule): The molecule to which both atoms belong.
        wiberg_bond_order (float): An attribute defaulting to 0, representing 
            the Wiberg bond order of the atom pair.

    Raises:
        AssertionError: If atom1 and atom2 do not belong to the same molecule.

    Methods:
        __repr__:
            Returns a string representation of the atom pair, including 
            the indices of the atoms within the molecule.
        __eq__:
            Compares two AtomPair objects and determines equality based on 
            the constituent atoms, regardless of their order.
        __contains__:
            Checks if a specified atom is one of the two atoms in the pair.

    Properties:
        attrs:
            Returns a list of attribute values, corresponding to the 
            attribute names defined in the `attr_names` class-level variable.
        shortest_path:
            Retrieves the shortest path of atoms connecting atom1 and 
            atom2 within the molecule.
        length_shortest_path:
            Provides the length of the shortest path (in bonds) between 
            the two atoms.
        distance:
            Calculates and returns the Euclidean distance between the two 
            atoms based on their 3D coordinates.
    """
    _length = 2
    attr_names = (
        'wiberg_bond_order',
        'length_shortest_path'
    )
    def __init__(self, atom1: Atom, atom2: Atom):
        self.atom1 = atom1
        self.atom2 = atom2
        assert self.atom1.mol is self.atom2.mol
        self.mol = self.atom1.mol
        self.wiberg_bond_order = 0

    def __repr__(self):
        return f"AtomPair({self.atom1.idx}, {self.atom2.idx})"

    def __eq__(self, other):
        return (other.atom1 == self.atom1 and other.atom2 == self.atom2) or (other.atom1 == self.atom2 and other.atom2 == self.atom1)

    def __contains__(self, check_atom):
        return check_atom is self.atom1 or check_atom is self.atom2

    @property
    def attrs(self):
        return [getattr(self, n) for n in self.attr_names]

    @property
    def shortest_path(self) -> list['Atom']:
        """
        Finds and returns the shortest path between two atoms in the molecule.

        This method computes the shortest path between atom1 and atom2 using the
        molecule's internal graph representation. The returned path is a list of
        Atom objects representing the sequence of atoms along the shortest route.

        It relies on the molecule's shortest_path method to calculate the path, 
        assuming that the graph representation of the molecule is already 
        configured correctly.

        @return: A list of Atom objects, representing the shortest path.
        """
        return self.mol.shortest_path(self.atom1, self.atom2)

    @property
    def length_shortest_path(self) -> int:
        """
        Returns the length of the shortest path in the graph.

        The length of the shortest path is calculated based on the precomputed
        shortest_path property. This value represents the number of edges in 
        the shortest path.

        @return: The length of the shortest path
        @rtype: int
        """
        return len(self.shortest_path)

    @property
    def distance(self):
        """
        Calculates the Euclidean distance between two points in 3D space. This property retrieves the coordinates 
        of two atoms and computes the straight-line distance between them using numpy's linear algebra norm 
        function. It assumes that the coordinates are stored as three-dimensional tuples or lists.

        Returns:
            float: The Euclidean distance between the two atoms based on their 3D coordinates.
        """
        x2, y2, z2 = self.atom2.coordinates
        x1, y1, z1 = self.atom1.coordinates
        return np.linalg.norm((x2-x1, y2-y1, z2-z1))


class AtomPairs(dict):
    """
    A specialized dictionary-like object for managing atom pairs and their associated data.

    This class extends Python's built-in dictionary and is specialized for handling 
    pairs of 'Atom' objects. It enforces that all keys are frozensets of two 'Atom' 
    objects and that all corresponding values are instances of 'AtomPair'. It 
    provides additional methods for managing atom pairs specific to a molecular 
    context, ensuring integrity and convenience in molecular data processing.

    Attributes:
        mol: The molecule to which the atom pairs belong.

    Initialization:
        __init__(*args, mol, **kwargs):
            Creates an instance of AtomPairs, initializing it with the provided 
            key-value pairs and the parent molecule 'mol'.

    Methods:
        __setitem__(self, atom_pair, value):
            Sets an 'AtomPair' instance with a frozenset of two 'Atom' objects as the key.

            Raises TypeError if the value is not of type 'AtomPair'.

        __getitem__(self, atom_pair):
            Retrieves the value associated with a given frozenset of two 'Atom' objects.

        __contains__(self, atom_pair):
            Checks if a given frozenset of two 'Atom' objects exists in the dictionary.

        getdefault(self, atom_pair):
            Retrieves the value associated with a given frozenset of two 'Atom' objects 
            or inserts a new default 'AtomPair' if the key does not exist.

        clear_not_exist_pairs(self):
            Removes all atom pairs from the dictionary where one or both atoms 
            no longer exist in the molecule.

        pair_distance(self):
            Property that returns an array of distances for all atom pairs.

        pairs_shortest_path(self):
            Returns a dictionary mapping frozensets of atom indices to their shortest 
            path distances.

        update_pairs(self):
            Updates the atom pairs dictionary by removing non-existent pairs and 
            adding any missing pairs for combinations of atoms in the molecule.

        idx_matrix(self):
            Property that returns a 2D array where rows correspond to the indices 
            of atoms in each atom pair.

    Raises:
        TypeError: If a non-AtomPair value is assigned to the dictionary.
    """
    def __init__(self, *args, mol, **kwargs):
        self.mol = mol
        super().__init__(*args, **kwargs)

    def __setitem__(self, atom_pair: tuple[Atom, Atom], value: AtomPair):
        if not isinstance(value, AtomPair):
            raise TypeError(f'the value in AtomPairs should be `AtomPair`')

        assert len(atom_pair) == 2  # length of atom must be 2
        assert all(isinstance(k, Atom) for k in atom_pair)  # all items should be atom
        super().__setitem__(frozenset(atom_pair), value)

    def __getitem__(self, atom_pair):
        assert len(atom_pair) == 2  # length of atom must be 2
        assert all(isinstance(k, Atom) for k in atom_pair)  # all items should be atom
        return super().__getitem__(frozenset(atom_pair))

    def __contains__(self, atom_pair):
        assert len(atom_pair) == 2
        return super().__contains__(frozenset(atom_pair))

    def getdefault(self, atom_pair):
        return self.setdefault(frozenset(atom_pair), AtomPair(*atom_pair))

    def clear_not_exist_pairs(self):
        mol_atoms = set(self.mol.atoms)
        to_remove = []
        for pair_key in self:
            if any(a not in mol_atoms for a in pair_key):
                to_remove.append(pair_key)

        for pair_key in to_remove:
            del self[pair_key]

    @property
    def pair_distance(self):
        return np.array([p.distance for p in self.values()])

    def pairs_shortest_path(self) -> dict[frozenset[int, int], int]:
        """
        Computes the shortest path lengths for all pairs of nodes and returns 
        them in a dictionary where the keys are frozensets representing the 
        pairs of nodes, and the values are the shortest path lengths.

        Returns
        -------
        dict[frozenset[int, int], int]
            A dictionary with keys as frozensets containing pairs of nodes and 
            values as the shortest path lengths between the nodes.
        """
        return {k: p.length_shortest_path for k, p in self.items()}

    def update_pairs(self):
        """
        Updates the current list of atom pairs for the molecule. This function first clears any
        pairs that no longer exist in the molecule and then updates the atom pairs to include
        all possible unique combinations of two atoms from the molecule.
        """
        self.clear_not_exist_pairs()
        for pair_key in combinations(self.mol.atoms, 2):
            self.setdefault(frozenset(pair_key), AtomPair(*pair_key))

    @property
    def idx_matrix(self) -> np.ndarray:
        """
        Return a matrix of indices for the elements in the collection.

        This property method generates a matrix where each element of the 
        matrix is the index (`idx`) of the corresponding element in the 
        input structure. It assumes the elements in the collection and 
        sub-collections have an `idx` attribute.

        Returns:
            np.array: A two-dimensional array where each entry corresponds
            to the `idx` value of an element from the input collection.
        """
        return np.array([[a.idx for a in p] for p in self], dtype=np.int64)


######################################################################################
def _load_ibl():
    """ Load ideal bond length data sheet, the sheet statistic from CSD database """
    data_path = osp.join(osp.dirname(__file__), 'ChemData', 'ideal_bond_length.json')
    with open(data_path) as file:
        raw_data =  json.load(file)
    ibl_data = {}
    for k, v in raw_data.items():
        list_k = json.loads(k)
        bond_order_dict = ibl_data.setdefault(frozenset(list_k[:2]), {})
        bond_order_dict[list_k[2]] = v
    return ibl_data

def _bond_order_setter(self: "Bond", key, bond_order):
    assert key == 'bond_order'
    self._default_attr_setter(self, key, bond_order)

    for atom in self.atoms:
        atom.calc_implicit_hydrogens()

class Bond(AtomSeq, MolBlock):
    """
    Represents a chemical bond between two atoms.

    This class models a bond in a molecular structure, providing attributes and methods for representing bond properties 
    such as bond order, length, aromaticity, and covalency. It supports operations for determining bond characteristics, 
    accessing bond-related geometric information, and interacting with associated atoms and molecular structures.
    """
    _ideal_bond_length = _load_ibl()

    _attrs_setter = {
        'bond_order': _bond_order_setter
    }

    _attrs_dict = {
        'bond_order': float,
        'constraint': bool,
        'id': int,
    }
    _attrs_enumerator = tuple(_attrs_dict.keys())

    # TODO: Modify the bond type to Enum type
    _bond_order_symbol = {
        0.: '?',
        1.: '-',
        1.5: '@',
        2.: '=',
        3.: '#'
    }
    _bond_order_names = {
        0.: 'Unknown',
        1.: 'Single',
        1.5: 'Aromatic',
        2.: 'Double',
        3.: 'Triple'
    }

    def __init__(self, atom1: Atom, atom2: Atom, **kwargs):
        super().__init__(atom1, atom2)
        # self.attrs = np.zeros(len(self._attrs_enumerator))
        self.attrs = np.array([kwargs.pop(a, 0.) for a in self._attrs_enumerator])

        # TODO: This invoke might cause some unexpected bug and error
        self.setattr(**kwargs)  # TODO: it's not recommended to be used to create an new bond obj

    def __repr__(self):
        return MolBlock.__repr__(self)

    def __eq__(self, other):
        return (other.atom1 == self.atom1 and other.atom2 == self.atom2) or (other.atom1 == self.atom2 and other.atom2 == self.atom1)

    def __hash__(self):
        return hash((self.atom1, self.atom2)) + hash((self.atom2, self.atom1))

    def __getattr__(self, item):
        try:
            return AtomSeq.__getattr__(self, item)
        except AttributeError:
            return MolBlock.__getattr__(self, item)

    def another_end(self, this_end: Atom):
        """
        Retrieves the opposite end of a bond from a given atom. This function determines if the provided
        atom is one of the ends of the bond and returns the other atom at the opposite end. If the given
        atom is not found at any end of the bond, an exception is raised.

        Args:
            this_end (Atom): The atom for which to find the opposite bond end.

        Returns:
            Atom: The atom at the opposite end of the bond.

        Raises:
            ValueError: If the provided atom is not one of the ends of the bond.
        """
        try:
            return self.atoms[abs(self.atoms.index(this_end) - 1)]
        except ValueError:
            raise ValueError("The given atom is in neither ends of the bond!")

    @property
    def bond_line(self) -> geometry.Line:
        """
        Returns a geometry.Line instance representing the bond line 
        between two atoms based on their coordinates.

        @return geometry.Line
            A line connecting the coordinates of the two atoms.
        """
        return geometry.Line(self.atom1.coordinates, self.atom2.coordinates)

    @property
    def idx(self) -> int:
        """
        Returns the index of the bond in the molecule's bond list.

        This property retrieves the index of the bond object within the list 
        of bonds in the associated molecule. It provides a way to identify 
        the bond's position for further processing or referencing in the 
        molecular structure.

        Returns:
            int: The zero-based index of the bond in the molecule's bonds list.
        """
        return self.mol.bonds.index(self)

    @property
    def ideal_bond_length(self) -> float:
        ibl_bond_order_dict = Bond._ideal_bond_length[frozenset([self.atom1.atomic_number, self.atom2.atomic_number])]
        return ibl_bond_order_dict[self.bond_order]

    @property
    def length(self) -> float:
        """
            Calculates and returns the length of the vector between two atomic positions.
            Returns:
                float: The Euclidean distance (length) between the two atomic vectors.
        """
        return float(np.linalg.norm(self.atom1.vector - self.atom2.vector))

    @property
    def label(self):
        """
        Returns the bond label represented as a string.

        Combines the labels of two atoms with a symbolic representation of the bond
        order to construct a descriptive identifier of the bond.

        :return: A string label identifying the bond.
        :rtype: str
        """
        return f"{self.atom1.label}{self._bond_order_symbol[self.bond_order]}{self.atom2.label}"

    @property
    def is_aromatic(self) -> bool:
        """
        Check if the molecule contains any aromatic ring.

        This property evaluates the presence of aromaticity within the molecule by 
        checking whether any of its constituent rings are aromatic.

        @return: Indicates if the molecule contains aromatic rings.

        """
        return any(r.is_aromatic for r in self.rings)

    @property
    def is_covalent(self) -> bool:
        """
        Determines if the bond is covalent.

        Checks whether any atom in the list of atoms is a metal.
        Returns True if no atoms are metals, indicating a covalent bond.

        @return: True if the bond is covalent, otherwise False
        @rtype: bool
        """
        return not any(a.is_metal for a in self.atoms)

    @property
    def is_heavy_covalent(self) -> bool:
        """
        Determine if a molecule is a heavy covalent molecule.

        This property checks if the molecule does not contain any metal atoms or 
        hydrogen (atomic number 1) among its constituent atoms. If all atoms in 
        the molecule qualify under this rule, the molecule is considered a 
        heavy covalent molecule.

        Returns
        -------
        bool
            True if the molecule is considered a heavy covalent molecule. 
            False otherwise.
        """
        return not any(a.is_metal or a.atomic_number == 1 for a in self.atoms)

    # @is_aromatic.setter
    # def is_aromatic(self, value: bool):
    #     if not self.in_ring:
    #         raise AttributeError("Can't assign a bond outside an ring to be ")
    #
    #     self.atom1.is_aromatic = value
    #     self.atom2.is_aromatic = value

    @property
    def is_metal_ligand_bond(self) -> bool:
        """
        Determine whether the bond is a metal-ligand bond.

        A metal-ligand bond is identified if one atom in the bond is a metal and the 
        other is not. This property verifies this condition and returns the result.

        @return: 
            A boolean indicating if the bond is a metal-ligand bond or not.
        """
        return (self.atom1.is_metal and not self.atom2.is_metal) or (not self.atom1.is_metal and self.atom2.is_metal)

    @property
    def rotatable(self) -> bool:
        """
        Determine if a bond is rotatable based on its properties.

        A bond is considered rotatable if it has a bond order of 1, is not part of an
        aromatic structure, and does not lie within a ring structure. This property can
        be useful for evaluating flexibility in molecular structures.

        @property

        Returns:
            bool: True if the bond is rotatable, otherwise False.
        """
        return (self.bond_order == 1) and (not self.is_aromatic) and (not self.in_ring)

    def bond_line_distance(
            self, other_bond: "Bond",
            relationship: bool = False
    ) -> Union[float, tuple[float, str]]:
        rela, dist = self.bond_line.distance_to_line(other_bond.bond_line)
        if relationship:
            return dist, rela
        return dist


class Angle(AtomSeq):
    """
    Represents an angle defined by three atoms in a sequence.

    This class determines the angular relationship between three atoms represented as 
    coordinates. It provides functionality to calculate the angle in degrees 
    and supports defining constraints for the angular relationship. Extends from 
    the AtomSeq class to handle sequences of atoms.

    Attributes:
    constraint: Defines if a constraint is applied to the angle. Default is False.
    """
    def __init__(self, a1: Atom, a2: Atom, a3: Atom):
        super().__init__(a1, a2, a3)
        self.constraint = False

    @property
    def degrees(self) -> float:
        return InternalCoordinates.calc_angle(
            self.atom1.coordinates,
            self.atom2.coordinates,
            self.atom3.coordinates
        )


class Torsion(AtomSeq):
    """
    Represents a torsion or dihedral angle defined by a sequence of four atoms.

    This class models a torsion angle in molecular geometry, which is the angle 
    formed by four atoms arranged in a sequence. It inherits from the AtomSeq 
    class to represent the ordered sequence of atoms defining the torsion. 
    This class also provides properties to compute metrics such as the torsion 
    angle in degrees and determine whether the torsion is rotatable.

    Attributes:
        constraint (bool): Indicates if a constraint is applied to the torsion 
            angle. Defaults to False.

    Methods:
        degrees (float): Returns the torsion angle in degrees based on the 
            coordinates of the atoms in the sequence.

        rotatable (bool): Determines if the torsion angle is rotatable based 
            on the properties of the connecting bonds.
    """
    def __init__(self, a: Atom, b: Atom, c: Atom, d: Atom):
        super().__init__(a, b, c, d)
        self.constraint = False

    @property
    def degrees(self) -> float:
        return InternalCoordinates.calc_dehedral(
            self.atom1.coordinates,
            self.atom2.coordinates,
            self.atom3.coordinates,
            self.atom4.coordinates
        )

    @property
    def rotatable(self) -> bool:
        """
        Returns whether the bond is rotatable.

        The property determines if the associated bond object is rotatable
        based on the characteristics of `bond2`. This is typically used 
        to ascertain flexibility or degrees of freedom in molecular 
        structures.

        @return: True if the bond is rotatable, False otherwise
        """
        return self.bond2.rotatable



class JointRing:
    """
    Represents a joint structure of multiple chemical rings.

    The JointRing class is designed to model and analyze a combined structure of
    multiple rings in a molecular context. It provides methods and properties to 
    query atom and bond connectivity, check neighborhood relationships, and verify
    specific kekulization constraints. It aims to support cheminformatics workflows 
    involving ring systems.

    Attributes:
        rings (tuple): Stores the constituent Ring objects. It is populated during 
            initialization with `*ring`.
        mol: The molecule associated with the first ring in the input. This attribute 
            assumes all provided rings belong to the same molecule.
        atoms (list): Unique list of all atoms shared among the provided rings.
        bonds (list): Unique list of all bonds shared among the provided rings.

    Methods:
        __contains__(item): Checks membership of an atom, bond, or ring within 
            the JointRing instance.
        empty: Property indicating whether the JointRing contains no rings.
        index(atom): Returns the positional index of the given atom within the 
            `atoms` list if it exists, otherwise it returns None.
        index_bond(bond): Provides the indices of the two atoms forming the bond 
            within the `atoms` list.
        atom_neigh_atom(atom): Fetches all neighboring atoms of a specified atom 
            that are within the JointRing.
        atom_neigh_bond(atom): Lists all bonds connected to a specified atom and 
            contained within the JointRing.
        bond_neigh_bond(bond): Retrieves all bonds neighboring a specified bond 
            while remaining part of the JointRing.
        check_kekulize: Validates the kekulization constraints of atoms in the 
            JointRing.
        kekulize: Placeholder for implementing kekulization on the JointRing.
    """
    def __init__(self, *ring):
        self.rings = ring
        self.mol = ring[0].mol
        self.atoms = list(set([a for r in ring for a in r.atoms]))
        self.bonds = list(set([b for r in ring for b in r.bonds]))

    def __contains__(self, item: Union[Atom, Bond, 'Ring']):
        if isinstance(item, Atom):
            if self.index(item):
                return True
            return False
        elif isinstance(item, Bond):
            if all(self.index_bond(item)):
                return True
            return False
        elif isinstance(item, Ring):
            if any(all(a in r.atoms for a in item.atoms) for r in self.rings):
                return True
            return False
        else:
            raise TypeError("Expected an instance of Atom or Bond or Ring, got {}".format(type(item)))

    @property
    def empty(self) -> bool:
        """
        Checks whether the object is empty based on the absence of rings.

        The property allows to determine if the object has any rings 
        present. If no rings are found, it indicates emptiness by returning 
        True. Otherwise, it returns False.

        Returns:
            bool: True if there are no rings; False otherwise.
        """
        return not self.rings

    def index(self, atom: Atom) -> Optional[int]:
        """
        Search for the index of a specified atom in the atom list.

        This method attempts to find the index of an atom in the list of atoms. If
        the atom does not exist in the list, the method will return None.

        Parameters:
            atom (Atom): The atom object to be searched in the list.

        Returns:
            Optional[int]: The index of the atom in the list if found, 
            otherwise None.

        Raises:
            ValueError: This exception is internally caught if the atom 
            is not found in the list.
        """
        try:
            return self.atoms.index(atom)
        except ValueError:
            return None

    def index_bond(self, bond: Bond):
        """
        Indexes the atoms associated with a bond in a molecular structure.

        Given a bond object, this method retrieves the indices of the two atoms
        that define the bond in the molecular structure. It assumes that the 
        bond object contains attributes `atom1` and `atom2`, which represent 
        the two atoms forming the bond. The function returns a tuple 
        containing their respective indices.

        Parameters:
            bond (Bond): The bond object containing two atoms.

        Returns:
            tuple[int, int]: A tuple containing the indices of `atom1` and 
            `atom2` in the structure.
        """
        return self.index(bond.atom1), self.index(bond.atom2)

    def atom_neigh_atom(self, atom: Atom):
        """
        Find and return the neighboring atoms from a given atom that are present within 
        the current collection or structure. The function filters the neighbors 
        of the given atom to include only those atoms belonging to the caller structure.

        Args:
            atom (Atom): The reference atom for which neighbors need to be identified.

        Returns:
            list: A list of neighboring atoms that are part of the current 
            structure or collection.
        """
        return [a for a in atom.neighbors if a in self]

    def atom_neigh_bond(self, atom: Atom):
        """
        Filters the bonds of a given atom to return only those that are part 
        of the current context.

        Summary:
        This method evaluates the bonds associated with a given atom and 
        returns a list containing only the bonds that exist within the 
        current context. The comparison is performed by checking each bond 
        against the bonds available in the object's state.

        Args:
            atom (Atom): The atom whose bonds are being filtered.

        Returns:
            list: A list of bonds connected to the atom and present 
            in the object's context.

        """
        return [b for b in atom.bonds if b in self]

    def bond_neigh_bond(self, bond: Bond):
        """
        Identifies and retrieves all bonds connected to the atoms of a given bond, excluding the 
        input bond itself, while ensuring they exist within the current context.

        Args:
            bond (Bond): The bond whose neighboring bonds will be identified.

        Returns:
            list[Bond]: A list of neighboring bonds connected to the atoms of the provided bond.
        """
        return [b for a in bond.atoms for b in a.bonds if (b is not bond and b in self)]

    def check_kekulize(self):
        """
        Checks whether the molecular structure adheres to Kekulé rules.

        This function evaluates the atomic properties of each atom in a given structure
        to determine if it satisfies Kekulé bonding rules. The checks focus on
        the atomic number, sum of heavy covalent bond orders, and the number of
        implicit hydrogens for specific atom types. If any atom violates these rules,
        the function returns False. Otherwise, it returns True once all atoms are validated.

        Returns:
            bool: True if the structure conforms to Kekulé rules, otherwise False.
        """
        for atom in self.atoms:
            if atom.atomic_number == 6:
                if atom.sum_heavy_cov_orders != 3:
                    return False
            if atom.atomic_number in [7, 15]:
                if atom.sum_heavy_cov_orders == 2 and atom.implicit_hydrogens != 1:
                    return False
                elif atom.sum_heavy_cov_orders == 3 and atom.implicit_hydrogens != 0:
                    return False
            if atom.atomic_number in  [5, 8, 16]:
                if atom.sum_heavy_cov_orders == 2:
                    return False

        return True


    def kekulize(self):
        raise NotImplementedError


class Ring(AtomSeq):
    """
    Represents a chemical ring structure composed of atoms connected by bonds.

    This class is used to model a ring structure in a molecule, which is a 
    closed sequence of atoms where the first atom is connected to the last 
    atom. It provides various methods and properties to analyze and manipulate 
    the ring. This includes determining aromaticity, checking for the presence 
    of metals or disordered structures, evaluating bonds, and generating 
    graph representations of the ring for further computational purposes.
    """
    def __init__(self, *atoms: Atom):
        super().__init__(*atoms)
        self._bonds = self._bonds + [self.mol.bond(self._atoms[0].idx, self._atoms[-1].idx)]

    def to_pair_edge(self):
        """
        The method to export all pairwise edges between every pair of atoms in the ring.
        This method would be useful to leverage Graph Neural Networks to encode the rings information,
        where every atom in the ring would be regarded to be close with each other, that
        each atom has edges with all of other atoms in the ring.

        Returns:
            np.ndarray: A numpy array consisting of all possible pairwise
            combinations of atom indices represented as edges.
        """
        return np.array([a1a2idx for a1a2idx in combinations(self.atoms_indices, 2)])

    @property
    def is_aromatic(self) -> bool:
        """
        Returns whether the molecule is aromatic.

        This property determines the aromaticity of the molecule by
        evaluating if all the atoms in the molecule are aromatic. The
        atoms are checked for their respective aromaticity status.

        Returns:
            bool: True if all the atoms in the molecule are aromatic,
            otherwise False.
        """
        return all(a.is_aromatic for a in self._atoms)

    @is_aromatic.setter
    def is_aromatic(self, value: bool):
        """
        Sets the aromatic property for all atoms in the molecule.

        This method updates the `is_aromatic` property for each individual atom
        contained in the molecule to the specified boolean value. The operation affects
        all atoms in the list of atoms associated with this molecule.

        Parameters:
            value (bool): A boolean value indicating whether the atoms should be 
                          set as aromatic or not.
        """
        for a in self._atoms:
            a.is_aromatic = value

    @property
    def is_disorder(self):
        """
        This property checks whether any pairwise distance within a given dataset
        is below a specified threshold, which indicates disorder in the data. 
        The threshold is set to 0.5. Returns a boolean value: True if disorder is 
        detected, otherwise False.

        @return: bool
            True if any pairwise distance in the dataset is less than 0.5, 
            False otherwise.
        """
        return np.any(self.pair_dist < 0.5)

    @property
    def has_metal(self):
        """
        Check if the structure contains any metallic elements.

        This property evaluates whether any of the atoms within the structure
        are classified as metallic based on their `is_metal` attribute.

        @return: True if at least one atom is metallic, otherwise False.
        :rtype: bool
        """
        return any(a.is_metal for a in self._atoms)

    def joint_with(self, other: "Ring") -> bool:
        """
        Determines if the current ring is joint with another ring by checking for 
        shared bonds. A joint is defined as sharing exactly one bond. 

        Parameters
        ----------
        other: Ring
            The other ring to check for connection.

        Returns
        -------
        bool
            Returns True if the current ring is joint with the other ring, 
            otherwise False.

        Raises
        ------
        TypeError
            Raised when the provided object is not of type Ring.
        ArithmeticError
            Raised when there are multiple intersecting bonds, which indicates 
            an inconsistency with the definition of a ring joint.
        """
        if not isinstance(other, Ring):
            raise TypeError(f"expected Ring but got {type(other)}")

        intersect_bonds = list(set(self.bonds) & set(other.bonds))
        if len(intersect_bonds) == 0 :
            return False
        elif len(intersect_bonds) == 1:
            return True
        else:
            raise ArithmeticError("incorrect Ring")

    def joint_ring(self, aromatic=True) -> Optional["JointRing"]:
        """
        Finds and constructs a JointRing involving the current ring.

        A JointRing represents a combination of rings in the molecule that are 
        connected or interact in a specified way. This function checks for joint 
        connections between the current ring and other rings in the molecule, 
        and optionally filters by aromaticity.

        Parameters:
        aromatic: bool
            Whether to consider only aromatic rings when joining.

        Returns:
        Optional[JointRing]
            A new JointRing object if one or more rings meet the conditions, 
            otherwise None.
        """
        if aromatic and not self.is_aromatic:
            return None

        rings = []
        for r in self.mol.rings:
            if r == self:
                rings.append(r)
            elif self.joint_with(r) and (not aromatic or r.is_aromatic):
                rings.append(r)

        if rings:
            return JointRing(*rings)

    def next_atom(self, atom: Atom, reverse: bool = False) -> Atom:
        """
        Returns the next atom in the sequence based on the current atom and the given 
        direction. If the current atom is at the end of the sequence when navigating 
        forward or at the beginning when navigating in reverse, it cycles to the 
        start of the sequence.

        Parameters
        ----------
        atom : Atom
            The current atom from which the navigation begins.
        reverse : bool, optional
            A flag indicating the navigation direction. If True, navigates in reverse;
            otherwise, navigates forward (default is False).

        Returns
        -------
        Atom
            The next atom in the sequence, or the first atom if the navigation wraps 
            around the sequence.
        """
        idx = self.atoms.index(atom)

        _next = -1 if reverse else 1

        if idx < len(self.atoms):
            return self.atoms[idx+_next]
        else:
            return self.atoms[0]

    @property
    def has_3d(self):
        """
        Check if the structure contains 3D coordinates.

        The property determines whether any atom in the structure has 3D coordinates
        different from the first atom.

        @return: True if any atom has 3D coordinates different from the first atom, False otherwise 
        @rtype: bool
        """
        return any(a.coordinates != self.atoms[0] for a in self.atoms)

    def determine_aromatic(self, inplace=False) -> bool:
        """
        Determines if the molecule is aromatic, applying Hückel's Rule.

        This method evaluates the aromatic behavior of a molecule based on its
        structure and specific electronic configuration. It includes conditions
        for neutral molecules and handles planar geometry checks for aromaticity.
        This operation can either return the result directly or modify the object's
        aromaticity status if `inplace` is set to True.

        Parameters
        ----------
        inplace : bool, default=False
            If True, the property ``is_aromatic`` of the object will be updated
            in-place based on the aromaticity check result.

        Returns
        -------
        bool
            True if the molecule is found to be aromatic, otherwise False.
        """
        if self.is_aromatic:
            return True

        def _neutral_mol_check():
            if not self.has_3d:
                # TODO: for neutral molecule just.
                pi_electron = 0
                for a in self._atoms:
                    if a.atomic_number == 6:
                        if len(a.heavy_neighbours) + a.implicit_hydrogens != 3:
                            return False
                        pi_electron += 1

                    elif a.atomic_number in (7, 15):
                        if len(a.heavy_neighbours) + a.implicit_hydrogens == 3:
                            pi_electron += 2
                        elif len(a.heavy_neighbours) + a.implicit_hydrogens == 2:
                            pi_electron += 1
                        else:
                            return False

                    elif a.atomic_number in (8, 16):
                        if len(a.heavy_neighbours) + a.implicit_hydrogens != 2:
                            return False
                        pi_electron += 2

                    elif a.atomic_number == 5:  # B
                        pi_electron += 0

                    else:
                        return False

                return (pi_electron - 2) % 4 == 0

            else:
                if not geometry.points_on_same_plane(*(a.coordinates for a in self.atoms)):
                    return False

                pi_electrons = []
                for a in self._atoms:
                    if a.atomic_number == 6:
                        if len(a.neigh_idx) > 3:
                            return False
                        pi_electrons.append((1,))

                    elif a.atomic_number in (7, 15):
                        if len(a.neigh_idx) > 3:
                            return False
                        pi_electrons.append((1, 2))

                    elif a.atomic_number in (8, 16):
                        if len(a.neigh_idx) != 2:
                            return False
                        pi_electrons.append((2,))

                return any((sum(pie) - 2) % 4 == 0 for pie in product(*pi_electrons))

        if self.has_metal:
            judge = False
        else:
            judge = _neutral_mol_check()

        if inplace:
            self.is_aromatic = judge
        return judge

    def is_bond_intersect_the_ring(self, bond: Bond) -> bool:
        """
        Determines if a given bond intersects with the ring structure.

        This method checks whether a bond intersects with the ring structure
        represented by the cycle. It ensures the bond is not already a part of
        the cycle, and then evaluates intersection using the geometric
        properties of the bond line and cycle.

        Args:
            bond: The bond to be checked for intersection with the ring.

        Returns:
            A boolean indicating whether the bond intersects the ring.
        """
        if bond in self._bonds:
            return False

        return self.cycle_places.is_line_intersect_the_cycle(bond.bond_line)

    @property
    def cycle_places(self) -> geometry.CyclePlanes:
        """
        Returns the cycle planes of the current molecular structure.

        The cycle planes are calculated based on the coordinates of the atoms
        present in the molecule. This property provides a geometry.CyclePlanes 
        object encapsulating the result.

        @return: geometry.CyclePlanes instance representing the cycle planes
        @rtype: geometry.CyclePlanes
        """
        return geometry.CyclePlanes(*[a.coordinates for a in self.atoms])

    def closest_edge_to_bond(self, bond: Bond) -> 'Bond':
        if bond not in self.mol.bonds:
            raise NotInSameMolecule(self, bond)

        min_dist_index = int(np.argmax([rb.bond_line_distance(bond) for rb in self._bonds]))
        return self._bonds[min_dist_index]

    def kekulize(self):
        """
        Kekulize method analyzes and adjusts the bond orders within a molecule to ensure the aromaticity rules 
        are consistently applied. This function modifies the bond orders directly in place to achieve a strict 
        Kekulé structure representation if the molecule has aromatic properties.
        """
        if not self.determine_aromatic(inplace=True):
            return

        def _refresh(bs):
            for b in bs:
                b.bond_order = 1

        _refresh(self._bonds)
        for bond in self._bonds:
            if all((end_atom.atomic_number not in [5, 8, 16] and eab.bond_order == 1)
                   for end_atom in bond.atoms for eab in end_atom.bonds):
                bond.bond_order = 2


class Conformers:
    """
    Representing an ensemble of conformers of a Molecule
    Class for managing and manipulating conformer data.

    This class is designed to handle conformer-related attributes such as 
    coordinates, energy, partial charges, spin multiplicity, Gibbs free energy, 
    forces, zero-point energy, thermodynamic properties, and heat capacity. 
    It provides methods to add new conformers, retrieve specific properties, 
    clear all stored data, and iterate over the stored conformers. This is 
    useful for computational chemistry or molecular modeling applications.

    Methods:
        __init__: Initializes an empty Conformers object with placeholders 
        for properties.
        __len__: Returns the number of stored conformers.
        __getitem__: Retrieves a dictionary of conformer attributes at 
        a specified index.
        __iter__: Iterates over stored conformers.
        index_attr: Retrieves a specific attribute by its name and index.
        add: Adds new conformer data, including coordinates and optional 
        energy.
        clear: Clears all stored conformer data.
        coordinates: Retrieves coordinates of a conformer by index.
        energy: Retrieves the energy of a conformer by index.
        gibbs: Retrieves the Gibbs free energy of a conformer by index.
        partial_charge: Retrieves partial charges of a conformer by index.
        force: Retrieves the forces on a conformer by index.
    """
    _attrs = (
        'coordinates',
        'energy',
        'partial_charge',
        'gibbs',
        'force',
        'zero_point',
        'spin_mult',
        'thermo',
        'capacity'
    )

    def __init__(self):
        self._coordinates = None
        self._energy = None
        self._partial_charges = None
        self._spin_mult = None
        self._gibbs = None
        self._force = None
        self._zero_point = None
        self._thermo = None
        self._capacity = None

    def __len__(self):
        if self._coordinates is None:
            return 0
        return len(self._coordinates)

    def __getitem__(self, idx):
        info = {}
        for attr in self._attrs:
            try:
                info[attr] = self.index_attr(attr, idx)
            except (IndexError, TypeError):
                continue

        return info

    def __iter__(self):
        return iter(self._coordinates)

    def index_attr(self, name, i):
        try:
            return getattr(self, f"_{name}")[i]
        except IndexError:
            raise IndexError(f'The number of conformers is {len(self._coordinates)}, but attempt to access the index {i}')
        except TypeError:
            try:
                return getattr(self, f"_{name}")
            except AttributeError:
                raise AttributeError(f"The conformers did not store {name} information")

    def add(self, coords: types.ArrayLike, energy: Union[types.ArrayLike, float] = None):
        """
        Adds new coordinates and optionally associated energy values to the conformer.

        This method stores molecular coordinates, formatted to ensure a consistent 
        three-dimensional structure. Optionally, it allows associating energy values 
        with the conformer, verifying compatibility between the number of energy values 
        and the number of coordinates. If coordinates and energy have been previously added, 
        the method appends the new values to the existing ones.

        Parameters:
            coords (types.ArrayLike): The molecular coordinates to add. Must be a 2D or 3D 
                                      array convertible to a consistent 3D array structure.
            energy (Union[types.ArrayLike, float], optional): Associated energy values, formatted 
                                                              as a scalar or a flattenable array.

        Raises:
            ValueError: If the number of provided energy values does not match the number 
                        of coordinate sets in the conformer.
        """
        coords = np.array(coords)

        if coords.ndim == 2:
            coords = coords.reshape((1,) + coords.shape)

        assert coords.ndim == 3
        assert coords.shape[-1] == 3

        if isinstance(self._coordinates, np.ndarray) and self._coordinates.shape[1] == coords.shape[1]:
            coords = np.vstack((self._coordinates, coords))

        self._coordinates = coords

        if energy is not None:
            if isinstance(energy, float):
                energy = np.array([energy])
            else:
                energy = np.array(energy).flatten()

            if self._energy is not None:
                energy = np.vstack((self._coordinates, coords))

            if len(energy) != len(self._coordinates):
                raise ValueError("The length of input energy in conformer nust matches with the coordinates in conformer")

            self._energy = energy

    def clear(self):
        for attr_name in self._attrs:
            setattr(self, f"_{attr_name}", None)

    def coordinates(self, i):
        return self.index_attr('coordinates', i)

    def energy(self, i):
        return self.index_attr('energy', i)

    def gibbs(self, i):
        return self.index_attr('gibbs', i)

    def partial_charge(self, i):
        return self.index_attr('partial_charge', i)

    def force(self, i):
        return self.index_attr('force', i)




class InternalCoordinates:
    """
    Represents and manages internal coordinates of a molecular system.

    This class facilitates the representation of internal coordinates using methods such
    as z-matrices. It also provides various mathematical utilities for calculations
    related to molecular geometry, such as bond lengths, angles, and dihedrals. The
    internal structure is represented by graph theory concepts, and it supports
    operations on graph representations of molecular linkages.

    Attributes:
    coordinates (np.ndarray): The coordinates of the system's atoms.
    link_matrix (np.ndarray): The connectivity data of the internal structure.
    """
    def __init__(self, coordinates: np.ndarray, link_matrix: np.ndarray):
        self.coordinates = coordinates
        self.link_matrix = link_matrix
        self._graph = nx.Graph()
        self._graph.add_edges_from(link_matrix)

    @staticmethod
    def calc_zmat_idx(graph_or_link: Union[nx.Graph, np.ndarray]) -> np.ndarray:
        """
            Calculates the Z-Matrix index for a given graph representation. This function constructs
            the Z-Matrix index for a molecular structure represented either as a NetworkX graph or as
            an adjacency list (numpy ndarray). The Z-Matrix index is used to uniquely represent the
            topology of molecules through connectivity information.

            Parameters:
            graph_or_link (Union[nx.Graph, np.ndarray]): The input graph representation. It can either be a
                NetworkX Graph object or a numpy ndarray of shape (n_edges, 2) representing edges.

            Returns:
            np.ndarray: A numpy array of shape (n_nodes, 4) where each node's connectivity pattern is
                described by its corresponding index path in the graph.

            Raises:
            TypeError: If the input parameter 'graph_or_link' is neither a NetworkX graph
                nor a numpy ndarray.
            RuntimeError: If an incorrect Z-Matrix is constructed due to issues in graph traversal.
        """
        if isinstance(graph_or_link, np.ndarray):
            assert graph_or_link.shape[1] == 2
            _graph = nx.Graph()
            _graph.add_edges_from(graph_or_link)
        elif isinstance(graph_or_link, nx.Graph):
            _graph = graph_or_link
        else:
            raise TypeError(f"The graph_or_link argument should be a nx.Graph or np.ndarray")

        # TODO: check the nodes id, the nodes key should from 0 to n-1

        # Create z-matrix index
        zmat_idx = -1 * np.ones((_graph.number_of_nodes(), 4), dtype=int)
        visited_nodes = set()
        num_nodes = len(_graph)
        node = 0
        while len(visited_nodes) < num_nodes:
            row = len(visited_nodes)
            visited_nodes.add(node)

            path = graph.graph_dfs_path(_graph, node, scope_nodes=visited_nodes, max_deep=min(row + 1, 4))
            zmat_idx[row, :row+1] = path

            try:
                node = min(n for vn in visited_nodes for n in _graph.neighbors(vn) if n not in visited_nodes)
            except ValueError:
                if not len(visited_nodes) == num_nodes:
                    raise RuntimeError("Get an incorrect Z-Matirx !!")

        return zmat_idx

    @staticmethod
    def calc_zmat(graph_or_link: Union[nx.Graph, np.ndarray], coords: np.ndarray) -> np.ndarray:
        """
        Computes the Z-matrix representation of molecular coordinates.

        This method computes the Z-matrix representation for a given molecular graph or connectivity
        matrix and a corresponding set of atomic coordinates. The Z-matrix is represented as a 
        NumPy array where each row contains the bond length, bond angle, and dihedral angle for 
        atoms in the molecule.

        Args:
            graph_or_link (Union[nx.Graph, np.ndarray]): The molecular representation either as 
                a networkx.Graph object or an adjacency matrix represented using NumPy.
            coords (np.ndarray): A 2D NumPy array of atomic coordinates with shape (N, 3), 
                where N is the number of atoms.

        Returns:
            np.ndarray: A 2D NumPy array representing the Z-matrix. Each row corresponds to an
                atom in the molecule, and the columns contain the bond length, bond angle, 
                and dihedral angle, respectively, calculated for that atom using its connectivity.
        """
        zmat_idx = InternalCoordinates.calc_zmat_idx(graph_or_link)
        zmat = -1 * np.ones((zmat_idx.shape[0], 3))

        for i, index in enumerate(zmat_idx):
            zmat[i, :i] = InternalCoordinates._calc_zmat_line(index, coords)

        return zmat

    @staticmethod
    def _calc_zmat_line(index: types.ArrayLike, coords: np.ndarray):
        """
            Calculates the Z-matrix parameters for a given set of indices and coordinates.

            This method computes bond length, bond angle, and dihedral angle depending 
            on the number of vectors derived from the input indices. It uses `numpy`
            for vector operations such as calculating norms, dot products, and cross 
            products. Angles are converted from radians to degrees. The resulting 
            parameters represent molecular geometry with respect to the provided indices.

            Args:
                index (types.ArrayLike): Array-like object containing indices that specify 
                atoms connected to the current atom. It must have a length of 4.

                coords (np.ndarray): A 2D numpy array where each row represents the

            Returns:
                list: A list of calculated Z-matrix parameters containing bond length, 
                bond angle, and optionally dihedral angle as applicable.
        """
        assert len(index) == 4

        vectors = [coords[index[i]] - coords[index[i-1]] for i in range(1, 4) if index[i] >= 0]

        results = []
        if len(vectors) >= 1:
            r = np.linalg.norm(vectors[0])
            results.append(r)

        if len(vectors) >= 2:
            r1 = np.linalg.norm(vectors[1])
            cos_a = np.dot(vectors[0], vectors[1]) / (r*r1)
            a = np.degrees(np.arccos(cos_a))
            results.append(a)

        if len(vectors) == 3:
            n1 = np.cross(vectors[0], vectors[1])
            n2 = np.cross(vectors[1], vectors[2])

            norm_n1 = np.linalg.norm(n1)
            norm_n2 = np.linalg.norm(n2)

            if norm_n1 == 0 or norm_n2 == 0:
                cos_dehedral = -1
            else:
                cos_dehedral = round(np.dot(n1, n2) / (norm_n1 * norm_n2), 8)

            dehedral = np.degrees(np.arccos(cos_dehedral))
            results.append(dehedral)

        return results

    @staticmethod
    def calc_angle(a: types.ArrayLike, b: types.ArrayLike, c: types.ArrayLike):
        """
            Calculates the angle defined by three points.

            This method computes the angle in radians at the middle point
            defined by the lines connecting three points `a`, `b`, and `c`.
            The calculation assumes the three points form two vectors
            emanating from the middle point, and the angle is evaluated
            between these two vectors.

            Args:
                a (types.ArrayLike): The coordinates of the first point.
                b (types.ArrayLike): The coordinates of the middle point.
                c (types.ArrayLike): The coordinates of the third point.

            Returns:
                float: The angle in radians between the vectors defined by
                the points.

            Raises:
                ValueError: If the input arrays are of incompatible shapes
                or do not represent valid points.
        """
        return InternalCoordinates._calc_angle(np.array(a), np.array(b), np.array(c))

    @staticmethod
    def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, return_degree: bool = True):
        """
            Calculate the angle formed by three points a, b, and c in a two-dimensional space. 
            The angle is computed using the dot product formula and can be returned in radians 
            or degrees.

            Args:
                a (np.ndarray): The first point in the 2D space.
                b (np.ndarray): The second point in the 2D space.
                c (np.ndarray): The third point in the 2D space.
                return_degree (bool): Whether to return the angle in degrees or radians. 
                    Defaults to True (degrees).

            Returns:
                float: The angle between the vectors ab and bc, in degrees or radians 
                depending on return_degree.

            Raises:
                None: This method does not raise exceptions.
        """
        # Calculate vectors ab and bc
        ab = b - a
        bc = c - b

        # Calculate the dot product and magnitudes
        abc_dot = np.dot(ab, bc)
        norm_ab = np.linalg.norm(ab)
        norm_bc = np.linalg.norm(bc)

        # Calculate the angle in radians
        cos_theta = abc_dot / (norm_ab * norm_bc)
        angle_radians = np.arccos(cos_theta)

        if return_degree:
            return np.degrees(angle_radians) # Convert to degrees
        else:
            return angle_radians

    @staticmethod
    def calc_dehedral(
            a: types.ArrayLike,
            b: types.ArrayLike,
            c: types.ArrayLike,
            d: types.ArrayLike,
            return_degree: bool = True
    ):
        """
        Calculate the dihedral angle formed by four points.

        This static method computes the dihedral angle, measured in degrees or radians,
        for four consecutive points in space. The input points are specified as array-like
        structures. The computation determines the angle formed by the planes defined
        by the four points.

        Parameters:
            a (ArrayLike): The first point in the sequence, given in Cartesian coordinates.
            b (ArrayLike): The second point in the sequence, given in Cartesian coordinates.
            c (ArrayLike): The third point in the sequence, given in Cartesian coordinates.
            d (ArrayLike): The fourth point in the sequence, given in Cartesian coordinates.
            return_degree (bool, optional): Determines the unit of the returned angle.
                If True, the angle is returned in degrees. If False, the angle is returned
                in radians. Defaults to True.

        Returns:
            float: The computed dihedral angle in degrees or radians, based on the value
                of the return_degree parameter.
        """
        return InternalCoordinates._calc_dehedral(np.array(a), np.array(b), np.array(c), np.array(d), return_degree)

    @staticmethod
    def _calc_dehedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, return_degree: bool = True):
        """
            Calculates the dehedral angle formed by four points in 3D space.

            This method computes the dehedral angle between the planes determined by 
            four consecutive points (A, B, C, D). The calculation is performed using 
            the vector cross products of the line segments between these points. The 
            result can be returned in either degrees or radians as specified.

            Args:
                a (np.ndarray): Coordinates of the first point (A), given as a 
                    1-dimensional numpy array.
                b (np.ndarray): Coordinates of the second point (B), given as a 
                    1-dimensional numpy array.
                c (np.ndarray): Coordinates of the third point (C), given as a 
                    1-dimensional numpy array.
                d (np.ndarray): Coordinates of the fourth point (D), given as a 
                    1-dimensional numpy array.
                return_degree (bool): Boolean flag indicating whether the angle should 
                    be returned in degrees (True) or in radians (False). Default is True.

            Returns:
                float: The calculated dehedral angle in degrees (when `return_degree` 
                is True) or radians (when `return_degree` is False).
        """
        # Calculate vectors ab, bc, and cd
        ab = b - a
        bc = c - b
        cd = d - c

        n1 = np.cross(ab, bc)
        n2 = np.cross(bc, cd)

        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)

        if norm_n1 == 0 or norm_n2 == 0:
            cos_dehedral = -1
        else:
            cos_dehedral = np.dot(n1, n2) / (norm_n1*norm_n2)

        dehedral_radians = np.arccos(cos_dehedral)
        if return_degree:
            return np.degrees(dehedral_radians) # Convert to degrees
        else:
            return dehedral_radians


if __name__ == "__main__":
    pass
