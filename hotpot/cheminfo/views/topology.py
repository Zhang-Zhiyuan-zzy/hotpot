"""Fixed ring and joint-ring views over current factual topology."""

from itertools import combinations
from typing import Optional, Union

import numpy as np

from ..structure.atom import Atom
from ..structure.bond import Bond
from ..structure.errors import NotInSameMolecule
from ..structure._topology import _selected_atom_rows
from .sequence import AtomSeq


def _geometry_namespaces():
    from ..operations._carrier_signatures import _geometry_types, _geometry_reports
    return _geometry_types(), _geometry_reports()


geometry, geometry_adapter = _geometry_namespaces()


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
        atoms (tuple): Fixed unique atoms in first-occurrence member order.
        bonds (tuple): Fixed unique bonds in first-occurrence member order.

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
    """
    def __setattr__(self, name, value):
        if '_view_source' in self.__dict__:
            raise AttributeError('JointRing is a structurally immutable view')
        object.__setattr__(self, name, value)

    @property
    def rings(self):
        return self._member_rings

    @property
    def atoms(self):
        return self._atoms

    @property
    def bonds(self):
        return self._bonds

    @property
    def mol(self):
        return self._mol

    def _check_source(self):
        from ._source import _check_ring_source
        _check_ring_source(self, prepare=True)

    def __reduce_ex__(self, protocol):
        from ._source import _restore_joint_ring, _view_source_current
        with self.mol._mutation_lock:
            current = _view_source_current(self.mol, self._view_source, self._atoms, self._view_uids)
            return (_restore_joint_ring, (type(self), self._member_rings, self._atoms, self._bonds,
                     self._view_geometry, self._view_representation, current))

    def __init__(self, *ring):
        from ._source import _bind_view, _ring_representation, _check_ring_source
        mol = ring[0].mol
        with mol._mutation_lock:
            for member in ring:
                if member.mol is not mol:
                    raise NotInSameMolecule(ring[0], member)
                _check_ring_source(member)
            self._member_rings = tuple(ring)
            self._mol = mol
            self._atoms = tuple({id(atom): atom for member in ring for atom in member.atoms}.values())
            self._bonds = tuple({id(bond): bond for member in ring for bond in member.bonds}.values())
            _bind_view(self, self._atoms)
            object.__setattr__(self, '_view_representation', _ring_representation(mol, self._atoms, self._bonds))

    def __contains__(self, item: Union[Atom, Bond, 'Ring']):
        if isinstance(item, Atom):
            return any(member is item for member in self._atoms)
        if isinstance(item, Bond):
            return any(member is item for member in self._bonds)
        if isinstance(item, Ring):
            return any(all(atom in member.atoms for atom in item.atoms) for member in self._member_rings)
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
        with self.mol._mutation_lock:
            self._check_source()
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
        with self.mol._mutation_lock:
            self._check_source()
            return [a for a in atom.neighbours if a in self]

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
        with self.mol._mutation_lock:
            self._check_source()
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
        with self.mol._mutation_lock:
            self._check_source()
            return [b for a in bond.atoms for b in a.bonds if (b is not bond and b in self)]

    def check_kekulize(self):
        from ..operations.chemistry import _check_joint_ring
        return _check_joint_ring(self)



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
        self._check_is_same_mol(*atoms)
        self._check_atom_number(*atoms)
        from ..operations.chemistry import _prepare_aromatic_representation
        from ._source import _bind_view, _ring_representation
        with atoms[0].mol._mutation_lock:
            mol = atoms[0].mol
            _selected_atom_rows(mol, atoms)
            _prepare_aromatic_representation(mol)
            self._atoms = tuple(atoms)
            self._bonds = tuple(mol.bond(first.idx, second.idx)
                                for first, second in zip(atoms, (*atoms[1:], atoms[0])))
            _bind_view(self, atoms)
            object.__setattr__(self, '_view_representation', _ring_representation(mol, atoms, self._bonds))

    def to_mol(self):
        from ._source import _check_ring_source
        with self.mol._mutation_lock:
            _check_ring_source(self, prepare=True)
            return super().to_mol()

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
        from ..operations.chemistry import _ring_aromatic_read
        return _ring_aromatic_read(self)


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
        from ..operations.geometry import _is_disorder
        return _is_disorder(self)

    @property
    def has_metal(self):
        from ._source import _check_ring_source
        with self.mol._mutation_lock:
            _check_ring_source(self)
            return any(atom.is_metal for atom in self._atoms)

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
        from ..operations.topology_queries import _ring_joint_with
        return _ring_joint_with(self, other, Ring)

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
        from ..operations.topology_queries import _joint_ring
        return _joint_ring(self, aromatic, JointRing)

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
        from ..operations.geometry import _ring_has_3d
        return _ring_has_3d(self)


    def relation_to_bond(
            self,
            bond: Bond,
    ) -> "geometry_adapter.BondRingFinding[Ring, Bond]":
        """Return the factual spatial relation between this ring and a bond."""
        from ..operations.geometry import _ring_relation_to_bond
        return _ring_relation_to_bond(self, bond, NotInSameMolecule)

    @property
    def geometry_cycle(self) -> geometry.Cycle:
        """Return this ring as an ordered geometric cycle boundary."""
        from ..operations.geometry import _ring_geometry_cycle
        return _ring_geometry_cycle(self)

    def closest_edge_to_bond(
            self,
            bond: Bond,
    ) -> Optional["geometry_adapter.RingEdgeDistance[Bond]"]:
        """Return the closest ring edge together with its distance evidence."""
        from ..operations.geometry import _closest_ring_edge_to_bond
        return _closest_ring_edge_to_bond(self, bond, NotInSameMolecule)
