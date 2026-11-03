"""The factual Bond carrier and bond-field mutation boundary."""

from copy import deepcopy
from typing import Optional

import numpy as np

from .atom import Atom
from ._fields import MolBlock, ATOM_SOURCE_CODES as _ATOM_SOURCE_CODES
from .bond_kind import BondKind
from .constraints import _bond_constraint_cell, _bond_raw_values
from ._commit import _FactTransaction, _commit_bond_facts
from ._identity import _capture_source
from ._topology import _selected_atom_rows
from ._state import _bond_member_at, _restore_bond_member, _restore_unregistered_bond
from ..views.sequence import AtomSeq


def _geometry_namespace():
    from ..operations._carrier_signatures import _geometry_types
    return _geometry_types()


geometry = _geometry_namespace()


def _bond_kind_from_order(bond_order: float) -> BondKind:
    return {
        1.0: BondKind.SINGLE,
        1.5: BondKind.AROMATIC,
        2.0: BondKind.DOUBLE,
        3.0: BondKind.TRIPLE,
    }.get(float(bond_order), BondKind.UNKNOWN)


def _coerce_bond_kind(value, bond_order: float) -> BondKind:
    if value is None:
        return _bond_kind_from_order(bond_order)
    if isinstance(value, BondKind):
        return value
    try:
        return BondKind(value)
    except ValueError:
        return BondKind[str(value).upper()]


def _bond_order_setter(self: 'Bond', key, bond_order):
    assert key == 'bond_order'
    with _FactTransaction(self.mol, producer=_ATOM_SOURCE_CODES['explicit']):
        current_kind = self._bond_kind
        kind = (_bond_kind_from_order(bond_order) if current_kind in {
            BondKind.SINGLE, BondKind.DOUBLE, BondKind.TRIPLE, BondKind.AROMATIC, BondKind.UNKNOWN,
        } else current_kind)
        values = _raw_bond_values(self)
        values[0] = float(bond_order)
        _commit_bond_facts(self.mol, self, _capture_source(self.mol), raw_values=values,
            kind=kind, direction=self._bond_direction, source_namespace=self._bond_source,
            metadata=self._bond_source_metadata)


class Bond(AtomSeq, MolBlock):
    """
    Represents a chemical bond between two atoms.

    This class models a bond in a molecular structure, providing attributes and methods for representing bond properties
    such as bond order, length, aromaticity, and covalency. It supports operations for determining bond characteristics,
    accessing bond-related geometric information, and interacting with associated atoms and molecular structures.
    """
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
        owner = self.mol
        with owner._mutation_lock:
            _selected_atom_rows(owner, (atom1, atom2))
            kind = kwargs.pop('bond_kind', None)
            direction = kwargs.pop('bond_direction', None)
            source = kwargs.pop('bond_source', None)
            metadata = deepcopy(kwargs.pop('bond_source_metadata', None) or {})
            values = np.array([kwargs.pop(name, 0.) for name in self._attrs_enumerator], dtype=np.float64)
            _initialize_bond_handle(self, owner, owner._allocate_bond_uid(), atom1, atom2, values,
                                    kind=_coerce_bond_kind(kind, values[0]), direction=direction,
                                    source_namespace=source, metadata=metadata)
            with _FactTransaction(owner, producer=_ATOM_SOURCE_CODES['explicit']):
                if kwargs:
                    raise TypeError(f"Unsupported Bond constructor fields: {', '.join(sorted(kwargs))}")

    @property
    def attrs(self):
        with self.mol._mutation_lock:
            return _raw_bond_values(self)

    @property
    def atom1(self) -> Atom:
        with self.mol._mutation_lock:
            return self._atoms[0]

    @property
    def atom2(self) -> Atom:
        with self.mol._mutation_lock:
            return self._atoms[1]

    def __reduce_ex__(self, protocol):
        with self.mol._mutation_lock:
            hint = self._uid
            member = _bond_member_at(self.mol, hint)
            row = hint if member is self else None
            if row is None:
                row = next((i for i, bond in enumerate(self.mol._bonds) if bond is self), None)
            if row is not None:
                return _restore_bond_member, (self.mol, row, self._uid)
            state = {key: value for key, value in self.__dict__.items()
                     if key not in ('_atoms', '_member_uid', '_uid_owner', '_attrs', '_constraint_values')}
            return _restore_unregistered_bond, (self.mol, self._uid, self.atom1, self.atom2, _raw_bond_values(self)), state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _set_bond_metadata(self, *, bond_kind, bond_direction=None, bond_source=None,
                           bond_source_metadata=None):
        metadata = deepcopy(bond_source_metadata or {})
        _commit_bond_facts(self.mol, self, _capture_source(self.mol), raw_values=_raw_bond_values(self),
            kind=_coerce_bond_kind(bond_kind, self.bond_order), direction=bond_direction,
            source_namespace=bond_source, metadata=metadata)

    @property
    def bond_kind(self) -> BondKind:
        with self.mol._mutation_lock:
            return self._bond_kind

    @property
    def bond_direction(self) -> Optional[str]:
        with self.mol._mutation_lock:
            return self._bond_direction

    @property
    def bond_source(self) -> Optional[str]:
        with self.mol._mutation_lock:
            return self._bond_source

    @property
    def bond_source_metadata(self) -> dict:
        with self.mol._mutation_lock:
            return deepcopy(self._bond_source_metadata)

    @property
    def attr_dict(self) -> dict:
        attrs = super().attr_dict
        attrs.update({
            'bond_kind': self.bond_kind,
            'bond_direction': self.bond_direction,
            'bond_source': self.bond_source,
            'bond_source_metadata': self.bond_source_metadata,
        })
        return attrs

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
    def bond_segment(self) -> geometry.Segment:
        """Return the finite geometric segment represented by this bond."""
        from ..operations.geometry import _bond_segment
        return _bond_segment(self)

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
        from ..operations.chemistry import _bond_ideal_bond_length
        return _bond_ideal_bond_length(self)

    @property
    def length(self) -> float:
        """
            Calculates and returns the length of the vector between two atomic positions.
            Returns:
                float: The Euclidean distance (length) between the two atomic vectors.
        """
        from ..operations.geometry import _bond_length
        return _bond_length(self)

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
        from ..operations.chemistry import _bond_aromatic_read
        return _bond_aromatic_read(self)

    @property
    def is_covalent(self) -> bool:
        """
        Determines if the bond is covalent.

        Checks whether any atom in the list of atoms is a metal.
        Returns True if no atoms are metals, indicating a covalent bond.

        @return: True if the bond is covalent, otherwise False
        @rtype: bool
        """
        from ..operations.chemistry import _bond_is_covalent
        return _bond_is_covalent(self)

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
        from ..operations.chemistry import _bond_is_heavy_covalent
        return _bond_is_heavy_covalent(self)

    @property
    def is_metal_ligand_bond(self) -> bool:
        """
        Determine whether the bond is a metal-ligand bond.

        A metal-ligand bond is identified if one atom in the bond is a metal and the
        other is not. This property verifies this condition and returns the result.

        @return:
            A boolean indicating if the bond is a metal-ligand bond or not.
        """
        from ..operations.chemistry import _bond_is_metal_ligand_bond
        return _bond_is_metal_ligand_bond(self)

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
        from ..operations.chemistry import _bond_rotatable
        return _bond_rotatable(self)

    def bond_segment_distance(self, other_bond: "Bond") -> float:
        """Return the minimum distance between two finite bond segments."""
        from ..operations.geometry import _bond_segment_distance
        return _bond_segment_distance(self, other_bond)


def _raw_bond_values(bond):
    return _bond_raw_values(bond._attrs, bond._constraint_values[bond._uid])


def _detach_bond(bond):
    object.__setattr__(bond, '_constraint_values', {bond._uid: bond._constraint_values[bond._uid]})


def _store_bond_field(bond, field, value):
    with _FactTransaction(bond.mol, producer=_ATOM_SOURCE_CODES['explicit']):
        values = _raw_bond_values(bond)
        values[bond._attrs_enumerator.index(field)] = float(value)
        _commit_bond_facts(bond.mol, bond, _capture_source(bond.mol), raw_values=values,
            kind=bond._bond_kind, direction=bond._bond_direction,
            source_namespace=bond._bond_source, metadata=bond._bond_source_metadata)


def _initialize_bond_handle(bond, mol, uid, atom1, atom2, raw_values, *, kind,
                            direction, source_namespace, metadata, aromatic=None):
    object.__setattr__(bond, '_atoms', (atom1, atom2))
    object.__setattr__(bond, '_bonds', [bond])
    object.__setattr__(bond, '_member_uid', uid)
    object.__setattr__(bond, '_uid_owner', mol._owner_token)
    raw_values = np.asarray(raw_values, dtype=np.float64)
    object.__setattr__(bond, '_attrs', raw_values[[0, 2]])
    object.__setattr__(bond, '_constraint_values', {uid: _bond_constraint_cell(raw_values[1])})
    object.__setattr__(bond, '_bond_kind', BondKind(kind))
    object.__setattr__(bond, '_bond_direction', direction)
    object.__setattr__(bond, '_display_facts', ())
    object.__setattr__(bond, '_bond_source', source_namespace)
    object.__setattr__(bond, '_bond_source_metadata', metadata)
    object.__setattr__(bond, '_is_aromatic', kind == 'aromatic' if aromatic is None else bool(aromatic))


def _new_bond_handle(mol, uid, atom1, atom2, raw_values, *, kind, direction,
                     source_namespace, metadata, aromatic=None):
    bond = object.__new__(Bond)
    _initialize_bond_handle(bond, mol, uid, atom1, atom2, raw_values, kind=kind,
                            direction=direction, source_namespace=source_namespace, metadata=metadata, aromatic=aromatic)
    return bond
