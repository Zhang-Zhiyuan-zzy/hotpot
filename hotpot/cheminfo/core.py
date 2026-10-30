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
import operator
from enum import Enum
from typing import Union, Literal, Iterable, Optional, Callable, Mapping
from copy import copy, deepcopy
from itertools import combinations, product
from threading import RLock as _RLock
from functools import wraps as _wraps

import cython
import numpy as np
from openbabel import pybel as pb, openbabel as ob

from hotpot.cheminfo.elements import elements
from hotpot.utils import types
import hotpot.cheminfo.obconvert as obc
from .rdconvert import to_rdmol
from .rules.mathematics import graph as graph_algorithms
from . import forcefields as ff
from .rules.mathematics import geometry
from .adapters.structure import geometry as geometry_adapter
from . import crystal as cryst
from .pubchem import pubchem_service
from .call_thermo import Thermo
from .structure._fields import (
    ATOM_FIELDS as _ATOM_FIELDS, ATOM_PRESENCE_ALL as _ATOM_PRESENCE_ALL,
    ATOM_SOURCE_CODES as _ATOM_SOURCE_CODES, FIELD_INDEX as _ATOM_FIELD_INDEX,
    FIELD_ADDRESSES as _ATOM_FIELD_ADDRESSES,
    FIELD_CODECS as _ATOM_FIELD_CODECS, FACT_FIELDS as _ATOM_FACT_FIELDS,
    DERIVED_FIELDS as _ATOM_DERIVED_FIELDS,
    ATOM_SCALAR_READONLY as _ATOM_SCALAR_READONLY,
)
from .structure._atom_table import _AtomFactTable
from .structure.constraints import _ConstraintSet
from .structure._identity import _capture_source, _hydrogen_install_inputs
from .structure._mutation import _AtomFactUpdate, _normalize_atom_facts
from .structure._commit import (
    _FactTransaction, _commit_atom_facts, _commit_coordinates, _commit_bond_facts,
    _commit_molecular_declarations, _commit_atom_sparse, _commit_topology,
    _current_transaction, _fact_engine, _commit_constraints,
)
from .structure._dependencies import (
    _dependency_state, _install_dependency_state, _remap_dependency_state,
    _mark_topology_dirty, _seed, GROUP_INDEX,
)
from .stereo import StereoPayload as _StereoPayload


PartialChargeModel = Literal[
    "eem", "eem2015ha", "eem2015hm", "eem2015hn",
    "eem2015ba", "eem2015bm", "eem2015bn",
    "gasteiger", "mmff94", "qeq", "qtpie", "eqeq",
    "fromfile", "none",
]

OPENBABEL_PARTIAL_CHARGE_MODELS = (
    "eem", "eem2015ha", "eem2015hm", "eem2015hn",
    "eem2015ba", "eem2015bm", "eem2015bn",
    "gasteiger", "mmff94", "qeq", "qtpie", "eqeq",
    "fromfile", "none",
)

# Chemistry policy deliberately kept outside the factual geometry package.
_MINIMUM_ATOM_SEPARATION = 0.5
_DEFAULT_BOND_RING_MAXIMUM_SIZE = 16

if sys.modules.get('hotpot.cheminfo._io', None) is None:
    from . import _io
else:
    _io = sys.modules['hotpot.cheminfo._io']

# Exceptions
class NotInSameMolecule(Exception):
    """ The exception is toggled when two objects are expected in a same Molecule, but not """
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2
        self.message = f"The {type(obj1).__name__} and {type(obj2).__name__} must belong to the same Molecule"
        super().__init__(self.message)
        
class ObjNotInMolecule(Exception):
    """ The exception is toggled when the given object not in a specific Molecule. """
    def __init__(self, obj, mol):
        self.obj = obj
        self.mol = mol
        self.message = f"The {type(obj).__name__} is not a current member of the Molecule"
        super().__init__(self.message)


class BondKind(str, Enum):
    """Semantic bond categories stored separately from numeric bond order."""

    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    AROMATIC = "aromatic"
    ZERO = "zero"
    DATIVE = "dative"
    UNKNOWN = "unknown"


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
        self._owner_token = object()
        self._mutation_lock = _RLock()
        self._next_atom_uid = 0
        self._next_bond_uid = 0
        self._model = None  # AI model
        self._environ = None

        self._atoms = []
        self._atom_table = _AtomFactTable()
        self._bonds = []
        self._topology_index = {}
        self._conformers = Conformers()
        self._conformers_index = 0

        self._pair_wiberg_bond_orders = {}
        self._constraints = _ConstraintSet()
        self._obmol = None
        self._row2idx = None
        self._crystal = None

        self._hided_metal_bonds = []
        self._hided_covalent_bonds = []
        self.properties = {}  # To store any mol properties

        self._stereo_payload = _StereoPayload(1, (), (), ())

        self._charge_value, self._charge_present, self._charge_source = 0, False, None
        self._spin_value, self._spin_present, self._spin_source = None, False, None

    def __dir__(self):
        return list(Conformers._attrs) + list(super(Molecule, self).__dir__())

    def __getattr__(self, item):
        if '_conformers' in self.__dict__ and item in Conformers._attrs:
            with self._mutation_lock:
                return self._conformers.index_attr(item, self._conformers_index)
        return object.__getattribute__(self, item)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.formula})"

    def __add__(self, other):
        clone = copy(self)
        clone.add_component(other)
        return clone

    def __copy__(self):
        return self.copy()

    def __getstate__(self):
        return _capture_molecule_pickle(self)

    def __setstate__(self, state):
        _restore_molecule_extra(self, state)

    def __reduce_ex__(self, protocol):
        record, owner, atom_uids, bond_uids, high_water, dependencies, state = self.__getstate__()
        return (_restore_molecule_raw,
                (type(self), record, owner, atom_uids, bond_uids, high_water, dependencies), state)

    @property
    def charge(self):
        with self._mutation_lock:
            return self._charge_value

    @charge.setter
    def charge(self, value):
        _commit_molecular_declarations(self, _capture_source(self), charge=(True, value, 'explicit'))

    @property
    def spin_mult(self):
        with self._mutation_lock:
            if self._spin_present:
                return self._spin_value
            return self._conformers.index_attr('spin_mult', self._conformers_index)

    @spin_mult.setter
    def spin_mult(self, value):
        _commit_molecular_declarations(self, _capture_source(self), spin_mult=(True, value, 'explicit'))

    def updata_atoms(self, *, atoms=None, **columns) -> None:
        selected = None if atoms is None else tuple(atoms)
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']):
            _atom_fact_engine()._commit_atom_columns(self, selected, columns)

    @classmethod
    def _build_molecule(cls, record, *, bond_metadata=None):
        return cls()._install_molecule(record, bond_metadata=bond_metadata)

    def _install_molecule(self, record, *, bond_metadata=None):
        _commit_topology(self, _capture_source(self), record,
                         atom_bindings=(None,) * record.atom_count,
                         bond_bindings=(None,) * len(record.bond_endpoints),
                         bond_metadata=bond_metadata)
        return self

    def _allocate_atom_uid(self):
        with self._mutation_lock:
            uid = self._next_atom_uid
            self._next_atom_uid += 1
            return uid

    def _allocate_bond_uid(self):
        with self._mutation_lock:
            uid = self._next_bond_uid
            self._next_bond_uid += 1
            return uid

    @property
    def _hided_bonds(self):
        return self._hided_metal_bonds + self._hided_covalent_bonds

    def _structural_bond_rows(self) -> tuple[tuple[int, int], ...]:
        atom_rows = {id(atom): row for row, atom in enumerate(self._atoms)}
        return tuple(
            (atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)])
            for bond in self._bonds + self._hided_bonds
        )

    def _set_remapped_stereo_payload(
            self,
            source: "Molecule",
            atom_mapping: Mapping[int, int],
            *,
            filter_bonds: bool,
    ) -> None:
        from .stereo import (
            _remap_stereo_payload,
            get_stereo_payload,
            set_stereo_payload,
        )

        allowed_bonds = self._structural_bond_rows() if filter_bonds else None
        set_stereo_payload(
            self,
            _remap_stereo_payload(
                get_stereo_payload(source),
                atom_mapping,
                allowed_bonds=allowed_bonds,
            ),
        )

    def _replace_atom_transport(self, target_row, source_atom):
        """Install one imported Atom transport row at its new member position."""
        from dataclasses import replace
        from .stereo import get_stereo_payload, set_stereo_payload

        source_row = source_atom.mol._topology_index.get(id(source_atom))
        imported = () if source_row is None else tuple(
            replace(fact, atom=target_row)
            for fact in get_stereo_payload(source_atom.mol).atom_facts
            if fact.atom == source_row[0]
        )
        payload = get_stereo_payload(self)
        atom_facts = []
        installed = False
        for fact in payload.atom_facts:
            if fact.atom == target_row:
                if not installed:
                    atom_facts.extend(imported)
                    installed = True
            else:
                atom_facts.append(fact)
        if not installed:
            atom_facts.extend(imported)
        set_stereo_payload(self, replace(payload, atom_facts=tuple(atom_facts)))

    def _remap_stereo_after_atom_removal(
            self,
            source_atoms: tuple["Atom", ...],
    ) -> None:
        destination_rows = {
            id(atom): row for row, atom in enumerate(self._atoms)
        }
        atom_mapping = {
            source_row: destination_rows[id(atom)]
            for source_row, atom in enumerate(source_atoms)
            if id(atom) in destination_rows
        }
        self._set_remapped_stereo_payload(
            self,
            atom_mapping,
            filter_bonds=False,
        )

    def _remove_stereo_bond_rows(
            self,
            removed_bond_rows: Iterable[tuple[int, int]],
    ) -> None:
        from .stereo import (
            StereoAnchorKind,
            _remap_stereo_payload,
            get_stereo_payload,
            set_stereo_payload,
        )

        payload = get_stereo_payload(self)
        removed_keys = {
            frozenset(atom_indices) for atom_indices in removed_bond_rows
        }
        allowed_bonds = list(self._structural_bond_rows())
        for record in payload.records:
            if record.anchor.kind is StereoAnchorKind.BOND:
                if frozenset(record.anchor.atom_indices) not in removed_keys:
                    allowed_bonds.append(record.anchor.atom_indices)
            elif record.anchor.kind is StereoAnchorKind.AXIS:
                constituent_bonds = tuple(zip(
                    record.anchor.atom_indices,
                    record.anchor.atom_indices[1:],
                ))
                if not any(
                    frozenset(atom_indices) in removed_keys
                    for atom_indices in constituent_bonds
                ):
                    allowed_bonds.extend(constituent_bonds)
        allowed_bonds.extend(
            fact.atom_indices
            for fact in payload.bond_facts
            if frozenset(fact.atom_indices) not in removed_keys
        )
        set_stereo_payload(
            self,
            _remap_stereo_payload(
                payload,
                {row: row for row in range(len(self._atoms))},
                allowed_bonds=allowed_bonds,
            ),
        )

    def copy(self):
        with self._mutation_lock:
            record, metadata = self._capture_factual_record(include_frames=False)
            dependencies = _remap_dependency_state(self._atom_table,
                tuple(atom._uid for atom in self._atoms), tuple(range(len(self._atoms))))
            annotations = dict(self.properties)
        clone = Molecule()
        _commit_topology(clone, _capture_source(clone), record,
                         atom_bindings=(None,) * record.atom_count,
                         bond_bindings=(None,) * len(record.bond_endpoints),
                         dependency_state=dependencies, bond_metadata=metadata)
        clone.properties = annotations
        return clone

    @property
    def crystal(self):
        return self._crystal

    def create_crystal(self, a, b, c, alpha, beta, gamma):
        self._crystal = cryst.Crystal(a, b, c, alpha, beta, gamma, mol=self)
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
        from .operations.chemistry import _require_implicit_hydrogens
        return sum(_require_implicit_hydrogens(self, rows=tuple(atom.idx for atom in self.heavy_atoms)))

    @property
    def has_hydrogens(self) -> bool:
        """
        Indicates whether the object contains any hydrogens.
        @return: Returns True if there are hydrogens, False otherwise.
        """
        return len(self.hydrogens) > 0

    def calc_mol_default_charge(self):
        from .operations.chemistry import _calculate_default_molecular_charge
        return _calculate_default_molecular_charge(self)



    def simple_paths(self, source=None, target=None, cutoff=None):
        from .operations.topology_queries import _simple_paths
        return _simple_paths(self, source, target, cutoff)

    def _set_coordinates(self, coordinates):
        self.coordinates = np.asarray(coordinates)

    def _update_topology(self, clear_conformers: bool = True, *,
                         _prepared_index=None, _constraint_rows=None) -> None:
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            if _prepared_index is self._topology_index and _constraint_rows is None:
                transaction.touch_caches()
            else:
                transaction.touch_topology()
            previous = self._topology_index
            self._topology_index = (_build_topology_index(
                self._atoms, self._bonds, self._hided_metal_bonds, self._hided_covalent_bonds)
                if _prepared_index is None else _prepared_index)
            if _constraint_rows is not None:
                angle_rows, torsion_rows = _constraint_rows
                self._constraints = _ConstraintSet(
                    tuple(tuple(self._atoms[row]._uid for row in entry) for entry in angle_rows),
                    tuple(tuple(self._atoms[row]._uid for row in entry) for entry in torsion_rows))
            elif self._topology_index is not previous:
                uids = {atom._uid for atom in self._atoms}
                self._constraints = self._constraints.subset(uids)
                if any(first not in uids or second not in uids for first, second in self._pair_wiberg_bond_orders):
                    self._pair_wiberg_bond_orders = {
                        pair: value for pair, value in self._pair_wiberg_bond_orders.items()
                        if all(uid in uids for uid in pair)}
                changed = tuple(atom._uid for atom in self._atoms if
                    tuple(id(bond) for bond in previous.get(id(atom), (None, ()))[1]) !=
                    tuple(id(bond) for bond in self._topology_index[id(atom)][1]))
                _mark_topology_dirty(self, changed)
            self._obmol = None
            self._row2idx = None
            if clear_conformers:
                self._conformers.clear()


    def _add_atom(self, atom):
        if atom.mol is self:
            raise ValueError('An Atom already owned by this Molecule cannot be inserted again')
        with atom.mol._mutation_lock:
            source_row = atom.mol._topology_index.get(id(atom))
            table = atom._detached_table if source_row is None else atom.mol._atom_table
            row = 0 if source_row is None else source_row[0]
            values, presence, sources = table.row(row), table.presence(row), table.sources(row)
            statuses = table._group_status[row].copy()
            pending_groups = tuple(atom._uid in seeds for seeds in table._dirty_seeds)
            sparse = {name: deepcopy(atom.__dict__[name]) for name in ('isotope', 'frac_valence') if name in atom.__dict__}
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            copied = _import_atom_row(self, values, presence=presence, sources=sources)
            new_row = self._topology_index[id(copied)][0]
            transaction.touch_dependency_status((new_row,), range(len(statuses)))
            self._atom_table._group_status[new_row] = statuses
            for group, pending_group in enumerate(pending_groups):
                _atom_fact_engine()._update_dependency_seeds(self, (group,), (copied._uid,), pending_group)
            for name, value in sparse.items():
                setattr(copied, name, value)
        return copied

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
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            atom1 = atom1 if isinstance(atom1, Atom) else self._atoms[atom1]
            atom2 = atom2 if isinstance(atom2, Atom) else self._atoms[atom2]

            if atom1.mol is not self or atom2.mol is not self:
                raise ValueError("at least one of atom1 and atom2 not on the molecule!")

            if atom1 is atom2:
                raise ValueError("a Molecule Bond cannot join an Atom to itself")

            first_record = self._topology_index[id(atom1)]
            second_record = self._topology_index[id(atom2)]
            incident = (
                first_record[1]
                if len(first_record[1]) <= len(second_record[1])
                else second_record[1]
            )
            if any(
                (bond.atom1 is atom1 and bond.atom2 is atom2)
                or (bond.atom1 is atom2 and bond.atom2 is atom1)
                for bond in incident
            ):
                raise ValueError(f"the bond between {atom1.idx} and {atom2.idx} is already exist!!")

            kwargs['bond_order'] = bond_order
            bond = Bond(atom1, atom2, **kwargs)
            transaction.touch_bond_membership((atom1, atom2))
            self._bonds.append(bond)
            _mark_topology_dirty(self, (atom1._uid, atom2._uid))

            for atom, record in ((atom1, first_record), (atom2, second_record)):
                bonds = tuple(sorted(
                    record[1] + (bond,),
                    key=lambda member: self._topology_index[
                        id(member.atom2 if member.atom1 is atom else member.atom1)
                    ][0],
                ))
                self._topology_index[id(atom)] = (record[0], bonds)

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
        source_atom = atom
        atom = self._add_atom(atom)
        self._update_topology()
        self._replace_atom_transport(atom.idx, source_atom)
        return atom

    def _replace_atom(self, original_atom_idx: int, new_atom: Union[int, str, 'Atom']):
        if isinstance(new_atom, int):
            new_atom = Atom(atomic_number=new_atom)
        elif isinstance(new_atom, str):
            new_atom = Atom(symbol=new_atom)
        elif not isinstance(new_atom, Atom):
            raise ValueError(f'Unsupported atom type {type(new_atom)}')
        original = self._atoms[original_atom_idx]
        target_row = self._topology_index[id(original)][0]
        source_atom = new_atom
        source_members = tuple(self._atoms)
        replacement = self._add_atom(new_atom)
        old_uids = tuple(atom._uid for atom in self._atoms)
        selection = tuple(len(self._atoms) - 1 if row == target_row else row
                          for row in range(len(self._atoms) - 1))
        next_atoms = list(source_members)
        next_atoms[target_row] = replacement
        state = _remap_dependency_state(self._atom_table, old_uids,
                                        tuple(atom._uid for atom in next_atoms), rows=selection)
        _detach_atom(original)
        table = self._atom_table.take(selection)
        _install_dependency_state(table, state)
        self._atom_table = table
        self._atoms[:] = next_atoms
        for bond in self._bonds + self._hided_bonds:
            if bond.atom1 is original or bond.atom2 is original:
                Bond._replace_atom(bond, original, replacement)
        self._topology_index = _build_topology_index(self._atoms, self._bonds,
                                                    self._hided_metal_bonds, self._hided_covalent_bonds)
        self._replace_atom_transport(target_row, source_atom)
        return original

    def build_all_pair_links(
            self,
            metal,
            threshold: Optional[float] = None,
            greedy=True,
            *,
            first_threshold: Optional[float] = None,
    ):
        from .AImodels.cbond.apply import build_all_possible_cbond
        return build_all_possible_cbond(
            self,
            metal,
            threshold,
            greedy,
            first_threshold=first_threshold,
        )

    def auto_pair_metal(
            self,
            metal,
            threshold: Optional[float] = None,
            greedy=True,
            probability: bool = False,
            *,
            first_threshold: Optional[float] = None,
    ) -> Union['Molecule', tuple['Molecule', float]]:
        from .AImodels.cbond.apply import auto_build_cbond
        if probability:
            return auto_build_cbond(
                self,
                metal,
                threshold,
                greedy,
                first_threshold=first_threshold,
            )
        else:
            return auto_build_cbond(
                self,
                metal,
                threshold,
                greedy,
                first_threshold=first_threshold,
            )[0]

    def replace_atom(
            self,
            original_atom_idx: int,
            new_atom: Union[int, str, "Atom"],
    ):
        """ Replaces an atom in the molecule to the other external, while inherit the environment of original one."""
        ori_atom = self._replace_atom(original_atom_idx, new_atom)
        self._update_topology()
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
        self._update_topology()
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

        self._update_topology()
        return bonds

    def add_component(self, component: 'Molecule'):
        from dataclasses import replace
        from .stereo import _remap_stereo_payload
        with component._mutation_lock:
            incoming, incoming_metadata = component._capture_factual_record(include_frames=False)
            incoming_dependencies = _dependency_state(component._atom_table)
            incoming_uids = tuple(atom._uid for atom in component._atoms)
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            transaction.touch_topology()
            current, current_metadata = self._capture_factual_record(include_frames=False)
            offset = current.atom_count
            if offset and incoming.cell is not None and current.cell != incoming.cell:
                raise ValueError('Components require compatible unit cells')
            charge = incoming.charge if not offset else (
                (True, current.charge[1] + incoming.charge[1], 'component_merge')
                if current.charge[0] and incoming.charge[0] else (False, None, None))
            spin = incoming.spin_mult if not offset else (False, None, None)
            payload = _remap_stereo_payload(incoming.stereo_payload,
                        {row: offset + row for row in range(incoming.atom_count)})
            payload = type(payload)(current.stereo_payload.schema_version,
                                    current.stereo_payload.records + payload.records,
                                    current.stereo_payload.atom_facts + payload.atom_facts,
                                    current.stereo_payload.bond_facts + payload.bond_facts)
            record = replace(current, atom_count=offset + incoming.atom_count,
                atom_values=current.atom_values + incoming.atom_values,
                atom_presence=current.atom_presence + incoming.atom_presence,
                atom_sources=current.atom_sources + incoming.atom_sources,
                atom_isotopes=current.atom_isotopes + tuple((offset + row, value) for row, value in incoming.atom_isotopes),
                atom_frac_valences=current.atom_frac_valences + tuple((offset + row, value) for row, value in incoming.atom_frac_valences),
                bond_endpoints=current.bond_endpoints + tuple((offset + a, offset + b) for a,b in incoming.bond_endpoints),
                bond_values=current.bond_values + incoming.bond_values,
                bond_partitions=current.bond_partitions + incoming.bond_partitions,
                bond_kinds=current.bond_kinds + incoming.bond_kinds,
                bond_aromatic=current.bond_aromatic + incoming.bond_aromatic,
                bond_directions=current.bond_directions + incoming.bond_directions,
                bond_sources=current.bond_sources + incoming.bond_sources,
                bond_metadata=current.bond_metadata + incoming.bond_metadata,
                charge=charge, spin_mult=spin, cell=current.cell if offset else incoming.cell,
                stereo_payload=payload,
                angle_constraints=current.angle_constraints + tuple(tuple(offset + row for row in item) for item in incoming.angle_constraints),
                torsion_constraints=current.torsion_constraints + tuple(tuple(offset + row for row in item) for item in incoming.torsion_constraints),
                pair_wiberg_bond_orders=current.pair_wiberg_bond_orders + tuple(
                    (offset + first, offset + second, value)
                    for first, second, value in incoming.pair_wiberg_bond_orders))
            old_atoms = tuple(self._atoms)
            old_bonds = tuple(self._bonds + self._hided_bonds)
            source_status, source_seeds = incoming_dependencies
            uid_map = dict(zip(incoming_uids, range(self._next_atom_uid, self._next_atom_uid + incoming.atom_count)))
            target_status, target_seeds = _dependency_state(self._atom_table)
            dependencies = (np.vstack((target_status, source_status)),
                tuple(frozenset(target) | frozenset(uid_map[uid] for uid in source if uid in uid_map)
                      for target, source in zip(target_seeds, source_seeds)))
            # Bindings follow the record's concatenated complete Bond order.
            _commit_topology(self, _capture_source(self), record,
                atom_bindings=old_atoms + (None,) * incoming.atom_count,
                bond_bindings=old_bonds + (None,) * len(incoming.bond_endpoints),
                dependency_state=dependencies, bond_metadata=current_metadata + incoming_metadata)

    def add_hydrogens(
            self,
            rm_polar_hs: bool = True,
            rng: Optional[np.random.Generator] = None,
    ):
        from .operations.chemistry import _add_hydrogens
        return _add_hydrogens(self, rm_polar_hs, rng)

    @property
    def cid(self) -> Optional[int]:
        return pubchem_service.smi_to_cid(self.smiles)

    @property
    def name(self) -> str:
        return pubchem_service.smi_to_name(self.smiles)

    @property
    def cas(self):
        return pubchem_service.smi_to_cas(self.smiles)

    def get_thermo(self, temp: float = 298.15, pressure: float = 101325.0) -> Thermo:
        return Thermo(self, temp=temp, pressure=pressure)

    def clear_constraints(self) -> None:
        """Clear the existing active coordinate/Bond and owned sequence constraints."""
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            transaction.touch_constraints()
            for atom in self.atoms:
                atom.constraint = False
            for bond in self.bonds:
                bond.constraint = False
            self._constraints = _ConstraintSet()

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
        if self._hided_metal_bonds:
            removed_bond_rows = tuple(
                (bond.a1idx, bond.a2idx)
                for bond in self._hided_metal_bonds
            )
            self._hided_metal_bonds = []
            self._remove_stereo_bond_rows(removed_bond_rows)

    def clear_hided_covalent_bonds(self) -> None:
        if self._hided_covalent_bonds:
            removed_bond_rows = tuple(
                (bond.a1idx, bond.a2idx)
                for bond in self._hided_covalent_bonds
            )
            self._hided_covalent_bonds = []
            self._remove_stereo_bond_rows(removed_bond_rows)

    def clear_hided_bonds(self) -> None:
        if self._hided_bonds:
            removed_bond_rows = tuple(
                (bond.a1idx, bond.a2idx) for bond in self._hided_bonds
            )
            self._hided_metal_bonds = []
            self._hided_covalent_bonds = []
            self._remove_stereo_bond_rows(removed_bond_rows)

    def restore_bonds(
            self,
            *bonds: "Bond",
            clear_conformers: bool = False,
    ) -> None:
        """Restore selected hidden bonds and refresh molecular topology once."""
        if not bonds:
            return

        bond_ids = tuple(id(bond) for bond in bonds)
        if len(set(bond_ids)) != len(bond_ids):
            raise ValueError("A hidden bond can only be restored once per call")

        hidden_bond_ids = {id(bond) for bond in self._hided_bonds}
        for bond, bond_id in zip(bonds, bond_ids):
            if bond_id not in hidden_bond_ids:
                raise ObjNotInMolecule(bond, self)

        restored_bond_ids = set(bond_ids)
        self._hided_metal_bonds = [
            bond for bond in self._hided_metal_bonds
            if id(bond) not in restored_bond_ids
        ]
        self._hided_covalent_bonds = [
            bond for bond in self._hided_covalent_bonds
            if id(bond) not in restored_bond_ids
        ]
        self._bonds.extend(bonds)
        self._update_topology(clear_conformers)

    def recover_hided_metal_ligand_bonds(self, clear_conformers: bool = False) -> None:
        """
        Restores metal-ligand bonds that were previously broken and updates the internal graph representation.

        This function recovers all previously broken metal-ligand bonds by adding them back
        to the existing bond list. Duplicate bonds are removed without changing their order.
        Additionally, it updates the internal graph representation to reflect the recovered bonds 
        and provides the option to clear conformer data. After recovery, the list of broken metal bonds 
        is reset.

        Arguments:
            clear_conformers (bool, optional): Determines whether existing conformer data should 
                                               be cleared during the graph update. Defaults to False.

        Returns:
            None
        """
        hidden_bonds = tuple(self._hided_metal_bonds)
        if hidden_bonds:
            self.restore_bonds(*hidden_bonds, clear_conformers=clear_conformers)
            logging.info(f"[green]Recover {len(hidden_bonds)} hided metal-ligand bonds[/]")

    def recover_hided_covalent_bonds(self, clear_conformers: bool = False) -> None:
        hidden_bonds = tuple(self._hided_covalent_bonds)
        if hidden_bonds:
            self.restore_bonds(*hidden_bonds, clear_conformers=clear_conformers)
            logging.info(f"[green]Recover {len(hidden_bonds)} hided covalent bonds[/]")

    @property
    def atom_pairs(self) -> "AtomPairs":
        """Return a fresh fixed pair table over the current molecular source."""
        return AtomPairs(mol=self)

    @property
    def angles(self) -> list["Angle"]:
        """Return fresh source-bound views; persisted constraints have one owner."""
        from .operations.geometry import _angle_views
        return _angle_views(self, Angle, _new_sequence_view)

    @property
    def torsions(self) -> list["Torsion"]:
        """Return fresh source-bound torsion views without retaining a cache."""
        from .operations.geometry import _torsion_views
        return _torsion_views(self, Torsion, _new_sequence_view)

    def assign_bond_order(self):
        from .operations.chemistry import _assign_bond_orders
        return _assign_bond_orders(self)

    @property
    def atom_matrix(self) -> np.ndarray:
        from .operations.chemistry import _atom_projection
        return _atom_projection(self)

    @property
    def atoms(self):
        with self._mutation_lock:
            return list(self._atoms)

    @property
    def c_bonds(self) -> list["Bond"]:
        """ Get all coordination bonds in the Molecule """
        return [b for b in self.bonds if b.is_metal_ligand_bond]

    @property
    def bonds(self) -> list["Bond"]:
        with self._mutation_lock:
            return list(self._bonds)

    def bond(self, a1idx: int, a2idx: int):
        """Return the active Bond joining two Atom indices."""
        with self._mutation_lock:
            first = self._atoms[a1idx]
            second = self._atoms[a2idx]
            first_bonds = self._topology_index[id(first)][1]
            second_bonds = self._topology_index[id(second)][1]
            incident = (
                first_bonds
                if len(first_bonds) <= len(second_bonds)
                else second_bonds
            )
            for bond in incident:
                if (
                    (bond.atom1 is first and bond.atom2 is second)
                    or (bond.atom1 is second and bond.atom2 is first)
                ):
                    return bond
            raise KeyError((a1idx, a2idx))

    def build3d(
            self,
            forcefield: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']] = 'UFF',
            *,
            algorithm: Literal["steepest", "conjugate"] = "conjugate",
            epochs: int = 100,
            steps_per_epoch: int = 100,
            add_hydrogens: bool = True,
            quality_level: Literal["off", "basic", "standard", "strict"] = "standard",
            quality_thresholds: Optional["ff.StructureAcceptanceThresholds"] = None,
            seed: Optional[int] = None,
            timeout: float = 1000.0,
            perturb_interval: Optional[int] = None,
            perturb_sigma: float = 0.5,
            stopping_criteria: Optional["ff.OptimizationStoppingCriteria"] = None,
            convergence_level: "ff.ConvergenceLevel" = (
                ff.DEFAULT_CONVERGENCE_LEVEL
            ),
            save_movie: bool = False,
            trajectory_start: Optional["ff.TrajectoryStart"] = None,
            trajectory_path: Optional["ff.TrajectoryPath"] = None,
            increasing_vdw: bool = False,
            vdw_cutoff_start: float = 0.0,
            vdw_cutoff_end: float = 12.5,
            candidate_count: Optional[int] = None,
            max_attempts: int = 50,
            candidate_warmup_steps: int = 500,
            candidate_score_steps: int = 1000,
            best_candidate_refine_steps: int = 3000,
            ligand_untangling_attempts: int = 20,
            coordination_restoration_attempts: int = 20,
            coordination_relaxation_steps: int = 100,
            complex_untangling_attempts: int = 30,
            coordination_geometry: Optional[str] = None,
    ):
        """Build and optimize this molecule with the appropriate FF workflow.

        ``candidate_count`` is reserved and currently has no effect.  Complex
        building uses one ligand starting geometry.  Future multi-conformer
        support will generate independent starts, optimize and gate each one,
        deduplicate or cluster them by geometry, rank them by topology,
        geometry, and energy evidence, and refine the selected conformer.
        """
        return ff.build_and_optimize(
            self,
            forcefield=forcefield,
            algorithm=algorithm,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            add_hydrogens=add_hydrogens,
            quality_level=quality_level,
            quality_thresholds=quality_thresholds,
            seed=seed,
            timeout=timeout,
            perturb_interval=perturb_interval,
            perturb_sigma=perturb_sigma,
            stopping_criteria=stopping_criteria,
            convergence_level=convergence_level,
            save_movie=save_movie,
            trajectory_start=trajectory_start,
            trajectory_path=trajectory_path,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
            candidate_count=candidate_count,
            max_attempts=max_attempts,
            candidate_warmup_steps=candidate_warmup_steps,
            candidate_score_steps=candidate_score_steps,
            best_candidate_refine_steps=best_candidate_refine_steps,
            ligand_untangling_attempts=ligand_untangling_attempts,
            coordination_restoration_attempts=coordination_restoration_attempts,
            coordination_relaxation_steps=coordination_relaxation_steps,
            complex_untangling_attempts=complex_untangling_attempts,
            coordination_geometry=coordination_geometry,
        )

    def update_mol_charge(self):
        """
        Updates the molecular charge by summing the charges of all constituent
        atoms. This method assumes the charges of individual atoms have been 
        correctly assigned beforehand.

        """
        self.charge = self.sum_atoms_charge

    @property
    def sum_atoms_charge(self) -> int:
        from .operations.chemistry import _atom_charge_sum
        return _atom_charge_sum(self)

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
        from .operations.chemistry import _assign_aromaticity
        return _assign_aromaticity(self, kekule=True)

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
        coordinates = self._conformers.index_attr('coordinates', i)
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['conformer_load']) as transaction:
            transaction.touch_frames()
            self.coordinates = coordinates
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
            *,
            algorithm: Literal["steepest", "conjugate"] = "conjugate",
            epochs: int = 100,
            steps_per_epoch: int = 100,
            complex_untangling_attempts: int = 30,
            add_hydrogens: bool = True,
            quality_level: Literal["off", "basic", "standard", "strict"] = "standard",
            quality_thresholds: Optional["ff.StructureAcceptanceThresholds"] = None,
            seed: Optional[int] = None,
            perturb_interval: Optional[int] = None,
            perturb_sigma: float = 0.5,
            stopping_criteria: Optional["ff.OptimizationStoppingCriteria"] = None,
            convergence_level: "ff.ConvergenceLevel" = (
                ff.DEFAULT_CONVERGENCE_LEVEL
            ),
            save_movie: bool = False,
            trajectory_start: Optional["ff.TrajectoryStart"] = None,
            trajectory_path: Optional["ff.TrajectoryPath"] = None,
            increasing_vdw: bool = False,
            vdw_cutoff_start: float = 0.0,
            vdw_cutoff_end: float = 12.5,
    ):
        """Automatically build and optimize with the appropriate workflow."""
        return ff.auto_optimize(
            self,
            forcefield=forcefield,
            algorithm=algorithm,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            complex_untangling_attempts=complex_untangling_attempts,
            add_hydrogens=add_hydrogens,
            quality_level=quality_level,
            quality_thresholds=quality_thresholds,
            seed=seed,
            perturb_interval=perturb_interval,
            perturb_sigma=perturb_sigma,
            stopping_criteria=stopping_criteria,
            convergence_level=convergence_level,
            save_movie=save_movie,
            trajectory_start=trajectory_start,
            trajectory_path=trajectory_path,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
        )

    def calc_atom_valence(self, assign_aromatic: Optional[bool] = None):
        from .operations.chemistry import _prepare_valence_and_hydrogens
        return _prepare_valence_and_hydrogens(self, assign_aromatic)

    def calc_implicit_hydrogens(self):
        from .operations.chemistry import _prepare_hydrogens
        _prepare_hydrogens(self, force=True)

    def _components_with_bindings(self):
        from .operations.topology_queries import _components
        return _components(self, Molecule, with_bindings=True)

    @property
    def components(self) -> list['Molecule']:
        from .operations.topology_queries import _components
        return _components(self, Molecule)

    def constraint_bonds_angles(self, exclude_metal_bonds: bool = True):
        """
        Constrains bonds and angles based on the provided conditions.

        This method applies constraints to the bonds and fresh angle views
        depending on whether they involve metals. Bonds and angles with
        metals are excluded from constraints if the flag `exclude_metal_bonds` is set
        to True.

        Args:
            exclude_metal_bonds (bool): A flag to indicate whether bonds and angles 
                involving metals should be excluded from constraints. Default is True.
        """
        if exclude_metal_bonds:
            for bond in self.bonds:
                if not bond.has_metal:
                    bond.constraint = True
            for angle in self.angles:
                if not angle.has_metal:
                    angle.constraint = True

    @property
    def coordinates(self) -> np.ndarray:
        with self._mutation_lock:
            return self._atom_table.coordinates()

    @coordinates.setter
    def coordinates(self, value: np.ndarray) -> None:
        coordinates = np.asarray(value, dtype=np.float64)
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']):
            assert coordinates.shape == (len(self._atoms), 3)
            update = _normalize_atom_facts(_capture_source(self), tuple(range(len(self._atoms))),
                        tuple(atom._uid for atom in self._atoms), {'coordinates': coordinates})
            _commit_coordinates(self, update)

    def _create_atom_from_array(self, attrs_array: np.ndarray) -> "Atom":
        return _import_atom_row(self, attrs_array)

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
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            transaction.touch_caches()
            atom = self._create_atom(**kwargs)
            self._update_topology(_prepared_index=self._topology_index)
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
            model: PartialChargeModel = "qeq",
            model_args: Optional[str] = None,
    ) -> tuple[float, ...]:
        """Calculate atomic partial charges with an OpenBabel charge model.

        Parameters
        ----------
        model
            OpenBabel charge-model plugin selected for the intended chemical
            domain:

            - ``gasteiger`` is a fast topology-based model for ordinary organic
              molecules.
            - ``mmff94`` is intended for molecules covered by MMFF94 atom types;
              use explicit hydrogens when their individual charges must be retained.
            - ``eem`` reproduces Bultinck B3LYP/6-31G*/MPA-style EEM charges.
              The ``eem2015`` variants use HF (``h``) or B3LYP (``b``) with
              AIM (``a``), MPA (``m``), or NPA (``n``) reference charges.
            - ``qeq`` is the Rappe-Goddard charge-equilibration model and is
              commonly used for inorganic and materials structures.
            - ``qtpie`` limits long-range charge transfer and is preferable to
              QEq for separated fragments and strongly distance-dependent cases.
            - ``eqeq`` is OpenBabel's extended charge-equilibration model.
            - ``fromfile`` reads externally supplied charges; pass its backend
              argument, normally a charge-file path, through ``model_args``.
            - ``none`` explicitly clears all partial charges to zero.

            EEM, QEq, QTPIE, and EQEq are geometry-dependent and require sensible
            coordinates and backend parameters for every element/type present.
            Available plugins can vary with the OpenBabel build.
        model_args
            Optional argument string forwarded unchanged to OpenBabel's
            ``OBChargeModel.ComputeCharges`` overload.

        Returns
        -------
        tuple[float, ...]
            Partial charges in Hotpot atom order. This method does not mutate
            ``Atom.partial_charge``; use :meth:`assign_partial_charge` to store
            them. Formal charges and ``Molecule.charge`` are never modified.
        """
        ob_charge_model = ob.OBChargeModel.FindType(model)
        if ob_charge_model is None:
            raise ValueError(
                f"OpenBabel charge model {model!r} is unavailable; "
                f"known models are {', '.join(OPENBABEL_PARTIAL_CHARGE_MODELS)}"
            )

        obmol = ob.OBMol(self.to_obmol())
        if model_args is None:
            succeeded = ob_charge_model.ComputeCharges(obmol)
        else:
            succeeded = ob_charge_model.ComputeCharges(obmol, model_args)
        if not succeeded:
            raise RuntimeError(
                f"OpenBabel charge model {model!r} could not calculate charges"
            )

        charges = tuple(float(value) for value in ob_charge_model.GetPartialCharges())
        if len(charges) != len(self.atoms):
            raise RuntimeError(
                f"OpenBabel charge model {model!r} returned {len(charges)} "
                f"charges for {len(self.atoms)} atoms"
            )
        if not np.all(np.isfinite(charges)):
            raise RuntimeError(
                f"OpenBabel charge model {model!r} returned non-finite charges"
            )
        return charges

    def assign_partial_charge(
            self,
            model: PartialChargeModel = "qeq",
            model_args: Optional[str] = None,
    ) -> tuple[float, ...]:
        """Calculate and store OpenBabel partial charges on all atoms.

        The model applicability and ``model_args`` contract are identical to
        :meth:`get_partial_charge`. Partial charges describe a model-dependent,
        continuous electron-density partition and are not integer formal charges;
        this method therefore leaves every ``Atom.formal_charge`` and the molecular
        total charge unchanged.

        Returns
        -------
        tuple[float, ...]
            Assigned charges in Hotpot atom order.
        """
        charges = self.get_partial_charge(model=model, model_args=model_args)
        for atom, charge in zip(self.atoms, charges):
            atom.partial_charge = charge

        if self._obmol is not None:
            for ob_atom, charge in zip(ob.OBMolAtomIter(self._obmol), charges):
                ob_atom.SetPartialCharge(charge)
        return charges

    @property
    def pair_dist(self) -> np.ndarray:
        from .operations.geometry import _pair_distances
        return _pair_distances(self)

    @property
    def dist_matrix(self) -> np.ndarray:
        from .operations.geometry import _distance_matrix
        return _distance_matrix(self)

    @property
    def element_counts(self):
        from .operations.descriptors import _element_counts
        return _element_counts(self)

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
        self._update_topology(clear_conformers)

    def hide_metal_ligand_bonds(self, clear_conformers: bool = False) -> None:
        """ break all bonds link with metals """
        metal_bonds = [b for b in self.bonds if b.is_metal_ligand_bond]
        self._hided_metal_bonds.extend(metal_bonds)

        for b in metal_bonds:
            self._bonds.remove(b)

        self._update_topology(clear_conformers)

    def graph_spectral(self, norm: Literal['infinite', 'min', 'l1', 'l2'] = 'l2'):
        from .operations.descriptors import _graph_spectral
        return _graph_spectral(self, norm)

    @property
    def formula(self) -> str:
        from .operations.descriptors import _formula
        return _formula(self)

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
        return any(
            pair_distance.measurement.distance < _MINIMUM_ATOM_SEPARATION
            for pair_distance in geometry_adapter.measure_atom_pair_distances(
                self,
                pair_scope="all",
            )
        )

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
        if len(self.atoms) < 2:
            return False
        reference = self.atoms[0].coordinates
        return any(
            bool(np.any(atom.coordinates != reference))
            for atom in self.atoms[1:]
        )

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

    def bond_ring_piercing_state(
            self,
            *,
            ring_scope: Literal["full_graph", "ligand_skeleton"] = "full_graph",
            max_ring_size: int = _DEFAULT_BOND_RING_MAXIMUM_SIZE,
    ) -> "geometry.PiercingState":
        """Return the aggregate factual bond--ring piercing state."""
        return geometry_adapter.determine_bond_ring_piercing_state(
            self,
            ring_scope=ring_scope,
            max_ring_size=max_ring_size,
        )

    def bond_ring_relations(
            self,
            *,
            ring_scope: Literal["full_graph", "ligand_skeleton"] = "full_graph",
            max_ring_size: int = _DEFAULT_BOND_RING_MAXIMUM_SIZE,
    ) -> "geometry_adapter.BondRingScanReport[Ring, Bond]":
        """Return the dense factual report for selected bond--ring pairs."""
        return geometry_adapter.scan_bond_ring_relations(
            self,
            ring_scope=ring_scope,
            max_ring_size=max_ring_size,
        )

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
    def mca_sites(self) -> dict["Atom", float]:
        """Important nucleophilic sites and their MCA values in kJ/mol."""
        if "_mca_sites" not in self.__dict__:
            raise AttributeError(
                "MCA sites have not been calculated for this molecule; "
                "call hotpot.cheminfo.calculator.mca(mol) first"
            )
        return dict(self._mca_sites)

    @property
    def is_organic(self) -> bool:
        """
        Determines whether a molecule is organic based on its atomic composition. 

        Organic molecules are defined as containing no metal atoms, at least one
        hydrogen atom (explicit or implicit), and at least one carbon atom.

        @return: True if the molecule meets the criteria for being organic; otherwise, False
        @rtype: bool
        """
        from .operations.chemistry import _require_implicit_hydrogens
        return (
            all(not a.is_metal for a in self.atoms) and
            (bool(self.hydrogens) or any(_require_implicit_hydrogens(self))) and
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

        self._update_topology(clear_conformers=False)

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

    def _remove_atoms_from_topology(self, atoms: Iterable[Union['Atom', int]]) -> None:
        source_atoms = tuple(self._atoms)
        atoms = tuple(atom if isinstance(atom, Atom) else self._atoms[atom] for atom in atoms)
        _selected_atom_rows(self, atoms)
        removed = {id(atom) for atom in atoms}
        retained = tuple(atom for atom in source_atoms if id(atom) not in removed)
        rows = tuple(self._topology_index[id(atom)][0] for atom in retained)
        affected = {atom._uid for atom in atoms}
        affected.update(neighbour._uid for atom in atoms for neighbour in atom.neighbours)
        for atom in atoms:
            _detach_atom(atom)
        _mark_topology_dirty(self, tuple(affected))
        state = _remap_dependency_state(self._atom_table, tuple(atom._uid for atom in source_atoms),
                                        tuple(atom._uid for atom in retained), rows=rows)
        table = self._atom_table.take(rows)
        _install_dependency_state(table, state)
        self._atom_table = table
        self._atoms[:] = retained
        for name in ('_bonds', '_hided_metal_bonds', '_hided_covalent_bonds'):
            getattr(self, name)[:] = [bond for bond in getattr(self, name)
                                     if all(id(atom) not in removed for atom in bond.atoms)]
        self._topology_index = _build_topology_index(self._atoms, self._bonds,
                                                    self._hided_metal_bonds, self._hided_covalent_bonds)
        self._remap_stereo_after_atom_removal(source_atoms)

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
        self.remove_atoms((atom,))

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
        self._remove_atoms_from_topology(atoms)
        self._update_topology()

    def remove_bond(self, bond: "Bond") -> None:
        """
        Removes a bond from the collection and updates the associated graph structure.

        Parameters:
            bond (Bond): The bond object to be removed.

        Returns:
            None
        """
        self.remove_bonds((bond,))

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
        bonds = tuple(bonds)
        removed_bond_rows = tuple(
            (bond.a1idx, bond.a2idx)
            for bond in bonds
        )
        _mark_topology_dirty(self, tuple(atom._uid for bond in bonds for atom in bond.atoms))
        for bond in bonds:
            self._bonds.remove(bond)
        self._remove_stereo_bond_rows(removed_bond_rows)
        self._update_topology()

    def force_remove_polar_hydrogens(self):
        self.remove_atoms([a for a in self._atoms if a.is_polar_hydrogen])

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

    def search_substructure(self, smarts: str, *, semantics="full_graph"):
        from .search import Substructure, Searcher
        sub = Substructure.from_smarts(smarts, semantics=semantics)
        searcher = Searcher(sub)
        return searcher.search(self)

    def set_default_valence(self):
        from .operations.chemistry import _prepare_valences
        return _prepare_valences(self)

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
    def _longest_path(node_count: int, edges: Iterable[tuple[int, int]],
                      _start_node: int = None, _end_node: int = None,
                      _scope_nodes: Iterable[int] = None):
        from .rules.mathematics.topology import _longest_path
        return _longest_path(node_count, edges, _start_node, _end_node, _scope_nodes)

    def longest_path(self):
        from .operations.topology_queries import _longest_path
        return _longest_path(self)

    def canonical_tree(self):
        from .operations.topology_queries import _canonical_tree
        return _canonical_tree(self)

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
        from .operations.topology_queries import _shortest_path_indices
        return _shortest_path_indices(self, atom1, atom2)

    def shortest_path(self, atom1, atom2) -> list['Atom']:
        from .operations.topology_queries import _shortest_path
        return _shortest_path(self, atom1, atom2)

    def shortest_paths_indices(self, atom1, atom2):
        """
        Find the deterministic shortest-path indices between two atoms.

        Parameters:
        atom1: The first atom for which the shortest path is to be calculated.
        atom2: The second atom for which the shortest path is to be calculated.

        Returns:
        list:
            A list of integers representing the path, or an empty list when the
            atoms are disconnected.
        """
        return self.shortest_path_index(atom1, atom2)

    @property
    def smiles(self) -> str:
        """ Return smiles string. """
        # return pb.readstring('smi', pb.Molecule(mol2obmol(self)[0]).write().strip()).write('can').strip()
        # return pb.Molecule(self.to_obmol()).write('can')
        return pb.readstring('mol2', pb.Molecule(self.to_obmol()).write('mol2')).write('can').split()[0]
        # return pb.readstring('smi', pb.Molecule(mol2obmol(self)[0]).write().strip()).write('can').strip()
        # return pb.Molecule(mol2obmol(self)[0]).write().strip()

    @property
    def kekulize_smiles(self) -> str:
        return pb.readstring('mol2', pb.Molecule(self.to_obmol()).write('mol2')).write('can', opt={'k': None}).split()[0]

    @property
    def rings(self) -> list["Ring"]:
        return self._materialize_rings()

    @property
    def cycle_basis(self) -> list[tuple[int, ...]]:
        from .operations.topology_queries import _cycle_basis
        return _cycle_basis(self)

    @property
    def all_cycle_basis(self) -> list[list[tuple[int, ...]]]:
        from .operations.topology_queries import _all_cycle_basis
        return _all_cycle_basis(self)

    @property
    def basis_rings(self) -> list["Ring"]:
        """Materialize Ring objects corresponding exactly to ``cycle_basis``."""
        return self._rings_from_cycles(self.cycle_basis)

    @property
    def all_basis_rings(self) -> list[list["Ring"]]:
        """Materialize Ring objects for every entry of ``all_cycle_basis``."""
        return [
            self._rings_from_cycles(basis)
            for basis in self.all_cycle_basis
        ]

    def _rings_from_cycles(self, cycles: Iterable[Iterable[int]]) -> list["Ring"]:
        from .operations.topology_queries import _rings_from_cycles
        return _rings_from_cycles(self, Ring, cycles)

    def _materialize_rings(
            self,
            *,
            ligand_skeleton: bool = False,
            max_size: Optional[int] = None,
            max_cycles: Optional[int] = graph_algorithms.DEFAULT_RELEVANT_CYCLE_LIMIT,
    ) -> list["Ring"]:
        from .operations.topology_queries import _ring_views
        return _ring_views(self, Ring, ligand_skeleton=ligand_skeleton,
                           max_size=max_size, max_cycles=max_cycles)

    def rings_for_scope(
            self,
            ring_scope: Literal["full_graph", "ligand_skeleton"],
            *,
            max_size: Optional[int] = None,
            max_cycles: Optional[int] = graph_algorithms.DEFAULT_RELEVANT_CYCLE_LIMIT,
    ) -> list["Ring"]:
        """Return a Relevant Cycle view for a molecular graph scope."""
        if ring_scope == "full_graph":
            return self._materialize_rings(
                max_size=max_size,
                max_cycles=max_cycles,
            )
        if ring_scope == "ligand_skeleton":
            return self._materialize_rings(
                ligand_skeleton=True,
                max_size=max_size,
                max_cycles=max_cycles,
            )
        raise ValueError(f"Unsupported ring scope: {ring_scope!r}")

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
        return self._materialize_rings(ligand_skeleton=True)



    def to_pyg_data(self, prefix: str = "", with_batch: bool = True):
        from ..plugins.PyG.data.utils import mol_to_pyg_data
        return mol_to_pyg_data(self, prefix, with_batch)

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
        """Translate current coordinates through the one coordinate commit."""
        from .operations.geometry import _translate
        return _translate(self, vector)





    @property
    def weight(self):
        from .operations.descriptors import _weight
        return _weight(self)

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


    def _capture_frame_state(self, *, rows=None):
        with self._mutation_lock, self._conformers._mutation_lock:
            values = []
            for name in Conformers._attrs:
                value = getattr(self._conformers, '_' + name, None)
                if isinstance(value, np.ndarray):
                    if rows is not None and name in ('coordinates', 'force', 'partial_charge'):
                        if value.ndim >= 2 and value.shape[1] == len(self._atoms):
                            value = value[:, tuple(rows)]
                        elif name == 'partial_charge' and value.ndim == 1 and len(value) == len(self._atoms):
                            value = value[list(rows)]
                    value = value.copy()
                else:
                    value = deepcopy(value)
                values.append(value)
            return tuple(values), self._conformers_index


    def _install_frame_state(self, values, index):
        prepared = tuple(np.array(value, copy=True) if isinstance(value, np.ndarray) else deepcopy(value)
                         for value in values)
        with self._conformers._mutation_lock:
            previous = tuple(getattr(self._conformers, '_' + name, None) for name in Conformers._attrs)
            old_index = self._conformers_index
            try:
                for name, value in zip(Conformers._attrs, prepared):
                    object.__setattr__(self._conformers, '_' + name, value)
                self._conformers_index = index
            except BaseException:
                for name, value in zip(Conformers._attrs, previous):
                    object.__setattr__(self._conformers, '_' + name, value)
                self._conformers_index = old_index
                raise


    def _capture_factual_record(self, *, rows=None, include_frames=True, copy_metadata=True):
        from .structure._builder import _make_molecule_record
        from .stereo import _remap_stereo_payload
        with self._mutation_lock:
            rows = tuple(range(len(self._atoms))) if rows is None else tuple(rows)
            mapping = {row: target for target, row in enumerate(rows)}
            atoms = tuple(self._atoms[row] for row in rows)
            complete = tuple((partition, bond) for partition, bonds in enumerate(
                             (self._bonds, self._hided_metal_bonds, self._hided_covalent_bonds)) for bond in bonds)
            bonds = tuple((partition, bond) for partition, bond in complete
                          if self._topology_index[id(bond.atom1)][0] in mapping
                          and self._topology_index[id(bond.atom2)][0] in mapping)
            endpoints = tuple((mapping[self._topology_index[id(bond.atom1)][0]],
                               mapping[self._topology_index[id(bond.atom2)][0]]) for _, bond in bonds)
            whole = rows == tuple(range(len(self._atoms)))
            payload = _remap_stereo_payload(self._stereo_payload, mapping,
                                            allowed_bonds=None if whole else endpoints)
            frames, index = self._capture_frame_state(rows=rows) if include_frames else ((None,) * 9, 0)
            uid_rows = {atom._uid: row for row, atom in enumerate(self._atoms)}
            def constraints(entries):
                return tuple(tuple(mapping[uid_rows[uid]] for uid in entry)
                             for entry in entries if all(uid in uid_rows and uid_rows[uid] in mapping
                                                         for uid in entry))
            metadata = tuple(bond._bond_source_metadata for _, bond in bonds)
            if copy_metadata:
                metadata = deepcopy(metadata)
            else:
                memo = {id(self): self, **{id(atom): atom for atom in self._atoms},
                        **{id(bond): bond for _, bond in complete}}
                metadata = deepcopy(metadata, memo)
            record = _make_molecule_record(
                atom_values=self._atom_table.select(_ATOM_FIELDS, rows=rows),
                atom_presence=self._atom_table.presence(rows), atom_sources=self._atom_table.sources(rows),
                atom_isotopes=tuple((row, atom.__dict__['isotope']) for row, atom in enumerate(atoms) if 'isotope' in atom.__dict__),
                atom_frac_valences=tuple((row, atom.__dict__['frac_valence']) for row, atom in enumerate(atoms) if 'frac_valence' in atom.__dict__),
                bond_endpoints=endpoints, bond_values=np.array([bond._attrs for _, bond in bonds], dtype=np.float64).reshape(-1, 3),
                bond_partitions=tuple(partition for partition, _ in bonds),
                bond_kinds=tuple(bond._bond_kind.value for _, bond in bonds),
                bond_directions=tuple(bond._bond_direction for _, bond in bonds),
                bond_sources=tuple(bond._bond_source for _, bond in bonds), bond_metadata=(None,) * len(bonds),
                bond_aromatic=tuple(bond._is_aromatic for _, bond in bonds),
                charge=(True, self._charge_value, self._charge_source) if whole and self._charge_present else (False, None, None),
                spin_mult=(True, self._spin_value, self._spin_source) if whole and self._spin_present else (False, None, None),
                cell=None if self._crystal is None else tuple(getattr(self._crystal, name) for name in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')),
                stereo_payload=payload, legacy_frame_values=frames, legacy_frame_index=index,
                angle_constraints=constraints(self._constraints.angles), torsion_constraints=constraints(self._constraints.torsions),
                identifier=('identifier' in self.__dict__, self.__dict__.get('identifier')),
                pair_wiberg_bond_orders=tuple((mapping[uid_rows[first]], mapping[uid_rows[second]], value)
                    for (first, second), value in self._pair_wiberg_bond_orders.items()
                    if first in uid_rows and second in uid_rows
                    and uid_rows[first] in mapping and uid_rows[second] in mapping))
            return record, metadata


    def _install_factual_record(self, record, *, atom_bindings=None, bond_bindings=None,
                                 dependency_state=None, bond_metadata=None):
        from .structure._builder import _thaw_canonical
        atom_bindings = (None,) * record.atom_count if atom_bindings is None else tuple(atom_bindings)
        bond_bindings = (None,) * len(record.bond_endpoints) if bond_bindings is None else tuple(bond_bindings)
        if len(atom_bindings) != record.atom_count or len(bond_bindings) != len(record.bond_endpoints):
            raise ValueError('Factual bindings must align with the import record')
        old_atoms = tuple(self._atoms)
        old_bonds = tuple(self._bonds + self._hided_bonds)
        old_hydrogen_inputs = _hydrogen_install_inputs(self) if dependency_state is not None else {}
        for bindings, current in ((atom_bindings, old_atoms), (bond_bindings, old_bonds)):
            current_ids = {id(member) for member in current}
            present = [member for member in bindings if member is not None]
            if len({id(member) for member in present}) != len(present) or any(
                    member._uid_owner is not self._owner_token or id(member) not in current_ids
                    for member in present):
                raise ValueError('Factual bindings require distinct current members of this owner')
        if bond_metadata is None:
            metadata = tuple(_thaw_canonical(value) or {} for value in record.bond_metadata)
        else:
            metadata = tuple(bond_metadata)
            if len(metadata) != len(record.bond_endpoints) or any(value != ('none',) for value in record.bond_metadata):
                raise ValueError('Opaque Bond metadata must be complete and exclude record metadata')
        table = _AtomFactTable(np.frombuffer(record.atom_values, dtype='<f8').reshape(record.atom_count, len(_ATOM_FIELDS)),
                               presence=np.frombuffer(record.atom_presence, dtype='<u4'),
                               sources=np.frombuffer(record.atom_sources, dtype='u1').reshape(record.atom_count, len(_ATOM_FIELDS)))
        first_atom, first_bond = self._next_atom_uid, self._next_bond_uid
        self._next_atom_uid += sum(atom is None for atom in atom_bindings)
        self._next_bond_uid += sum(bond is None for bond in bond_bindings)
        atom_uids = iter(range(first_atom, self._next_atom_uid))
        bond_uids = iter(range(first_bond, self._next_bond_uid))
        atoms = tuple(_new_atom_handle(self, next(atom_uids)) if atom is None else atom for atom in atom_bindings)
        pair_wiberg_bond_orders = {
            tuple(sorted((atoms[first]._uid, atoms[second]._uid))): value
            for first, second, value in record.pair_wiberg_bond_orders}
        values = np.frombuffer(record.bond_values, dtype='<f8').reshape(-1, 3)
        bonds = []
        for row, binding in enumerate(bond_bindings):
            first, second = record.bond_endpoints[row]
            kwargs = dict(kind=record.bond_kinds[row], direction=record.bond_directions[row],
                          source_namespace=record.bond_sources[row], metadata=metadata[row], aromatic=record.bond_aromatic[row])
            if binding is None:
                bond = _new_bond_handle(self, next(bond_uids), atoms[first], atoms[second], values[row], **kwargs)
            else:
                bond = binding
                _initialize_bond_handle(bond, self, bond._uid, atoms[first], atoms[second], values[row], **kwargs)
            bonds.append(bond)
        partitions = tuple([bond for bond, partition in zip(bonds, record.bond_partitions) if partition == wanted]
                           for wanted in range(3))
        topology_index = _build_topology_index(atoms, *partitions)
        if dependency_state is not None:
            _install_dependency_state(table, dependency_state)
        elif old_atoms:
            old_ids = {id(atom) for atom in old_atoms}
            retained_uids = {atom._uid for atom in atoms if id(atom) in old_ids}
            for row, atom in enumerate(atoms):
                previous = self._topology_index.get(id(atom))
                if previous is not None:
                    table._group_status[row] = self._atom_table._group_status[previous[0]]
            table._dirty_seeds = tuple(set(seeds).intersection(retained_uids) for seeds in self._atom_table._dirty_seeds)
        retained_ids = {id(atom) for atom in atoms}
        for atom in old_atoms:
            if id(atom) not in retained_ids:
                _detach_atom(atom)
        sparse_isotopes, sparse_valences = dict(record.atom_isotopes), dict(record.atom_frac_valences)
        transaction = _current_transaction(self)
        for row, atom in enumerate(atoms):
            if transaction is not None:
                transaction.touch_sparse(atom)
            atom.__dict__.pop('_detached_table', None)
            for name, sparse in (('isotope', sparse_isotopes), ('frac_valence', sparse_valences)):
                if row in sparse:
                    atom.__dict__[name] = sparse[row]
                else:
                    atom.__dict__.pop(name, None)
        self._pair_wiberg_bond_orders = pair_wiberg_bond_orders
        self._atom_table = table
        self._atoms[:] = atoms
        self._bonds[:], self._hided_metal_bonds[:], self._hided_covalent_bonds[:] = partitions
        self._update_topology(clear_conformers=False, _prepared_index=topology_index,
                             _constraint_rows=(record.angle_constraints, record.torsion_constraints))
        for name, declaration in (('charge', record.charge), ('spin', record.spin_mult)):
            present, value, source = declaration
            object.__setattr__(self, '_' + name + '_present', present)
            object.__setattr__(self, '_' + name + '_value', value if present else (0 if name == 'charge' else None))
            object.__setattr__(self, '_' + name + '_source', source)
        self._stereo_payload = record.stereo_payload if record.stereo_payload is not None else _StereoPayload(1, (), (), ())
        if dependency_state is None:
            _mark_topology_dirty(self, tuple(atom._uid for atom in atoms))
        elif old_hydrogen_inputs:
            changed = tuple(uid for uid, inputs in _hydrogen_install_inputs(self).items()
                            if uid in old_hydrogen_inputs and old_hydrogen_inputs[uid] != inputs)
            if changed:
                _seed(self, GROUP_INDEX['IMPLICIT_H'], changed)
        self._crystal = None if record.cell is None else cryst.Crystal(*record.cell, mol=self)
        self._install_frame_state(tuple(_thaw_canonical(value) for value in record.legacy_frame_values), record.legacy_frame_index)
        if record.identifier[0]:
            self.identifier = record.identifier[1]
        else:
            self.__dict__.pop('identifier', None)
        return self



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
    """
    _attrs_dict = {}
    _attrs_setter = {}
    _default_attrs = {}
    _attrs_enumerator = tuple(_attrs_dict.keys())

    @property
    def _uid(self):
        return self._member_uid

    def __repr__(self):
        return f"{self.__class__.__name__}({self.label})"

    def __copy__(self):
        raise PermissionError(f"The {self.__class__.__name__} not allow to copy")

    def __dir__(self):
        return list(self._attrs_enumerator) + list(super(MolBlock, self).__dir__())

    def __getattr__(self, item):
        if item in self._attrs_dict:
            if isinstance(self, Atom):
                return _atom_numeric_value(self, item)
            with self.mol._mutation_lock:
                return self._attrs_dict[item](self._attrs[self._attrs_enumerator.index(item)])
        return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        if isinstance(self, Atom):
            _atom_fact_engine()
            Atom.__setattr__(self, key, value)
        elif isinstance(self, Bond) and key in self._attrs_dict:
            if key == 'bond_order':
                _bond_order_setter(self, key, value)
            else:
                _store_bond_field(self, key, value)
        else:
            object.__setattr__(self, key, value)

    @staticmethod
    def _default_attr_setter(self, key, value):
        if isinstance(self, Atom):
            _store_atom_fields(self, {key: value})
        else:
            _store_bond_field(self, key, value)

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



#######################################################################
#######################################################################
# Define attributes setters
def _atomic_number_setter(self: 'Atom', key, atomic_number):
    _store_atom_fields(self, {'atomic_number': atomic_number})

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

    _attrs_dict = dict(_ATOM_FIELD_CODECS)

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

    def __init__(self, mol: Molecule = None, **kwargs):
        owner = mol if mol is not None else Molecule()
        uid = owner._allocate_atom_uid()
        allowed = {'atomic_number', 'symbol', 'formal_charge', 'implicit_hydrogens',
                   'coordinates', 'x', 'y', 'z', 'id', 'constraint',
                   'x_constraint', 'y_constraint', 'z_constraint'}
        unknown = kwargs.keys() - allowed
        if unknown:
            raise TypeError(f"Unsupported Atom constructor fields: {', '.join(sorted(unknown))}")
        constraints = {name: kwargs.pop(name) for name in tuple(kwargs)
                       if name in ('constraint', 'x_constraint', 'y_constraint', 'z_constraint')}
        with _FactTransaction(owner, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            row = len(owner._atom_table)
            update = _normalize_atom_facts(_capture_source(owner), (row,), (uid,), kwargs)
            values = np.zeros(19, dtype=np.float64)
            presence = 0
            sources = np.zeros((1, 19), dtype=np.uint8)
            for field, column in zip(update.fields, update.values):
                index = _ATOM_FIELD_INDEX[field]
                values[index] = column[0]
                presence |= 1 << index
                sources[0, index] = _ATOM_SOURCE_CODES['explicit']
            for axis in ('x_constraint', 'y_constraint', 'z_constraint'):
                if axis in constraints or 'constraint' in constraints:
                    value = constraints.get(axis, constraints.get('constraint'))
                    if 'constraint' in constraints and axis in constraints and bool(value) != bool(constraints['constraint']):
                        raise ValueError(f'Conflicting Atom constructor aliases for {axis}')
                    index = _ATOM_FIELD_INDEX[axis]
                    values[index] = bool(value)
                    presence |= 1 << index
                    sources[0, index] = _ATOM_SOURCE_CODES['explicit']
            _initialize_atom_handle(self, owner, uid)
            transaction.touch_registration(self)
            owner._atom_table._append_prepared(values[None, :], np.asarray([presence], dtype=np.uint32), sources)
            owner._atoms.append(self)
            owner._topology_index[id(self)] = (row, ())
            _seed(owner, GROUP_INDEX['FORMAL_CHARGE_COVERAGE'], (uid,))

    @property
    def attrs(self):
        from .operations.chemistry import _atom_projection
        with self.mol._mutation_lock:
            record = self.mol._topology_index.get(id(self))
            if record is None:
                return _detached_atom_projection(self)
            return _atom_projection(self.mol, rows=(record[0],))[0]

    def __reduce_ex__(self, protocol):
        with self.mol._mutation_lock:
            record = self.mol._topology_index.get(id(self))
            if record is not None:
                return _restore_atom_member, (self.mol, record[0], self._uid)
            state = self._detached_table.__getstate__()
            extra = {key: value for key, value in self.__dict__.items()
                     if key not in ('_mol', '_member_uid', '_uid_owner', '_detached_table')}
            return _restore_detached_atom, (self.mol, self._uid, state), extra

    def __setstate__(self, state):
        self.__dict__.update(state)


    def _add_atom(
            self,
            atom: Union[str, int, "Atom"] = 1,
            bond_order: int = 1,
            atom_attrs: dict = None,
            bond_attrs: dict = None
    ):
        if isinstance(atom, str):
            from .reference.properties import _atomic_number_for_symbol
            atom = _atomic_number_for_symbol(atom)
        if isinstance(atom, int):
            atom = self.mol._create_atom(atomic_number=atom, **(atom_attrs or {}))
        else:
            atom = self.mol._add_atom(atom)
            if atom_attrs:
                self.mol.updata_atoms(atoms=(atom,), **atom_attrs)
        self.mol._add_bond(self, atom, bond_order, **(bond_attrs or {}))
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
        getattr(self.mol, '_update_topology')()
        return atom

    @staticmethod
    def random_point_on_sphere(
            radius: float = 1.,
            rng: Optional[np.random.Generator] = None,
    ):
        """
        Generates a random point on the surface of a sphere with a given radius.

        The function uses uniformly distributed random numbers to compute a point
        on the sphere by generating the polar angle (theta) and azimuthal angle
        (phi) in spherical coordinates, and converting them into Cartesian
        coordinates (x, y, z). The generated point lies on the sphere's surface.

        Args:
            radius (float): The radius of the sphere. Default is 1.0.
            rng (numpy.random.Generator, optional): Generator used for sampling.
                When omitted, NumPy's process-global random generator is used.

        Returns:
            tuple[float, float, float]: Cartesian coordinates (x, y, z) of the
            point on the sphere's surface.
        """
        # 随机生成极角 theta 和方位角 phi
        random = np.random.rand if rng is None else rng.random
        theta = np.arccos(2 * random() - 1)  # 0 到 pi
        phi = 2 * np.pi * random()  # 0 到 2pi

        # convert to Cartesian coordination
        x = np.sin(theta) * np.cos(phi) * radius
        y = np.sin(theta) * np.sin(phi) * radius
        z = np.cos(theta) * radius

        return x, y, z

    @property
    def period(self) -> int:
        return self.n

    @property
    def group(self):
        from .rules.chemistry.elements import _group
        return _group(self.atomic_number)

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

    def _add_hydrogens(
            self,
            num: int = None,
            rm_polar_hs: bool = True,
            rng: Optional[np.random.Generator] = None,
    ) -> (int, list["Atom"]):
        from .operations.chemistry import _add_atom_hydrogens
        return _add_atom_hydrogens(self, num, rm_polar_hs, rng)

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
        with _FactTransaction(self.mol, producer=_ATOM_SOURCE_CODES['implicit_hydrogens']) as transaction:
            transaction.touch_topology()
            add_or_rm, hydrogens = self._add_hydrogens(num)
            self.mol._update_topology()
            return hydrogens

    @property
    def bonds(self) -> list['Bond']:
        with self.mol._mutation_lock:
            record = self.mol._topology_index.get(id(self))
            return [] if record is None else list(record[1])

    def _calc_implicit_hydrogens(self) -> int:
        # TODO: Implement by C++
        from .operations.chemistry import _atom_implicit_proposal
        return _atom_implicit_proposal(self)


    def calc_implicit_hydrogens(self):
        from .operations.chemistry import _prepare_hydrogens
        _prepare_hydrogens(self.mol, rows=(self.idx,), force=True)

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
        _store_atom_fields(self, dict.fromkeys(('x_constraint', 'y_constraint', 'z_constraint'), value))

    @property
    def coordinates(self):
        with self.mol._mutation_lock:
            return Atom._coord_getter(self)

    @coordinates.setter
    def coordinates(self, value: types.ArrayLike):
        x, y, z = value
        _store_atom_fields(self, {'x': x, 'y': y, 'z': z})

    @property
    def exact_mass(self):
        from .rules.chemistry.elements import _exact_mass
        return _exact_mass(self.atomic_number)

    @property
    def electronegativity(self) -> Optional[float]:
        from .rules.chemistry.elements import _electronegativity
        return _electronegativity(self.atomic_number)

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
        from .rules.chemistry.elements import _covalent_radius
        return _covalent_radius(self.atomic_number)

    @property
    def density(self) -> float:
        from .rules.chemistry.elements import _density
        return _density(self.atomic_number)

    @property
    def hyb(self) -> int:
        from .operations.chemistry import _atom_hybridization
        return _atom_hybridization(self)

    @property
    def idx(self) -> int:
        with self.mol._mutation_lock:
            return self.mol._topology_index[id(self)][0]

    @property
    def mca(self) -> float:
        """Predicted methyl cation affinity of this atom in kJ/mol."""
        if "_mca" not in self.__dict__:
            raise AttributeError(
                "MCA has not been calculated for this atom; "
                "call hotpot.cheminfo.calculator.mca(atom.mol) first"
            )
        return self._mca

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
    def is_lanthanide(self):
        from .rules.chemistry.elements import _is_lanthanide
        return _is_lanthanide(self.atomic_number)

    @property
    def is_actinide(self):
        from .rules.chemistry.elements import _is_actinide
        return _is_actinide(self.atomic_number)

    @property
    def is_noble_gases(self):
        from .rules.chemistry.elements import _is_noble_gas
        return _is_noble_gas(self.atomic_number)

    @property
    def is_halogens(self):
        from .rules.chemistry.elements import _is_halogen
        return _is_halogen(self.atomic_number)

    @property
    def is_metal(self):
        from .rules.chemistry.elements import _is_metal
        return _is_metal(self.atomic_number)

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
        from .rules.chemistry.elements import _mass
        return _mass(self.atomic_number)

    @property
    def mol(self) -> Molecule:
        """Return the immutable molecule owner, including after detachment."""
        return self._mol

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
        """Return adjacent Atom indices in deterministic order."""
        return np.array([
            self.mol._topology_index[id(
                bond.atom2 if bond.atom1 is self else bond.atom1
            )][0]
            for bond in self.bonds
        ])

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
    def h_count(self):
        return len(self.hydrogens)

    @property
    def connectivity(self):
        return len(self.bonds)

    @property
    def neighbours(self) -> list['Atom']:
        """Return adjacent Atoms in deterministic index order."""
        return [
            bond.atom2 if bond.atom1 is self else bond.atom1
            for bond in self.bonds
        ]

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
        from .rules.chemistry.bonding.valence import _missing_electrons
        return _missing_electrons(self.atomic_number)

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
        from .rules.chemistry.elements import _electron_configuration
        return np.asarray(_electron_configuration(self.atomic_number), dtype=int)


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
        from .rules.chemistry.elements import _fill_electron_shell
        n, l, conf = _fill_electron_shell(self.atomic_number)
        return n, l, conf + [0] * (5 - len(conf))

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

        from .operations.chemistry import _require_implicit_hydrogens
        hydrogen_count = _require_implicit_hydrogens(self.mol, rows=(self.idx,))[0]
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

        _state += hydrogen_count

        if _state > 0:
            _state += _temp_state
        elif _state < 0:
            _state -= _temp_state

        return _state

    def get_formal_charge(self) -> cython.int:
        from .operations.chemistry import _atom_formal_charge
        return _atom_formal_charge(self)

    def get_valence(self) -> cython.int:
        from .operations.chemistry import _atom_valence
        return _atom_valence(self)

    def set_valence_to_default(self):
        from .operations.chemistry import _atom_valence
        _atom_valence(self)


    @property
    def sum_bond_orders(self) -> int:
        from .operations.chemistry import _atom_order_sum
        return _atom_order_sum(self, 'all')

    @property
    def sum_heavy_cov_orders(self) -> int:
        from .operations.chemistry import _atom_order_sum
        return _atom_order_sum(self, 'heavy')

    @property
    def sum_covalent_orders(self) -> int:
        from .operations.chemistry import _atom_order_sum
        return _atom_order_sum(self, 'covalent')

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
        from .reference.properties import _ELEMENT_SYMBOLS
        return _ELEMENT_SYMBOLS[self.atomic_number]

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
        from .reference.properties import _ELEMENT_SYMBOLS
        self.atomic_number = _ELEMENT_SYMBOLS.index(value)

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

    def __setattr__(self, name, value):
        if '_view_source' in self.__dict__ and (
                name in ('_atoms', '_bonds') or name.startswith('_view_') or re.fullmatch(r'(?:atom|bond)\d+|a\d+idx', name)):
            raise AttributeError('View membership is read-only')
        super().__setattr__(name, value)

    def __reduce_ex__(self, protocol):
        if '_view_source' not in self.__dict__:
            return object.__reduce_ex__(self, protocol)
        from .views._source import _restore_sequence_view, _view_source_current
        with self.mol._mutation_lock:
            current = _view_source_current(self.mol, self._view_source, self._atoms, self._view_uids)
            if '_view_representation' in self.__dict__:
                from .views._source import _restore_ring_view
                return (_restore_ring_view, (type(self), self._atoms, self._bonds,
                         self._view_geometry, self._view_representation, current))
            return (_restore_sequence_view,
                    (type(self), self._atoms, self._bonds, self._view_geometry, current))

    def __init__(self, *atoms: Atom):
        self._check_is_same_mol(*atoms)
        self._check_atom_number(*atoms)
        self._atoms = atoms

        with self.mol._mutation_lock:
            if isinstance(self, Bond):
                self._bonds = [self]
            else:
                self._bonds = [self.mol.bond(atoms[i].idx, atoms[i+1].idx) for i in range(len(atoms) - 1)]
            if not isinstance(self, (Bond, Ring)):
                from .views._source import _bind_view
                self._bonds = tuple(self._bonds)
                _bind_view(self, atoms)

    def __repr__(self):
        return f"{self.__class__.__name__}" + '(' + ''.join(a.symbol for a in self.atoms) + ')'

    def __getattr__(self, item):
        if re.match(r"atom\d+", item):
            idx = int(item[4:]) - 1
            with self.mol._mutation_lock:
                return self._atoms[idx]

        elif re.match(r"bond\d+", item):
            idx = int(item[4:]) - 1
            if '_view_source' in self.__dict__:
                return self._bonds[idx]
            atom_start = self._atoms[idx]
            atom_end = self._atoms[idx+1]
            return list(set(atom_start.bonds) & set(atom_end.bonds))[0]

        elif re.match(r"a\d+idx", item):
            len_idx = len(item) - 4
            idx = int(item[1:1+len_idx]) - 1
            with self.mol._mutation_lock:
                self._check_topology_source()
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
        if '_view_source' in self.__dict__:
            raise AttributeError('View membership is read-only')
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

    def _check_topology_source(self):
        if '_view_source' in self.__dict__:
            from .views._source import _check_view_source
            _check_view_source(self.mol, self._view_source, self._atoms, self._view_uids)

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
    def atoms(self) -> tuple[Atom, ...]:
        with self.mol._mutation_lock:
            return tuple(self._atoms)

    @property
    def atoms_indices(self):
        with self.mol._mutation_lock:
            self._check_topology_source()
            return [a.idx for a in self._atoms]

    @property
    def bonds(self) -> list["Bond"]:
        return list(self._bonds)

    @property
    def mol(self) -> Molecule:
        return self._atoms[0].mol

    @property
    def has_metal(self):
        with self.mol._mutation_lock:
            self._check_topology_source()
            return any(a.is_metal for a in self._atoms)

    def to_mol(self):
        with self.mol._mutation_lock:
            self._check_topology_source()
            record, metadata = self.mol._capture_factual_record(rows=tuple(atom.idx for atom in self._atoms), include_frames=False)
        mol = Molecule()
        _commit_topology(mol, _capture_source(mol), record,
                         atom_bindings=(None,) * record.atom_count,
                         bond_bindings=(None,) * len(record.bond_endpoints), bond_metadata=metadata)
        return mol


class AtomPair:
    """Fixed pair membership with source-checked mathematics and supplied observations."""

    __slots__ = ('_atoms', '_view_source', '_view_uids', '_view_geometry')
    _length = 2
    attr_names = ('wiberg_bond_order', 'length_shortest_path')

    def __init__(self, atom1: Atom, atom2: Atom):
        from .views._source import _bind_view
        assert atom1.mol is atom2.mol
        self._atoms = (atom1, atom2)
        _bind_view(self, self._atoms)

    def __reduce_ex__(self, protocol):
        from .views._source import _restore_pair_view, _view_source_current
        with self.mol._mutation_lock:
            current = _view_source_current(self.mol, self._view_source, self._atoms, self._view_uids)
            return _restore_pair_view, (type(self), self._atoms, self._view_geometry, current)

    @property
    def atoms(self):
        return self._atoms

    @property
    def atom1(self):
        return self._atoms[0]

    @property
    def atom2(self):
        return self._atoms[1]

    @property
    def mol(self):
        return self.atom1.mol

    def __repr__(self):
        from .views._source import _check_view_source
        with self.mol._mutation_lock:
            _check_view_source(self.mol, self._view_source, self._atoms, self._view_uids)
            return f"AtomPair({self.atom1.idx}, {self.atom2.idx})"

    def __eq__(self, other):
        if not isinstance(other, AtomPair):
            return NotImplemented
        return frozenset(self._atoms) == frozenset(other._atoms)

    def __contains__(self, atom):
        return atom is self.atom1 or atom is self.atom2

    @property
    def wiberg_bond_order(self):
        with self.mol._mutation_lock:
            if any(atom._uid_owner is not self.mol._owner_token or atom._uid != uid
                   or id(atom) not in self.mol._topology_index
                   for atom, uid in zip(self._atoms, self._view_uids)):
                raise RuntimeError('Pair observation members changed')
            return self.mol._pair_wiberg_bond_orders.get(tuple(sorted(self._view_uids)), 0)

    @wiberg_bond_order.setter
    def wiberg_bond_order(self, value):
        from .structure._commit import _commit_pair_wiberg
        _commit_pair_wiberg(self.mol, self._view_source[0], self._atoms, self._view_uids, value)

    @property
    def attrs(self):
        return [getattr(self, name) for name in self.attr_names]

    @property
    def shortest_path(self) -> list['Atom']:
        from .views._source import _check_view_source
        with self.mol._mutation_lock:
            _check_view_source(self.mol, self._view_source, self._atoms, self._view_uids)
            return self.mol.shortest_path(self.atom1, self.atom2)

    @property
    def length_shortest_path(self) -> int:
        return len(self.shortest_path)

    @property
    def distance(self):
        from .rules.mathematics.view_geometry import _pair_distance
        from .views._source import _view_coordinates
        return _pair_distance(*_view_coordinates(self))


class AtomPairs(Mapping):
    """Caller-owned fixed pair table; scientific observations remain at their owner."""

    __slots__ = ('_mol', '_atoms', '_members', '_items', '_view_source', '_view_uids', '_geometry')

    def __init__(self, *args, mol, **kwargs):
        from types import MappingProxyType
        from .views._source import _capture_view_source, _geometry_snapshot
        with mol._mutation_lock:
            self._mol = mol
            self._atoms = tuple(mol._atoms)
            self._members = MappingProxyType({atom: row for row, atom in enumerate(self._atoms)})
            self._items = MappingProxyType(dict(*args, **kwargs)) if args or kwargs else None
            self._view_source = _capture_view_source(mol)
            self._view_uids = tuple(atom._uid for atom in self._atoms)
            self._geometry = _geometry_snapshot(mol, self._atoms)[1]

    def __reduce_ex__(self, protocol):
        from .views._source import _restore_pair_table, _view_source_current
        with self.mol._mutation_lock:
            current = _view_source_current(self.mol, self._view_source, self._atoms, self._view_uids)
            items = None if self._items is None else tuple(self._items.items())
            return (_restore_pair_table,
                    (type(self), self.mol, self._atoms, self._geometry, current, items))

    @property
    def mol(self):
        return self._mol

    def __len__(self):
        count = len(self._atoms)
        return len(self._items) if self._items is not None else count * (count - 1) // 2

    def __iter__(self):
        return iter(self._items) if self._items is not None else (
            frozenset(atoms) for atoms in combinations(self._atoms, 2))

    def __getitem__(self, atoms):
        key = frozenset(atoms)
        if self._items is not None:
            return self._items[key]
        if len(key) != 2 or any(atom not in self._members for atom in key):
            raise KeyError(key)
        atoms = tuple(atoms)
        pair = object.__new__(AtomPair)
        pair._atoms = atoms
        pair._view_source = self._view_source
        pair._view_uids = tuple(atom._uid for atom in atoms)
        rows = tuple(self._members[atom] for atom in atoms)
        from .views._source import _member_geometry
        pair._view_geometry = _member_geometry(self._geometry, rows, len(self._atoms))
        return pair

    def getdefault(self, atom_pair):
        return self[atom_pair]

    def copy(self):
        return dict(self.items())

    def update_pairs(self):
        from .views._source import _check_view_source
        with self.mol._mutation_lock:
            _check_view_source(self.mol, self._view_source, self._atoms, self._view_uids)

    @property
    def pair_distance(self):
        if self._items is not None:
            return np.array([pair.distance for pair in self.values()])
        from .rules.mathematics.view_geometry import _pair_distances
        from .views._source import _check_view_source, _geometry_snapshot
        with self.mol._mutation_lock:
            _check_view_source(self.mol, self._view_source, self._atoms, self._view_uids)
            coordinates, identity = _geometry_snapshot(self.mol, self._atoms)
            if identity != self._geometry:
                raise RuntimeError('View geometry source changed; request a new view')
            return _pair_distances(coordinates)

    def pairs_shortest_path(self) -> dict:
        return {key: pair.length_shortest_path for key, pair in self.items()}

    @property
    def idx_matrix(self) -> np.ndarray:
        with self.mol._mutation_lock:
            self.update_pairs()
            return np.array([[atom.idx for atom in key] for key in self], dtype=np.int64)


######################################################################################
def _bond_order_setter(self: 'Bond', key, bond_order):
    assert key == 'bond_order'
    with _FactTransaction(self.mol, producer=_ATOM_SOURCE_CODES['explicit']):
        current_kind = self._bond_kind
        kind = (_bond_kind_from_order(bond_order) if current_kind in {
            BondKind.SINGLE, BondKind.DOUBLE, BondKind.TRIPLE, BondKind.AROMATIC, BondKind.UNKNOWN,
        } else current_kind)
        values = self._attrs.copy()
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
            return self._attrs.copy()

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
            groups = (self.mol._bonds, self.mol._hided_metal_bonds, self.mol._hided_covalent_bonds)
            hint = self._uid
            member = _bond_member_at(self.mol, hint)
            row = hint if member is self else None
            if row is None:
                offset = 0
                for group in groups:
                    local = next((i for i, bond in enumerate(group) if bond is self), None)
                    if local is not None:
                        row = offset + local
                        break
                    offset += len(group)
            if row is not None:
                return _restore_bond_member, (self.mol, row, self._uid)
            state = {key: value for key, value in self.__dict__.items()
                     if key not in ('_atoms', '_member_uid', '_uid_owner', '_attrs')}
            return _restore_unregistered_bond, (self.mol, self._uid, self.atom1, self.atom2, self._attrs.copy()), state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _set_bond_metadata(self, *, bond_kind, bond_direction=None, bond_source=None,
                           bond_source_metadata=None):
        metadata = deepcopy(bond_source_metadata or {})
        _commit_bond_facts(self.mol, self, _capture_source(self.mol), raw_values=self._attrs.copy(),
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
        return geometry_adapter.segment_from_bond(self)

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
        from .reference.properties import _ideal_bond_lengths

        ibl_bond_order_dict = _ideal_bond_lengths()[frozenset([self.atom1.atomic_number, self.atom2.atomic_number])]
        return ibl_bond_order_dict[self.bond_order]

    @property
    def length(self) -> float:
        """
            Calculates and returns the length of the vector between two atomic positions.
            Returns:
                float: The Euclidean distance (length) between the two atomic vectors.
        """
        from .operations.geometry import _bond_length
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
        from .operations.chemistry import _bond_aromatic_read
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

    def bond_segment_distance(self, other_bond: "Bond") -> float:
        """Return the minimum distance between two finite bond segments."""
        return geometry.segment_segment_distance(
            self.bond_segment,
            other_bond.bond_segment,
        )


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

    @property
    def degrees(self) -> float:
        from .rules.mathematics.internal_coordinates import InternalCoordinates
        from .views._source import _view_coordinates
        return InternalCoordinates.calc_angle(*_view_coordinates(self))


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

    @property
    def degrees(self) -> float:
        from .rules.mathematics.internal_coordinates import InternalCoordinates
        from .views._source import _view_coordinates
        return InternalCoordinates.calc_dehedral(*_view_coordinates(self))

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
        with self.mol._mutation_lock:
            self._check_topology_source()
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
        from .views._source import _check_ring_source
        _check_ring_source(self, prepare=True)

    def __reduce_ex__(self, protocol):
        from .views._source import _restore_joint_ring, _view_source_current
        with self.mol._mutation_lock:
            current = _view_source_current(self.mol, self._view_source, self._atoms, self._view_uids)
            return (_restore_joint_ring, (type(self), self._member_rings, self._atoms, self._bonds,
                     self._view_geometry, self._view_representation, current))

    def __init__(self, *ring):
        from .views._source import _bind_view, _ring_representation, _check_ring_source
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
        from .operations.chemistry import _check_joint_ring
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
        from .operations.chemistry import _prepare_aromatic_representation
        from .views._source import _bind_view, _ring_representation
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
        from .views._source import _check_ring_source
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
        from .operations.chemistry import _ring_aromatic_read
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
        return any(
            pair_distance.measurement.distance < _MINIMUM_ATOM_SEPARATION
            for pair_distance in geometry_adapter.measure_atom_pair_distances(
                self,
                pair_scope="all",
            )
        )

    @property
    def has_metal(self):
        from .views._source import _check_ring_source
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
        if not isinstance(other, Ring):
            raise TypeError(f"expected Ring but got {type(other)}")
        from .views._source import _check_ring_source
        with self.mol._mutation_lock:
            _check_ring_source(self, prepare=True)
            if other.mol is not self.mol:
                return False
            _check_ring_source(other, prepare=True)
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
        from .views._source import _check_ring_source
        with self.mol._mutation_lock:
            _check_ring_source(self, prepare=True)
            if aromatic and not self.is_aromatic:
                return None
            rings = []
            for ring in self.mol.rings:
                if ring == self:
                    rings.append(ring)
                elif self.joint_with(ring) and (not aromatic or ring.is_aromatic):
                    rings.append(ring)
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
        from .views._source import _view_coordinates
        with self.mol._mutation_lock:
            coordinates = _view_coordinates(self)
            return any(tuple(row) != self.atoms[0] for row in coordinates)


    def relation_to_bond(
            self,
            bond: Bond,
    ) -> "geometry_adapter.BondRingFinding[Ring, Bond]":
        """Return the factual spatial relation between this ring and a bond."""
        if self.mol is not bond.mol or bond not in self.mol.bonds:
            raise NotInSameMolecule(self, bond)
        return geometry_adapter.determine_bond_ring_relation(self, bond)

    @property
    def geometry_cycle(self) -> geometry.Cycle:
        """Return this ring as an ordered geometric cycle boundary."""
        return geometry_adapter.cycle_from_ring(self)

    def closest_edge_to_bond(
            self,
            bond: Bond,
    ) -> Optional["geometry_adapter.RingEdgeDistance[Bond]"]:
        """Return the closest ring edge together with its distance evidence."""
        if self.mol is not bond.mol or bond not in self.mol.bonds:
            raise NotInSameMolecule(self, bond)
        measurement = geometry.closest_cycle_edge(
            self.geometry_cycle,
            bond.bond_segment,
        )
        if measurement is None:
            return None
        return geometry_adapter.RingEdgeDistance(
            bond=self.bonds[measurement.edge_index],
            measurement=measurement,
        )



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
        self._mutation_lock = _RLock()
        for name in self._attrs:
            setattr(self, '_' + name, None)

    def __getstate__(self):
        with self._mutation_lock:
            return {key: value.copy() if isinstance(value, np.ndarray) else deepcopy(value)
                    for key, value in self.__dict__.items() if key != '_mutation_lock'}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mutation_lock = _RLock()

    def __len__(self):
        with self._mutation_lock:
            return 0 if self._coordinates is None else len(self._coordinates)

    def __getitem__(self, idx):
        with self._mutation_lock:
            info = {}
            for attr in self._attrs:
                try:
                    info[attr] = self.index_attr(attr, idx)
                except (IndexError, TypeError):
                    continue
            return info

    def __iter__(self):
        with self._mutation_lock:
            values = None if self._coordinates is None else self._coordinates.copy()
            return iter(values)

    def index_attr(self, name, i):
        with self._mutation_lock:
            try:
                value = getattr(self, f'_{name}')[i]
            except IndexError:
                raise IndexError(f'The number of conformers is {len(self._coordinates)}, but attempt to access the index {i}')
            except TypeError:
                try:
                    value = getattr(self, f'_{name}')
                except AttributeError:
                    raise AttributeError(f'The conformers did not store {name} information')
            return value.copy() if isinstance(value, np.ndarray) else deepcopy(value)

    def add(self, coords: types.ArrayLike, energy: Union[types.ArrayLike, float] = None):
        coords = np.array(coords)
        if coords.ndim == 2:
            coords = coords.reshape((1,) + coords.shape)
        assert coords.ndim == 3
        assert coords.shape[-1] == 3
        supplied_energy = energy is not None
        if supplied_energy:
            energy = np.array([energy]) if isinstance(energy, float) else np.array(energy).flatten()
        with self._mutation_lock:
            if isinstance(self._coordinates, np.ndarray) and self._coordinates.shape[1] == coords.shape[1]:
                coords = np.vstack((self._coordinates, coords))
            if supplied_energy:
                # Preserve the characterized route until the Stage7D replacement.
                if self._energy is not None:
                    energy = np.vstack((coords, coords))
                if len(energy) != len(coords):
                    raise ValueError('The length of input energy in conformer nust matches with the coordinates in conformer')
            previous = self._coordinates, self._energy
            try:
                self._coordinates = coords
                if supplied_energy:
                    self._energy = energy
            except BaseException:
                self._coordinates, self._energy = previous
                raise

    def clear(self):
        with self._mutation_lock:
            previous = tuple(getattr(self, '_' + name) for name in self._attrs)
            try:
                for name in self._attrs:
                    setattr(self, '_' + name, None)
            except BaseException:
                for name, value in zip(self._attrs, previous):
                    setattr(self, '_' + name, value)
                raise

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






if __name__ == "__main__":
    pass


# Stage3 carrier primitives. Chemistry stays in the existing facade bodies.
def _selected_atom_rows(mol, atoms, *, public=False):
    rows = []
    seen = set()
    for atom in atoms:
        if atom.mol is not mol:
            if public:
                raise NotInSameMolecule(mol, atom)
            raise ValueError('Selected Atom belongs to another Molecule')
        record = mol._topology_index.get(id(atom))
        if record is None:
            if public:
                raise ObjNotInMolecule(atom, mol)
            raise KeyError(id(atom))
        row = record[0]
        if (atom._uid_owner is not mol._owner_token or mol._atoms[row] is not atom
                or id(atom) in seen):
            raise ValueError('Atom selection requires unique current members')
        seen.add(id(atom))
        rows.append(row)
    return tuple(rows)


def _atom_numeric_value(atom, field):
    with atom.mol._mutation_lock:
        record = atom.mol._topology_index.get(id(atom))
        table, row = (atom._detached_table, 0) if record is None else (atom.mol._atom_table, record[0])
        return _ATOM_FIELD_CODECS[field](table.raw_index(row, _ATOM_FIELD_INDEX[field]))


_ATOM_FACT_ENGINE = None


def _atom_fact_engine():
    global _ATOM_FACT_ENGINE
    if _ATOM_FACT_ENGINE is None:
        engine = _fact_engine()
        engine._install_atom_dispatch(Atom, _atomic_number_setter, _commit_atom_sparse,
                                      NotInSameMolecule, ObjNotInMolecule)
        _ATOM_FACT_ENGINE = engine
    return _ATOM_FACT_ENGINE


def _store_atom_fields(atom, columns, *, producer=None):
    if columns.keys() & _ATOM_SCALAR_READONLY:
        raise AttributeError('Calculated Atom projections are read-only; use the complete Molecule assignment')
    engine = _atom_fact_engine()
    producer = _ATOM_SOURCE_CODES['explicit'] if producer is None else producer
    if len(columns) == 1:
        field, value = next(iter(columns.items()))
        engine._commit_atom_scalar(atom, _ATOM_FIELD_INDEX[field], value, producer=producer)
        return
    engine._commit_atom_fields(atom, columns, producer=producer)


def _store_bond_field(bond, field, value):
    with _FactTransaction(bond.mol, producer=_ATOM_SOURCE_CODES['explicit']):
        values = bond._attrs.copy()
        values[bond._attrs_enumerator.index(field)] = float(value)
        _commit_bond_facts(bond.mol, bond, _capture_source(bond.mol), raw_values=values,
            kind=bond._bond_kind, direction=bond._bond_direction,
            source_namespace=bond._bond_source, metadata=bond._bond_source_metadata)


def _initialize_atom_handle(atom, mol, uid):
    object.__setattr__(atom, '_mol', mol)
    object.__setattr__(atom, '_member_uid', uid)
    object.__setattr__(atom, '_uid_owner', mol._owner_token)


def _new_atom_handle(mol, uid):
    atom = object.__new__(Atom)
    _initialize_atom_handle(atom, mol, uid)
    return atom


def _import_atom_row(mol, values, *, presence=None, sources=None):
    """Install one supplied factual row through the private import boundary."""
    values = np.asarray(values, dtype=np.float64)
    prepared = _AtomFactTable(values[None, :], presence=presence, sources=sources)
    with _FactTransaction(mol, producer=_ATOM_SOURCE_CODES['legacy_array']) as transaction:
        atom = _new_atom_handle(mol, mol._allocate_atom_uid())
        transaction.touch_registration(atom)
        row = mol._atom_table._append_prepared(prepared.matrix(), prepared.presence(), prepared.sources())
        mol._atoms.append(atom)
        mol._topology_index[id(atom)] = (row, ())
        _mark_topology_dirty(mol, (atom._uid,))
        return atom


def _detached_atom_projection(atom):
    from .rules.chemistry.elements import _electron_configuration
    from .operations.chemistry import _atom_valence, _hydrogen_display_value
    table = atom._detached_table
    values = table.row(0)
    values[1:7] = _electron_configuration(atom.atomic_number)
    values[13] = _atom_valence(atom)
    field = _ATOM_FIELD_INDEX['implicit_hydrogens']
    from .structure._fields import FIELD_PRESENCE_BITS
    from .structure._dependencies import _FactStatus
    pending = any(atom._uid in seeds for seeds in table._dirty_seeds[:2])
    if (table._group_status[0, GROUP_INDEX['IMPLICIT_H']] != _FactStatus.CURRENT
            or not int(table._presence[0]) & FIELD_PRESENCE_BITS[field] or pending):
        values[field] = _hydrogen_display_value(table, 0, pending=pending)
    return values


def _initialize_bond_handle(bond, mol, uid, atom1, atom2, raw_values, *, kind,
                            direction, source_namespace, metadata, aromatic=None):
    object.__setattr__(bond, '_atoms', (atom1, atom2))
    object.__setattr__(bond, '_bonds', [bond])
    object.__setattr__(bond, '_member_uid', uid)
    object.__setattr__(bond, '_uid_owner', mol._owner_token)
    object.__setattr__(bond, '_attrs', np.array(raw_values, dtype=np.float64, copy=True))
    object.__setattr__(bond, '_bond_kind', BondKind(kind))
    object.__setattr__(bond, '_bond_direction', direction)
    object.__setattr__(bond, '_bond_source', source_namespace)
    object.__setattr__(bond, '_bond_source_metadata', metadata)
    object.__setattr__(bond, '_is_aromatic', kind == 'aromatic' if aromatic is None else bool(aromatic))


def _new_bond_handle(mol, uid, atom1, atom2, raw_values, *, kind, direction,
                     source_namespace, metadata, aromatic=None):
    bond = object.__new__(Bond)
    _initialize_bond_handle(bond, mol, uid, atom1, atom2, raw_values, kind=kind,
                            direction=direction, source_namespace=source_namespace, metadata=metadata, aromatic=aromatic)
    return bond


def _new_sequence_view(cls, atoms, rows, complete_paths, *, source, uids, geometry):
    from .views._source import _initialize_view_source, _member_geometry
    view = object.__new__(cls)
    members = tuple(atoms[row] for row in rows)
    object.__setattr__(view, '_atoms', members)
    object.__setattr__(view, '_bonds', tuple(complete_paths.get(frozenset((id(first), id(second))))
                                          for first, second in zip(members, members[1:])))
    _initialize_view_source(view, source, tuple(uids[row] for row in rows),
                            _member_geometry(geometry, rows, len(atoms)))
    return view


def _build_topology_index(atoms, bonds, hidden_metal=(), hidden_covalent=()):
    rows = {}
    incident = {}
    for row, atom in enumerate(atoms):
        if id(atom) in rows:
            raise ValueError('a Molecule cannot contain the same Atom twice')
        if atoms and atom.mol is not atoms[0].mol:
            raise ValueError('every Atom member must belong to its Molecule')
        rows[id(atom)] = row
        incident[id(atom)] = []
    complete = tuple(bonds) + tuple(hidden_metal) + tuple(hidden_covalent)
    hidden_ids = [id(bond) for bond in tuple(hidden_metal) + tuple(hidden_covalent)]
    if len(set(hidden_ids)) != len(hidden_ids):
        raise ValueError('a hidden Bond may occur in only one hidden collection')
    if {id(bond) for bond in bonds}.intersection(hidden_ids):
        raise ValueError('a Bond cannot be both active and hidden')
    for bond in complete:
        if id(bond.atom1) not in rows or id(bond.atom2) not in rows:
            raise ValueError('every Bond endpoint must belong to its Molecule')
    pairs = set()
    for bond in bonds:
        first, second = rows[id(bond.atom1)], rows[id(bond.atom2)]
        if first == second:
            raise ValueError('a Molecule Bond cannot join an Atom to itself')
        pair = tuple(sorted((first, second)))
        if pair in pairs:
            raise ValueError('a Molecule cannot contain parallel active Bonds')
        pairs.add(pair)
        incident[id(bond.atom1)].append(bond)
        incident[id(bond.atom2)].append(bond)
    return {id(atom): (rows[id(atom)], tuple(sorted(incident[id(atom)], key=lambda bond:
                rows[id(bond.atom2 if bond.atom1 is atom else bond.atom1)]))) for atom in atoms}


def _seal_detached_table(table):
    table._blocks = tuple(np.frombuffer(block.tobytes(), dtype=block.dtype).reshape(block.shape)
                          for block in table._blocks)
    from .structure._atom_table import _column_views
    table._columns = _column_views(table._blocks)
    table._presence = np.frombuffer(table._presence.tobytes(), dtype=table._presence.dtype)
    table._sources = np.frombuffer(table._sources.tobytes(), dtype=table._sources.dtype).reshape(table._sources.shape)
    table._group_status = np.frombuffer(table._group_status.tobytes(), dtype=np.uint8).reshape(table._group_status.shape)
    table._dirty_seeds = tuple(frozenset(seeds) for seeds in table._dirty_seeds)
    return table


def _detach_atom(atom):
    from .structure._dependencies import _FactStatus
    row, bonds = atom.mol._topology_index[id(atom)]
    table = atom.mol._atom_table.take((row,))
    local_uids = {atom._uid, *(bond.atom2._uid if bond.atom1 is atom else bond.atom1._uid for bond in bonds)}
    if (int(table._sources[0, _ATOM_FIELD_INDEX['implicit_hydrogens']]) == _ATOM_SOURCE_CODES['implicit_hydrogens']
            and table._group_status[0, GROUP_INDEX['IMPLICIT_H']] == _FactStatus.CURRENT
            and any(local_uids.intersection(seeds) for seeds in table._dirty_seeds[:2])):
        table._group_status[0, GROUP_INDEX['IMPLICIT_H']] = _FactStatus.NEEDS_RECALC
    table._dirty_seeds = tuple({atom._uid}.intersection(seeds) for seeds in table._dirty_seeds)
    object.__setattr__(atom, '_detached_table', _seal_detached_table(table))










def _capture_molecule_pickle(mol):
    with mol._mutation_lock:
        record, metadata = mol._capture_factual_record(include_frames=False, copy_metadata=False)
        bonds = tuple(mol._bonds + mol._hided_bonds)
        owned = {'_mutation_lock', '_owner_token', '_atoms', '_bonds', '_hided_metal_bonds', '_hided_covalent_bonds',
                 '_atom_table', '_topology_index', '_next_atom_uid', '_next_bond_uid', '_conformers',
                 '_conformers_index', '_stereo_payload', '_crystal', '_charge_value', '_charge_present',
                 '_charge_source', '_spin_value', '_spin_present', '_spin_source',
                 '_constraints', '_obmol', '_row2idx', '_pair_wiberg_bond_orders'}
        state = {key: value for key, value in mol.__dict__.items() if key not in owned}
        state['properties'] = dict(mol.properties)
        graph_memo = {id(mol): mol, **{id(atom): atom for atom in mol._atoms},
                      **{id(bond): bond for bond in bonds}}
        atom_state = tuple({key: value for key, value in atom.__dict__.items()
                            if key not in ('_mol', '_member_uid', '_uid_owner', '_detached_table', 'isotope', 'frac_valence')}
                           for atom in mol._atoms)
        bond_state = tuple({key: value for key, value in bond.__dict__.items()
                            if key not in ('_atoms', '_bonds', '_member_uid', '_uid_owner', '_attrs',
                                           '_bond_kind', '_bond_direction', '_bond_source', '_bond_source_metadata', '_is_aromatic')}
                           for bond in bonds)
        crystal_state = None if mol._crystal is None else dict(mol._crystal.__dict__)
        extra = (state, metadata, atom_state, bond_state, mol._capture_frame_state(), crystal_state)
        return (record, mol._owner_token, tuple(atom._uid for atom in mol._atoms),
                tuple(bond._uid for bond in bonds), (mol._next_atom_uid, mol._next_bond_uid),
                _dependency_state(mol._atom_table), extra)


def _restore_molecule_raw(cls, record, owner, atom_uids, bond_uids, high_water, dependencies):
    mol = cls()
    mol._owner_token = owner
    mol._install_factual_record(record, dependency_state=dependencies)
    for atom, uid in zip(mol._atoms, atom_uids):
        object.__setattr__(atom, '_member_uid', uid)
    for bond, uid in zip(mol._bonds + mol._hided_bonds, bond_uids):
        object.__setattr__(bond, '_member_uid', uid)
    mol._constraints = _ConstraintSet(
        tuple(tuple(atom_uids[row] for row in entry) for entry in record.angle_constraints),
        tuple(tuple(atom_uids[row] for row in entry) for entry in record.torsion_constraints))
    mol._pair_wiberg_bond_orders = {
        tuple(sorted((atom_uids[first], atom_uids[second]))): value
        for first, second, value in record.pair_wiberg_bond_orders}
    mol._next_atom_uid, mol._next_bond_uid = high_water
    return mol


def _restore_molecule_extra(mol, state):
    attributes, metadata, atom_state, bond_state, frames, crystal_state = state
    mol.__dict__.update(attributes)
    for atom, extra in zip(mol._atoms, atom_state):
        atom.__dict__.update(extra)
    for bond, extra, supplied in zip(mol._bonds + mol._hided_bonds, bond_state, metadata):
        bond.__dict__.update(extra)
        object.__setattr__(bond, '_bond_source_metadata', supplied)
    mol._install_frame_state(*frames)
    if crystal_state is not None:
        mol._crystal.__dict__.update(crystal_state)


def _restore_atom_member(mol, row, uid):
    if row >= len(mol._atoms) or mol._atoms[row]._uid != uid:
        raise RuntimeError('Serialized Atom membership changed during capture')
    return mol._atoms[row]


def _restore_bond_member(mol, row, uid):
    bond = _bond_member_at(mol, row)
    if bond is None or bond._uid != uid:
        raise RuntimeError('Serialized Bond membership changed during capture')
    return bond


def _bond_member_at(mol, row):
    for group in (mol._bonds, mol._hided_metal_bonds, mol._hided_covalent_bonds):
        if row < len(group):
            return group[row]
        row -= len(group)
    return None


def _restore_detached_atom(mol, uid, state):
    atom = _new_atom_handle(mol, uid)
    table = object.__new__(_AtomFactTable)
    table.__setstate__(state)
    object.__setattr__(atom, '_detached_table', _seal_detached_table(table))
    return atom


def _restore_unregistered_bond(mol, uid, first, second, values):
    return _new_bond_handle(mol, uid, first, second, values, kind=BondKind.UNKNOWN,
                            direction=None, source_namespace=None, metadata={})


def _numeric_property(field):
    index, codec = _ATOM_FIELD_INDEX[field], _ATOM_FIELD_CODECS[field]

    def read(atom):
        if field in _ATOM_DERIVED_FIELDS:
            if field == 'valence':
                return atom.get_valence()
            return int(atom.electron_configuration[index - 1])
        if field == 'is_aromatic':
            from .operations.chemistry import _atom_aromatic_read
            return _atom_aromatic_read(atom)
        if field == 'implicit_hydrogens':
            from .operations.chemistry import _implicit_hydrogen_read
            return _implicit_hydrogen_read(atom)
        mol = atom._mol
        with mol._mutation_lock:
            record = mol._topology_index.get(id(atom))
            if record is None:
                return codec(atom._detached_table._columns[index][0])
            return codec(mol._atom_table._columns[index][record[0]])

    return property(read)


def _constraint_property(scope):
    def read(view):
        from .views._source import _check_view_source
        with view.mol._mutation_lock:
            _check_view_source(view.mol, view._view_source, view.atoms, view._view_uids)
            return view._view_uids in getattr(view.mol._constraints, scope)

    def write(view, value):
        from .views._source import _check_view_source
        with view.mol._mutation_lock:
            _check_view_source(view.mol, view._view_source, view.atoms, view._view_uids)
            entries = ((view.atoms, value),)
            if scope == 'angles':
                _commit_constraints(view.mol, view._view_source[0], angle_entries=entries)
            else:
                _commit_constraints(view.mol, view._view_source[0], torsion_entries=entries)

    return property(read, write)


for _field in _ATOM_FIELDS:
    if _field not in Atom.__dict__:
        setattr(Atom, _field, _numeric_property(_field))


def _transactional_legacy(method, *, topology=False, frames=False, constraints=False):
    @_wraps(method)
    def guarded(self, *args, **kwargs):
        mol = self if isinstance(self, Molecule) else self.mol
        with _FactTransaction(mol, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            if topology:
                transaction.touch_topology()
                transaction.touch_dependencies()
            if frames:
                transaction.touch_frames()
            if constraints:
                transaction.touch_constraints()
            return method(self, *args, **kwargs)
    return guarded


for _method in (
        'add_atom', '_replace_atom', 'replace_atom', 'add_bond', 'add_bonds',
        'remove_atom', 'remove_atoms', 'remove_bond', 'remove_bonds', 'add_hydrogens',
        'clear_metal_ligand_bonds', 'clear_hided_covalent_bonds', 'clear_hided_bonds',
        'restore_bonds', 'recover_hided_metal_ligand_bonds', 'recover_hided_covalent_bonds',
        'hide_bonds', 'hide_metal_ligand_bonds', 'create_crystal'):
    setattr(Molecule, _method, _transactional_legacy(getattr(Molecule, _method), topology=True))
for _method in ('calc_atom_valence', 'calc_implicit_hydrogens', 'assign_partial_charge',
                'refresh_atom_id'):
    setattr(Molecule, _method, _transactional_legacy(getattr(Molecule, _method)))
for _method in ('conformer_add', 'conformer_clear'):
    setattr(Molecule, _method, _transactional_legacy(getattr(Molecule, _method), frames=True))
for _method in ('clear_constraints', 'constraint_bonds_angles'):
    setattr(Molecule, _method, _transactional_legacy(getattr(Molecule, _method), constraints=True))
Angle.constraint = _constraint_property('angles')
Torsion.constraint = _constraint_property('torsions')
del _field, _method
