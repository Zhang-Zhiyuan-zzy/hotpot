"""Canonical molecular facts and thin operation facades."""

from copy import copy, deepcopy
from functools import wraps as _wraps
from itertools import combinations
from threading import RLock as _RLock
from typing import Union, Literal, Iterable, Optional, Mapping

import numpy as np

from hotpot.utils import types
from ._fields import ATOM_FIELDS as _ATOM_FIELDS, ATOM_SOURCE_CODES as _ATOM_SOURCE_CODES
from ._atom_table import _AtomFactTable
from .constraints import _ConstraintSet, _split_atom_rows, _join_atom_rows, _filter_constraint_members
from ._topology import _build_topology_index, _selected_atom_rows
from ._identity import _capture_source, _check_source, _hydrogen_install_inputs
from ._results import (
    _current_mca_result, _copy_mca_result, _invalidate_results,
    _partial_charge_record, _install_partial_charge_record,
)
from ._mutation import _normalize_atom_facts
from ._commit import (
    _FactTransaction, _commit_coordinates, _commit_molecular_declarations,
    _commit_topology, _current_transaction,
)
from ._dependencies import (
    _install_dependency_state, _remap_dependency_state, _mark_topology_dirty, _seed, GROUP_INDEX,
)
from .stereo import StereoPayload as _StereoPayload
from ._stereo import _get_stereo_payload, _install_stereo_storage
from .conformer import (
    Conformers, _bound_conformers, _add_source_frames, _clear_frames, _load_frame,
    _current_frame_value, _assign_frame_value, _capture_frame_records,
    _install_frame_records, _copy_source_frames, _frame_spin,
)
from ._state import _capture_molecule_pickle, _restore_molecule_raw, _restore_molecule_extra


def _bind_declaration_contracts():
    from ..operations._carrier_signatures import (
        _forcefield_contracts, _geometry_types, _geometry_reports, _graph_defaults,
    )
    return (_graph_defaults(), _forcefield_contracts(), _geometry_types(), _geometry_reports())


# Existing eager annotation/default objects; calculations enter named operations.
graph_algorithms, ff, geometry, geometry_adapter = _bind_declaration_contracts()


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

_DEFAULT_BOND_RING_MAXIMUM_SIZE = 16


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

        self._atoms = []
        self._atom_table = _AtomFactTable()
        self._bonds = []
        self._topology_index = {}
        self._source_frames = ()
        self._conformer_results = ()
        self._frame_result = None
        self._conformers_index = 0
        self._conformers = _bound_conformers(self)

        self._mca_result = None
        self._partial_charge_result = None
        self._pair_wiberg_bond_orders = {}
        self._constraints = _ConstraintSet()
        self._crystal = None

        self.properties = {}  # To store any mol properties

        self._stereo_schema = 1
        self._stereo_records = ()
        self._stereo_source = ((), ())

        self._charge_value, self._charge_present, self._charge_source = 0, False, None
        self._spin_value, self._spin_present, self._spin_source = None, False, None

    def __dir__(self):
        return list(Conformers._attrs) + list(super(Molecule, self).__dir__())

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
            return _frame_spin(self)

    @spin_mult.setter
    def spin_mult(self, value):
        _commit_molecular_declarations(self, _capture_source(self), spin_mult=(True, value, 'explicit'))

    def updata_atoms(self, *, atoms=None, **columns) -> None:
        from .atom import _atom_fact_engine
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

    def _structural_bond_rows(self) -> tuple[tuple[int, int], ...]:
        atom_rows = {id(atom): row for row, atom in enumerate(self._atoms)}
        return tuple(
            (atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)])
            for bond in self._bonds
        )

    def _set_remapped_stereo_payload(
            self,
            source: "Molecule",
            atom_mapping: Mapping[int, int],
            *,
            filter_bonds: bool,
    ) -> None:
        from ._stereo import _get_stereo_payload
        from ..operations.chemistry import _remap_stereo_transport, _set_stereo_payload

        allowed_bonds = self._structural_bond_rows() if filter_bonds else None
        _set_stereo_payload(
            self,
            _remap_stereo_transport(
                _get_stereo_payload(source),
                atom_mapping,
                allowed_bonds=allowed_bonds,
            ),
        )

    def _replace_atom_transport(self, target_row, source_atom):
        """Install current occurrences and available source H at a member row."""
        from dataclasses import replace
        from .stereo import AtomTransportRecord, HydrogenTransportState
        from ..operations.chemistry import _set_stereo_payload

        with source_atom.mol._mutation_lock:
            source_row = source_atom.mol._topology_index.get(id(source_atom))
            if source_row is None:
                imported = tuple(AtomTransportRecord(
                    target_row, isotope, radical,
                    HydrogenTransportState(implicit, bracket, no_implicit) if present else None,
                    None, namespace)
                    for isotope, radical, present, implicit, bracket, no_implicit, namespace
                    in source_atom._detached_table._stereo_facts[0])
            else:
                source_payload = _get_stereo_payload(source_atom.mol)
                imported = () if source_payload is None else tuple(
                    replace(fact, atom=target_row) for fact in source_payload.atom_facts
                    if fact.atom == source_row[0])
        payload = _get_stereo_payload(self)
        if payload is None:
            if not imported:
                return
            payload = _StereoPayload(1, (), (), ())
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
        _set_stereo_payload(self, replace(payload, atom_facts=tuple(atom_facts)))

    def _remap_stereo_after_atom_removal(self, source_atoms, payload):
        from ..operations.chemistry import _remap_stereo_transport

        destination_rows = {id(atom): row for row, atom in enumerate(self._atoms)}
        mapping = {row: destination_rows[id(atom)] for row, atom in enumerate(source_atoms)
                   if id(atom) in destination_rows}
        _install_stereo_storage(self, None if payload is None else _remap_stereo_transport(payload, mapping))

    def _remove_stereo_bond_rows(
            self,
            removed_bond_rows: Iterable[tuple[int, int]],
    ) -> None:
        from .stereo import StereoAnchorKind
        from ._stereo import _get_stereo_payload
        from ..operations.chemistry import _remap_stereo_transport, _set_stereo_payload

        payload = _get_stereo_payload(self)
        if payload is None:
            return
        removed_keys = {
            frozenset(atom_indices) for atom_indices in removed_bond_rows
        }
        allowed_bonds = [rows for rows in self._structural_bond_rows()
                         if frozenset(rows) not in removed_keys]
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
        _set_stereo_payload(
            self,
            _remap_stereo_transport(
                payload,
                {row: row for row in range(len(self._atoms))},
                allowed_bonds=allowed_bonds,
            ),
        )

    def copy(self):
        with self._mutation_lock:
            if self._partial_charge_result is not None and not self._partial_charge_result.invalidated_by:
                _check_source(self, self._partial_charge_result.source)
            record, metadata = self._capture_factual_record(include_frames=False)
            dependencies = _remap_dependency_state(self._atom_table,
                tuple(atom._uid for atom in self._atoms), tuple(range(len(self._atoms))))
            clone = Molecule()
            _commit_topology(clone, _capture_source(clone), record,
                             atom_bindings=(None,) * record.atom_count,
                             bond_bindings=(None,) * len(record.bond_endpoints),
                             dependency_state=dependencies, bond_metadata=metadata)
            clone.properties = dict(self.properties)
            _copy_mca_result(self, clone)
            _copy_source_frames(self, clone)
            return clone

    @property
    def crystal(self):
        return self._crystal

    def create_crystal(self, a, b, c, alpha, beta, gamma, *, space_group=None,
                       site_relations=None, bond_images=None):
        from ..operations.crystal import _create_crystal
        return _create_crystal(self, (a, b, c, alpha, beta, gamma),
                               space_group, site_relations, bond_images)

    @staticmethod
    def read_one(src, fmt=None, **kwargs):
        """ Just read the first Molecule instance in the src """
        from ..operations.io import MolReader
        return next(MolReader(src, fmt, **kwargs))

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
        from ..operations.chemistry import _molecular_implicit_hydrogen_sum
        return _molecular_implicit_hydrogen_sum(self)

    @property
    def has_hydrogens(self) -> bool:
        """
        Indicates whether the object contains any hydrogens.
        @return: Returns True if there are hydrogens, False otherwise.
        """
        return len(self.hydrogens) > 0

    def calc_mol_default_charge(self):
        from ..operations.chemistry import _calculate_default_molecular_charge
        return _calculate_default_molecular_charge(self)



    def simple_paths(self, source=None, target=None, cutoff=None):
        from ..operations.topology_queries import _simple_paths
        return _simple_paths(self, source, target, cutoff)

    def _update_topology(self, clear_conformers: bool = True, *,
                         _prepared_index=None, _constraint_rows=None) -> None:
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            if _prepared_index is self._topology_index and _constraint_rows is None:
                transaction.touch_frames()
                transaction.touch_constraints()
            else:
                transaction.touch_topology()
            previous = self._topology_index
            self._topology_index = (_build_topology_index(self._atoms, self._bonds)
                if _prepared_index is None else _prepared_index)
            if _constraint_rows is not None:
                angle_rows, torsion_rows = _constraint_rows
                self._constraints = _ConstraintSet(
                    tuple(tuple(self._atoms[row]._uid for row in entry) for entry in angle_rows),
                    tuple(tuple(self._atoms[row]._uid for row in entry) for entry in torsion_rows),
                    self._constraints.atom_axes, self._constraints.bond_values)
            elif self._topology_index is not previous:
                uids = {atom._uid for atom in self._atoms}
                _filter_constraint_members(self)
                if any(first not in uids or second not in uids for first, second in self._pair_wiberg_bond_orders):
                    self._pair_wiberg_bond_orders = {
                        pair: value for pair, value in self._pair_wiberg_bond_orders.items()
                        if all(uid in uids for uid in pair)}
                changed = tuple(atom._uid for atom in self._atoms if
                    tuple(id(bond) for bond in previous.get(id(atom), (None, ()))[1]) !=
                    tuple(id(bond) for bond in self._topology_index[id(atom)][1]))
                if changed:
                    _mark_topology_dirty(self, changed)
            from ._crystal import _prune_crystal_members
            _prune_crystal_members(self)


    def _add_atom(self, atom):
        from .atom import _atom_fact_engine, _import_atom_row
        if atom.mol is self:
            raise ValueError('An Atom already owned by this Molecule cannot be inserted again')
        with atom.mol._mutation_lock:
            source_row = atom.mol._topology_index.get(id(atom))
            table = atom._detached_table if source_row is None else atom.mol._atom_table
            row = 0 if source_row is None else source_row[0]
            values, presence, sources = _join_atom_rows(table, atom._constraint_values,
                                                       (atom._uid,), rows=(row,))
            values = values[0]
            statuses = table._group_status[row].copy()
            pending_groups = tuple(atom._uid in seeds for seeds in table._dirty_seeds)
            sparse = {name: deepcopy(atom.__dict__[name]) for name in ('frac_valence',) if name in atom.__dict__}
            charge_result = (None if source_row is None else _partial_charge_record(
                atom.mol, (row,), whole_context=len(atom.mol._atoms) == 1 and not atom.mol._bonds))
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            copied = _import_atom_row(self, values, presence=presence, sources=sources,
                                      partial_charge_result=charge_result)
            new_row = self._topology_index[id(copied)][0]
            transaction.touch_dependency_status((new_row,), range(len(statuses)))
            self._atom_table._group_status[new_row] = statuses
            for group, pending_group in enumerate(pending_groups):
                _atom_fact_engine()._update_dependency_seeds(self, (group,), (copied._uid,), pending_group)
            for name, value in sparse.items():
                setattr(copied, name, value)
            self._replace_atom_transport(new_row, atom)
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
        from .atom import Atom
        from .bond import Bond
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
            transaction.touch_bond(bond)
            transaction.touch_constraint_value(self._constraints.bond_values, bond._uid)
            self._constraints.bond_values[bond._uid] = bond._constraint_values[bond._uid]
            object.__setattr__(bond, '_constraint_values', self._constraints.bond_values)
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
        atom = self._add_atom(atom)
        self._update_topology()
        return atom

    def _replace_atom(self, original_atom_idx: int, new_atom: Union[int, str, 'Atom']):
        from .atom import Atom, _detach_atom
        from .bond import Bond
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
        _detach_atom(original)
        self._replace_atom_transport(target_row, source_atom)
        from ..operations.chemistry import _remap_stereo_transport
        payload = _get_stereo_payload(self)
        if payload is not None:
            payload = _remap_stereo_transport(payload,
                                             {row: row for row in range(len(source_members))})
        old_uids = tuple(atom._uid for atom in self._atoms)
        selection = tuple(len(self._atoms) - 1 if row == target_row else row
                          for row in range(len(self._atoms) - 1))
        next_atoms = list(source_members)
        next_atoms[target_row] = replacement
        state = _remap_dependency_state(self._atom_table, old_uids,
                                        tuple(atom._uid for atom in next_atoms), rows=selection)
        table = self._atom_table.take(selection)
        _install_dependency_state(table, state)
        self._atom_table = table
        self._atoms[:] = next_atoms
        for bond in self._bonds:
            if bond.atom1 is original or bond.atom2 is original:
                Bond._replace_atom(bond, original, replacement)
        self._topology_index = _build_topology_index(self._atoms, self._bonds)
        _filter_constraint_members(self)
        _install_stereo_storage(self, payload)
        return original

    def build_all_pair_links(
            self,
            metal,
            threshold: Optional[float] = None,
            greedy=True,
            *,
            first_threshold: Optional[float] = None,
    ):
        from ..operations.inference.cbond.apply import build_all_possible_cbond
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
        from ..operations.inference.cbond.apply import auto_build_cbond
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
        from ..operations.topology import _replace_atom
        return _replace_atom(self, original_atom_idx, new_atom)

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
        from ..operations.topology import _add_component
        return _add_component(self, component)


    def add_hydrogens(
            self,
            rm_polar_hs: bool = True,
            rng: Optional[np.random.Generator] = None,
    ):
        from ..operations.chemistry import _add_hydrogens
        return _add_hydrogens(self, rm_polar_hs, rng)

    @property
    def cid(self) -> Optional[int]:
        from ..operations.descriptors import _cid
        return _cid(self)

    @property
    def name(self) -> str:
        from ..operations.descriptors import _name
        return _name(self)

    @property
    def cas(self):
        from ..operations.descriptors import _cas
        return _cas(self)

    def get_thermo(self, temp: float = 298.15, pressure: float = 101325.0) -> "Thermo":
        from ..operations.thermo import _get_thermo
        return _get_thermo(self, temp=temp, pressure=pressure)

    def clear_constraints(self) -> None:
        """Clear coordinate, complete Bond, and owned sequence constraints."""
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            transaction.touch_constraints()
            for atom in self.atoms:
                atom.constraint = False
            for bond in self.bonds:
                bond.constraint = False
            if self._constraints.angles or self._constraints.torsions:
                _invalidate_results(self, ('constraints',), transaction)
                self._constraints = _ConstraintSet(atom_axes=self._constraints.atom_axes,
                                                   bond_values=self._constraints.bond_values)

    @property
    def atom_pairs(self) -> "AtomPairs":
        """Return a fresh fixed pair table over the current molecular source."""
        from ..views.pairs import AtomPairs
        return AtomPairs(mol=self)

    @property
    def angles(self) -> list["Angle"]:
        """Return fresh source-bound views; persisted constraints have one owner."""
        from ..views.geometry import Angle
        from ..views.sequence import _new_sequence_view
        from ..operations.geometry import _angle_views
        return _angle_views(self, Angle, _new_sequence_view)

    @property
    def torsions(self) -> list["Torsion"]:
        """Return fresh source-bound torsion views without retaining a cache."""
        from ..views.geometry import Torsion
        from ..views.sequence import _new_sequence_view
        from ..operations.geometry import _torsion_views
        return _torsion_views(self, Torsion, _new_sequence_view)

    def assign_bond_order(self):
        from ..operations.chemistry import _assign_bond_orders
        return _assign_bond_orders(self)

    @property
    def atom_matrix(self) -> np.ndarray:
        from ..operations.chemistry import _atom_projection
        return _atom_projection(self)

    @property
    def atoms(self):
        with self._mutation_lock:
            return list(self._atoms)

    @property
    def c_bonds(self) -> list["Bond"]:
        """ Get all coordination bonds in the Molecule """
        from ..operations.topology_queries import _c_bonds
        return _c_bonds(self)

    @property
    def bonds(self) -> list["Bond"]:
        with self._mutation_lock:
            return list(self._bonds)

    def bond(self, a1idx: int, a2idx: int):
        """Return the Bond joining two Atom indices in the complete topology."""
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
    ):
        """Build and optimize this molecule with the appropriate FF workflow.

        ``candidate_count`` is reserved and currently has no effect.  Complex
        building uses one ligand starting geometry.  Future multi-conformer
        support will generate independent starts, optimize and gate each one,
        deduplicate or cluster them by geometry, rank them by topology,
        geometry, and energy evidence, and refine the selected conformer.
        """
        from ..operations.forcefields import build_and_optimize
        return build_and_optimize(
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
        )

    def update_mol_charge(self):
        """
        Updates the molecular charge by summing the charges of all constituent
        atoms. This method assumes the charges of individual atoms have been
        correctly assigned beforehand.

        """
        from ..operations.chemistry import _update_molecular_charge
        return _update_molecular_charge(self)

    @property
    def sum_atoms_charge(self) -> int:
        from ..operations.chemistry import _atom_charge_sum
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
        from ..operations.chemistry import _molecular_default_unpaired_electrons
        return _molecular_default_unpaired_electrons(self)

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
        from ..operations.chemistry import _molecular_default_spin_mult
        return _molecular_default_spin_mult(self)

    def determine_rings_aromatic(self):
        from ..operations.chemistry import _assign_aromaticity
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
        return _load_frame(self, i)

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
        return _add_source_frames(self, coords, energy)

    def conformer_clear(self):
        """
        Clears all stored conformers from the internal conformer storage.

        This method is used to empty the internal list of conformers,
        removing all previously stored data. This can be useful when resetting
        or preparing the object for storing a new set of conformers.

        Raises:
            No exceptions are raised by this method.
        """
        return _clear_frames(self)

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
        from ..operations.forcefields import auto_optimize
        return auto_optimize(
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
        from ..operations.chemistry import _calculate_valence_and_hydrogens
        return _calculate_valence_and_hydrogens(self, assign_aromatic)

    def calc_implicit_hydrogens(self):
        from ..operations.chemistry import _calculate_hydrogens
        _calculate_hydrogens(self, stacklevel=5)

    @property
    def components(self) -> list['Molecule']:
        from ..operations.topology_queries import _components
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
        from ..operations.geometry import _constraint_bonds_angles
        return _constraint_bonds_angles(self, exclude_metal_bonds)

    @property
    def coordinates(self) -> np.ndarray:
        """Independent Cartesian coordinates in angstrom from the Atom fact table."""
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

    @property
    def fractional_coordinates(self) -> np.ndarray:
        """Independent unwrapped fractional coordinates in the associated Crystal."""
        from ..operations.crystal import _coordinates
        with self._mutation_lock:
            if self._crystal is None:
                raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: fractional coordinates require an associated Crystal')
            return self._crystal.to_fractional(_coordinates(self))

    @fractional_coordinates.setter
    def fractional_coordinates(self, value: np.ndarray) -> None:
        with self._mutation_lock:
            if self._crystal is None:
                raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: fractional coordinates require an associated Crystal')
            self.coordinates = self._crystal.to_cartesian(value)

    def _create_atom_from_array(self, attrs_array: np.ndarray) -> "Atom":
        from .atom import _import_atom_row
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
        from .atom import Atom
        return Atom(self, **kwargs)

    def create_atom(self, **kwargs) -> 'Atom':
        with _FactTransaction(self, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            transaction.touch_frames()
            transaction.touch_constraints()
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
        from ..operations.conversion import _get_partial_charge
        return _get_partial_charge(self, model, model_args, OPENBABEL_PARTIAL_CHARGE_MODELS)

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
        from ..operations.conversion import _assign_partial_charge
        return _assign_partial_charge(self, model, model_args)

    @property
    def pair_dist(self) -> np.ndarray:
        from ..operations.geometry import _pair_distances
        return _pair_distances(self)

    @property
    def dist_matrix(self) -> np.ndarray:
        from ..operations.geometry import _distance_matrix
        return _distance_matrix(self)

    @property
    def element_counts(self):
        from ..operations.descriptors import _element_counts
        return _element_counts(self)

    def graph_spectral(self, norm: Literal['infinite', 'min', 'l1', 'l2'] = 'l2'):
        from ..operations.descriptors import _graph_spectral
        return _graph_spectral(self, norm)

    @property
    def formula(self) -> str:
        from ..operations.descriptors import _formula
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
        from ..operations.geometry import _is_disorder
        return _is_disorder(self)

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
        from ..operations.geometry import _molecule_has_3d
        return _molecule_has_3d(self)

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
        from ..operations.geometry import _molecule_bond_ring_piercing_state
        return _molecule_bond_ring_piercing_state(
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
        from ..operations.geometry import _molecule_bond_ring_relations
        return _molecule_bond_ring_relations(
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
        with self._mutation_lock:
            prediction = _current_mca_result(self)
            return {self._atoms[site.atom_index]: site.mca_kj_mol for site in prediction.sites}

    @property
    def is_organic(self) -> bool:
        """
        Determines whether a molecule is organic based on its atomic composition.

        Organic molecules are defined as containing no metal atoms, at least one
        hydrogen atom (explicit or implicit), and at least one carbon atom.

        @return: True if the molecule meets the criteria for being organic; otherwise, False
        @rtype: bool
        """
        from ..operations.descriptors import _is_organic
        return _is_organic(self)

    @property
    def is_full_halogenated(self) -> bool:
        """
        Determines whether the molecule is derived from an organic mol by replace all hydrogen
        to halogenated.
        """
        from ..operations.descriptors import _is_full_halogenated
        return _is_full_halogenated(self)

    def link_atoms(self, assign_bond_order: bool = True):
        """
        Links all atoms of the molecule together and updates the molecular graph.
        This method can also assign bond orders between atoms if specified.

        Args:
            assign_bond_order (bool): Indicates whether to assign bond orders.
                Defaults to True.
        """
        from ..operations.topology import _link_atoms
        return _link_atoms(self, assign_bond_order)

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
        from .atom import Atom, _detach_atom
        from .bond import _detach_bond
        source_atoms = tuple(self._atoms)
        payload = _get_stereo_payload(self)
        transaction = _current_transaction(self)
        if transaction is not None:
            transaction.touch_topology()
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
        for bond in self._bonds:
            if any(id(atom) in removed for atom in bond.atoms):
                _detach_bond(bond)
        self._bonds[:] = [bond for bond in self._bonds
                         if all(id(atom) not in removed for atom in bond.atoms)]
        self._topology_index = _build_topology_index(self._atoms, self._bonds)
        _filter_constraint_members(self)
        self._remap_stereo_after_atom_removal(source_atoms, payload)
        from ._crystal import _prune_crystal_members
        _prune_crystal_members(self)

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
        from .bond import _detach_bond
        bonds = tuple(bonds)
        removed_bond_rows = tuple(
            (bond.a1idx, bond.a2idx)
            for bond in bonds
        )
        displays = tuple(bond._display_facts for bond in bonds)
        _mark_topology_dirty(self, tuple(atom._uid for bond in bonds for atom in bond.atoms))
        self._remove_stereo_bond_rows(removed_bond_rows)
        for bond, display in zip(bonds, displays):
            row = self._bonds.index(bond)
            _detach_bond(self._bonds[row])
            object.__setattr__(bond, '_display_facts', display)
            self._bonds.pop(row)
        self._update_topology()

    def force_remove_polar_hydrogens(self):
        from ..operations.topology import _remove_hydrogens
        return _remove_hydrogens(self, polar_only=True)

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
        from ..operations.topology import _remove_hydrogens
        return _remove_hydrogens(self)

    def remove_metals(self):
        """
        Removes metal atoms from the structure by calling the remove_atoms method
        on the list of metal atoms.

        Returns:
            None
        """
        self.remove_atoms(self.metals)

    def search_substructure(self, smarts: str, *, semantics="full_graph"):
        from ..operations.search.search import _search_substructure
        return _search_substructure(self, smarts, semantics=semantics)

    def set_default_valence(self):
        from ..operations.chemistry import _prepare_valences
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
        from ..operations.descriptors import _similarity
        return _similarity(self, other, method, norm)

    def longest_path(self):
        from ..operations.topology_queries import _longest_path
        return _longest_path(self)

    def canonical_tree(self):
        from ..operations.topology_queries import _canonical_tree
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
        from ..operations.conversion import _inchi
        return _inchi(self)

    def shortest_path_index(self, atom1, atom2) -> list[int]:
        from ..operations.topology_queries import _shortest_path_indices
        return _shortest_path_indices(self, atom1, atom2)

    def shortest_path(self, atom1, atom2) -> list['Atom']:
        from ..operations.topology_queries import _shortest_path
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
        from ..operations.conversion import _smiles
        return _smiles(self)

    @property
    def kekulize_smiles(self) -> str:
        from ..operations.conversion import _smiles
        return _smiles(self, kekulize=True)

    @property
    def rings(self) -> list["Ring"]:
        return self._materialize_rings()

    @property
    def cycle_basis(self) -> list[tuple[int, ...]]:
        from ..operations.topology_queries import _cycle_basis
        return _cycle_basis(self)

    @property
    def all_cycle_basis(self) -> list[list[tuple[int, ...]]]:
        from ..operations.topology_queries import _all_cycle_basis
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
        from ..views.topology import Ring
        from ..operations.topology_queries import _rings_from_cycles
        return _rings_from_cycles(self, Ring, cycles)

    def _materialize_rings(
            self,
            *,
            ligand_skeleton: bool = False,
            max_size: Optional[int] = None,
            max_cycles: Optional[int] = graph_algorithms.DEFAULT_RELEVANT_CYCLE_LIMIT,
    ) -> list["Ring"]:
        from ..views.topology import Ring
        from ..operations.topology_queries import _ring_views
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
        from ..operations.topology_queries import _rings_for_scope
        return _rings_for_scope(self, ring_scope, max_size=max_size, max_cycles=max_cycles)

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
        from ..operations.topology_queries import _aromatic_joint_rings
        return _aromatic_joint_rings(self)

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
        from ..operations.topology_queries import _rings_small
        return _rings_small(self)

    @property
    def ligand_rings(self) -> list["Ring"]:
        return self._materialize_rings(ligand_skeleton=True)



    def to_pyg(self):
        """Return independent useful molecular tensors and structured observations."""
        from ..operations.conversion import _to_pyg
        return _to_pyg(self)

    @classmethod
    def from_pyg(cls, data, *, field_map=None):
        """Build molecular facts from the current declared PyG fields."""
        from ..operations.conversion import _from_pyg
        return _from_pyg(cls, data, field_map=field_map)

    def to_obmol(self) -> "ob.OBMol":
        """Return a fresh Open Babel projection of the current molecular facts."""
        from ..operations.conversion import mol2obmol
        return mol2obmol(self)[0]

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
        from ..operations.conversion import to_rdmol
        return to_rdmol(self)

    def to_pybel_mol(self) -> "pb.Molecule":
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
        from ..operations.conversion import _to_pybel_mol
        return _to_pybel_mol(self)

    def translation(self, vector: types.ArrayLike):
        """Translate current coordinates through the one coordinate commit."""
        from ..operations.geometry import _translate
        return _translate(self, vector)





    @property
    def weight(self):
        from ..operations.descriptors import _weight
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
        from ..operations.io import MolWriter
        writer = MolWriter(filename, fmt, overwrite=overwrite, **kwargs)
        return writer.write(self, write_single=write_single)


    def _capture_factual_record(self, *, rows=None, include_frames=True, copy_metadata=True, selection=None):
        from .bond import _raw_bond_values
        from ._builder import _make_molecule_record
        from ._selection import _selected_bonds
        from ..operations.chemistry import _remap_stereo_transport
        from ._crystal import _capture_crystal_record
        with self._mutation_lock:
            whole_capture = rows is None and selection is None
            rows = tuple(range(len(self._atoms))) if rows is None else tuple(rows)
            mapping = {row: target for target, row in enumerate(rows)}
            atoms = tuple(self._atoms[row] for row in rows)
            complete = tuple(self._bonds)
            selected = complete if selection is None else _selected_bonds(self, selection)
            bonds = tuple(bond for bond in selected
                          if self._topology_index[id(bond.atom1)][0] in mapping
                          and self._topology_index[id(bond.atom2)][0] in mapping)
            endpoints = tuple((mapping[self._topology_index[id(bond.atom1)][0]],
                               mapping[self._topology_index[id(bond.atom2)][0]]) for bond in bonds)
            whole = rows == tuple(range(len(self._atoms)))
            charge_context = whole and len(bonds) == len(complete) and all(
                first is second for first, second in zip(bonds, complete))
            payload = _get_stereo_payload(self)
            if not whole_capture and payload is not None:
                payload = _remap_stereo_transport(payload, mapping,
                                                allowed_bonds=None if whole else endpoints)
            source_frames, conformer_results, frame_selection = (
                _capture_frame_records(self, rows=rows) if include_frames else ((), (), None))
            uid_rows = {atom._uid: row for row, atom in enumerate(self._atoms)}
            def constraints(entries):
                return tuple(tuple(mapping[uid_rows[uid]] for uid in entry)
                             for entry in entries if all(uid in uid_rows and uid_rows[uid] in mapping
                                                         for uid in entry))
            metadata = tuple(bond._bond_source_metadata for bond in bonds)
            if copy_metadata:
                metadata = deepcopy(metadata)
            else:
                memo = {id(self): self, **{id(atom): atom for atom in self._atoms},
                        **{id(bond): bond for bond in complete}}
                metadata = deepcopy(metadata, memo)
            atom_values, atom_presence, atom_sources = _join_atom_rows(
                self._atom_table, self._constraints.atom_axes,
                tuple(atom._uid for atom in atoms), rows=rows)
            record = _make_molecule_record(
                atom_values=atom_values,
                atom_presence=atom_presence, atom_sources=atom_sources,
                partial_charge_result=_partial_charge_record(self, rows, whole_context=charge_context),
                atom_frac_valences=tuple((row, atom.__dict__['frac_valence']) for row, atom in enumerate(atoms) if 'frac_valence' in atom.__dict__),
                bond_endpoints=endpoints, bond_values=np.array([_raw_bond_values(bond) for bond in bonds], dtype=np.float64).reshape(-1, 3),
                bond_kinds=tuple(bond._bond_kind.value for bond in bonds),
                bond_directions=tuple(bond._bond_direction for bond in bonds),
                bond_sources=tuple(bond._bond_source for bond in bonds), bond_metadata=(None,) * len(bonds),
                bond_aromatic=tuple(bond._is_aromatic for bond in bonds),
                charge=(True, self._charge_value, self._charge_source) if whole and self._charge_present else (False, None, None),
                spin_mult=(True, self._spin_value, self._spin_source) if whole and self._spin_present else (False, None, None),
                cell=None if self._crystal is None else _capture_crystal_record(self._crystal, atoms, bonds),
                stereo_payload=payload, source_frames=source_frames,
                conformer_results=conformer_results, frame_selection=frame_selection,
                angle_constraints=constraints(self._constraints.angles), torsion_constraints=constraints(self._constraints.torsions),
                identifier=('identifier' in self.__dict__, self.__dict__.get('identifier')),
                pair_wiberg_bond_orders=tuple((mapping[uid_rows[first]], mapping[uid_rows[second]], value)
                    for (first, second), value in self._pair_wiberg_bond_orders.items()
                    if first in uid_rows and second in uid_rows
                    and uid_rows[first] in mapping and uid_rows[second] in mapping))
            return record, metadata


    def _install_factual_record(self, record, *, atom_bindings=None, bond_bindings=None,
                                 dependency_state=None, bond_metadata=None, _install_frames=True, _install_crystal=True):
        from .atom import _detach_atom, _new_atom_handle
        from .bond import _detach_bond, _initialize_bond_handle, _new_bond_handle
        from ._builder import _thaw_canonical
        atom_bindings = (None,) * record.atom_count if atom_bindings is None else tuple(atom_bindings)
        bond_bindings = (None,) * len(record.bond_endpoints) if bond_bindings is None else tuple(bond_bindings)
        if len(atom_bindings) != record.atom_count or len(bond_bindings) != len(record.bond_endpoints):
            raise ValueError('Factual bindings must align with the import record')
        _invalidate_results(self, ('representation', 'geometry'))
        old_atoms = tuple(self._atoms)
        old_bonds = tuple(self._bonds)
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
        table_values, presence, sources, atom_cells = _split_atom_rows(
            np.frombuffer(record.atom_values, dtype='<f8').reshape(record.atom_count, len(_ATOM_FIELDS)),
            presence=np.frombuffer(record.atom_presence, dtype='<u4'),
            sources=np.frombuffer(record.atom_sources, dtype='u1').reshape(record.atom_count, len(_ATOM_FIELDS)))
        table = _AtomFactTable(table_values, presence=presence, sources=sources)
        first_atom, first_bond = self._next_atom_uid, self._next_bond_uid
        self._next_atom_uid += sum(atom is None for atom in atom_bindings)
        self._next_bond_uid += sum(bond is None for bond in bond_bindings)
        atom_uids = iter(range(first_atom, self._next_atom_uid))
        bond_uids = iter(range(first_bond, self._next_bond_uid))
        atoms = tuple(_new_atom_handle(self, next(atom_uids), cell=cell) if atom is None else atom
                      for atom, cell in zip(atom_bindings, atom_cells))
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
        topology_index = _build_topology_index(atoms, bonds)
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
        retained_bonds = {id(bond) for bond in bonds}
        for bond in old_bonds:
            if id(bond) not in retained_bonds:
                _detach_bond(bond)
        atom_axes = {atom._uid: cell for atom, cell in zip(atoms, atom_cells)}
        bond_values = {bond._uid: bond._constraint_values[bond._uid] for bond in bonds}
        sparse_valences = dict(record.atom_frac_valences)
        transaction = _current_transaction(self)
        for row, atom in enumerate(atoms):
            if transaction is not None:
                transaction.touch_sparse(atom)
                if atom_bindings[row] is None:
                    transaction.touch_constraint_binding(atom)
            atom.__dict__.pop('_detached_table', None)
            object.__setattr__(atom, '_constraint_values', atom_axes)
            if row in sparse_valences:
                atom.__dict__['frac_valence'] = sparse_valences[row]
            else:
                atom.__dict__.pop('frac_valence', None)
        for bond, binding in zip(bonds, bond_bindings):
            if transaction is not None and binding is None:
                transaction.touch_bond(bond)
            object.__setattr__(bond, '_constraint_values', bond_values)
        self._constraints = _ConstraintSet(atom_axes=atom_axes, bond_values=bond_values)
        self._pair_wiberg_bond_orders = pair_wiberg_bond_orders
        self._atom_table = table
        self._atoms[:] = atoms
        self._bonds[:] = bonds
        self._topology_index = topology_index
        # A missing complete-record transport retains the established schema1 default.
        _install_stereo_storage(self, record.stereo_payload if record.stereo_payload is not None
                                else _StereoPayload(1, (), (), ()))
        self._update_topology(clear_conformers=False, _prepared_index=topology_index,
                             _constraint_rows=(record.angle_constraints, record.torsion_constraints))
        for name, declaration in (('charge', record.charge), ('spin', record.spin_mult)):
            present, value, source = declaration
            object.__setattr__(self, '_' + name + '_present', present)
            object.__setattr__(self, '_' + name + '_value', value if present else (0 if name == 'charge' else None))
            object.__setattr__(self, '_' + name + '_source', source)
        if dependency_state is None:
            _mark_topology_dirty(self, tuple(atom._uid for atom in atoms))
        elif old_hydrogen_inputs:
            changed = tuple(uid for uid, inputs in _hydrogen_install_inputs(self).items()
                            if uid in old_hydrogen_inputs and old_hydrogen_inputs[uid] != inputs)
            if changed:
                _seed(self, GROUP_INDEX['IMPLICIT_H'], changed)
        if _install_crystal:
            from ._crystal import _install_crystal_record
            _install_crystal_record(self, record.cell)
        if _install_frames:
            _install_frame_records(self, record.source_frames, record.conformer_results, record.frame_selection)
        _install_partial_charge_record(self, record.partial_charge_result)
        if record.identifier[0]:
            self.identifier = record.identifier[1]
        else:
            self.__dict__.pop('identifier', None)
        return self


def _frame_observation_property(name):
    def read(mol):
        with mol._mutation_lock:
            return _current_frame_value(mol, name)

    def write(mol, value):
        _assign_frame_value(mol, name, value)

    return property(read, write)


for _observation in ('energy', 'partial_charge', 'gibbs', 'force', 'zero_point', 'thermo', 'capacity'):
    setattr(Molecule, _observation, _frame_observation_property(_observation))
del _observation


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
        'remove_atom', 'remove_atoms', 'remove_bond', 'remove_bonds', 'add_hydrogens'):
    setattr(Molecule, _method, _transactional_legacy(getattr(Molecule, _method), topology=True))

for _method in ('calc_atom_valence', 'calc_implicit_hydrogens',
                'refresh_atom_id'):
    setattr(Molecule, _method, _transactional_legacy(getattr(Molecule, _method)))

for _method in ('conformer_add', 'conformer_clear'):
    setattr(Molecule, _method, _transactional_legacy(getattr(Molecule, _method), frames=True))

for _method in ('clear_constraints', 'constraint_bonds_angles'):
    setattr(Molecule, _method, _transactional_legacy(getattr(Molecule, _method), constraints=True))
del _method
