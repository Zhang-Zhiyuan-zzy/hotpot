"""Closed factual commits and one operation-local undo transaction.

The held owner lock is the publication boundary. Legacy eager setters may read
staged values, while nested commits join the same journal and external readers
wait until publication or restoration completes.
"""

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import fields, is_dataclass
from struct import unpack

import numpy as np

from ._fields import ATOM_SOURCE_CODES, COORDINATE_FIELDS, FIELD_INDEX
from ._identity import _check_source
from ._atom_table import _restore_atom_write
from ._dependencies import _mark_atom_facts

__all__ = ()

_UNCHANGED = object()
_EXPLICIT_SOURCE = ATOM_SOURCE_CODES["explicit"]
_TRANSACTIONS = ContextVar("hotpot_fact_transactions", default=())
_DECLARATIONS = ("_charge_value", "_charge_present", "_charge_source",
                 "_spin_value", "_spin_present", "_spin_source")
_FRAME_FIELDS = ("_coordinates", "_energy", "_partial_charge", "_spin_mult", "_gibbs",
                 "_force", "_zero_point", "_thermo", "_capacity")
_TOPOLOGY_FIELDS = ("_atoms", "_bonds", "_hided_metal_bonds", "_hided_covalent_bonds",
                    "_angles", "_torsions", "_rings", "_ring_indices_cache", "_ligand_rings",
                    "_ligand_rings_signature", "_obmol", "_row2idx", "_crystal", "_atom_pairs",
                    "identifier", "properties")
_BOND_FIELDS = ("_bond_kind", "_bond_direction", "_bond_source", "_bond_source_metadata")
_CACHE_FIELDS = ("_angles", "_torsions", "_rings", "_ring_indices_cache", "_ligand_rings",
                 "_ligand_rings_signature", "_obmol", "_row2idx")


def _current_transaction(mol):
    for transaction in reversed(_TRANSACTIONS.get()):
        if transaction.mol is mol:
            return transaction
    return None


def _restore_fields(owner, state):
    for name, value in state:
        if value is _UNCHANGED:
            owner.__dict__.pop(name, None)
        else:
            owner.__dict__[name] = value


def _table_storage(table):
    return tuple(getattr(table, name) for name in table.__slots__)


def _restore_storage(table, state):
    for name, value in zip(table.__slots__, state):
        setattr(table, name, value)


class _FactTransaction:
    """One lock and closed-domain undo journal; never retained on a carrier."""

    __slots__ = ("mol", "producer", "_joined", "_token", "_cells", "_statuses", "_seeds",
                 "_bonds", "_sparse", "_declarations", "_stereo", "_frames", "_topology",
                 "_registrations", "_bond_membership", "_cache_state", "_constraints", "_failed")

    def __init__(self, mol, *, producer):
        self.mol = mol
        self.producer = producer
        self._joined = None
        self._token = None

    def _initialize_journal(self):
        self._cells = []
        self._statuses = self._seeds = self._bonds = self._sparse = None
        self._declarations = None
        self._stereo = _UNCHANGED
        self._frames = None
        self._topology = None
        self._registrations = None
        self._bond_membership = None
        self._cache_state = None
        self._constraints = None
        self._failed = False

    def __enter__(self):
        self.mol._mutation_lock.acquire()
        parent = _current_transaction(self.mol)
        if parent is not None:
            self._joined = parent
            return parent
        self._initialize_journal()
        self._token = _TRANSACTIONS.set(_TRANSACTIONS.get() + (self,))
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if self._joined is not None:
                if kind is not None:
                    self._joined._failed = True
                return False
            if kind is not None or self._failed:
                self._rollback()
                if kind is None:
                    raise RuntimeError("A nested factual transaction failed; the operation was rolled back")
        finally:
            if self._token is not None:
                _TRANSACTIONS.reset(self._token)
            if self._joined is None:
                self._clear()
            self.mol._mutation_lock.release()
        return False

    def _clear(self):
        self._cells.clear()
        self._statuses = self._seeds = self._bonds = self._sparse = None
        self._constraints = self._registrations = None
        self._declarations = self._frames = self._topology = None
        self._bond_membership = None
        self._cache_state = None
        self._stereo = _UNCHANGED
        self._joined = self._token = None

    def touch_dependency_status(self, rows, groups):
        status = self.mol._atom_table._group_status
        if self._statuses is None:
            self._statuses = {}
        for row in rows:
            for group in groups:
                key = (id(status), row, group)
                if key not in self._statuses:
                    self._statuses[key] = (status, status[row, group])

    def touch_dependency_seeds(self, group, uids):
        seeds = self.mol._atom_table._dirty_seeds[group]
        if self._seeds is None:
            self._seeds = {}
        for uid in uids:
            key = (id(seeds), uid)
            if key not in self._seeds:
                self._seeds[key] = (seeds, uid in seeds)

    def touch_dependencies(self):
        table = self.mol._atom_table
        self.touch_dependency_status(range(len(table)), range(len(table._dirty_seeds)))
        for group, seeds in enumerate(table._dirty_seeds):
            self.touch_dependency_seeds(group, tuple(seeds))

    def touch_bond(self, bond):
        if self._bonds is None:
            self._bonds = {}
        if id(bond) not in self._bonds:
            self._bonds[id(bond)] = (bond, bond._attrs.copy(),
                                    tuple((name, bond.__dict__.get(name, _UNCHANGED))
                                          for name in _BOND_FIELDS), bond.atom1, bond.atom2)

    def touch_sparse(self, atom):
        if self._sparse is None:
            self._sparse = {}
        if id(atom) not in self._sparse:
            self._sparse[id(atom)] = (atom, tuple((name, atom.__dict__.get(name, _UNCHANGED))
                                                for name in ("isotope", "frac_valence")))

    def touch_declarations(self):
        if self._declarations is None:
            self._declarations = tuple((name, self.mol.__dict__.get(name, _UNCHANGED))
                                       for name in _DECLARATIONS)

    def touch_stereo(self):
        if self._stereo is _UNCHANGED:
            self._stereo = self.mol._stereo_payload

    def touch_frames(self):
        if self._frames is not None:
            return
        frames = self.mol._conformers
        with frames._mutation_lock:
            self._frames = (frames, self.mol._conformers_index,
                            tuple((name, None if getattr(frames, name) is None
                                   else np.array(getattr(frames, name), copy=True))
                                  for name in _FRAME_FIELDS))

    def touch_constraints(self):
        for view in tuple(self.mol._angles) + tuple(self.mol._torsions):
            self.touch_constraint(view)

    def touch_constraint(self, view):
        if self._constraints is None:
            self._constraints = {}
        if id(view) not in self._constraints:
            self._constraints[id(view)] = (view, view.__dict__.get('constraint', _UNCHANGED))

    def touch_caches(self):
        if self._topology is not None or self._cache_state is not None:
            return
        values = []
        for name in _CACHE_FIELDS:
            value = self.mol.__dict__[name]
            contents = list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else None
            values.append((name, value, contents))
        self._cache_state = tuple(values)
        self.touch_frames()
        self.touch_constraints()

    def touch_registration(self, atom):
        if self._topology is not None:
            return
        table = self.mol._atom_table
        if self._registrations is None:
            self._registrations = []
        self._registrations.append((atom, table, _table_storage(table), len(self.mol._atoms)))

    def touch_bond_membership(self, atoms):
        if self._topology is not None:
            return
        if self._bond_membership is None:
            self._bond_membership = (self.mol._bonds, len(self.mol._bonds), self.mol._topology_index, {})
        index, entries = self._bond_membership[2:]
        for atom in atoms:
            entries.setdefault(id(atom), index[id(atom)])

    def touch_topology(self):
        if self._topology is not None:
            return
        mol = self.mol
        values = []
        for name in _TOPOLOGY_FIELDS:
            value = mol.__dict__.get(name, _UNCHANGED)
            contents = list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else None
            values.append((name, value, contents))
        table = mol._atom_table
        self._topology = (table, _table_storage(table), mol._topology_index,
                          dict(mol._topology_index), tuple(values),
                          tuple((atom, atom.__dict__.get("_detached_table", _UNCHANGED)) for atom in mol._atoms))
        for bond in tuple(mol._bonds) + tuple(mol._hided_metal_bonds) + tuple(mol._hided_covalent_bonds):
            self.touch_bond(bond)
        for view in tuple(mol._angles) + tuple(mol._torsions):
            self.touch_constraint(view)
        self.touch_declarations()
        self.touch_stereo()
        self.touch_frames()
        self.touch_dependencies()

    def _rollback(self):
        mol = self.mol
        if self._topology is not None:
            table, storage, index, entries, values, detached = self._topology
            _restore_storage(table, storage)
            mol._atom_table = table
            for name, value, contents in values:
                if contents is not None:
                    if isinstance(value, list):
                        value[:] = contents
                    else:
                        value.clear()
                        value.update(contents)
                _restore_fields(mol, ((name, value),))
            index.clear()
            index.update(entries)
            mol._topology_index = index
            for atom, value in detached:
                _restore_fields(atom, (("_detached_table", value),))
        if self._bond_membership is not None:
            bonds, length, index, entries = self._bond_membership
            del bonds[length:]
            index.update(entries)
            mol._bonds, mol._topology_index = bonds, index
        for atom, table, storage, length in reversed(self._registrations or ()):
            _restore_storage(table, storage)
            del mol._atoms[length:]
            mol._topology_index.pop(id(atom), None)
            atom.__dict__.pop("_detached_table", None)
        for state in reversed(self._cells):
            _restore_atom_write(state)
        for (_, row, group), (status, value) in (self._statuses or {}).items():
            status[row, group] = value
        for (_, uid), (seeds, present) in (self._seeds or {}).items():
            if present:
                seeds.add(uid)
            else:
                seeds.discard(uid)
        for bond, raw, fields, first, second in (self._bonds or {}).values():
            bond.__dict__["_attrs"] = raw
            _restore_fields(bond, fields)
            bond.__dict__["_atoms"] = (first, second)
        for atom, fields in (self._sparse or {}).values():
            _restore_fields(atom, fields)
        if self._declarations is not None:
            _restore_fields(mol, self._declarations)
        if self._stereo is not _UNCHANGED:
            mol._stereo_payload = self._stereo
        if self._frames is not None:
            frames, index, fields = self._frames
            with frames._mutation_lock:
                _restore_fields(frames, fields)
            mol._conformers, mol._conformers_index = frames, index
        for view, constraint in (self._constraints or {}).values():
            _restore_fields(view, (("constraint", constraint),))
        if self._cache_state is not None:
            for name, value, contents in self._cache_state:
                if contents is not None:
                    if isinstance(value, list):
                        value[:] = contents
                    else:
                        value.clear()
                        value.update(contents)
                _restore_fields(mol, ((name, value),))


def _check_atom_rows(mol, update):
    atoms, topology, owner = mol._atoms, mol._topology_index, mol._owner_token
    for row, uid in zip(update.rows, update.atom_uids):
        if row >= len(atoms):
            raise ValueError("Atom update member is no longer current")
        atom = atoms[row]
        if (atom._mol is not mol or atom._member_uid != uid or atom._uid_owner is not owner
                or topology.get(id(atom), (None,))[0] != row):
            raise ValueError("Atom update member or UID is no longer current")


def _commit_atom_facts(mol, update):
    with _FactTransaction(mol, producer=_EXPLICIT_SOURCE) as transaction:
        _check_source(mol, update.source)
        _check_atom_rows(mol, update)
        if len(update.rows) == 1:
            selection = update.rows[0]
            if len(update.fields) == 1:
                code = update.sources[0]
                prepared = ((FIELD_INDEX[update.fields[0]], unpack("=d", update.value_bytes[0])[0],
                             update.presence[0], transaction.producer if code == _EXPLICIT_SOURCE else code),)
            else:
                prepared = tuple((FIELD_INDEX[field], unpack("=d", value)[0], present,
                                  transaction.producer if code == _EXPLICIT_SOURCE else code)
                                 for field, value, present, code in
                                 zip(update.fields, update.value_bytes, update.presence, update.sources))
        else:
            selection = (slice(0, len(update.rows)) if update.rows == tuple(range(len(mol._atoms)))
                         else np.asarray(update.rows, dtype=np.intp))
            prepared = tuple((FIELD_INDEX[field], np.frombuffer(value, dtype=np.float64), present,
                              transaction.producer if code == _EXPLICIT_SOURCE else code)
                             for field, value, present, code in
                             zip(update.fields, update.value_bytes, update.presence, update.sources))
        mol._atom_table._write_prepared(selection, prepared, transaction=transaction)
        _mark_atom_facts(mol, update.rows, update.fields, transaction=transaction,
                         atom_uids=update.atom_uids)


def _commit_coordinates(mol, update):
    if any(field not in COORDINATE_FIELDS for field in update.fields):
        raise ValueError("Coordinate updates require canonical XYZ fields")
    _commit_atom_facts(mol, update)


def _commit_atom_sparse(mol, atom, *, isotope=_UNCHANGED, frac_valence=_UNCHANGED):
    from ._dependencies import _seed, GROUP_INDEX
    with _FactTransaction(mol, producer=_EXPLICIT_SOURCE) as transaction:
        entry = mol._topology_index.get(id(atom))
        if atom.mol is not mol or entry is None or mol._atoms[entry[0]] is not atom:
            raise ValueError("Sparse Atom update requires a current member")
        transaction.touch_sparse(atom)
        if isotope is not _UNCHANGED:
            atom.__dict__["isotope"] = isotope
        if frac_valence is not _UNCHANGED:
            atom.__dict__["frac_valence"] = frac_valence
            _seed(mol, GROUP_INDEX["VALENCE"], (atom._uid,))


def _commit_bond_facts(mol, bond, binding, *, raw_values, kind, direction,
                       source_namespace, metadata):
    from ._dependencies import _mark_topology_dirty
    raw_values = np.array(raw_values, dtype=np.float64, copy=True)
    if raw_values.shape != (3,):
        raise ValueError("Bond facts require the complete raw three-slot row")
    metadata = deepcopy(metadata)
    with _FactTransaction(mol, producer=_EXPLICIT_SOURCE) as transaction:
        _check_source(mol, binding)
        if bond.mol is not mol:
            raise ValueError("Bond facts require the current owner")
        for atom in (bond.atom1, bond.atom2):
            entry = mol._topology_index.get(id(atom))
            if atom.mol is not mol or entry is None or mol._atoms[entry[0]] is not atom:
                raise ValueError("Bond endpoints must be current members")
        transaction.touch_bond(bond)
        bond.__dict__["_attrs"] = raw_values
        for name, value in zip(_BOND_FIELDS, (kind, direction, source_namespace, metadata)):
            bond.__dict__[name] = value
        _mark_topology_dirty(mol, (bond.atom1._uid, bond.atom2._uid))


def _commit_molecular_declarations(mol, binding, *, charge=_UNCHANGED, spin_mult=_UNCHANGED):
    from ._dependencies import _mark_declarations_dirty
    with _FactTransaction(mol, producer=_EXPLICIT_SOURCE) as transaction:
        _check_source(mol, binding)
        transaction.touch_declarations()
        for prefix, value in (("charge", charge), ("spin", spin_mult)):
            if value is not _UNCHANGED:
                present, declared, source = value
                for suffix, item in (("present", present), ("value", declared), ("source", source)):
                    mol.__dict__[f"_{prefix}_{suffix}"] = item
        _mark_declarations_dirty(mol)


def _immutable_stereo_tree(value):
    if is_dataclass(value):
        return all(_immutable_stereo_tree(getattr(value, field.name)) for field in fields(value))
    if type(value) is tuple:
        return all(_immutable_stereo_tree(item) for item in value)
    return type(value) is not list


def _commit_stereo(mol, source, payload):
    from ._builder import _stereo
    sealed = _stereo(payload)
    payload = payload if _immutable_stereo_tree(payload) else sealed
    with _FactTransaction(mol, producer=_EXPLICIT_SOURCE) as transaction:
        _check_source(mol, source)
        transaction.touch_stereo()
        mol._stereo_payload = payload


def _commit_constraints(mol, source, *, atom_update=None, bond_entries=(),
                        angle_entries=(), torsion_entries=()):
    with _FactTransaction(mol, producer=_EXPLICIT_SOURCE) as transaction:
        _check_source(mol, source)
        for entries, membership in ((angle_entries, mol._angles), (torsion_entries, mol._torsions)):
            for view, _ in entries:
                if not any(member is view for member in membership):
                    raise ValueError("Constraint view is no longer current")
        if atom_update is not None:
            if any(field not in ("x_constraint", "y_constraint", "z_constraint") for field in atom_update.fields):
                raise ValueError("Atom constraints require canonical constraint axes")
            _commit_atom_facts(mol, atom_update)
        for bond, value in bond_entries:
            if not any(member is bond for member in tuple(mol._bonds) + tuple(mol._hided_metal_bonds) + tuple(mol._hided_covalent_bonds)):
                raise ValueError("Constraint Bond is no longer current")
            transaction.touch_bond(bond)
            bond._attrs[2] = value
        for view, value in tuple(angle_entries) + tuple(torsion_entries):
            transaction.touch_constraint(view)
            view.__dict__["constraint"] = value


def _commit_topology(mol, source, record, *, atom_bindings, bond_bindings,
                     dependency_state=None, bond_metadata=None):
    with _FactTransaction(mol, producer=_EXPLICIT_SOURCE) as transaction:
        _check_source(mol, source)
        transaction.touch_topology()
        mol._install_factual_record(record, atom_bindings=atom_bindings,
                                bond_bindings=bond_bindings, dependency_state=dependency_state,
                                bond_metadata=bond_metadata)
