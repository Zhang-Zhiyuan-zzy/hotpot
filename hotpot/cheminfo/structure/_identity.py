"""Task-local source bindings for the staged fact-store migration.

Identities describe the current owners, not a second retained molecular state.
Complete Bond order follows the sole authoritative Molecule Bond list.
Source-frame identities inspect only explicit source values and provenance.
Generated attachment applicability excludes starting geometry and source frames;
historical operation input is retained separately. Captures and commits share
one owner lock; table columns are captured directly once. The private MCA
representation excludes unused spin from the shared facts.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json

import numpy as np

from ._fields import FIELD_INDEX, ATOM_FIELDS, FIELD_PRESENCE_BITS, FACT_FIELDS
from ._source_identity import _SourceIdentity
from ._stereo import _get_stereo_payload
from .constraints import _select_atom_values, _select_atom_metadata, _bond_raw_values


__all__ = ()


_DOMAINS = ("topology", "representation", "geometry", "cell", "constraints", "source_frames", "forcefield_inputs", "bonding_inputs", "mca_representation", "charge_representation", "frame_representation")
_REPRESENTATION_FIELDS = tuple(FIELD_INDEX[name] for name in (
    "atomic_number", "formal_charge", "implicit_hydrogens", "is_aromatic",
))
_COORDINATE_FIELDS = tuple(FIELD_INDEX[name] for name in ("x", "y", "z"))
_CONSTRAINT_FIELDS = tuple(FIELD_INDEX[name] for name in (
    "x_constraint", "y_constraint", "z_constraint",
))
_BONDING_FIELDS = (*_REPRESENTATION_FIELDS, *_COORDINATE_FIELDS)


def _canonical(value):
    """Encode supported scientific/source values without object repr or hash."""
    if isinstance(value, Enum):
        return ["enum", type(value).__module__, type(value).__qualname__, _canonical(value.value)]
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return ["array", value.dtype.str, value.shape, _canonical(value.tolist())]
        return ["array", value.dtype.str, value.shape, value.tobytes(order="C").hex()]
    if is_dataclass(value) and not isinstance(value, type):
        return ["record", type(value).__module__, type(value).__qualname__,
                [(field.name, _canonical(getattr(value, field.name))) for field in fields(value)]]
    if isinstance(value, Mapping):
        entries = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        entries.sort(key=lambda entry: json.dumps(entry[0], sort_keys=True, separators=(",", ":")))
        return ["mapping", entries]
    if isinstance(value, (tuple, list)):
        return ["sequence", [_canonical(item) for item in value]]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if value is None or isinstance(value, (str, bool, int, float)):
        return [type(value).__name__, value]
    raise TypeError(f"Unsupported source identity value: {type(value).__name__}")


def _digest(domain, value):
    encoded = json.dumps([domain, _canonical(value)], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _complete_bonds(mol):
    return tuple(mol._bonds)


def _topology(mol, bonds):
    return (tuple(atom._uid for atom in mol._atoms),
            tuple((bond._uid, bond.atom1._uid, bond.atom2._uid) for bond in bonds))


def _atom_columns(mol, indices):
    fields = tuple(ATOM_FIELDS[index] for index in indices)
    return _select_atom_values(mol._atom_table, mol._constraints.atom_axes,
                               tuple(atom._uid for atom in mol._atoms), fields)


def _atom_metadata(mol, indices):
    return _select_atom_metadata(mol._atom_table, mol._constraints.atom_axes,
                                 tuple(atom._uid for atom in mol._atoms), indices)


def _declaration(mol, prefix):
    return tuple(mol.__dict__[f"_{prefix}_{suffix}"] for suffix in ("present", "value", "source"))



def _current_spin(mol):
    from .conformer import _frame_spin
    if mol._spin_present:
        return "declared", _declaration(mol, "spin")
    spin = _frame_spin(mol)
    return ("selected_frame", spin) if spin is not None else ("absent", None)



def _hydrogen_transport_inputs(atom_uids, payload):
    """Effective current H-rule inputs; historical transport is not an owner."""
    facts = {} if payload is None else {fact.atom: fact for fact in payload.atom_facts}
    inputs = []
    for row, uid in enumerate(atom_uids):
        fact = facts.get(row)
        current = None if fact is None else fact.current_hydrogens
        inputs.append((uid, 0 if fact is None or fact.radical_electrons is None else fact.radical_electrons,
                       0 if current is None or current.bracket_explicit_hydrogens is None
                       else current.bracket_explicit_hydrogens,
                       False if current is None or current.no_implicit is None else current.no_implicit))
    return tuple(inputs)


def _hydrogen_inputs(atom, transport_inputs):
    """Immutable local inputs for H only; other bonding projections stay intact."""
    mol = atom.mol
    entry = mol._topology_index.get(id(atom))
    table, row, bonds = ((atom._detached_table, 0, ()) if entry is None
                         else (mol._atom_table, entry[0], entry[1]))
    number_field = FIELD_INDEX['atomic_number']
    charge_field = FIELD_INDEX['formal_charge']
    def number(index):
        return int(table._columns[number_field][index]) if (
            int(table._presence[index]) & FIELD_PRESENCE_BITS[number_field]) else 0
    neighbours, orders, kinds, flags, roles = [], [], [], [], []
    for bond in bonds:
        other = bond.atom2 if bond.atom1 is atom else bond.atom1
        neighbours.append(number(mol._topology_index[id(other)][0]))
        orders.append(float(bond._attrs[0]))
        kinds.append(bond._bond_kind.value)
        flags.append(bond._is_aromatic)
        direction = bond._bond_direction
        donor = (bond.atom1 if direction == 'atom1_to_atom2'
                 else bond.atom2 if direction == 'atom2_to_atom1' else None)
        roles.append(None if donor is None else 'donor' if donor is atom else 'acceptor')
    radical, bracket, no_implicit = transport_inputs.get(atom._uid, (0, 0, False))
    charge = int(table._columns[charge_field][row]) if (
        int(table._presence[row]) & FIELD_PRESENCE_BITS[charge_field]) else 0
    return (number(row), tuple(neighbours), tuple(orders),
            bool(table._columns[FIELD_INDEX['is_aromatic']][row]), charge,
            tuple(kinds), tuple(flags), tuple(roles), radical, bracket, no_implicit)



def _hydrogen_install_inputs(mol):
    """Bindings for retained H status across a complete factual installation."""
    transport = {entry[0]: entry[1:] for entry in _hydrogen_transport_inputs(
        tuple(atom._uid for atom in mol._atoms), _get_stereo_payload(mol))}
    table = mol._atom_table
    field = FIELD_INDEX['implicit_hydrogens']
    inputs = {}
    for atom in mol._atoms:
        row, bonds = mol._topology_index[id(atom)]
        inputs[atom._uid] = (
            _hydrogen_inputs(atom, transport),
            tuple((bond._uid, bond.atom2._uid if bond.atom1 is atom else bond.atom1._uid) for bond in bonds),
            bool(int(table._presence[row]) & FIELD_PRESENCE_BITS[field]),
            int(table._sources[row, field]), table._columns[field][row].tobytes(),
        )
    return inputs


def _stereo_inputs(mol):
    """Capture canonical owners directly; no assembled transport mirror."""
    return (mol._stereo_schema, mol._stereo_records, mol._stereo_source,
            tuple(mol._atom_table._stereo_facts[:len(mol._atoms)]),
            tuple((bond._uid, bond._display_facts) for bond in mol._bonds))


def _partial_charge_inputs(mol):
    attachment = mol._partial_charge_result
    return None if attachment is None else (attachment.payload, attachment.invalidated_by)


def _content_identities(mol, domains):
    needs_topology = any(domain not in ("cell", "source_frames") for domain in domains)
    bonds = _complete_bonds(mol) if needs_topology else ()
    topology = _topology(mol, bonds) if needs_topology else None
    identities = []
    for domain in domains:
        if domain == "topology":
            content = topology
        elif domain == 'bonding_inputs':
            # Closed numeric dependencies of the Stage5 rules/OB perception.
            # Opaque Bond annotations and legacy result frames are not inputs.
            content = (topology, _atom_columns(mol, _BONDING_FIELDS),
                       _atom_metadata(mol, _BONDING_FIELDS),
                       tuple((bond._uid, bond.bond_order, bond.bond_kind, bond._is_aromatic,
                              bond.bond_direction, bond.bond_source) for bond in bonds),
                       _hydrogen_transport_inputs(tuple(atom._uid for atom in mol._atoms), _get_stereo_payload(mol)),
                       _partial_charge_inputs(mol))
        elif domain in ("representation", "mca_representation", "charge_representation", "frame_representation"):
            content = (
                topology,
                _atom_columns(mol, _REPRESENTATION_FIELDS),
                _atom_metadata(mol, _REPRESENTATION_FIELDS),
                tuple((bond.bond_order, bond.bond_kind, bond._is_aromatic, bond.bond_direction,
                       bond.bond_source, bond.bond_source_metadata) for bond in bonds),
                _stereo_inputs(mol),
                ("charge", _declaration(mol, "charge")),
            )
            if domain == "representation":
                content += (_current_spin(mol),)
            elif domain == "frame_representation":
                content += (("spin", _declaration(mol, "spin")),)
        elif domain == "geometry":
            content = (topology, _atom_columns(mol, _COORDINATE_FIELDS),
                       _atom_metadata(mol, _COORDINATE_FIELDS))
        elif domain == "cell":
            cell = mol._crystal
            content = None if cell is None else cell._state[0]
        elif domain == "constraints":
            content = (
                topology,
                tuple(tuple(bool(value) for value in row)
                      for row in _atom_columns(mol, _CONSTRAINT_FIELDS)),
                tuple((bond._uid, bool(bond.constraint)) for bond in bonds),
                mol._constraints.angles,
                mol._constraints.torsions,
            )
        elif domain == "source_frames":
            from .conformer import _source_frame_inputs
            content = _source_frame_inputs(mol)
        else:  # Closed numeric input domain; opaque metadata is separate.
            from .conformer import _frame_numeric_inputs
            indices = tuple(FIELD_INDEX[field] for field in FACT_FIELDS if FIELD_INDEX[field] not in _COORDINATE_FIELDS)
            content = (
                topology, _atom_columns(mol, indices), _atom_metadata(mol, indices),
                tuple((atom._uid, tuple((name, atom.__dict__[name])
                                       for name in ("frac_valence",) if name in atom.__dict__))
                      for atom in mol._atoms),
                tuple((bond._uid, _bond_raw_values(bond._attrs, bond._constraint_values[bond._uid]), bond.bond_kind, bond._is_aromatic,
                       bond.bond_direction, bond.bond_source) for bond in bonds),
                _declaration(mol, "charge"), _current_spin(mol), _stereo_inputs(mol),
                _frame_numeric_inputs(mol),
                tuple(sorted(mol._pair_wiberg_bond_orders.items())), _partial_charge_inputs(mol),
            )
        identities.append((domain, _digest(domain, content)))
    return tuple(identities)


def _capture_source(mol, domains=()):
    """Capture only named dependencies once; an empty tuple binds owner only."""
    domains = tuple(dict.fromkeys(domains))
    for domain in domains:
        if domain not in _DOMAINS:
            raise ValueError(f"Unknown source dependency domain: {domain!r}")
    with mol._mutation_lock:
        if not domains:
            return _SourceIdentity(mol._owner_token, ())
        return _SourceIdentity(mol._owner_token, _content_identities(mol, domains))


def _check_source(mol, source):
    """Reject a foreign owner or a changed captured dependency before commit."""
    with mol._mutation_lock:
        if source.owner is not mol._owner_token:
            raise RuntimeError("Molecular source owner changed; recalculate before committing")
        if not source.dependencies:
            return
        domains = tuple(domain for domain, _ in source.dependencies)
        for domain in domains:
            if domain not in _DOMAINS:
                raise ValueError(f"Unknown source dependency domain: {domain!r}")
        current = _content_identities(mol, domains)
        for (domain, previous), (_, observed) in zip(source.dependencies, current):
            if previous != observed:
                raise RuntimeError(
                    f"Molecular source dependency {domain!r} changed; recalculate before committing"
                )
