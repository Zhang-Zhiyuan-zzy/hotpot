"""Task-local source bindings for the staged fact-store migration.

Identities describe the current owners, not a second retained molecular state.
Until Stage7B, complete Bond order is active, hidden-metal, hidden-covalent;
moving a Bond between those partitions may conservatively change an identity.
Until Stage7D, source frames retain their legacy coordinate-only meaning. The
forcefield_inputs domain additionally covers every retained numeric/sparse
input consumed before its final Stage5/7 ownership cutover. Captures and
commits share one owner lock; table columns are captured directly once.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json

import numpy as np

from ._fields import FIELD_INDEX, ATOM_FIELDS, FIELD_PRESENCE_BITS, FACT_FIELDS


__all__ = ()


_DOMAINS = ("topology", "representation", "geometry", "cell", "constraints", "source_frames", "forcefield_inputs", "bonding_inputs")
_REPRESENTATION_FIELDS = tuple(FIELD_INDEX[name] for name in (
    "atomic_number", "formal_charge", "implicit_hydrogens", "is_aromatic",
))
_COORDINATE_FIELDS = tuple(FIELD_INDEX[name] for name in ("x", "y", "z"))
_CONSTRAINT_FIELDS = tuple(FIELD_INDEX[name] for name in (
    "x_constraint", "y_constraint", "z_constraint",
))
_BONDING_FIELDS = (*_REPRESENTATION_FIELDS, FIELD_INDEX['partial_charge'], *_COORDINATE_FIELDS)


@dataclass(frozen=True, eq=False)
class _SourceIdentity:
    owner: object
    dependencies: tuple[tuple[str, str], ...]


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
    if value is None or isinstance(value, (str, bool, int, float)):
        return [type(value).__name__, value]
    raise TypeError(f"Unsupported source identity value: {type(value).__name__}")


def _digest(domain, value):
    encoded = json.dumps([domain, _canonical(value)], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _complete_bonds(mol):
    return tuple(mol._bonds) + tuple(mol._hided_metal_bonds) + tuple(mol._hided_covalent_bonds)


def _topology(mol, bonds):
    return (tuple(atom._uid for atom in mol._atoms),
            tuple((bond._uid, bond.atom1._uid, bond.atom2._uid) for bond in bonds))


def _atom_columns(mol, indices):
    fields = tuple(ATOM_FIELDS[index] for index in indices)
    return mol._atom_table.select(fields)


def _atom_metadata(mol, indices):
    table = mol._atom_table
    mask = sum(FIELD_PRESENCE_BITS[index] for index in indices)
    return (table._presence[:len(table)] & mask,
            table._sources[:len(table), indices])


def _declaration(mol, prefix):
    return tuple(mol.__dict__[f"_{prefix}_{suffix}"] for suffix in ("present", "value", "source"))



def _current_spin(mol):
    if mol._spin_present:
        return "declared", _declaration(mol, "spin")
    if mol._conformers._spin_mult is not None:
        return "selected_legacy_frame", mol._conformers.index_attr("spin_mult", mol._conformers_index)
    return "absent", None



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
        tuple(atom._uid for atom in mol._atoms), mol._stereo_payload)}
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


def _content_identities(mol, domains):
    bonds = _complete_bonds(mol) if any(domain != "cell" for domain in domains) else ()
    topology = _topology(mol, bonds) if bonds or any(domain != "cell" for domain in domains) else None
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
                       _hydrogen_transport_inputs(tuple(atom._uid for atom in mol._atoms), mol._stereo_payload))
        elif domain == "representation":
            content = (
                topology,
                _atom_columns(mol, _REPRESENTATION_FIELDS),
                _atom_metadata(mol, _REPRESENTATION_FIELDS),
                tuple((bond.bond_order, bond.bond_kind, bond._is_aromatic, bond.bond_direction,
                       bond.bond_source, bond.bond_source_metadata) for bond in bonds),
                mol._stereo_payload,
                tuple((atom._uid, atom.__dict__["isotope"])
                      for atom in mol._atoms if "isotope" in atom.__dict__),
                ("charge", _declaration(mol, "charge")),
                _current_spin(mol),
            )
        elif domain == "geometry":
            content = (topology, _atom_columns(mol, _COORDINATE_FIELDS),
                       _atom_metadata(mol, _COORDINATE_FIELDS))
        elif domain == "cell":
            cell = mol._crystal
            content = None if cell is None else tuple(
                getattr(cell, name) for name in ("a", "b", "c", "alpha", "beta", "gamma")
            )
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
            with mol._conformers._mutation_lock:
                content = topology, mol._conformers._coordinates
                identities.append((domain, _digest(domain, content)))
                continue
        else:  # Closed legacy numeric input domain; opaque metadata is separate.
            indices = tuple(FIELD_INDEX[field] for field in FACT_FIELDS if FIELD_INDEX[field] not in _COORDINATE_FIELDS)
            with mol._conformers._mutation_lock:
                content = (
                    topology, _atom_columns(mol, indices), _atom_metadata(mol, indices),
                    tuple((atom._uid, tuple((name, atom.__dict__[name])
                                           for name in ("isotope", "frac_valence") if name in atom.__dict__))
                          for atom in mol._atoms),
                    tuple((bond._uid, bond._attrs, bond.bond_kind, bond._is_aromatic,
                           bond.bond_direction, bond.bond_source)
                          for bond in bonds),
                    _declaration(mol, "charge"), _current_spin(mol), mol._stereo_payload,
                    tuple(getattr(mol._conformers, name) for name in (
                        "_energy", "_partial_charge", "_spin_mult", "_gibbs", "_force",
                        "_zero_point", "_thermo", "_capacity")),
                    mol._conformers_index,
                    tuple(sorted(mol._pair_wiberg_bond_orders.items())),
                )
                identities.append((domain, _digest(domain, content)))
                continue
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
        with mol._conformers._mutation_lock:
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
        with mol._conformers._mutation_lock:
            current = _content_identities(mol, domains)
        for (domain, previous), (_, observed) in zip(source.dependencies, current):
            if previous != observed:
                raise RuntimeError(
                    f"Molecular source dependency {domain!r} changed; recalculate before committing"
                )
