"""Complete supplied aromatic representations for tests of read-only views."""

from dataclasses import fields, is_dataclass, replace
import struct

import numpy as np

from hotpot import Molecule
from hotpot.cheminfo.operations.chemistry import _bonding_snapshot
from hotpot.cheminfo.rules.chemistry.bonding.kekule import _aromatic_flags_assignment
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX


def immutable_snapshot(value):
    """Compare complete nested facts by declared fields and exact numeric bytes."""
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((field.name, immutable_snapshot(getattr(value, field.name)))
                                   for field in fields(value)))
    if isinstance(value, np.ndarray):
        return (type(value), value.dtype.str, tuple(value.shape), value.tobytes(order='C'))
    if isinstance(value, np.generic):
        return (type(value), value.dtype.str, value.tobytes())
    if type(value) is float:
        return (float, struct.pack('>d', value))
    if type(value) in (tuple, list):
        return (type(value), tuple(immutable_snapshot(item) for item in value))
    if type(value) is dict:
        return (dict, tuple((immutable_snapshot(key), immutable_snapshot(item))
                            for key, item in value.items()))
    return (type(value), value)


def factual_state(mol):
    record, metadata = mol._capture_factual_record()
    return immutable_snapshot(record), immutable_snapshot(metadata)


def supplied_aromatic_flags(mol, flags):
    flags = tuple(flags)
    assignment = _aromatic_flags_assignment(_bonding_snapshot(mol), flags)
    record, metadata = mol._capture_factual_record()
    values = np.frombuffer(record.atom_values, dtype='<f8').reshape(record.atom_count, 19).copy()
    presence = np.frombuffer(record.atom_presence, dtype='<u4').copy()
    sources = np.frombuffer(record.atom_sources, dtype='u1').reshape(record.atom_count, 19).copy()
    field = FIELD_INDEX['is_aromatic']
    values[:, field] = flags
    presence |= 1 << field
    sources[:, field] = ATOM_SOURCE_CODES['explicit']
    aromatic_by_uid = dict(zip(assignment.bond_uids, assignment.bond_aromatic))
    complete = mol._bonds
    return Molecule._build_molecule(replace(record,
        atom_values=values.tobytes(), atom_presence=presence.tobytes(), atom_sources=sources.tobytes(),
        bond_aromatic=tuple(aromatic_by_uid.get(bond._uid, bond._is_aromatic) for bond in complete)),
        bond_metadata=metadata)
