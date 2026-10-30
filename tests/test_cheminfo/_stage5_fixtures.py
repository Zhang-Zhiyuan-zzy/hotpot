"""Complete supplied aromatic representations for tests of read-only views."""

from dataclasses import fields, replace

import numpy as np

from hotpot import Molecule
from hotpot.cheminfo.operations.chemistry import _bonding_snapshot
from hotpot.cheminfo.rules.chemistry.bonding.kekule import _aromatic_flags_assignment
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX


def factual_state(mol):
    record, metadata = mol._capture_factual_record()
    return tuple(getattr(record, field.name) for field in fields(record)), metadata


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
    complete = mol._bonds + mol._hided_metal_bonds + mol._hided_covalent_bonds
    return Molecule._build_molecule(replace(record,
        atom_values=values.tobytes(), atom_presence=presence.tobytes(), atom_sources=sources.tobytes(),
        bond_aromatic=tuple(aromatic_by_uid.get(bond._uid, bond._is_aromatic) for bond in complete)),
        bond_metadata=metadata)
