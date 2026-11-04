"""Complete placed-coordinate maps validated by the sole identification rule."""

from dataclasses import replace

import numpy as np

from ...structure._fields import ATOM_FIELDS, FIELD_INDEX, FIELD_PRESENCE_BITS
from ...structure.stereo import StereoPayload
from ...structure.symmetry import SpaceGroup
from .stereochemistry.algebra import _remap_stereo_payload
from .crystal_identification import (
    _IdentificationMismatch, _IdentificationUnavailable, _identify_generated_record,
)


__all__ = ()
_XYZ = tuple(FIELD_INDEX[name] for name in ('x', 'y', 'z'))


def _coordinate_domains(first, second, tolerance):
    left = np.frombuffer(first.atom_values, dtype='<f8').reshape(-1, len(ATOM_FIELDS))[:, _XYZ]
    right = np.frombuffer(second.atom_values, dtype='<f8').reshape(-1, len(ATOM_FIELDS))[:, _XYZ]
    return tuple(tuple(map(int, np.flatnonzero(np.linalg.norm(right - point, axis=1) <= tolerance)))
                 for point in left)


def _matching_unavailable(record):
    presence = np.frombuffer(record.atom_presence, dtype='<u4')
    element = FIELD_PRESENCE_BITS[FIELD_INDEX['atomic_number']]
    if np.any((presence & element) == 0):
        return 'required element facts are absent'
    return None


def _pair_record(first, second):
    """Transient disjoint union; no live results or invented source bindings."""
    offset = first.atom_count
    payloads = (first.stereo_payload, None if second.stereo_payload is None else
                _remap_stereo_payload(second.stereo_payload, {row: offset + row for row in range(second.atom_count)}))
    present = tuple(payload for payload in payloads if payload is not None)
    payload = None if not present else StereoPayload(present[0].schema_version,
        tuple(entry for item in present for entry in item.records),
        tuple(entry for item in present for entry in item.atom_facts),
        tuple(entry for item in present for entry in item.bond_facts))
    count = first.atom_count + second.atom_count
    edges = first.bond_endpoints + tuple((a + offset, b + offset) for a, b in second.bond_endpoints)
    group = SpaceGroup(1)
    origin = (first.cell[1], group, (), (), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    cell = (first.cell[0], first.cell[1], group,
            ((), first.cell[3][1] + second.cell[3][1], first.cell[3][2] + second.cell[3][2]),
            ((0, 0, 0),) * len(edges), 'cell', origin, (None,) * count, (None,) * len(edges))
    return replace(first, atom_count=count, atom_values=first.atom_values + second.atom_values,
        atom_presence=first.atom_presence + second.atom_presence,
        atom_sources=first.atom_sources + second.atom_sources,
        atom_frac_valences=first.atom_frac_valences + tuple((row + offset, value) for row, value in second.atom_frac_valences),
        bond_endpoints=edges, bond_values=first.bond_values + second.bond_values,
        bond_kinds=first.bond_kinds + second.bond_kinds,
        bond_directions=first.bond_directions + second.bond_directions,
        bond_sources=first.bond_sources + second.bond_sources,
        bond_metadata=first.bond_metadata + second.bond_metadata,
        bond_aromatic=first.bond_aromatic + second.bond_aromatic,
        charge=(False, None, None), spin_mult=(False, None, None), cell=cell, stereo_payload=payload,
        angle_constraints=first.angle_constraints + tuple(tuple(row + offset for row in entry) for entry in second.angle_constraints),
        torsion_constraints=first.torsion_constraints + tuple(tuple(row + offset for row in entry) for entry in second.torsion_constraints),
        partial_charge_result=None, source_frames=(), conformer_results=(), frame_selection=None,
        pair_wiberg_bond_orders=())


def _complete_mapping(record, mapping, count, tolerance):
    classes = tuple((row, count + mapping[row]) for row in range(count))
    try:
        _identify_generated_record(record, classes, coordinate_tolerance=tolerance)
    except _IdentificationMismatch:
        return False, None
    except _IdentificationUnavailable as exc:
        return None, str(exc)
    return True, None
