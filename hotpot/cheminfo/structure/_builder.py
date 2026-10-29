"""Private complete import records and a closed immutable value codec."""

from __future__ import annotations

from dataclasses import dataclass, fields
import operator
from typing import TYPE_CHECKING

import numpy as np

from ._fields import ATOM_FIELDS, ATOM_PRESENCE_ALL, ATOM_SOURCE_CODES

if TYPE_CHECKING:
    from ..stereo import StereoPayload


__all__ = ()


_BOND_KINDS = frozenset({"single", "double", "triple", "aromatic", "zero", "dative", "unknown"})
_SCALAR_TYPES = {"bool": bool, "int": int, "float": float, "str": str, "bytes": bytes}


def _bytes(value):
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("Record buffers require bytes-like values")
    return bytes(value)


def _dtype(value):
    dtype = np.dtype(value)
    if (dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None
            or dtype.metadata is not None):
        raise TypeError("Value codec requires an ordinary non-object NumPy dtype")
    return dtype


def _freeze_value(value):
    """Encode supported values into immutable, explicitly tagged tuples."""
    if value is None:
        return ("none",)
    if isinstance(value, np.generic):
        dtype = _dtype(value.dtype)
        return ("numpy_scalar", dtype.str, value.tobytes()[:dtype.itemsize])
    if isinstance(value, np.ndarray):
        dtype = _dtype(value.dtype)
        return ("ndarray", dtype.str, tuple(value.shape), value.tobytes(order="C"))
    for tag, scalar_type in _SCALAR_TYPES.items():
        if type(value) is scalar_type:
            return (tag, value)
    if type(value) in (list, tuple):
        return ("list" if type(value) is list else "tuple", tuple(map(_freeze_value, value)))
    if type(value) is dict:
        return ("dict", tuple((_freeze_value(key), _freeze_value(item)) for key, item in value.items()))
    raise TypeError(f"Unsupported import value type: {type(value).__name__}")


def _canonical_value(value):
    """Validate already encoded values without introducing another tag layer."""
    value = tuple(value)
    if not value:
        raise ValueError("Encoded import values require a tag")
    if value == ("none",):
        return value
    tag = value[0]
    if tag in _SCALAR_TYPES and len(value) == 2:
        if type(value[1]) is not _SCALAR_TYPES[tag]:
            raise TypeError("Encoded scalar type does not match its tag")
        return value
    if tag in ("list", "tuple") and len(value) == 2:
        return (tag, tuple(map(_canonical_value, value[1])))
    if tag == "dict" and len(value) == 2:
        pairs = []
        keys = set()
        for pair in value[1]:
            key, item = pair
            key, item = _canonical_value(key), _canonical_value(item)
            decoded_key = _thaw_canonical(key)
            if decoded_key in keys:
                raise ValueError("Encoded dictionaries require unique keys")
            keys.add(decoded_key)
            pairs.append((key, item))
        return (tag, tuple(pairs))
    if tag in ("ndarray", "numpy_scalar") and len(value) == (4 if tag == "ndarray" else 3):
        dtype = _dtype(value[1])
        shape = tuple(operator.index(size) for size in value[2]) if tag == "ndarray" else ()
        data = _bytes(value[-1])
        if any(size < 0 for size in shape) or len(data) != dtype.itemsize * _size(shape):
            raise ValueError("Encoded NumPy bytes do not match shape and dtype")
        # A zero-byte shape can still exceed NumPy's dimension/stride limits.
        # Check its actual header without allocating an array-sized owner.
        np.ndarray(shape, dtype=dtype, buffer=data)
        return (tag, dtype.str, shape, data) if tag == "ndarray" else (tag, dtype.str, data)
    raise TypeError("Unsupported or malformed encoded import value")


def _size(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def _thaw_canonical(value):
    tag = value[0]
    if tag == "none":
        return None
    if tag in _SCALAR_TYPES:
        return value[1]
    if tag == "list":
        return list(map(_thaw_canonical, value[1]))
    if tag == "tuple":
        return tuple(map(_thaw_canonical, value[1]))
    if tag == "dict":
        return {_thaw_canonical(key): _thaw_canonical(item) for key, item in value[1]}
    if tag == "numpy_scalar":
        return np.ndarray((), dtype=value[1], buffer=value[2])[()]
    return np.ndarray(value[2], dtype=value[1], buffer=value[3])


def _thaw_value(value):
    """Decode into fresh containers and array headers backed by immutable bytes."""
    return _thaw_canonical(_canonical_value(value))


def _rows(values, width, count):
    result = tuple(tuple(operator.index(row) for row in item) for item in values)
    if any(len(item) != width or any(row < 0 or row >= count for row in item) for item in result):
        raise ValueError("Import row references have invalid width or range")
    return result


def _sparse(values, count, convert):
    result = tuple((operator.index(row), convert(value)) for row, value in values)
    if any(row < 0 or row >= count for row, _ in result) or len({row for row, _ in result}) != len(result):
        raise ValueError("Sparse import rows require unique rows within the Atom table")
    return result


def _pair_values(values, count):
    values = tuple(tuple(entry) for entry in values)
    if any(len(entry) != 3 for entry in values):
        raise ValueError("Supplied pair observations require row/row/value triples")
    pairs = _rows((entry[:2] for entry in values), 2, count)
    pairs = tuple(tuple(sorted(pair)) for pair in pairs)
    if any(first == second for first, second in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("Supplied pair observations require distinct unique Atom pairs")
    return tuple((first, second, float(entry[2])) for (first, second), entry in zip(pairs, values))


def _declared(value):
    present, item, source = value
    if type(present) is not bool:
        raise TypeError("Molecular declaration presence must be boolean")
    if not present and (item is not None or source is not None):
        raise ValueError("Absent molecular declarations carry no value or source")
    if present and ((type(item) not in (bool, int, float)
                     and not isinstance(item, (np.bool_, np.integer, np.floating)))
                    or (source is not None and type(source) is not str)):
        raise TypeError("Present molecular declarations require a real scalar and optional source")
    return present, item.item() if isinstance(item, np.generic) else item, source


def _stereo(payload):
    if payload is None:
        return None
    from ..stereo import (
        AtomTransportRecord, BondTransportRecord, HydrogenTransportState,
        StereoAnchor, StereoAnchorKind, StereoCarrier, StereoCarrierKind,
        StereoGroupKind, StereoGroupRef, StereoPayload, StereoRecord,
    )

    # Field order and node identities are the existing transport schema.
    # (None, T) is optional T; (tuple, T[, width]) is a tuple sequence.
    schemas = {
        StereoPayload: (int, (tuple, StereoRecord), (tuple, AtomTransportRecord),
                        (tuple, BondTransportRecord)),
        StereoRecord: (StereoAnchor, (tuple, StereoCarrier), str,
                       (None, StereoGroupRef), str, str),
        StereoAnchor: (StereoAnchorKind, (tuple, int)),
        StereoCarrier: (StereoCarrierKind, (None, int)),
        StereoGroupRef: (str, StereoGroupKind, int),
        AtomTransportRecord: (int, (None, int), (None, int),
                              (None, HydrogenTransportState),
                              (None, HydrogenTransportState), (None, str)),
        HydrogenTransportState: ((None, int), (None, int), (None, bool)),
        BondTransportRecord: ((tuple, int, 2), (None, str)),
    }

    def copy_value(value, expected):
        if type(expected) is tuple:
            marker, item_type = expected[:2]
            if marker is None:
                return None if value is None else copy_value(value, item_type)
            if type(value) not in (tuple, list):
                raise TypeError("Stereo sequence fields require a list or tuple")
            result = tuple(copy_value(item, item_type) for item in value)
            if len(expected) == 3 and len(result) != expected[2]:
                raise ValueError("Stereo tuple field has the wrong width")
            if type(value) is tuple and all(first is second for first, second in zip(value, result)):
                return value
            return result
        if type(value) is not expected:
            raise TypeError(f"Stereo schema field requires {expected.__name__}")
        if expected in schemas:
            original = tuple(getattr(value, field.name) for field in fields(value))
            sealed = tuple(copy_value(item, field_type)
                           for item, field_type in zip(original, schemas[expected]))
            return value if all(first is second for first, second in zip(original, sealed)) else expected(*sealed)
        return value

    return copy_value(payload, StereoPayload)


@dataclass(frozen=True, eq=False)
class MoleculeRecord:
    atom_count: int
    atom_values: bytes
    atom_presence: bytes
    atom_sources: bytes
    atom_isotopes: tuple
    atom_frac_valences: tuple
    bond_endpoints: tuple
    bond_values: bytes
    bond_partitions: tuple
    bond_kinds: tuple
    bond_directions: tuple
    bond_sources: tuple
    bond_metadata: tuple
    charge: tuple
    spin_mult: tuple
    cell: tuple | None
    stereo_payload: StereoPayload | None
    legacy_frame_values: tuple
    legacy_frame_index: int
    angle_constraints: tuple
    torsion_constraints: tuple
    identifier: tuple
    pair_wiberg_bond_orders: tuple = ()

    def __post_init__(self):
        count = operator.index(self.atom_count)
        if count < 0:
            raise ValueError("Atom count must be nonnegative")
        atom_values, presence, sources, bond_values = map(
            _bytes, (self.atom_values, self.atom_presence, self.atom_sources, self.bond_values),
        )
        width = len(ATOM_FIELDS)
        if (len(atom_values) != count * width * 8 or len(presence) != count * 4
                or len(sources) != count * width):
            raise ValueError(f"Atom import buffers do not match (N,{width}), (N,), (N,{width})")
        if np.any(np.frombuffer(presence, dtype="<u4") > ATOM_PRESENCE_ALL):
            raise ValueError("Atom presence contains undeclared bits")
        if not np.isin(np.frombuffer(sources, dtype="u1"), tuple(ATOM_SOURCE_CODES.values())).all():
            raise ValueError("Atom sources contain an undeclared producer code")
        endpoints = _rows(self.bond_endpoints, 2, count)
        bond_count = len(endpoints)
        partitions = tuple(operator.index(value) for value in self.bond_partitions)
        kinds, directions, bond_sources = map(tuple, (self.bond_kinds, self.bond_directions, self.bond_sources))
        metadata = tuple(map(_canonical_value, self.bond_metadata))
        if any(value[0] not in ("none", "dict") for value in metadata):
            raise TypeError("Bond metadata requires dict or None at each root")
        if (len(bond_values) != bond_count * 3 * 8 or any(
                len(column) != bond_count for column in (partitions, kinds, directions, bond_sources, metadata))):
            raise ValueError("Bond import values and metadata must align with endpoints")
        if any(value not in (0, 1, 2) for value in partitions) or any(
                type(value) is not str or value not in _BOND_KINDS for value in kinds):
            raise ValueError("Bond import contains an undeclared partition or kind")
        if any(value is not None and type(value) is not str for value in (*directions, *bond_sources)):
            raise TypeError("Bond direction and source must be strings or None")
        frames = tuple(map(_canonical_value, self.legacy_frame_values))
        if len(frames) != 9:
            raise ValueError("Legacy frame import requires all nine Conformers slots")
        cell = None if self.cell is None else tuple(map(float, self.cell))
        if cell is not None and len(cell) != 6:
            raise ValueError("Cell import requires six parameters")
        identifier = tuple(self.identifier)
        if (len(identifier) != 2 or type(identifier[0]) is not bool
                or (identifier[1] is not None and type(identifier[1]) is not str)
                or (not identifier[0] and identifier[1] is not None)):
            raise ValueError("Identifier import requires an explicit presence/value pair")
        normalized = dict(
            atom_count=count, atom_values=atom_values, atom_presence=presence, atom_sources=sources,
            atom_isotopes=_sparse(self.atom_isotopes, count, lambda value: None if value is None else operator.index(value)),
            atom_frac_valences=_sparse(self.atom_frac_valences, count, float),
            bond_endpoints=endpoints, bond_values=bond_values, bond_partitions=partitions,
            bond_kinds=kinds, bond_directions=directions, bond_sources=bond_sources,
            bond_metadata=metadata, charge=_declared(self.charge), spin_mult=_declared(self.spin_mult),
            cell=cell, stereo_payload=_stereo(self.stereo_payload), legacy_frame_values=frames,
            legacy_frame_index=operator.index(self.legacy_frame_index),
            angle_constraints=_rows(self.angle_constraints, 3, count),
            torsion_constraints=_rows(self.torsion_constraints, 4, count), identifier=identifier,
            pair_wiberg_bond_orders=_pair_values(self.pair_wiberg_bond_orders, count),
        )
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    def __reduce__(self):
        return type(self), tuple(getattr(self, field.name) for field in fields(self))

    def __deepcopy__(self, memo):
        return self


def _make_molecule_record(*, atom_values, atom_presence, atom_sources, atom_isotopes,
                          atom_frac_valences, bond_endpoints, bond_values, bond_partitions,
                          bond_kinds, bond_directions, bond_sources, bond_metadata, charge,
                          spin_mult, cell, stereo_payload, legacy_frame_values,
                          legacy_frame_index, angle_constraints, torsion_constraints, identifier,
                          pair_wiberg_bond_orders=()):
    """Encode one complete supplied import without chemistry or owner mutation."""
    values = np.asarray(atom_values, dtype="<f8")
    width = len(ATOM_FIELDS)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"Atom import values require shape (N,{width})")
    count = len(values)
    presence, sources = np.asarray(atom_presence), np.asarray(atom_sources)
    if presence.shape != (count,) or sources.shape != (count, width):
        raise ValueError(f"Atom import metadata requires shapes (N,) and (N,{width})")
    if (presence.dtype.kind not in "iu" or sources.dtype.kind not in "iu"
            or np.any(presence < 0) or np.any(presence > ATOM_PRESENCE_ALL)
            or not np.isin(sources, tuple(ATOM_SOURCE_CODES.values())).all()):
        raise ValueError("Atom import metadata contains undeclared masks or producer codes")
    endpoints = tuple(bond_endpoints)
    bonds = np.asarray(bond_values, dtype="<f8")
    if bonds.shape != (len(endpoints), 3):
        raise ValueError("Bond import values require shape (B,3)")
    return MoleculeRecord(
        count, values.tobytes(order="C"), presence.astype("<u4").tobytes(),
        sources.astype("u1").tobytes(), atom_isotopes, atom_frac_valences,
        endpoints, bonds.tobytes(order="C"), bond_partitions, bond_kinds,
        bond_directions, bond_sources, tuple(map(_freeze_value, bond_metadata)),
        charge, spin_mult, cell, stereo_payload, tuple(map(_freeze_value, legacy_frame_values)),
        legacy_frame_index, angle_constraints, torsion_constraints, identifier, pair_wiberg_bond_orders,
    )
