"""Private complete import records and a closed immutable value codec."""

from __future__ import annotations

from dataclasses import dataclass, fields
import operator
from typing import TYPE_CHECKING

import numpy as np

from ._fields import ATOM_FIELDS, ATOM_PRESENCE_ALL, ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from ._results import PartialChargeSet

if TYPE_CHECKING:
    from .stereo import StereoPayload


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
    from .stereo import (
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


def _frame_records(source_frames, conformer_results, frame_selection, count):
    """Validate the closed frame transport and preserve shared observations."""
    from .conformer import CoordinateFrame, ConformerEnsembleResult, EnergyForceResult

    observations = {}

    def subjects(uids):
        if any(uid < 0 or uid >= count for uid in uids):
            raise ValueError("Frame subjects require canonical Atom rows")
        return uids

    def observation(value):
        if value is None:
            return None
        if type(value) is not EnergyForceResult:
            raise TypeError("Frame observations require EnergyForceResult")
        key = id(value)
        if key not in observations:
            subjects(value.atom_uids)
            observations[key] = EnergyForceResult(
                value.atom_uids, value.values, value.frame_indices, value.method,
                value.method_version, value.parameters, value.units, value.provenance)
        return observations[key]

    sources = []
    for frame in source_frames:
        if type(frame) is not CoordinateFrame:
            raise TypeError("Source frame transport requires CoordinateFrame values")
        sources.append(CoordinateFrame(subjects(frame.atom_uids), frame.coordinates,
            observation(frame.observations), frame.observation_index, frame.provenance))
    generated = []
    for entry in conformer_results:
        payload, marker = entry
        if type(payload) is not ConformerEnsembleResult:
            raise TypeError("Generated frame transport requires ConformerEnsembleResult values")
        marker = tuple(marker)
        if any(domain not in ("frame_representation", "cell", "constraints") for domain in marker):
            raise ValueError("Generated frames require finite applicability markers")
        generated.append((ConformerEnsembleResult(subjects(payload.atom_uids), payload.batch,
            observation(payload.observations), payload.input_source), marker))
    cursor, selected = (0, None) if frame_selection is None else frame_selection
    cursor = operator.index(cursor)
    if selected is not None:
        (payload, row), marker = selected
        payload = observation(payload)
        row, marker = operator.index(row), tuple(marker)
        if payload is None or not 0 <= row < len(payload.frame_indices):
            raise ValueError("Selected frame observations require an existing observation row")
        if any(domain not in ("frame_representation", "geometry", "cell", "constraints") for domain in marker):
            raise ValueError("Selected frame observations require finite applicability markers")
        selected = ((payload, row), marker)
    return tuple(sources), tuple(generated), (cursor, selected)


@dataclass(frozen=True, eq=False, kw_only=True)
class MoleculeRecord:
    atom_count: int
    atom_values: bytes
    atom_presence: bytes
    atom_sources: bytes
    atom_isotopes: tuple
    atom_frac_valences: tuple
    bond_endpoints: tuple
    bond_values: bytes
    bond_kinds: tuple
    bond_directions: tuple
    bond_sources: tuple
    bond_metadata: tuple
    charge: tuple
    spin_mult: tuple
    cell: tuple | None
    stereo_payload: StereoPayload | None
    source_frames: tuple
    conformer_results: tuple
    frame_selection: tuple | None
    angle_constraints: tuple
    torsion_constraints: tuple
    identifier: tuple
    pair_wiberg_bond_orders: tuple = ()
    bond_aromatic: tuple[bool, ...] = ()
    partial_charge_result: tuple[PartialChargeSet, tuple[str, ...]] | None = None

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
        charge_field = FIELD_INDEX["partial_charge"]
        charge_bit = FIELD_PRESENCE_BITS[charge_field]
        masks = np.frombuffer(presence, dtype="<u4").copy()
        charge_present = (masks & charge_bit) != 0
        result = self.partial_charge_result
        if result is not None:
            payload, marker = result
            if charge_present.any():
                raise ValueError("An explicit partial-charge result excludes supplied Atom charge bits")
            if (type(payload) is not PartialChargeSet or payload.atom_uids != tuple(range(count))
                    or any(domain not in ("charge_representation", "geometry") for domain in marker)):
                raise ValueError("Record partial charges require canonical complete rows and finite dependency markers")
            result = payload, tuple(marker)
        codes = np.frombuffer(sources, dtype="u1").reshape(count, width).copy()
        values = np.frombuffer(atom_values, dtype="<f8").reshape(count, width).copy()
        if result is None and charge_present.any():
            result = (PartialChargeSet(
                tuple(range(count)),
                tuple(float(value) if present else None
                      for value, present in zip(values[:, charge_field], charge_present)),
                "supplied", None, (), "e", tuple(map(int, codes[:, charge_field])),
            ), ())
        masks &= np.uint32(~charge_bit & ATOM_PRESENCE_ALL)
        codes[:, charge_field] = ATOM_SOURCE_CODES["default"]
        values[:, charge_field] = np.nan
        atom_values, presence, sources = values.tobytes(), masks.tobytes(), codes.tobytes()
        endpoints = _rows(self.bond_endpoints, 2, count)
        bond_count = len(endpoints)
        kinds, directions, bond_sources = map(tuple, (self.bond_kinds, self.bond_directions, self.bond_sources))
        aromatic = tuple(self.bond_aromatic) if self.bond_aromatic else tuple(kind == 'aromatic' for kind in kinds)
        metadata = tuple(map(_canonical_value, self.bond_metadata))
        if any(value[0] not in ("none", "dict") for value in metadata):
            raise TypeError("Bond metadata requires dict or None at each root")
        if (len(bond_values) != bond_count * 3 * 8 or any(
                len(column) != bond_count for column in (kinds, directions, bond_sources, metadata, aromatic))):
            raise ValueError("Bond import values and metadata must align with endpoints")
        if any(type(value) is not str or value not in _BOND_KINDS for value in kinds):
            raise ValueError("Bond import contains an undeclared kind")
        if any(value is not None and type(value) is not str for value in (*directions, *bond_sources)):
            raise TypeError("Bond direction and source must be strings or None")
        source_frames, conformer_results, frame_selection = _frame_records(
            self.source_frames, self.conformer_results, self.frame_selection, count)
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
            bond_endpoints=endpoints, bond_values=bond_values,
            bond_kinds=kinds, bond_directions=directions, bond_sources=bond_sources,
            bond_metadata=metadata, charge=_declared(self.charge), spin_mult=_declared(self.spin_mult),
            cell=cell, stereo_payload=_stereo(self.stereo_payload), source_frames=source_frames,
            conformer_results=conformer_results, frame_selection=frame_selection,
            angle_constraints=_rows(self.angle_constraints, 3, count),
            torsion_constraints=_rows(self.torsion_constraints, 4, count), identifier=identifier,
            pair_wiberg_bond_orders=_pair_values(self.pair_wiberg_bond_orders, count),
            bond_aromatic=tuple(bool(value) for value in aromatic), partial_charge_result=result,
        )
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    def __reduce__(self):
        return _restore_molecule_record, (
            type(self), 4, tuple(getattr(self, field.name) for field in fields(self)),
        )

    def __deepcopy__(self, memo):
        return self


def _restore_molecule_record(record_type, schema_version, field_values):
    if schema_version != 4 or len(field_values) != 25:
        raise ValueError("MoleculeRecord pickle requires schema version 4 and 25 fields")
    return record_type(**dict(zip((field.name for field in fields(MoleculeRecord)), field_values)))


def _make_molecule_record(*, atom_values, atom_presence, atom_sources, atom_isotopes,
                          atom_frac_valences, bond_endpoints, bond_values,
                          bond_kinds, bond_directions, bond_sources, bond_metadata, charge,
                          spin_mult, cell, stereo_payload, source_frames,
                          conformer_results, frame_selection, angle_constraints, torsion_constraints, identifier,
                          pair_wiberg_bond_orders=(), bond_aromatic=(), partial_charge_result=None):
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
        atom_count=count, atom_values=values.tobytes(order="C"),
        atom_presence=presence.astype("<u4").tobytes(), atom_sources=sources.astype("u1").tobytes(),
        atom_isotopes=atom_isotopes, atom_frac_valences=atom_frac_valences,
        bond_endpoints=endpoints, bond_values=bonds.tobytes(order="C"), bond_kinds=bond_kinds,
        bond_directions=bond_directions, bond_sources=bond_sources,
        bond_metadata=tuple(map(_freeze_value, bond_metadata)), charge=charge, spin_mult=spin_mult,
        cell=cell, stereo_payload=stereo_payload,
        source_frames=source_frames, conformer_results=conformer_results,
        frame_selection=frame_selection, angle_constraints=angle_constraints,
        torsion_constraints=torsion_constraints, identifier=identifier,
        pair_wiberg_bond_orders=pair_wiberg_bond_orders, bond_aromatic=bond_aromatic,
        partial_charge_result=partial_charge_result,
    )
