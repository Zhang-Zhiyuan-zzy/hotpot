"""Closed, immutable Atom update candidates; no carrier installation here.

Selection and owner checks happen at the Molecule boundary. This preparation
retains the legacy float64 domain; it does not infer chemistry or alter the
permissions of the legacy setters before their owning-stage migration.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np

from ..reference.properties import _atomic_number_for_symbol
from ._fields import (
    ATOM_BATCH_READONLY, ATOM_BATCH_WRITABLE, ATOM_FIELDS, ATOM_SOURCE_CODES,
    COORDINATE_FIELDS, FIELD_INDEX,
)
from ._identity import _SourceIdentity


__all__ = ()


def _column_candidate(value, count, field):
    candidate = np.asarray(value, dtype=np.float64)
    if candidate.shape == ():
        return np.full(count, candidate, dtype=np.float64)
    if candidate.shape != (count,):
        raise ValueError(f"Atom field {field!r} requires a scalar or shape ({count},)")
    return candidate


@dataclass(frozen=True, eq=False)
class _AtomFactUpdate:
    source: _SourceIdentity
    rows: tuple[int, ...]
    atom_uids: tuple[int, ...]
    fields: tuple[str, ...]
    value_bytes: tuple[bytes, ...]
    presence: tuple[bool, ...]
    sources: tuple[int, ...]

    def __post_init__(self):
        rows = tuple(operator.index(row) for row in self.rows)
        uids = tuple(operator.index(uid) for uid in self.atom_uids)
        if (len(rows) != len(uids) or len(set(rows)) != len(rows)
                or len(set(uids)) != len(uids) or any(row < 0 for row in rows)
                or any(uid < 0 for uid in uids)):
            raise ValueError("Atom updates require aligned unique nonnegative rows and UIDs")
        fields = tuple(self.fields)
        if len(set(fields)) != len(fields) or any(field not in FIELD_INDEX for field in fields):
            raise ValueError("Atom updates require unique canonical fields")
        if not len(fields) == len(self.value_bytes) == len(self.presence) == len(self.sources):
            raise ValueError("Atom update values and metadata must align with fields")
        presence = tuple(self.presence)
        if any(not isinstance(value, (bool, np.bool_)) for value in presence):
            raise TypeError("Atom update presence must be boolean")
        sources = tuple(operator.index(value) for value in self.sources)
        if any(value not in ATOM_SOURCE_CODES.values() for value in sources):
            raise ValueError("Atom update source must be a declared producer code")
        value_bytes = tuple(bytes(value) for value in self.value_bytes)
        if any(len(value) != len(rows) * np.dtype(np.float64).itemsize for value in value_bytes):
            raise ValueError("Atom update bytes must encode one float64 per selected row")
        for name, value in (
            ("rows", rows), ("atom_uids", uids), ("fields", fields),
            ("value_bytes", value_bytes), ("presence", tuple(bool(value) for value in presence)),
            ("sources", sources),
        ):
            object.__setattr__(self, name, value)

    @property
    def values(self):
        # Headers are task-local: changing dtype/shape cannot alter the candidate.
        return tuple(np.frombuffer(value, dtype=np.float64) for value in self.value_bytes)

    def __reduce__(self):
        return type(self), (
            self.source, self.rows, self.atom_uids, self.fields, self.value_bytes,
            self.presence, self.sources,
        )

    def __deepcopy__(self, memo):
        # The complete candidate is immutable; retain its opaque source owner.
        return self


def _normalize_atom_facts(source, rows, atom_uids, columns):
    """Encode the approved scalar/batch fields into one complete candidate."""
    for field in columns:
        if field in ATOM_BATCH_READONLY:
            raise AttributeError(f"Atom field {field!r} is not writable through updata_atoms")
        if field not in ATOM_BATCH_WRITABLE:
            raise TypeError(f"Unknown Atom update field: {field!r}")
    rows = tuple(rows)
    count = len(rows)
    prepared = {}

    def add(field, value):
        value = _column_candidate(value, count, field)
        if field in prepared and not np.array_equal(prepared[field], value, equal_nan=True):
            raise ValueError(f"Conflicting Atom update aliases for {field!r}")
        prepared[field] = value

    for field, value in columns.items():
        if field == "symbol":
            symbols = np.asarray(value)
            if symbols.shape == ():
                add("atomic_number", _atomic_number_for_symbol(symbols.item()))
            elif symbols.shape == (count,):
                add("atomic_number", [_atomic_number_for_symbol(item) for item in symbols])
            else:
                raise ValueError(f"Atom symbols require a scalar or shape ({count},)")
        elif field == "coordinates":
            coordinates = np.asarray(value, dtype=np.float64)
            if coordinates.shape == (3,):
                for name, item in zip(COORDINATE_FIELDS, coordinates):
                    add(name, item)
            elif coordinates.shape == (count, 3):
                for index, name in enumerate(COORDINATE_FIELDS):
                    add(name, coordinates[:, index])
            else:
                raise ValueError(f"Atom coordinates require shape (3,) or ({count}, 3)")
        else:
            add(field, value)
    fields = tuple(field for field in ATOM_FIELDS if field in prepared)
    return _AtomFactUpdate(
        source, rows, tuple(atom_uids), fields,
        tuple(prepared[field].tobytes() for field in fields), (True,) * len(fields),
        (ATOM_SOURCE_CODES["explicit"],) * len(fields),
    )
