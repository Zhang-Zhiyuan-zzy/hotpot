"""One private numeric Atom store, without chemistry or molecule membership.

Two float64 blocks own only factual numeric slots. The historical19-column
transport shape is projected with unavailable P4 slots; operations calculate
electron and valence views. No derived numeric value is retained here.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping

import numpy as np

from ._fields import (
    ATOM_FIELDS, ATOM_PRESENCE_ALL, ATOM_SOURCE_CODES, FIELD_ADDRESSES,
    FIELD_CODECS, FIELD_INDEX, FIELD_PRESENCE_BITS, SCALAR_FIELDS, DERIVED_FIELDS,
)


__all__ = ()

_SOURCE_CODES = frozenset(ATOM_SOURCE_CODES.values())


def _numeric_rows(rows):
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(ATOM_FIELDS):
        raise ValueError(f"Atom rows must have shape (N, {len(ATOM_FIELDS)})")
    return values


def _storage(capacity):
    return (
        np.empty((capacity, len(SCALAR_FIELDS)), dtype=np.float64),
        np.empty((capacity, 3), dtype=np.float64),
    )


def _column_views(blocks):
    return tuple(None if address is None else blocks[address[0]][:, address[1]]
                 for address in FIELD_ADDRESSES)


def _selected_blocks(blocks, selection):
    if isinstance(selection, slice):
        return blocks[0][selection].copy(), blocks[1][selection].copy()
    # Advanced row indexing already allocates independent arrays.
    return blocks[0][selection], blocks[1][selection]


def _metadata_array(value, shape, dtype, maximum, name):
    candidate = np.asarray(value)
    if candidate.shape != () and candidate.shape != shape:
        raise ValueError(f"Atom {name} requires a scalar or shape {shape}")
    if candidate.size and candidate.dtype.kind not in "iu":
        raise TypeError(f"Atom {name} requires integer values")
    if candidate.size and (candidate.min() < 0 or candidate.max() > maximum):
        raise ValueError(f"Atom {name} is outside its declared range")
    return np.full(shape, candidate, dtype=dtype) if candidate.shape == () else candidate.astype(dtype, copy=True)


def _metadata_rows(count, presence, sources):
    masks = _metadata_array(
        ATOM_PRESENCE_ALL if presence is None else presence,
        (count,), np.uint32, ATOM_PRESENCE_ALL, "presence",
    )
    if sources is None:
        supplied = (masks[:, None] & np.asarray(FIELD_PRESENCE_BITS, dtype=np.uint32)) != 0
        codes = np.where(supplied, ATOM_SOURCE_CODES["legacy_array"], ATOM_SOURCE_CODES["default"]).astype(np.uint8)
    else:
        codes = _metadata_array(sources, (count, len(ATOM_FIELDS)), np.uint8, max(_SOURCE_CODES), "sources")
        if not np.isin(codes, tuple(_SOURCE_CODES)).all():
            raise ValueError("Atom sources contain an undeclared source code")
    return masks, codes


def _present_value(value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("Atom update presence must be a boolean")
    return bool(value)


def _source_value(value):
    value = operator.index(value)
    if value not in _SOURCE_CODES:
        raise ValueError("Atom update source is not a declared source code")
    return value


class _AtomFactTable:
    __slots__ = ("_blocks", "_columns", "_size", "_presence", "_sources", "_group_status", "_dirty_seeds")

    def __init__(self, rows=None, *, presence=None, sources=None):
        values = np.empty((0, len(ATOM_FIELDS))) if rows is None else _numeric_rows(rows)
        masks, codes = _metadata_rows(len(values), presence, sources)
        blocks = (values[:, [FIELD_INDEX[name] for name in SCALAR_FIELDS]].copy(),
                  values[:, 10:13].copy())
        for field in DERIVED_FIELDS:
            index = FIELD_INDEX[field]
            masks &= np.uint32(~FIELD_PRESENCE_BITS[index] & ATOM_PRESENCE_ALL)
            codes[:, index] = ATOM_SOURCE_CODES["default"]
        self._blocks, self._columns, self._size = blocks, _column_views(blocks), len(values)
        self._presence, self._sources = masks, codes
        from ._dependencies import _initialize_dependencies
        _initialize_dependencies(self)

    def __len__(self):
        return self._size

    def __getstate__(self):
        # NumPy serializes views independently, so persist only the owners.
        # Spare capacity may be uninitialized and is recreated, never read.
        return (
            self._size, len(self._blocks[0]),
            tuple(block[:self._size].copy() for block in self._blocks),
            self._presence[:self._size].copy(), self._sources[:self._size].copy(),
            self._group_status[:self._size].copy(), tuple(frozenset(seed) for seed in self._dirty_seeds),
        )

    def __setstate__(self, state):
        size, capacity, values, presence, sources, status, seeds = state
        blocks = _storage(capacity)
        for block, value in zip(blocks, values):
            block[:size] = value
        masks = np.empty(capacity, dtype=np.uint32)
        codes = np.empty((capacity, len(ATOM_FIELDS)), dtype=np.uint8)
        masks[:size], codes[:size] = presence, sources
        self._blocks, self._columns, self._size = blocks, _column_views(blocks), size
        self._presence, self._sources = masks, codes
        self._group_status = np.empty((capacity, status.shape[1]), dtype=np.uint8)
        self._group_status[:size] = status
        self._dirty_seeds = tuple(set(seed) for seed in seeds)

    def _install_storage(self, blocks, columns, size, presence, sources):
        previous = self._blocks, self._columns, self._size, self._presence, self._sources
        try:
            self._blocks, self._columns, self._size = blocks, columns, size
            self._presence, self._sources = presence, sources
        except BaseException:
            self._blocks, self._columns, self._size, self._presence, self._sources = previous
            raise

    def _row(self, row):
        row = operator.index(row)
        if row < 0:
            row += self._size
        if not 0 <= row < self._size:
            raise IndexError("Atom row is outside the current table")
        return row

    def _rows(self, rows, *, unique=False):
        if rows is None:
            return slice(0, self._size), self._size
        if isinstance(rows, slice):
            start, stop, step = rows.indices(self._size)
            indices = np.arange(start, stop, step, dtype=np.intp)
        elif isinstance(rows, (int, np.integer)):
            row = self._row(rows)
            return slice(row, row + 1), 1
        else:
            indices = np.asarray(rows)
            if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
                raise TypeError("Atom row selection must be a one-dimensional integer sequence")
            # Check before intp conversion, so oversized unsigned indices cannot wrap.
            if indices.size and (indices.min() < -self._size or indices.max() >= self._size):
                raise IndexError("Atom row is outside the current table")
            indices = np.array(indices, dtype=np.intp, copy=True)
            indices[indices < 0] += self._size
        if unique and len(np.unique(indices)) != len(indices):
            raise ValueError("Atom update/remap rows cannot contain duplicates")
        return indices, len(indices)

    def raw_index(self, row, field_index):
        """Read a raw numeric slot by pre-resolved field index."""
        row = operator.index(row)
        if row < 0:
            row += self._size
        if not 0 <= row < self._size:
            raise IndexError("Atom row is outside the current table")
        return float(self._columns[field_index][row])

    def raw(self, row, field):
        row = operator.index(row)
        if row < 0:
            row += self._size
        if not 0 <= row < self._size:
            raise IndexError("Atom row is outside the current table")
        return float(self._columns[FIELD_INDEX[field]][row])

    def read(self, row, field):
        row = operator.index(row)
        if row < 0:
            row += self._size
        if not 0 <= row < self._size:
            raise IndexError("Atom row is outside the current table")
        return FIELD_CODECS[field](self._columns[FIELD_INDEX[field]][row])

    def row(self, row):
        row = self._row(row)
        return self.select(ATOM_FIELDS, rows=(row,))[0]

    def column(self, field, rows=None):
        selection, _ = self._rows(rows)
        return self._columns[FIELD_INDEX[field]][selection].copy()

    def coordinates(self, rows=None):
        selection, _ = self._rows(rows)
        return self._blocks[1][selection].copy()

    def matrix(self):
        return self.select(ATOM_FIELDS)

    def presence(self, rows=None):
        selection, _ = self._rows(rows)
        return self._presence[selection].copy()

    def sources(self, rows=None):
        selection, _ = self._rows(rows)
        return self._sources[selection].copy()

    def select(self, fields, *, rows=None, dtype="float64"):
        indices = tuple(FIELD_INDEX[field] for field in fields)
        selection, count = self._rows(rows)
        values = np.empty((count, len(indices)), dtype=dtype)
        for target, index in enumerate(indices):
            column = self._columns[index]
            values[:, target] = np.nan if column is None else column[selection]
        return values

    def update(self, rows, columns, *, present=True, source=ATOM_SOURCE_CODES["explicit"]):
        """Install numeric values, presence and provenance as one touched set.

        Presence/source options may be scalar or have exactly the canonical
        keys in ``columns``. Scalar writes remain O(touched fields); neither
        path copies untargeted rows or fields for rollback.
        """
        mapped_presence = isinstance(present, Mapping)
        mapped_source = isinstance(source, Mapping)
        if mapped_presence and present.keys() != columns.keys():
            raise ValueError("Atom presence mapping must match the updated fields")
        if mapped_source and source.keys() != columns.keys():
            raise ValueError("Atom source mapping must match the updated fields")
        if not columns:
            return
        selection, count = self._rows(rows, unique=True)
        if count == 1:
            selection = selection.start if isinstance(selection, slice) else int(selection[0])
        prepared = []
        for field, value in columns.items():
            if field in DERIVED_FIELDS:
                raise AttributeError(f"{field} is a calculated read-only projection")
            index = FIELD_INDEX[field]
            prepared.append((index, value,
                             _present_value(present[field] if mapped_presence else present),
                             _source_value(source[field] if mapped_source else source)))
        self._write_prepared(selection, prepared)

    def _write_prepared(self, selection, prepared, *, transaction=None):
        """Install through the sole native physical writer and undo boundary."""
        from ._commit import _fact_engine
        _fact_engine()._write_prepared(self, selection, prepared, transaction)

    def append(self, rows, *, presence=None, sources=None):
        values = _numeric_rows(rows)
        masks, codes = _metadata_rows(len(values), presence, sources)
        return self._append_prepared(values, masks, codes)

    def _append_prepared(self, values, masks, codes):
        """Append operation-owned rows with already normalized metadata."""
        first = self._size
        if not len(values):
            return first
        size = first + len(values)
        capacity = len(self._blocks[0])
        if size > capacity:
            capacity = max(size, capacity + max(1, capacity // 2))
            blocks = _storage(capacity)
            for target, source in zip(blocks, self._blocks):
                target[:first] = source[:first]
            next_presence = np.empty(capacity, dtype=np.uint32)
            next_sources = np.empty((capacity, len(ATOM_FIELDS)), dtype=np.uint8)
            next_presence[:first] = self._presence[:first]
            next_sources[:first] = self._sources[:first]
        else:
            blocks = self._blocks
            next_presence, next_sources = self._presence, self._sources
        blocks[0][first:size] = values[:, [FIELD_INDEX[name] for name in SCALAR_FIELDS]]
        blocks[1][first:size] = values[:, 10:13]
        next_presence[first:size] = masks
        next_sources[first:size] = codes
        for field in DERIVED_FIELDS:
            index = FIELD_INDEX[field]
            next_presence[first:size] &= np.uint32(~FIELD_PRESENCE_BITS[index] & ATOM_PRESENCE_ALL)
            next_sources[first:size, index] = ATOM_SOURCE_CODES["default"]
        columns = self._columns if blocks is self._blocks else _column_views(blocks)
        from ._dependencies import _status_from_presence
        if capacity > len(self._group_status):
            status = np.empty((capacity, self._group_status.shape[1]), dtype=np.uint8)
            status[:first] = self._group_status[:first]
        else:
            status = self._group_status
        status[first:size] = _status_from_presence(next_presence[first:size])
        self._install_storage(blocks, columns, size, next_presence, next_sources)
        self._group_status = status
        return first

    def remap(self, rows):
        selection, size = self._rows(rows, unique=True)
        # Preserve spare capacity across deletion/reordering so a subsequent
        # append can reuse it. Logical rows still install as one prepared set.
        blocks = _storage(len(self._blocks[0]))
        blocks[0][:size] = self._blocks[0][selection]
        blocks[1][:size] = self._blocks[1][selection]
        next_presence = np.empty_like(self._presence)
        next_sources = np.empty_like(self._sources)
        next_presence[:size] = self._presence[selection]
        next_sources[:size] = self._sources[selection]
        status = np.empty((len(blocks[0]), self._group_status.shape[1]), dtype=np.uint8)
        status[:size] = self._group_status[selection]
        self._install_storage(blocks, _column_views(blocks), size, next_presence, next_sources)
        self._group_status = status

    def take(self, rows):
        selection, size = self._rows(rows)
        blocks = _selected_blocks(self._blocks, selection)
        presence = self._presence[selection].copy()
        sources = self._sources[selection].copy()
        result = object.__new__(type(self))
        result._blocks, result._columns, result._size = blocks, _column_views(blocks), size
        result._presence, result._sources = presence, sources
        result._group_status = self._group_status[selection].copy()
        result._dirty_seeds = tuple(set(seed) for seed in self._dirty_seeds)
        return result

    def copy(self):
        return self.take(None)
