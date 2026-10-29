"""One private numeric Atom store, without chemistry or molecule membership.

Two float64 blocks preserve every historical numeric slot. The 19 column
views only address those blocks; they own no data. Callers serialize mutations
and perform owner/source checks at the Molecule commit boundary.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping

import numpy as np

from ._fields import (
    ATOM_FIELDS, ATOM_PRESENCE_ALL, ATOM_SOURCE_CODES, FIELD_ADDRESSES,
    FIELD_CODECS, FIELD_INDEX, FIELD_PRESENCE_BITS, SCALAR_FIELDS,
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
    return tuple(blocks[block][:, offset] for block, offset in FIELD_ADDRESSES)


def _selected_blocks(blocks, selection):
    if isinstance(selection, slice):
        return blocks[0][selection].copy(), blocks[1][selection].copy()
    # Advanced row indexing already allocates independent arrays.
    return blocks[0][selection], blocks[1][selection]


def _scalar_value(value):
    if isinstance(value, (int, float, np.number)):
        return float(value)
    candidate = np.asarray(value, dtype=np.float64)
    if candidate.shape == ():
        return float(candidate)
    if candidate.shape == (1,):
        return float(candidate[0])
    raise ValueError("A single-row Atom update requires a scalar or one value")


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
    __slots__ = ("_blocks", "_columns", "_size", "_presence", "_sources")

    def __init__(self, rows=None, *, presence=None, sources=None):
        values = np.empty((0, len(ATOM_FIELDS))) if rows is None else _numeric_rows(rows)
        masks, codes = _metadata_rows(len(values), presence, sources)
        blocks = (
            np.concatenate((values[:, :10], values[:, 13:]), axis=1),
            values[:, 10:13].copy(),
        )
        self._blocks, self._columns, self._size = blocks, _column_views(blocks), len(values)
        self._presence, self._sources = masks, codes

    def __len__(self):
        return self._size

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
        values = np.empty(len(ATOM_FIELDS), dtype=np.float64)
        values[:10] = self._blocks[0][row, :10]
        values[10:13] = self._blocks[1][row]
        values[13:] = self._blocks[0][row, 10:]
        return values

    def column(self, field, rows=None):
        selection, _ = self._rows(rows)
        return self._columns[FIELD_INDEX[field]][selection].copy()

    def coordinates(self, rows=None):
        selection, _ = self._rows(rows)
        return self._blocks[1][selection].copy()

    def matrix(self):
        return np.concatenate((
            self._blocks[0][:self._size, :10],
            self._blocks[1][:self._size],
            self._blocks[0][:self._size, 10:],
        ), axis=1)

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
            values[:, target] = self._columns[index][selection]
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
        if isinstance(rows, (int, np.integer)):
            row = self._row(rows)
            if len(columns) == 1:
                field = next(iter(columns))
                index = FIELD_INDEX[field]
                column = self._columns[index]
                value = _scalar_value(columns[field])
                supplied = _present_value(present[field] if mapped_presence else present)
                code = _source_value(source[field] if mapped_source else source)
                if not (column.flags.writeable and self._presence.flags.writeable and self._sources.flags.writeable):
                    raise ValueError("Atom update targets read-only storage")
                previous = column[row]
                old_presence = self._presence[row]
                old_source = self._sources[row, index]
                mask = FIELD_PRESENCE_BITS[index]
                next_presence = int(old_presence) | mask if supplied else int(old_presence) & (ATOM_PRESENCE_ALL ^ mask)
                try:
                    column[row] = value
                    self._sources[row, index] = code
                    self._presence[row] = next_presence
                except BaseException:
                    column[row] = previous
                    self._sources[row, index] = old_source
                    self._presence[row] = old_presence
                    raise
                return

            prepared = tuple(
                (FIELD_INDEX[field], self._columns[FIELD_INDEX[field]], _scalar_value(value),
                 _present_value(present[field] if mapped_presence else present),
                 _source_value(source[field] if mapped_source else source))
                for field, value in columns.items()
            )
            if (not self._presence.flags.writeable or not self._sources.flags.writeable
                    or any(not column.flags.writeable for _, column, _, _, _ in prepared)):
                raise ValueError("Atom update targets read-only storage")
            previous = tuple(column[row] for _, column, _, _, _ in prepared)
            old_sources = tuple(self._sources[row, index] for index, _, _, _, _ in prepared)
            old_presence = self._presence[row]
            next_presence = int(old_presence)
            for index, _, _, supplied, _ in prepared:
                mask = FIELD_PRESENCE_BITS[index]
                next_presence = next_presence | mask if supplied else next_presence & (ATOM_PRESENCE_ALL ^ mask)
            try:
                for index, column, value, _, code in prepared:
                    column[row] = value
                    self._sources[row, index] = code
                self._presence[row] = next_presence
            except BaseException:
                for (index, column, _, _, _), value, code in zip(prepared, previous, old_sources):
                    column[row] = value
                    self._sources[row, index] = code
                self._presence[row] = old_presence
                raise
            return

        selection, count = self._rows(rows, unique=True)
        prepared = []
        for field, value in columns.items():
            index = FIELD_INDEX[field]
            column = self._columns[index]
            candidate = np.asarray(value, dtype=np.float64)
            if candidate.shape == ():
                candidate = np.full(count, candidate, dtype=np.float64)
            elif candidate.shape == (count,):
                candidate = candidate.copy()
            else:
                raise ValueError(f"Atom field {field!r} requires a scalar or shape ({count},)")
            supplied = _present_value(present[field] if mapped_presence else present)
            code = _source_value(source[field] if mapped_source else source)
            prepared.append((index, column, candidate, supplied, code))
        if (not self._presence.flags.writeable or not self._sources.flags.writeable
                or any(not column.flags.writeable for _, column, _, _, _ in prepared)):
            raise ValueError("Atom update targets read-only storage")
        previous = tuple(column[selection].copy() for _, column, _, _, _ in prepared)
        old_sources = tuple(self._sources[selection, index].copy() for index, _, _, _, _ in prepared)
        old_presence = self._presence[selection].copy()
        next_presence = old_presence.copy()
        for index, _, _, supplied, _ in prepared:
            mask = FIELD_PRESENCE_BITS[index]
            if supplied:
                next_presence |= np.uint32(mask)
            else:
                next_presence &= np.uint32(ATOM_PRESENCE_ALL ^ mask)
        try:
            for index, column, candidate, _, code in prepared:
                column[selection] = candidate
                self._sources[selection, index] = code
            self._presence[selection] = next_presence
        except BaseException:
            for (index, column, _, _, _), value, code in zip(prepared, previous, old_sources):
                column[selection] = value
                self._sources[selection, index] = code
            self._presence[selection] = old_presence
            raise

    def append(self, rows, *, presence=None, sources=None):
        values = _numeric_rows(rows)
        masks, codes = _metadata_rows(len(values), presence, sources)
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
        blocks[0][first:size, :10] = values[:, :10]
        blocks[0][first:size, 10:] = values[:, 13:]
        blocks[1][first:size] = values[:, 10:13]
        next_presence[first:size] = masks
        next_sources[first:size] = codes
        columns = self._columns if blocks is self._blocks else _column_views(blocks)
        self._install_storage(blocks, columns, size, next_presence, next_sources)
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
        self._install_storage(blocks, _column_views(blocks), size, next_presence, next_sources)

    def take(self, rows):
        selection, size = self._rows(rows)
        blocks = _selected_blocks(self._blocks, selection)
        presence = self._presence[selection].copy()
        sources = self._sources[selection].copy()
        result = object.__new__(type(self))
        result._blocks, result._columns, result._size = blocks, _column_views(blocks), size
        result._presence, result._sources = presence, sources
        return result

    def copy(self):
        return self.take(None)
