"""One private numeric Atom store, without chemistry or molecule membership.

Two float64 blocks preserve every historical numeric slot. The 19 column
views only address those blocks; they own no data. Callers serialize mutations
and perform owner/source checks at the Molecule commit boundary.
"""

from __future__ import annotations

import operator

import numpy as np

from ._fields import ATOM_FIELDS, FIELD_ADDRESSES, FIELD_CODECS, FIELD_INDEX, SCALAR_FIELDS


__all__ = ()


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


class _AtomFactTable:
    __slots__ = ("_blocks", "_columns", "_size")

    def __init__(self, rows=None):
        values = np.empty((0, len(ATOM_FIELDS))) if rows is None else _numeric_rows(rows)
        blocks = (
            np.concatenate((values[:, :10], values[:, 13:]), axis=1),
            values[:, 10:13].copy(),
        )
        self._blocks, self._columns, self._size = blocks, _column_views(blocks), len(values)

    def __len__(self):
        return self._size

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

    def select(self, fields, *, rows=None, dtype="float64"):
        indices = tuple(FIELD_INDEX[field] for field in fields)
        selection, count = self._rows(rows)
        values = np.empty((count, len(indices)), dtype=dtype)
        for target, index in enumerate(indices):
            values[:, target] = self._columns[index][selection]
        return values

    def update(self, rows, columns):
        """Convert the complete touched write set before installing any value.

        Scalar writes stay O(number of touched fields), without row arrays or
        whole-table copies. Both paths restore the interrupted write as well
        as earlier writes; all user conversion occurs before installation.
        """
        if isinstance(rows, (int, np.integer)):
            row = self._row(rows)
            if len(columns) == 1:
                field = next(iter(columns))
                column = self._columns[FIELD_INDEX[field]]
                value = _scalar_value(columns[field])
                if not column.flags.writeable:
                    raise ValueError("Atom update targets a read-only column")
                previous = column[row]
                try:
                    column[row] = value
                except BaseException:
                    column[row] = previous
                    raise
                return

            prepared = tuple(
                (self._columns[FIELD_INDEX[field]], _scalar_value(value))
                for field, value in columns.items()
            )
            if any(not column.flags.writeable for column, _ in prepared):
                raise ValueError("Atom update targets a read-only column")
            previous = tuple(column[row] for column, _ in prepared)
            try:
                for column, value in prepared:
                    column[row] = value
            except BaseException:
                for (column, _), value in zip(prepared, previous):
                    column[row] = value
                raise
            return

        selection, count = self._rows(rows, unique=True)
        prepared = []
        for field, value in columns.items():
            column = self._columns[FIELD_INDEX[field]]
            candidate = np.asarray(value, dtype=np.float64)
            if candidate.shape == ():
                candidate = np.full(count, candidate, dtype=np.float64)
            elif candidate.shape == (count,):
                candidate = candidate.copy()
            else:
                raise ValueError(f"Atom field {field!r} requires a scalar or shape ({count},)")
            prepared.append((column, candidate))
        if any(not column.flags.writeable for column, _ in prepared):
            raise ValueError("Atom update targets a read-only column")
        previous = tuple(column[selection].copy() for column, _ in prepared)
        try:
            for column, candidate in prepared:
                column[selection] = candidate
        except BaseException:
            for (column, _), value in zip(prepared, previous):
                column[selection] = value
            raise

    def append(self, rows):
        values = _numeric_rows(rows)
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
        else:
            blocks = self._blocks
        blocks[0][first:size, :10] = values[:, :10]
        blocks[0][first:size, 10:] = values[:, 13:]
        blocks[1][first:size] = values[:, 10:13]
        columns = self._columns if blocks is self._blocks else _column_views(blocks)
        self._blocks, self._columns, self._size = blocks, columns, size
        return first

    def remap(self, rows):
        selection, size = self._rows(rows, unique=True)
        # Preserve spare capacity across deletion/reordering so a subsequent
        # append can reuse it. Logical rows still install as one prepared set.
        blocks = _storage(len(self._blocks[0]))
        blocks[0][:size] = self._blocks[0][selection]
        blocks[1][:size] = self._blocks[1][selection]
        self._blocks, self._columns, self._size = blocks, _column_views(blocks), size

    def take(self, rows):
        selection, size = self._rows(rows)
        blocks = _selected_blocks(self._blocks, selection)
        result = object.__new__(type(self))
        result._blocks, result._columns, result._size = blocks, _column_views(blocks), size
        return result

    def copy(self):
        return self.take(None)
