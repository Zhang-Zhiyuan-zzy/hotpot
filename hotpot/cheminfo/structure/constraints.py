"""Private persisted constraints and bit-preserving transport projections."""

from dataclasses import dataclass, field
from struct import unpack_from

import numpy as np

from ._fields import (
    ATOM_FIELDS, ATOM_PRESENCE_ALL, ATOM_SOURCE_CODES, CONSTRAINT_FIELDS,
    FIELD_INDEX, FIELD_PRESENCE_BITS, TABLE_FIELDS, TABLE_PRESENCE_ALL,
)


__all__ = ()

_EMPTY_ATOM_CELL = bytes(28)


def _atom_axis_value(cell, axis):
    return unpack_from("<d", cell, axis * 8)[0]


def _bond_constraint_value(cell):
    return unpack_from("<d", cell)[0]


def _bond_constraint_cell(value):
    return np.asarray(value, dtype="<f8").tobytes()


def _bond_raw_values(values, cell):
    result = np.empty(3, dtype=np.float64)
    result[0], result[2] = values
    result[1] = _bond_constraint_value(cell)
    return result


def _split_atom_rows(rows, *, presence=None, sources=None):
    """Split the historical supplied schema once into table rows and cells."""
    from ._atom_table import _metadata_array, _SOURCE_CODES
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(ATOM_FIELDS):
        raise ValueError(f"Atom rows must have shape (N, {len(ATOM_FIELDS)})")
    masks = _metadata_array(ATOM_PRESENCE_ALL if presence is None else presence,
                            (len(values),), np.uint32, ATOM_PRESENCE_ALL, "presence")
    if sources is None:
        supplied = (masks[:, None] & np.asarray(FIELD_PRESENCE_BITS, dtype=np.uint32)) != 0
        codes = np.where(supplied, ATOM_SOURCE_CODES["legacy_array"],
                         ATOM_SOURCE_CODES["default"]).astype(np.uint8)
    else:
        codes = _metadata_array(sources, values.shape, np.uint8, max(_SOURCE_CODES), "sources")
        if not np.isin(codes, tuple(_SOURCE_CODES)).all():
            raise ValueError("Atom sources contain an undeclared source code")
    width = len(TABLE_FIELDS)
    raw_axes = np.asarray(values[:, width:], dtype="<f8")
    cells = tuple(raw_axes[row].tobytes() + bytes((int(mask) >> width,))
                  + codes[row, width:].tobytes() for row, mask in enumerate(masks))
    return (values[:, :width], masks & TABLE_PRESENCE_ALL, codes[:, :width], cells)


def _select_atom_values(table, cells, uids, fields=ATOM_FIELDS, *, rows=None, dtype="float64"):
    """Join only requested raw fields for explicitly aligned selected UIDs."""
    selection, count = table._rows(rows)
    result = np.empty((count, len(fields)), dtype=dtype)
    for target, name in enumerate(fields):
        index = FIELD_INDEX[name]
        if name in CONSTRAINT_FIELDS:
            axis = index - len(TABLE_FIELDS)
            result[:, target] = tuple(_atom_axis_value(cells[uid], axis) for uid in uids)
        else:
            column = table._columns[index]
            result[:, target] = np.nan if column is None else column[selection]
    return result


def _select_atom_metadata(table, cells, uids, indices, *, rows=None):
    """Join selected source columns and their historical presence bits."""
    selection, count = table._rows(rows)
    masks = table._presence[selection].copy()
    sources = np.empty((count, len(indices)), dtype=np.uint8)
    axis_cells = tuple(cells[uid] for uid in uids) if any(index >= len(TABLE_FIELDS) for index in indices) else ()
    for target, index in enumerate(indices):
        if index < len(TABLE_FIELDS):
            sources[:, target] = table._sources[selection, index]
        else:
            axis = index - len(TABLE_FIELDS)
            sources[:, target] = tuple(cell[25 + axis] for cell in axis_cells)
    if axis_cells:
        masks |= np.asarray(tuple(cell[24] << len(TABLE_FIELDS) for cell in axis_cells), dtype=np.uint32)
    masks &= sum(FIELD_PRESENCE_BITS[index] for index in indices)
    return masks, sources


def _join_atom_rows(table, cells, uids, *, rows=None):
    values = _select_atom_values(table, cells, uids, rows=rows)
    masks, sources = _select_atom_metadata(table, cells, uids, tuple(range(len(ATOM_FIELDS))), rows=rows)
    return values, masks, sources


def _updated_entries(current, updates):
    entries = dict.fromkeys(current)
    for members, value in updates:
        if value:
            entries[members] = None
        else:
            entries.pop(members, None)
    return tuple(entries)


@dataclass(frozen=True)
class _ConstraintSet:
    angles: tuple[tuple[int, int, int], ...] = ()
    torsions: tuple[tuple[int, int, int, int], ...] = ()
    atom_axes: dict[int, bytes] = field(default_factory=dict)
    bond_values: dict[int, bytes] = field(default_factory=dict)

    def updated(self, *, angles=(), torsions=()):
        return type(self)(
            _updated_entries(self.angles, angles),
            _updated_entries(self.torsions, torsions),
            self.atom_axes, self.bond_values,
        )

    def subset(self, members, bonds):
        return type(self)(
            tuple(entry for entry in self.angles if all(uid in members for uid in entry)),
            tuple(entry for entry in self.torsions if all(uid in members for uid in entry)),
            {uid: self.atom_axes[uid] for uid in members},
            {uid: self.bond_values[uid] for uid in bonds},
        )


def _bind_constraint_members(mol):
    for atom in mol._atoms:
        object.__setattr__(atom, '_constraint_values', mol._constraints.atom_axes)
    for bond in mol._bonds:
        object.__setattr__(bond, '_constraint_values', mol._constraints.bond_values)


def _filter_constraint_members(mol):
    atoms = {atom._uid for atom in mol._atoms}
    bonds = {bond._uid for bond in mol._bonds}
    if mol._constraints.atom_axes.keys() == atoms and mol._constraints.bond_values.keys() == bonds:
        return
    mol._constraints = mol._constraints.subset(atoms, bonds)
    _bind_constraint_members(mol)
