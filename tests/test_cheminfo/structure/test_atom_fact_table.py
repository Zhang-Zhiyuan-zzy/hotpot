"""Numeric ownership, atomic updates and remapping before facade integration."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hotpot import Atom
from hotpot.cheminfo.structure._atom_table import _AtomFactTable
from hotpot.cheminfo.structure._fields import (
    ALIAS_TARGETS,
    ATOM_BATCH_WRITABLE,
    ATOM_FIELDS,
    ATOM_SCHEMA,
    FIELD_INDEX,
)


def _rows(count=4):
    values = np.arange(count * 19, dtype=np.float64).reshape(count, 19) + 0.25
    values[:, FIELD_INDEX["atomic_number"]] = 6.5
    values[:, FIELD_INDEX["is_aromatic"]] = 2.5
    values[:, FIELD_INDEX["x_constraint"]] = -2.5
    values[:, FIELD_INDEX["formal_charge"]] = -32769.5
    values[:, FIELD_INDEX["implicit_hydrogens"]] = 65537.25
    values[:, FIELD_INDEX["id"]] = float(2 ** 70)
    return values


class _WriteThenRaise(np.ndarray):
    """A one-shot failure after data changed, including KeyboardInterrupt."""

    def __new__(cls, column):
        value = column.view(cls)
        value.armed = True
        return value

    def __array_finalize__(self, source):
        self.armed = getattr(source, "armed", False)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.armed:
            self.armed = False
            raise KeyboardInterrupt("injected after a completed column write")


def test_schema_preserves_model_order_and_does_not_publish_internal_columns():
    assert ATOM_FIELDS == (
        "atomic_number", "n", "s", "p", "d", "f", "g", "formal_charge",
        "partial_charge", "is_aromatic", "x", "y", "z", "valence",
        "implicit_hydrogens", "id", "x_constraint", "y_constraint", "z_constraint",
    )
    assert ALIAS_TARGETS["coordinates"] == ("x", "y", "z")
    assert ATOM_BATCH_WRITABLE == {
        "atomic_number", "symbol", "formal_charge", "implicit_hydrogens",
        "id", "coordinates", "x", "y", "z",
    }
    with pytest.raises(TypeError):
        FIELD_INDEX["mca"] = 19


def test_raw_numeric_domain_is_preserved_separately_from_scalar_codecs():
    source = _rows()
    expected = source.copy()
    table = _AtomFactTable(source)
    atom = Atom(attrs_array=expected[0].copy(), update_electron_config=False)
    source[:] = -1
    np.testing.assert_array_equal(table.matrix(), expected)
    assert table.matrix().dtype == np.float64
    for index, (field, codec) in enumerate(ATOM_SCHEMA):
        assert table.raw_index(0, index) == expected[0, index]
        assert table.raw(0, field) == expected[0, index]
        assert table.read(0, field) == codec(expected[0, index])
        assert table.read(0, field) == getattr(atom, field)
        np.testing.assert_array_equal(table.column(field), expected[:, index])
    assert table.raw(0, "is_aromatic") == 2.5
    assert table.read(0, "is_aromatic") is True


@pytest.mark.parametrize("export", [
    lambda table: table.row(0),
    lambda table: table.column("x"),
    lambda table: table.coordinates(),
    lambda table: table.matrix(),
    lambda table: table.select(("id", "is_aromatic", "x")),
    lambda table: table.column("x", (2, 0)),
])
def test_each_numeric_projection_is_an_independent_writable_array(export):
    expected = _rows()
    table = _AtomFactTable(expected)
    returned = export(table)
    assert returned.flags.writeable
    for block in table._blocks:
        assert not np.shares_memory(returned, block)
    returned[:] = -901
    np.testing.assert_array_equal(table.matrix(), expected)


def test_private_selection_preserves_field_row_order_and_requested_dtype():
    source = _rows()
    table = _AtomFactTable(source)
    selected = table.select(("z", "atomic_number", "z"), rows=(3, 0), dtype="float32")
    np.testing.assert_array_equal(selected, source[np.ix_((3, 0), (12, 0, 12))].astype("float32"))
    assert selected.dtype == np.float32
    assert table.select((), rows=(3, 0)).shape == (2, 0)
    np.testing.assert_array_equal(table.coordinates((2, 0)), source[[2, 0], 10:13])


def test_empty_tables_and_empty_selections_keep_explicit_shapes():
    table = _AtomFactTable()
    assert len(table) == 0
    assert table.matrix().shape == (0, 19)
    assert table.coordinates().shape == (0, 3)
    assert table.column("x").shape == (0,)
    assert table.select(("x", "id")).shape == (0, 2)
    table.update(None, {"x": 8})
    assert table.append(np.empty((0, 19))) == 0
    table.remap([])
    assert table.copy().matrix().shape == (0, 19)


def test_scalar_and_batch_updates_share_the_same_raw_value_semantics():
    expected = _rows()
    table = _AtomFactTable(expected)
    table.update(1, {"atomic_number": 8.5, "is_aromatic": 0.5, "x": -3.5})
    expected[1, [0, 9, 10]] = (8.5, 0.5, -3.5)
    charges = np.array([2.25, -4.5])
    table.update((3, 0), {"formal_charge": charges, "y": 7})
    expected[[3, 0], 7] = charges
    expected[[3, 0], 11] = 7
    charges[:] = -900
    table.update(2, {"z": [42.5]})
    expected[2, 12] = 42.5
    np.testing.assert_array_equal(table.matrix(), expected)


@pytest.mark.parametrize("rows, columns, exception", [
    (0, {"x": 3, "y": object()}, (TypeError, ValueError)),
    (0, {"x": 3, "y": [1, 2]}, ValueError),
    ((0, 1), {"x": [3, 4], "y": [2]}, ValueError),
    ((0, 1), {"x": [3, 4], "y": [2, object()]}, (TypeError, ValueError)),
    ((0, 1), {"x": 3, "unregistered": 2}, KeyError),
    ((0, 0), {"x": 3}, ValueError),
    ((0, 40), {"x": 3}, IndexError),
    ((0.5, 1), {"x": 3}, TypeError),
    (np.array([2 ** 64 - 1], dtype=np.uint64), {"x": 3}, IndexError),
])
def test_failed_update_preserves_the_complete_previous_table(rows, columns, exception):
    expected = _rows()
    table = _AtomFactTable(expected)
    with pytest.raises(exception):
        table.update(rows, columns)
    np.testing.assert_array_equal(table.matrix(), expected)


@pytest.mark.parametrize("rows", [1, (1, 3), None])
def test_read_only_target_rejects_the_complete_write_before_install(rows):
    expected = _rows()
    table = _AtomFactTable(expected)
    columns = list(table._columns)
    columns[FIELD_INDEX["x"]] = _WriteThenRaise(columns[FIELD_INDEX["x"]])
    table._columns = tuple(columns)
    table._columns[FIELD_INDEX["y"]].flags.writeable = False
    with pytest.raises(ValueError):
        table.update(rows, {"x": 100, "y": 200})
    assert table._columns[FIELD_INDEX["x"]].armed
    np.testing.assert_array_equal(table.matrix(), expected)


@pytest.mark.parametrize("rows, updates", [
    (1, {"y": 200}),
    (1, {"x": 100, "y": 200, "z": 300}),
    ((1, 3), {"y": (200, 250)}),
    ((1, 3), {"x": 100, "y": (200, 250), "z": 300}),
    (None, {"x": 100, "y": 200, "z": 300}),
])
def test_interrupted_write_restores_current_earlier_and_uninstalled_columns(rows, updates):
    expected = _rows()
    table = _AtomFactTable(expected)
    columns = list(table._columns)
    columns[FIELD_INDEX["y"]] = _WriteThenRaise(columns[FIELD_INDEX["y"]])
    table._columns = tuple(columns)
    with pytest.raises(KeyboardInterrupt, match="after a completed column write"):
        table.update(rows, updates)
    assert not table._columns[FIELD_INDEX["y"]].armed
    np.testing.assert_array_equal(table.matrix(), expected)


@pytest.mark.parametrize("read", [
    lambda table, row: table.raw_index(row, FIELD_INDEX["x"]),
    lambda table, row: table.raw(row, "x"),
    lambda table, row: table.read(row, "x"),
])
def test_scalar_reads_normalize_negative_rows_and_never_read_spare_capacity(read):
    table = _AtomFactTable(_rows())
    table.append(_rows(1))
    assert len(table._blocks[0]) > len(table)
    assert read(table, -1) == table.matrix()[-1, FIELD_INDEX["x"]]
    assert read(table, np.int64(0)) == table.matrix()[0, FIELD_INDEX["x"]]
    with pytest.raises(IndexError):
        read(table, len(table))
    with pytest.raises(IndexError):
        read(table, -len(table) - 1)
    with pytest.raises(TypeError):
        read(table, 1.5)


def test_append_growth_rebinds_column_views_without_duplicate_numeric_owners():
    expected = _rows(1)
    table = _AtomFactTable(expected)
    for _ in range(12):
        addition = _rows(1) + len(table)
        original = addition.copy()
        assert table.append(addition) == len(expected)
        expected = np.concatenate((expected, original))
        addition[:] = -800
        np.testing.assert_array_equal(table.matrix(), expected)
        assert table._blocks[1].flags.c_contiguous
        assert all(sum(np.shares_memory(column, block) for block in table._blocks) == 1
                   for column in table._columns)
    assert len(table._blocks[0]) >= len(table)
    with pytest.raises(IndexError):
        table.raw(len(table), "x")


@pytest.mark.parametrize("bad", [np.zeros(19), np.zeros((2, 18)), [[object()] * 19]])
def test_failed_append_does_not_change_rows_or_capacity(bad):
    table = _AtomFactTable(_rows())
    original = table.matrix()
    blocks = table._blocks
    with pytest.raises((ValueError, TypeError)):
        table.append(bad)
    assert table._blocks is blocks
    np.testing.assert_array_equal(table.matrix(), original)


def test_remap_and_independent_tables_keep_all_numeric_fields_and_selection_order():
    original = _rows()
    table = _AtomFactTable(original)
    clone = table.copy()
    subset = table.take((2, 0))
    table.remap((3, 1, 0))
    np.testing.assert_array_equal(table.matrix(), original[[3, 1, 0]])
    np.testing.assert_array_equal(subset.matrix(), original[[2, 0]])
    table.update(0, {"id": -20})
    subset.update(0, {"formal_charge": -80})
    np.testing.assert_array_equal(clone.matrix(), original)
    for other in (clone, subset):
        assert all(not np.shares_memory(left, right)
                   for left in table._blocks for right in other._blocks)
    current = table.matrix()
    with pytest.raises(ValueError):
        table.remap((0, 0))
    np.testing.assert_array_equal(table.matrix(), current)
    table.remap([])
    assert len(table) == 0
    assert table.matrix().shape == (0, 19)


def test_remap_keeps_capacity_for_a_subsequent_append_without_shared_rows():
    original = _rows(6)
    table = _AtomFactTable(original)
    table.remap((4, 2, 0, 1))
    blocks = table._blocks
    assert len(blocks[0]) == 6
    addition = _rows(2) + 100
    table.append(addition)
    assert table._blocks is blocks
    np.testing.assert_array_equal(table.matrix(), np.concatenate((original[[4, 2, 0, 1]], addition)))
    assert len(table.copy()._blocks[0]) == len(table)


def test_storage_leaf_import_loads_no_carrier_or_execution_backend(tmp_path):
    module = sys.modules[_AtomFactTable.__module__]
    package = Path(module.__file__).parent
    code = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location('_table_leaf', {str(package / '__init__.py')!r}, submodule_search_locations=[{str(package)!r}])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from _table_leaf._atom_table import _AtomFactTable
assert _AtomFactTable().matrix().shape == (0, 19)
for name in ('hotpot', 'rdkit', 'openbabel', 'scipy', 'thermo'):
    assert not any(module == name or module.startswith(name + '.') for module in sys.modules), name
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
