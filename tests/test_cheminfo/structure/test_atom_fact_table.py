"""Numeric ownership, atomic updates and remapping before facade integration."""

from copy import deepcopy
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

from hotpot import Atom
from hotpot.cheminfo.structure._commit import _native_fail_after
from hotpot.cheminfo.structure._atom_table import _AtomFactTable
from hotpot.cheminfo.structure._fields import (
    ALIAS_TARGETS,
    ATOM_BATCH_WRITABLE,
    ATOM_FIELDS,
    ATOM_PRESENCE_ALL,
    ATOM_SCHEMA,
    ATOM_SOURCE_CODES,
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


@pytest.fixture(params=("pickle", "deepcopy"))
def roundtrip(request):
    if request.param == "pickle":
        return lambda table: pickle.loads(pickle.dumps(table))
    return deepcopy


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
    presence, sources = table.presence(), table.sources()
    with pytest.raises(exception):
        table.update(rows, columns)
    np.testing.assert_array_equal(table.matrix(), expected)
    np.testing.assert_array_equal(table.presence(), presence)
    np.testing.assert_array_equal(table.sources(), sources)


@pytest.mark.parametrize("rows", [1, (1, 3), None])
def test_read_only_target_rejects_the_complete_write_before_install(rows):
    expected = _rows()
    table = _AtomFactTable(expected)
    table._columns[FIELD_INDEX["y"]].flags.writeable = False
    with _native_fail_after("value"), pytest.raises(ValueError):
        table.update(rows, {"x": 100, "y": 200})
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
    presence, sources = table.presence(), table.sources()
    count = len(table) if rows is None else 1 if isinstance(rows, int) else len(rows)
    occurrence = tuple(updates).index("y") * count + 1
    with _native_fail_after("value", occurrence), pytest.raises(KeyboardInterrupt, match="native fact store"):
        table.update(rows, updates)
    np.testing.assert_array_equal(table.matrix(), expected)
    np.testing.assert_array_equal(table.presence(), presence)
    np.testing.assert_array_equal(table.sources(), sources)


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


def test_roundtrip_rebuilds_column_views_and_preserves_independent_metadata(roundtrip):
    expected = _rows()
    masks = np.array([0, ATOM_PRESENCE_ALL, 1, 2], dtype=np.uint32)
    codes = np.arange(4 * 19, dtype=np.uint8).reshape(4, 19) % len(ATOM_SOURCE_CODES)
    original = _AtomFactTable(expected, presence=masks, sources=codes)
    restored = roundtrip(original)
    np.testing.assert_array_equal(restored.matrix(), expected)
    np.testing.assert_array_equal(restored.presence(), masks)
    np.testing.assert_array_equal(restored.sources(), codes)
    assert all(sum(np.shares_memory(column, block) for block in restored._blocks) == 1
               for column in restored._columns)
    assert all(not np.shares_memory(left, right)
               for left in restored._blocks for right in original._blocks)
    assert not np.shares_memory(restored._presence, original._presence)
    assert not np.shares_memory(restored._sources, original._sources)

    restored.update(0, {"x": -123, "formal_charge": 88}, source=ATOM_SOURCE_CODES["rdkit"])
    restored.update((3, 1), {"z": (13, 17)}, present=False)
    expected[0, [10, 7]] = (-123, 88)
    expected[[3, 1], 12] = (13, 17)
    masks[0] |= (1 << 10) | (1 << 7)
    masks[[3, 1]] &= np.uint32(ATOM_PRESENCE_ALL ^ (1 << 12))
    codes[0, [10, 7]] = ATOM_SOURCE_CODES["rdkit"]
    codes[[3, 1], 12] = ATOM_SOURCE_CODES["explicit"]
    np.testing.assert_array_equal(restored.matrix(), expected)
    np.testing.assert_array_equal(restored.coordinates(), expected[:, 10:13])
    np.testing.assert_array_equal(restored.presence(), masks)
    np.testing.assert_array_equal(restored.sources(), codes)
    for row in range(len(restored)):
        np.testing.assert_array_equal(restored.row(row), expected[row])
        for field in ("x", "z", "formal_charge"):
            assert restored.raw(row, field) == expected[row, FIELD_INDEX[field]]
    np.testing.assert_array_equal(original.matrix(), _rows())
    assert original.presence()[0] == 0


def test_roundtrip_preserves_spare_capacity_and_rebinds_after_append_remap(roundtrip):
    original = _AtomFactTable(_rows(6), presence=0, sources=ATOM_SOURCE_CODES["rdkit"])
    original.remap((4, 2, 0))
    expected = original.matrix()
    restored = roundtrip(original)
    blocks = restored._blocks
    assert len(blocks[0]) == 6
    addition = _rows(2) + 100
    restored.append(addition)
    assert restored._blocks is blocks
    expected = np.concatenate((expected, addition))
    restored.remap((4, 0, 3))
    expected = expected[[4, 0, 3]]
    restored.update(1, {"x": -123})
    expected[1, 10] = -123
    np.testing.assert_array_equal(restored.matrix(), expected)
    np.testing.assert_array_equal(restored.coordinates(), expected[:, 10:13])
    assert restored.raw(1, "x") == -123
    np.testing.assert_array_equal(restored.presence(), [ATOM_PRESENCE_ALL, 1 << 10, ATOM_PRESENCE_ALL])
    assert restored.sources()[1, 0] == ATOM_SOURCE_CODES["rdkit"]
    assert restored.sources()[0, 0] == ATOM_SOURCE_CODES["legacy_array"]
    np.testing.assert_array_equal(original.matrix(), _rows(6)[[4, 2, 0]])


@pytest.mark.parametrize("capacity", (0, 6))
def test_empty_roundtrip_can_append_and_grow_with_one_numeric_owner(roundtrip, capacity):
    original = _AtomFactTable(_rows(capacity))
    original.remap([])
    restored = roundtrip(original)
    assert len(restored) == 0 and len(restored._blocks[0]) == capacity
    assert restored.matrix().shape == (0, 19)
    assert restored.presence().shape == (0,)
    assert restored.sources().shape == (0, 19)
    with pytest.raises(IndexError):
        restored.raw(0, "x")
    expected = _rows(capacity + 1)
    restored.append(expected, presence=0, sources=ATOM_SOURCE_CODES["openbabel"])
    restored.update(-1, {"y": -77})
    expected[-1, 11] = -77
    np.testing.assert_array_equal(restored.matrix(), expected)
    np.testing.assert_array_equal(restored.coordinates(), expected[:, 10:13])
    assert restored.raw(-1, "y") == -77
    assert restored.presence()[-1] == 1 << 11
    assert restored.sources()[0, 0] == ATOM_SOURCE_CODES["openbabel"]
    assert len(original) == 0


def test_pickle_does_not_serialize_unused_numeric_or_metadata_slots():
    table = _AtomFactTable(_rows(6))
    table.remap((0,))
    marker = b"unused!!"
    table._blocks[0][1:] = np.frombuffer(marker, dtype=np.float64)[0]
    table._blocks[1][1:] = np.frombuffer(marker, dtype=np.float64)[0]
    table._presence[1:].view(np.uint8)[:len(marker)] = np.frombuffer(marker, dtype=np.uint8)
    table._sources[1:].ravel()[:len(marker)] = np.frombuffer(marker, dtype=np.uint8)
    serialized = pickle.dumps(table, protocol=pickle.HIGHEST_PROTOCOL)
    assert marker not in serialized
    restored = pickle.loads(serialized)
    assert len(restored._blocks[0]) == 6 and len(restored) == 1
    np.testing.assert_array_equal(restored.matrix(), _rows(6)[:1])


def test_presence_distinguishes_absent_and_explicit_zero_without_changing_numeric_slots():
    table = _AtomFactTable(np.zeros((2, 19)), presence=0)
    table.update(0, {"formal_charge": 0})
    assert table.raw(0, "formal_charge") == table.raw(1, "formal_charge") == 0
    np.testing.assert_array_equal(table.presence(), [1 << FIELD_INDEX["formal_charge"], 0])
    assert table.sources()[0, FIELD_INDEX["formal_charge"]] == ATOM_SOURCE_CODES["explicit"]
    assert table.sources()[1, FIELD_INDEX["formal_charge"]] == ATOM_SOURCE_CODES["default"]
    assert table.presence().dtype == np.uint32
    assert table.sources().dtype == np.uint8


def test_raw_rows_default_to_legacy_supplied_metadata_and_empty_shapes_are_explicit():
    table = _AtomFactTable(np.zeros((2, 19)))
    np.testing.assert_array_equal(table.presence(), np.full(2, ATOM_PRESENCE_ALL, dtype=np.uint32))
    np.testing.assert_array_equal(table.sources(), np.full((2, 19), ATOM_SOURCE_CODES["legacy_array"], dtype=np.uint8))
    empty = _AtomFactTable(np.empty((0, 19)), presence=[], sources=np.empty((0, 19)))
    assert empty.presence().shape == (0,)
    assert empty.sources().shape == (0, 19)
    with pytest.raises(TypeError):
        ATOM_SOURCE_CODES["custom"] = 10


def test_mixed_field_metadata_is_installed_with_one_numeric_write_set():
    table = _AtomFactTable(_rows(), presence=0)
    columns = {"atomic_number": 8, "n": 2, "x": 0}
    present = {"atomic_number": True, "n": True, "x": False}
    source = {
        "atomic_number": ATOM_SOURCE_CODES["explicit"],
        "n": ATOM_SOURCE_CODES["electron_configuration"],
        "x": ATOM_SOURCE_CODES["openbabel"],
    }
    table.update((2, 0), columns, present=present, source=source)
    mask = (1 << FIELD_INDEX["atomic_number"]) | (1 << FIELD_INDEX["n"])
    np.testing.assert_array_equal(table.presence(), [mask, 0, mask, 0])
    for field, value in columns.items():
        np.testing.assert_array_equal(table.column(field, (2, 0)), [value, value])
        np.testing.assert_array_equal(table.sources((2, 0))[:, FIELD_INDEX[field]], [source[field], source[field]])
    # An absent toolkit slot can still retain its explicit source diagnostic.
    assert table.sources()[0, FIELD_INDEX["x"]] == ATOM_SOURCE_CODES["openbabel"]
    table.update(0, {"n": 0}, present=False, source=ATOM_SOURCE_CODES["default"])
    assert table.presence()[0] == 1 << FIELD_INDEX["atomic_number"]


@pytest.mark.parametrize("option, value", [
    ("present", 1),
    ("present", {"x": True}),
    ("present", {"x": True, "y": True, "z": True}),
    ("source", 255),
    ("source", 1.5),
    ("source", {"x": ATOM_SOURCE_CODES["explicit"]}),
])
@pytest.mark.parametrize("rows", [0, (0, 2)])
def test_invalid_metadata_rejects_numeric_and_metadata_changes_together(rows, option, value):
    table = _AtomFactTable(_rows(), presence=0)
    numeric, presence, sources = table.matrix(), table.presence(), table.sources()
    with pytest.raises((TypeError, ValueError)):
        table.update(rows, {"x": 100, "y": 200}, **{option: value})
    np.testing.assert_array_equal(table.matrix(), numeric)
    np.testing.assert_array_equal(table.presence(), presence)
    np.testing.assert_array_equal(table.sources(), sources)


@pytest.mark.parametrize("metadata", ["_presence", "_sources"])
@pytest.mark.parametrize("rows, columns", [
    (1, {"x": 100}),
    (1, {"x": 100, "y": 200}),
    ((1, 3), {"x": 100, "y": 200}),
    (None, {"x": 100, "y": 200}),
])
def test_metadata_write_interruption_restores_numeric_presence_and_source(metadata, rows, columns):
    table = _AtomFactTable(_rows(), presence=0)
    numeric, presence, sources = table.matrix(), table.presence(), table.sources()
    boundary = "presence" if metadata == "_presence" else "source"
    with _native_fail_after(boundary), pytest.raises(KeyboardInterrupt):
        table.update(rows, columns)
    np.testing.assert_array_equal(table.matrix(), numeric)
    np.testing.assert_array_equal(table.presence(), presence)
    np.testing.assert_array_equal(table.sources(), sources)


@pytest.mark.parametrize("metadata", ["_presence", "_sources"])
@pytest.mark.parametrize("rows", [1, (1, 3), None])
def test_read_only_metadata_is_rejected_before_numeric_install(metadata, rows):
    table = _AtomFactTable(_rows(), presence=0)
    numeric, presence, sources = table.matrix(), table.presence(), table.sources()
    getattr(table, metadata).flags.writeable = False
    with _native_fail_after("value"), pytest.raises(ValueError):
        table.update(rows, {"x": 100})
    np.testing.assert_array_equal(table.matrix(), numeric)
    np.testing.assert_array_equal(table.presence(), presence)
    np.testing.assert_array_equal(table.sources(), sources)


def test_metadata_inputs_exports_growth_remap_and_copies_preserve_exact_rows():
    masks = np.array([0, 1 << FIELD_INDEX["x"], ATOM_PRESENCE_ALL], dtype=np.uint32)
    codes = np.arange(3 * 19, dtype=np.uint8).reshape(3, 19) % len(ATOM_SOURCE_CODES)
    expected_masks, expected_codes = masks.copy(), codes.copy()
    table = _AtomFactTable(_rows(3), presence=masks, sources=codes)
    masks[:] = 0
    codes[:] = 0
    table.append(_rows(2), presence=[0, 1 << FIELD_INDEX["id"]], sources=ATOM_SOURCE_CODES["rdkit"])
    expected_masks = np.concatenate((expected_masks, [0, 1 << FIELD_INDEX["id"]])).astype(np.uint32)
    expected_codes = np.concatenate((expected_codes, np.full((2, 19), ATOM_SOURCE_CODES["rdkit"], dtype=np.uint8)))
    table.remap((4, 0, 2))
    expected_masks, expected_codes = expected_masks[[4, 0, 2]], expected_codes[[4, 0, 2]]
    clone, subset = table.copy(), table.take((2, 0))
    np.testing.assert_array_equal(table.presence(), expected_masks)
    np.testing.assert_array_equal(table.sources(), expected_codes)
    np.testing.assert_array_equal(subset.presence(), expected_masks[[2, 0]])
    np.testing.assert_array_equal(subset.sources(), expected_codes[[2, 0]])
    returned_masks, returned_codes = table.presence(), table.sources()
    returned_masks[:] = 0
    returned_codes[:] = 0
    table.update(0, {"x": -1}, source=ATOM_SOURCE_CODES["mca_dto"])
    np.testing.assert_array_equal(clone.presence(), expected_masks)
    np.testing.assert_array_equal(clone.sources(), expected_codes)
    for other in (clone, subset):
        assert not np.shares_memory(table._presence, other._presence)
        assert not np.shares_memory(table._sources, other._sources)
    assert clone._sources.shape == (len(clone), 19)
    assert subset._presence.shape == (len(subset),)


@pytest.mark.parametrize("kwargs", [
    {"presence": [0, ATOM_PRESENCE_ALL + 1]},
    {"presence": [0.5, 0]},
    {"sources": 255},
    {"sources": np.zeros((2, 18), dtype=np.uint8)},
])
def test_invalid_append_metadata_leaves_values_metadata_and_capacity_unchanged(kwargs):
    table = _AtomFactTable(_rows(), presence=0)
    numeric, presence, sources = table.matrix(), table.presence(), table.sources()
    blocks, masks_owner, sources_owner = table._blocks, table._presence, table._sources
    with pytest.raises((TypeError, ValueError)):
        table.append(_rows(2), **kwargs)
    assert table._blocks is blocks
    assert table._presence is masks_owner
    assert table._sources is sources_owner
    np.testing.assert_array_equal(table.matrix(), numeric)
    np.testing.assert_array_equal(table.presence(), presence)
    np.testing.assert_array_equal(table.sources(), sources)


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
