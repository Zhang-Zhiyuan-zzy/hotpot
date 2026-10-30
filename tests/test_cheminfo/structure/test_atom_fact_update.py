"""Closed candidate preparation, independently of the pending owner cutover."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import pickle

import numpy as np
import pytest

from hotpot.cheminfo.reference.properties import _atomic_number_for_symbol
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
from hotpot.cheminfo.structure._identity import _SourceIdentity
from hotpot.cheminfo.structure._mutation import _normalize_atom_facts


def prepare(columns, rows=(2, 0), uids=(17, 8)):
    return _normalize_atom_facts(_SourceIdentity(object(), ()), rows, uids, columns)


def values(update):
    return dict(zip(update.fields, update.values))


def test_aliases_agree_and_selection_order_is_preserved():
    update = prepare({
        "x": [1, 4], "coordinates": [[1, 2, 3], [4, 5, 6]],
        "symbol": ["c", "In"], "atomic_number": [6, 49],
        "formal_charge": 0, "id": [2**63 + 4096, 1e30],
    })
    assert update.rows == (2, 0)
    assert update.atom_uids == (17, 8)
    assert update.fields == ("atomic_number", "formal_charge", "x", "y", "z", "id")
    np.testing.assert_array_equal(values(update)["atomic_number"], [6, 49])
    np.testing.assert_array_equal(values(update)["formal_charge"], [0, 0])
    np.testing.assert_array_equal(values(update)["z"], [3, 6])
    np.testing.assert_array_equal(values(update)["id"], [2**63 + 4096, 1e30])
    assert update.presence == (True,) * 6  # Explicit zero is supplied.
    assert update.sources == (ATOM_SOURCE_CODES["explicit"],) * 6


def test_scalar_and_batch_prepare_the_same_one_row_candidate():
    scalar = prepare({"symbol": "Fe", "coordinates": (1, 2, 3)}, (7,), (19,))
    batch = prepare({"atomic_number": [26], "coordinates": [[1, 2, 3]]}, (7,), (19,))
    assert scalar.fields == batch.fields
    for left, right in zip(scalar.values, batch.values):
        np.testing.assert_array_equal(left, right)


def test_candidates_are_isolated_and_cannot_reenable_numpy_writes():
    coordinate_input = np.arange(6.0).reshape(2, 3)
    update = prepare({"coordinates": coordinate_input})
    coordinate_input[:] = -1
    np.testing.assert_array_equal(values(update)["x"], [0, 3])
    for column in update.values:
        assert column.dtype == np.float64
        with pytest.raises(ValueError):
            column.setflags(write=True)
        with pytest.raises(ValueError):
            column[0] = 123
    with pytest.raises(FrozenInstanceError):
        update.rows = (0, 1)


@pytest.mark.parametrize("copy_candidate", [deepcopy, lambda x: pickle.loads(pickle.dumps(x))])
def test_serialized_candidates_remain_immutable(copy_candidate):
    original = prepare({"x": [1, 2]})
    copied = copy_candidate(original)
    np.testing.assert_array_equal(values(copied)["x"], [1, 2])
    with pytest.raises(ValueError):
        copied.values[0].setflags(write=True)


def test_deepcopy_retains_the_original_source_binding():
    update = prepare({"implicit_hydrogens": 3})
    assert deepcopy(update).source.owner is update.source.owner


def test_numpy_header_changes_cannot_mutate_the_saved_candidate():
    update = prepare({"x": [1, 2]})
    header = update.values[0]
    header.dtype = np.uint64
    header.shape = (2, 1)
    for candidate in (update, deepcopy(update), pickle.loads(pickle.dumps(update))):
        assert candidate.values[0].dtype == np.float64
        assert candidate.values[0].shape == (2,)
        np.testing.assert_array_equal(candidate.values[0], [1, 2])


def test_pickle_preserves_joint_owner_binding_and_rejects_standalone_binding():
    from hotpot import Molecule
    from hotpot.cheminfo.structure._identity import _capture_source, _check_source

    mol = Molecule()
    atom = mol.create_atom(symbol="C")
    update = _normalize_atom_facts(_capture_source(mol), (atom.idx,), (atom._uid,), {"x": 2})
    restored_mol, restored_update = pickle.loads(pickle.dumps((mol, update)))
    _check_source(restored_mol, restored_update.source)
    with pytest.raises(RuntimeError, match="owner"):
        _check_source(mol, pickle.loads(pickle.dumps(update)).source)


@pytest.mark.parametrize("columns", [
    {"symbol": "C", "atomic_number": 7},
    {"coordinates": (1, 2, 3), "x": [1, 9]},
    {"x": [1]}, {"symbol": ["C"]},
    {"coordinates": np.zeros((1, 3))},
    {"coordinates": (1, 2)}, {"formal_charge": [[0], [1]]},
])
def test_incomplete_or_conflicting_input_is_rejected(columns):
    with pytest.raises(ValueError):
        prepare(columns)


@pytest.mark.parametrize("name", [
    "n", "s", "p", "d", "f", "g", "valence", "partial_charge",
    "is_aromatic", "constraint", "x_constraint", "idx", "mol", "mca", "attrs",
])
def test_batch_permissions_do_not_expose_internal_or_derived_writes(name):
    with pytest.raises(AttributeError):
        prepare({name: 0})


def test_unknown_field_is_not_an_arbitrary_attribute_writer():
    with pytest.raises(TypeError):
        prepare({"new_attrs": 0})


@pytest.mark.parametrize("rows,uids", [((0, 0), (1, 2)), ((0, 1), (2, 2)),
                                       ((0,), (1, 2)), ((-1,), (1,))])
def test_invalid_row_uid_alignment_is_rejected(rows, uids):
    with pytest.raises(ValueError):
        prepare({"x": 1}, rows, uids)


def test_empty_selection_accepts_empty_columns_and_coordinate_broadcast():
    update = prepare({"coordinates": (1, 2, 3), "symbol": []}, (), ())
    assert update.fields == ("atomic_number", "x", "y", "z")
    assert all(column.shape == (0,) for column in update.values)
    assert prepare({}).fields == ()


def test_equal_nan_aliases_preserve_the_existing_raw_numeric_domain():
    update = prepare({"coordinates": (np.nan, 2, 3), "x": np.nan, "formal_charge": 2.5})
    np.testing.assert_array_equal(values(update)["x"], [np.nan, np.nan])
    np.testing.assert_array_equal(values(update)["formal_charge"], [2.5, 2.5])


@pytest.mark.parametrize("changes,exception", [
    ({"fields": ("x", "x")}, ValueError),
    ({"presence": ()}, ValueError),
    ({"presence": (1,)}, TypeError),
    ({"sources": (250,)}, ValueError),
])
def test_direct_candidate_metadata_remains_closed(changes, exception):
    with pytest.raises(exception):
        replace(prepare({"x": 1}), **changes)


def test_symbol_translation_reuses_existing_order_without_json_number_errors():
    from hotpot.cheminfo.elements import Element
    from openbabel import openbabel as ob

    for symbol in Element.symbols[1:]:
        for spelling in (symbol, symbol.lower(), symbol.upper(), symbol.swapcase()):
            assert _atomic_number_for_symbol(spelling) == ob.GetAtomicNum(spelling)
    for spelling in ("D", "T", "d", "t", "X", "*", "Carbon", "", "ſ", "ı", "ſn", "ın"):
        assert _atomic_number_for_symbol(spelling) == ob.GetAtomicNum(spelling)


class _UnconvertedDerivedValue:
    def __float__(self):
        raise AssertionError("Derived input must be rejected before numeric conversion")

    def __array__(self, dtype=None, copy=None):
        raise AssertionError("Mixed input must be rejected before numeric conversion")


def _stored_atom_state(mol):
    table = mol._atom_table
    return (tuple(block[:len(table)].tobytes() for block in table._blocks),
            table.presence().tobytes(), table.sources().tobytes(),
            table._group_status[:len(table)].tobytes(),
            tuple(frozenset(seed) for seed in table._dirty_seeds))


@pytest.mark.parametrize("field", ["n", "s", "p", "d", "f", "g", "valence"])
@pytest.mark.parametrize("path", ["normalize_scalar", "normalize_mixed", "commit_scalar",
                                  "commit_mixed", "commit_candidate", "write_prepared"])
def test_private_native_ingress_rejects_derived_fields_atomically(field, path):
    from hotpot import Molecule
    from hotpot.cheminfo.structure._commit import _fact_engine
    from hotpot.cheminfo.structure._fields import FIELD_INDEX
    from hotpot.cheminfo.structure._identity import _capture_source
    from hotpot.cheminfo.structure._mutation import _normalize_atom_fields, _normalize_atom_scalar

    mol = Molecule()
    atom = mol.create_atom(atomic_number=6, implicit_hydrogens=4)
    table = mol._atom_table
    source = _capture_source(mol)
    engine = _fact_engine()
    producer = ATOM_SOURCE_CODES["explicit"]
    value = _UnconvertedDerivedValue()
    table._dirty_seeds[0].add(atom._uid)
    before = _stored_atom_state(mol)
    with pytest.raises(AttributeError, match="Calculated electron and valence"):
        if path == "normalize_scalar":
            _normalize_atom_scalar(source, 0, atom._uid, field, value)
        elif path == "normalize_mixed":
            _normalize_atom_fields(source, (0,), (atom._uid,), {"x": value, field: value})
        elif path == "commit_scalar":
            engine._commit_atom_scalar(atom, FIELD_INDEX[field], value, producer=producer)
        elif path == "commit_mixed":
            engine._commit_atom_fields(atom, {"x": value, field: value}, producer=producer)
        elif path == "commit_candidate":
            update = _normalize_atom_fields(source, (0,), (atom._uid,), {"x": 9, "y": 7})
            object.__setattr__(update, "fields", ("x", field))
            engine._commit_atom_facts(mol, update)
        else:
            engine._write_prepared(table, 0, [(FIELD_INDEX["x"], 9, True, producer),
                                              (FIELD_INDEX[field], value, True, producer)])
    assert _stored_atom_state(mol) == before


def test_private_native_ingress_preserves_internal_aromatic_commits():
    from hotpot import Molecule
    from hotpot.cheminfo.structure._commit import _commit_atom_facts, _fact_engine
    from hotpot.cheminfo.structure._dependencies import GROUP_INDEX, _FactStatus
    from hotpot.cheminfo.structure._fields import FIELD_INDEX, FIELD_PRESENCE_BITS
    from hotpot.cheminfo.structure._identity import _capture_source
    from hotpot.cheminfo.structure._mutation import _normalize_atom_fields, _normalize_atom_scalar

    mol = Molecule()
    atom = mol.create_atom(atomic_number=6, implicit_hydrogens=4)
    source = _capture_source(mol)
    producer = ATOM_SOURCE_CODES["explicit"]
    for update in (_normalize_atom_scalar(source, 0, atom._uid, "is_aromatic", True),
                   _normalize_atom_fields(source, (0,), (atom._uid,), {"x": 9, "is_aromatic": True})):
        _commit_atom_facts(mol, update)
    engine = _fact_engine()
    engine._commit_atom_scalar(atom, FIELD_INDEX["is_aromatic"], False, producer=producer)
    engine._commit_atom_fields(atom, {"x": 8, "is_aromatic": True}, producer=producer)
    table = mol._atom_table
    index = FIELD_INDEX["is_aromatic"]
    assert table.raw(0, "is_aromatic") == 1
    assert int(table.presence()[0]) & FIELD_PRESENCE_BITS[index]
    assert table.sources()[0, index] == producer
    group = GROUP_INDEX["AROMATIC_REPRESENTATION"]
    assert table._group_status[0, group] == _FactStatus.CURRENT
    assert atom._uid not in table._dirty_seeds[group]


def test_derived_field_guard_preserves_complete_aromatic_assignment():
    import math

    from hotpot import Molecule
    from hotpot.cheminfo.operations import chemistry
    from hotpot.cheminfo.rules.chemistry.bonding.kekule import _calculate_aromatic_assignment
    from hotpot.cheminfo.structure._commit import _commit_aromatic_assignment
    from hotpot.cheminfo.structure._dependencies import GROUP_INDEX, _FactStatus

    mol = Molecule()
    for row in range(6):
        angle = row * math.pi / 3
        mol.create_atom(atomic_number=6, coordinates=(math.cos(angle), math.sin(angle), 0.))
    for row in range(6):
        mol.add_bond(row, (row + 1) % 6, 1.)
    snapshot = chemistry._bonding_snapshot(mol)
    assignment = _calculate_aromatic_assignment(snapshot, kekule=True)
    _commit_aromatic_assignment(mol, assignment, snapshot=snapshot)
    assert mol._atom_table.column("is_aromatic").tolist() == [1.] * 6
    assert tuple(bond._is_aromatic for bond in mol.bonds) == (True,) * 6
    assert tuple(bond.bond_order for bond in mol.bonds) == assignment.bond_orders
    assert (mol._atom_table._group_status[:6, GROUP_INDEX["AROMATIC_REPRESENTATION"]]
            == _FactStatus.CURRENT).all()
