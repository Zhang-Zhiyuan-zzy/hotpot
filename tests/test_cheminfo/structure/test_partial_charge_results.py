"""One selected charge owner, finite applicability and native publication."""

from dataclasses import FrozenInstanceError, fields, replace
from copy import deepcopy
import pickle

import numpy as np
import pytest

from hotpot import Atom, Molecule
from hotpot.cheminfo.structure import _identity, _results
from hotpot.cheminfo.structure._commit import (
    _FactTransaction, _commit_atom_facts, _fact_engine, _native_fail_after,
)
from hotpot.cheminfo.structure._dependencies import GROUP_INDEX, _FactStatus, _dependency_state
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from hotpot.cheminfo.structure._mutation import _normalize_atom_fields
from hotpot.cheminfo.structure._results import (
    PartialChargeSet, _ResultAttachment, _capture_mca_source,
    _capture_partial_charge_source, _current_partial_charge_result,
    _import_partial_charges, _partial_charge_projection, _select_partial_charge_result,
)


def molecule():
    mol = Molecule()
    Atom(mol, atomic_number=6, formal_charge=0, implicit_hydrogens=0, coordinates=(0.0, 0.0, 0.0))
    Atom(mol, atomic_number=8, formal_charge=0, implicit_hydrogens=0, coordinates=(1.4, 0.0, 0.0))
    mol.add_bond(0, 1)
    _ = mol.atom_matrix
    return mol


def select(mol, values=(0.0, -0.25)):
    return _import_partial_charges(mol, values, producer=ATOM_SOURCE_CODES['rdkit'])


def test_absence_zero_and_incomplete_selection_are_distinct_with_no_scalar_store():
    mol = molecule()
    assert tuple(atom.partial_charge for atom in mol.atoms) == (None, None)
    assert np.isnan(mol.atom_matrix[:, FIELD_INDEX['partial_charge']]).all()
    assert mol._atom_table._columns[FIELD_INDEX['partial_charge']] is None
    payload = select(mol, (0.0, None))
    assert tuple(atom.partial_charge for atom in mol.atoms) == (0.0, None)
    assert _current_partial_charge_result(mol) is payload
    assert _partial_charge_projection(mol) == (0.0, None)
    assert mol.atom_matrix[0, 8] == 0.0 and np.isnan(mol.atom_matrix[1, 8])
    assert not (mol._atom_table.presence() & (1 << 8)).any()
    assert (mol._atom_table.sources()[:, 8] == ATOM_SOURCE_CODES['default']).all()
    assert all('partial_charge' not in atom.__dict__ for atom in mol.atoms)
    assert payload.method == 'supplied' and payload.method_version is None
    assert payload.units == 'e' and payload.provenance == (ATOM_SOURCE_CODES['rdkit'],) * 2


def test_payload_has_exactly_the_seven_immutable_fields_and_isolated_inputs():
    mol = molecule()
    uids, values, parameters, provenance = [0, 1], [0.0, None], [['model_args', None]], [3, 0]
    payload = PartialChargeSet(uids, values, 'gasteiger', 'test-version', parameters, 'e', provenance)
    assert tuple(field.name for field in fields(payload)) == (
        'atom_uids', 'values', 'method', 'method_version', 'parameters', 'units', 'provenance',
    )
    uids[0], values[0], parameters[0][0], provenance[0] = 9, 3.0, 'changed', 7
    assert payload.atom_uids == (0, 1) and payload.values == (0.0, None)
    assert payload.parameters == (('model_args', None),) and payload.provenance == (3, 0)
    for copied in (deepcopy(payload), pickle.loads(pickle.dumps(payload))):
        assert copied == payload
        with pytest.raises(FrozenInstanceError):
            copied.values = ()
    with pytest.raises(ValueError, match='complete unique Atom'):
        replace(payload, atom_uids=(0, 0))
    with pytest.raises(ValueError, match='complete unique Atom'):
        replace(payload, values=(0.0,))


def test_scalar_reads_never_hash_check_calculate_or_build_whole_row_mapping(monkeypatch):
    mol = molecule()
    select(mol)

    def forbidden(*args, **kwargs):
        pytest.fail('A scalar selected-charge read entered an operation or whole-source check')

    for module in (_identity, _results):
        monkeypatch.setattr(module, '_capture_source', forbidden)
        monkeypatch.setattr(module, '_check_source', forbidden)
    monkeypatch.setattr(_results, '_current_partial_charge_result', forbidden)
    monkeypatch.setattr(Molecule, 'get_partial_charge', forbidden)
    monkeypatch.setattr(Molecule, 'assign_partial_charge', forbidden)
    for _ in range(20):
        assert tuple(atom.partial_charge for atom in mol.atoms) == (0.0, -0.25)


@pytest.mark.parametrize('path', ['scalar', 'fields', 'columns', 'prepared'])
def test_each_native_publication_path_marks_geometry_stale(path):
    mol = molecule()
    payload = select(mol)
    engine = _fact_engine()
    if path == 'scalar':
        mol.atoms[0].x = 2.0
    elif path == 'fields':
        engine._commit_atom_fields(mol.atoms[0], {'x': 2.0, 'id': 42}, producer=1)
    elif path == 'columns':
        mol.updata_atoms(x=(2.0, 1.4))
    else:
        update = _normalize_atom_fields(_capture_source(mol), (0,), (mol.atoms[0]._uid,), {'x': 2.0})
        _commit_atom_facts(mol, update)
    assert mol._partial_charge_result.payload is payload
    assert mol._partial_charge_result.invalidated_by == ('geometry',)
    with pytest.raises(RuntimeError, match=r'geometry.*assign_partial_charge'):
        _ = mol.atoms[0].partial_charge
    assert _partial_charge_projection(mol) == (None, None)
    assert np.isnan(mol.atom_matrix[:, 8]).all()
    assert payload.values == (0.0, -0.25)


@pytest.mark.parametrize('change', ['value', 'signed_zero', 'presence', 'producer'])
def test_native_dependency_change_detects_bits_presence_and_producer(change):
    mol = molecule()
    select(mol)
    original = mol._partial_charge_result
    if change == 'signed_zero':
        mol.atoms[0].x = -0.0
    else:
        field = 'formal_charge'
        update = _normalize_atom_fields(_capture_source(mol), (0,), (mol.atoms[0]._uid,),
            {field: 1.0 if change == 'value' else 0.0},
            present=change != 'presence', producer=3 if change == 'producer' else 1)
        _commit_atom_facts(mol, update)
    assert mol._partial_charge_result is not original
    assert mol._partial_charge_result.invalidated_by == (
        ('geometry',) if change == 'signed_zero' else ('charge_representation',))


@pytest.mark.parametrize('change', ['order', 'metadata'])
def test_bond_signed_zero_and_nested_metadata_changes_invalidate_charge_without_mca(change):
    mol = molecule()
    bond = mol.bonds[0]
    bond.bond_order = 0.0
    bond._set_bond_metadata(bond_kind=bond.bond_kind, bond_source_metadata={'nested': {'value': 0.0}})
    payload = select(mol)
    assert mol._mca_result is None
    if change == 'order':
        bond.bond_order = -0.0
    else:
        bond._set_bond_metadata(bond_kind=bond.bond_kind,
                                bond_source_metadata={'nested': {'value': -0.0}})
    assert mol._partial_charge_result.payload is payload
    assert mol._partial_charge_result.invalidated_by == ('charge_representation',)


@pytest.mark.parametrize('outer', [False, True])
@pytest.mark.parametrize('boundary', ['value', 'source', 'presence', 'status', 'result_invalidation'])
def test_native_failure_restores_exact_charge_attachment_and_facts(outer, boundary):
    mol = molecule()
    field = 'implicit_hydrogens' if boundary == 'status' else 'x'
    if boundary == 'status':
        # Upstream writes enqueue invalidation; a materialized H write performs
        # the actual status store from NEEDS_RECALC to CURRENT.
        group = GROUP_INDEX['IMPLICIT_H']
        mol._atom_table._group_status[0, group] = _FactStatus.NEEDS_RECALC
        mol._atom_table._dirty_seeds[group].add(mol.atoms[0]._uid)
    select(mol)
    before = mol._partial_charge_result
    raw = tuple(block[:len(mol.atoms)].tobytes() for block in mol._atom_table._blocks)
    metadata = mol._atom_table.presence().tobytes(), mol._atom_table.sources().tobytes()
    statuses, seeds = _dependency_state(mol._atom_table)

    def write():
        with _native_fail_after(boundary):
            _fact_engine()._commit_atom_scalar(mol.atoms[0], FIELD_INDEX[field], 2.0, producer=3)

    with pytest.raises(KeyboardInterrupt, match='native fact store'):
        if outer:
            with _FactTransaction(mol, producer=1):
                write()
        else:
            write()
    assert mol._partial_charge_result is before
    assert tuple(block[:len(mol.atoms)].tobytes() for block in mol._atom_table._blocks) == raw
    assert (mol._atom_table.presence().tobytes(), mol._atom_table.sources().tobytes()) == metadata
    actual_statuses, actual_seeds = _dependency_state(mol._atom_table)
    np.testing.assert_array_equal(actual_statuses, statuses)
    assert actual_seeds == seeds
    assert mol.atoms[0].partial_charge == 0.0


def test_invalidation_reselection_and_outer_rollback_restore_original_exact_slot():
    mol = molecule()
    select(mol)
    original = mol._partial_charge_result
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1):
            mol.atoms[0].x = 2.0
            select(mol, (0.5, -0.5))
            assert mol.atoms[0].partial_charge == 0.5
            raise KeyboardInterrupt('abort after charge reselection')
    assert mol._partial_charge_result is original
    assert mol.atoms[0].x == 0.0 and mol.atoms[0].partial_charge == 0.0


def test_failed_complete_selection_and_nested_poison_restore_previous_selection():
    mol = molecule()
    payload = select(mol)
    before = mol._partial_charge_result
    bad = replace(payload, atom_uids=tuple(reversed(payload.atom_uids)))
    with pytest.raises(ValueError, match='complete ordered Atom'):
        _select_partial_charge_result(mol, bad, _capture_partial_charge_source(mol), payload.atom_uids)
    assert mol._partial_charge_result is before
    with pytest.raises(RuntimeError, match='nested factual'):
        with _FactTransaction(mol, producer=1):
            select(mol, (0.5, -0.5))
            try:
                _select_partial_charge_result(mol, bad, _capture_partial_charge_source(mol), payload.atom_uids)
            except ValueError:
                pass
    assert mol._partial_charge_result is before


def test_source_and_membership_drift_reject_complete_selection():
    mol = molecule()
    payload = select(mol)
    source = _capture_partial_charge_source(mol)
    mol.atoms[0].x = 2.0
    before = mol._partial_charge_result
    with pytest.raises(RuntimeError, match='geometry'):
        _select_partial_charge_result(mol, payload, source, payload.atom_uids)
    assert mol._partial_charge_result is before
    source = _capture_partial_charge_source(mol)
    with pytest.raises(RuntimeError, match='membership'):
        _select_partial_charge_result(mol, payload, source, tuple(reversed(payload.atom_uids)))
    assert mol._partial_charge_result is before


def test_charge_selection_retains_mca_and_never_depends_on_its_own_selected_values():
    mol = molecule()
    charge_source = _capture_partial_charge_source(mol)
    mca = _ResultAttachment(object(), _capture_mca_source(mol))
    mol._mca_result = mca
    select(mol)
    _check_source(mol, charge_source)
    select(mol, (0.25, None))
    _check_source(mol, charge_source)
    assert mol._mca_result is mca
    assert tuple(name for name, _ in charge_source.dependencies) == ('charge_representation', 'geometry')


def test_foreign_owner_is_integrity_error_for_both_strict_and_optional_projection():
    mol, foreign = molecule(), molecule()
    payload = select(mol)
    mol._partial_charge_result = _ResultAttachment(payload, _capture_partial_charge_source(foreign), ('geometry',))
    for read in (lambda: mol.atoms[0].partial_charge,
                 lambda: _partial_charge_projection(mol),
                 lambda: _current_partial_charge_result(mol)):
        with pytest.raises(RuntimeError, match='owner'):
            read()
