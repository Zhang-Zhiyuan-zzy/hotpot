"""Selected-result currentness at the native physical publication boundary."""

from contextlib import nullcontext
import sys

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.operations.inference.mca.results import (
    AtomPrediction, MoleculePrediction, SitePrediction,
)
from hotpot.cheminfo.structure._atom_table import _AtomFactTable
from hotpot.cheminfo.structure._commit import (
    _FactTransaction, _fact_engine, _native_fail_after,
)
from hotpot.cheminfo.structure._dependencies import _dependency_state
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX
from hotpot.cheminfo.structure._identity import _capture_source
from hotpot.cheminfo.structure._mutation import _normalize_atom_scalar
from hotpot.cheminfo.structure._results import (
    _capture_mca_source, _import_partial_charges, _prepare_atom_result_updates,
    _select_mca_result,
)


_EXPLICIT = ATOM_SOURCE_CODES['explicit']
_INGRESSES = ('dispatch', 'scalar', 'fields', 'columns', 'prepared')


def _molecule():
    mol = Molecule()
    for number in (6, 8):
        atom = mol.create_atom(atomic_number=number)
        _fact_engine()._commit_atom_fields(
            atom, {'x': 0.0, 'y': 0.0, 'z': 0.0, 'formal_charge': 0.0,
                   'implicit_hydrogens': 0.0, 'is_aromatic': 0.0},
            producer=_EXPLICIT,
        )
    return mol


def _select(mol, value=10.0):
    prediction = MoleculePrediction(
        'C.O', 0, (SitePrediction(1, 'O', 'fixture', value),), 'fixture',
        tuple(AtomPrediction(row, atom.symbol, value + row)
              for row, atom in enumerate(mol.atoms)),
    )
    _select_mca_result(mol, prediction, _capture_mca_source(mol),
                       tuple(atom._uid for atom in mol.atoms))
    return mol._mca_result


def _write(mol, ingress, field, value, *, producer=_EXPLICIT, present=True):
    atom = mol.atoms[0]
    engine = _fact_engine()
    if ingress == 'dispatch':
        setattr(atom, field, value)
    elif ingress == 'scalar':
        engine._commit_atom_scalar(atom, FIELD_INDEX[field], value,
                                   producer=producer, present=present)
    elif ingress == 'fields':
        engine._commit_atom_fields(atom, {field: value}, producer=producer, present=present)
    elif ingress == 'columns':
        with _FactTransaction(mol, producer=producer):
            engine._commit_atom_columns(mol, (atom,), {field: [value]})
    elif ingress == 'prepared':
        update = _normalize_atom_scalar(_capture_source(mol), 0, atom._uid, field, value,
                                        producer=producer, present=present)
        engine._commit_atom_facts(mol, update)


def _state(mol):
    table = mol._atom_table
    status, seeds = _dependency_state(table)
    return (tuple(block[:len(table)].tobytes() for block in table._blocks),
            table.presence().tobytes(), table.sources().tobytes(), status.tobytes(), seeds,
            tuple(sorted(mol._constraints.atom_axes.items())))


@pytest.mark.parametrize('ingress', _INGRESSES)
@pytest.mark.parametrize('field,value,domain', [('x', 2.5, 'geometry'),
                                               ('formal_charge', -1.0, 'mca_representation')])
def test_all_native_molecular_ingresses_invalidate_the_selected_attachment(ingress, field, value, domain):
    mol = _molecule()
    before = _select(mol)
    historical = before.payload.to_dict()
    _write(mol, ingress, field, value)
    after = mol._mca_result
    assert after is not before
    assert after.payload is before.payload and after.source is before.source
    assert after.invalidated_by == (domain,)
    assert before.invalidated_by == () and before.payload.to_dict() == historical
    with pytest.raises(RuntimeError, match=domain):
        _ = mol.atoms[0].mca
    with pytest.raises(RuntimeError, match=domain):
        _ = mol.mca_sites


@pytest.mark.parametrize('field,value,domain', [
    ('atomic_number', 7.0, 'mca_representation'), ('formal_charge', -1.0, 'mca_representation'),
    ('implicit_hydrogens', 2.0, 'mca_representation'), ('is_aromatic', 1.0, 'mca_representation'),
    ('x', 1.0, 'geometry'), ('y', 1.0, 'geometry'), ('z', 1.0, 'geometry'),
])
def test_each_subscribed_numeric_field_reaches_the_common_native_hook(field, value, domain):
    mol = _molecule()
    _select(mol)
    _write(mol, 'fields', field, value)
    assert mol._mca_result.invalidated_by == (domain,)


@pytest.mark.parametrize('ingress', ('scalar', 'fields', 'prepared'))
@pytest.mark.parametrize('change', ('value_bits', 'presence', 'source'))
def test_native_comparison_includes_signed_zero_presence_and_producer(ingress, change):
    mol = _molecule()
    _select(mol)
    _write(mol, ingress, 'x', -0.0 if change == 'value_bits' else 0.0,
           present=change != 'presence',
           producer=ATOM_SOURCE_CODES['rdkit'] if change == 'source' else _EXPLICIT)
    assert mol._mca_result.invalidated_by == ('geometry',)
    if change == 'value_bits':
        assert np.signbit(mol._atom_table.raw(0, 'x'))


@pytest.mark.parametrize('ingress', _INGRESSES)
@pytest.mark.parametrize('selected', (False, True))
def test_noop_or_absent_selection_skips_native_result_preparation(ingress, selected):
    mol = _molecule()
    before = _select(mol) if selected else None
    calls = []
    code = _prepare_atom_result_updates.__code__
    previous_profile = sys.getprofile()

    def profile(frame, event, argument):
        if event == 'call' and frame.f_code is code:
            calls.append(frame.f_locals['changed_fields'])

    sys.setprofile(profile)
    try:
        # Equal bits, producer and presence with a selection; an actual write
        # without one. Neither path should enter Python result preparation.
        with _native_fail_after('result_invalidation'):
            _write(mol, ingress, 'x', 0.0 if selected else 3.0)
    finally:
        sys.setprofile(previous_profile)
    assert calls == []
    assert mol._mca_result is before
    if selected:
        assert mol.atoms[0].mca == 10.0


def test_bulk_preparation_reports_only_fields_with_an_actual_change_in_any_selected_row():
    mol = _molecule()
    _select(mol)
    calls = []
    code = _prepare_atom_result_updates.__code__
    previous_profile = sys.getprofile()

    def profile(frame, event, argument):
        if event == 'call' and frame.f_code is code:
            calls.append(tuple(frame.f_locals['changed_fields']))

    sys.setprofile(profile)
    try:
        with _FactTransaction(mol, producer=_EXPLICIT):
            _fact_engine()._commit_atom_columns(
                mol, None, {'formal_charge': [0.0, 0.0], 'x': [0.0, 2.0]},
            )
    finally:
        sys.setprofile(previous_profile)
    assert calls == [('x',)]
    assert mol._mca_result.invalidated_by == ('geometry',)


def test_equal_absent_value_bits_and_named_producer_are_a_true_native_noop():
    mol = _molecule()
    producer = ATOM_SOURCE_CODES['rdkit']
    _write(mol, 'fields', 'x', -0.0, producer=producer, present=False)
    attachment = _select(mol)
    with _native_fail_after('result_invalidation'):
        _write(mol, 'prepared', 'x', -0.0, producer=producer, present=False)
    assert mol._mca_result is attachment


@pytest.mark.parametrize('field,value', [('id', 42), ('x_constraint', 1)])
def test_unsubscribed_numeric_and_constraint_writes_retain_selected_result(field, value):
    mol = _molecule()
    before = _select(mol)
    with _native_fail_after('result_invalidation'):
        _write(mol, 'fields', field, value)
    assert mol._mca_result is before
    assert mol.atoms[0].mca == 10.0


@pytest.mark.parametrize('ingress', _INGRESSES)
def test_charge_write_rejection_and_complete_selection_retain_exact_mca_result(ingress):
    mol = _molecule()
    before = _select(mol)
    facts = _state(mol)
    with _native_fail_after('result_invalidation'):
        with pytest.raises(AttributeError):
            _write(mol, ingress, 'partial_charge', 0.5)
    assert _state(mol) == facts
    assert mol._partial_charge_result is None
    assert mol._mca_result is before

    with _native_fail_after('result_invalidation'):
        _import_partial_charges(mol, (0.5, None), producer=_EXPLICIT)
    assert _state(mol) == facts
    assert mol._partial_charge_result.payload.values == (0.5, None)
    assert mol._partial_charge_result.invalidated_by == ()
    assert tuple(atom.partial_charge for atom in mol.atoms) == (0.5, None)
    assert mol._mca_result is before
    assert mol.atoms[0].mca == 10.0


def test_native_invalidation_accumulates_only_new_subscribed_domains():
    mol = _molecule()
    _select(mol)
    _write(mol, 'scalar', 'x', 1.0)
    geometry = mol._mca_result
    with _native_fail_after('result_invalidation'):
        _write(mol, 'fields', 'z', 2.0)
    assert mol._mca_result is geometry
    _write(mol, 'prepared', 'formal_charge', -1.0)
    assert set(mol._mca_result.invalidated_by) == {'geometry', 'mca_representation'}


@pytest.mark.parametrize('ingress,outer', [(name, outer) for name in ('dispatch', 'scalar', 'fields', 'prepared')
                                          for outer in (False, True)] + [('columns', True)])
@pytest.mark.parametrize('exception', (KeyboardInterrupt, MemoryError))
def test_native_result_publication_failure_restores_exact_selection_and_all_fact_state(ingress, outer, exception):
    mol = _molecule()
    attachment = _select(mol)
    before = _state(mol)
    with pytest.raises(exception, match='injected after native fact store') as caught:
        with (_FactTransaction(mol, producer=_EXPLICIT) if outer else nullcontext()):
            with _native_fail_after('result_invalidation', exception=exception):
                _write(mol, ingress, 'x', 6.0)
    assert type(caught.value) is exception
    assert _state(mol) == before
    assert mol._mca_result is attachment
    assert mol.atoms[0].mca == 10.0


@pytest.mark.parametrize('boundary', ('enrollment', 'value', 'source', 'presence',
                                      'status', 'seed_add', 'seed_remove', 'result_invalidation'))
def test_existing_failure_scopes_keep_the_attachment_when_native_transaction_is_poisoned(boundary):
    mol = _molecule()
    table = mol._atom_table
    for seeds in table._dirty_seeds:
        seeds.clear()
    table._group_status[:, 1] = 1
    field, value = ('implicit_hydrogens', 3.0) if boundary in ('status', 'seed_remove') else ('x', 6.0)
    if boundary == 'seed_remove':
        table._dirty_seeds[1].add(mol.atoms[0]._uid)
    attachment = _select(mol)
    before = _state(mol)
    with pytest.raises(RuntimeError, match='nested factual'):
        with _FactTransaction(mol, producer=_EXPLICIT):
            try:
                with _native_fail_after(boundary):
                    _write(mol, 'scalar', field, value)
            except KeyboardInterrupt:
                pass
    assert _state(mol) == before
    assert mol._mca_result is attachment


def test_native_invalidation_then_python_selection_rolls_back_to_original_attachment():
    mol = _molecule()
    attachment = _select(mol)
    before = _state(mol)
    original = MemoryError('after native invalidation and replacement selection')
    with pytest.raises(MemoryError) as caught:
        with _FactTransaction(mol, producer=_EXPLICIT):
            _write(mol, 'scalar', 'x', 8.0)
            assert mol._mca_result.invalidated_by == ('geometry',)
            selected = _select(mol, 30.0)
            assert selected is not attachment and selected.invalidated_by == ()
            assert mol.atoms[0].mca == 30.0
            raise original
    assert caught.value is original
    assert _state(mol) == before
    assert mol._mca_result is attachment
    assert mol.atoms[0].mca == 10.0


def test_private_scratch_table_has_no_molecular_result_or_dependency_hook():
    mol = _molecule()
    attachment = _select(mol)
    before = _state(mol)
    scratch = _AtomFactTable(np.zeros((2, 16)))
    with _native_fail_after('result_invalidation'):
        scratch.update(0, {'x': 9.0, 'formal_charge': -1.0})
    assert scratch.raw(0, 'x') == 9.0 and scratch.raw(0, 'formal_charge') == -1.0
    assert _state(mol) == before and mol._mca_result is attachment


@pytest.mark.parametrize('ingress', _INGRESSES)
def test_nonempty_ensemble_tuple_alone_reaches_native_result_preparation(ingress):
    from hotpot.cheminfo.operations.conformers.contracts import ConformerBatch
    from hotpot.cheminfo.structure.conformer import _publish_conformer_ensemble

    mol = _molecule()
    _publish_conformer_ensemble(
        mol, ConformerBatch(mol.coordinates[None], (0,)),
        ((1.0,), None, None, None, None, None, None, None), 0,
        input_source=_capture_source(mol), method='native-hook-fixture')
    mol._frame_result = None
    before = mol._conformer_results
    _write(mol, ingress, 'formal_charge', -1.0)
    assert mol._frame_result is None
    assert mol._conformer_results is not before
    assert mol._conformer_results[0].payload is before[0].payload
    assert mol._conformer_results[0].invalidated_by == ('frame_representation',)
