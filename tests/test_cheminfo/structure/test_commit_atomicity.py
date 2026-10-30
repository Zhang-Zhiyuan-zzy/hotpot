"""Operation-local journal atomicity, source bindings and publication locks."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

from hotpot import Atom, Molecule
from hotpot.cheminfo.stereo import StereoPayload
from hotpot.cheminfo.structure._commit import (
    _native_fail_after, _FactTransaction, _commit_atom_facts, _commit_atom_sparse, _commit_bond_facts,
    _commit_molecular_declarations, _commit_stereo, _current_transaction,
)
from hotpot.cheminfo.structure._dependencies import _dependency_state
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from hotpot.cheminfo.structure._mutation import _normalize_atom_scalar


def _molecule():
    mol = Molecule()
    for number in (6, 8, 7):
        mol.create_atom(atomic_number=number)
    mol.add_bond(0, 1)
    return mol


def _update(mol, row, field, value, **kwargs):
    return _normalize_atom_scalar(_capture_source(mol), row, mol.atoms[row]._uid, field, value, **kwargs)


def test_nested_commits_restore_raw_bits_metadata_sparse_and_status_on_baseexception():
    mol = _molecule()
    table = mol._atom_table
    before = table.matrix().tobytes(), table.presence(), table.sources(), _dependency_state(table)
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=13) as outer:
            _commit_atom_facts(mol, _update(mol, 0, 'x', -0.0))
            with _FactTransaction(mol, producer=1) as inner:
                assert inner is outer
                _commit_atom_facts(mol, _update(mol, 0, 'x', 19.5))
                _commit_atom_facts(mol, _update(mol, 1, 'formal_charge', -1))
                _commit_atom_sparse(mol, mol.atoms[0], isotope=13, frac_valence=2.5)
            raise KeyboardInterrupt('after nested writes')
    assert table.matrix().tobytes() == before[0]
    np.testing.assert_array_equal(table.presence(), before[1])
    np.testing.assert_array_equal(table.sources(), before[2])
    after = _dependency_state(table)
    np.testing.assert_array_equal(after[0], before[3][0])
    assert after[1] == before[3][1]
    assert 'isotope' not in mol.atoms[0].__dict__
    assert 'frac_valence' not in mol.atoms[0].__dict__
    assert _current_transaction(mol) is None
    assert not outer._cells and not outer._seeds and outer._topology is None


def test_caught_nested_failure_still_prevents_outer_publication():
    mol = _molecule()
    before = mol.atom_matrix.tobytes()
    with pytest.raises(RuntimeError, match='nested factual'):
        with _FactTransaction(mol, producer=1):
            try:
                with _FactTransaction(mol, producer=1):
                    _commit_atom_facts(mol, _update(mol, 0, 'x', 99))
                    raise ValueError('caught by legacy caller')
            except ValueError:
                pass
    assert mol.atom_matrix.tobytes() == before


def test_operation_producer_is_inherited_but_named_computation_is_retained():
    mol = _molecule()
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['xtb']):
        _commit_atom_facts(mol, _update(mol, 0, 'x', 9))
        _commit_atom_facts(mol, _update(mol, 0, 'implicit_hydrogens', 3, producer=ATOM_SOURCE_CODES['implicit_hydrogens']))
    codes = mol._atom_table.sources(0)[0]
    assert codes[FIELD_INDEX['x']] == ATOM_SOURCE_CODES['xtb']
    assert codes[FIELD_INDEX['implicit_hydrogens']] == ATOM_SOURCE_CODES['implicit_hydrogens']


def test_scalar_raw_write_keeps_negative_zero_and_immutable_candidate():
    mol = _molecule()
    candidate = _update(mol, 0, 'x', -0.0)
    assert not candidate.values[0].flags.writeable
    _commit_atom_facts(mol, candidate)
    assert np.signbit(mol._atom_table.raw(0, 'x'))
    with pytest.raises(ValueError):
        candidate.values[0].flags.writeable = True


def test_source_and_uid_checks_precede_any_write():
    mol = _molecule()
    candidate = _update(mol, 0, 'x', 999)
    before = mol.atom_matrix.tobytes()
    foreign = replace(candidate, source=_capture_source(_molecule()))
    with pytest.raises(RuntimeError, match='owner'):
        _commit_atom_facts(mol, foreign)
    stale = replace(candidate, atom_uids=(1000,))
    with pytest.raises(ValueError, match='UID'):
        _commit_atom_facts(mol, stale)
    assert mol.atom_matrix.tobytes() == before


def _complete_state(mol):
    from tests.test_cheminfo._stage5_fixtures import factual_state
    status, seeds = _dependency_state(mol._atom_table)
    raw = factual_state(mol), status.tobytes(), seeds
    return raw, mol.atom_matrix.tobytes()


def test_eager_element_failure_restores_all_prior_slots(monkeypatch):
    mol = _molecule()
    mol.atom_matrix.tobytes()  # Settle lazy chemistry before the tested transaction.
    before = _complete_state(mol)
    def forbidden(atom):
        raise AssertionError('Element publication must not execute a valence rule')
    monkeypatch.setattr(Atom, 'get_valence', forbidden)
    with _native_fail_after('value'), pytest.raises(KeyboardInterrupt, match='native fact store'):
        mol.atoms[0].atomic_number = 7
    assert _complete_state(mol) == before


def test_bond_endpoint_failure_rolls_back_bond_and_prior_endpoint(monkeypatch):
    mol = _molecule()
    mol.atom_matrix.tobytes()  # Settle lazy chemistry before the tested transaction.
    before = _complete_state(mol)
    bond = mol.bonds[0]
    before_bond = bond.attrs.tobytes(), bond.bond_kind
    def forbidden(atom):
        raise AssertionError('Bond publication must not execute an endpoint H rule')
    monkeypatch.setattr(Atom, '_calc_implicit_hydrogens', forbidden)
    # The second real seed store fails after the Bond and a prior seed changed.
    with _native_fail_after('seed_add', occurrence=2), pytest.raises(KeyboardInterrupt, match='native fact store'):
        mol.bonds[0].bond_order = 2
    assert mol.bonds[0] is bond
    assert bond.attrs.tobytes() == before_bond[0]
    assert bond.bond_kind is before_bond[1]
    assert _complete_state(mol) == before


def test_cross_domain_topology_undo_restores_handles_endpoints_frames_and_declarations():
    mol = _molecule()
    mol.conformer_add(mol.coordinates, energy=3.0)
    original_atoms, original_bonds = tuple(mol.atoms), tuple(mol.bonds)
    index = mol._topology_index
    table = mol._atom_table
    before = mol.atom_matrix.tobytes()
    with pytest.raises(RuntimeError, match='publication failure'):
        with _FactTransaction(mol, producer=1) as transaction:
            transaction.touch_topology()
            mol.remove_atom(mol.atoms[0])
            _commit_molecular_declarations(mol, _capture_source(mol), charge=(True, 5, 'test'))
            _commit_stereo(mol, _capture_source(mol), StereoPayload(1, (), (), ()))
            raise RuntimeError('publication failure')
    assert tuple(mol.atoms) == original_atoms
    assert tuple(mol.bonds) == original_bonds
    assert mol._topology_index is index and mol._atom_table is table
    assert mol.atom_matrix.tobytes() == before
    assert mol.bonds[0].atom1 is original_atoms[0]
    assert mol.bonds[0].atom2 is original_atoms[1]
    assert '_detached_table' not in original_atoms[0].__dict__
    assert len(mol.conformers) == 1
    assert mol.conformers[0]['energy'] == 3.0
    assert mol.charge == 0


def test_reader_and_source_capture_wait_for_outer_publication():
    mol = _molecule()
    attempted, finished = Event(), Event()
    def read():
        attempted.set()
        values = mol.atom_matrix
        source = _capture_source(mol, ('geometry',))
        finished.set()
        return values, source
    with ThreadPoolExecutor(max_workers=1) as pool:
        with _FactTransaction(mol, producer=1):
            _commit_atom_facts(mol, _update(mol, 0, 'x', 3))
            future = pool.submit(read)
            assert attempted.wait(2)
            assert not finished.wait(0.05)
            _commit_atom_facts(mol, _update(mol, 1, 'x', 5))
        values, source = future.result(timeout=2)
    assert values[:2, FIELD_INDEX['x']].tolist() == [3, 5]
    _check_source(mol, source)


def test_sealed_stereo_identity_is_preserved_and_mutable_sequences_are_isolated():
    mol = _molecule()
    sealed = StereoPayload(1, (), (), ())
    _commit_stereo(mol, _capture_source(mol), sealed)
    assert mol._stereo_payload is sealed
    mutable = StereoPayload(1, [], [], [])
    _commit_stereo(mol, _capture_source(mol), mutable)
    assert mol._stereo_payload is not mutable
    assert mol._stereo_payload.records == ()
    mutable.records.append('outside')
    assert mol._stereo_payload.records == ()


@pytest.mark.parametrize('topology_edit', ['grow', 'remove', 'remove_then_grow'])
def test_overlapping_bulk_scalar_undo_targets_original_buffers_after_topology_edit(topology_edit):
    mol = _molecule()
    table = mol._atom_table
    atoms = tuple(mol.atoms)
    before = table.matrix().tobytes(), table.presence(), table.sources(), _dependency_state(table)
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=13):
            mol.updata_atoms(formal_charge=-1, coordinates=(1., 2., 3.))
            atoms[1].x = -0.0
            if topology_edit == 'grow':
                # Force a physical-buffer replacement in the active table.
                for _ in range(len(table._blocks[0]) + 1):
                    mol.create_atom(atomic_number=6)
            else:
                mol.remove_atom(atoms[0])
                if topology_edit == 'remove_then_grow':
                    mol.create_atom(atomic_number=6)
            mol.updata_atoms(formal_charge=2, coordinates=(9., 8., 7.))
            atoms[1].formal_charge = 3
            raise KeyboardInterrupt('after overlapping writes and changed storage')
    assert tuple(mol.atoms) == atoms
    assert mol._atom_table is table
    assert table.matrix().tobytes() == before[0]
    np.testing.assert_array_equal(table.presence(), before[1])
    np.testing.assert_array_equal(table.sources(), before[2])
    after = _dependency_state(table)
    np.testing.assert_array_equal(after[0], before[3][0])
    assert after[1] == before[3][1]


@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('target', ['value', 'presence', 'source', 'status'])
def test_commit_restores_write_that_raises_after_mutation(batch, target):
    mol = _molecule()
    table = mol._atom_table
    field = 'implicit_hydrogens' if target == 'status' else 'formal_charge'
    value = 2 if target == 'status' else -2
    if target == 'status':
        table._group_status[:, 1] = 1
    before = table.matrix().tobytes(), table.presence(), table.sources(), _dependency_state(table)
    with _native_fail_after(target), pytest.raises(KeyboardInterrupt, match='native fact store'):
        if batch:
            mol.updata_atoms(**{field: value})
        else:
            setattr(mol.atoms[0], field, value)
    assert table.matrix().tobytes() == before[0]
    np.testing.assert_array_equal(table.presence(), before[1])
    np.testing.assert_array_equal(table.sources(), before[2])
    after = _dependency_state(table)
    np.testing.assert_array_equal(after[0], before[3][0])
    assert after[1] == before[3][1]
