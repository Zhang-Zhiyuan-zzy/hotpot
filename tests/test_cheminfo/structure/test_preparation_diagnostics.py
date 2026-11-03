"""Observed preparation failures retain facts and report their affected rows."""

import math
import re
import warnings

import pytest

from hotpot import Molecule
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._dependencies import (
    GROUP_INDEX, _FactStatus, _dependency_state, _install_dependency_state,
)
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from tests.test_cheminfo._stage5_fixtures import factual_state, supplied_aromatic_flags


H_FIELD = FIELD_INDEX['implicit_hydrogens']
H_GROUP = GROUP_INDEX['IMPLICIT_H']


def _state(mol):
    status, seeds = _dependency_state(mol._atom_table)
    return factual_state(mol), status.tobytes(), seeds


def _present(mol, row):
    return bool(int(mol._atom_table._presence[row]) & FIELD_PRESENCE_BITS[H_FIELD])


def _missing_input(case):
    mol = Molecule()
    atom = mol.create_atom(id=91, **({} if case == 'site' else {'atomic_number': 6}))
    if case != 'site':
        mol.create_atom(id=17, **({} if case == 'neighbour' else {'atomic_number': 8}))
        kind = {'neighbour': 'single', 'kind': 'unknown', 'direction': 'dative'}[case]
        mol.add_bond(0, 1, 0. if case == 'kind' else 1., bond_kind=kind)
    return mol, atom


def _mixed_rows():
    mol = Molecule()
    mol.create_atom(atomic_number=6, id=91)
    mol.create_atom(id=17)
    return mol


def _command(mol, command):
    if command == 'atom':
        return mol.atoms[1].calc_implicit_hydrogens()
    if command == 'hydrogens':
        return mol.calc_implicit_hydrogens()
    return mol.calc_atom_valence()


@pytest.mark.parametrize('case,code,reason', [
    ('site', 'HYDROGEN.MISSING_SITE_ELEMENT', 'site element is missing'),
    ('neighbour', 'HYDROGEN.MISSING_NEIGHBOUR_ELEMENT', 'neighbour_atom_index=1 element is missing'),
    ('kind', 'HYDROGEN.UNKNOWN_BOND_KIND', 'bond_kind=unknown'),
    ('direction', 'HYDROGEN.MISSING_DATIVE_DIRECTION', 'dative direction is missing'),
])
def test_explicit_atom_diagnostic_names_actual_missing_input_and_matches_strict(case, code, reason):
    mol, atom = _missing_input(case)
    with pytest.warns(RuntimeWarning) as observed:
        result = atom.calc_implicit_hydrogens()
    assert result is None
    assert len(observed) == 1
    message = str(observed[0].message)
    assert re.findall(r'\bHYDROGEN\.[A-Z_]+\b', message) == [code]
    assert 'atom_index=0' in message
    assert '91' not in message
    assert reason in message.partition(': ')[2]
    if case != 'site':
        assert 'bond_index=0' in message
    assert mol._atom_table._group_status[0, H_GROUP] == _FactStatus.UNRESOLVED
    assert not _present(mol, 0)
    assert mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['default']
    with pytest.raises(ValueError) as failure:
        chemistry._require_implicit_hydrogens(mol, rows=(0,))
    assert str(failure.value) == message


@pytest.mark.parametrize('case', ['site', 'neighbour', 'kind', 'direction'])
def test_lazy_display_preparation_and_scientific_extraction_are_quiet(case):
    mol, atom = _missing_input(case)
    before = _state(mol)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        facts = extract_molecular_facts(mol)
        assert _state(mol) == before
        chemistry._prepare_hydrogens(mol)
        assert atom.implicit_hydrogens == 0
        assert atom.attrs[14] == 0
    assert observed == []
    assert facts.hydrogen_inventory.implicit_counts[0] is None
    assert not _present(mol, 0)


@pytest.mark.parametrize('command', ['atom', 'hydrogens', 'valence'])
def test_explicit_commands_keep_complete_rows_and_report_only_unavailable_rows(command):
    mol = _mixed_rows()
    calculate = {
        'atom': mol.atoms[1].calc_implicit_hydrogens,
        'hydrogens': mol.calc_implicit_hydrogens,
        'valence': mol.calc_atom_valence,
    }[command]
    with pytest.warns(RuntimeWarning) as observed:
        result = calculate()
    assert result is None
    assert len(observed) == 1
    assert observed[0].filename == __file__
    assert 'HYDROGEN.MISSING_SITE_ELEMENT' in str(observed[0].message)
    assert 'atom_index=1' in str(observed[0].message)
    assert 'atom_index=0' not in str(observed[0].message)
    assert mol._atom_table._group_status[:2, H_GROUP].tolist() == [
        _FactStatus.CURRENT, _FactStatus.UNRESOLVED]
    assert _present(mol, 0) and not _present(mol, 1)
    assert chemistry._require_implicit_hydrogens(mol, rows=(0,)) == (4,)
    assert mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['implicit_hydrogens']


def test_atom_calculation_does_not_report_unrequested_rows_prepared_as_dependencies():
    mol = _mixed_rows()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        assert mol.atoms[0].calc_implicit_hydrogens() is None
    assert observed == []
    assert mol._atom_table._group_status[:2, H_GROUP].tolist() == [
        _FactStatus.CURRENT, _FactStatus.UNRESOLVED]
    assert mol.atoms[0].implicit_hydrogens == 4
    assert not _present(mol, 1)


@pytest.mark.parametrize('command', ['atom', 'hydrogens', 'valence'])
def test_promoted_warning_rolls_back_the_complete_explicit_command(command):
    mol = _mixed_rows()
    before = _state(mol)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(RuntimeWarning, match=r'HYDROGEN\.MISSING_SITE_ELEMENT.*atom_index=1'):
            _command(mol, command)
    assert _state(mol) == before
    assert not _present(mol, 0)
    assert not _present(mol, 1)


@pytest.mark.parametrize('command', ['atom', 'hydrogens', 'valence'])
def test_explicit_preparation_joins_caller_transaction_and_restores_all_facts(command):
    mol = _mixed_rows()
    before = _state(mol)
    with pytest.raises(RuntimeError, match='caller rollback'):
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['explicit']):
            mol.atoms[0].atomic_number = 8
            with pytest.warns(RuntimeWarning, match=r'HYDROGEN\.MISSING_SITE_ELEMENT'):
                _command(mol, command)
            assert chemistry._require_implicit_hydrogens(mol, rows=(0,)) == (2,)
            raise RuntimeError('caller rollback')
    assert _state(mol) == before


def test_caught_promoted_warning_still_poisoned_the_joined_caller_transaction():
    mol = _mixed_rows()
    before = _state(mol)
    with pytest.raises(RuntimeError, match='nested factual transaction failed'):
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['explicit']):
            mol.atoms[0].atomic_number = 8
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                with pytest.raises(RuntimeWarning, match=r'HYDROGEN\.MISSING_SITE_ELEMENT'):
                    mol.calc_implicit_hydrogens()
    assert _state(mol) == before


@pytest.mark.parametrize('command', ['atom', 'hydrogens', 'valence'])
def test_explicit_known_zero_remains_authoritative_and_silent(command):
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol.create_atom(implicit_hydrogens=0)
    source = mol._atom_table._sources[1, H_FIELD]
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        assert _command(mol, command) is None
    assert observed == []
    assert chemistry._require_implicit_hydrogens(mol, rows=(1,)) == (0,)
    assert _present(mol, 1)
    assert mol._atom_table._sources[1, H_FIELD] == source == ATOM_SOURCE_CODES['explicit']


def _noncyclic_source():
    mol = Molecule()
    mol.create_atom(atomic_number=6, id=91)
    mol.create_atom(atomic_number=6, id=17)
    mol.add_bond(0, 1)
    return supplied_aromatic_flags(mol, (True, False))


@pytest.mark.parametrize('consumer', ['atom', 'bond', 'strict'])
def test_aromatic_failure_names_affected_row_and_noncyclic_source_without_overwriting_it(consumer):
    mol = _noncyclic_source()
    before = factual_state(mol)
    with pytest.raises(RuntimeError) as failure:
        if consumer == 'atom':
            _ = mol.atoms[1].is_aromatic
        elif consumer == 'bond':
            _ = mol.bonds[0].is_aromatic
        else:
            chemistry._require_aromatic_representation(mol, rows=(1,))
    message = str(failure.value)
    assert 'AROMATIC.NONCYCLIC_SOURCE' in message
    affected = (0, 1) if consumer == 'bond' else (1,)
    assert tuple(map(int, re.findall(r'AROMATIC\.NONCYCLIC_SOURCE atom_index=(\d+)', message))) == affected
    assert 'source_atom_indices=(0,)' in message
    assert 'noncyclic' in message.lower()
    assert factual_state(mol) == before
    assert mol._atom_table.column('is_aromatic').tolist() == [1., 0.]
    assert mol._atom_table._sources[0, FIELD_INDEX['is_aromatic']] == ATOM_SOURCE_CODES['explicit']
    assert mol._atom_table._group_status[:2, GROUP_INDEX['AROMATIC_REPRESENTATION']].tolist() == [
        _FactStatus.CONFLICT, _FactStatus.CONFLICT]


def test_hydrogen_representation_conflict_warning_and_strict_failure_agree():
    mol = _noncyclic_source()
    with pytest.warns(RuntimeWarning) as observed:
        assert mol.atoms[1].calc_implicit_hydrogens() is None
    assert len(observed) == 1
    message = str(observed[0].message)
    assert 'HYDROGEN.REPRESENTATION_CONFLICT' in message
    assert 'atom_index=1' in message
    assert 'AROMATIC.NONCYCLIC_SOURCE' in message
    assert re.search(r'source[^;]*\b0\b', message)
    with pytest.raises(RuntimeError) as failure:
        chemistry._require_implicit_hydrogens(mol, rows=(1,))
    assert str(failure.value) == message
    assert mol._atom_table.column('is_aromatic').tolist() == [1., 0.]
    assert not _present(mol, 1)


def _benzene():
    mol = Molecule()
    for row in range(6):
        angle = row * math.pi / 3
        mol.create_atom(atomic_number=6, coordinates=(math.cos(angle), math.sin(angle), 0.))
    for row in range(6):
        mol.add_bond(row, (row + 1) % 6, 2. if row % 2 == 0 else 1.)
    return supplied_aromatic_flags(mol, (True,) * 6)


def test_equivalent_kekule_projection_is_valid_and_success_remains_silent():
    mol = _benzene()
    source_flags = mol._atom_table.column('is_aromatic').tobytes()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        assert all(atom.is_aromatic for atom in mol.atoms)
        assert all(bond.is_aromatic for bond in mol.bonds)
        assert chemistry._require_aromatic_representation(mol) is None
        assert mol.calc_implicit_hydrogens() is None
    assert observed == []
    assert chemistry._require_implicit_hydrogens(mol) == (1,) * 6
    assert tuple(bond.bond_order for bond in mol.bonds) == (2., 1., 2., 1., 2., 1.)
    # The source aromatic field is retained when the equivalent projection prepares H.
    assert mol._atom_table.column('is_aromatic').tolist() == [1.] * 6
    assert mol._atom_table._sources[:6, FIELD_INDEX['is_aromatic']].tolist() == [ATOM_SOURCE_CODES['explicit']] * 6
    assert mol._atom_table.column('is_aromatic').tobytes() == source_flags


def test_independent_failed_and_valid_samples_keep_their_own_results():
    failed = _noncyclic_source()
    with pytest.raises(RuntimeError, match=r'AROMATIC\.NONCYCLIC_SOURCE'):
        chemistry._require_aromatic_representation(failed)
    failed_facts = factual_state(failed)
    valid = _benzene()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        assert valid.calc_implicit_hydrogens() is None
        assert chemistry._require_implicit_hydrogens(valid) == (1,) * 6
    assert observed == []
    assert factual_state(failed) == failed_facts
    assert failed._atom_table.column('is_aromatic').tolist() == [1., 0.]


def test_promoted_conflict_warning_restores_aromatic_source_and_dependency_state():
    mol = _noncyclic_source()
    before = _state(mol)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(RuntimeWarning, match=r'HYDROGEN\.REPRESENTATION_CONFLICT'):
            mol.calc_implicit_hydrogens()
    assert _state(mol) == before
    assert mol._atom_table.column('is_aromatic').tolist() == [1., 0.]


def _stored_unavailable(*, aromatic=False):
    """Restore a terminal status whose persisted facts have no retained cause."""
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    chemistry._prepare_hydrogens(mol)
    chemistry._prepare_aromatic_representation(mol)
    status, seeds = _dependency_state(mol._atom_table)
    status[0, H_GROUP] = _FactStatus.UNRESOLVED
    if aromatic:
        status[0, GROUP_INDEX['AROMATIC_REPRESENTATION']] = _FactStatus.UNRESOLVED
    _install_dependency_state(mol._atom_table, (status, seeds))
    return mol


def test_stored_unavailability_reports_actual_status_without_inventing_a_missing_input():
    mol = _stored_unavailable()
    before = _state(mol)
    with pytest.raises(ValueError) as failure:
        chemistry._require_implicit_hydrogens(mol)
    message = str(failure.value)
    assert re.findall(r'\bHYDROGEN\.[A-Z_]+\b', message) == ['HYDROGEN.UNAVAILABLE']
    assert 'atom_index=0' in message
    assert 'status=UNRESOLVED' in message
    assert 'missing' not in message.lower()
    assert _state(mol) == before


def test_stored_aromatic_unavailability_names_status_without_inventing_source_conflict():
    mol = _stored_unavailable(aromatic=True)
    before = _state(mol)
    table = mol._atom_table
    raw_values = table.matrix().tobytes()
    presence = int(table._presence[0])
    sources = table.sources().copy()
    assert _present(mol, 0)
    assert sources[0, H_FIELD] == ATOM_SOURCE_CODES['implicit_hydrogens']
    with pytest.raises(RuntimeError) as failure:
        chemistry._require_aromatic_representation(mol)
    aromatic_message = str(failure.value)
    assert 'AROMATIC.NO_CURRENT_ASSIGNMENT' in aromatic_message
    assert 'atom_index=0' in aromatic_message
    assert 'status=UNRESOLVED' in aromatic_message
    assert 'NONCYCLIC_SOURCE' not in aromatic_message
    assert _state(mol) == before
    with pytest.raises(ValueError) as failure:
        chemistry._require_implicit_hydrogens(mol)
    message = str(failure.value)
    assert 'HYDROGEN.AROMATIC_UNRESOLVED' in message
    assert 'atom_index=0' in message
    assert aromatic_message in message
    assert table.matrix().tobytes() == raw_values
    assert int(table._presence[0]) == presence & ~FIELD_PRESENCE_BITS[H_FIELD]
    sources[0, H_FIELD] = ATOM_SOURCE_CODES['default']
    assert table.sources().tobytes() == sources.tobytes()
    assert not _present(mol, 0)
    assert table._group_status[0, H_GROUP] == _FactStatus.UNRESOLVED
    assert table._group_status[0, GROUP_INDEX['AROMATIC_REPRESENTATION']] == _FactStatus.UNRESOLVED


def test_independent_unresolved_and_complete_hydrogen_samples_continue_normally():
    failed, atom = _missing_input('site')
    with pytest.warns(RuntimeWarning, match=r'HYDROGEN\.MISSING_SITE_ELEMENT'):
        assert atom.calc_implicit_hydrogens() is None
    failed_state = _state(failed)
    complete = Molecule()
    complete.create_atom(atomic_number=8)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        assert complete.calc_implicit_hydrogens() is None
    assert observed == []
    assert chemistry._require_implicit_hydrogens(complete) == (2,)
    assert _state(failed) == failed_state
    assert not _present(failed, 0)


def test_valid_component_does_not_inherit_disconnected_aromatic_source_failure():
    mol = Molecule()
    mol.add_component(_noncyclic_source())
    mol.add_component(_benzene())
    with pytest.raises(RuntimeError) as failure:
        chemistry._require_aromatic_representation(mol)
    message = str(failure.value)
    assert tuple(map(int, re.findall(r'AROMATIC\.NONCYCLIC_SOURCE atom_index=(\d+)', message))) == (0, 1)
    assert 'source_atom_indices=(0,)' in message
    assert 'AROMATIC.NO_CURRENT_ASSIGNMENT' not in message
    assert chemistry._require_aromatic_representation(mol, rows=tuple(range(2, 8))) is None
    assert all(atom.is_aromatic for atom in mol.atoms[2:])
    assert mol._atom_table.column('is_aromatic').tolist() == [1., 0.] + [1.] * 6
