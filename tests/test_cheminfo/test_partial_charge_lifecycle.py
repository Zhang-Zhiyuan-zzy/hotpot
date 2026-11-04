"""Selected partial charges survive only the explicitly supported lifecycles."""

from copy import deepcopy
from dataclasses import fields, replace
import pickle

import numpy as np
import pytest

from hotpot import Atom, Molecule
from hotpot.cheminfo.structure._builder import MoleculeRecord, _restore_molecule_record
from hotpot.cheminfo.structure._commit import _FactTransaction, _commit_topology
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX
from hotpot.cheminfo.structure._results import (
    PartialChargeSet, _ResultAttachment, _capture_partial_charge_source,
    _import_partial_charges,
)
from hotpot.cheminfo.structure._selection import _capture_topology_selection
from hotpot.cheminfo.operations.topology_queries import _project_molecule


def molecule():
    mol = Molecule()
    for number, x in ((6, 0.0), (8, 1.4), (7, 3.0)):
        Atom(mol, atomic_number=number, formal_charge=0, implicit_hydrogens=0, coordinates=(x, 0.0, 0.0))
    mol.add_bond(0, 1)
    _ = mol.atom_matrix
    return mol


def select(mol, values=(0.0, -0.25, None)):
    with mol._mutation_lock:
        uids = tuple(atom._uid for atom in mol.atoms)
        payload = PartialChargeSet(uids, values, 'gasteiger', 'known-version',
            (('model_args', None),), 'e', (ATOM_SOURCE_CODES['openbabel'],) * len(uids))
        from hotpot.cheminfo.structure._results import _select_partial_charge_result
        return _select_partial_charge_result(mol, payload, _capture_partial_charge_source(mol), uids)


def assert_payload(payload, values):
    assert payload.values == values
    assert payload.method == 'gasteiger' and payload.method_version == 'known-version'
    assert payload.parameters == (('model_args', None),) and payload.units == 'e'


@pytest.mark.parametrize('stale', [False, True])
def test_exact_copy_preserves_method_presence_provenance_and_staleness(stale):
    mol = molecule()
    select(mol)
    mol.properties['partial_charge'] = {'annotation': 'independent'}
    if stale:
        mol.atoms[0].x = 0.5
    marker = mol._partial_charge_result.invalidated_by
    clone = mol.copy()
    attachment = clone._partial_charge_result
    assert attachment.source.owner is clone._owner_token
    assert clone._owner_token is not mol._owner_token
    assert attachment.payload.atom_uids == tuple(atom._uid for atom in clone.atoms)
    assert_payload(attachment.payload, (0.0, -0.25, None))
    assert attachment.payload.provenance == mol._partial_charge_result.payload.provenance
    assert attachment.invalidated_by == marker
    if stale:
        with pytest.raises(RuntimeError, match='geometry'):
            _ = clone.atoms[0].partial_charge
    else:
        _check_source(clone, attachment.source)
        assert tuple(atom.partial_charge for atom in clone.atoms) == (0.0, -0.25, None)
    clone.properties['new'] = 1
    assert 'new' not in mol.properties


def test_current_whole_copy_checks_original_source_before_rebinding():
    mol, foreign = molecule(), molecule()
    payload = select(mol)
    mol._partial_charge_result = _ResultAttachment(payload, _capture_partial_charge_source(foreign))
    with pytest.raises(RuntimeError, match='owner'):
        mol.copy()


@pytest.mark.parametrize('stale', [False, True])
def test_current_record_transports_explicit_payload_and_marker_without_raw_charge_bits(stale):
    mol = molecule()
    select(mol)
    if stale:
        mol.atoms[0].x = 0.5
    record, metadata = mol._capture_factual_record(include_frames=False)
    assert tuple(field.name for field in fields(record))[-1] == 'partial_charge_result'
    assert len(fields(record)) == 24
    payload, marker = record.partial_charge_result
    assert payload.atom_uids == (0, 1, 2)
    assert_payload(payload, (0.0, -0.25, None))
    assert marker == mol._partial_charge_result.invalidated_by
    assert np.isnan(np.frombuffer(record.atom_values, dtype='<f8').reshape(3, 19)[:, 8]).all()
    assert not (np.frombuffer(record.atom_presence, dtype='<u4') & (1 << 8)).any()
    assert (np.frombuffer(record.atom_sources, dtype='u1').reshape(3, 19)[:, 8] == 0).all()
    restore, (record_type, version, values) = record.__reduce__()
    assert version == 6 and len(values) == 24
    for old_version in (0, 1, 2, 3, 4, 5):
        with pytest.raises(ValueError, match='version 6 and 24 fields'):
            _restore_molecule_record(record_type, old_version, values)
    transported = restore(record_type, version, values)
    rebuilt = Molecule._build_molecule(transported, bond_metadata=metadata)
    assert rebuilt._partial_charge_result.invalidated_by == marker
    assert_payload(rebuilt._partial_charge_result.payload, payload.values)
    if stale:
        with pytest.raises(RuntimeError, match='geometry'):
            _ = rebuilt.atoms[0].partial_charge
    else:
        _check_source(rebuilt, rebuilt._partial_charge_result.source)
        assert rebuilt.atoms[0].partial_charge == 0.0


def test_legacy_record_inputs_move_presence_values_and_producer_once():
    mol = molecule()
    record, _ = mol._capture_factual_record(include_frames=False)
    values = np.frombuffer(record.atom_values, dtype='<f8').reshape(3, 19).copy()
    presence = np.frombuffer(record.atom_presence, dtype='<u4').copy()
    sources = np.frombuffer(record.atom_sources, dtype='u1').reshape(3, 19).copy()
    values[:, 8] = (0.0, 99.0, -0.25)
    presence[[0, 2]] |= 1 << 8
    sources[:, 8] = (3, 0, 4)
    supplied = replace(record, atom_values=values.tobytes(), atom_presence=presence.tobytes(),
                       atom_sources=sources.tobytes())
    payload, marker = supplied.partial_charge_result
    assert payload.values == (0.0, None, -0.25) and payload.provenance == (3, 0, 4)
    assert payload.method == 'supplied' and payload.method_version is None and marker == ()
    assert np.isnan(np.frombuffer(supplied.atom_values, dtype='<f8').reshape(3, 19)[:, 8]).all()
    replaced = replace(supplied, identifier=(True, 'unchanged charge'))
    assert replaced.partial_charge_result == supplied.partial_charge_result
    with pytest.raises(ValueError, match='excludes supplied Atom charge bits'):
        replace(supplied, atom_presence=presence.tobytes())
    with pytest.raises(ValueError, match='canonical complete rows'):
        replace(supplied, partial_charge_result=(replace(payload, atom_uids=(8, 9, 10)), marker))


def test_graph_and_row_subsets_preserve_matching_history_with_stale_context():
    mol = molecule()
    select(mol)
    selection = _capture_topology_selection(mol, excluded_bonds=(mol.bonds[0],))
    projected = _project_molecule(mol, Molecule, selection=selection)
    assert_payload(projected._partial_charge_result.payload, (0.0, -0.25, None))
    assert 'charge_representation' in projected._partial_charge_result.invalidated_by
    for component in mol.components:
        payload = component._partial_charge_result.payload
        original_rows = tuple(next(row for row, atom in enumerate(mol.atoms)
                                   if atom.atomic_number == member.atomic_number)
                              for member in component.atoms)
        assert payload.values == tuple((0.0, -0.25, None)[row] for row in original_rows)
        assert 'charge_representation' in component._partial_charge_result.invalidated_by
        with pytest.raises(RuntimeError, match='charge_representation'):
            _ = component.atoms[0].partial_charge
    record, metadata = mol._capture_factual_record(rows=(1, 0), include_frames=False)
    subset = Molecule._build_molecule(record, bond_metadata=metadata)
    assert_payload(subset._partial_charge_result.payload, (-0.25, 0.0))
    assert subset._partial_charge_result.payload.atom_uids == (0, 1)
    assert 'charge_representation' in subset._partial_charge_result.invalidated_by


def test_capture_after_remove_and_add_maps_historical_uids_and_fills_unobserved_rows():
    mol = molecule()
    select(mol)
    removed = mol.atoms[0]
    mol.remove_atom(removed)
    mol.create_atom(atomic_number=1, formal_charge=0, implicit_hydrogens=0)
    assert removed.partial_charge is None and np.isnan(removed.attrs[8])
    record, metadata = mol._capture_factual_record(include_frames=False)
    payload, marker = record.partial_charge_result
    assert payload.values == (-0.25, None, None)
    assert payload.provenance == (4, 4, 0)
    assert marker and payload.atom_uids == (0, 1, 2)
    rebuilt = Molecule._build_molecule(record, bond_metadata=metadata)
    assert rebuilt._partial_charge_result.payload.values == payload.values
    assert rebuilt._partial_charge_result.invalidated_by == marker


def test_nonempty_merge_retains_receiver_history_and_empty_merge_rebinds_incoming():
    receiver, incoming = molecule(), molecule()
    select(receiver)
    _import_partial_charges(incoming, (0.5, None, -0.5), producer=ATOM_SOURCE_CODES['rdkit'])
    incoming_slot = incoming._partial_charge_result
    receiver.add_component(incoming)
    result = receiver._partial_charge_result
    assert_payload(result.payload, (0.0, -0.25, None, None, None, None))
    assert result.invalidated_by and result.payload.provenance == (4, 4, 4, 0, 0, 0)
    assert incoming._partial_charge_result is incoming_slot
    assert incoming.atoms[0].partial_charge == 0.5
    empty = Molecule()
    empty.add_component(incoming)
    assert empty._partial_charge_result.payload.values == (0.5, None, -0.5)
    assert empty._partial_charge_result.invalidated_by == ()
    _check_source(empty, empty._partial_charge_result.source)
    assert empty.atoms[0].partial_charge == 0.5


def noncontiguous_reordered_molecule():
    mol = molecule()
    mol.remove_atom(0)
    mol.create_atom(atomic_number=16, formal_charge=0, implicit_hydrogens=0)
    original = tuple(mol.atoms)
    rows = (2, 0, 1)
    record, metadata = mol._capture_factual_record(rows=rows, include_frames=False)
    _commit_topology(mol, _capture_source(mol), record,
        atom_bindings=tuple(original[row] for row in rows), bond_bindings=(), bond_metadata=metadata)
    _ = mol.atom_matrix
    select(mol, (0.25, None, -0.25))
    assert tuple(atom._uid for atom in mol.atoms) == (3, 1, 2)
    return mol


@pytest.mark.parametrize('transport', [deepcopy, lambda value: pickle.loads(pickle.dumps(value))], ids=['deepcopy', 'pickle'])
@pytest.mark.parametrize('stale', [False, True])
def test_pickle_and_deepcopy_preserve_noncontiguous_reordered_uid_owner_sharing(transport, stale):
    mol = noncontiguous_reordered_molecule()
    if stale:
        mol.atoms[0].x = 1.0
    restored, member = transport((mol, mol.atoms[1]))
    assert member is restored.atoms[1] and member.mol is restored
    assert tuple(atom._uid for atom in restored.atoms) == (3, 1, 2)
    attachment = restored._partial_charge_result
    assert attachment.source.owner is restored._owner_token
    assert attachment.payload.atom_uids == (3, 1, 2)
    assert_payload(attachment.payload, (0.25, None, -0.25))
    assert attachment.invalidated_by == mol._partial_charge_result.invalidated_by
    if stale:
        with pytest.raises(RuntimeError, match='geometry'):
            _ = restored.atoms[0].partial_charge
    else:
        _check_source(restored, attachment.source)
        assert tuple(atom.partial_charge for atom in restored.atoms) == (0.25, None, -0.25)


def test_working_record_publication_then_rollback_restores_exact_selection():
    mol = molecule()
    select(mol)
    original = mol._partial_charge_result
    work = mol.copy()
    _import_partial_charges(work, (0.5, -0.5, None), producer=ATOM_SOURCE_CODES['forcefield'])
    record, metadata = work._capture_factual_record(include_frames=False)
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1):
            _commit_topology(mol, _capture_source(mol), record,
                atom_bindings=tuple(mol.atoms), bond_bindings=tuple(mol.bonds), bond_metadata=metadata)
            assert mol.atoms[0].partial_charge == 0.5
            raise KeyboardInterrupt('abort working record publication')
    assert mol._partial_charge_result is original and mol.atoms[0].partial_charge == 0.0


@pytest.mark.parametrize('change', ['spin', 'labels', 'annotations', 'constraints', 'frames'])
def test_non_input_mutations_retain_exact_charge_selection(change):
    mol = molecule()
    select(mol)
    original = mol._partial_charge_result
    if change == 'spin':
        mol.spin_mult = 3
    elif change == 'labels':
        mol.identifier = 'renamed'
        mol.atoms[0].id = 41
    elif change == 'annotations':
        mol.properties['partial_charge'] = [10.0, 20.0]
    elif change == 'constraints':
        mol.atoms[0].constraint = True
        mol.bonds[0].constraint = True
    else:
        mol.conformer_add(mol.coordinates)
        mol.conformers.clear()
    assert mol._partial_charge_result is original
    _check_source(mol, original.source)
    assert mol.atoms[0].partial_charge == 0.0


def test_legacy_row_import_preserves_supplied_observations_without_physical_charge_storage():
    mol = Molecule()
    rows = np.zeros((2, 19), dtype=np.float64)
    rows[:, 0] = (6, 8)
    rows[:, 8] = (0.0, -0.25)
    first = mol._create_atom_from_array(rows[0])
    assert first.partial_charge == 0.0
    mol._create_atom_from_array(rows[1])
    attachment = mol._partial_charge_result
    assert attachment.payload.values == (0.0, -0.25)
    assert attachment.payload.method == 'supplied' and attachment.invalidated_by
    assert mol._atom_table._columns[8] is None
    assert not (mol._atom_table.presence() & (1 << 8)).any()
