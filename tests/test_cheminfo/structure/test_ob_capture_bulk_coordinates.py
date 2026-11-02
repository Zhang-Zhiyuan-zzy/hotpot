"""Atomic coordinate capture and complete write coverage for the OB projector."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple
from threading import Event
import struct

import numpy as np
import pytest
from openbabel import openbabel as ob

from hotpot import Molecule, read_mol
from hotpot.cheminfo import BondKind
from hotpot.cheminfo.operations import chemistry, conversion, io
from hotpot.cheminfo.adapters.openbabel import conversion as ob_adapter
from hotpot.cheminfo.adapters.io.formats import openbabel as ob_format
from hotpot.cheminfo.structure._builder import _freeze_value
from hotpot.cheminfo.structure._atom_table import _AtomFactTable
from hotpot.cheminfo.structure._fields import FIELD_INDEX


def _facts(mol):
    record, metadata = mol._capture_factual_record()
    return astuple(record), _freeze_value(metadata)


def _coordinates(obmol):
    return np.array([(atom.GetX(), atom.GetY(), atom.GetZ())
                     for atom in ob.OBMolAtomIter(obmol)])


def test_capture_holds_owner_lock_against_concurrent_coordinate_commit(monkeypatch):
    mol = read_mol('CCO', 'smi')
    before = np.arange(9.).reshape(3, 3) / 8
    after = before + 7.
    mol.coordinates = before
    mol.to_obmol()
    table = mol._atom_table
    real = _AtomFactTable.coordinates
    capture_started, release_capture, commit_attempted, commit_completed = (Event() for _ in range(4))
    captures = []

    def capture(value, rows=None):
        if value is table:
            captures.append(rows)
            capture_started.set()
            assert release_capture.wait(5)
        return real(value, rows)

    def commit():
        commit_attempted.set()
        mol.coordinates = after
        commit_completed.set()

    monkeypatch.setattr(_AtomFactTable, 'coordinates', capture)
    with ThreadPoolExecutor(max_workers=2) as pool:
        projection = pool.submit(mol.to_obmol)
        try:
            assert capture_started.wait(5)
            mutation = pool.submit(commit)
            assert commit_attempted.wait(5)
            assert not commit_completed.wait(.05)
        finally:
            release_capture.set()
        projected = projection.result(timeout=5)
        mutation.result(timeout=5)
    assert captures == [None]
    np.testing.assert_array_equal(_coordinates(projected), before)
    np.testing.assert_array_equal(mol.coordinates, after)
    assert not {'_obmol', '_row2idx'} & mol.__dict__.keys()


def test_capture_failure_keeps_completed_aromatic_preparation_before_backend_atoms(monkeypatch):
    mol = Molecule()
    first = mol.create_atom(atomic_number=6, formal_charge=0, coordinates=(0., 0., 0.))
    second = mol.create_atom(atomic_number=6, formal_charge=0, coordinates=(1.54, 0., 0.))
    mol.add_bond(first, second, 1.)
    events, containers, prepared_facts = [], [], []
    prepare = chemistry._require_aromatic_representation
    new_container = conversion._openbabel._new_obmol
    real_coordinates = _AtomFactTable.coordinates

    def aromatic(owner):
        events.append('aromatic:enter')
        result = prepare(owner)
        prepared_facts.append(_facts(owner))
        events.append('aromatic:returned')
        return result

    def container():
        result = new_container()
        containers.append(result)
        events.append('BeginModify:returned')
        return result

    def coordinates(value, rows=None):
        if value is mol._atom_table:
            events.append('coordinates:attempt')
            raise ValueError('Task88 injected coordinate snapshot failure')
        return real_coordinates(value, rows)

    def atom(*args, **kwargs):
        events.append('NewAtom:attempt')
        raise RuntimeError('Task88 competing backend atom failure')

    monkeypatch.setattr(chemistry, '_require_aromatic_representation', aromatic)
    monkeypatch.setattr(conversion._openbabel, '_new_obmol', container)
    monkeypatch.setattr(_AtomFactTable, 'coordinates', coordinates)
    monkeypatch.setattr(ob.OBMol, 'NewAtom', atom)
    with pytest.raises(ValueError, match='^Task88 injected coordinate snapshot failure$'):
        conversion.mol2obmol(mol)
    assert events == ['aromatic:enter', 'aromatic:returned', 'BeginModify:returned', 'coordinates:attempt']
    assert len(containers) == 1
    assert containers[0].NumAtoms() == containers[0].NumBonds() == 0
    assert _facts(mol) == prepared_facts[0]
    assert not {'_obmol', '_row2idx'} & mol.__dict__.keys()


@pytest.mark.parametrize('frame_count', [0, 1, 2])
@pytest.mark.parametrize('write_single', [False, True])
def test_each_write_uses_fresh_value_for_all_frames_and_preserves_distinct_current_frame(
        monkeypatch, tmp_path, frame_count, write_single):
    mol = read_mol('CCO', 'smi')
    frame = np.arange(9.).reshape(3, 3) / 8
    current = frame + 10.
    mol.coordinates = current
    for index in range(frame_count):
        mol.conformer_add(frame + index)
    held = mol.to_obmol()
    before = _facts(mol)
    projections, serialized, updates = [], [], []
    pack, serialize, update = ob_adapter._factual_obmol, ob_format._write_string, ob_adapter.set_obmol_coordinates

    def projection(*args, **kwargs):
        result = pack(*args, **kwargs)
        projections.append(result[0])
        return result

    def write(converter, value):
        serialized.append((value, _coordinates(value)))
        return serialize(converter, value)

    def coordinates(value, values):
        updates.append((value, np.array(values)))
        return update(value, values)

    monkeypatch.setattr(ob_adapter, '_factual_obmol', projection)
    monkeypatch.setattr(ob_format, '_write_string', write)
    monkeypatch.setattr(ob_adapter, 'set_obmol_coordinates', coordinates)
    expected = [current] if write_single or frame_count == 0 else [frame + i for i in range(frame_count)]
    writer = io.MolWriter(None, fmt='sdf')
    for index in range(2):
        text = writer.write(mol, write_single=write_single)
        assert text.count('$$$$') == len(expected)
        path = tmp_path / f'write-{index}.sdf'
        path.write_text(text)
        readback = list(io.MolReader(path, 'sdf'))
        assert len(readback) == len(expected)
        np.testing.assert_array_equal([m.coordinates for m in readback], expected)
        assert len(list(io.MolReader(text, 'sdf'))) == 1
    assert len(projections) == 2 and projections[0] is not projections[1]
    for index, value in enumerate(projections):
        selected = serialized[index * len(expected):(index + 1) * len(expected)]
        assert all(parent is value for parent, _ in selected)
        np.testing.assert_array_equal([coords for _, coords in selected], expected)
    assert len(updates) == (0 if write_single or frame_count == 0 else 2 * frame_count)
    np.testing.assert_array_equal(_coordinates(held), current)
    assert _facts(mol) == before
    assert not any(isinstance(value, ob.OBMol) for value in writer.__dict__.values())
    assert not {'_obmol', '_row2idx'} & mol.__dict__.keys()


@pytest.mark.parametrize('bits', [
    0x0000000000000000, 0x8000000000000000, 0x7ff0000000000000,
    0xfff0000000000000, 0x7ff8000000000042, 0xfff8000000000043,
    0x7ff0000000000001, 0x0000000000000001, 0x8000000000000001,
    0x7fefffffffffffff,
], ids=lambda bits: f'{bits:016x}')
def test_capture_retains_exact_scalar_types_float_bits_and_unbounded_integers(bits):
    mol = read_mol('CCO', 'smi')
    table = mol._atom_table
    value = struct.unpack('>d', struct.pack('>Q', bits))[0]
    # These deliberately raw numeric facts characterize capture only. They
    # must never be passed to a chemistry backend or interpreted as valid Z.
    for field in ('partial_charge', 'is_aromatic', 'x', 'y', 'z'):
        table._columns[FIELD_INDEX[field]][1] = value
    table._columns[FIELD_INDEX['atomic_number']][0] = 6.75
    table._columns[FIELD_INDEX['formal_charge']][0] = -1.75
    table._columns[FIELD_INDEX['atomic_number']][1] = float(2 ** 80)
    table._columns[FIELD_INDEX['formal_charge']][1] = -float(2 ** 100)
    with mol._mutation_lock:
        rows, bonds, cell = conversion._capture_ob_values(mol, tuple(mol.bonds))
    assert type(rows) is tuple and len(rows) == 3
    for row in rows:
        assert type(row) is tuple
        assert tuple(type(value) for value in row) == (int, int, float, tuple, float)
        assert len(row[3]) == 3 and all(type(value) is float for value in row[3])
    assert rows[0][:2] == (6, -1)
    assert rows[1][:2] == (2 ** 80, -(2 ** 100))
    assert rows[2][0] == 8
    for captured in (rows[1][2], rows[1][4], *rows[1][3]):
        assert struct.pack('>d', captured) == struct.pack('>Q', bits)
    assert type(bonds) is tuple and len(bonds) == 2 and cell is None


@pytest.mark.parametrize(('slots', 'error_type', 'message'), [
    ([(0, 'formal_charge', float('nan')), (1, 'atomic_number', float('inf'))],
     ValueError, 'cannot convert float NaN to integer'),
    ([(1, 'atomic_number', float('inf')), (1, 'formal_charge', float('nan'))],
     OverflowError, 'cannot convert float infinity to integer'),
    ([(1, 'formal_charge', float('-inf')), (2, 'atomic_number', float('nan'))],
     OverflowError, 'cannot convert float infinity to integer'),
], ids=['earlier-row-formal-charge', 'atomic-number-before-formal-charge', 'later-row-nan'])
def test_capture_malformed_integer_error_preserves_row_and_field_precedence(slots, error_type, message):
    mol = read_mol('CCO', 'smi')
    table = mol._atom_table
    for row, field, value in slots:
        table._columns[FIELD_INDEX[field]][row] = value
    before = tuple(block[:len(table)].tobytes() for block in table._blocks)
    with mol._mutation_lock, pytest.raises(error_type) as caught:
        conversion._capture_ob_values(mol, tuple(mol.bonds))
    assert type(caught.value) is error_type and caught.value.args == (message,)
    assert tuple(block[:len(table)].tobytes() for block in table._blocks) == before


def test_capture_rows_remain_detached_and_next_capture_reads_current_values():
    mol = read_mol('CCO', 'smi')
    before = np.arange(9.).reshape(3, 3) / 8
    mol.coordinates = before
    mol._atom_table.update((1,), {'formal_charge': 0., 'partial_charge': 0.})
    with mol._mutation_lock:
        previous, _, _ = conversion._capture_ob_values(mol, tuple(mol.bonds))
        mol.coordinates = before + 7.
        mol._atom_table.update((1,), {'formal_charge': -1., 'partial_charge': .375})
        current, _, _ = conversion._capture_ob_values(mol, tuple(mol.bonds))
    assert previous[1] == (6, 0, 0., (.375, .5, .625), 0.)
    assert current[1] == (6, -1, .375, (7.375, 7.5, 7.625), 0.)
    assert previous is not current and previous[1] is not current[1]
    assert previous[1][3] is not current[1][3]


@pytest.mark.parametrize('malformed', [False, True], ids=['complete-atoms-before-cutoff', 'atom-error-before-cutoff'])
def test_capture_finishes_atom_rows_before_first_dative_and_never_reads_cell(monkeypatch, malformed):
    mol = Molecule()
    for number in (7, 29, 8):
        mol.create_atom(atomic_number=number)
    mol.coordinates = np.array(((0., 0., 0.), (1., 0., 0.), (2., 0., 0.)))
    dative = mol.add_bond(0, 1, bond_kind=BondKind.DATIVE, bond_direction='atom1_to_atom2')
    mol._atom_table._columns[FIELD_INDEX['formal_charge']][2] = float('nan') if malformed else -2.75
    cell_reads = []
    def unread_cell(owner):
        cell_reads.append(owner)
        raise AssertionError('cell was read after first dative')
    monkeypatch.setattr(Molecule, 'crystal', property(unread_cell))
    with mol._mutation_lock:
        if malformed:
            with pytest.raises(ValueError) as caught:
                conversion._capture_ob_values(mol, (dative, object()))
            assert type(caught.value) is ValueError
            assert caught.value.args == ('cannot convert float NaN to integer',)
        else:
            rows, bonds, cell = conversion._capture_ob_values(mol, (dative, object()))
            assert len(rows) == 3
            assert rows[2] == (8, -2, 0., (2., 0., 0.), 0.)
            assert bonds == (('dative',),) and cell is None
    assert cell_reads == []
