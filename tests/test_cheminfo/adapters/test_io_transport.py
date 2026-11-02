"""One projection per complete write, frame transport and public I/O behavior."""
from dataclasses import astuple
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from openbabel import openbabel as ob

import hotpot
from hotpot import Molecule, read_mol
from hotpot.cheminfo.operations import conversion, io
from hotpot.cheminfo.adapters.openbabel import conversion as ob_adapter
from hotpot.cheminfo.structure._builder import _freeze_value


def _facts(mol):
    record, metadata = mol._capture_factual_record()
    return astuple(record), _freeze_value(metadata)


def _frames():
    mol = read_mol('CCO', 'smi')
    coordinates = np.arange(9.).reshape(3, 3) / 8
    mol.coordinates = coordinates
    mol.conformer_add(coordinates)
    mol.conformer_add(coordinates + 1.)
    mol.to_obmol()
    return mol, coordinates


def test_public_reader_writer_exports_are_the_operation_definitions():
    cheminfo = importlib.import_module('hotpot.cheminfo')
    for name in ('MolReader', 'MolWriter', 'read_mol'):
        assert getattr(hotpot, name) is getattr(cheminfo, name) is getattr(io, name)
    assert io.MolReader.__module__ == io.MolWriter.__module__ == 'hotpot.cheminfo.operations.io'


def test_multiframe_write_projects_once_reuses_frames_and_preserves_string_path_cardinality(monkeypatch, tmp_path):
    mol, coordinates = _frames()
    before = _facts(mol)
    calls = {'operation': 0, 'adapter': 0}
    operation, packing = conversion.mol2obmol, ob_adapter._factual_obmol
    serialized_coordinates = []
    original_write = ob.OBConversion.WriteString
    def project(*args, **kwargs):
        calls['operation'] += 1
        return operation(*args, **kwargs)
    def pack(*args, **kwargs):
        calls['adapter'] += 1
        return packing(*args, **kwargs)
    def serialize(self, value, *args, **kwargs):
        serialized_coordinates.append(tuple((a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(value)))
        return original_write(self, value, *args, **kwargs)
    monkeypatch.setattr(conversion, 'mol2obmol', project)
    monkeypatch.setattr(ob_adapter, '_factual_obmol', pack)
    monkeypatch.setattr(ob.OBConversion, 'WriteString', serialize)
    writer = io.MolWriter(None, fmt='sdf')
    text = writer.write(mol, write_single=False)
    assert calls == {'operation': 1, 'adapter': 1}
    assert text.count('$$$$') == 2
    np.testing.assert_array_equal(serialized_coordinates, [coordinates, coordinates + 1.])
    assert _facts(mol) == before
    assert not any(isinstance(value, ob.OBMol) for value in writer.__dict__.values())
    second = writer.write(mol, write_single=False)
    assert calls == {'operation': 2, 'adapter': 2}
    assert second.count('$$$$') == 2
    np.testing.assert_array_equal(serialized_coordinates, [coordinates, coordinates + 1., coordinates, coordinates + 1.])
    assert _facts(mol) == before
    # Keep the inherited string reader's first-record behavior separate from
    # actual all-frame file-path readback.
    string_records = list(io.MolReader(text, 'sdf'))
    path = tmp_path / 'two-frames.sdf'
    path.write_text(text)
    path_records = list(io.MolReader(path, 'sdf'))
    second_path = tmp_path / 'second-write.sdf'
    second_path.write_text(second)
    second_records = list(io.MolReader(second_path, 'sdf'))
    assert len(string_records) == 1 and len(path_records) == len(second_records) == 2
    np.testing.assert_array_equal(string_records[0].coordinates, coordinates)
    np.testing.assert_array_equal(path_records[0].coordinates, coordinates)
    np.testing.assert_array_equal(path_records[1].coordinates, coordinates + 1.)
    np.testing.assert_array_equal(second_records[0].coordinates, coordinates)
    np.testing.assert_array_equal(second_records[1].coordinates, coordinates + 1.)
    current = mol.to_obmol()
    np.testing.assert_array_equal([(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(current)], coordinates)
    assert not {'_obmol', '_row2idx'} & mol.__dict__.keys()


def test_single_frame_option_and_file_return_overwrite_behavior(tmp_path):
    mol, coordinates = _frames()
    before = _facts(mol)
    text = mol.write(fmt='sdf', write_single=True)
    assert text.count('$$$$') == 1
    path = tmp_path / 'single.sdf'
    assert mol.write(path, fmt='sdf', write_single=True) is None
    assert len(list(io.MolReader(path, 'sdf'))) == 1
    np.testing.assert_array_equal(io.read_mol(path, 'sdf').coordinates, coordinates)
    with pytest.raises(FileExistsError):
        mol.write(path, fmt='sdf')
    assert mol.write(path, fmt='sdf', overwrite=True, write_single=False) is None
    assert len(list(io.MolReader(path, 'sdf'))) == 2
    assert _facts(mol) == before


def test_plugin_registry_order_kwargs_precedence_and_decorator_return_are_preserved(monkeypatch):
    fmt = 'stage7a-contract'
    monkeypatch.setitem(io.MolWriter._plugins, fmt, {})
    events = []
    def pre(writer, mol, *args, **kwargs):
        events.append(('pre', args, kwargs['winner']))
    def write(writer, mol, *args, **kwargs):
        events.append(('write', args, kwargs['winner']))
        return 'original'
    def post(writer, script, mol, *args, **kwargs):
        events.append(('post', script, args, kwargs['winner']))
        return script + '-post'
    assert io.MolWriter.add_plugin(fmt, 'pre')(pre) is None
    assert io.MolWriter.add_plugin(fmt, 'write')(write) is None
    assert io.MolWriter.add_plugin(fmt, 'post')(post) is None
    assert io.MolWriter._plugins[fmt] == {'pre': pre, 'write': write, 'post': post}
    writer = io.MolWriter(None, fmt=fmt, winner='constructor')
    assert writer.write(Molecule(), 'argument', winner='call') == 'original-post'
    assert events == [('pre', ('argument',), 'constructor'),
                      ('write', ('argument',), 'constructor'),
                      ('post', 'original', ('argument',), 'constructor')]


def test_gjf_charge_failure_rolls_back_complete_facts_and_frames(monkeypatch):
    mol, _ = _frames()
    before = _facts(mol)
    def failed_charge(self):
        self.atoms[0].implicit_hydrogens = 9
        self.atoms[0].formal_charge = 2
        raise RuntimeError('charge preparation failure')
    monkeypatch.setattr(Molecule, 'calc_mol_default_charge', failed_charge)
    callback = io.MolWriter._plugins['gjf']['pre']
    with pytest.raises(RuntimeError, match='charge preparation failure'):
        callback(SimpleNamespace(kwargs={}, ob_opt={}), mol)
    assert _facts(mol) == before
