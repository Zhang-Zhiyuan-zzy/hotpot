"""Accepted export capture precedence after actual lazy preparations."""
from collections import Counter
from dataclasses import astuple

import pytest
from openbabel import openbabel as ob
from rdkit import Chem

from hotpot import Atom, Molecule
from hotpot.cheminfo.operations import chemistry, conversion
from hotpot.cheminfo.adapters.rdkit import stereo_transport
from hotpot.cheminfo.structure._builder import _freeze_value
from hotpot.cheminfo.structure._atom_table import _AtomFactTable


def _facts(mol):
    record, metadata = mol._capture_factual_record()
    return astuple(record), _freeze_value(metadata)


def _two_carbons():
    mol = Molecule()
    first = mol.create_atom(atomic_number=6, formal_charge=0, coordinates=(0., 0., 0.))
    second = mol.create_atom(atomic_number=6, formal_charge=0, coordinates=(1.54, 0., 0.))
    mol.add_bond(first, second, 1.)
    mol.charge = 0
    return mol


class Observation:
    def __init__(self, mol, monkeypatch):
        self.mol, self.patch = mol, monkeypatch
        self.events, self.returns, self.completed_facts = [], {}, {}
        self.containers = []

    def watch(self, owner, name):
        real = getattr(owner, name)
        def watched(*args, **kwargs):
            self.events.append(name + ':enter')
            result = real(*args, **kwargs)
            self.returns.setdefault(name, []).append(result)
            self.completed_facts[name] = _facts(self.mol)
            self.events.append(name + ':returned')
            return result
        self.patch.setattr(owner, name, watched)

    def preparations(self):
        for name in ('_require_implicit_hydrogens', '_prepare_hydrogens',
                     '_require_aromatic_representation', '_prepare_aromatic_representation'):
            self.watch(chemistry, name)
        for name in ('_validate_payload_references', '_payload_atom_facts'):
            self.watch(stereo_transport, name)

    def row(self, atom):
        return self.mol._topology_index[id(atom)][0]


def test_rd_capture_precedes_atom_failure_after_preparation(monkeypatch):
    mol = _two_carbons()
    observation = Observation(mol, monkeypatch)
    observation.preparations()
    real_rwmol = Chem.RWMol
    real_hybridization = chemistry._atom_hybridization
    def empty_container(*args, **kwargs):
        value = real_rwmol(*args, **kwargs)
        observation.containers.append(value)
        observation.events.append('RWMol:created')
        assert value.GetNumAtoms() == value.GetNumBonds() == 0
        return value
    def fail_atom(*args, **kwargs):
        observation.events.append('Chem.Atom:attempt')
        raise RuntimeError('S7A injected RD atom0 construction')
    def hybridization(atom):
        row = observation.row(atom)
        observation.events.append('hybridization:' + str(row))
        if row == 1:
            raise ValueError('S7A injected atom1 hybridization')
        return real_hybridization(atom)
    monkeypatch.setattr(Chem, 'RWMol', empty_container)
    monkeypatch.setattr(Chem, 'Atom', fail_atom)
    monkeypatch.setattr(chemistry, '_atom_hybridization', hybridization)
    with pytest.raises(ValueError, match='^S7A injected atom1 hybridization$'):
        conversion.to_rdmol(mol)
    counts = Counter(observation.events)
    assert observation.returns['_require_implicit_hydrogens'] == [(3, 3)]
    assert counts['_require_aromatic_representation:returned'] == 1
    assert counts['_prepare_hydrogens:returned'] >= 1
    assert counts['_prepare_aromatic_representation:returned'] >= 1
    assert counts['_validate_payload_references:returned'] == 1
    assert counts['_payload_atom_facts:returned'] == 1
    assert counts['Chem.Atom:attempt'] == 0
    assert len(observation.containers) == 1
    assert observation.containers[0].GetNumAtoms() == observation.containers[0].GetNumBonds() == 0
    selected = {'_require_implicit_hydrogens:enter', '_require_aromatic_representation:enter',
                'RWMol:created', '_validate_payload_references:enter', '_payload_atom_facts:enter',
                'hybridization:0', 'hybridization:1'}
    assert [event for event in observation.events if event in selected] == [
        '_require_implicit_hydrogens:enter', '_require_aromatic_representation:enter',
        'RWMol:created', '_validate_payload_references:enter', '_payload_atom_facts:enter',
        'hybridization:0', 'hybridization:1']
    assert _facts(mol) == observation.completed_facts['_payload_atom_facts']
    assert not {'_obmol', '_row2idx'} & mol.__dict__.keys()


def test_ob_capture_precedes_atom_failure_after_preparation(monkeypatch):
    mol = _two_carbons()
    observation = Observation(mol, monkeypatch)
    observation.preparations()
    real_obmol, real_begin = ob.OBMol, ob.OBMol.BeginModify
    real_new_atom, real_end = ob.OBMol.NewAtom, ob.OBMol.EndModify
    def empty_container(*args, **kwargs):
        value = real_obmol(*args, **kwargs)
        observation.containers.append(value)
        observation.events.append('OBMol:created')
        assert value.NumAtoms() == value.NumBonds() == 0
        return value
    def begin(value, *args, **kwargs):
        result = real_begin(value, *args, **kwargs)
        observation.events.append('BeginModify:returned')
        assert value.NumAtoms() == value.NumBonds() == 0
        return result
    def new_atom(*args, **kwargs):
        observation.events.append('NewAtom:attempt')
        return real_new_atom(*args, **kwargs)
    def fail_atomic_number(*args, **kwargs):
        observation.events.append('SetAtomicNum:attempt')
        raise RuntimeError('S7A injected OB atom0 SetAtomicNum')
    def end(*args, **kwargs):
        observation.events.append('EndModify:attempt')
        return real_end(*args, **kwargs)
    def coordinates(table, rows=None):
        assert table is mol._atom_table and rows is None
        observation.events.append('coordinates:snapshot')
        raise ValueError('S7A injected coordinate snapshot')
    monkeypatch.setattr(real_obmol, 'BeginModify', begin)
    monkeypatch.setattr(real_obmol, 'NewAtom', new_atom)
    monkeypatch.setattr(real_obmol, 'EndModify', end)
    monkeypatch.setattr(ob.OBAtom, 'SetAtomicNum', fail_atomic_number)
    monkeypatch.setattr(ob, 'OBMol', empty_container)
    monkeypatch.setattr(_AtomFactTable, 'coordinates', coordinates)
    with pytest.raises(ValueError, match='^S7A injected coordinate snapshot$'):
        conversion.mol2obmol(mol)
    counts = Counter(observation.events)
    assert counts['_require_aromatic_representation:returned'] == 1
    assert counts['_prepare_aromatic_representation:returned'] >= 1
    assert counts['_require_implicit_hydrogens:enter'] == counts['_prepare_hydrogens:enter'] == 1
    assert observation.returns['_require_implicit_hydrogens'] == [(3, 3)]
    assert counts['_validate_payload_references:enter'] == counts['_payload_atom_facts:enter'] == 0
    assert counts['NewAtom:attempt'] == counts['SetAtomicNum:attempt'] == counts['EndModify:attempt'] == 0
    assert len(observation.containers) == 1
    assert observation.containers[0].NumAtoms() == observation.containers[0].NumBonds() == 0
    selected = {'_require_implicit_hydrogens:enter', '_require_aromatic_representation:enter',
                'OBMol:created', 'BeginModify:returned', 'coordinates:snapshot'}
    assert [event for event in observation.events if event in selected] == [
        '_require_implicit_hydrogens:enter', '_require_aromatic_representation:enter',
        'OBMol:created', 'BeginModify:returned', 'coordinates:snapshot']
    assert _facts(mol) == observation.completed_facts['_require_aromatic_representation']
    assert not {'_obmol', '_row2idx'} & mol.__dict__.keys()
