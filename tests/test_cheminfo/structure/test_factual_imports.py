"""Toolkit decoder effects at the factual-import boundary."""

from dataclasses import replace
from types import ModuleType, SimpleNamespace
import importlib
import sys

import numpy as np
import pytest
from openbabel import openbabel as ob
from rdkit import Chem

from hotpot import Atom, Molecule
from hotpot.cheminfo import obconvert, rdconvert

core = importlib.import_module(Molecule.__module__)
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from hotpot.cheminfo.stereo import StereoPayload


def _obmol():
    value = ob.OBMol()
    value.NewAtom().SetAtomicNum(6)
    value.NewAtom().SetAtomicNum(8)
    value.AddBond(1, 2, 2)
    value.SetTotalCharge(0)
    return value


@pytest.mark.parametrize('convert,source', [(obconvert.obmol2mol, _obmol),
                                         (rdconvert.from_rdmol, lambda: Chem.MolFromSmiles('CO'))])
def test_converter_uses_same_target_and_one_topology_build_without_public_setters(monkeypatch, convert, source):
    target = Molecule()
    token, lock = target._owner_token, target._mutation_lock
    target._next_atom_uid, target._next_bond_uid = 7, 11
    seen = []
    original = core._build_topology_index
    def count(*args, **kwargs):
        seen.append(len(args[0]))
        return original(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('decoder used public setter construction')
    monkeypatch.setattr(core, '_build_topology_index', count)
    monkeypatch.setattr(Molecule, '_create_atom', forbidden)
    monkeypatch.setattr(Molecule, '_add_bond', forbidden)
    monkeypatch.setattr(Atom, 'get_valence', forbidden)
    assert convert(source(), target) is target
    assert seen == [2]
    assert target._owner_token is token and target._mutation_lock is lock
    assert [atom._uid for atom in target.atoms] == [7, 8]
    assert target.bonds[0]._uid == 11


@pytest.mark.parametrize('convert', [obconvert.obmol2mol, rdconvert.from_rdmol])
@pytest.mark.parametrize('partition', ['atom', 'hidden_metal', 'hidden_covalent'])
def test_nonempty_converter_target_is_rejected_before_source_access_and_mutation(convert, partition):
    target = Molecule()
    if partition == 'atom':
        target.create_atom(atomic_number=6)
    else:
        getattr(target, '_hided_' + ('metal' if partition == 'hidden_metal' else 'covalent') + '_bonds').append(object())
    table, index = target._atom_table, target._topology_index
    high_water = target._next_atom_uid, target._next_bond_uid
    with pytest.raises(ValueError, match='empty Atom and complete-Bond'):
        convert(object(), target)
    assert target._atom_table is table and target._topology_index is index
    assert (target._next_atom_uid, target._next_bond_uid) == high_water


def test_rdkit_presence_distinguishes_absent_coordinates_partial_charge_and_supplied_zero():
    source = Chem.MolFromSmiles('CO')
    source.GetAtomWithIdx(0).SetDoubleProp('_GasteigerCharge', 0.)
    mol = rdconvert.from_rdmol(source, Molecule())
    masks, codes = mol._atom_table.presence(), mol._atom_table.sources()
    assert masks[0] & FIELD_PRESENCE_BITS[FIELD_INDEX['partial_charge']]
    assert not masks[1] & FIELD_PRESENCE_BITS[FIELD_INDEX['partial_charge']]
    assert not any(mask & FIELD_PRESENCE_BITS[FIELD_INDEX['x']] for mask in masks)
    assert not any(mask & FIELD_PRESENCE_BITS[FIELD_INDEX['id']] for mask in masks)
    assert codes[0, FIELD_INDEX['partial_charge']] == ATOM_SOURCE_CODES['rdkit']
    assert codes[1, FIELD_INDEX['partial_charge']] == ATOM_SOURCE_CODES['default']
    assert codes[0, FIELD_INDEX['s']] == ATOM_SOURCE_CODES['electron_configuration']
    assert mol._charge_present and mol._charge_source == 'rdkit' and mol.charge == 0


@pytest.mark.parametrize('convert,source,keeps_stereo', [
    (obconvert.obmol2mol, _obmol, True),
    (rdconvert.from_rdmol, Chem.Mol, False),
])
def test_empty_preconfigured_target_preserves_cell_spin_identifier_and_clears_all_frames(convert, source, keeps_stereo):
    target = Molecule()
    target.create_crystal(5, 6, 7, 90, 90, 90)
    target.spin_mult = 3
    target.identifier = 'existing'
    payload = StereoPayload(1, (), (), ())
    target._stereo_payload = payload
    for index, name in enumerate(target.conformers._attrs):
        setattr(target.conformers, '_' + name, np.array([index + 1]))
    target._conformers_index = 0
    convert(source(), target)
    assert tuple(getattr(target.crystal, name) for name in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')) == (5, 6, 7, 90, 90, 90)
    assert target.spin_mult == 3 and target.identifier == 'existing'
    assert all(getattr(target.conformers, '_' + name) is None for name in target.conformers._attrs)
    assert (target._stereo_payload is payload) is keeps_stereo


def test_failed_decoder_install_restores_preconfigured_empty_target(monkeypatch):
    target = Molecule()
    target.charge = 5
    original = Molecule._install_frame_state
    def fail(self, values, index):
        original(self, values, index)
        raise RuntimeError('frame publication')
    monkeypatch.setattr(Molecule, '_install_frame_state', fail)
    with pytest.raises(RuntimeError, match='frame publication'):
        rdconvert.from_rdmol(Chem.MolFromSmiles('CO'), target)
    assert not target.atoms and not target.bonds and target.charge == 5
    assert target._next_atom_uid == 2 and target._next_bond_uid == 1


def test_cclib_prepares_all_nine_frames_units_and_last_coordinates_before_science(monkeypatch):
    from hotpot.cheminfo._io import _io
    coordinates = np.arange(12.).reshape(2, 2, 3)
    energies = np.array([-10., -11.])
    charges = np.array([-0.25, 0.25])
    forces = np.arange(12.).reshape(2, 2, 3) / 10
    data = SimpleNamespace(atomnos=np.array([6, 8]), atomcoords=coordinates,
        scfenergies=energies, atomcharges={'mulliken': charges},
        charge=0, mult=3, freeenergy=-1.25, zpve=0.125)
    monkeypatch.setitem(sys.modules, 'cclib', SimpleNamespace(ccopen=lambda value: SimpleNamespace(parse=lambda: data)))
    monkeypatch.setattr(_io, '_extract_g16_thermo', lambda lines: (0.75, 0.25))
    monkeypatch.setattr(_io, '_extract_force_matrix', lambda lines, numbers: forces)
    events = []
    def science(name):
        def observe(mol, *args, **kwargs):
            events.append(name)
            assert mol._charge_present and mol._charge_source == 'cclib'
            assert mol._spin_present and mol.spin_mult == 3
            assert mol._conformers_index == -1
            np.testing.assert_array_equal(mol.coordinates, coordinates[-1])
            np.testing.assert_array_equal(mol.conformers._coordinates, coordinates)
            np.testing.assert_array_equal(mol.conformers._force, forces)
        return observe
    for name in ('link_atoms', 'determine_rings_aromatic', 'assign_bond_order', 'calc_atom_valence'):
        monkeypatch.setattr(Molecule, name, science(name))
    reader = object.__new__(_io.MolReader)
    reader.src, reader.src_type, reader.fmt = 'fake quantum output', 'str', 'log'
    mol = next(reader._cclib_read())
    assert events == ['link_atoms', 'determine_rings_aromatic', 'assign_bond_order', 'calc_atom_valence']
    expected = (coordinates, energies, np.array([[0, 0], charges]), -1.25 * 27.211386245988,
                forces, 0.125 * 27.211386245988, 3, 0.75, 0.25)
    for name, value in zip(mol.conformers._attrs, expected):
        np.testing.assert_array_equal(getattr(mol.conformers, '_' + name), value)
    coordinates[:] = 99
    energies[:] = 99
    forces[:] = 99
    assert mol.conformers._coordinates[0, 0, 0] == 0
    assert mol.conformers._energy.tolist() == [-10, -11]
    assert mol.conformers._force[0, 0, 0] == 0
    codes = mol._atom_table.sources()
    assert np.all(codes[:, FIELD_INDEX['x']] == ATOM_SOURCE_CODES['cclib'])
    assert np.all(codes[:, FIELD_INDEX['s']] == ATOM_SOURCE_CODES['electron_configuration'])
    assert not np.any(mol._atom_table.presence() & FIELD_PRESENCE_BITS[FIELD_INDEX['partial_charge']])


def _fake_ccdc(monkeypatch):
    package = ModuleType('ccdc')
    molecule = ModuleType('ccdc.molecule')
    molecule.Molecule = type('Molecule', (), {})
    molecule.Atom = type('Atom', (), {})
    molecule.Bond = type('Bond', (), {'BondType': SimpleNamespace(all_bond_types=lambda: {'Single': 1, 'Double': 2})})
    package.molecule = molecule
    monkeypatch.setitem(sys.modules, 'ccdc', package)
    monkeypatch.setitem(sys.modules, 'ccdc.molecule', molecule)
    monkeypatch.delitem(sys.modules, 'hotpot.plugins.ccdc_api.convert', raising=False)
    return importlib.import_module('hotpot.plugins.ccdc_api.convert')


def test_ccdc_fake_boundary_uses_atoms_collection_returns_carrier_and_installs_no_chemistry(monkeypatch):
    conversion = _fake_ccdc(monkeypatch)
    atoms = [SimpleNamespace(label='C1', atomic_number=6, formal_charge=0,
                            partial_charge=0.25, coordinates=(1, 2, 3), rings=()),
             SimpleNamespace(label='O1', atomic_number=8, formal_charge=-1,
                             partial_charge=-0.25, coordinates=(4, 5, 6), rings=())]
    normalized = []
    source = SimpleNamespace(atoms=atoms, bonds=[SimpleNamespace(atoms=tuple(atoms), bond_type='Double')],
                             normalise_labels=lambda: normalized.append(True))
    def forbidden(*args, **kwargs):
        raise AssertionError('CCDC factual import used chemistry')
    monkeypatch.setattr(Atom, 'get_valence', forbidden)
    monkeypatch.setattr(Atom, '_calc_implicit_hydrogens', forbidden)
    mol = conversion.mol_from_ccdc_to_hp(source)
    assert normalized == [True]
    assert isinstance(mol, Molecule)
    assert [atom.atomic_number for atom in mol.atoms] == [6, 8]
    assert [atom.formal_charge for atom in mol.atoms] == [0, -1]
    assert mol.coordinates.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert mol.bonds[0].bond_order == 2
    assert not np.any(mol._atom_table.presence() & FIELD_PRESENCE_BITS[FIELD_INDEX['valence']])
    assert np.all(mol._atom_table.sources()[:, FIELD_INDEX['formal_charge']] == ATOM_SOURCE_CODES['ccdc'])
    assert not mol._charge_present


def test_g16_prewrite_rolls_back_chemical_stores_if_charge_calculation_fails(monkeypatch):
    from hotpot.cheminfo._io import g16
    from hotpot.cheminfo._io._io import MolWriter
    to_gjf_preprocess = MolWriter._plugins["gjf"]["pre"]
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    before = mol.atom_matrix.tobytes()
    def failed_charge(self):
        self.atoms[0].implicit_hydrogens = 9
        raise RuntimeError('charge calculation failed')
    monkeypatch.setattr(Molecule, 'calc_mol_default_charge', failed_charge)
    with pytest.raises(RuntimeError, match='charge calculation failed'):
        to_gjf_preprocess(SimpleNamespace(kwargs={}, ob_opt={}), mol)
    assert mol.atom_matrix.tobytes() == before
