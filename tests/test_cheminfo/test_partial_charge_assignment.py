"""Explicit charge calculation, atomic selection and optional toolkit transport."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple
import json

import numpy as np
import pytest
from openbabel import openbabel as ob
from rdkit import Chem

from hotpot import Molecule, read_mol
from hotpot.cheminfo.operations import conversion
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from hotpot.cheminfo.structure._results import _import_partial_charges


def _prepared():
    mol = read_mol('CO', 'smi')
    mol.coordinates = ((0., 0., 0.), (1.43, 0., 0.))
    mol.to_obmol()
    return mol


def _charges(mol):
    return tuple(atom.partial_charge for atom in mol.atoms)


def test_assignment_calls_existing_calculation_once_without_fresh_owner_lock(monkeypatch):
    mol = _prepared()
    calls = []

    def calculate(owner, model, model_args):
        calls.append((owner, model, model_args))
        assert not owner._mutation_lock._is_owned()
        return (0., -.125)

    monkeypatch.setattr(Molecule, 'get_partial_charge', calculate)
    assert mol.assign_partial_charge('gasteiger', 'arguments') == (0., -.125)
    assert calls == [(mol, 'gasteiger', 'arguments')]
    result = mol._partial_charge_result.payload
    assert result.atom_uids == tuple(atom._uid for atom in mol.atoms)
    assert result.values == _charges(mol) == (0., -.125)
    assert result.method == 'gasteiger'
    assert result.method_version == ob.OBReleaseVersion()
    assert result.parameters == (('model_args', 'arguments'),)
    assert result.units == 'e'
    assert result.provenance == (ATOM_SOURCE_CODES['openbabel'],) * 2
    assert not np.any(mol._atom_table.presence() & FIELD_PRESENCE_BITS[FIELD_INDEX['partial_charge']])


def test_failed_calculation_preserves_previous_exact_selection_and_facts(monkeypatch):
    mol = _prepared()
    _import_partial_charges(mol, (.1, -.1), producer=ATOM_SOURCE_CODES['explicit'])
    before = mol._partial_charge_result
    record = astuple(mol._capture_factual_record()[0])

    def fail(owner, **kwargs):
        raise RuntimeError('charge backend failed')

    monkeypatch.setattr(Molecule, 'get_partial_charge', fail)
    with pytest.raises(RuntimeError, match='charge backend failed'):
        mol.assign_partial_charge()
    assert mol._partial_charge_result is before
    assert astuple(mol._capture_factual_record()[0]) == record


def test_failed_calculation_keeps_completed_aromatic_preparation(monkeypatch):
    mol = Molecule()
    mol.create_atom(atomic_number=6, formal_charge=0, coordinates=(0., 0., 0.))
    mol.create_atom(atomic_number=6, formal_charge=0, coordinates=(1.54, 0., 0.))
    mol.add_bond(0, 1, 1.)
    _import_partial_charges(mol, (.1, -.1), producer=ATOM_SOURCE_CODES['explicit'])
    payload = mol._partial_charge_result.payload
    prepared = []
    prepare = conversion._chemistry._require_aromatic_representation

    def aromatic(owner):
        value = prepare(owner)
        prepared.append(astuple(owner._capture_factual_record()[0]))
        return value

    def fail(owner, **kwargs):
        raise RuntimeError('charge backend failed after preparation')

    monkeypatch.setattr(conversion._chemistry, '_require_aromatic_representation', aromatic)
    monkeypatch.setattr(Molecule, 'get_partial_charge', fail)
    with pytest.raises(RuntimeError, match='failed after preparation'):
        mol.assign_partial_charge()
    assert len(prepared) == 1
    assert astuple(mol._capture_factual_record()[0]) == prepared[0]
    assert mol._partial_charge_result.payload is payload


@pytest.mark.parametrize('mutation', ['geometry', 'membership'])
def test_source_drift_during_calculation_rejects_complete_selection(monkeypatch, mutation):
    mol = _prepared()
    _import_partial_charges(mol, (.1, -.1), producer=ATOM_SOURCE_CODES['explicit'])
    payload = mol._partial_charge_result.payload
    calls = []

    def calculate(owner, **kwargs):
        calls.append(True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            if mutation == 'geometry':
                pool.submit(setattr, owner.atoms[0], 'x', .5).result(timeout=5)
            else:
                pool.submit(owner.create_atom, atomic_number=1).result(timeout=5)
        return (0., 0.)

    monkeypatch.setattr(Molecule, 'get_partial_charge', calculate)
    with pytest.raises(RuntimeError):
        mol.assign_partial_charge()
    assert calls == [True]
    assert mol._partial_charge_result.payload is payload
    assert mol._partial_charge_result.invalidated_by
    if mutation == 'geometry':
        assert mol.atoms[0].x == .5
    else:
        assert len(mol.atoms) == 3


def test_assignment_inside_outer_transaction_restores_exact_selection_on_rollback(monkeypatch):
    mol = _prepared()
    _import_partial_charges(mol, (.1, -.1), producer=ATOM_SOURCE_CODES['explicit'])
    before = mol._partial_charge_result
    monkeypatch.setattr(Molecule, 'get_partial_charge', lambda self, **kwargs: (.2, -.2))
    with pytest.raises(RuntimeError, match='outer rollback'):
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['explicit']):
            mol.assign_partial_charge()
            assert _charges(mol) == (.2, -.2)
            raise RuntimeError('outer rollback')
    assert mol._partial_charge_result is before
    assert _charges(mol) == (.1, -.1)


def test_real_openbabel_calculation_is_nonselecting_then_explicitly_selected(tmp_path):
    mol = _prepared()
    formal_charges = tuple(atom.formal_charge for atom in mol.atoms)
    net_charge = mol.charge
    assert _charges(mol) == (None, None)
    observed = mol.get_partial_charge('gasteiger')
    assert isinstance(observed, tuple) and len(observed) == 2
    assert np.isfinite(observed).all()
    assert _charges(mol) == (None, None)
    selected = mol.assign_partial_charge('gasteiger')
    assert isinstance(selected, tuple) and np.isfinite(selected).all()
    assert _charges(mol) == selected
    assert tuple(atom.formal_charge for atom in mol.atoms) == formal_charges
    assert mol.charge == net_charge
    (tmp_path / 'actual-charge-observation.json').write_text(json.dumps({
        'input_smiles': 'CO', 'coordinates_angstrom': mol.coordinates.tolist(),
        'formal_charges': formal_charges, 'molecular_charge': net_charge,
        'method': 'gasteiger', 'method_version': ob.OBReleaseVersion(),
        'model_args': None, 'units': 'e', 'get_values': observed,
        'assigned_values': selected,
        'interpretation': 'Named-method observations; no universal chemical truth or charge-sum claim.',
    }, indent=2) + '\n')


@pytest.mark.parametrize(('returned', 'succeeded', 'message'), [
    ((0.,), True, 'returned 1 charges for 2 atoms'),
    ((0., float('nan')), True, 'non-finite charges'),
    ((0., float('inf')), True, 'non-finite charges'),
    ((0., 0.), False, 'could not calculate charges'),
])
def test_calculation_retains_backend_count_and_finite_contract(monkeypatch, returned, succeeded, message):
    mol = _prepared()
    monkeypatch.setattr(conversion._openbabel, '_charge_model', lambda model: object())
    monkeypatch.setattr(conversion._openbabel, '_compute_charges', lambda *args: (object(), succeeded))
    monkeypatch.setattr(conversion._openbabel, '_partial_charges', lambda model: returned)
    with pytest.raises(RuntimeError, match=message):
        mol.get_partial_charge()
    assert mol._partial_charge_result is None


def test_unavailable_model_keeps_explicit_local_failure(monkeypatch):
    mol = _prepared()
    monkeypatch.setattr(conversion._openbabel, '_charge_model', lambda model: None)
    with pytest.raises(ValueError, match='unavailable; known models'):
        mol.get_partial_charge('unknown')
    assert mol._partial_charge_result is None


@pytest.mark.parametrize('values', [(None, None), (0., -.25), (None, 0.), (.25, None)])
@pytest.mark.parametrize('toolkit', ['rdkit', 'openbabel'])
def test_optional_charge_roundtrip_preserves_none_and_zero_through_toolkit_copy(values, toolkit):
    mol = _prepared()
    _import_partial_charges(mol, values, producer=ATOM_SOURCE_CODES['explicit'])
    if toolkit == 'rdkit':
        external = Chem.Mol(mol.to_rdmol())
        assert tuple(atom.HasProp('_GasteigerCharge') for atom in external.GetAtoms()) == tuple(
            value is not None for value in values)
        restored = conversion.from_rdmol(external, Molecule())
    else:
        external = ob.OBMol(mol.to_obmol())
        assert external.HasPartialChargesPerceived() == any(value is not None for value in values)
        restored = conversion.obmol2mol(external, Molecule())
    assert _charges(restored) == values
    expected_code = ATOM_SOURCE_CODES[toolkit]
    if any(value is not None for value in values):
        assert restored._partial_charge_result.payload.method == 'supplied'
        assert restored._partial_charge_result.payload.provenance == tuple(
            expected_code if value is not None else ATOM_SOURCE_CODES['default'] for value in values)


def test_openbabel_absent_import_never_reads_or_perceives_charges(monkeypatch):
    source = ob.OBMol()
    source.NewAtom().SetAtomicNum(6)
    source.NewAtom().SetAtomicNum(8)
    source.AddBond(1, 2, 1)
    assert not source.HasPartialChargesPerceived()

    def forbidden(atom):
        raise AssertionError('absent conversion attempted charge perception')

    monkeypatch.setattr(ob.OBAtom, 'GetPartialCharge', forbidden)
    restored = conversion.obmol2mol(source, Molecule())
    assert _charges(restored) == (None, None)
    assert not source.HasPartialChargesPerceived()


def test_stale_charge_export_omits_values_without_recalculation(monkeypatch):
    mol = _prepared()
    _import_partial_charges(mol, (0., -.25), producer=ATOM_SOURCE_CODES['explicit'])
    mol.atoms[0].x = .5

    def forbidden(*args, **kwargs):
        raise AssertionError('conversion calculated partial charges')

    monkeypatch.setattr(Molecule, 'get_partial_charge', forbidden)
    for toolkit in ('rdkit', 'openbabel'):
        if toolkit == 'rdkit':
            external = mol.to_rdmol()
            assert not any(atom.HasProp('_GasteigerCharge') for atom in external.GetAtoms())
            restored = conversion.from_rdmol(external, Molecule())
        else:
            external = mol.to_obmol()
            assert not external.HasPartialChargesPerceived()
            restored = conversion.obmol2mol(external, Molecule())
        assert _charges(restored) == (None, None)
    with pytest.raises(RuntimeError):
        _charges(mol)
