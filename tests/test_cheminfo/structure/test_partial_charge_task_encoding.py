"""Task charge encoding preserves observations and trained feature meanings."""

from queue import Queue

import numpy as np
import pytest

from hotpot import Atom, Molecule, read_mol
from hotpot.cheminfo.AImodels import data_extract
from hotpot.cheminfo.forcefields.native_packing import pack_complex_session_input
from hotpot.cheminfo.operations.chemistry import _atom_projection
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
from hotpot.cheminfo.structure._results import _import_partial_charges


def _prepared_molecule():
    mol = read_mol("CCO")
    _atom_projection(mol, strict_hydrogens=True)
    return mol


@pytest.mark.parametrize("values", [None, (None, 0.0, -0.25)])
def test_model_charge_zero_encoding_preserves_first_fifteen_fields_and_metal_feature(values):
    mol = _prepared_molecule()
    if values is not None:
        _import_partial_charges(mol, values, producer=ATOM_SOURCE_CODES["explicit"])
    selected = mol._partial_charge_result
    direct = mol.atom_matrix

    features = data_extract.extract_atom_attrs(mol)
    expected_charges = [0.0, 0.0, 0.0 if values is None else -0.25]

    assert features["x_names"] == Atom._attrs_enumerator[:15] + ("is_metal",)
    assert features["x"].shape == (3, 16)
    assert features["x"].dtype == np.float64
    assert features["x"][:, 8].tolist() == expected_charges
    other_columns = [column for column in range(15) if column != 8]
    np.testing.assert_array_equal(features["x"][:, other_columns], direct[:, other_columns])
    assert features["x"][:, 15].tolist() == [0.0, 0.0, 0.0]
    assert features["x"][:, 0].tolist() == [6.0, 6.0, 8.0]
    assert features["x"][:, 1:7].tolist() == [
        [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 4.0, 0.0, 0.0, 0.0],
    ]
    assert features["x"][:, 13:15].tolist() == [[4.0, 3.0], [4.0, 2.0], [2.0, 1.0]]
    assert np.isnan(direct[0, 8])
    assert mol._partial_charge_result is selected
    assert tuple(atom.partial_charge for atom in mol.atoms) == (
        (None,) * 3 if values is None else values
    )


def test_model_stale_charge_encoding_is_finite_and_retains_historical_values():
    mol = _prepared_molecule()
    _import_partial_charges(mol, (-0.25, 0.0, 0.25), producer=ATOM_SOURCE_CODES["explicit"])
    mol.atoms[0].x = 1.0
    selected = mol._partial_charge_result

    features = data_extract.extract_atom_attrs(mol)

    assert np.isfinite(features["x"]).all()
    assert features["x"][:, 8].tolist() == [0.0, 0.0, 0.0]
    assert np.isnan(mol.atom_matrix[:, 8]).all()
    assert mol._partial_charge_result is selected
    assert selected.invalidated_by
    assert selected.payload.values == (-0.25, 0.0, 0.25)


def test_atomic_number_only_model_request_does_not_consult_charge(monkeypatch):
    mol = _prepared_molecule()

    def forbidden(*args):
        pytest.fail("atomic-number-only task consulted partial charges")

    monkeypatch.setattr(data_extract, "_partial_charge_projection", forbidden)
    assert data_extract.extract_atom_attrs(mol, atomic_number_only=True)["x"].tolist() == [6, 6, 8]


@pytest.mark.parametrize("stale", [False, True])
def test_forcefield_session_uses_finite_charge_encoding_without_writeback(stale):
    mol = _prepared_molecule()
    _import_partial_charges(mol, (None, 0.0, -0.25), producer=ATOM_SOURCE_CODES["explicit"])
    if stale:
        mol.atoms[0].x = 1.0
    selected = mol._partial_charge_result

    session = pack_complex_session_input(mol)

    assert session.partial_charges.tolist() == ([0.0] * 3 if stale else [0.0, 0.0, -0.25])
    assert session.partial_charges.dtype == np.float64
    assert session.partial_charges.flags.c_contiguous
    assert np.isfinite(session.partial_charges).all()
    assert mol._partial_charge_result is selected
    assert selected.payload.values == (None, 0.0, -0.25)


@pytest.mark.parametrize("multiprocess_path", [False, True], ids=["serial", "worker"])
def test_tmqm_imports_one_complete_supplied_charge_set_after_topology(monkeypatch, multiprocess_path):
    from hotpot.ds.providers import tmqm

    generator = object.__new__(tmqm.DataGenerator)
    xyz = (2, (("C", "0", "0", "0"), ("N", "1.4", "0", "0")))
    csv = ["charge-fixture"] + ["0"] * 8
    charges = [0.0, -0.25, "Total -0.25"]
    text = (xyz, csv, charges, [])
    events = []

    def link_atoms(mol):
        assert mol._partial_charge_result is None
        assert tuple(atom.partial_charge for atom in mol.atoms) == (None, None)
        mol.add_bond(0, 1, 1.0)
        events.append("topology")

    def import_charges(mol, values, **kwargs):
        assert len(mol.bonds) == 1
        events.append(("charges", values, kwargs["producer"]))
        return _import_partial_charges(mol, values, **kwargs)

    monkeypatch.setattr(Molecule, "link_atoms", link_atoms)
    monkeypatch.setattr(tmqm, "_import_partial_charges", import_charges)
    if multiprocess_path:
        generator.text2mol_queues = [Queue()]
        generator.result_queues = [Queue()]
        generator.text2mol_queues[0].put(text)
        generator.text2mol_queues[0].put(None)
        generator._mp_mol_construct(0)
        mol = generator.result_queues[0].get_nowait()
        assert generator.result_queues[0].get_nowait() is None
    else:
        generator._xyz_generator = iter((xyz,))
        generator._csv_generator = iter((csv,))
        generator._q_generator = iter((charges,))
        generator._bo_generator = iter(([],))
        mol = generator.next_mol()

    assert events == ["topology", ("charges", (0.0, -0.25), ATOM_SOURCE_CODES["tmqm"])]
    assert mol.identifier == "charge-fixture"
    assert tuple(atom.symbol for atom in mol.atoms) == ("C", "N")
    assert tuple(atom.partial_charge for atom in mol.atoms) == (0.0, -0.25)
    result = mol._partial_charge_result
    assert not result.invalidated_by
    assert result.payload.method == "supplied"
    assert result.payload.provenance == (ATOM_SOURCE_CODES["tmqm"],) * 2
