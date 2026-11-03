"""Producer boundaries for canonical source frames and current observations."""

from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.adapters.io.formats import g16
from hotpot.cheminfo.structure._builder import _thaw_value
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES


def _gaussian_record(monkeypatch, *, charges=None, forces=None, energies=None,
                     gibbs=None, zero_point=None):
    coordinates = np.arange(18., dtype=float).reshape(3, 2, 3)
    data = SimpleNamespace(atomnos=np.array([6, 8]), atomcoords=coordinates,
        atomcharges={} if charges is None else {"mulliken": charges},
        charge=0, mult=1)
    if energies is not None:
        data.scfenergies = energies
    if gibbs is not None:
        data.freeenergy = gibbs
    if zero_point is not None:
        data.zpve = zero_point
    monkeypatch.setattr(g16, "_extract_force_matrix", lambda *args: (
        np.empty((0,)) if forces is None else forces))
    monkeypatch.setattr(g16, "_extract_g16_thermo", lambda lines: (None, None))
    mol = Molecule()
    record = g16._cclib_record(data, "", "str", mol._capture_factual_record()[0])
    mol._install_molecule(record)
    return mol, coordinates


@pytest.mark.parametrize("charge", [None, np.zeros(2), np.array([-.25, .25])],
                         ids=["absent", "explicit-zero", "last-only"])
def test_gaussian_import_distinguishes_absent_zero_and_final_only_mulliken(monkeypatch, charge):
    mol, coordinates = _gaussian_record(monkeypatch, charges=charge)
    assert len(mol._source_frames) == 3 and not mol._conformer_results
    np.testing.assert_array_equal(mol.coordinates, coordinates[-1])
    assert mol.conformers[0]["partial_charge"] is None
    assert mol.conformers[1]["partial_charge"] is None
    if charge is None:
        assert mol.conformers[2]["partial_charge"] is None
    else:
        np.testing.assert_array_equal(mol.conformers[2]["partial_charge"], charge)
        raw = _thaw_value(mol._source_frames[-1].observations.values[1])
        assert raw.shape == (2,)
        np.testing.assert_array_equal(raw, charge)
    # Frame import never selects Atom charges as an implicit side effect.
    assert tuple(atom.partial_charge for atom in mol.atoms) == (None, None)


def test_gaussian_scalar_scope_and_ordered_histories_preserve_zero_and_conversions(monkeypatch):
    forces = np.arange(18.).reshape(3, 2, 3) / 10
    energies = np.array([0., -1., -2.])
    mol, _ = _gaussian_record(monkeypatch, forces=forces, energies=energies,
                              gibbs=0., zero_point=.125)
    for index in range(3):
        frame = mol.conformers[index]
        assert frame["energy"] == energies[index]
        assert frame["gibbs"] == 0.
        assert frame["zero_point"] == pytest.approx(.125 * 27.211386245988)
        np.testing.assert_array_equal(frame["force"], forces[index])
    forces[:] = 99
    np.testing.assert_array_equal(mol.conformers[0]["force"], np.arange(6.).reshape(2, 3) / 10)


@pytest.mark.parametrize("field", ["energy", "force", "partial_charge"])
def test_gaussian_unequal_histories_are_retained_raw_with_no_invented_mapping(monkeypatch, field):
    raw = {"energy": np.array([-1., -2.]),
           "force": np.arange(12.).reshape(2, 2, 3),
           "partial_charge": np.array([[.1, -.1], [.2, -.2]])}[field]
    argument = {"energy": "energies", "force": "forces", "partial_charge": "charges"}[field]
    with pytest.warns(RuntimeWarning, match=(
            rf"frame_observation_unmapped field={field} coordinates=3 observations=2")):
        mol, coordinates = _gaussian_record(monkeypatch, **{argument: raw})
    assert mol.conformers_number == 3
    np.testing.assert_array_equal(mol.coordinates, coordinates[-1])
    assert all(mol.conformers[index][field] is None for index in range(3))
    column = ("energy", "partial_charge", "gibbs", "force", "zero_point",
              "spin_mult", "thermo", "capacity").index(field)
    np.testing.assert_array_equal(_thaw_value(mol._source_frames[0].observations.values[column]), raw)


@pytest.mark.parametrize("worker", [False, True], ids=["serial", "worker"])
def test_tmqm_publishes_energy_once_after_topology_and_full_charge_import(monkeypatch, worker):
    from hotpot.ds.providers import tmqm
    from hotpot.cheminfo.structure.conformer import _select_frame_observations

    generator = object.__new__(tmqm.DataGenerator)
    xyz = (2, (("C", "0", "0", "0"), ("N", "1.4", "0", "0")))
    csv = ["frame-fixture", "0"] + ["0"] * 7
    charges = [0., -.25, "Total -0.25"]
    events = []

    def link_atoms(mol):
        assert mol._frame_result is None
        mol.add_bond(0, 1, 1.)
        events.append("topology")

    original_import = tmqm._import_partial_charges
    def import_charges(mol, values, **kwargs):
        events.append("charges")
        return original_import(mol, values, **kwargs)

    def select(mol, values, **kwargs):
        assert len(mol.bonds) == 1
        assert tuple(atom.partial_charge for atom in mol.atoms) == (0., -.25)
        assert values == (0., None, None, None, None, None, None, None)
        assert kwargs["producer"] == "tmqm"
        events.append("observations")
        return _select_frame_observations(mol, values, **kwargs)

    monkeypatch.setattr(Molecule, "link_atoms", link_atoms)
    monkeypatch.setattr(tmqm, "_import_partial_charges", import_charges)
    monkeypatch.setattr(tmqm, "_select_frame_observations", select)
    if worker:
        generator.text2mol_queues, generator.result_queues = [Queue()], [Queue()]
        generator.text2mol_queues[0].put((xyz, csv, charges, []))
        generator.text2mol_queues[0].put(None)
        generator._mp_mol_construct(0)
        mol = generator.result_queues[0].get_nowait()
        assert generator.result_queues[0].get_nowait() is None
    else:
        generator._xyz_generator, generator._csv_generator = iter((xyz,)), iter((csv,))
        generator._q_generator, generator._bo_generator = iter((charges,)), iter(([],))
        mol = generator.next_mol()
    assert events == ["topology", "charges", "observations"]
    assert mol.energy == 0. and not mol._frame_result.invalidated_by
    assert not mol._source_frames and not mol._conformer_results
    assert mol._partial_charge_result.payload.provenance == (ATOM_SOURCE_CODES["tmqm"],) * 2
    mol.atoms[0].x += 1
    with pytest.raises(RuntimeError, match="frame_observation_stale"):
        _ = mol.energy
