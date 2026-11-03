"""Gaussian log parsing plus an opt-in official-executable integration."""

from pathlib import Path
import shutil
import warnings

import numpy as np
import pytest

import hotpot as hp
from hotpot.plugins.qm.gaussian import GaussOut, run_gaussian


FIXTURE = Path(__file__).parents[1] / "input" / "Am_BuPh-BPPhen.log"


def test_gaussian_log_parser_preserves_numeric_results_with_tolerance() -> None:
    output = GaussOut.read_file(FIXTURE)
    parsed = output.parse()

    assert not output.is_error
    assert parsed["coordinates"].shape == (73, 3)
    assert parsed["force"].shape == (54, 73, 3)
    assert np.isfinite(parsed["coordinates"]).all()
    assert np.isfinite(parsed["force"]).all()
    assert parsed["energy"] == pytest.approx(-60574.40216122108, abs=1.0e-8)
    assert parsed["gibbs"] == pytest.approx(-60560.09243823103, abs=1.0e-8)
    assert parsed["zero_point"] == pytest.approx(16.2662864132, abs=1.0e-9)
    assert parsed["capacity"] == pytest.approx(0.00616935257, rel=1.0e-8)


@pytest.mark.external
def test_official_gaussian_single_point_uses_dynamic_files(tmp_path: Path) -> None:
    if shutil.which("g16") is None:
        pytest.skip("Gaussian 16 executable is not available")
    molecule = hp.read_mol("C", "smi")
    molecule.build3d(seed=17, epochs=1, steps_per_epoch=20)

    output = run_gaussian(
        molecule,
        route="#p hf/sto-3g sp",
        gjf_save_path=tmp_path / "methane.gjf",
        log_save_path=tmp_path / "methane.log",
        err_save_path=tmp_path / "methane.err",
        test=False,
    )

    assert not output.is_error
    assert np.isfinite(output.parse()["energy"])


def test_existing_log_import_and_update_keep_scoped_observations_without_engine():
    """The real log is a calculation artifact; no Gaussian executable is used."""
    output = GaussOut.read_file(FIXTURE)
    parsed = output.parse()
    with warnings.catch_warnings(record=True) as findings:
        warnings.simplefilter("always")
        molecule = hp.read_mol(FIXTURE, "g16log")
    assert molecule.conformers_number > 1
    assert molecule.conformers[0]["partial_charge"] is None
    np.testing.assert_array_equal(molecule.conformers[-1]["partial_charge"], parsed["partial_charge"])
    np.testing.assert_array_equal(molecule.coordinates, parsed["coordinates"])
    observations = molecule._source_frames[0].observations
    from hotpot.cheminfo.structure._builder import _thaw_value
    raw_forces = _thaw_value(observations.values[3])
    np.testing.assert_array_equal(raw_forces, parsed["force"])
    if len(raw_forces) != molecule.conformers_number:
        assert molecule.conformers[-1]["force"] is None
        assert any("frame_observation_unmapped field=force" in str(item.message)
                   for item in findings)
    sources = molecule._source_frames
    output.update_mol(molecule)
    assert molecule._source_frames is sources
    assert molecule.energy == pytest.approx(parsed["energy"], abs=1.e-8)
    assert molecule.gibbs == pytest.approx(parsed["gibbs"], abs=1.e-8)
    assert molecule.zero_point == pytest.approx(parsed["zero_point"], abs=1.e-9)
    assert molecule.capacity == pytest.approx(parsed["capacity"], rel=1.e-8)
    np.testing.assert_array_equal(molecule.force, parsed["force"])
    np.testing.assert_array_equal([atom.partial_charge for atom in molecule.atoms], parsed["partial_charge"])
    assert not molecule._frame_result.invalidated_by
    assert not molecule._partial_charge_result.invalidated_by
    assert molecule._partial_charge_result.payload.method == "mulliken"
    np.testing.assert_array_equal(molecule.mol_orbital_energies, parsed["mol_orbital_energies"])


def test_gaussian_update_publishes_all_observations_after_geometry_and_declarations(monkeypatch):
    from hotpot.cheminfo.structure import conformer, _results
    molecule = hp.Molecule()
    molecule.create_atom(atomic_number=6)
    molecule.create_atom(atomic_number=8)
    molecule.add_bond(0, 1, 1.)
    coordinates = np.arange(6.).reshape(2, 3)
    charges = np.zeros(2)
    forces = np.ones((2, 2, 3))
    attrs = {"atomic_number": np.array([6, 8]), "coordinates": coordinates,
             "partial_charge": charges, "energy": 0., "gibbs": None,
             "zero_point": 0., "spin": 1, "spin_mult": 1, "charge": 0,
             "force": forces, "thermo": None, "capacity": 0.,
             "mol_orbital_energies": np.array([[-1., 1.]])}
    output = GaussOut("")
    monkeypatch.setattr(output, "parse", lambda: dict(attrs))
    events = []
    select = conformer._select_frame_observations
    import_charges = _results._import_partial_charges

    def publish(owner, values, **kwargs):
        np.testing.assert_array_equal(owner.coordinates, coordinates)
        assert owner.charge == 0 and owner.spin_mult == 1
        events.append("observations")
        return select(owner, values, **kwargs)

    def charges_after(owner, values, **kwargs):
        assert owner.energy == 0. and owner.zero_point == 0.
        events.append("charges")
        return import_charges(owner, values, **kwargs)

    monkeypatch.setattr(conformer, "_select_frame_observations", publish)
    monkeypatch.setattr(_results, "_import_partial_charges", charges_after)
    output.update_mol(molecule)
    assert events == ["observations", "charges"]
    assert molecule.energy == 0. and molecule.gibbs is None
    assert molecule.capacity == 0.
    np.testing.assert_array_equal(molecule.force, forces)
    assert tuple(atom.partial_charge for atom in molecule.atoms) == (0., 0.)
    assert not molecule._source_frames and not molecule._conformer_results
