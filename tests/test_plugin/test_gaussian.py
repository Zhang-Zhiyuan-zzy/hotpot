"""Gaussian log parsing plus an opt-in official-executable integration."""

from pathlib import Path
import shutil
from types import SimpleNamespace
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
    assert parsed["spin"] == parsed["spin_mult"]
    assert np.isfinite(parsed["mol_orbital_energies"]).all()
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
    annotations = dict(molecule.properties)
    assert output.update_mol(molecule) is None
    assert molecule._source_frames is sources
    assert molecule.charge == parsed["charge"]
    assert molecule.spin_mult == parsed["spin_mult"]
    assert molecule.energy == pytest.approx(parsed["energy"], abs=1.e-8)
    assert molecule.gibbs == pytest.approx(parsed["gibbs"], abs=1.e-8)
    assert molecule.zero_point == pytest.approx(parsed["zero_point"], abs=1.e-9)
    assert molecule.capacity == pytest.approx(parsed["capacity"], rel=1.e-8)
    np.testing.assert_array_equal(molecule.force, parsed["force"])
    np.testing.assert_array_equal([atom.partial_charge for atom in molecule.atoms], parsed["partial_charge"])
    assert not molecule._frame_result.invalidated_by
    assert not molecule._partial_charge_result.invalidated_by
    assert molecule._partial_charge_result.payload.method == "mulliken"
    assert "spin" not in molecule.__dict__
    assert "mol_orbital_energies" not in molecule.__dict__
    assert molecule.properties == annotations
    report = output.parse()
    assert report["spin"] == parsed["spin"]
    np.testing.assert_array_equal(report["mol_orbital_energies"], parsed["mol_orbital_energies"])


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
    monkeypatch.setattr(output, "parse", lambda: attrs)
    parsed_values = dict(attrs)
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
    assert output.update_mol(molecule) is None
    assert output.parse().keys() == parsed_values.keys()
    assert all(output.parse()[name] is value for name, value in parsed_values.items())
    assert "spin" not in molecule.__dict__
    assert "mol_orbital_energies" not in molecule.__dict__
    assert molecule.properties == {}
    assert events == ["observations", "charges"]
    assert molecule.energy == 0. and molecule.gibbs is None
    assert molecule.capacity == 0.
    np.testing.assert_array_equal(molecule.force, forces)
    assert tuple(atom.partial_charge for atom in molecule.atoms) == (0., 0.)
    assert not molecule._source_frames and not molecule._conformer_results


@pytest.mark.parametrize("carrier", [hp.Molecule, hp.Atom])
def test_carriers_have_no_generic_attribute_update_method(carrier):
    assert not hasattr(carrier, "setattr")


@pytest.mark.parametrize("boundary", ["observations", "partial_charges"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_gaussian_update_rolls_back_all_imports_on_publication_failure(
        monkeypatch, boundary, error_type):
    from hotpot.cheminfo.structure import conformer, _results
    from hotpot.cheminfo.structure._dependencies import _dependency_state
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES

    molecule = hp.Molecule()
    molecule.create_atom(atomic_number=6)
    molecule.create_atom(atomic_number=8)
    molecule.add_bond(0, 1, 1.)
    molecule.coordinates = np.array([[0., 0., 0.], [1.4, 0., 0.]])
    molecule.charge = 0
    molecule.spin_mult = 1
    _ = molecule.atom_matrix  # Prepare implicit-H facts before attaching observations.
    molecule.conformer_add(molecule.coordinates, -3.)
    molecule.conformer_load(0)
    _results._import_partial_charges(
        molecule, np.array([-.2, .2]),
        producer=ATOM_SOURCE_CODES["explicit"], method="supplied-test")
    molecule.properties["label"] = "before Gaussian update"
    owners = (molecule._source_frames, molecule._conformer_results,
              molecule._frame_result, molecule._partial_charge_result)
    coordinates = molecule.coordinates
    matrix = molecule.atom_matrix
    presence, sources = molecule._atom_table.presence(), molecule._atom_table.sources()
    status, seeds = _dependency_state(molecule._atom_table)
    declarations = (molecule._charge_value, molecule._charge_present,
                    molecule._charge_source, molecule._spin_value,
                    molecule._spin_present, molecule._spin_source)
    cursor = molecule._conformers_index
    attrs = {"atomic_number": np.array([6, 8]), "coordinates": coordinates + 2.,
             "partial_charge": np.array([-.1, .1]), "energy": -5., "gibbs": None,
             "zero_point": 0., "spin": 3, "spin_mult": 3, "charge": 2,
             "force": np.ones((2, 3)), "thermo": None, "capacity": 0.,
             "mol_orbital_energies": np.array([[-1., 1.]])}
    output = GaussOut("")
    monkeypatch.setattr(output, "parse", lambda: attrs)
    parsed_values = dict(attrs)
    failure = error_type("Gaussian publication interrupted")
    owner, name = ((conformer, "_select_frame_observations")
                   if boundary == "observations" else (_results, "_import_partial_charges"))
    publish = getattr(owner, name)

    def publish_then_fail(*args, **kwargs):
        publish(*args, **kwargs)
        raise failure

    monkeypatch.setattr(owner, name, publish_then_fail)
    with pytest.raises(error_type) as caught:
        output.update_mol(molecule)
    assert caught.value is failure
    assert molecule._source_frames is owners[0]
    assert molecule._conformer_results is owners[1]
    assert molecule._frame_result is owners[2]
    assert molecule._partial_charge_result is owners[3]
    assert molecule._conformers_index == cursor
    np.testing.assert_array_equal(molecule.coordinates, coordinates)
    np.testing.assert_array_equal(molecule.atom_matrix, matrix)
    np.testing.assert_array_equal(molecule._atom_table.presence(), presence)
    np.testing.assert_array_equal(molecule._atom_table.sources(), sources)
    after_status, after_seeds = _dependency_state(molecule._atom_table)
    np.testing.assert_array_equal(after_status, status)
    assert after_seeds == seeds
    assert (molecule._charge_value, molecule._charge_present, molecule._charge_source,
            molecule._spin_value, molecule._spin_present, molecule._spin_source) == declarations
    assert molecule.energy == -3.
    assert tuple(atom.partial_charge for atom in molecule.atoms) == (-.2, .2)
    assert "spin" not in molecule.__dict__
    assert "mol_orbital_energies" not in molecule.__dict__
    assert molecule.properties == {"label": "before Gaussian update"}
    assert output.parse().keys() == parsed_values.keys()
    assert all(output.parse()[name] is value for name, value in parsed_values.items())


@pytest.mark.parametrize("thermo_facts, expected", [
    ({}, (None, None)),
    ({"freeenergy": 0., "zpve": 0.}, (0., 0.)),
    ({"freeenergy": -.25, "zpve": .125},
     (-.25 * 27.211386245988, .125 * 27.211386245988)),
], ids=["absent", "explicit-zero", "finite-hartree"])
def test_gaussian_parse_and_update_preserve_optional_thermochemistry(
        monkeypatch, thermo_facts, expected):
    """Exercise the parser with supplied cclib data, without a Gaussian engine."""
    from hotpot.plugins.qm.gaussian import gauss

    coordinates = np.array([[[0., 0., 0.], [1.4, 0., 0.]]])
    data = SimpleNamespace(atomnos=np.array([6, 8]), atomcoords=coordinates,
        atomcharges={"mulliken": np.zeros(2)}, scfenergies=np.array([-5.]),
        mult=1, charge=0, moenergies=[[-1., 1.]], **thermo_facts)
    monkeypatch.setattr(gauss.cclib, "ccopen", lambda stream: SimpleNamespace(parse=lambda: data))
    output = GaussOut("")
    parsed = output.parse()
    assert (parsed["gibbs"], parsed["zero_point"]) == expected
    assert parsed["energy"] == -5.
    assert parsed["spin"] == parsed["spin_mult"] == 1
    np.testing.assert_array_equal(parsed["mol_orbital_energies"], [[-1., 1.]])
    series = output.export_pandas_series()
    for name, value in zip(("gibbs", "zero_point"), expected):
        if value is None:
            assert np.isnan(series[name])
        else:
            assert series[name] == value
    molecule = hp.Molecule()
    molecule.create_atom(atomic_number=6)
    molecule.create_atom(atomic_number=8)
    assert output.update_mol(molecule) is None
    assert (molecule.gibbs, molecule.zero_point) == expected
    assert molecule.energy == -5.
    assert molecule.charge == 0 and molecule.spin_mult == 1
    assert "spin" not in molecule.__dict__
    assert "mol_orbital_energies" not in molecule.__dict__
    assert molecule.properties == {}
