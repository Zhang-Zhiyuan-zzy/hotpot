from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo import forcefields as ff
from hotpot.cheminfo.core import Molecule


def _bare_molecule():
    return object.__new__(Molecule)


def test_molecule_build3d_is_a_single_forcefield_facade(monkeypatch):
    molecule = _bare_molecule()
    expected = object()
    quality_thresholds = {"standard_minimum_distance": 0.4}
    calls = []

    def fake_build_and_optimize(current, **options):
        calls.append((current, options))
        return expected

    monkeypatch.setattr(ff, "build_and_optimize", fake_build_and_optimize)

    result = molecule.build3d(
        forcefield="GAFF",
        epochs=3,
        steps_per_epoch=7,
        add_hydrogens=False,
        quality_level="basic",
        quality_thresholds=quality_thresholds,
        seed=19,
        timeout=2.5,
        candidate_count=2,
        max_attempts=9,
    )

    assert result is expected
    assert len(calls) == 1
    current, options = calls[0]
    assert current is molecule
    assert options["forcefield"] == "GAFF"
    assert options["epochs"] == 3
    assert options["steps_per_epoch"] == 7
    assert options["add_hydrogens"] is False
    assert options["quality_level"] == "basic"
    assert options["quality_thresholds"] is quality_thresholds
    assert options["seed"] == 19
    assert options["timeout"] == 2.5
    assert options["candidate_count"] == 2
    assert options["max_attempts"] == 9


def test_molecule_optimize_is_a_single_forcefield_facade(monkeypatch):
    molecule = _bare_molecule()
    expected = object()
    quality_thresholds = {"standard_minimum_distance": 0.45}
    calls = []

    def fake_auto_optimize(current, **options):
        calls.append((current, options))
        return expected

    monkeypatch.setattr(ff, "auto_optimize", fake_auto_optimize)

    result = molecule.optimize(
        forcefield="MMFF94s",
        epochs=4,
        steps_per_epoch=11,
        quality_level="strict",
        quality_thresholds=quality_thresholds,
        seed=23,
    )

    assert result is expected
    assert len(calls) == 1
    current, options = calls[0]
    assert current is molecule
    assert options["forcefield"] == "MMFF94s"
    assert options["epochs"] == 4
    assert options["steps_per_epoch"] == 11
    assert options["quality_level"] == "strict"
    assert options["quality_thresholds"] is quality_thresholds
    assert options["seed"] == 23


def test_legacy_molecule_forcefield_entrypoints_are_removed():
    assert not hasattr(Molecule, "complexes_build_optimize_")
    assert not hasattr(Molecule, "optimize_complexes")
    assert not hasattr(ff, "OBBuilder")
    assert not hasattr(ff, "ForceFields")


@pytest.mark.parametrize("add_hydrogens", (False, True))
def test_build3d_captures_the_requested_hydrogen_policy(
    monkeypatch,
    add_hydrogens,
):
    molecule = read_mol("CC", "smi")
    captured = []

    def capture(current, **options):
        captured.append(options["allow_added_hydrogens"])
        return object()

    monkeypatch.setattr(ff, "_capture_workflow_topology", capture)
    monkeypatch.setattr(
        ff,
        "_hydrogenated_working_copy",
        lambda current, **options: current,
    )
    monkeypatch.setattr(ff, "ob_build", lambda current: None)
    monkeypatch.setattr(
        ff.geo,
        "evaluate_geometry_quality",
        lambda *args, **options: SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(ff, "_commit_working_copy", lambda *args: None)

    ff.build3d(molecule, add_hydrogens=add_hydrogens)

    assert captured == [add_hydrogens]


def test_complexes_build_translates_legacy_options_once(monkeypatch):
    molecule = object()
    captured = {}

    def implementation(current, forcefield, **options):
        captured.update(options)
        return current, forcefield

    monkeypatch.setattr(ff, "_complexes_build_impl", implementation)

    result = ff.complexes_build(
        molecule,
        "MMFF94",
        steps=7,
        step_size=11,
        perturb_steps=3,
        save_screenshot=True,
        build_times=2,
        init_opt_steps=13,
        second_opt_steps=17,
        min_energy_opt_steps=19,
        increasing_Vdw=True,
        Vdw_cutoff_start=1.5,
        Vdw_cutoff_end=9.5,
    )

    assert result == (molecule, "MMFF94")
    assert captured == {
        "epochs": 7,
        "steps_per_epoch": 11,
        "perturb_interval": 3,
        "save_movie": True,
        "candidate_count": 2,
        "candidate_warmup_steps": 13,
        "candidate_score_steps": 17,
        "best_candidate_refine_steps": 19,
        "increasing_vdw": True,
        "vdw_cutoff_start": 1.5,
        "vdw_cutoff_end": 9.5,
    }


def test_complexes_build_rejects_conflicting_legacy_and_current_options():
    with pytest.raises(TypeError, match="Conflicting values"):
        ff.complexes_build(
            object(),
            epochs=3,
            steps=4,
        )


def test_build_and_optimize_dispatches_complex_once(monkeypatch):
    molecule = SimpleNamespace(has_metal=True)
    expected = object()
    calls = []

    def fake_complexes_build(current, forcefield, **options):
        calls.append((current, forcefield, options))
        return expected

    monkeypatch.setattr(ff, "complexes_build", fake_complexes_build)

    result = ff.build_and_optimize(
        molecule,
        "MMFF94s",
        epochs=2,
        steps_per_epoch=13,
        timeout=4.0,
        seed=29,
    )

    assert result is expected
    assert len(calls) == 1
    current, forcefield, options = calls[0]
    assert current is molecule
    assert forcefield == "MMFF94s"
    assert options["epochs"] == 2
    assert options["steps_per_epoch"] == 13
    assert options["timeout"] == 4.0
    assert options["seed"] == 29


def test_build_and_optimize_organic_builds_then_optimizes_once(monkeypatch):
    molecule = SimpleNamespace(has_metal=False)
    working = SimpleNamespace(has_metal=False)
    expected = object()
    calls = []

    monkeypatch.setattr(
        ff,
        "_hydrogenated_working_copy",
        lambda current, *, add_hydrogens: working,
    )
    monkeypatch.setattr(
        ff,
        "build3d",
        lambda current, **options: calls.append(("build", current, options)),
    )

    def fake_optimize(current, forcefield, **options):
        calls.append(("optimize", current, forcefield, options))
        return expected

    monkeypatch.setattr(ff, "optimize", fake_optimize)
    monkeypatch.setattr(
        ff,
        "_commit_working_copy",
        lambda current, completed: calls.append(("commit", current, completed)),
    )

    result = ff.build_and_optimize(
        molecule,
        "GAFF",
        epochs=3,
        steps_per_epoch=17,
        add_hydrogens=True,
    )

    assert result is expected
    assert [call[0] for call in calls] == ["build", "optimize", "commit"]
    assert calls[0][1] is working
    assert calls[0][2]["add_hydrogens"] is True
    assert calls[1][1] is working
    assert calls[1][2] == "GAFF"
    assert calls[1][3]["add_hydrogens"] is False
    assert calls[1][3]["epochs"] == 3
    assert calls[1][3]["steps_per_epoch"] == 17
    assert calls[2][1:] == (molecule, working)


@pytest.mark.parametrize(
    ("has_metal", "expected_function", "expected_forcefield"),
    ((False, "ordinary", None), (True, "complex", None)),
)
def test_auto_optimize_dispatches_by_molecule_type(
    monkeypatch,
    has_metal,
    expected_function,
    expected_forcefield,
):
    molecule = SimpleNamespace(has_metal=has_metal)
    calls = []
    expected = object()

    def fake_ordinary(current, forcefield, **options):
        calls.append(("ordinary", current, forcefield, options))
        return expected

    def fake_complex(current, forcefield, **options):
        calls.append(("complex", current, forcefield, options))
        return expected

    monkeypatch.setattr(ff, "optimize", fake_ordinary)
    monkeypatch.setattr(ff, "optimize_complex", fake_complex)

    result = ff.auto_optimize(
        molecule,
        epochs=5,
        steps_per_epoch=19,
        seed=31,
    )

    assert result is expected
    assert len(calls) == 1
    function, current, forcefield, options = calls[0]
    assert function == expected_function
    assert current is molecule
    assert forcefield == expected_forcefield
    assert options["epochs"] == 5
    assert options["steps_per_epoch"] == 19
    assert options["seed"] == 31


def test_optimize_on_metal_molecule_does_not_build_ligand_proxies(monkeypatch):
    molecule = SimpleNamespace(has_metal=True)
    working = SimpleNamespace(has_metal=True)
    expected = object()
    calls = []

    monkeypatch.setattr(
        ff,
        "_capture_workflow_topology",
        lambda current, **options: "topology",
    )
    monkeypatch.setattr(
        ff,
        "_hydrogenated_working_copy",
        lambda current, *, add_hydrogens, seed=None: working,
    )
    monkeypatch.setattr(
        ff,
        "_build_complex_working",
        lambda *args, **kwargs: pytest.fail("ordinary optimize built ligand proxies"),
    )

    def fake_run(current, **options):
        calls.append((current, options))
        return expected

    monkeypatch.setattr(ff, "_run_optimizer_on_working", fake_run)
    monkeypatch.setattr(
        ff,
        "_commit_working_copy",
        lambda current, completed: calls.append((current, completed)),
    )

    result = ff.optimize(
        molecule,
        "UFF",
        epochs=2,
        steps_per_epoch=7,
        add_hydrogens=False,
    )

    assert result is expected
    assert calls[0][0] is working
    assert calls[0][1]["requested_forcefield"] == "UFF"
    assert calls[0][1]["effective_forcefield"] == "UFF"
    assert calls[1] == (molecule, working)


def test_build3d_only_embeds_coordinates(monkeypatch):
    molecule = read_mol("c1ccccc1", "smi")
    calls = []

    monkeypatch.setattr(
        ff,
        "_capture_workflow_topology",
        lambda current, **options: "topology",
    )
    monkeypatch.setattr(
        ff,
        "_hydrogenated_working_copy",
        lambda current, *, add_hydrogens, seed=None: current,
    )
    monkeypatch.setattr(ff, "ob_build", lambda current: calls.append(("build", current)))
    monkeypatch.setattr(
        ff.geo,
        "evaluate_geometry_quality",
        lambda *args, **kwargs: SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(
        ff,
        "_run_optimizer_on_working",
        lambda *args, **kwargs: pytest.fail("build3d invoked optimization"),
    )
    monkeypatch.setattr(
        ff,
        "_commit_working_copy",
        lambda current, completed: calls.append(("commit", current, completed)),
    )

    report = ff.build3d(molecule, add_hydrogens=False, seed=37)

    assert calls == [("build", molecule), ("commit", molecule, molecule)]
    assert report.atom_count == len(molecule.atoms)


def test_perturb_only_changes_coordinates():
    original_coordinates = np.arange(9, dtype=float).reshape(3, 3)
    metadata = {"name": "probe"}
    molecule = SimpleNamespace(
        coordinates=original_coordinates.copy(),
        metadata=metadata,
        charge=2,
    )

    result = ff.perturb(molecule, sigma=0.1, seed=41)

    assert result is molecule.coordinates
    assert not np.array_equal(molecule.coordinates, original_coordinates)
    assert molecule.metadata is metadata
    assert molecule.charge == 2
    assert set(vars(molecule)) == {"coordinates", "metadata", "charge"}


@pytest.mark.parametrize("forcefield", ("UFF", "MMFF94s"))
def test_ordinary_benzene_forcefield_request_reaches_optimizer_unchanged(
    monkeypatch,
    forcefield,
):
    molecule = read_mol("c1ccccc1", "smi")
    calls = []
    expected = object()

    monkeypatch.setattr(
        ff,
        "_capture_workflow_topology",
        lambda current, **options: "topology",
    )
    monkeypatch.setattr(
        ff,
        "_hydrogenated_working_copy",
        lambda current, *, add_hydrogens, seed=None: current,
    )

    def fake_run(current, **options):
        calls.append((current, options))
        return expected

    monkeypatch.setattr(ff, "_run_optimizer_on_working", fake_run)
    monkeypatch.setattr(ff, "_commit_working_copy", lambda current, completed: None)

    result = ff.optimize(molecule, forcefield, add_hydrogens=False)

    assert result is expected
    assert calls[0][0] is molecule
    assert calls[0][1]["requested_forcefield"] == forcefield
    assert calls[0][1]["effective_forcefield"] == forcefield


def test_organic_combined_workflow_requests_hydrogen_addition_once(monkeypatch):
    molecule = SimpleNamespace(has_metal=False, atoms=(), hydrogens=())
    hydrogen_requests = []
    expected = object()

    monkeypatch.setattr(
        ff,
        "_capture_workflow_topology",
        lambda current, **options: "topology",
    )

    def fake_working_copy(current, *, add_hydrogens, seed=None):
        hydrogen_requests.append(add_hydrogens)
        return SimpleNamespace(has_metal=False, atoms=(), hydrogens=())

    monkeypatch.setattr(ff, "_hydrogenated_working_copy", fake_working_copy)
    monkeypatch.setattr(ff, "ob_build", lambda current: None)
    monkeypatch.setattr(
        ff.geo,
        "evaluate_geometry_quality",
        lambda *args, **kwargs: SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(ff, "_run_optimizer_on_working", lambda *args, **kwargs: expected)
    monkeypatch.setattr(ff, "_commit_working_copy", lambda current, completed: None)

    result = ff.build_and_optimize(molecule, add_hydrogens=True)

    assert result is expected
    assert hydrogen_requests.count(True) == 1
    assert hydrogen_requests == [False, True, False]


@pytest.mark.parametrize(
    "public_function",
    (
        ff.build_and_optimize,
        ff.auto_optimize,
        ff.build3d,
        ff.optimize,
        ff.perturb,
        ff.build_complex3d,
        ff.optimize_complex,
        ff.complexes_build,
        ff.prepare_coordination_geometry,
    ),
)
def test_public_forcefield_functions_reject_an_unknown_option(public_function):
    with pytest.raises(TypeError):
        public_function(object(), misspelled_option=True)
