"""Package-boundary tests for the versioned force-field implementation."""

from __future__ import annotations

import importlib
import inspect
import pickle
import subprocess
import sys

import numpy as np
import pytest

from hotpot import read_mol


PACKAGE_NAME = "hotpot.cheminfo.forcefields"
MODERN_FACADE_NAME = f"{PACKAGE_NAME}.ff"
PYTHON39_FACADE_NAME = f"{PACKAGE_NAME}.ff39"

PUBLIC_FUNCTIONS = (
    "capture_topology",
    "evaluate_structure_acceptance",
    "is_structure_accepted",
    "perturb",
    "collect_coordination_environments",
    "prepare_coordination_geometry",
    "build3d",
    "optimize",
    "build_complex3d",
    "optimize_complex",
    "complexes_build",
    "build_and_optimize",
    "auto_optimize",
)

SHARED_FUNCTIONS = (
    "capture_topology",
    "evaluate_structure_acceptance",
    "is_structure_accepted",
    "perturb",
    "collect_coordination_environments",
    "prepare_coordination_geometry",
    "optimize",
    "optimize_complex",
    "auto_optimize",
)

WORKER_ADAPTED_FUNCTIONS = (
    "build3d",
    "build_complex3d",
    "complexes_build",
    "build_and_optimize",
)

TRAJECTORY_CONTRACTS = (
    "TrajectoryStart",
    "TrajectoryStage",
    "TrajectoryEvent",
    "AtomIdentity",
    "BondTopology",
    "BondTopologyRevision",
    "RingFrameEvidence",
    "CoordinationFrameEvidence",
    "OptimizationFrameEvidence",
    "ForceFieldFrame",
    "ForceFieldTrajectory",
    "ForceFieldTrajectoryArchive",
)

TRAJECTORY_TYPE_ALIASES = (
    "TrajectoryPath",
    "FrameEvidence",
)

PUBLIC_DATA_CONTRACTS = (
    "ForceFieldRunReport",
    "Build3DReport",
    "CandidateRejection",
    "RingUntanglingReport",
    "CoordinationBondRestorationReport",
    "ComplexBuildDiagnostics",
    "BuildWorkerResult",
    "ForceFieldWorkflowReport",
    "BuildAndOptimizeReport",
    "ComplexBuildReport",
    "ForceFieldSetupReport",
    "AcceptanceCheck",
    "StructureAcceptanceThresholds",
    "ForceFieldValidationReport",
    "CoordinationEnvironment",
    "CoordinationGeometryCandidate",
    "CoordinationGeometryResult",
)

PUBLIC_EXCEPTIONS = (
    "ForceFieldError",
    "ForceFieldSetupError",
    "BuildWorkerError",
    "BuildTimeoutError",
    "ComplexBuildError",
    "ComplexBuildWarning",
    "ComplexBuildWorkerError",
    "ComplexBuildTimeoutError",
    "GeometryQualityError",
    "GeometryQualityWarning",
)

BACKEND_COMPATIBILITY_EXPORTS = (
    "_CandidateOptimizationResult",
    "_energy_factor_to_kj",
    "_find_forcefield_prototype",
    "_forcefield_energy_in_kj",
    "_get_forcefield",
    "_make_constraints",
    "_ob_build",
    "_resolve_complex_forcefield",
    "_resolve_organic_forcefield",
    "_seed_openbabel_random",
    "_serialized_forcefield_call",
    "_setup_forcefield_backend",
    "_single_ob_optimization",
)

WORKING_COPY_COMPATIBILITY_EXPORTS = (
    "_commit_working_copy",
    "_copy_molecule_metadata",
    "_hydrogenated_working_copy",
    "_make_worker_mol",
)


def _selected_facade_name() -> str:
    if sys.version_info[:2] == (3, 9):
        return PYTHON39_FACADE_NAME
    return MODERN_FACADE_NAME


def test_package_exports_the_selected_facade_contract():
    package = importlib.import_module(PACKAGE_NAME)
    selected_facade = importlib.import_module(_selected_facade_name())

    assert package.__all__ == selected_facade.__all__
    for name in package.__all__:
        assert getattr(package, name) is getattr(selected_facade, name)


def test_facades_expose_the_same_public_symbols_and_function_signatures():
    modern_facade = importlib.import_module(MODERN_FACADE_NAME)
    python39_facade = importlib.import_module(PYTHON39_FACADE_NAME)

    assert modern_facade.__all__ == python39_facade.__all__

    modern_functions = tuple(
        name
        for name in modern_facade.__all__
        if inspect.isfunction(getattr(modern_facade, name))
    )
    python39_functions = tuple(
        name
        for name in python39_facade.__all__
        if inspect.isfunction(getattr(python39_facade, name))
    )
    assert modern_functions == PUBLIC_FUNCTIONS
    assert python39_functions == PUBLIC_FUNCTIONS

    for name in PUBLIC_FUNCTIONS:
        assert inspect.signature(getattr(modern_facade, name)) == inspect.signature(
            getattr(python39_facade, name)
        )


def test_facades_export_shared_trajectory_contracts_by_identity():
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")
    trajectory = importlib.import_module(f"{PACKAGE_NAME}.trajectory")
    modern = importlib.import_module(MODERN_FACADE_NAME)
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)

    for name in (*TRAJECTORY_CONTRACTS, *TRAJECTORY_TYPE_ALIASES):
        assert name in shared.__all__
        assert getattr(modern, name) is getattr(shared, name)
        assert getattr(python39, name) is getattr(shared, name)
    for name in TRAJECTORY_CONTRACTS:
        assert getattr(shared, name) is getattr(trajectory, name)


def test_facades_export_shared_forcefield_contracts_by_identity():
    contracts = importlib.import_module(f"{PACKAGE_NAME}.contracts")
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")
    modern = importlib.import_module(MODERN_FACADE_NAME)
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)

    for name in (*PUBLIC_DATA_CONTRACTS, *PUBLIC_EXCEPTIONS):
        contract = getattr(shared, name)
        assert getattr(modern, name) is contract
        assert getattr(python39, name) is contract
        assert getattr(contracts, name) is contract
        assert pickle.loads(pickle.dumps(contract)) is contract


@pytest.mark.parametrize(
    "name",
    (*PUBLIC_DATA_CONTRACTS, *PUBLIC_EXCEPTIONS),
)
def test_legacy_utils_pickle_paths_resolve_to_canonical_contracts(name):
    contracts = importlib.import_module(f"{PACKAGE_NAME}.contracts")
    legacy_global = f"c{PACKAGE_NAME}.utils\n{name}\n.".encode("ascii")

    assert pickle.loads(legacy_global) is getattr(contracts, name)


def test_coordinate_api_is_reexported_without_wrapping():
    coordinates = importlib.import_module(f"{PACKAGE_NAME}.coordinates")
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")
    modern = importlib.import_module(MODERN_FACADE_NAME)
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)

    assert shared.perturb is coordinates.perturb
    assert modern.perturb is coordinates.perturb
    assert python39.perturb is coordinates.perturb
    assert pickle.loads(pickle.dumps(coordinates.perturb)) is coordinates.perturb


def test_topology_api_is_reexported_without_wrapping():
    topology = importlib.import_module(f"{PACKAGE_NAME}.topology")
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")
    modern = importlib.import_module(MODERN_FACADE_NAME)
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)

    for name in (*topology.__all__,):
        exported = getattr(topology, name)
        assert getattr(shared, name) is exported
        assert getattr(modern, name) is exported
        assert getattr(python39, name) is exported
        assert pickle.loads(pickle.dumps(exported)) is exported


def test_coordination_api_is_reexported_without_wrapping():
    coordination = importlib.import_module(f"{PACKAGE_NAME}.coordination")
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")
    modern = importlib.import_module(MODERN_FACADE_NAME)
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)

    for name in coordination.__all__:
        exported = getattr(coordination, name)
        assert getattr(shared, name) is exported
        assert getattr(modern, name) is exported
        assert getattr(python39, name) is exported
        assert pickle.loads(pickle.dumps(exported)) is exported


def test_acceptance_api_is_reexported_without_wrapping():
    acceptance = importlib.import_module(f"{PACKAGE_NAME}.acceptance")
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")
    modern = importlib.import_module(MODERN_FACADE_NAME)
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)

    for name in acceptance.__all__:
        exported = getattr(acceptance, name)
        assert getattr(shared, name) is exported
        assert getattr(modern, name) is exported
        assert getattr(python39, name) is exported
        assert pickle.loads(pickle.dumps(exported)) is exported


def test_backend_compatibility_names_are_reexported_without_wrapping():
    backend = importlib.import_module(f"{PACKAGE_NAME}.backend")
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")

    for name in BACKEND_COMPATIBILITY_EXPORTS:
        assert getattr(shared, name) is getattr(backend, name)

    assert shared._WORKER_LIFECYCLE_LOCK is backend._WORKER_LIFECYCLE_LOCK
    assert shared._OPENBABEL_FORCEFIELD_LOCK is backend._OPENBABEL_FORCEFIELD_LOCK

    for name in ("_CandidateOptimizationResult", "_get_forcefield", "_ob_build"):
        exported = getattr(backend, name)
        assert pickle.loads(pickle.dumps(exported)) is exported


def test_working_copy_helpers_are_reexported_without_wrapping():
    working_copy = importlib.import_module(f"{PACKAGE_NAME}.working_copy")
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")

    for name in WORKING_COPY_COMPATIBILITY_EXPORTS:
        assert getattr(shared, name) is getattr(working_copy, name)


def test_facades_only_wrap_worker_adapted_workflows():
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")
    modern = importlib.import_module(MODERN_FACADE_NAME)
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)

    for name in SHARED_FUNCTIONS:
        assert getattr(modern, name) is getattr(shared, name)
        assert getattr(python39, name) is getattr(shared, name)

    for name in WORKER_ADAPTED_FUNCTIONS:
        assert getattr(modern, name) is getattr(shared, name)
        assert getattr(python39, name) is not getattr(shared, name)


def test_worker_adapted_facades_share_canonical_documentation():
    modern = importlib.import_module(MODERN_FACADE_NAME)
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)

    for name in WORKER_ADAPTED_FUNCTIONS:
        assert getattr(modern, name).__doc__
        assert getattr(python39, name).__doc__ == getattr(modern, name).__doc__


@pytest.mark.parametrize("facade_name", (MODERN_FACADE_NAME, PYTHON39_FACADE_NAME))
@pytest.mark.parametrize("function_name", PUBLIC_FUNCTIONS)
def test_public_forcefield_functions_are_pickleable(facade_name, function_name):
    facade = importlib.import_module(facade_name)
    function = getattr(facade, function_name)

    assert pickle.loads(pickle.dumps(function)) is function


def test_python39_build3d_injects_the_legacy_seeded_worker(monkeypatch):
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")
    legacy = importlib.import_module(f"{PACKAGE_NAME}.utils39")
    expected = object()
    received = {}

    def build_workflow(molecule, **options):
        received.update(options)
        return expected

    monkeypatch.setattr(shared, "_build3d_workflow", build_workflow)

    assert python39.build3d(object(), seed=17) is expected
    assert received["worker_target"] is legacy._seeded_ob_build_worker


@pytest.mark.parametrize(
    ("function_name", "workflow_name", "worker_options"),
    (
        (
            "build_complex3d",
            "_build_complex3d_workflow",
            {"worker_target": "_build_ligand_proxies_worker"},
        ),
        (
            "complexes_build",
            "_complexes_build_workflow",
            {"worker_target": "_build_ligand_proxies_worker"},
        ),
        (
            "build_and_optimize",
            "_build_and_optimize_workflow",
            {
                "seeded_build_worker": "_seeded_ob_build_worker",
                "complex_build_worker": "_build_ligand_proxies_worker",
            },
        ),
    ),
)
def test_python39_complex_workflows_inject_legacy_workers(
    monkeypatch,
    function_name,
    workflow_name,
    worker_options,
):
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")
    legacy = importlib.import_module(f"{PACKAGE_NAME}.utils39")
    expected = object()
    received = {}

    def workflow(molecule, forcefield, **options):
        received.update(options)
        return expected

    monkeypatch.setattr(shared, workflow_name, workflow)

    assert getattr(python39, function_name)(object()) is expected
    for option_name, worker_name in worker_options.items():
        assert received[option_name] is getattr(legacy, worker_name)


@pytest.mark.parametrize(
    ("function_name", "workflow_name", "trajectory_start"),
    (
        (
            "build_complex3d",
            "_build_complex3d_workflow",
            "coordination_restoration",
        ),
        (
            "complexes_build",
            "_complexes_build_workflow",
            "coordination_restoration",
        ),
        ("build_and_optimize", "_build_and_optimize_workflow", None),
    ),
)
def test_python39_complex_workflows_forward_trajectory_options(
    monkeypatch,
    function_name,
    workflow_name,
    trajectory_start,
    tmp_path,
):
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)
    shared = importlib.import_module(f"{PACKAGE_NAME}.utils")
    expected = object()
    received = {}

    def workflow(molecule, forcefield, **options):
        received.update(options)
        return expected

    monkeypatch.setattr(shared, workflow_name, workflow)

    start = (
        None
        if trajectory_start is None
        else python39.TrajectoryStart(trajectory_start)
    )
    assert getattr(python39, function_name)(
        object(),
        trajectory_start=start,
        trajectory_path=tmp_path,
    ) is expected
    assert received["trajectory_start"] is start
    assert received["trajectory_path"] is tmp_path


def test_fresh_import_selects_only_the_runtime_facade():
    expected_facade = _selected_facade_name()
    unselected_facade = (
        MODERN_FACADE_NAME
        if expected_facade == PYTHON39_FACADE_NAME
        else PYTHON39_FACADE_NAME
    )
    script = f"""
import importlib
import sys

package = importlib.import_module({PACKAGE_NAME!r})
expected_facade = importlib.import_module({expected_facade!r})

assert {expected_facade!r} in sys.modules
assert {unselected_facade!r} not in sys.modules
assert package.__all__ == expected_facade.__all__
for name in package.__all__:
    assert getattr(package, name) is getattr(expected_facade, name)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="The modern import-isolation contract applies to Python 3.10+.",
)
def test_modern_import_does_not_load_python39_or_legacy_ctypes_adapter():
    script = f"""
import importlib
import sys

importlib.import_module({PACKAGE_NAME!r})
utils = importlib.import_module({PACKAGE_NAME + '.utils'!r})

assert {PYTHON39_FACADE_NAME!r} not in sys.modules
assert {PACKAGE_NAME + '.utils39'!r} not in sys.modules
assert not hasattr(utils, 'ctypes')
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("module_name", "worker_name"),
    (
        (f"{PACKAGE_NAME}.utils", "_build_ligand_proxies_worker"),
        (f"{PACKAGE_NAME}.utils", "_seeded_ob_build_worker"),
        (f"{PACKAGE_NAME}.utils39", "_build_ligand_proxies_worker"),
        (f"{PACKAGE_NAME}.utils39", "_seeded_ob_build_worker"),
    ),
)
def test_spawn_worker_targets_are_pickleable(module_name, worker_name):
    module = importlib.import_module(module_name)
    worker = getattr(module, worker_name)

    restored_worker = pickle.loads(pickle.dumps(worker))

    assert restored_worker is worker


def _seeded_coordinates(seed):
    molecule = read_mol("CCCCCCOC(=O)NCCCCC", "smi")
    package = importlib.import_module(PACKAGE_NAME)
    package.build3d(molecule, seed=seed, timeout=30.0)
    return molecule.coordinates


def test_selected_seed_adapter_is_repeatable_and_seed_sensitive():
    first = _seeded_coordinates(101)
    repeated = _seeded_coordinates(101)
    different = _seeded_coordinates(103)

    np.testing.assert_array_equal(first, repeated)
    assert not np.allclose(first, different)


def test_facades_have_matching_unseeded_behavior():
    modern = importlib.import_module(MODERN_FACADE_NAME)
    python39 = importlib.import_module(PYTHON39_FACADE_NAME)
    modern_mol = read_mol("CCO", "smi")
    python39_mol = read_mol("CCO", "smi")

    modern_build = modern.build3d(modern_mol, timeout=30.0)
    python39_build = python39.build3d(python39_mol, timeout=30.0)
    modern_run = modern.optimize(
        modern_mol,
        epochs=1,
        steps_per_epoch=1,
        add_hydrogens=False,
        quality_level="off",
    )
    python39_run = python39.optimize(
        python39_mol,
        epochs=1,
        steps_per_epoch=1,
        add_hydrogens=False,
        quality_level="off",
    )

    assert type(modern_build) is type(python39_build)
    assert modern_build.added_hydrogen_count == python39_build.added_hydrogen_count
    assert modern_build.quality_report.passed == python39_build.quality_report.passed
    assert type(modern_run) is type(python39_run)
    assert modern_run.energy_unit == python39_run.energy_unit == "kJ/mol"
    assert [atom.atomic_number for atom in modern_mol.atoms] == [
        atom.atomic_number for atom in python39_mol.atoms
    ]
    assert len(modern_mol.bonds) == len(python39_mol.bonds)

    messages = []
    for facade in (modern, python39):
        with pytest.raises(ValueError) as error:
            facade.build_complex3d(read_mol("[Zn].N", "smi"))
        messages.append(str(error.value))
    assert messages[0] == messages[1]
