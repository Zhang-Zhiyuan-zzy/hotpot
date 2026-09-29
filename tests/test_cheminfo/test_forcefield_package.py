"""Package-boundary tests for the canonical force-field implementation."""

from __future__ import annotations

import importlib
import inspect
import pickle
import subprocess
import sys

import numpy as np

from hotpot import read_mol


PACKAGE_NAME = "hotpot.cheminfo.forcefields"
FACADE_NAME = f"{PACKAGE_NAME}.ff"

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
    "OptimizationStoppingCriteria",
    "ForceFieldRunReport",
    "Build3DReport",
    "CandidateRejection",
    "RingUntanglingReport",
    "CoordinationBondRestorationReport",
    "ComplexBuildDiagnostics",
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


def test_package_exports_the_canonical_facade_contract():
    package = importlib.import_module(PACKAGE_NAME)
    facade = importlib.import_module(FACADE_NAME)

    assert package.__all__ == facade.__all__
    for name in package.__all__:
        assert getattr(package, name) is getattr(facade, name)


def test_facade_exposes_the_expected_public_functions():
    facade = importlib.import_module(FACADE_NAME)

    functions = tuple(
        name for name in facade.__all__ if inspect.isfunction(getattr(facade, name))
    )
    assert functions == PUBLIC_FUNCTIONS


def test_build_worker_result_is_an_internal_contract():
    facade = importlib.import_module(FACADE_NAME)
    contracts = importlib.import_module(f"{PACKAGE_NAME}.contracts")

    assert "BuildWorkerResult" not in facade.__all__
    assert not hasattr(facade, "BuildWorkerResult")
    assert contracts.BuildWorkerResult.__module__ == f"{PACKAGE_NAME}.contracts"


def test_facade_exports_trajectory_contracts_by_identity():
    facade = importlib.import_module(FACADE_NAME)
    trajectory = importlib.import_module(f"{PACKAGE_NAME}.trajectory")

    for name in (*TRAJECTORY_CONTRACTS, *TRAJECTORY_TYPE_ALIASES):
        assert name in facade.__all__
    for name in TRAJECTORY_CONTRACTS:
        assert getattr(facade, name) is getattr(trajectory, name)


def test_facade_exports_forcefield_contracts_by_identity():
    contracts = importlib.import_module(f"{PACKAGE_NAME}.contracts")
    facade = importlib.import_module(FACADE_NAME)

    for name in (*PUBLIC_DATA_CONTRACTS, *PUBLIC_EXCEPTIONS):
        contract = getattr(contracts, name)
        assert getattr(facade, name) is contract
        assert pickle.loads(pickle.dumps(contract)) is contract


def test_specialized_apis_are_reexported_without_wrapping():
    facade = importlib.import_module(FACADE_NAME)
    modules = (
        importlib.import_module(f"{PACKAGE_NAME}.acceptance"),
        importlib.import_module(f"{PACKAGE_NAME}.coordination"),
        importlib.import_module(f"{PACKAGE_NAME}.coordinates"),
        importlib.import_module(f"{PACKAGE_NAME}.topology"),
        importlib.import_module(f"{PACKAGE_NAME}.workflows"),
    )

    for module in modules:
        for name in module.__all__:
            exported = getattr(module, name)
            assert getattr(facade, name) is exported
            assert pickle.loads(pickle.dumps(exported)) is exported


def test_removed_compatibility_modules_are_not_importable():
    for module_name in ("ff39", "utils", "utils39"):
        try:
            importlib.import_module(f"{PACKAGE_NAME}.{module_name}")
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"obsolete force-field module remains importable: {module_name}")


def test_fresh_import_loads_only_the_canonical_facade():
    script = f"""
import importlib
import sys

package = importlib.import_module({PACKAGE_NAME!r})
facade = importlib.import_module({FACADE_NAME!r})

assert {FACADE_NAME!r} in sys.modules
for obsolete in ('ff39', 'utils', 'utils39'):
    assert {PACKAGE_NAME!r} + '.' + obsolete not in sys.modules
assert package.__all__ == facade.__all__
for name in package.__all__:
    assert getattr(package, name) is getattr(facade, name)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_public_forcefield_functions_are_pickleable():
    facade = importlib.import_module(FACADE_NAME)

    for function_name in PUBLIC_FUNCTIONS:
        function = getattr(facade, function_name)
        assert pickle.loads(pickle.dumps(function)) is function


def test_spawn_worker_targets_are_pickleable():
    workers = importlib.import_module(f"{PACKAGE_NAME}.workers")

    for worker_name in ("_build_ligand_proxies_worker", "_seeded_ob_build_worker"):
        worker = getattr(workers, worker_name)
        assert pickle.loads(pickle.dumps(worker)) is worker


def _seeded_coordinates(seed):
    molecule = read_mol("CCCCCCOC(=O)NCCCCC", "smi")
    package = importlib.import_module(PACKAGE_NAME)
    package.build3d(molecule, seed=seed, timeout=30.0)
    return molecule.coordinates


def test_seeded_build_is_repeatable_and_seed_sensitive():
    first = _seeded_coordinates(101)
    repeated = _seeded_coordinates(101)
    different = _seeded_coordinates(103)

    np.testing.assert_array_equal(first, repeated)
    assert not np.allclose(first, different)
