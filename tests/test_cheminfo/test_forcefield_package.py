"""Package-boundary tests for the versioned force-field implementation."""

from __future__ import annotations

import importlib
import inspect
import pickle
import subprocess
import sys

import pytest


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
