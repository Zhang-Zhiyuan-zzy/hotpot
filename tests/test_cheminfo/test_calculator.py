"""Regression contracts for the calculator package."""

from __future__ import annotations

import subprocess
import sys

import hotpot.cheminfo.calculator as calculator


PUBLIC_CALCULATOR_NAMES = (
    "mca",
    "extract_molecular_facts",
    "resolve_scientific_state",
    "project_state",
    "resolve_and_project_state",
    "MolecularFacts",
    "StateRequest",
    "StatePolicy",
    "ScientificOperationRequirements",
    "ScientificStateBundle",
    "BackendStateRequest",
    "BackendProjectionResult",
    "ScientificStateWorkflowResult",
)


def test_calculator_package_exposes_only_the_declared_public_objects() -> None:
    assert tuple(calculator.__all__) == PUBLIC_CALCULATOR_NAMES
    assert all(hasattr(calculator, name) for name in PUBLIC_CALCULATOR_NAMES)


def test_removed_root_calculator_module_is_not_importable() -> None:
    completed = subprocess.run(
        (sys.executable, "-c", "import hotpot.calculator"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "No module named 'hotpot.calculator'" in completed.stderr


def test_importing_calculator_does_not_import_the_mca_runtime() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import hotpot.cheminfo.calculator; "
            "assert 'hotpot.cheminfo.AImodels.mca' not in sys.modules",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
