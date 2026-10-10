"""Regression contracts for the calculator package."""

from __future__ import annotations

import subprocess
import sys

import hotpot
import hotpot.cheminfo.calculator as calculator
import pytest


PUBLIC_CALCULATOR_NAMES = (
    "Calculator",
    "MolChargeCalculator",
    "formal_charge",
    "infer_charge",
    "infer_lowest_spin",
    "mca",
    "resolve_electronic_state",
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


def test_legacy_molecular_charge_calculator_has_deterministic_fixtures() -> None:
    charge_calculator = calculator.MolChargeCalculator()

    assert charge_calculator(hotpot.read_mol("CCO", "smi")) == 0
    assert charge_calculator(hotpot.read_mol("[NH4+]", "smi")) == 1
    assert charge_calculator(hotpot.read_mol("CC(=O)[O-]", "smi")) == 0
    assert charge_calculator(hotpot.read_mol("O[Te](F)(F)(F)(F)F", "smi")) == -1

    with pytest.raises(ValueError, match="Unknown molecule fragment"):
        charge_calculator(hotpot.read_mol("ClCl", "smi"))


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
