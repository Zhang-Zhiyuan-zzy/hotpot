"""Regression contracts for the pre-split calculator façade."""

from __future__ import annotations

import subprocess
import sys

import hotpot
import hotpot.calculator as public_calculator
import hotpot.cheminfo.calculator as calculator_implementation
import pytest


PUBLIC_CALCULATOR_NAMES = (
    "Calculator",
    "MolChargeCalculator",
    "formal_charge",
    "mca",
)


def test_root_and_cheminfo_facades_share_the_four_public_objects() -> None:
    assert tuple(public_calculator.__all__) == PUBLIC_CALCULATOR_NAMES
    for name in PUBLIC_CALCULATOR_NAMES:
        assert getattr(public_calculator, name) is getattr(
            calculator_implementation,
            name,
        )


def test_legacy_molecular_charge_calculator_has_deterministic_fixtures() -> None:
    calculator = public_calculator.MolChargeCalculator()

    assert calculator(hotpot.read_mol("CCO", "smi")) == 0
    assert calculator(hotpot.read_mol("[NH4+]", "smi")) == 1
    assert calculator(hotpot.read_mol("CC(=O)[O-]", "smi")) == 0
    assert calculator(hotpot.read_mol("O[Te](F)(F)(F)(F)F", "smi")) == -1

    with pytest.raises(ValueError, match="Unknown molecule fragment"):
        calculator(hotpot.read_mol("ClCl", "smi"))


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
