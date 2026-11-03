"""The relocated MCA facade is explicit and lazy; calculator packages retire."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hotpot.cheminfo.operations.inference import mca as inference


PUBLIC_INFERENCE_NAMES = (
    "mca", "AtomPrediction", "MCACpuPredictor", "MCAGpuOutOfMemoryError",
    "MCAGpuPredictor", "MoleculePrediction", "SitePrediction", "predict_mca",
    "predict_mca_cpu", "predict_mca_gpu",
)


def test_inference_package_exposes_only_the_declared_public_objects() -> None:
    assert tuple(inference.__all__) == PUBLIC_INFERENCE_NAMES
    assert all(hasattr(inference, name) for name in PUBLIC_INFERENCE_NAMES)


@pytest.mark.parametrize("module", ("hotpot.calculator", "hotpot.cheminfo.calculator"))
def test_removed_calculator_modules_are_not_importable(module: str) -> None:
    completed = subprocess.run(
        (sys.executable, "-c", f"import {module}"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert f"No module named '{module}'" in completed.stderr


def test_importing_inference_facade_does_not_import_the_mca_runtime() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import hotpot.cheminfo.operations.inference.mca; "
            "assert 'hotpot.cheminfo.operations.inference.mca.api' not in sys.modules; "
            "assert 'hotpot.cheminfo.adapters.inference.runtime' not in sys.modules",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
