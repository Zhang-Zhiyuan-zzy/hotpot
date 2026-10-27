"""Gaussian log parsing plus an opt-in official-executable integration."""

from pathlib import Path
import shutil

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
