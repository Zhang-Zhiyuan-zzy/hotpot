from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import hotpot


def test_official_gfn2_singlepoint_matches_public_result_contract() -> None:
    if os.environ.get("HOTPOT_XTB_INTEGRATION") != "1":
        pytest.skip("set HOTPOT_XTB_INTEGRATION=1 to run the official xTB test")
    executable = shutil.which("xtb")
    if executable is None:
        pytest.skip("official xTB executable is not available on PATH")

    try:
        from hotpot.plugins.xtb.contracts import GFNXTBMethod, XTBTask
        from hotpot.plugins.xtb.workflow import run_gfn_xtb
    except ModuleNotFoundError:
        pytest.xfail("Phase 9 public xTB workflow API is not implemented yet")

    mol = hotpot.read_mol("[H]C([H])([H])O[H]", "smi")
    mol.coordinates = np.asarray(
        (
            (-0.63, 0.90, 0.00),
            (0.00, 0.00, 0.00),
            (-0.63, -0.90, 0.00),
            (0.00, 0.00, 1.00),
            (1.42, 0.00, 0.00),
            (1.82, 0.72, 0.00),
        ),
        dtype=float,
    )
    report = run_gfn_xtb(
        mol,
        method=GFNXTBMethod.GFN2_XTB,
        task=XTBTask.SINGLEPOINT,
        charge=0,
        unpaired_electrons=0,
        executable=executable,
    )

    assert report.return_code == 0
    assert np.isfinite(report.energy_hartree)
    assert report.atom_order_verified is True
    assert report.coordinates_committed is False
