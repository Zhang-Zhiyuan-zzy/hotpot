"""Opt-in parity checks against an installed official xTB executable."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import hotpot
from hotpot.plugins.xtb.adapter import prepare_xtb_input
from hotpot.plugins.xtb.contracts import GFNXTBMethod, XTBTask
from hotpot.plugins.xtb.workflow import run_gfn_xtb, run_gfnff


_GFNFF_ENERGY = re.compile(
    r'"total energy"\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)'
)


def _official_executable() -> Path:
    if os.environ.get("HOTPOT_XTB_INTEGRATION") != "1":
        pytest.skip("set HOTPOT_XTB_INTEGRATION=1 to run official xTB parity tests")
    configured = os.environ.get("HOTPOT_XTB_EXECUTABLE")
    resolved = configured or shutil.which("xtb")
    if resolved is None or not Path(resolved).is_file():
        pytest.skip("official xTB executable is not available")
    return Path(resolved).resolve()


def _methanol():
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
    return mol


def _single_thread_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    return environment


def _direct_run(
    executable: Path,
    work_directory: Path,
    method_arguments: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    input_path = work_directory / "input.xyz"
    prepare_xtb_input(_methanol(), input_path)
    spin_arguments = () if method_arguments == ("--gfnff",) else ("--uhf", "0")
    return subprocess.run(
        (
            str(executable),
            str(input_path),
            *method_arguments,
            "--sp",
            "--json",
            "--chrg",
            "0",
            *spin_arguments,
        ),
        cwd=work_directory,
        env=_single_thread_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_official_gfn2_wrapper_matches_direct_cli_energy(tmp_path: Path) -> None:
    executable = _official_executable()
    report = run_gfn_xtb(
        _methanol(),
        method=GFNXTBMethod.GFN2_XTB,
        task=XTBTask.SINGLEPOINT,
        charge=0,
        unpaired_electrons=0,
        executable=executable,
        environment=_single_thread_environment(),
    )

    direct_directory = tmp_path / "direct-gfn2"
    direct_directory.mkdir()
    direct = _direct_run(executable, direct_directory, ("--gfn", "2"))
    assert direct.returncode == 0, direct.stdout + direct.stderr
    direct_result = json.loads(
        (direct_directory / "xtbout.json").read_text(encoding="utf-8")
    )

    assert report.energy_hartree == pytest.approx(
        float(direct_result["total energy"]),
        abs=1.0e-12,
    )
    assert report.atom_order_verified is True
    assert report.coordinates_committed is False
    assert report.backend_info.executable == executable


def test_official_gfnff_wrapper_matches_direct_cli_energy(tmp_path: Path) -> None:
    executable = _official_executable()
    report = run_gfnff(
        _methanol(),
        task=XTBTask.SINGLEPOINT,
        charge=0,
        executable=executable,
        environment=_single_thread_environment(),
    )

    direct_directory = tmp_path / "direct-gfnff"
    direct_directory.mkdir()
    direct = _direct_run(executable, direct_directory, ("--gfnff",))
    assert direct.returncode == 0, direct.stdout + direct.stderr
    direct_text = (direct_directory / "gfnff_lists.json").read_text(
        encoding="utf-8"
    )
    match = _GFNFF_ENERGY.search(direct_text)

    assert match is not None
    assert report.energy_hartree == pytest.approx(
        float(match.group(1).replace("D", "E").replace("d", "e")),
        abs=1.0e-12,
    )
    assert report.atom_order_verified is True
    assert report.coordinates_committed is False
