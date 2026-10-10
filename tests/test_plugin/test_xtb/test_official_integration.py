"""Opt-in parity checks against an installed official xTB executable."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

import hotpot
from hotpot.cheminfo.core import Molecule
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


def _methanol() -> Molecule:
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


def _single_atom(smiles: str) -> Molecule:
    mol = hotpot.read_mol(smiles, "smi")
    mol.coordinates = np.asarray(((0.0, 0.0, 0.0),), dtype=float)
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
    mol: Molecule,
    method_arguments: tuple[str, ...],
    *,
    task: XTBTask,
    charge: int,
    unpaired_electrons: Optional[int],
) -> subprocess.CompletedProcess[str]:
    input_path = work_directory / "input.xyz"
    prepare_xtb_input(mol, input_path)
    spin_arguments = (
        ()
        if unpaired_electrons is None
        else ("--uhf", str(unpaired_electrons))
    )
    return subprocess.run(
        (
            str(executable),
            str(input_path),
            *method_arguments,
            "--sp" if task is XTBTask.SINGLEPOINT else "--opt",
            "--json",
            "--chrg",
            str(charge),
            *spin_arguments,
        ),
        cwd=work_directory,
        env=_single_thread_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def _xyz_symbols_and_coordinates(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[0])
    records = tuple(line.split() for line in lines[2:])
    assert len(records) == atom_count
    assert all(len(record) == 4 for record in records)
    return (
        tuple(record[0] for record in records),
        np.asarray(
            tuple(tuple(float(value) for value in record[1:]) for record in records),
            dtype=float,
        ),
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
    direct = _direct_run(
        executable,
        direct_directory,
        _methanol(),
        ("--gfn", "2"),
        task=XTBTask.SINGLEPOINT,
        charge=0,
        unpaired_electrons=0,
    )
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
    direct = _direct_run(
        executable,
        direct_directory,
        _methanol(),
        ("--gfnff",),
        task=XTBTask.SINGLEPOINT,
        charge=0,
        unpaired_electrons=None,
    )
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


def test_official_gfn2_optimization_matches_direct_cli_result(
    tmp_path: Path,
) -> None:
    executable = _official_executable()
    optimized_mol = _methanol()
    report = run_gfn_xtb(
        optimized_mol,
        method=GFNXTBMethod.GFN2_XTB,
        task=XTBTask.OPTIMIZE,
        charge=0,
        unpaired_electrons=0,
        executable=executable,
        environment=_single_thread_environment(),
    )

    direct_directory = tmp_path / "direct-gfn2-optimize"
    direct_directory.mkdir()
    direct = _direct_run(
        executable,
        direct_directory,
        _methanol(),
        ("--gfn", "2"),
        task=XTBTask.OPTIMIZE,
        charge=0,
        unpaired_electrons=0,
    )
    assert direct.returncode == 0, direct.stdout + direct.stderr
    direct_result = json.loads(
        (direct_directory / "xtbout.json").read_text(encoding="utf-8")
    )
    direct_symbols, direct_coordinates = _xyz_symbols_and_coordinates(
        direct_directory / "xtbopt.xyz"
    )

    assert report.energy_hartree == pytest.approx(
        float(direct_result["total energy"]),
        abs=1.0e-12,
    )
    assert direct_symbols == tuple(atom.symbol for atom in optimized_mol.atoms)
    np.testing.assert_allclose(
        np.asarray(optimized_mol.coordinates, dtype=float),
        direct_coordinates,
        rtol=0.0,
        atol=1.0e-8,
    )
    assert report.atom_order_verified is True
    assert report.coordinates_committed is True
    assert report.converged is True


@pytest.mark.parametrize(
    ("smiles", "charge", "unpaired_electrons"),
    (("[Cl-]", -1, 0), ("[H]", 0, 1)),
    ids=("ionic-chloride", "hydrogen-radical"),
)
def test_official_gfn2_wrapper_matches_direct_cli_for_charged_and_open_shell_states(
    tmp_path: Path,
    smiles: str,
    charge: int,
    unpaired_electrons: int,
) -> None:
    executable = _official_executable()
    report = run_gfn_xtb(
        _single_atom(smiles),
        method=GFNXTBMethod.GFN2_XTB,
        task=XTBTask.SINGLEPOINT,
        charge=charge,
        unpaired_electrons=unpaired_electrons,
        executable=executable,
        environment=_single_thread_environment(),
    )

    direct_directory = tmp_path / f"direct-state-{charge}-{unpaired_electrons}"
    direct_directory.mkdir()
    direct = _direct_run(
        executable,
        direct_directory,
        _single_atom(smiles),
        ("--gfn", "2"),
        task=XTBTask.SINGLEPOINT,
        charge=charge,
        unpaired_electrons=unpaired_electrons,
    )
    assert direct.returncode == 0, direct.stdout + direct.stderr
    direct_result = json.loads(
        (direct_directory / "xtbout.json").read_text(encoding="utf-8")
    )

    assert report.energy_hartree == pytest.approx(
        float(direct_result["total energy"]),
        abs=1.0e-12,
    )
    assert report.charge == charge
    assert report.unpaired_electrons == unpaired_electrons
    assert report.argv[-4:] == (
        "--chrg",
        str(charge),
        "--uhf",
        str(unpaired_electrons),
    )
    assert report.atom_order_verified is True
    assert report.coordinates_committed is False
