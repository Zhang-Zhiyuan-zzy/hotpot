"""Molecule, XYZ and xTB artifact adapter contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol, Tuple

import numpy as np
import pytest

import hotpot
from hotpot.cheminfo.core import Molecule
from hotpot.plugins.xtb.adapter import (
    commit_xtb_coordinates,
    parse_xtb_artifacts,
    prepare_xtb_input,
)
from hotpot.plugins.xtb.backend import probe_xtb_backend
from hotpot.plugins.xtb.contracts import (
    XTBInputError,
    XTBGeometry,
    XTBMethod,
    XTBRequest,
    XTBResultError,
    XTBRunReport,
    XTBTask,
)
from hotpot.plugins.xtb.runner import run_xtb


class FakeXTBFactory(Protocol):
    def __call__(
        self,
        scenario: str = "success",
        directory_name: str = "fake_xtb",
    ) -> Path: ...


def _explicit_methanol() -> Molecule:
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


def _run_fake(
    executable: Path,
    work_directory: Path,
    mol: Molecule,
    method: XTBMethod,
    task: XTBTask,
) -> Tuple[XTBGeometry, XTBRunReport]:
    work_directory.mkdir()
    expected = prepare_xtb_input(mol, work_directory / "input.xyz")
    report = run_xtb(
        XTBRequest(
            backend_info=probe_xtb_backend(
                executable=executable,
                environment=dict(os.environ),
            ),
            method=method,
            task=task,
            input_path=work_directory / "input.xyz",
            work_directory=work_directory,
            charge=0,
            unpaired_electrons=None if method is XTBMethod.GFNFF else 0,
            environment=dict(os.environ),
        )
    )
    return expected, report


def test_prepare_rejects_implicit_hydrogens_and_degenerate_geometry(
    tmp_path: Path,
) -> None:
    with pytest.raises(XTBInputError):
        prepare_xtb_input(hotpot.read_mol("CO", "smi"), tmp_path / "implicit.xyz")

    mol = _explicit_methanol()
    mol.coordinates = np.zeros((len(mol.atoms), 3), dtype=float)
    with pytest.raises(XTBInputError):
        prepare_xtb_input(mol, tmp_path / "coincident.xyz")


def test_prepare_writes_complete_strict_xyz(tmp_path: Path) -> None:
    mol = _explicit_methanol()
    geometry = prepare_xtb_input(mol, tmp_path / "input.xyz")
    lines = (tmp_path / "input.xyz").read_text(encoding="utf-8").splitlines()

    assert int(lines[0]) == len(mol.atoms)
    assert len(lines) == len(mol.atoms) + 2
    assert tuple(line.split()[0] for line in lines[2:]) == geometry.symbols
    assert all(len(line.split()) == 4 for line in lines[2:])


def test_prepare_accepts_a_finite_single_atom_geometry(tmp_path: Path) -> None:
    mol = hotpot.read_mol("[He]", "smi")

    geometry = prepare_xtb_input(mol, tmp_path / "helium.xyz")

    assert geometry.symbols == ("He",)
    assert geometry.coordinates == ((0.0, 0.0, 0.0),)


@pytest.mark.parametrize("method", (XTBMethod.GFN2_XTB, XTBMethod.GFNFF))
def test_parse_accepts_method_specific_results_and_commits_only_on_request(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
    method: XTBMethod,
) -> None:
    mol = _explicit_methanol()
    initial = mol.coordinates.copy()
    expected, report = _run_fake(
        fake_xtb_factory(),
        tmp_path / method.value,
        mol,
        method,
        XTBTask.OPTIMIZE,
    )
    if method is XTBMethod.GFNFF:
        path = report.artifacts["gfnff_lists.json"].path
        text = path.read_text(encoding="utf-8")
        path.write_text(text[:-1] + ',\n"alist": [********]\n}\n', encoding="utf-8")

    result = parse_xtb_artifacts(report, expected)
    np.testing.assert_array_equal(mol.coordinates, initial)
    assert result.energy_hartree == pytest.approx(-5.123456789)
    assert result.atom_order_verified is True

    mol.to_obmol()
    assert mol._obmol is not None
    commit_xtb_coordinates(mol, result)
    assert not np.array_equal(mol.coordinates, initial)
    assert mol._obmol is None
    assert mol._row2idx is None


@pytest.mark.parametrize(
    "scenario",
    (
        "nonfinite_energy",
        "nonfinite_coordinates",
        "truncated_geometry",
        "reordered_elements",
    ),
)
def test_invalid_artifacts_raise_with_report_without_mutation(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
    scenario: str,
) -> None:
    mol = _explicit_methanol()
    initial = mol.coordinates.copy()
    expected, report = _run_fake(
        fake_xtb_factory(scenario),
        tmp_path / scenario,
        mol,
        XTBMethod.GFN2_XTB,
        XTBTask.OPTIMIZE,
    )

    with pytest.raises(XTBResultError) as error:
        parse_xtb_artifacts(report, expected)

    assert error.value.report is report
    np.testing.assert_array_equal(mol.coordinates, initial)


def test_optional_partial_charges_require_exact_atom_count(
    fake_xtb_factory: FakeXTBFactory,
    tmp_path: Path,
) -> None:
    mol = _explicit_methanol()
    expected, report = _run_fake(
        fake_xtb_factory(),
        tmp_path / "charges",
        mol,
        XTBMethod.GFN2_XTB,
        XTBTask.SINGLEPOINT,
    )
    path = report.artifacts["xtbout.json"].path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["partial charges"] = [0.0]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(XTBResultError):
        parse_xtb_artifacts(report, expected)
