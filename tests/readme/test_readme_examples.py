"""Regression tests for every executable example shown in README.md."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import hotpot as hp
from hotpot.calculator import mca
from hotpot.cheminfo import forcefields as ff


ROOT = Path(__file__).resolve().parents[2]


def _run_hotpot(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hotpot", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_readme_primary_cli_surface() -> None:
    output = _run_hotpot("--help").stdout

    assert "mca" in output
    assert "cbond" in output
    assert "ff" in output


def test_readme_mca_cli_output() -> None:
    output = _run_hotpot("mca", "c1ccccc1CN", "--device", "cpu").stdout.strip()

    assert output == (
        "No.  Atom  MCA(kJ/mol)  is_Nuc_site\n"
        "1    C     319.00       False\n"
        "2    C     304.00       False\n"
        "3    C     329.25       False\n"
        "4    C     303.50       False\n"
        "5    C     318.75       False\n"
        "6    C     324.00       False\n"
        "7    C     314.50       False\n"
        "8    N     489.50       True"
    )


def test_readme_cbond_cli_output() -> None:
    output = _run_hotpot("cbond", "Eu", "CN", "--device", "cpu").stdout.strip()

    assert output == "C[NH2+][Eu]"


def test_readme_forcefield_cli_output(tmp_path: Path) -> None:
    output_path = tmp_path / "ethane.mol2"
    report_path = tmp_path / "ethane.json"

    result = _run_hotpot(
        "ff",
        "CC",
        "--epochs",
        "1",
        "--steps-per-epoch",
        "20",
        "--quality",
        "standard",
        "--seed",
        "2026",
        "--report",
        str(report_path),
        "-o",
        str(output_path),
    )

    assert result.stdout == ""
    molecule = hp.read_mol(output_path)
    assert len(molecule.atoms) == 8
    assert ff.evaluate_structure_acceptance(molecule, level="standard").passed
    assert '"status": "ok"' in report_path.read_text(encoding="utf-8")


def test_readme_cbond_forcefield_pipeline(tmp_path: Path) -> None:
    cbond = _run_hotpot("cbond", "Eu", "CN", "--device", "cpu")
    output_path = tmp_path / "eu-methylamine.mol2"
    forcefield = subprocess.run(
        [
            sys.executable,
            "-m",
            "hotpot",
            "ff",
            "-",
            "--input-format",
            "smi",
            "--epochs",
            "1",
            "--steps-per-epoch",
            "20",
            "--quality",
            "off",
            "--seed",
            "2026",
            "-o",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        input=cbond.stdout,
    )

    assert forcefield.stdout == ""
    molecule = hp.read_mol(output_path)
    assert [atom.symbol for atom in molecule.metals] == ["Eu"]
    assert len(molecule.c_bonds) == 1


def test_readme_mca_python_api_output() -> None:
    molecule = hp.read_mol("c1ccccc1CN")

    mca(molecule, device="cpu")

    assert molecule.atoms[7].mca == pytest.approx(489.5)
    assert molecule.atoms[7].label == "N7"
    assert tuple(
        (atom.symbol, value) for atom, value in molecule.mca_sites.items()
    ) == (("N", pytest.approx(489.5)),)


def test_readme_core_ring_and_aromaticity_output() -> None:
    molecule = hp.read_mol("c1ccccc1O")

    assert len(molecule.atoms) == 7
    assert [len(ring.atoms) for ring in molecule.rings] == [6]
    assert [ring.is_aromatic for ring in molecule.rings] == [True]


def test_readme_conversion_output() -> None:
    molecule = hp.read_mol("CCN")

    assert hp.to_hotpot_mol(molecule) is molecule
    assert len(hp.to_hotpot_mol(molecule.to_rdmol()).atoms) == 3
    assert len(hp.to_hotpot_mol(molecule.to_obmol()).atoms) == 3


def test_readme_graph_spectrum_output() -> None:
    phenol = hp.read_mol("c1ccc(O)cc1", "smi").graph_spectral()
    benzoic_acid = hp.read_mol("c1ccccc1C(=O)O", "smi").graph_spectral()
    reordered_phenol = hp.read_mol("c1ccccc1O", "smi").graph_spectral()

    assert phenol.vectors.shape == (6, 13)
    assert benzoic_acid.vectors.shape == (6, 15)
    assert phenol | benzoic_acid == pytest.approx(0.907590226292854)
    assert phenol | reordered_phenol == pytest.approx(1.0)


def test_readme_thermo_output() -> None:
    molecule = hp.read_mol("c1ccc(O)cc1", "smi")
    thermo = molecule.get_thermo(temp=298.15, pressure=101325)

    assert thermo.Tc == pytest.approx(694.2)
    assert thermo.Psat == pytest.approx(80.20201686360379)


def test_readme_ring_wedge_public_interface() -> None:
    assert hp.RingWedge.__name__ == "RingWedge"
