"""Regression tests for every executable example shown in README.md."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import hotpot as hp
from hotpot.cheminfo.calculator import mca
from hotpot.cheminfo import forcefields as ff


ROOT = Path(__file__).resolve().parents[2]
COMPLEX_EXTRACTANT_SMILES = (
    "O=C(N(C)CCC)C(C=C1)=NC2=C1C=CC3=C2N=C("
    "C4=NC(C(C)(C)CCC5(C)C)=C5N=N4)C=C3"
)


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
    assert "xtb" in output
    assert "run" in output


def test_readme_xtb_documentation_surface() -> None:
    output = _run_hotpot("xtb", "--doc").stdout

    assert "hotpot xtb" in output
    assert "explicit-atom 3D" in output
    assert "GFN-FF" in output
    assert "standard input" in output.lower()


def test_readme_pipeline_documentation_surface() -> None:
    output = _run_hotpot("run", "--doc").stdout

    assert "hotpot run" in output
    assert "::" in output
    assert "results" in output.lower()


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
    output = _run_hotpot(
        "cbond",
        "Eu",
        COMPLEX_EXTRACTANT_SMILES,
        "--device",
        "cpu",
        "--bond-detail",
    ).stdout.strip()

    assert output == (
        "CCCN(C1=[O][Eu@]23n4c1ccc1c4c4n3c("
        "-[c]3n2nc2c(n3)C(C)(C)CCC2(C)C)ccc4cc1)C\n"
        "Cbond Detail:\n"
        "AtomIdx  Atom  Score\n"
        "10       N     4.04834\n"
        "17       N     7.02889\n"
        "0        O     6.10205\n"
        "32       N     4.98739\n"
        "-- End --"
    )


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


def test_readme_forcefield_validation_python_output() -> None:
    molecule = hp.read_mol("CC")
    molecule.build3d(
        forcefield="UFF",
        epochs=1,
        steps_per_epoch=20,
        quality_level="standard",
        seed=2026,
    )

    report = ff.evaluate_structure_acceptance(molecule, level="standard")

    assert report.passed


def test_readme_cbond_forcefield_pipeline(tmp_path: Path) -> None:
    cbond = _run_hotpot(
        "cbond",
        "Eu",
        COMPLEX_EXTRACTANT_SMILES,
        "--device",
        "cpu",
    )
    output_path = tmp_path / "eu-extractant.mol2"
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
    assert len(molecule.c_bonds) == 4


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
