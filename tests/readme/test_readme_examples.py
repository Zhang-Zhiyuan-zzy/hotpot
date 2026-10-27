"""Execute the examples directly from the root README Markdown source."""

from __future__ import annotations

import contextlib
import io
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import hotpot as hp
from hotpot.cheminfo import forcefields as ff


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
FENCE = re.compile(r"```(?P<language>\w+)\n(?P<body>.*?)\n```", re.DOTALL)


def _blocks(test_name: str) -> tuple[tuple[str, str], ...]:
    text = README.read_text(encoding="utf-8")
    reference = f"tests/readme/test_readme_examples.py::{test_name}"
    start = text.index(reference) + len(reference)
    next_reference = text.find("<!-- Verified by", start)
    section = text[start:] if next_reference < 0 else text[start:next_reference]
    return tuple(
        (match.group("language"), match.group("body"))
        for match in FENCE.finditer(section)
    )


def _command(test_name: str) -> tuple[str, ...]:
    language, source = _blocks(test_name)[0]
    assert language == "bash"
    logical = source.replace("\\\n", " ")
    logical = " ".join(
        line[2:] if line.startswith("$ ") else line.strip()
        for line in logical.splitlines()
    )
    arguments = tuple(shlex.split(logical))
    assert arguments[0] == "hotpot"
    return arguments[1:]


def _run_hotpot(
    arguments: tuple[str, ...],
    *,
    cwd: Path = ROOT,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if not environment.get("HOTPOT_TEST_INSTALLED_PACKAGE"):
        pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(ROOT) if not pythonpath else f"{ROOT}{os.pathsep}{pythonpath}"
        )
    return subprocess.run(
        [sys.executable, "-m", "hotpot", *arguments],
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        input=input_text,
    )


def _run_python(test_name: str) -> tuple[str, dict[str, object]]:
    language, source = _blocks(test_name)[0]
    assert language == "python"
    namespace: dict[str, object] = {"__name__": "__readme_example__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(source, f"README.md::{test_name}", "exec"), namespace)
    return output.getvalue().strip(), namespace


def _documented_output(test_name: str) -> str:
    language, body = _blocks(test_name)[1]
    assert language == "text"
    return body.strip()


def test_readme_primary_cli_surface() -> None:
    output = _run_hotpot(("--help",)).stdout

    for command in ("mca", "cbond", "ff", "xtb", "run", "ds"):
        assert command in output


def test_readme_xtb_documentation_surface() -> None:
    output = _run_hotpot(("xtb", "--doc")).stdout

    assert "hotpot xtb" in output
    assert "explicit-atom 3D" in output
    assert "GFN-FF" in output
    assert "standard input" in output.lower()


def test_readme_pipeline_documentation_surface() -> None:
    output = _run_hotpot(("run", "--doc")).stdout

    assert "hotpot run" in output
    assert "::" in output
    assert "results" in output.lower()


@pytest.mark.model_artifact
def test_readme_mca_cli_output() -> None:
    lines = _run_hotpot(_command("test_readme_mca_cli_output")).stdout.splitlines()
    expected = _documented_output("test_readme_mca_cli_output").splitlines()

    assert lines[0] == expected[0]
    assert len(lines) == len(expected)
    for actual_line, expected_line in zip(lines[1:], expected[1:]):
        actual = actual_line.split()
        documented = expected_line.split()
        assert actual[:2] == documented[:2]
        assert float(actual[2]) == pytest.approx(float(documented[2]), abs=0.25)
        assert actual[3] == documented[3]


@pytest.mark.model_artifact
def test_readme_cbond_cli_output() -> None:
    lines = _run_hotpot(_command("test_readme_cbond_cli_output")).stdout.splitlines()
    expected = _documented_output("test_readme_cbond_cli_output").splitlines()

    assert lines[:3] == expected[:3]
    assert lines[-1] == expected[-1]
    assert len(lines) == len(expected)
    for actual_line, expected_line in zip(lines[3:-1], expected[3:-1]):
        actual = actual_line.split()
        documented = expected_line.split()
        assert actual[:2] == documented[:2]
        assert float(actual[2]) == pytest.approx(float(documented[2]), abs=1.0e-4)


def test_readme_forcefield_cli_output(tmp_path: Path) -> None:
    result = _run_hotpot(
        _command("test_readme_forcefield_cli_output"),
        cwd=tmp_path,
    )

    assert result.stdout == ""
    molecule = hp.read_mol(tmp_path / "ethane.mol2")
    assert len(molecule.atoms) == 8
    assert ff.evaluate_structure_acceptance(molecule, level="standard").passed
    assert '"status": "ok"' in (tmp_path / "ethane.json").read_text()


@pytest.mark.model_artifact
def test_readme_cbond_forcefield_pipeline(tmp_path: Path) -> None:
    arguments = _command("test_readme_cbond_forcefield_pipeline")
    separator = arguments.index("|")
    cbond = _run_hotpot(arguments[:separator], cwd=tmp_path)
    forcefield = _run_hotpot(
        arguments[separator + 2 :],
        cwd=tmp_path,
        input_text=cbond.stdout,
    )

    assert forcefield.stdout == ""
    molecule = hp.read_mol(tmp_path / "eu-extractant.mol2")
    assert [atom.symbol for atom in molecule.metals] == ["Eu"]
    assert len(molecule.c_bonds) == 4


@pytest.mark.model_artifact
def test_readme_mca_python_api_output() -> None:
    output, _ = _run_python("test_readme_mca_python_api_output")
    assert output == _documented_output("test_readme_mca_python_api_output")


def test_readme_core_ring_and_aromaticity_output() -> None:
    output, _ = _run_python("test_readme_core_ring_and_aromaticity_output")
    assert output == _documented_output("test_readme_core_ring_and_aromaticity_output")


def test_readme_conversion_output() -> None:
    output, _ = _run_python("test_readme_conversion_output")
    assert output == _documented_output("test_readme_conversion_output")


def test_readme_thermo_output() -> None:
    output, _ = _run_python("test_readme_thermo_output")
    actual = tuple(map(float, output.splitlines()))
    documented = tuple(
        map(float, _documented_output("test_readme_thermo_output").splitlines())
    )

    assert actual == pytest.approx(documented, rel=1.0e-8, abs=1.0e-8)


def test_readme_graph_spectrum_output() -> None:
    output, _ = _run_python("test_readme_graph_spectrum_output")
    actual = output.splitlines()
    documented = _documented_output("test_readme_graph_spectrum_output").splitlines()

    assert actual[:2] == documented[:2]
    assert tuple(map(float, actual[2:])) == pytest.approx(
        tuple(map(float, documented[2:])),
        abs=1.0e-12,
    )


def test_readme_native_rdwrapper_output() -> None:
    output, namespace = _run_python("test_readme_native_rdwrapper_output")
    batch = namespace["batch"]

    assert output == str(batch.coordinates.shape)
    assert batch.coordinates.shape[0] > 0
    assert batch.coordinates.shape[1:] == (3, 3)
    assert np.isfinite(batch.coordinates).all()


def test_readme_forcefield_validation_python_output() -> None:
    output, namespace = _run_python(
        "test_readme_forcefield_validation_python_output"
    )

    assert output == _documented_output(
        "test_readme_forcefield_validation_python_output"
    )
    assert namespace["report"].passed


def test_readme_ring_wedge_public_interface() -> None:
    assert hp.RingWedge.__name__ == "RingWedge"
