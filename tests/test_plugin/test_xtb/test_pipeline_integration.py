"""Composition tests for xTB shell streams and controlled pipelines."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import hotpot
from hotpot.pipeline.contracts import MolecularPayload, MolecularRecord, StageSpec
from hotpot.pipeline.runner import run_pipeline
from hotpot.plugins.xtb.contracts import XTBMethod
from hotpot.plugins.xtb.stream import (
    XTBStreamRecord,
    read_sdf_records,
    write_sdf_records,
)


def _water():
    mol = hotpot.read_mol("[H]O[H]", "smi")
    mol.coordinates = np.asarray(
        (
            (-0.75, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.75, 0.0, 0.0),
        ),
        dtype=float,
    )
    return mol


def _python_environment() -> dict[str, str]:
    environment = dict(os.environ)
    project_root = str(Path(__file__).resolve().parents[3])
    current_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        project_root if not current_path else f"{project_root}{os.pathsep}{current_path}"
    )
    return environment


def test_standalone_shell_composes_gfnff_then_gfn2_as_pure_sdf(
    fake_xtb_factory,
) -> None:
    executable = fake_xtb_factory()
    source_sdf = write_sdf_records((XTBStreamRecord(_water()),))
    common = (
        sys.executable,
        "-m",
        "hotpot",
        "xtb",
        "-",
        "--input-format",
        "sdf",
        "--xtb-executable",
        str(executable),
    )
    gfnff = subprocess.Popen(
        (*common, "--method", "gfnff", "--task", "optimize"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_python_environment(),
    )
    gfn2 = subprocess.Popen(
        (*common, "--method", "gfn2", "--task", "singlepoint"),
        stdin=gfnff.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_python_environment(),
    )
    assert gfnff.stdin is not None
    assert gfnff.stdout is not None
    gfnff.stdout.close()
    gfnff.stdin.write(source_sdf)
    gfnff.stdin.close()
    output, gfn2_stderr = gfn2.communicate(timeout=60)
    gfnff_stderr = "" if gfnff.stderr is None else gfnff.stderr.read()
    gfnff_status = gfnff.wait(timeout=60)

    assert gfnff_status == 0, gfnff_stderr
    assert gfn2.returncode == 0, gfn2_stderr
    records = read_sdf_records(output)
    assert len(records) == 1
    assert records[0].metadata is not None
    assert records[0].metadata.method is XTBMethod.GFN2_XTB
    assert records[0].metadata.total_charge == 0
    assert records[0].metadata.unpaired_electrons == 0
    assert records[0].metadata.energy_hartree == -5.123456789
    assert "TOTAL ENERGY" not in output


@pytest.mark.parametrize("include_gfnff", (False, True))
def test_controlled_pipeline_composes_optional_gfnff_and_gfn2_with_artifacts(
    fake_xtb_factory,
    tmp_path: Path,
    include_gfnff: bool,
) -> None:
    executable = fake_xtb_factory()
    results_directory = tmp_path / f"controlled-results-{include_gfnff}"
    common = ("--xtb-executable", str(executable))
    specs = [
        StageSpec(
            "xtb",
            ("--method", "gfn2", "--task", "singlepoint", *common),
        )
    ]
    if include_gfnff:
        specs.insert(
            0,
            StageSpec(
                "xtb",
                ("--method", "gfnff", "--task", "optimize", *common),
            ),
        )
    result = run_pipeline(
        tuple(specs),
        results_directory=results_directory,
        initial_payload=MolecularPayload((MolecularRecord(_water()),)),
    )

    assert result.status.value == "succeeded"
    assert result.payload is not None
    assert result.payload.records[0].electronic_state is not None
    assert result.payload.records[0].electronic_state.unpaired_electrons == 0
    manifest = json.loads(
        (results_directory / "manifest.json").read_text(encoding="utf-8")
    )
    expected_methods = ["gfnff", "gfn2"] if include_gfnff else ["gfn2"]
    assert [stage["name"] for stage in manifest["stages"]] == [
        "xtb"
    ] * len(expected_methods)
    assert [
        stage["report"]["xtb"]["requested_method"]
        for stage in manifest["stages"]
    ] == expected_methods
    assert (results_directory / "final.sdf").is_file()
    assert all(stage["artifacts"] for stage in manifest["stages"])
