"""Opt-in mixed-batch CLI checks against an installed official xTB executable."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import hotpot
from hotpot.plugins.xtb import cli
from hotpot.plugins.xtb.stream import read_sdf_records


pytestmark = pytest.mark.external


def _official_executable() -> Path:
    if os.environ.get("HOTPOT_XTB_INTEGRATION") != "1":
        pytest.skip("set HOTPOT_XTB_INTEGRATION=1 to run official xTB CLI tests")
    configured = os.environ.get("HOTPOT_XTB_EXECUTABLE")
    resolved = configured or shutil.which("xtb")
    if resolved is None or not Path(resolved).is_file():
        pytest.skip("official xTB executable is not available")
    return Path(resolved).resolve()


@pytest.mark.parametrize("jobs", (1, 2), ids=("serial", "parallel"))
def test_official_cli_keeps_mixed_batch_failures_local(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    jobs: int,
) -> None:
    executable = _official_executable()
    # Full graph hydrogens keep this batch independent of implicit-H normalization.
    fixtures = (
        ("[H]O[H]", ((-0.7, 0.0, 0.0), (0.0, 0.0, 0.0), (0.7, 0.0, 0.0))),
        ("[H]", ((0.0, 0.0, 0.0),)),
        (
            "[H]C([H])([H])[H]",
            (
                (0.63, 0.63, 0.63),
                (0.0, 0.0, 0.0),
                (0.63, -0.63, -0.63),
                (-0.63, 0.63, -0.63),
                (-0.63, -0.63, 0.63),
            ),
        ),
    )
    molecules = []
    for smiles, coordinates in fixtures:
        mol = hotpot.read_mol(smiles, "smi")
        mol.coordinates = np.asarray(coordinates, dtype=float)
        molecules.append(mol)
    input_path = tmp_path / "input.sdf"
    input_text = "".join(
        mol.write(fmt="sdf", write_single=True) for mol in molecules
    )
    input_path.write_text(input_text, encoding="utf-8")
    input_records = read_sdf_records(input_text)
    assert len(input_records) == 3
    assert [tuple(atom.symbol for atom in record.mol.atoms) for record in input_records] == [
        ("H", "O", "H"),
        ("H",),
        ("H", "C", "H", "H", "H"),
    ]
    output_path = tmp_path / "output.sdf"
    report_path = tmp_path / "report.json"

    returncode = cli.main(
        (
            str(input_path),
            "--method", "gfn2",
            "--task", "singlepoint",
            "--charge", "0",
            "--unpaired-electrons", "0",
            "--xtb-executable", str(executable),
            "--threads", "1",
            "--jobs", str(jobs),
            "--post-check", "off",
            "--output", str(output_path),
            "--report", str(report_path),
            "--native-log", str(tmp_path / "native.log"),
            "--work-directory", str(tmp_path / "work"),
            "--keep-work-directory",
        )
    )

    captured = capsys.readouterr()
    assert returncode == 1
    assert captured.out == ""
    assert "record 1:" in captured.err
    assert "HARD-013.UHF_PARITY" in captured.err
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report["results"]
    assert len(results) == 3
    assert [result["index"] for result in results] == [0, 1, 2]
    assert [result["status"] for result in results] == ["ok", "error", "ok"]
    failure = results[1]
    assert failure["error"]["type"] == "XTBStateResolutionError"
    assert "HARD-013.UHF_PARITY" in failure["error"]["message"]
    assert "xtb_report" not in failure

    output_records = read_sdf_records(output_path.read_text(encoding="utf-8"))
    assert len(output_records) == 2
    for output_record, index in zip(output_records, (0, 2)):
        xtb_report = results[index]["xtb_report"]
        assert math.isfinite(xtb_report["energy_hartree"])
        assert xtb_report["charge"] == 0
        assert xtb_report["unpaired_electrons"] == 0
        assert xtb_report["atom_order_verified"] is True
        assert xtb_report["coordinates_committed"] is False
        assert tuple(atom.symbol for atom in output_record.mol.atoms) == tuple(
            atom.symbol for atom in input_records[index].mol.atoms
        )
        np.testing.assert_array_equal(
            output_record.mol.coordinates,
            input_records[index].mol.coordinates,
        )
        assert output_record.metadata is not None
        assert math.isfinite(output_record.metadata.energy_hartree)
        assert output_record.metadata.energy_hartree == xtb_report["energy_hartree"]
    assert input_path.read_text(encoding="utf-8") == input_text
