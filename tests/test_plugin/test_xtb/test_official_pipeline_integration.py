"""Opt-in end-to-end checks with the CBond model and official xTB."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from hotpot.pipeline.contracts import MolecularPayload, StageSpec
from hotpot.pipeline.runner import run_pipeline
from hotpot.plugins.xtb.stream import read_sdf_records


_LIGAND = "NCCO"
_METAL = "Zn"


def _integration_resources() -> Path:
    if os.environ.get("HOTPOT_CBOND_INTEGRATION") != "1":
        pytest.skip(
            "set HOTPOT_CBOND_INTEGRATION=1 to use the real CBond model"
        )
    if os.environ.get("HOTPOT_XTB_INTEGRATION") != "1":
        pytest.skip("set HOTPOT_XTB_INTEGRATION=1 to use official xTB")

    configured = os.environ.get("HOTPOT_XTB_EXECUTABLE")
    resolved = configured or shutil.which("xtb")
    if resolved is None or not Path(resolved).is_file():
        pytest.skip("official xTB executable is not available")
    return Path(resolved).resolve()


def _stage_specs(
    executable: Path,
    include_gfnff: bool,
) -> tuple[StageSpec, ...]:
    specs = [
        StageSpec("cbond", (_METAL, _LIGAND, "--device", "cpu")),
        StageSpec(
            "ff",
            (
                "--route",
                "complex",
                "--rebuild",
                "--quality",
                "standard",
                "--seed",
                "7",
                "--timeout",
                "300",
            ),
        ),
    ]
    xtb_options = (
        "--xtb-executable",
        str(executable),
        "--threads",
        "1",
        "--post-check",
        "standard",
    )
    if include_gfnff:
        specs.append(
            StageSpec(
                "xtb",
                ("--method", "gfnff", "--task", "optimize", *xtb_options),
            )
        )
    specs.append(
        StageSpec(
            "xtb",
            ("--method", "gfn2", "--task", "singlepoint", *xtb_options),
        )
    )
    return tuple(specs)


def _quality_passed(report: dict[str, object]) -> bool:
    return bool(report["passed"]) and all(
        bool(check["passed"]) for check in report["checks"]
    )


@pytest.mark.parametrize("include_gfnff", (False, True))
def test_official_controlled_coordination_pipeline(
    tmp_path: Path,
    include_gfnff: bool,
) -> None:
    executable = _integration_resources()
    results_directory = tmp_path / f"official-coordination-{include_gfnff}"

    result = run_pipeline(
        _stage_specs(executable, include_gfnff),
        results_directory=results_directory,
        initial_payload=MolecularPayload(()),
    )

    assert result.status.value == "succeeded"
    assert result.payload is not None
    assert len(result.payload.records) == 1
    state = result.payload.records[0].electronic_state
    assert state is not None
    assert state.charge == 2
    assert state.unpaired_electrons == 0

    manifest_path = results_directory / "manifest.json"
    final_path = results_directory / "final.sdf"
    assert manifest_path.is_file()
    assert final_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"

    expected_names = ["cbond", "ff"]
    expected_methods = ["gfn2"]
    if include_gfnff:
        expected_names.append("xtb")
        expected_methods.insert(0, "gfnff")
    expected_names.append("xtb")
    assert [stage["name"] for stage in manifest["stages"]] == expected_names
    assert all(stage["status"] == "succeeded" for stage in manifest["stages"])

    for stage in manifest["stages"]:
        stage_directory = results_directory / stage["directory"]
        assert (stage_directory / "manifest.json").is_file()
        assert (stage_directory / "output.sdf").is_file()
        assert (stage_directory / "report.json").is_file()

    cbond_report = manifest["stages"][0]["report"]["cbond"]
    assert cbond_report["metal"] == _METAL
    assert cbond_report["donor_indices"] == [0, 3]

    forcefield_report = manifest["stages"][1]["report"]["forcefield"][0]
    assert _quality_passed(forcefield_report["quality_report"])
    assert forcefield_report["quality_report"]["level"] == "standard"

    xtb_stages = manifest["stages"][2:]
    xtb_reports = [stage["report"]["xtb"] for stage in xtb_stages]
    assert [report["requested_method"] for report in xtb_reports] == expected_methods
    assert [report["task"] for report in xtb_reports] == (
        ["optimize", "singlepoint"]
        if include_gfnff
        else ["singlepoint"]
    )
    assert all(report["process_succeeded"] for report in xtb_reports)
    assert all(report["converged"] for report in xtb_reports)
    assert all(math.isfinite(report["energy_hartree"]) for report in xtb_reports)
    assert all(report["backend"]["version"] for report in xtb_reports)
    assert all(
        report["electronic_state"]["charge"] == 2 for report in xtb_reports
    )
    assert all(
        report["electronic_state"]["unpaired_electrons"] == 0
        for report in xtb_reports
    )
    assert all(
        _quality_passed(stage["report"]["post_check"][0])
        for stage in xtb_stages
    )

    final_records = read_sdf_records(final_path.read_text(encoding="utf-8"))
    assert len(final_records) == 1
    final_mol = final_records[0].mol
    assert final_mol.has_metal
    metal = next(atom for atom in final_mol.atoms if atom.symbol == _METAL)
    assert {atom.symbol for atom in metal.neighbours} == {"N", "O"}
    assert final_mol.has_3d
    assert np.isfinite(np.asarray(final_mol.coordinates, dtype=float)).all()
