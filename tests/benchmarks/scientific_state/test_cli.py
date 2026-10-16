"""CLI tests for the scientific-state pre-activation baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .__main__ import main


def test_cli_runs_smiles_baseline_without_claiming_activation(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "input.smi"
    source.write_text("[Eu]N unknown-metal-bond\n", encoding="utf-8")
    output = tmp_path / "output"

    assert main(
        (
            "--smiles-corpus",
            str(source),
            "--output",
            str(output),
            "--no-replay",
        )
    ) == 0

    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["requested_count"] == 1
    assert summary["resolved_count"] == 1
    assert summary["activation_coverage_is_final"] is False
    assert summary["replay_checked_count"] == 0
    assert "pre-activation baseline" in capsys.readouterr().out


def test_cli_applies_only_digest_bound_manifest_coordination_bonds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "complex.smi"
    source.write_text("[Eu].N\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "topology.jsonl"
    manifest.write_text(
        "{"
        '"case_id":"eu-n",'
        '"format":"smi",'
        '"store_path":"complex.smi",'
        f'"source_sha256":"{digest}",'
        '"coordination_bonds":[[0,1]]'
        "}\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    assert main(
        (
            "--topology-manifest",
            str(manifest),
            "--structure-root",
            str(tmp_path),
            "--output",
            str(output),
            "--no-replay",
        )
    ) == 0

    record = json.loads(
        (output / "records.jsonl").read_text(encoding="utf-8")
    )
    assert record["coordination_bond_count"] == 1
    assert record["unknown_bond_count"] == 0
    assert record["profile_finding_codes"] == [
        "PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE"
    ]
