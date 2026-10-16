"""CLI tests for the scientific-state pre-activation baseline."""

from __future__ import annotations

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
