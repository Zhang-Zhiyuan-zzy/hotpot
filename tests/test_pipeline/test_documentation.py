"""Executable examples from the controlled-pipeline API guide."""

from pathlib import Path

import hotpot
from hotpot.pipeline import (
    MolecularPayload,
    MolecularRecord,
    StageSpec,
    run_pipeline,
)


def test_pipeline_readme_python_example(tmp_path: Path) -> None:
    mol = hotpot.read_mol("CC")
    result = run_pipeline(
        (
            StageSpec(
                "ff",
                (
                    "--route",
                    "organic",
                    "--epochs",
                    "1",
                    "--steps-per-epoch",
                    "5",
                    "--quality",
                    "off",
                    "--seed",
                    "1",
                ),
            ),
        ),
        results_directory=tmp_path / "run-001",
        initial_payload=MolecularPayload((MolecularRecord(mol),)),
    )

    assert result.status.value == "succeeded"
    assert (result.results_directory / "final.sdf").is_file()
