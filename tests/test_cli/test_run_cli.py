"""Strict contracts for the controlled ``hotpot run`` command.

These tests intentionally remain strict expected failures until the Phase 15
controller implementation lands.  Imports of the planned package stay inside
the tests so this Phase 14 contract file remains collectable on the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.xfail(
    strict=True,
    reason="Phase 15 controlled molecular pipeline is not implemented yet",
)


def _write_workflow(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_inline_and_json_workflows_normalize_to_identical_stage_specs(
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.cli import load_json_workflow, parse_inline_stages
    from hotpot.pipeline.contracts import StageSpec

    inline = parse_inline_stages(
        (
            "cbond",
            "Eu",
            "O=C(O)C",
            "::",
            "ff",
            "--route",
            "complex",
            "--forcefield",
            "uff",
            "::",
            "xtb",
            "--method",
            "gfnff",
            "--task",
            "optimize",
            "::",
            "xtb",
            "--method",
            "gfn2",
            "--task",
            "optimize",
        )
    )
    workflow_path = _write_workflow(
        tmp_path / "workflow.json",
        {
            "stages": [
                {"name": "cbond", "argv": ["Eu", "O=C(O)C"]},
                {
                    "name": "ff",
                    "argv": ["--route", "complex", "--forcefield", "uff"],
                },
                {
                    "name": "xtb",
                    "argv": ["--method", "gfnff", "--task", "optimize"],
                },
                {
                    "name": "xtb",
                    "argv": ["--method", "gfn2", "--task", "optimize"],
                },
            ]
        },
    )

    expected = (
        StageSpec("cbond", ("Eu", "O=C(O)C")),
        StageSpec(
            "ff",
            ("--route", "complex", "--forcefield", "uff"),
        ),
        StageSpec(
            "xtb",
            ("--method", "gfnff", "--task", "optimize"),
        ),
        StageSpec(
            "xtb",
            ("--method", "gfn2", "--task", "optimize"),
        ),
    )
    assert inline == expected
    assert load_json_workflow(workflow_path) == expected


@pytest.mark.parametrize(
    "tokens",
    (
        (),
        ("::", "ff"),
        ("ff", "::"),
        ("ff", "::", "::", "xtb"),
    ),
)
def test_inline_workflow_rejects_empty_stages(tokens) -> None:
    from hotpot.pipeline.cli import parse_inline_stages
    from hotpot.pipeline.contracts import PipelineDefinitionError

    with pytest.raises(PipelineDefinitionError):
        parse_inline_stages(tokens)


def test_json_workflow_rejects_an_empty_stage_sequence(tmp_path: Path) -> None:
    from hotpot.pipeline.cli import load_json_workflow
    from hotpot.pipeline.contracts import PipelineDefinitionError

    workflow_path = _write_workflow(
        tmp_path / "empty.json",
        {"stages": []},
    )
    with pytest.raises(PipelineDefinitionError):
        load_json_workflow(workflow_path)


def test_only_the_exact_double_colon_token_is_a_stage_separator() -> None:
    from hotpot.pipeline.cli import parse_inline_stages
    from hotpot.pipeline.contracts import StageSpec

    assert parse_inline_stages(("ff", ":", "xtb", "left::right")) == (
        StageSpec("ff", (":", "xtb", "left::right")),
    )


@pytest.mark.parametrize("stage_name", ("shell", "python", "unknown-stage"))
def test_inline_workflow_rejects_unknown_stage_names(stage_name: str) -> None:
    from hotpot.pipeline.cli import parse_inline_stages
    from hotpot.pipeline.contracts import PipelineDefinitionError

    with pytest.raises(PipelineDefinitionError):
        parse_inline_stages((stage_name, "--version"))


def test_json_workflow_rejects_unknown_stage_names(tmp_path: Path) -> None:
    from hotpot.pipeline.cli import load_json_workflow
    from hotpot.pipeline.contracts import PipelineDefinitionError

    workflow_path = _write_workflow(
        tmp_path / "unknown.json",
        {"stages": [{"name": "unknown-stage", "argv": []}]},
    )
    with pytest.raises(PipelineDefinitionError):
        load_json_workflow(workflow_path)


@pytest.mark.parametrize(
    "document",
    (
        '{"stages": [], "stages": []}',
        '{"stages": [{"name": "ff", "name": "xtb", "argv": []}]}',
        '{"stages": [{"name": "ff", "argv": [], "argv": []}]}',
    ),
)
def test_json_workflow_rejects_duplicate_keys(
    tmp_path: Path,
    document: str,
) -> None:
    from hotpot.pipeline.cli import load_json_workflow
    from hotpot.pipeline.contracts import PipelineDefinitionError

    workflow_path = tmp_path / "duplicate.json"
    workflow_path.write_text(document, encoding="utf-8")
    with pytest.raises(PipelineDefinitionError):
        load_json_workflow(workflow_path)


@pytest.mark.parametrize(
    "payload",
    (
        {
            "stages": [{"name": "ff", "argv": []}],
            "version": 1,
        },
        {
            "stages": [
                {"name": "ff", "argv": [], "environment": {"A": "B"}}
            ]
        },
    ),
)
def test_json_workflow_rejects_extra_fields(
    tmp_path: Path,
    payload,
) -> None:
    from hotpot.pipeline.cli import load_json_workflow
    from hotpot.pipeline.contracts import PipelineDefinitionError

    workflow_path = _write_workflow(tmp_path / "extra.json", payload)
    with pytest.raises(PipelineDefinitionError):
        load_json_workflow(workflow_path)


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {},
        {"stages": {}},
        {"stages": [1]},
        {"stages": [{"name": 1, "argv": []}]},
        {"stages": [{"name": "ff", "argv": "--route complex"}]},
        {"stages": [{"name": "ff", "argv": ["--route", 1]}]},
    ),
)
def test_json_workflow_rejects_wrong_schema_types(
    tmp_path: Path,
    payload,
) -> None:
    from hotpot.pipeline.cli import load_json_workflow
    from hotpot.pipeline.contracts import PipelineDefinitionError

    workflow_path = _write_workflow(tmp_path / "wrong-type.json", payload)
    with pytest.raises(PipelineDefinitionError):
        load_json_workflow(workflow_path)


def test_inline_and_json_workflow_sources_are_mutually_exclusive(
    tmp_path: Path,
    capsys,
) -> None:
    from hotpot.pipeline import cli

    workflow_path = _write_workflow(
        tmp_path / "workflow.json",
        {"stages": [{"name": "ff", "argv": []}]},
    )
    results_directory = tmp_path / "results"

    status = cli.main(
        [
            str(workflow_path),
            "--results-dir",
            str(results_directory),
            "--",
            "ff",
        ]
    )

    assert status == 2
    assert "cannot" in capsys.readouterr().err.lower()
    assert not results_directory.exists()


def test_stage_arguments_are_never_interpreted_by_a_shell(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline import cli
    from hotpot.pipeline.contracts import PipelineRunResult, PipelineStatus

    marker = tmp_path / "shell-was-run"
    ligand = f"C; touch {marker}"
    captured_specs = []

    def fake_run_pipeline(specs, *, results_directory, initial_payload=None):
        captured_specs.extend(specs)
        return PipelineRunResult(
            status=PipelineStatus.SUCCEEDED,
            payload=None,
            results_directory=results_directory,
            stage_results=(),
        )

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    status = cli.main(
        [
            "--results-dir",
            str(tmp_path / "results"),
            "--",
            "cbond",
            "Eu",
            ligand,
        ]
    )

    assert status == 0
    assert captured_specs[0].argv == ("Eu", ligand)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("pipeline_status", "expected_exit_status"),
    (("succeeded", 0), ("failed", 1)),
)
def test_cli_maps_pipeline_status_to_process_exit_status(
    monkeypatch,
    tmp_path: Path,
    pipeline_status: str,
    expected_exit_status: int,
) -> None:
    from hotpot.pipeline import cli
    from hotpot.pipeline.contracts import PipelineRunResult, PipelineStatus

    def fake_run_pipeline(specs, *, results_directory, initial_payload=None):
        assert specs
        return PipelineRunResult(
            status=PipelineStatus(pipeline_status),
            payload=None,
            results_directory=results_directory,
            stage_results=(),
        )

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    assert cli.main(
        [
            "--results-dir",
            str(tmp_path / pipeline_status),
            "--",
            "ff",
            "--route",
            "organic",
        ]
    ) == expected_exit_status


def test_preflight_failure_does_not_create_results_directory(
    tmp_path: Path,
    capsys,
) -> None:
    from hotpot.pipeline import cli

    results_directory = tmp_path / "must-not-exist"
    status = cli.main(
        [
            "--results-dir",
            str(results_directory),
            "--",
            "ff",
            "--route",
            "not-a-route",
        ]
    )

    assert status == 2
    assert capsys.readouterr().err
    assert not results_directory.exists()


def test_help_and_documentation_do_not_require_a_workflow(capsys) -> None:
    from hotpot import __main__ as hotpot_main

    with pytest.raises(SystemExit) as help_exit:
        hotpot_main.main(["run", "--help"])
    assert help_exit.value.code == 0
    help_output = capsys.readouterr().out
    assert "usage: hotpot run" in help_output
    assert "--results-dir" in help_output

    with pytest.raises(SystemExit) as doc_exit:
        hotpot_main.main(["run", "--doc"])
    assert doc_exit.value.code == 0
    documentation = capsys.readouterr().out
    assert "hotpot run" in documentation
    assert "::" in documentation
    assert "results" in documentation.lower()
