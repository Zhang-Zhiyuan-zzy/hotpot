"""Strict contracts for ordered molecular-pipeline execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest


pytestmark = pytest.mark.xfail(
    strict=True,
    reason="Phase 15 pipeline controller implementation is pending",
)


def _water():
    import hotpot

    mol = hotpot.read_mol("[H]O[H]", "smi")
    mol.coordinates = (
        (-0.75, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.75, 0.0, 0.0),
    )
    return mol


def _fake_stage_factory(operation: Callable):
    class PreparedStage:
        def __init__(self, spec) -> None:
            self.spec = spec

        def execute(self, payload, context):
            return operation(self.spec, payload, context)

    class Stage:
        def prepare(self, spec):
            return PreparedStage(spec)

    return Stage()


def test_pipeline_runs_stages_in_order_and_passes_payload_in_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        MolecularRecord,
        PipelineStatus,
        StageResult,
        StageSpec,
        StageStatus,
    )
    from hotpot.pipeline import runner

    source_payload = MolecularPayload((MolecularRecord(_water()),))
    observed: list[tuple[str, int, int]] = []
    first_output: list[MolecularPayload] = []

    def first(spec, payload, context):
        output = MolecularPayload(
            (MolecularRecord(payload.records[0].molecule.copy()),)
        )
        output.records[0].molecule.atoms[0].coordinates = (-0.80, 0.0, 0.0)
        first_output.append(output)
        observed.append((spec.name, id(payload), context.stage_index))
        return StageResult(StageStatus.SUCCEEDED, output)

    def second(spec, payload, context):
        assert payload is first_output[0]
        observed.append((spec.name, id(payload), context.stage_index))
        return StageResult(StageStatus.SUCCEEDED, payload)

    stages = {
        "first": _fake_stage_factory(first),
        "second": _fake_stage_factory(second),
    }
    monkeypatch.setattr(runner, "get_stage", stages.__getitem__)

    result = runner.run_pipeline(
        (StageSpec("first", ()), StageSpec("second", ())),
        results_directory=tmp_path / "ordered",
        initial_payload=source_payload,
    )

    assert result.status is PipelineStatus.SUCCEEDED
    assert result.payload is first_output[0]
    assert result.failed_stage_index is None
    assert [name for name, _, _ in observed] == ["first", "second"]
    assert [index for _, _, index in observed] == [0, 1]


def test_execution_failure_is_persisted_and_stops_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        MolecularRecord,
        PipelineExecutionError,
        PipelineStatus,
        StageExecutionError,
        StageResult,
        StageSpec,
        StageStatus,
    )
    from hotpot.pipeline import runner

    calls: list[str] = []

    def succeed(spec, payload, context):
        calls.append(spec.name)
        return StageResult(StageStatus.SUCCEEDED, payload)

    def fail(spec, payload, context):
        calls.append(spec.name)
        (context.stage_directory / "native.log").write_text(
            "backend stopped\n",
            encoding="utf-8",
        )
        raise StageExecutionError("backend stopped")

    def must_not_run(spec, payload, context):
        calls.append(spec.name)
        raise AssertionError("a downstream stage ran after an execution failure")

    stages = {
        "ok": _fake_stage_factory(succeed),
        "broken": _fake_stage_factory(fail),
        "later": _fake_stage_factory(must_not_run),
    }
    monkeypatch.setattr(runner, "get_stage", stages.__getitem__)
    results_directory = tmp_path / "failed"

    with pytest.raises(PipelineExecutionError) as captured:
        runner.run_pipeline(
            (
                StageSpec("ok", ()),
                StageSpec("broken", ()),
                StageSpec("later", ()),
            ),
            results_directory=results_directory,
            initial_payload=MolecularPayload((MolecularRecord(_water()),)),
        )

    assert calls == ["ok", "broken"]
    assert captured.value.stage_error.args == ("backend stopped",)
    assert captured.value.run_result.status is PipelineStatus.FAILED
    assert captured.value.run_result.failed_stage_index == 1
    assert len(captured.value.run_result.stage_results) == 1
    assert not (results_directory / "final.sdf").exists()
    failed_directories = tuple((results_directory / "stages").glob("01-broken*.failed"))
    assert len(failed_directories) == 1
    assert (failed_directories[0] / "native.log").read_text(encoding="utf-8") == (
        "backend stopped\n"
    )


def test_quality_failure_returns_terminal_evidence_and_stops_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        MolecularRecord,
        PipelineStatus,
        StageResult,
        StageSpec,
        StageStatus,
    )
    from hotpot.pipeline import runner

    terminal_payload: list[MolecularPayload] = []
    downstream_calls = 0

    def quality_fail(spec, payload, context):
        output = MolecularPayload(
            (MolecularRecord(payload.records[0].molecule.copy()),)
        )
        terminal_payload.append(output)
        return StageResult(
            StageStatus.QUALITY_FAILED,
            output,
            report={"quality_gate": "failed"},
            stderr="terminal geometry did not pass\n",
        )

    def later(spec, payload, context):
        nonlocal downstream_calls
        downstream_calls += 1
        return StageResult(StageStatus.SUCCEEDED, payload)

    stages = {
        "gate": _fake_stage_factory(quality_fail),
        "later": _fake_stage_factory(later),
    }
    monkeypatch.setattr(runner, "get_stage", stages.__getitem__)
    results_directory = tmp_path / "quality-failed"

    result = runner.run_pipeline(
        (StageSpec("gate", ()), StageSpec("later", ())),
        results_directory=results_directory,
        initial_payload=MolecularPayload((MolecularRecord(_water()),)),
    )

    assert result.status is PipelineStatus.FAILED
    assert result.payload is terminal_payload[0]
    assert result.failed_stage_index == 0
    assert len(result.stage_results) == 1
    assert result.stage_results[0].status is StageStatus.QUALITY_FAILED
    assert downstream_calls == 0
    assert not (results_directory / "final.sdf").exists()
    manifest = json.loads(
        (results_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["failed_stage_index"] == 0


def test_repeated_xtb_stages_keep_order_and_distinct_artifact_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        MolecularRecord,
        PipelineStatus,
        StageResult,
        StageSpec,
        StageStatus,
    )
    from hotpot.pipeline import runner

    observed_argv: list[tuple[str, ...]] = []

    def execute(spec, payload, context):
        observed_argv.append(spec.argv)
        (context.stage_directory / "method.txt").write_text(
            spec.argv[1],
            encoding="utf-8",
        )
        return StageResult(StageStatus.SUCCEEDED, payload)

    stage = _fake_stage_factory(execute)
    monkeypatch.setattr(runner, "get_stage", lambda name: stage)
    results_directory = tmp_path / "repeated-xtb"

    result = runner.run_pipeline(
        (
            StageSpec("xtb", ("--method", "gfnff")),
            StageSpec("xtb", ("--method", "gfn2")),
        ),
        results_directory=results_directory,
        initial_payload=MolecularPayload((MolecularRecord(_water()),)),
    )

    assert result.status is PipelineStatus.SUCCEEDED
    assert observed_argv == [
        ("--method", "gfnff"),
        ("--method", "gfn2"),
    ]
    stage_directories = sorted(
        path for path in (results_directory / "stages").iterdir() if path.is_dir()
    )
    assert len(stage_directories) == 2
    assert stage_directories[0] != stage_directories[1]
    assert (stage_directories[0] / "method.txt").read_text(encoding="utf-8") == (
        "gfnff"
    )
    assert (stage_directories[1] / "method.txt").read_text(encoding="utf-8") == (
        "gfn2"
    )
