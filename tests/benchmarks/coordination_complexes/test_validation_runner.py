"""Focused, non-native tests for the Plan-17 validation runner."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from .configuration import STANDARD_SETTINGS
from .validation_acceptance import GATE_IDS
from . import validation_runner as runner


def _runtime(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "hotpot_version": "0.5.4.0",
        "imported_hotpot_version": "0.5.4.0",
        "rdkit_version": runner.PINNED_RDKIT_VERSION,
        "native": True,
        "openbabel_version": "3.2.1",
        "python_version": "3.13.0",
        "python_implementation": "CPython",
        "platform": "fixture",
    }
    value.update(updates)
    return value


def _git(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {"commit": "a" * 40, "dirty": False}
    value.update(updates)
    return value


def _cohort() -> dict[str, object]:
    case = {
        "index": 1,
        "smiles": "N[Am]",
        "seed": STANDARD_SETTINGS.seed + 1,
        "settings": dict(STANDARD_SETTINGS.to_manifest()),
        "status": "cbond",
        "semantic_graph_sha256": "b" * 64,
        "graph": {"atoms": [], "bonds": [], "stereo": {}},
        "coordination": {},
    }
    return {
        "schema_version": 1,
        "canonical_sha256": "c" * 64,
        "source": {"sha256": "d" * 64, "physical_line_count": 187},
        "protocol": {
            "metal": "Am",
            "settings": dict(STANDARD_SETTINGS.to_manifest()),
        },
        "partition": {
            "cbond_count": 182,
            "no_cbond_count": 5,
            "cbond_indices": [1],
            "no_cbond_indices": [],
        },
        "cases": [case],
    }


class _Record:
    def __init__(self, value: dict[str, object]):
        self.value = value

    def to_dict(self) -> dict[str, object]:
        return self.value


def _gate_records(passed: bool) -> list[dict[str, object]]:
    return [
        {"gate_id": gate_id, "passed": passed, "diagnostics": []}
        for gate_id in GATE_IDS
    ]


def _evaluator(case, execution):
    del case, execution
    evidence = _Record(
        {
            "frames": [
                {
                    "role": "selected",
                    "energy": {"value": -12.5},
                    "rms_gradient": {"value": 0.12},
                    "max_gradient": {"value": 0.34},
                }
            ]
        }
    )
    acceptance = _Record(
        {
            "case_id": "extractant-001",
            "route": "F1",
            "passed": True,
            "gates": _gate_records(True),
            "diagnostics": [],
        }
    )
    return evidence, acceptance


def test_formal_start_requires_pinned_native_clean_runtime() -> None:
    runner.validate_formal_start(_runtime(), _git())

    with pytest.raises(RuntimeError, match="native RDKit 2026.09.1"):
        runner.validate_formal_start(_runtime(rdkit_version="2025.09.3"), _git())
    with pytest.raises(RuntimeError, match="native RDKit backend"):
        runner.validate_formal_start(_runtime(native=False), _git())
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        runner.validate_formal_start(_runtime(), _git(dirty=True))


def test_cohort_is_built_once_then_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohort = _cohort()
    calls = []

    def build(*, source_path):
        calls.append(source_path)
        return cohort

    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    monkeypatch.setattr(runner, "build_validation_cohort", build)
    monkeypatch.setattr(runner, "write_validation_cohort", write)
    monkeypatch.setattr(runner, "load_validation_cohort", lambda path: cohort)
    output = tmp_path / "cohort.json"

    assert runner.build_frozen_cohort(output) is cohort
    assert output.is_file()
    assert len(calls) == 1
    assert runner.build_frozen_cohort(output, reuse=True) is cohort
    assert len(calls) == 1


def test_run_case_uses_frozen_semantics_and_writes_only_accepted_structure(
    tmp_path: Path,
) -> None:
    cohort = _cohort()
    observed = {}

    def reconstruct(case):
        observed["case"] = case
        return "reconstructed-without-cbond"

    def execute(molecule, route, settings, case_index):
        observed["execution"] = (molecule, route, settings, case_index)

        class Archive:
            def write(self, path, *, include_sdf):
                assert include_sdf is False
                (path / "main").mkdir(parents=True)
                (path / "archive.json").write_text("{}\n", encoding="utf-8")
                (path / "main" / "coordinates.npz").write_bytes(b"trajectory")

        return SimpleNamespace(
            route=route,
            report=SimpleNamespace(
                build=None,
                optimization=SimpleNamespace(
                    elapsed_seconds=2.0,
                    trajectory=Archive(),
                ),
            ),
            proposal={"status": "SUCCESS"},
            proposal_manifest=({"coordinate_sha256": "e" * 64},),
            proposal_seconds=1.0,
            forcefield_seconds=2.0,
            elapsed_seconds=3.0,
        )

    def write(execution, acceptance, path):
        del execution, acceptance
        path.write_text("accepted selected structure\n", encoding="utf-8")
        return {"atom_row_mapping": [0, 1]}

    record = runner.run_case(
        cohort,
        case_index=1,
        route="F1",
        run_id="validation-f1",
        output_root=tmp_path,
        formal=False,
        runtime=_runtime(),
        git=_git(),
        executor=execute,
        evaluator=_evaluator,
        structure_writer=write,
        cohort_validator=lambda value: None,
        molecule_builder=reconstruct,
    )

    assert observed["case"] is cohort["cases"][0]
    assert observed["execution"] == (
        "reconstructed-without-cbond",
        "F1",
        STANDARD_SETTINGS,
        1,
    )
    assert record["passed"] is True
    assert record["status"] == "automated_passed"
    assert record["manual_review"]["status"] == "pending_human_review"
    assert record["proposal"]["frozen_manifest"][0]["coordinate_sha256"] == "e" * 64
    assert record["optimization"] == {
        "best_energy": -12.5,
        "rms_gradient": 0.12,
        "max_gradient": 0.34,
    }
    selected = Path(record["selected_structure"]["path"])
    assert selected.is_file()
    assert selected.name == "selected.sdf"
    assert record["trajectory_artifact"]["file_count"] == 2
    assert record["trajectory_artifact"]["includes_sdf"] is False
    assert len(record["trajectory_artifact"]["sha256"]) == 64
    result_path = runner.case_result_path(
        tmp_path,
        route="F1",
        run_id="validation-f1",
        case_index=1,
    )
    assert json.loads(result_path.read_text(encoding="utf-8"))["passed"] is True


def test_failure_preserves_traceback_has_all_gates_and_is_resumable(
    tmp_path: Path,
) -> None:
    cohort = _cohort()

    def fail(molecule, route, settings, case_index):
        del molecule, route, settings, case_index
        raise RuntimeError("fixture route failed")

    arguments = {
        "case_index": 1,
        "route": "C1",
        "run_id": "validation-c1",
        "output_root": tmp_path,
        "formal": False,
        "runtime": _runtime(),
        "git": _git(),
        "executor": fail,
        "cohort_validator": lambda value: None,
        "molecule_builder": lambda case: object(),
    }
    record = runner.run_case(cohort, **arguments)

    assert record["passed"] is False
    assert record["selected_structure"] is None
    assert record["error"]["type"] == "RuntimeError"
    assert "fixture route failed" in record["error"]["traceback"]
    assert [gate["gate_id"] for gate in record["acceptance"]["gates"]] == list(GATE_IDS)
    assert not list(tmp_path.glob("**/selected.sdf"))

    arguments["resume"] = True
    arguments["molecule_builder"] = lambda case: pytest.fail(
        "resume reconstructed a completed case"
    )
    assert runner.run_case(cohort, **arguments) == json.loads(
        runner.case_result_path(
            tmp_path,
            route="C1",
            run_id="validation-c1",
            case_index=1,
        ).read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("route", "entrypoint", "artifact_name"),
    (
        ("F2", "execute_provider_route", "frozen-proposals.json"),
        ("C1", "execute_whole_complex_route", "whole-complex-proposal.json"),
    ),
)
def test_routes_archive_proposal_even_when_forcefield_workflow_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    entrypoint: str,
    artifact_name: str,
) -> None:
    observed = {}

    def execute(
        molecule,
        route,
        settings,
        case_index,
        *,
        proposal_artifact_path,
    ):
        del molecule, route, settings, case_index
        proposal_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "proposal_count": 1,
            "proposals": [{"coordinate_sha256": "f" * 64}],
            "canonical_sha256": "e" * 64,
        }
        proposal_artifact_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        observed["path"] = proposal_artifact_path
        raise RuntimeError("force field failed after proposal freeze")

    monkeypatch.setattr(runner, entrypoint, execute)
    record = runner.run_case(
        _cohort(),
        case_index=1,
        route=route,
        run_id=f"validation-{route.lower()}",
        output_root=tmp_path,
        formal=False,
        runtime=_runtime(),
        git=_git(),
        cohort_validator=lambda value: None,
        molecule_builder=lambda case: object(),
    )

    artifact = observed["path"]
    assert artifact.is_file()
    assert artifact.name == artifact_name
    assert record["passed"] is False
    assert record["proposal"]["record"] is None
    assert record["proposal"]["archived_artifact"] == {
        "path": str(artifact.resolve()),
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "valid_json": True,
        "canonical_sha256": "e" * 64,
        "proposal_count": 1,
    }
    assert "force field failed after proposal freeze" in record["error"]["message"]


def _summary_record(
    *,
    run_id: str,
    route: str,
    case_index: int,
    passed: bool,
    seconds: float,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "route": route,
        "case_index": case_index,
        "passed": passed,
        "status": "automated_passed" if passed else "automated_failed",
        "acceptance": {"gates": _gate_records(passed)},
        "timing": {"runner_elapsed_seconds": seconds},
    }


def test_summary_uses_exact_denominators_timing_and_f1_repeat_classification(
    tmp_path: Path,
) -> None:
    specs = runner.validation_run_specs(
        run_prefix="round-1",
        routes=("F1",),
        f1_repeat=True,
    )
    configuration = {
        "run_specs": list(specs),
        "case_indices": [1, 2],
    }
    runner._atomic_write_json(
        tmp_path / "manifest.json",
        {
            "schema": runner.RUN_MANIFEST_SCHEMA,
            "scientific_configuration": configuration,
        },
    )
    records = (
        _summary_record(
            run_id="round-1-f1", route="F1", case_index=1, passed=True, seconds=1.0
        ),
        _summary_record(
            run_id="round-1-f1", route="F1", case_index=2, passed=False, seconds=3.0
        ),
        _summary_record(
            run_id="round-1-f1-repeat",
            route="F1",
            case_index=1,
            passed=True,
            seconds=2.0,
        ),
        _summary_record(
            run_id="round-1-f1-repeat",
            route="F1",
            case_index=2,
            passed=True,
            seconds=4.0,
        ),
    )
    for record in records:
        path = runner.case_result_path(
            tmp_path,
            route="F1",
            run_id=record["run_id"],
            case_index=record["case_index"],
        )
        runner._atomic_write_json(path, record)

    summary = runner.summarize_results(tmp_path)

    assert summary["expected_result_count"] == 4
    assert summary["observed_result_count"] == 4
    assert summary["complete"] is True
    assert summary["runs"][0]["expected_count"] == 2
    assert summary["runs"][0]["automated_accepted_count"] == 1
    assert summary["runs"][0]["gates"]["A01"]["denominator"] == 2
    timing = summary["runs"][0]["timing"]
    assert timing["observed_count"] == 2
    assert timing["p50_seconds"] == 2.0
    assert timing["p95_seconds"] == pytest.approx(2.9)
    assert timing["max_seconds"] == 3.0
    assert timing["sum_seconds"] == 4.0
    assert summary["overall_timing"]["sum_seconds"] == 10.0
    assert summary["f1_repeat"]["acceptance_classification_consistent"] is False
    assert summary["f1_repeat"]["acceptance_mismatch_case_indices"] == [2]
    assert summary["human_review"] == {
        "status": "not_performed",
        "default_replacement_authorized": False,
    }


def test_formal_matrix_requires_all_routes_cases_and_full_f1_repeat() -> None:
    specs = runner.validation_run_specs(run_prefix="formal-round")

    runner._validate_formal_matrix_contract(specs, (1, 2), (1, 2))
    with pytest.raises(ValueError, match="all 182"):
        runner._validate_formal_matrix_contract(specs, (1,), (1, 2))
    with pytest.raises(ValueError, match="complete F1 repeat"):
        runner._validate_formal_matrix_contract(specs[:-1], (1, 2), (1, 2))


def test_cli_exposes_required_commands() -> None:
    help_text = runner._parser().format_help()
    assert "build-cohort" in help_text
    assert "run-case" in help_text
    assert "run" in help_text
    assert "summarize" in help_text
