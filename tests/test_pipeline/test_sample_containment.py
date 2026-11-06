"""Per-input denominators and exact sample control for FF/xTB adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _molecule():
    import hotpot

    mol = hotpot.read_mol("[H]O[H]", "smi")
    mol.coordinates = np.asarray(((-0.75, 0.0, 0.0), (0.0, 0.5, 0.0), (0.75, 0.0, 0.0)))
    return mol


def _quality(*failed_names, severity="error"):
    from hotpot.cheminfo.operations.forcefields.contracts import AcceptanceCheck, ForceFieldValidationReport

    checks = (() if "finite_coordinates" in failed_names else (AcceptanceCheck("finite_coordinates", True),)) + tuple(
        AcceptanceCheck(name, False, severity=severity, measured=0.1, threshold=0.5,
                        atom_indices=(0, 1), message="observed distance")
        for name in failed_names
    )
    return ForceFieldValidationReport(not failed_names or severity == "warning", checks)


@dataclass(frozen=True)
class _FFReport:
    quality_report: object
    trajectory: None = None


def _xtb_report(*, converged=True, method="gfn2", return_code=0):
    from hotpot.plugins._harness import NativeProcessResult, ProcessProvenance
    from hotpot.plugins.xtb.contracts import XTBBackendInfo, XTBMethod, XTBRunReport, XTBTask

    provenance = ProcessProvenance(
        executable=Path("/fake/xtb"), executable_sha256="a" * 64,
        argv=("xtb", "input.xyz"), cwd=Path("/fake/xtb-workspace"),
        return_code=return_code, stdout="native stdout", stderr="native stderr",
        elapsed_seconds=0.1, artifacts=(),
    )
    backend = XTBBackendInfo(
        provenance.executable, "6.7.1", None, provenance.executable_sha256,
        NativeProcessResult(("xtb", "--version"), provenance.cwd, 0, "xtb version 6.7.1", "", 0.01),
        86, 86,
    )
    method = XTBMethod(method)
    return XTBRunReport(
        backend_info=backend, effective_method=method, task=XTBTask.OPTIMIZE,
        charge=0, unpaired_electrons=None if method is XTBMethod.GFNFF else 0,
        converged=converged, artifacts={}, provenance=provenance,
        energy_hartree=-76.0, coordinates_committed=True,
    )


def _state_projection():
    from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts
    from hotpot.cheminfo.operations.calculation.scientific_state.state_resolver import resolve_scientific_state
    from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import ScientificOperationRequirements, ScientificQuantity
    from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest

    state = resolve_scientific_state(
        extract_molecular_facts(_molecule()), StateRequest(net_charge=0, two_s=0),
        requirements=ScientificOperationRequirements(
            "sample-containment-test",
            frozenset((ScientificQuantity.NET_CHARGE, ScientificQuantity.ELECTRONIC_STATE)),
            frozenset(),
        ),
    )
    projection = SimpleNamespace(
        scientific_state_identity=state.execution_decision.selected_state_identity,
        projection_convention="test-projection",
        execution_decision=state.execution_decision,
    )
    return state, projection


def _prepare(kind, monkeypatch, tmp_path, *, argv=(), quality=None, errors=None, convergence=True):
    from hotpot.pipeline.contracts import MolecularPayload, MolecularRecord, StageContext, StageSpec

    records = tuple(MolecularRecord(_molecule()) for _ in range(3))
    calls = []
    errors = {} if errors is None else errors
    quality = _quality() if quality is None else quality
    if kind == "ff":
        from hotpot.cheminfo.operations.forcefields import stage

        def run(molecule, options, **kwargs):
            index = len(calls)
            calls.append(molecule)
            if index in errors:
                raise errors[index]
            return _FFReport(quality)

        monkeypatch.setattr(stage, "execute_forcefield_route", run)
    else:
        from hotpot.plugins.xtb import stage

        state, projection = _state_projection()

        def run(record, options, work_directory):
            index = len(calls)
            calls.append(record.molecule)
            if index in errors:
                raise errors[index]
            return _xtb_report(converged=convergence), state, projection

        monkeypatch.setattr(stage, "_run_record", run)
        monkeypatch.setattr(stage, "_post_check", lambda mol, level: quality)
    spec = StageSpec(kind, argv)
    stage_directory = tmp_path / kind
    stage_directory.mkdir()
    context = StageContext(tmp_path, stage_directory, 0, spec)
    return stage.get_stage().prepare(spec), MolecularPayload(records), context, calls


def _known_error(kind):
    if kind == "ff":
        from hotpot.cheminfo.operations.forcefields.contracts import ForceFieldSetupError

        return ForceFieldSetupError("unavailable"), "FF-SETUP-UNAVAILABLE"
    from hotpot.plugins.xtb.contracts import XTBResultError

    return XTBResultError("unavailable", _xtb_report()), "XTB-RESULT-UNAVAILABLE"


@pytest.mark.parametrize("kind", ("ff", "xtb"))
def test_middle_known_failure_continues_to_trailing_success(kind, monkeypatch, tmp_path):
    from hotpot.pipeline.contracts import StageStatus

    error, code = _known_error(kind)
    prepared, payload, context, calls = _prepare(kind, monkeypatch, tmp_path, errors={1: error})
    result = prepared.execute(payload, context)
    assert calls == [record.molecule for record in payload.records]
    assert result.status is StageStatus.SUCCEEDED
    assert tuple(record.molecule for record in result.payload.records) == (payload.records[0].molecule, payload.records[2].molecule)
    report = result.report
    assert tuple(report[key] for key in ("sample_outcomes_version", "input_count", "attempted_count", "output_count", "unavailable_count", "sample_break_count", "not_attempted_count")) == (1, 3, 3, 2, 1, 0, 0)
    rows = report["sample_outcomes"]
    assert [row["input_index"] for row in rows] == [0, 1, 2]
    assert [row["output_index"] for row in rows] == [0, None, 1]
    assert [row["outcome"] for row in rows] == ["returned", "unavailable", "returned"]
    assert rows[1]["failure"]["code"] == code
    assert rows[1]["failure"]["type"] == type(error).__name__
    assert rows[1]["failure"]["message"] == "unavailable"
    if kind == "xtb":
        assert [row["report_index"] for row in rows] == [0, 1, 2]
        assert report["xtb"][rows[1]["report_index"]]["provenance"]["stdout"] == "native stdout"
        assert "native_report" not in rows[1]["failure"]
        assert "record 1 stdout" in (context.stage_directory / "native.log").read_text()
    else:
        assert "report_index" not in rows[1]
        assert [rows[i]["report_index"] for i in (0, 2)] == [0, 1]


@pytest.mark.parametrize("error_type", ("XTBExecutionError", "XTBResultError"))
def test_xtb_single_unavailable_sample_retains_one_indexed_native_report(error_type, monkeypatch, tmp_path):
    from hotpot.pipeline.contracts import MolecularPayload, StageStatus
    from hotpot.plugins.xtb import contracts

    native_failure = error_type == "XTBExecutionError"
    native_report = _xtb_report(converged=not native_failure, return_code=17 if native_failure else 0)
    error = getattr(contracts, error_type)("unavailable", native_report)
    prepared, payload, context, calls = _prepare("xtb", monkeypatch, tmp_path, errors={0: error})
    result = prepared.execute(MolecularPayload(payload.records[:1]), context)

    assert len(calls) == 1
    assert result.status is StageStatus.QUALITY_FAILED
    assert result.payload.records == ()
    report = result.report
    assert (report["input_count"], report["attempted_count"], report["output_count"], report["unavailable_count"]) == (1, 1, 0, 1)
    row, = report["sample_outcomes"]
    assert row["input_index"] == 0
    assert row["output_index"] is None
    assert row["outcome"] == "unavailable"
    assert row["report_index"] == 0
    assert row["failure"] == {
        "code": error.sample_failure_code,
        "type": error_type,
        "message": "unavailable",
    }
    assert isinstance(report["xtb"], dict)
    assert report["xtb"]["provenance"]["stdout"] == "native stdout"
    assert report["xtb"]["provenance"]["stderr"] == "native stderr"
    assert report["xtb"]["provenance"]["return_code"] == (17 if native_failure else 0)
    log = (context.stage_directory / "native.log").read_text()
    assert "record 0 stdout" in log and "native stdout" in log
    assert "record 0 stderr" in log and "native stderr" in log


def test_xtb_summary_preserves_state_binding_and_projection_decision(monkeypatch, tmp_path):
    prepared, payload, context, _ = _prepare("xtb", monkeypatch, tmp_path)
    result = prepared.execute(payload, context)

    for record, report in zip(result.payload.records, result.report["xtb"]):
        assert {"requested_method", "effective_method", "task", "process_succeeded"}.isdisjoint(report)
        state = record.scientific_state
        assert report["scientific_state"] == {
            "facts_identity": state.facts.identity.chemical_identity_digest,
            "resolution_status": state.scientific_resolution.status.value,
            "decision_action": state.execution_decision.action.value,
            "selected_state_identity": state.execution_decision.selected_state_identity.digest,
        }
        assert report["backend_projection"] == {
            "projection_convention": "test-projection",
            "decision_source": state.execution_decision.source.value,
        }


@pytest.mark.parametrize("kind", ("ff", "xtb"))
@pytest.mark.parametrize("unknown_kind", ("value", "unrecognized-code"))
def test_unknown_error_with_same_text_is_fatal_and_preserves_cause(kind, unknown_kind, monkeypatch, tmp_path):
    from hotpot.pipeline.contracts import StageExecutionError

    error = ValueError("unavailable")
    if unknown_kind == "unrecognized-code":
        error, _ = _known_error(kind)
        error.sample_failure_code = "UNKNOWN"
    prepared, payload, context, calls = _prepare(kind, monkeypatch, tmp_path, errors={1: error})
    with pytest.raises(StageExecutionError) as caught:
        prepared.execute(payload, context)
    assert caught.value.__cause__ is error
    assert len(calls) == 2


@pytest.mark.parametrize("kind", ("ff", "xtb"))
@pytest.mark.parametrize("severity", ("error", "warning"))
def test_nonhard_findings_retain_finite_output_and_unchanged_check(kind, severity, monkeypatch, tmp_path):
    from hotpot.pipeline.contracts import StageStatus

    quality = _quality("atom_too_close", severity=severity)
    prepared, payload, context, _ = _prepare(kind, monkeypatch, tmp_path, quality=quality)
    result = prepared.execute(payload, context)
    assert result.status is StageStatus.SUCCEEDED
    assert len(result.payload.records) == 3
    finding = result.report["sample_outcomes"][0]["findings"][0]
    assert finding == {"code": f"{kind.upper()}:atom_too_close", "name": "atom_too_close", "passed": False, "severity": severity, "measured": 0.1, "threshold": 0.5, "atom_indices": (0, 1), "bond_indices": (), "message": "observed distance"}


@pytest.mark.parametrize("kind", ("ff", "xtb"))
@pytest.mark.parametrize("scope", ("sample", "batch", "both"))
def test_explicit_break_retains_observation_and_complete_denominator(kind, scope, monkeypatch, tmp_path):
    from hotpot.pipeline.contracts import StageStatus

    code = f"{kind.upper()}:atom_too_close"
    argv = ("--sample-break-on", code) if scope == "sample" else ("--batch-break-on", code)
    if scope == "both":
        argv += ("--sample-break-on", code)
    prepared, payload, context, calls = _prepare(kind, monkeypatch, tmp_path, argv=argv, quality=_quality("atom_too_close"))
    result = prepared.execute(payload, context)
    rows = result.report["sample_outcomes"]
    assert result.status is StageStatus.QUALITY_FAILED
    if scope == "sample":
        assert len(calls) == 3
        assert not result.break_pipeline
        assert result.report["sample_break_count"] == 3
        assert [row["outcome"] for row in rows] == ["sample-break"] * 3
        assert result.payload.records == ()
        for index in range(3):
            observation = context.stage_directory / "observations" / f"{index:04d}.sdf"
            assert rows[index]["observation_path"] == f"observations/{index:04d}.sdf"
            assert observation.is_file()
            assert "$$$$" in observation.read_text()
        assert len(result.artifacts) >= 3
    else:
        assert len(calls) == 1
        assert result.break_pipeline
        assert rows[0]["break_scope"] == "batch"
        assert [row["outcome"] for row in rows] == ["returned", "not-attempted", "not-attempted"]
        assert [row["output_index"] for row in rows] == [0, None, None]
        assert len(result.payload.records) == 1
        assert result.payload.records[0].molecule is payload.records[0].molecule
        assert result.report["attempted_count"] == 1
        assert result.report["not_attempted_count"] == 2


@pytest.mark.parametrize("kind", ("ff", "xtb"))
def test_required_frame_failure_omits_output_but_retains_actual_report(kind, monkeypatch, tmp_path):
    from hotpot.pipeline.contracts import StageStatus

    prepared, payload, context, calls = _prepare(kind, monkeypatch, tmp_path, quality=_quality("finite_coordinates"))
    result = prepared.execute(payload, context)
    assert len(calls) == 3
    assert result.status is StageStatus.QUALITY_FAILED
    assert result.payload.records == ()
    assert result.report["unavailable_count"] == 3
    assert all("observation_path" not in row for row in result.report["sample_outcomes"])
    assert not (context.stage_directory / "observations").exists()
    assert [row["report_index"] for row in result.report["sample_outcomes"]] == [0, 1, 2]
    method_reports = result.report["forcefield" if kind == "ff" else "xtb"]
    assert len(method_reports) == 3


@pytest.mark.parametrize("kind", ("ff", "xtb"))
def test_break_code_parser_is_exact_and_repeatable(kind, monkeypatch, tmp_path):
    from hotpot.pipeline.contracts import PipelineDefinitionError, StageSpec

    code = f"{kind.upper()}:atom_too_close"
    prepared, _, _, _ = _prepare(kind, monkeypatch, tmp_path, argv=("--sample-break-on", code, "--sample-break-on", code))
    options = prepared if kind == "ff" else prepared.options
    assert options.sample_break_on == (code, code)
    assert options.batch_break_on == ()
    from hotpot.cheminfo.operations.forcefields import stage as ff
    from hotpot.plugins.xtb import stage as xtb
    stage = ff if kind == "ff" else xtb
    with pytest.raises(PipelineDefinitionError):
        stage.get_stage().prepare(StageSpec(kind, ("--sample-break-on", code.upper())))


def test_xtb_false_convergence_is_a_finding_with_usable_output(monkeypatch, tmp_path):
    from hotpot.pipeline.contracts import StageStatus

    prepared, payload, context, _ = _prepare("xtb", monkeypatch, tmp_path, convergence=False)
    result = prepared.execute(payload, context)
    assert result.status is StageStatus.SUCCEEDED
    assert len(result.payload.records) == 3
    assert result.report["sample_outcomes"][0]["findings"][0]["code"] == "XTB:convergence"


@pytest.mark.parametrize("anchor", ("initial", "cbond"))
def test_duplicate_content_composes_ff_xtb_by_position_with_persisted_anchor(anchor, monkeypatch, tmp_path):
    from hotpot.cheminfo.operations.forcefields import stage as ff
    from hotpot.plugins.xtb import stage as xtb
    from hotpot.pipeline import runner
    from hotpot.pipeline.contracts import MolecularPayload, MolecularRecord, PipelineStatus, StageResult, StageSpec, StageStatus
    from hotpot.cheminfo.operations.forcefields.contracts import ForceFieldSetupError
    from hotpot.plugins.xtb.contracts import XTBResultError

    molecules = tuple(_molecule() for _ in range(4))
    records = tuple(MolecularRecord(molecule) for molecule in molecules)
    ff_calls = []
    xtb_calls = []
    state, projection = _state_projection()

    def run_ff(mol, options, **kwargs):
        index = len(ff_calls)
        ff_calls.append(mol)
        if index == 1:
            raise ForceFieldSetupError("middle FF unavailable")
        return _FFReport(_quality())

    def run_xtb(record, options, work_directory):
        index = len(xtb_calls)
        xtb_calls.append(record.molecule)
        if index == 1:
            raise XTBResultError("middle xTB unavailable", _xtb_report())
        return _xtb_report(), state, projection

    class CBond:
        def prepare(self, spec):
            return self

        def execute(self, payload, context):
            assert not payload.records
            return StageResult(StageStatus.SUCCEEDED, MolecularPayload(records), report={"cbond": {"structures": tuple({"rank": index + 1} for index in range(4))}})

    monkeypatch.setattr(ff, "execute_forcefield_route", run_ff)
    monkeypatch.setattr(xtb, "_run_record", run_xtb)
    monkeypatch.setattr(xtb, "_post_check", lambda mol, level: None)
    stages = {"ff": ff.get_stage(), "xtb": xtb.get_stage(), "cbond": CBond()}
    monkeypatch.setattr(runner, "get_stage", stages.__getitem__)
    specs = (StageSpec("ff", ()), StageSpec("xtb", ()))
    payload = MolecularPayload(records)
    if anchor == "cbond":
        specs = (StageSpec("cbond", ()),) + specs
        payload = MolecularPayload(())
    results_directory = tmp_path / "run"
    result = runner.run_pipeline(specs, results_directory=results_directory, initial_payload=payload)
    assert result.status is PipelineStatus.SUCCEEDED
    assert ff_calls == list(molecules)
    assert xtb_calls == [molecules[0], molecules[2], molecules[3]]
    assert tuple(record.molecule for record in result.payload.records) == (molecules[0], molecules[3])
    first, second = result.stage_results[-2:]
    ff_rows = first.report["sample_outcomes"]
    xtb_rows = second.report["sample_outcomes"]
    # Positional edge composition attributes equal-content molecules to ordinals 0 and 3.
    upstream = {row["output_index"]: row["input_index"] for row in ff_rows if row["output_index"] is not None}
    assert [upstream[row["input_index"]] for row in xtb_rows if row["output_index"] is not None] == [0, 3]
    stage_offset = int(anchor == "cbond")
    ff_directory = results_directory / "stages" / f"{stage_offset:02d}-ff"
    xtb_directory = results_directory / "stages" / f"{stage_offset + 1:02d}-xtb"
    assert (ff_directory / "output.sdf").read_text().count("$$$$") == 3
    assert (xtb_directory / "output.sdf").read_text().count("$$$$") == 2
    assert json.loads((ff_directory / "report.json").read_text())["input_count"] == 4
    xtb_report = json.loads((xtb_directory / "report.json").read_text())
    failure_row = xtb_report["sample_outcomes"][1]
    assert "native_report" not in failure_row["failure"]
    assert xtb_report["xtb"][failure_row["report_index"]]["provenance"]["stdout"] == "native stdout"
    manifest = json.loads((results_directory / "manifest.json").read_text())
    assert manifest["stages"][-1]["report"] == xtb_report
    if anchor == "cbond":
        generated = results_directory / "stages" / "00-cbond"
        assert (generated / "output.sdf").read_text().count("$$$$") == 4
        assert [item["rank"] for item in json.loads((generated / "report.json").read_text())["cbond"]["structures"]] == [1, 2, 3, 4]


@pytest.mark.parametrize("kind", ("ff", "xtb"))
@pytest.mark.parametrize("scope", ("sample", "batch"))
def test_break_on_known_unavailable_code_never_manufactures_output(kind, scope, monkeypatch, tmp_path):
    from hotpot.pipeline.contracts import StageStatus

    error, code = _known_error(kind)
    prepared, payload, context, calls = _prepare(kind, monkeypatch, tmp_path, errors={1: error}, argv=(f"--{scope}-break-on", code))
    result = prepared.execute(payload, context)
    rows = result.report["sample_outcomes"]
    assert rows[1]["outcome"] == "unavailable"
    assert rows[1]["output_index"] is None
    assert rows[1]["break_scope"] == scope
    assert result.report["sample_break_count"] == 0
    if scope == "sample":
        assert len(calls) == 3
        assert result.status is StageStatus.SUCCEEDED
        assert not result.break_pipeline
        assert len(result.payload.records) == 2
    else:
        assert len(calls) == 2
        assert result.status is StageStatus.QUALITY_FAILED
        assert result.break_pipeline
        assert len(result.payload.records) == 1
        assert rows[2]["outcome"] == "not-attempted"


def test_ff_error_trajectories_and_worker_diagnostics_are_retained(monkeypatch, tmp_path):
    from hotpot.cheminfo.operations.forcefields.contracts import BuildWorkerError

    class Archive:
        def write(self, path):
            path.mkdir(parents=True)
            (path / "trajectory.json").write_text('{"frames": ["actual retained frame"]}')

    error = BuildWorkerError("NativeSetupError", "unavailable", "worker original traceback")
    error.sample_failure_code = "FF-SETUP-UNAVAILABLE"
    error.trajectory = Archive()
    error.ligand_build_attempts = (Archive(),)
    prepared, payload, context, _ = _prepare("ff", monkeypatch, tmp_path, errors={1: error})
    result = prepared.execute(payload, context)
    failure = result.report["sample_outcomes"][1]["failure"]
    assert failure["worker_error"] == {"type": "NativeSetupError", "message": "unavailable", "traceback": "worker original traceback"}
    assert failure["trajectory_available"] is True
    assert failure["ligand_build_attempt_count"] == 1
    assert (context.stage_directory / "errors/0001/trajectory/trajectory.json").is_file()
    assert (context.stage_directory / "errors/0001/ligand-0000/trajectory.json").is_file()
    assert len(result.artifacts) == 2


def test_xtb_post_check_uses_only_existing_geometry_acceptance_owner(monkeypatch):
    from hotpot.cheminfo.operations.forcefields import acceptance
    from hotpot.plugins.xtb import stage

    mol = _molecule()
    expected = _quality("atom_too_close")
    calls = []

    def evaluate(molecule, *, level):
        calls.append((molecule, level))
        return expected

    monkeypatch.setattr(acceptance, "_evaluate_geometry_acceptance", evaluate)
    assert stage._post_check(mol, "strict") is expected
    assert calls == [(mol, "strict")]
    assert stage._post_check(mol, "off") is None
    assert len(calls) == 1


def test_xtb_local_state_failure_retains_scientific_bundle_and_projection_findings(monkeypatch, tmp_path):
    from hotpot.cheminfo.adapters.calculation.contracts import BackendIdentity
    from hotpot.cheminfo.operations.calculation.scientific_state.execution_selection import (
        BackendProjectionFinding, BackendProjectionResult,
        BackendProjectionStatus, BackendStateRequest,
    )
    from hotpot.plugins.xtb.state_resolution import XTBStateResolutionError

    state, _ = _state_projection()
    projection = BackendProjectionResult(
        BackendProjectionStatus.BACKEND_UNSUPPORTED,
        BackendIdentity("test-unsupported-context"), BackendStateRequest(), None,
        (BackendProjectionFinding("test-domain", "outside this backend domain"),),
    )
    error = XTBStateResolutionError(state, projection)
    prepared, payload, context, calls = _prepare("xtb", monkeypatch, tmp_path, errors={1: error})
    result = prepared.execute(payload, context)
    assert len(calls) == 3
    failure = result.report["sample_outcomes"][1]["failure"]
    assert failure["code"] == "XTB-STATE-UNAVAILABLE"
    assert failure["scientific_state"]["facts"]["identity"]["chemical_identity_digest"] == state.facts.identity.chemical_identity_digest
    assert failure["projection_result"]["findings"][0]["code"] == "test-domain"
    assert failure["projection_result"]["status"] == "backend_unsupported"


@pytest.mark.parametrize("kind", ("ff", "xtb"))
@pytest.mark.parametrize("error_type", (RuntimeError, TypeError))
def test_other_exception_types_propagate_exact_object_without_later_attempt(kind, error_type, monkeypatch, tmp_path):
    error = error_type("unknown execution failure")
    prepared, payload, context, calls = _prepare(kind, monkeypatch, tmp_path, errors={1: error})
    with pytest.raises(error_type) as caught:
        prepared.execute(payload, context)
    assert caught.value is error
    assert len(calls) == 2


@pytest.mark.parametrize("metric", ("finite_final_energy", "finite_rms_gradient", "finite_max_gradient"))
def test_ff_unavailable_required_metric_saves_defined_geometry_and_path(metric, monkeypatch, tmp_path):
    from hotpot.cheminfo.operations.forcefields.contracts import AcceptanceCheck, ForceFieldValidationReport
    from hotpot.pipeline.contracts import StageStatus

    quality = ForceFieldValidationReport(
        False,
        (
            AcceptanceCheck("coordinate_shape", True),
            AcceptanceCheck("finite_coordinates", True),
            AcceptanceCheck("topology", True),
            AcceptanceCheck(metric, False, measured=None, threshold="finite", message="required metric unavailable"),
        ),
    )
    prepared, payload, context, calls = _prepare("ff", monkeypatch, tmp_path, quality=quality)
    result = prepared.execute(payload, context)
    assert len(calls) == 3
    assert result.status is StageStatus.QUALITY_FAILED
    assert result.payload.records == ()
    assert result.report["unavailable_count"] == 3
    assert len(result.report["forcefield"]) == 3
    for index, row in enumerate(result.report["sample_outcomes"]):
        assert row["outcome"] == "unavailable"
        assert row["output_index"] is None
        assert row["report_index"] == index
        assert row["observation_path"] == f"observations/{index:04d}.sdf"
        assert row["findings"][0]["code"] == f"FF:{metric}"
        observation = context.stage_directory / row["observation_path"]
        assert observation.read_text().count("$$$$") == 1
    assert len(result.artifacts) == 3


@pytest.mark.parametrize("frame_check", ("coordinate_shape", "finite_coordinates", "topology_atom_identity"))
def test_ff_invalid_frame_with_missing_metric_is_not_serialized_as_observation(frame_check, monkeypatch, tmp_path):
    quality = _quality(frame_check, "finite_final_energy")
    prepared, payload, context, _ = _prepare("ff", monkeypatch, tmp_path, quality=quality)
    result = prepared.execute(payload, context)
    assert result.payload.records == ()
    assert result.report["unavailable_count"] == 3
    assert all("observation_path" not in row for row in result.report["sample_outcomes"])
    assert not (context.stage_directory / "observations").exists()


def test_xtb_stage_import_with_post_check_off_keeps_ff_stage_unloaded():
    import subprocess
    import sys

    source = """
import sys
from hotpot.plugins.xtb import stage
from hotpot.pipeline.contracts import StageSpec
prepared = stage.get_stage().prepare(StageSpec('xtb', ('--post-check', 'off')))
assert prepared.options.post_check == 'off'
assert 'hotpot.cheminfo.operations.forcefields.stage' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
