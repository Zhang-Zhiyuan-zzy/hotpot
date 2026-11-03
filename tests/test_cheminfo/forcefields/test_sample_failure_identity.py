"""Local failure identity originates at bounded producers and survives transport."""

from __future__ import annotations

import pickle
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot.cheminfo.conformers import ConformerBatch, ConformerSamplingError
from hotpot import Molecule
from hotpot.cheminfo.operations.forcefields import backend, ligand, stage, workers
from hotpot.cheminfo.operations.forcefields.contracts import (
    BuildTimeoutError,
    BuildWorkerError,
    BuildWorkerResult,
    ComplexBuildDiagnostics,
    ComplexBuildError,
    ComplexBuildTimeoutError,
    ComplexBuildWorkerError,
    ForceFieldError,
    ForceFieldSetupError,
    GeometryQualityError,
    RingUntanglingReport,
)


@pytest.mark.parametrize(
    "error, code",
    (
        (ForceFieldError("unknown"), None),
        (ComplexBuildError("unknown"), None),
        (BuildWorkerError("ForceFieldSetupError", "known-looking message", None), None),
        (ComplexBuildWorkerError("GeometryQualityError", "unknown", None), None),
        (ForceFieldSetupError("setup"), "FF-SETUP-UNAVAILABLE"),
        (BuildTimeoutError("timeout"), "FF-BUILD-TIMEOUT"),
        (ComplexBuildTimeoutError("timeout"), "FF-COMPLEX-BUILD-TIMEOUT"),
        (GeometryQualityError(None), "FF-NO-USABLE-FRAME"),
    ),
)
def test_only_bounded_error_classes_supply_a_default_code(error, code):
    assert error.sample_failure_code == code


def test_worker_envelope_appends_optional_code_after_existing_positional_fields():
    assert tuple(field.name for field in fields(BuildWorkerResult)) == (
        "status", "coordinates", "diagnostics", "error_type", "error_message",
        "traceback", "ligand_build_attempts", "sample_failure_code",
    )
    result = BuildWorkerResult("error", None, None, "ValueError", "raw", "trace", ())
    assert result.sample_failure_code is None
    assert result.error_message == "raw"


def test_native_setup_translation_preserves_cause_and_report():
    cause = RuntimeError("native setup rejected")
    with pytest.raises(ForceFieldSetupError) as caught:
        backend._raise_forcefield_setup_error(
            cause, requested_forcefield="UFF", effective_forcefield="UFF", stage="setup",
        )
    assert caught.value.sample_failure_code == "FF-SETUP-UNAVAILABLE"
    assert caught.value.__cause__ is cause
    assert str(caught.value) == str(cause)
    assert caught.value.report.effective_forcefield == "UFF"
    assert caught.value.report.stage == "setup"


def test_ob_builder_false_result_alone_has_a_local_failure_code(monkeypatch):
    monkeypatch.setattr(backend, "build_molecule", lambda mol: SimpleNamespace(succeeded=False))
    with pytest.raises(ForceFieldError) as caught:
        backend._ob_build(object())
    assert caught.value.sample_failure_code == "FF-BUILD-UNAVAILABLE"
    assert str(caught.value) == "Open Babel could not build initial 3D coordinates"

    cause = ValueError("Open Babel could not build initial 3D coordinates")

    def fail(mol):
        raise cause

    monkeypatch.setattr(backend, "build_molecule", fail)
    with pytest.raises(ValueError) as unknown:
        backend._ob_build(object())
    assert unknown.value is cause
    assert getattr(unknown.value, "sample_failure_code", None) is None


def _molecule():
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=6)
    mol.add_bond(0, 1, bond_order=1.0)
    return mol


def _build_ligand(mol, sampler):
    return ligand._build_ligand_proxies(
        mol, max_attempts=1, candidate_warmup_steps=1, candidate_score_steps=1,
        best_candidate_refine_steps=1, effective_forcefield="UFF", sampler=sampler,
    )


@pytest.mark.parametrize("code", (None, "FF-BUILD-UNAVAILABLE"))
def test_sampler_wrapper_forwards_only_its_actual_cause_code(monkeypatch, code):
    mol = _molecule()
    cause = ConformerSamplingError("ForceFieldSetupError: FF-SETUP-UNAVAILABLE")
    if code is not None:
        cause.sample_failure_code = code

    def sample(mol, request):
        raise cause

    monkeypatch.setattr(ligand, "capture_topology", lambda *args, **kwargs: object())
    with pytest.raises(ComplexBuildError) as caught:
        _build_ligand(mol, SimpleNamespace(sample=sample))
    assert caught.value.sample_failure_code == code
    assert caught.value.__cause__ is cause
    assert str(caught.value) == "Component 0 failed its single conformer provider call"
    assert caught.value.diagnostics.attempt_count == 1
    assert caught.value.diagnostics.rejected_candidates[0].reason == str(cause)


def test_exact_candidate_exhaustion_is_identified_and_retains_rejection(monkeypatch):
    mol = _molecule()
    monkeypatch.setattr(ligand, "capture_topology", lambda *args, **kwargs: object())

    def setup_failure(*args, **kwargs):
        raise ForceFieldSetupError("candidate setup unavailable")

    monkeypatch.setattr(ligand, "_single_ob_optimization", setup_failure)
    sampler = SimpleNamespace(
        sample=lambda component, request: ConformerBatch(
            np.zeros((1, len(component.atoms), 3)), (0,),
        ),
    )
    with pytest.raises(ComplexBuildError) as caught:
        _build_ligand(mol, sampler)
    assert caught.value.sample_failure_code == "FF-LIGAND-CANDIDATES-EXHAUSTED"
    assert caught.value.diagnostics.attempt_count == 1
    assert caught.value.diagnostics.accepted_candidates == 0
    assert caught.value.diagnostics.rejected_candidates[0].reason == "candidate setup unavailable"



def _patch_ligand_recovery(monkeypatch, *, first_candidate_passes=True):
    """Keep geometry deterministic while exercising the real recovery controller."""
    mol = _molecule()
    trace = []
    coordinates = np.array(((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)))
    checkpoint = SimpleNamespace(
        state=ligand.geo.PiercingState.DOES_NOT_PIERCE,
        piercings=(), undetermined_pair_count=0, excluded_ring_count=0, max_ring_size=16,
    )
    untangling_report = RingUntanglingReport(3, 0, 0, 0, 0, True)

    def sample(component, request):
        trace.append(("sample",))
        return ConformerBatch(coordinates[None, :, :], (0,))

    def optimize(component, forcefield, steps):
        trace.append(("optimize", steps))
        return backend._CandidateOptimizationResult(1.0, "kJ/mol", False)

    def untangle(*args, **kwargs):
        trace.append(("untangle", kwargs["short_steps"], kwargs["settling_steps"]))
        return SimpleNamespace(checkpoint_report=checkpoint, energy=1.0, report=untangling_report)

    def acceptance(*args, **kwargs):
        trace.append(("accept",))
        return SimpleNamespace(passed=first_candidate_passes, failures=())

    monkeypatch.setattr(ligand, "capture_topology", lambda *args, **kwargs: object())
    monkeypatch.setattr(ligand, "_single_ob_optimization", optimize)
    monkeypatch.setattr(ligand, "_scan_ring_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(ligand, "_record_ring_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(ligand, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(ligand, "evaluate_structure_acceptance_at_checkpoint", acceptance)
    return mol, SimpleNamespace(sample=sample), trace, coordinates


def _run_recovery_ligand(mol, sampler):
    return ligand._build_ligand_proxies(
        mol, max_attempts=3, candidate_warmup_steps=11, candidate_score_steps=22,
        best_candidate_refine_steps=33, ligand_untangling_attempts=3,
        effective_forcefield="UFF", sampler=sampler,
    )


@pytest.mark.parametrize("failure_point", ("warmup", "ring_repair"))
@pytest.mark.parametrize("prior_fallback", (False, True))
def test_unknown_candidate_error_propagates_exactly_without_more_attempts_or_fallback(
    monkeypatch, failure_point, prior_fallback,
):
    mol, sampler, trace, _ = _patch_ligand_recovery(
        monkeypatch, first_candidate_passes=not prior_fallback,
    )
    cause = ForceFieldError("unknown candidate integrity failure")
    calls = []
    owner_name = "_single_ob_optimization" if failure_point == "warmup" else "_untangle_ring_piercings"
    operation = getattr(ligand, owner_name)

    def fail_at_candidate(*args, **kwargs):
        calls.append(1)
        if len(calls) == (2 if prior_fallback else 1):
            raise cause
        return operation(*args, **kwargs)

    monkeypatch.setattr(ligand, owner_name, fail_at_candidate)
    with pytest.raises(ForceFieldError) as caught:
        _run_recovery_ligand(mol, sampler)
    assert caught.value is cause
    assert caught.value.sample_failure_code is None
    assert len(calls) == (2 if prior_fallback else 1)
    assert trace.count(("sample",)) == 1
    assert ("optimize", 33) not in trace
    assert trace.count(("accept",)) == (1 if prior_fallback else 0)


@pytest.mark.parametrize("failure_point", ("warmup", "ring_repair"))
def test_known_candidate_setup_rejection_keeps_attempt_budget_and_diagnostics(monkeypatch, failure_point):
    mol, sampler, trace, _ = _patch_ligand_recovery(monkeypatch)
    cause = ForceFieldSetupError("candidate setup unavailable")
    rejected_calls = []
    owner_name = "_single_ob_optimization" if failure_point == "warmup" else "_untangle_ring_piercings"

    def setup_failure(*args, **kwargs):
        rejected_calls.append(1)
        raise cause

    monkeypatch.setattr(ligand, owner_name, setup_failure)
    with pytest.raises(ComplexBuildError) as caught:
        _run_recovery_ligand(mol, sampler)
    assert caught.value.sample_failure_code == "FF-LIGAND-CANDIDATES-EXHAUSTED"
    assert len(rejected_calls) == 3
    assert caught.value.diagnostics.attempt_count == 3
    assert tuple(rejection.attempt for rejection in caught.value.diagnostics.rejected_candidates) == (1, 2, 3)
    assert all(rejection.reason == str(cause) for rejection in caught.value.diagnostics.rejected_candidates)
    assert trace.count(("sample",)) == 1
    assert trace.count(("accept",)) == 0


@pytest.mark.parametrize("error_type", (ForceFieldError, ForceFieldSetupError))
def test_refinement_recovers_only_known_setup_and_unknown_error_propagates_exactly(monkeypatch, error_type):
    mol, sampler, trace, candidate_coordinates = _patch_ligand_recovery(monkeypatch)
    cause = error_type("refinement failed")
    optimize = ligand._single_ob_optimization
    calls = []

    def fail_refinement(component, forcefield, steps):
        calls.append(steps)
        if steps == 33:
            component.coordinates = component.coordinates + 7.0
            raise cause
        return optimize(component, forcefield, steps)

    monkeypatch.setattr(ligand, "_single_ob_optimization", fail_refinement)
    if error_type is ForceFieldError:
        with pytest.raises(ForceFieldError) as caught:
            _run_recovery_ligand(mol, sampler)
        assert caught.value is cause
        assert caught.value.sample_failure_code is None
    else:
        coordinates, diagnostics = _run_recovery_ligand(mol, sampler)
        np.testing.assert_array_equal(coordinates, candidate_coordinates)
        assert diagnostics.attempt_count == 1
        assert diagnostics.accepted_candidates == 1
        assert len(diagnostics.rejected_candidates) == 1
        assert diagnostics.rejected_candidates[0].reason == "refined candidate: refinement failed"
        assert diagnostics.warning_messages == (
            "Component 0: long refinement failed; retaining the medium-optimized candidate",
        )
    assert calls == [11, 33]
    assert trace.count(("sample",)) == 1
    assert trace.count(("accept",)) == 1
    assert trace.count(("untangle", 11, 22)) == 1

def test_only_two_existing_route_preconditions_get_private_value_error_codes():
    with pytest.raises(ValueError) as coordinates:
        stage._requires_build(
            SimpleNamespace(has_3d=False), SimpleNamespace(rebuild=False, optimize_only=True),
        )
    assert coordinates.value.sample_failure_code == "FF-REQUEST-COORDINATES"
    assert str(coordinates.value) == "--optimize-only requires non-coincident existing coordinates"
    with pytest.raises(ValueError) as method:
        stage._validate_forcefield_request(
            SimpleNamespace(has_metal=True), SimpleNamespace(route="auto", forcefield="MMFF94"),
        )
    assert method.value.sample_failure_code == "FF-REQUEST-METHOD"
    assert type(method.value) is type(coordinates.value)
    assert type(method.value).__name__ not in stage.__all__
    with pytest.raises(ValueError) as unknown:
        backend._resolve_organic_forcefield("unknown")
    assert getattr(unknown.value, "sample_failure_code", None) is None


class _Connection:
    def __init__(self, result=None, *, eof=False, ready=True):
        self.result = result
        self.eof = eof
        self.ready = ready
        self.closed = False

    def send(self, result):
        self.result = pickle.loads(pickle.dumps(result))

    def poll(self, timeout):
        return self.ready

    def recv(self):
        if self.eof:
            raise EOFError("closed")
        return self.result

    def close(self):
        self.closed = True


class _Process:
    sentinel = object()

    def __init__(self, exitcode=0):
        self.exitcode = exitcode
        self.started = False

    def start(self):
        self.started = True

    def join(self, timeout):
        pass

    def is_alive(self):
        return False


def _receive(monkeypatch, connection, error_type, *, exitcode=0, exited=True, timeout_type=BuildTimeoutError):
    process = _Process(exitcode)
    send = _Connection()
    monkeypatch.setattr(workers, "wait_for_connections", lambda *args, **kwargs: [process.sentinel] if exited else [])
    try:
        return workers._receive_worker_result(
            process, connection, send, timeout=1.0, worker_error_type=error_type,
            timeout_error_type=timeout_type,
        )
    finally:
        assert connection.closed and send.closed
        assert process.started


@pytest.mark.parametrize("sender", ("ordinary", "complex"))
@pytest.mark.parametrize("code", (None, "FF-BUILD-UNAVAILABLE", "EXPLICIT-PRODUCER-CODE"))
def test_child_sender_and_parent_receiver_preserve_exact_code_and_raw_error(monkeypatch, sender, code):
    diagnostics = ComplexBuildDiagnostics(1, 0, (), 0.0)
    message = "FF-SETUP-UNAVAILABLE: same text for every code"
    cause = ValueError(message) if code is None else ComplexBuildError(message, diagnostics)
    if code is not None:
        cause.sample_failure_code = code

    def fail(*args, **kwargs):
        raise cause

    mol = SimpleNamespace(_update_topology=lambda **kwargs: None)
    connection = _Connection()
    monkeypatch.setattr(workers._backend, "_seed_openbabel_random", lambda seed: None)
    if sender == "ordinary":
        monkeypatch.setattr(workers, "_ob_build", fail)
        workers._seeded_ob_build_worker(mol, connection, 11)
        error_type = BuildWorkerError
    else:
        monkeypatch.setattr(workers, "_build_ligand_proxies", fail)
        workers._build_ligand_proxies_worker(mol, connection, 1, 1, 1, 1, "UFF", None)
        error_type = ComplexBuildWorkerError
    assert connection.closed
    assert connection.result.sample_failure_code == code
    with pytest.raises(error_type) as caught:
        _receive(monkeypatch, connection, error_type)
    assert caught.value.sample_failure_code == code
    assert caught.value.error_type == type(cause).__name__
    assert caught.value.error_message == str(cause)
    assert "raise cause" in caught.value.worker_traceback
    if sender == "complex" and code is not None:
        assert caught.value.diagnostics == diagnostics


@pytest.mark.parametrize("error_type", (BuildWorkerError, ComplexBuildWorkerError))
@pytest.mark.parametrize(
    "case, expected_type",
    (
        ("eof", "WorkerProtocolError"),
        ("shutdown", "WorkerShutdownError"),
        ("missing_exit", "WorkerShutdownError"),
        ("exit", "WorkerExitError"),
        ("envelope", "WorkerProtocolError"),
        ("status", "WorkerProtocolError"),
        ("success", "WorkerProtocolError"),
    ),
)
def test_protocol_and_process_failures_never_inherit_result_code(monkeypatch, error_type, case, expected_type):
    result = BuildWorkerResult("error", sample_failure_code="FF-BUILD-UNAVAILABLE")
    if case == "envelope":
        result = SimpleNamespace(sample_failure_code="FF-BUILD-UNAVAILABLE")
    elif case == "status":
        result = BuildWorkerResult("invalid", sample_failure_code="FF-BUILD-UNAVAILABLE")
    elif case == "success":
        result = BuildWorkerResult("ok", sample_failure_code="FF-BUILD-UNAVAILABLE")
    connection = _Connection(result, eof=case == "eof")
    exitcode = None if case == "missing_exit" else 7 if case == "exit" else 0
    with pytest.raises(error_type) as caught:
        _receive(monkeypatch, connection, error_type, exitcode=exitcode, exited=case != "shutdown")
    assert caught.value.error_type == expected_type
    assert caught.value.sample_failure_code is None


@pytest.mark.parametrize(
    "error_type, timeout_type, code",
    (
        (BuildWorkerError, BuildTimeoutError, "FF-BUILD-TIMEOUT"),
        (ComplexBuildWorkerError, ComplexBuildTimeoutError, "FF-COMPLEX-BUILD-TIMEOUT"),
    ),
)
def test_existing_bounded_worker_timeout_has_its_own_code(monkeypatch, error_type, timeout_type, code):
    with pytest.raises(timeout_type) as caught:
        _receive(monkeypatch, _Connection(ready=False), error_type, timeout_type=timeout_type)
    assert caught.value.sample_failure_code == code
    assert isinstance(caught.value, TimeoutError)


@pytest.mark.parametrize("coordinates", (np.zeros((2, 2)), np.full((2, 3), np.nan)))
@pytest.mark.parametrize("error_type", (BuildWorkerError, ComplexBuildWorkerError))
def test_invalid_success_coordinates_remain_protocol_failure(coordinates, error_type):
    result = BuildWorkerResult("ok", coordinates=coordinates, sample_failure_code="FF-BUILD-UNAVAILABLE")
    with pytest.raises(error_type) as caught:
        workers._validated_worker_coordinates(result, expected_atom_count=2, worker_error_type=error_type)
    assert caught.value.error_type == "WorkerProtocolError"
    assert caught.value.sample_failure_code is None
