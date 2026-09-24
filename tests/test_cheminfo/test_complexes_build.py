import multiprocessing as mp
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.forcefields import backend as ob_backend
from hotpot.cheminfo.forcefields import ligand
from hotpot.cheminfo.forcefields import repair
from hotpot.cheminfo.forcefields import utils as ff
from hotpot.cheminfo.forcefields import workers
from hotpot.cheminfo.forcefields import workflows


def _send_large_worker(connection):
    diagnostics = ff.ComplexBuildDiagnostics(1, 1, (), 0.0)
    connection.send(
        ff.BuildWorkerResult(
            status="ok",
            coordinates=np.ones((250_000, 3)),
            diagnostics=diagnostics,
        )
    )
    connection.close()


def _share_single_ob_optimization(monkeypatch):
    monkeypatch.setattr(
        repair,
        "_single_ob_optimization",
        ligand._single_ob_optimization,
    )


def _send_error_worker(connection):
    connection.send(
        ff.BuildWorkerResult(
            status="error",
            error_type="ValueError",
            error_message="dative conversion failed",
            traceback="worker traceback",
        )
    )
    connection.close()


def _blocking_worker(connection):
    time.sleep(10.0)


def _blocking_complex_worker(molecule, connection, *args):
    time.sleep(10.0)


def _failing_complex_worker(molecule, connection, *args):
    connection.send(
        ff.BuildWorkerResult(
            status="error",
            error_type="RuntimeError",
            error_message="deliberate public worker failure",
            traceback="worker traceback",
        )
    )
    connection.close()


def _ligand_fallback_warning_worker(molecule, connection, *args):
    diagnostics = ff.ComplexBuildDiagnostics(
        1,
        0,
        (),
        0.0,
        ("no ligand candidate passed; retaining the best usable attempt",),
    )
    connection.send(
        ff.BuildWorkerResult(
            status="ok",
            coordinates=molecule.coordinates,
            diagnostics=diagnostics,
        )
    )
    connection.close()


def _failing_ligand_trajectory_worker(molecule, connection, *args):
    trajectory = ff.ForceFieldTrajectory.from_molecule(
        molecule,
        start=ff.TrajectoryStart.LIGAND_BUILD,
    )
    frame = trajectory.record_molecule(
        molecule,
        stage=ff.TrajectoryStage.LIGAND_BUILD,
        event=ff.TrajectoryEvent.TERMINAL,
    )
    trajectory.select(frame.index)
    connection.send(
        ff.BuildWorkerResult(
            status="error",
            error_type="ForceFieldError",
            error_message="deliberate ligand build failure",
            ligand_build_attempts=(trajectory,),
        )
    )
    connection.close()


def _malformed_worker(connection):
    connection.send("not a BuildWorkerResult")
    connection.close()


def _invalid_status_worker(connection):
    connection.send(
        ff.BuildWorkerResult(
            status="unknown",
            coordinates=np.zeros((1, 3)),
        )
    )
    connection.close()


def _incomplete_success_worker(connection):
    connection.send(ff.BuildWorkerResult(status="ok"))
    connection.close()


def _send_seed_worker(connection):
    diagnostics = ff.ComplexBuildDiagnostics(0, 0, (), 0.0)
    connection.send(
        ff.BuildWorkerResult(
            status="ok",
            coordinates=np.asarray([[float(os.environ["OB_RANDOM_SEED"]), 0.0, 0.0]]),
            diagnostics=diagnostics,
        )
    )
    connection.close()


def _send_small_worker(connection):
    diagnostics = ff.ComplexBuildDiagnostics(0, 0, (), 0.0)
    connection.send(
        ff.BuildWorkerResult(
            status="ok",
            coordinates=np.zeros((1, 3)),
            diagnostics=diagnostics,
        )
    )
    connection.close()


def _send_coordinates_only_worker(connection):
    connection.send(
        ff.BuildWorkerResult(
            status="ok",
            coordinates=np.zeros((1, 3)),
        )
    )
    connection.close()


def _send_then_exit_slowly_worker(connection):
    _send_small_worker(connection)
    time.sleep(0.05)


def _send_tagged_worker(connection, tag):
    diagnostics = ff.ComplexBuildDiagnostics(1, 1, (), 0.0)
    connection.send(
        ff.BuildWorkerResult(
            status="ok",
            coordinates=np.asarray([[float(tag), 0.0, 0.0]]),
            diagnostics=diagnostics,
        )
    )
    connection.close()


class _NeverReadyConnection:
    def __init__(self):
        self.closed = False

    def poll(self, timeout):
        return False

    def close(self):
        self.closed = True


class _ReadyConnection(_NeverReadyConnection):
    def __init__(self, result):
        super().__init__()
        self.result = result

    def poll(self, timeout):
        return True

    def recv(self):
        return self.result


class _StubbornProcess:
    def __init__(self):
        self.started = False
        self.terminated = False
        self.killed = False
        self.join_calls = []
        self.sentinel = object()

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started and not self.killed

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class _DelayedExitcodeProcess:
    def __init__(self):
        self.started = False
        self.exitcode = None
        self.join_calls = []
        self.sentinel = object()

    def start(self):
        self.started = True

    def is_alive(self):
        return False

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        if timeout is not None:
            self.exitcode = 0


class _DummyAtom:
    def __init__(self, atom_id):
        self.id = atom_id
        self.coordinates = np.zeros(3)


class _DummyComponent:
    has_metal = False

    def __init__(self):
        self.coordinates = np.zeros((2, 3))
        self.atoms = [_DummyAtom(0), _DummyAtom(1)]
        self.bonds = []
        self.hidden = []

    def recover_hided_covalent_bonds(self, clear_conformers=False):
        return None

    def hide_bonds(self, *bonds, clear_conformers=False):
        self.hidden.extend(bonds)

    def restore_bonds(self, *bonds, clear_conformers=False):
        for bond in bonds:
            self.hidden.remove(bond)

    @property
    def has_bond_ring_intersection(self):
        raise AssertionError("forcefields must call geometry directly")

    @property
    def intersection_bonds_rings(self):
        raise AssertionError("forcefields must call geometry directly")


class _DummyComplex:
    def __init__(self, component):
        self.component = component
        self.coordinates = np.zeros((2, 3))
        self.charge = 0
        self.properties = {}
        self._model = None
        self._environ = None
        self._crystal = None

    def __copy__(self):
        return self

    def hide_metal_ligand_bonds(self, clear_conformers=False):
        return None

    def recover_hided_metal_ligand_bonds(self, clear_conformers=False):
        return None

    @property
    def components(self):
        return [self.component]

    def update_atoms_attrs_from_id_dict(self, updates):
        return None


def _bond_ring_finding(ring, bond):
    return SimpleNamespace(
        target=SimpleNamespace(
            ring=SimpleNamespace(ring=ring),
            bond=SimpleNamespace(bond=bond),
        )
    )


def _piercing_report(*ring_bond_pairs):
    pair_count = len(ring_bond_pairs)
    return SimpleNamespace(
        state=(
            geo.PiercingState.PIERCES
            if ring_bond_pairs
            else geo.PiercingState.DOES_NOT_PIERCE
        ),
        piercings=tuple(
            _bond_ring_finding(ring, bond)
            for ring, bond in ring_bond_pairs
        ),
        undetermined=(),
        undetermined_pair_count=0,
        selected_ring_count=len({id(ring) for ring, _ in ring_bond_pairs}),
        excluded_ring_count=0,
        candidate_pair_count=pair_count,
        aabb_separated_pair_count=0,
        exact_pair_count=pair_count,
        piercing_pair_count=pair_count,
        does_not_pierce_pair_count=0,
        scan_complete=True,
        max_ring_size=16,
        ring_scope="ligand_skeleton",
    )


def _pipe_process(target):
    context = mp.get_context("fork")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(send_connection,))
    return process, receive_connection, send_connection


def _molecule_state(molecule):
    return (
        tuple(
            (atom.idx, atom.id, atom.atomic_number, atom.formal_charge)
            for atom in molecule.atoms
        ),
        tuple(
            sorted(
                (
                    min(bond.a1idx, bond.a2idx),
                    max(bond.a1idx, bond.a2idx),
                    bond.bond_order,
                    bond.bond_kind,
                )
                for bond in molecule.bonds
            )
        ),
        molecule.coordinates.copy(),
    )


def _assert_molecule_state_unchanged(molecule, expected):
    current = _molecule_state(molecule)
    assert current[:2] == expected[:2]
    np.testing.assert_array_equal(current[2], expected[2])


def _active_child_pids():
    return {process.pid for process in mp.active_children()}


def test_pipe_receives_large_result_before_joining_worker():
    process, receive_connection, send_connection = _pipe_process(_send_large_worker)
    result = workers._receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=5.0,
    )
    assert result.coordinates.shape == (250_000, 3)
    assert process.exitcode == 0


def test_pipe_propagates_worker_error_with_original_diagnostics():
    process, receive_connection, send_connection = _pipe_process(_send_error_worker)
    with pytest.raises(ff.ComplexBuildWorkerError) as caught:
        workers._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=5.0,
        )
    assert caught.value.error_type == "ValueError"
    assert caught.value.error_message == "dative conversion failed"
    assert caught.value.worker_traceback == "worker traceback"


def test_generic_build_worker_accepts_coordinates_without_complex_diagnostics():
    process, receive_connection, send_connection = _pipe_process(
        _send_coordinates_only_worker
    )

    result = workers._receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=5.0,
        require_diagnostics=False,
        worker_error_type=ff.BuildWorkerError,
        timeout_error_type=ff.BuildTimeoutError,
        operation="building initial coordinates",
    )

    assert result.coordinates.shape == (1, 3)
    assert result.diagnostics is None


def test_generic_build_worker_uses_generic_failure_type():
    process, receive_connection, send_connection = _pipe_process(_send_error_worker)

    with pytest.raises(ff.BuildWorkerError) as caught:
        workers._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=5.0,
            require_diagnostics=False,
            worker_error_type=ff.BuildWorkerError,
            timeout_error_type=ff.BuildTimeoutError,
            operation="building initial coordinates",
        )

    assert caught.value.error_type == "ValueError"
    assert caught.value.error_message == "dative conversion failed"
    assert caught.value.worker_traceback == "worker traceback"


def test_generic_build_worker_uses_generic_timeout_type():
    process, receive_connection, send_connection = _pipe_process(_blocking_worker)

    with pytest.raises(ff.BuildTimeoutError, match="building initial coordinates"):
        workers._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=0.02,
            require_diagnostics=False,
            worker_error_type=ff.BuildWorkerError,
            timeout_error_type=ff.BuildTimeoutError,
            operation="building initial coordinates",
        )


def test_timeout_terminates_and_joins_worker():
    process, receive_connection, send_connection = _pipe_process(_blocking_worker)
    with pytest.raises(ff.ComplexBuildTimeoutError):
        workers._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=0.02,
        )
    assert not process.is_alive()


@pytest.mark.parametrize(
    ("worker", "expected_error"),
    (
        (_blocking_complex_worker, ff.ComplexBuildTimeoutError),
        (_failing_complex_worker, ff.ComplexBuildWorkerError),
    ),
    ids=("timeout", "worker-failure"),
)
def test_public_complex_build_failure_is_transactional_and_reaps_worker(
    monkeypatch,
    worker,
    expected_error,
):
    molecule = read_mol("[Zn](N)(N)", "smi")
    original = _molecule_state(molecule)
    child_pids_before = _active_child_pids()
    fork_context = mp.get_context("fork")
    monkeypatch.setattr(workflows.mp, "get_context", lambda method: fork_context)
    monkeypatch.setattr(workflows, "_build_ligand_proxies_worker", worker)

    with pytest.raises(expected_error):
        ff.build_complex3d(
            molecule,
            candidate_count=1,
            max_attempts=1,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            timeout=0.02,
            add_hydrogens=False,
        )

    _assert_molecule_state_unchanged(molecule, original)
    assert _active_child_pids() <= child_pids_before


def test_timeout_escalates_to_kill_when_worker_ignores_termination():
    process = _StubbornProcess()
    receive_connection = _NeverReadyConnection()
    send_connection = _NeverReadyConnection()

    with pytest.raises(ff.ComplexBuildTimeoutError):
        workers._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=0.0,
        )

    assert process.terminated
    assert process.killed
    assert receive_connection.closed
    assert send_connection.closed


def test_malformed_worker_protocol_fails_explicitly():
    process, receive_connection, send_connection = _pipe_process(_malformed_worker)
    with pytest.raises(ff.ComplexBuildWorkerError) as caught:
        workers._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=5.0,
        )
    assert caught.value.error_type == "WorkerProtocolError"


def test_invalid_worker_status_fails_explicitly():
    process, receive_connection, send_connection = _pipe_process(
        _invalid_status_worker
    )
    with pytest.raises(ff.ComplexBuildWorkerError) as caught:
        workers._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=5.0,
        )
    assert caught.value.error_type == "WorkerProtocolError"


def test_incomplete_success_worker_protocol_fails_explicitly():
    process, receive_connection, send_connection = _pipe_process(
        _incomplete_success_worker
    )
    with pytest.raises(ff.ComplexBuildWorkerError) as caught:
        workers._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=5.0,
        )
    assert caught.value.error_type == "WorkerProtocolError"


def test_seed_is_forwarded_to_worker_without_mutating_parent_environment(monkeypatch):
    monkeypatch.setenv("OB_RANDOM_SEED", "parent")
    process, receive_connection, send_connection = _pipe_process(_send_seed_worker)

    result = workers._receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=5.0,
        seed=37,
    )

    assert result.coordinates[0, 0] == 37.0
    assert os.environ["OB_RANDOM_SEED"] == "parent"


def test_worker_start_and_reaping_use_the_lifecycle_lock(monkeypatch):
    class CountingLock:
        entered = 0

        def __enter__(self):
            self.entered += 1

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    lock = CountingLock()
    monkeypatch.setattr(ob_backend, "_WORKER_LIFECYCLE_LOCK", lock)
    process, receive_connection, send_connection = _pipe_process(_send_small_worker)

    workers._receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=5.0,
    )

    assert lock.entered == 3


def test_successful_worker_receives_a_separate_exit_grace_period(monkeypatch):
    monkeypatch.setattr(ob_backend, "_WORKER_EXIT_GRACE_SECONDS", 0.5)
    process, receive_connection, send_connection = _pipe_process(
        _send_then_exit_slowly_worker
    )

    result = workers._receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=2.0,
    )

    assert result.status == "ok"
    assert process.exitcode == 0


def test_ready_sentinel_is_followed_by_bounded_exitcode_refresh(monkeypatch):
    monkeypatch.setattr(ob_backend, "_WORKER_EXIT_GRACE_SECONDS", 0.25)
    monkeypatch.setattr(
        workers,
        "wait_for_connections",
        lambda objects, timeout: list(objects),
    )
    diagnostics = ff.ComplexBuildDiagnostics(0, 0, (), 0.0)
    receive_connection = _ReadyConnection(
        ff.BuildWorkerResult(
            status="ok",
            coordinates=np.zeros((1, 3)),
            diagnostics=diagnostics,
        )
    )
    send_connection = _NeverReadyConnection()
    process = _DelayedExitcodeProcess()

    result = workers._receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=2.0,
    )

    assert result.status == "ok"
    assert process.exitcode == 0
    assert process.join_calls == [0.25, 5.0]


def test_successful_message_does_not_hide_a_worker_that_fails_to_exit(monkeypatch):
    monkeypatch.setattr(ob_backend, "_WORKER_EXIT_GRACE_SECONDS", 0.25)
    wait_calls = []

    def wait_for_exit(objects, timeout):
        wait_calls.append((objects, timeout))
        return []

    monkeypatch.setattr(workers, "wait_for_connections", wait_for_exit)
    diagnostics = ff.ComplexBuildDiagnostics(0, 0, (), 0.0)
    receive_connection = _ReadyConnection(
        ff.BuildWorkerResult(
            status="ok",
            coordinates=np.zeros((1, 3)),
            diagnostics=diagnostics,
        )
    )
    send_connection = _NeverReadyConnection()
    process = _StubbornProcess()

    with pytest.raises(ff.ComplexBuildWorkerError, match="did not terminate"):
        workers._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=2.0,
        )

    assert wait_calls == [((process.sentinel,), 0.25)]
    assert process.join_calls == [5.0, 5.0]
    assert process.terminated
    assert process.killed
    assert receive_connection.closed
    assert send_connection.closed


def test_repeated_complex_worker_requests_do_not_cross_or_leak_processes():
    context = mp.get_context("fork")
    child_pids_before = _active_child_pids()
    received_tags = []

    for tag in range(24):
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_send_tagged_worker,
            args=(send_connection, tag),
        )
        result = workers._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=2.0,
        )
        received_tags.append(int(result.coordinates[0, 0]))
        assert process.exitcode == 0

    assert received_tags == list(range(24))
    assert _active_child_pids() <= child_pids_before


@pytest.mark.parametrize(
    "coordinates",
    (
        np.zeros((1, 3)),
        np.full((2, 3), np.nan),
    ),
    ids=("wrong-shape", "nonfinite"),
)
def test_complex_worker_coordinates_are_validated_before_native_optimization(
    coordinates,
):
    diagnostics = ff.ComplexBuildDiagnostics(1, 1, (), 0.0)
    result = ff.BuildWorkerResult(
        status="ok",
        coordinates=coordinates,
        diagnostics=diagnostics,
    )

    with pytest.raises(ff.ComplexBuildWorkerError) as caught:
        workers._validated_worker_coordinates(result, expected_atom_count=2)

    assert caught.value.error_type == "WorkerProtocolError"


def test_generic_worker_coordinate_validation_uses_generic_failure_type():
    result = ff.BuildWorkerResult(
        status="ok",
        coordinates=np.zeros((1, 3)),
    )

    with pytest.raises(ff.BuildWorkerError) as caught:
        workers._validated_worker_coordinates(
            result,
            expected_atom_count=2,
            worker_error_type=ff.BuildWorkerError,
        )

    assert caught.value.error_type == "WorkerProtocolError"


def test_candidate_attempts_are_bounded_and_use_geometry_relations(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    calls = {"build": 0, "perturb": 0, "screen": 0, "closest": 0}
    ring_sizes = []

    def fake_build(current):
        calls["build"] += 1

    monkeypatch.setattr(ligand, "_ob_build", fake_build)

    def perturb(coordinates, **kwargs):
        calls["perturb"] += 1
        return np.asarray(coordinates).copy()

    monkeypatch.setattr(ligand, "_perturbed_coordinates", perturb)
    monkeypatch.setattr(
        ligand,
        "_single_ob_optimization",
lambda *args, **kwargs: ob_backend._CandidateOptimizationResult(1.0, "kJ/mol", False),
    )
    _share_single_ob_optimization(monkeypatch)
    monkeypatch.setattr(
        ligand,
        "capture_topology",
        lambda mol, **options: object(),
    )

    def screen_relations(*args, **kwargs):
        calls["screen"] += 1
        ring_sizes.append(kwargs["max_ring_size"])
        return _piercing_report(("ring", "probe"))

    def closest(*args, **kwargs):
        calls["closest"] += 1
        return SimpleNamespace(a1idx=0, a2idx=1)

    intersection_failure = ff.AcceptanceCheck(
        name="bond_ring_intersection",
        passed=False,
        measured=(0, 1, 2),
        threshold=False,
        atom_indices=(3, 4),
        bond_indices=(2,),
    )
    monkeypatch.setattr(geo, "screen_bond_ring_relations", screen_relations)
    watch = (
        repair._WatchedRingPiercing(
            repair._BondRingPairKey((0, 1, 2), (3, 4)),
            ((0, 1),),
        ),
    )
    monkeypatch.setattr(
        repair,
        "_ring_piercing_watch",
        lambda *args, **kwargs: watch,
    )
    monkeypatch.setattr(
        repair,
        "_scan_ring_piercing_watch",
        lambda *args, **kwargs: repair._RingPiercingWatchResult(
            geo.PiercingState.PIERCES,
            watch,
        ),
    )
    monkeypatch.setattr(repair, "_select_ring_opening_edge", closest)
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(
            passed=False,
            failures=(intersection_failure,),
        ),
    )

    _, diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=3,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
        ligand_untangling_attempts=1,
    )

    assert calls == {"build": 1, "perturb": 2, "screen": 9, "closest": 3}
    assert ring_sizes == [16] * 9
    assert diagnostics.attempt_count == 3
    assert diagnostics.accepted_candidates == 0
    assert len(diagnostics.rejected_candidates) == 3
    assert all(
        rejection.quality_failures == (intersection_failure,)
        for rejection in diagnostics.rejected_candidates
    )
    assert "no candidate passed the basic geometry gate" in (
        diagnostics.warning_messages[0]
    )
    assert len(diagnostics.ligand_untangling) == 1


@pytest.mark.parametrize(
    "piercing_state",
    (
        geo.PiercingState.DOES_NOT_PIERCE,
        geo.PiercingState.UNDETERMINED,
    ),
)
def test_ligand_proxy_only_opens_rings_for_confirmed_piercing(
    monkeypatch,
    piercing_state,
):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    calls = {"screen": 0, "acceptance": 0}

    monkeypatch.setattr(ligand, "_ob_build", lambda current: None)
    monkeypatch.setattr(
        ligand,
        "_single_ob_optimization",
        lambda *args, **kwargs: ob_backend._CandidateOptimizationResult(
            1.0,
            "kJ/mol",
            False,
        ),
    )
    _share_single_ob_optimization(monkeypatch)
    monkeypatch.setattr(ligand, "capture_topology", lambda *args, **kwargs: object())

    def screen_relations(*args, **kwargs):
        calls["screen"] += 1
        report = _piercing_report()
        report.state = piercing_state
        return report

    def accept(*args, **kwargs):
        calls["acceptance"] += 1
        return SimpleNamespace(passed=True, failures=())

    monkeypatch.setattr(
        geo,
        "screen_bond_ring_relations",
        screen_relations,
    )
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        accept,
    )

    coordinates, diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
    )

    np.testing.assert_array_equal(coordinates, molecule.coordinates)
    assert diagnostics.accepted_candidates == 1
    assert diagnostics.rejected_candidates == ()
    assert calls == {"screen": 3, "acceptance": 2}
    assert component.hidden == []


def test_default_ligand_proxy_search_stops_after_one_accepted_candidate(
    monkeypatch,
):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    build_calls = 0
    quality_calls = 0
    first_failure = ff.AcceptanceCheck(
        name="minimum_distance",
        passed=False,
        measured=0.2,
        threshold=0.4,
    )

    def build(current):
        nonlocal build_calls
        build_calls += 1
        current.coordinates = np.full((2, 3), float(build_calls))

    monkeypatch.setattr(ligand, "_ob_build", build)
    monkeypatch.setattr(
        ligand,
        "_single_ob_optimization",
        lambda *args, **kwargs: ob_backend._CandidateOptimizationResult(
            1.0,
            "kJ/mol",
            False,
        ),
    )
    _share_single_ob_optimization(monkeypatch)
    monkeypatch.setattr(ligand, "capture_topology", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        geo,
        "screen_bond_ring_relations",
        lambda *args, **kwargs: _piercing_report(),
    )
    def quality(*args, **kwargs):
        nonlocal quality_calls
        quality_calls += 1
        return SimpleNamespace(
            passed=quality_calls >= 2,
            failures=() if quality_calls >= 2 else (first_failure,),
        )

    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        quality,
    )

    _, diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=10,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
    )

    assert build_calls == 1
    assert quality_calls == 3
    assert diagnostics.accepted_candidates == 1
    assert diagnostics.warning_messages == ()


def test_ligand_proxy_search_stops_after_first_accepted_candidate(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    build_calls = 0
    optimization_steps = []

    def build(current):
        nonlocal build_calls
        build_calls += 1
        current.coordinates = np.full((2, 3), float(build_calls))

    monkeypatch.setattr(ligand, "_ob_build", build)
    def optimize(current, forcefield, steps):
        optimization_steps.append(steps)
        return ob_backend._CandidateOptimizationResult(
            float(current.coordinates[0, 0]),
            "kJ/mol",
            False,
        )

    monkeypatch.setattr(ligand, "_single_ob_optimization", optimize)
    _share_single_ob_optimization(monkeypatch)
    monkeypatch.setattr(ligand, "capture_topology", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        geo,
        "screen_bond_ring_relations",
        lambda *args, **kwargs: _piercing_report(),
    )
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(passed=True, failures=()),
    )

    _, diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=2,
        candidate_warmup_steps=1,
        candidate_score_steps=2,
        best_candidate_refine_steps=3,
        effective_forcefield="UFF",
    )

    assert build_calls == 1
    assert optimization_steps == [1, 2, 3]
    assert diagnostics.accepted_candidates == 1
    assert diagnostics.warning_messages == ()


def test_ligand_fallback_warning_is_emitted_by_the_parent_process(monkeypatch):
    molecule = read_mol("[Zn](N)", "smi")
    fork_context = mp.get_context("fork")
    monkeypatch.setattr(workflows.mp, "get_context", lambda method: fork_context)

    with pytest.warns(
        ff.ComplexBuildWarning,
        match="no ligand candidate passed",
    ):
        prepared = workflows._prepare_complex_working_mol(
            molecule,
            effective_forcefield="UFF",
            max_attempts=2,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            timeout=2.0,
            add_hydrogens=False,
            seed=None,
            coordination_geometry=None,
            worker_target=_ligand_fallback_warning_worker,
        )

    assert prepared.diagnostics.warning_messages == (
        "no ligand candidate passed; retaining the best usable attempt",
    )


def test_failed_ligand_build_persists_recorded_attempts(monkeypatch, tmp_path):
    molecule = read_mol("[Zn](N)", "smi")
    trajectory_path = tmp_path / "failed-complex-build"
    fork_context = mp.get_context("fork")
    monkeypatch.setattr(workflows.mp, "get_context", lambda method: fork_context)

    with pytest.raises(ff.ComplexBuildWorkerError) as caught:
        workflows._prepare_complex_working_mol(
            molecule,
            effective_forcefield="UFF",
            max_attempts=1,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            timeout=2.0,
            add_hydrogens=False,
            seed=None,
            trajectory_start=ff.TrajectoryStart.LIGAND_BUILD,
            trajectory_path=trajectory_path,
            coordination_geometry=None,
            worker_target=_failing_ligand_trajectory_worker,
        )

    restored = ff.ForceFieldTrajectoryArchive.read(trajectory_path)
    assert caught.value.trajectory is not None
    assert len(restored.main) == 0
    assert len(restored.ligand_build_attempts) == 1
    assert restored.ligand_build_attempts[0][0].event is ff.TrajectoryEvent.TERMINAL


def test_builder_failure_stops_after_the_single_builder_call(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    calls = 0

    def fail_build(current):
        nonlocal calls
        calls += 1
        raise ff.ForceFieldError("builder failed")

    monkeypatch.setattr(ligand, "_ob_build", fail_build)
    monkeypatch.setattr(
        ligand,
        "capture_topology",
        lambda mol, **options: object(),
    )

    with pytest.raises(ff.ComplexBuildError) as caught:
        ligand._build_ligand_proxies(
            molecule,
            max_attempts=3,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            effective_forcefield="UFF",
        )

    assert calls == 1
    assert caught.value.diagnostics.attempt_count == 1
    assert len(caught.value.diagnostics.rejected_candidates) == 1


def test_builder_failure_does_not_restore_unrelated_hidden_ring_bonds(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    recovery_calls = 0

    def fail_build(current):
        raise ff.ForceFieldError("builder failed")

    def recover(clear_conformers=False):
        nonlocal recovery_calls
        recovery_calls += 1

    component.recover_hided_covalent_bonds = recover
    monkeypatch.setattr(ligand, "_ob_build", fail_build)
    monkeypatch.setattr(
        ligand,
        "capture_topology",
        lambda mol, **options: object(),
    )

    with pytest.raises(ff.ComplexBuildError):
        ligand._build_ligand_proxies(
            molecule,
            max_attempts=2,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            effective_forcefield="UFF",
        )

    assert recovery_calls == 0


def test_ligand_retries_share_one_built_root_and_record_distinct_branches(
    monkeypatch,
):
    molecule = read_mol("[Zn](N)", "smi")
    trajectories = []
    build_inputs = []
    perturb_inputs = []
    failure = ff.AcceptanceCheck(
        name="minimum_distance",
        passed=False,
        measured=0.2,
        threshold=0.4,
    )

    def build(component_mol):
        build_inputs.append(component_mol.coordinates.copy())
        component_mol.coordinates = np.ones_like(component_mol.coordinates)

    def perturb(coordinates, **kwargs):
        perturb_inputs.append(np.asarray(coordinates).copy())
        return np.asarray(coordinates) + len(perturb_inputs)

    def untangle(component_mol, *args, **kwargs):
        return repair._RingUntanglingResult(
            report=ff.RingUntanglingReport(
                attempt_limit=1,
                attempts_completed=0,
                initial_piercing_count=0,
                final_piercing_count=0,
                minimum_piercing_count=0,
                resolved=True,
            ),
            energy=float(component_mol.coordinates[0, 0]),
            checkpoint_report=kwargs["checkpoint_report"],
        )

    monkeypatch.setattr(ligand, "_ob_build", build)
    monkeypatch.setattr(ligand, "_perturbed_coordinates", perturb)
    monkeypatch.setattr(
        ligand,
        "_single_ob_optimization",
        lambda *args, **kwargs: ob_backend._CandidateOptimizationResult(
            1.0,
            "kJ/mol",
            False,
        ),
    )
    monkeypatch.setattr(ligand, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(
            passed=False,
            failures=(failure,),
        ),
    )

    ligand._build_ligand_proxies(
        molecule,
        max_attempts=3,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
        trajectory_attempts=trajectories,
    )

    assert len(build_inputs) == 1
    assert len(perturb_inputs) == 2
    assert all(np.array_equal(frame, np.ones_like(frame)) for frame in perturb_inputs)
    events = tuple(
        frame.event
        for trajectory in trajectories
        for frame in trajectory
    )
    assert events.count(ff.TrajectoryEvent.BUILD_COMPLETE) == 1
    assert events.count(ff.TrajectoryEvent.PERTURBED) == 2


def test_candidate_rejection_preserves_geometry_failure_details(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    failure = ff.AcceptanceCheck(
        name="bond_length",
        passed=False,
        measured=31.0,
        threshold=30.0,
        atom_indices=(0, 1),
        bond_indices=(0,),
    )

    monkeypatch.setattr(ligand, "_ob_build", lambda current: None)
    monkeypatch.setattr(
        ligand,
        "_single_ob_optimization",
        lambda *args, **kwargs: ob_backend._CandidateOptimizationResult(1.0, "kJ/mol", False),
    )
    _share_single_ob_optimization(monkeypatch)
    monkeypatch.setattr(
        ligand,
        "capture_topology",
        lambda mol, **options: object(),
    )
    monkeypatch.setattr(
        geo,
        "screen_bond_ring_relations",
        lambda *args, **kwargs: _piercing_report(),
    )
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(
            passed=False,
            failures=(failure,),
        ),
    )

    _, diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
    )

    rejection = diagnostics.rejected_candidates[-1]
    assert diagnostics.accepted_candidates == 0
    assert "no candidate passed the basic geometry gate" in (
        diagnostics.warning_messages[0]
    )
    assert rejection.quality_failures == (failure,)
    assert "bond_length" in rejection.reason
    assert "measured=31.0" in rejection.reason
    assert "threshold=30.0" in rejection.reason
    assert "atom_indices=(0, 1)" in rejection.reason
    assert "bond_indices=(0,)" in rejection.reason


def test_failed_refinement_retains_the_medium_optimized_candidate(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    quality_calls = 0
    failure = ff.AcceptanceCheck(
        name="minimum_distance",
        passed=False,
        measured=0.12,
        threshold=0.40,
        atom_indices=(0, 1),
    )

    monkeypatch.setattr(ligand, "_ob_build", lambda current: None)
    monkeypatch.setattr(
        ligand,
        "_single_ob_optimization",
        lambda *args, **kwargs: ob_backend._CandidateOptimizationResult(1.0, "kJ/mol", False),
    )
    _share_single_ob_optimization(monkeypatch)
    monkeypatch.setattr(
        ligand,
        "capture_topology",
        lambda mol, **options: object(),
    )
    monkeypatch.setattr(
        geo,
        "screen_bond_ring_relations",
        lambda *args, **kwargs: _piercing_report(),
    )

    def quality(*args, **kwargs):
        nonlocal quality_calls
        quality_calls += 1
        assert kwargs["forcefield_stage"] == "candidate"
        assert "converged" not in kwargs["forcefield_report"]
        assert "rms_gradient" not in kwargs["forcefield_report"]
        assert "max_gradient" not in kwargs["forcefield_report"]
        passed = quality_calls == 1
        return SimpleNamespace(
            passed=passed,
            failures=() if passed else (failure,),
        )

    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        quality,
    )

    _, diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
    )

    assert quality_calls == 2
    rejection = diagnostics.rejected_candidates[-1]
    assert rejection.quality_failures == (failure,)
    assert "minimum_distance" in rejection.reason
    assert "measured=0.12" in rejection.reason
    assert "threshold=0.4" in rejection.reason
    assert "atom_indices=(0, 1)" in rejection.reason
    assert diagnostics.warning_messages[-1].endswith(
        "long refinement failed the basic geometry gate; retaining the "
        "medium-optimized candidate"
    )


def test_refined_intersection_retains_all_structured_geometry_evidence(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    relation_calls = 0
    quality_calls = 0
    failure = ff.AcceptanceCheck(
        name="bond_ring_intersection",
        passed=False,
        measured=(0, 1, 2),
        threshold=False,
        atom_indices=(3, 4),
        bond_indices=(2,),
    )
    gate_failure = ff.AcceptanceCheck(
        name="atom_too_close",
        passed=False,
        measured=0.2,
        threshold=0.4,
        atom_indices=(0, 1),
    )

    monkeypatch.setattr(ligand, "_ob_build", lambda current: None)
    monkeypatch.setattr(
        ligand,
        "_single_ob_optimization",
        lambda *args, **kwargs: ob_backend._CandidateOptimizationResult(1.0, "kJ/mol", False),
    )
    _share_single_ob_optimization(monkeypatch)
    monkeypatch.setattr(
        ligand,
        "capture_topology",
        lambda mol, **options: object(),
    )

    def screen_relations(*args, **kwargs):
        nonlocal relation_calls
        relation_calls += 1
        return _piercing_report(
            *(("ring", "probe"),) if relation_calls > 2 else ()
        )

    monkeypatch.setattr(
        geo,
        "screen_bond_ring_relations",
        screen_relations,
    )

    def untangle(current, *args, **kwargs):
        checkpoint = kwargs["checkpoint_report"]
        count = len(checkpoint.piercings)
        return repair._RingUntanglingResult(
            report=ff.RingUntanglingReport(
                attempt_limit=kwargs["attempt_limit"],
                attempts_completed=0,
                initial_piercing_count=count,
                final_piercing_count=count,
                minimum_piercing_count=count,
                resolved=count == 0,
            ),
            energy=1.0,
            checkpoint_report=checkpoint,
        )

    monkeypatch.setattr(ligand, "_untangle_ring_piercings", untangle)

    def quality(*args, **kwargs):
        nonlocal quality_calls
        quality_calls += 1
        assert kwargs["forcefield_stage"] == "candidate"
        return SimpleNamespace(
            passed=quality_calls == 1,
            failures=() if quality_calls == 1 else (failure, gate_failure),
        )

    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        quality,
    )

    _, diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
    )

    rejection = diagnostics.rejected_candidates[-1]
    assert rejection.quality_failures == (failure, gate_failure)
    assert "bond_ring_intersection" in rejection.reason
    assert "atom_too_close" in rejection.reason


def test_best_unqualified_ligand_candidate_is_selected(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    state = {"build": 0, "proposal": 1}
    perturb_inputs = []
    failure = ff.AcceptanceCheck(
        name="minimum_distance",
        passed=False,
        measured=0.10,
        threshold=0.40,
        atom_indices=(0, 1),
    )

    def build(current):
        state["build"] += 1
        current.coordinates = np.full((2, 3), float(state["build"]))

    def optimize(current, forcefield, steps):
        marker = float(current.coordinates[0, 0])
        energies = {1.0: 0.0, 2.0: -100.0, 3.0: 100.0}
        return ob_backend._CandidateOptimizationResult(
            energies[marker],
            "kJ/mol",
            False,
        )

    def perturb(coordinates, **kwargs):
        perturb_inputs.append(np.asarray(coordinates).copy())
        state["proposal"] += 1
        return np.full_like(coordinates, float(state["proposal"]))

    def quality(current, **options):
        return SimpleNamespace(passed=False, failures=(failure,))

    piercing_counts = iter((2, 1, 1))

    def untangle(current, *args, **kwargs):
        count = next(piercing_counts)
        return repair._RingUntanglingResult(
            report=ff.RingUntanglingReport(
                attempt_limit=1,
                attempts_completed=0,
                initial_piercing_count=count,
                final_piercing_count=count,
                minimum_piercing_count=count,
                resolved=False,
            ),
            energy={1: 0.0, 2: -100.0, 3: 100.0}[
                int(current.coordinates[0, 0])
            ],
            checkpoint_report=kwargs["checkpoint_report"],
        )

    monkeypatch.setattr(ligand, "_ob_build", build)
    monkeypatch.setattr(ligand, "_perturbed_coordinates", perturb)
    monkeypatch.setattr(ligand, "_single_ob_optimization", optimize)
    monkeypatch.setattr(
        ligand,
        "capture_topology",
        lambda mol, **options: object(),
    )
    monkeypatch.setattr(
        ligand,
        "_untangle_ring_piercings",
        untangle,
    )
    monkeypatch.setattr(
        ligand,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: _piercing_report(),
    )
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        quality,
    )

    _, diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=3,
        candidate_warmup_steps=1,
        candidate_score_steps=2,
        best_candidate_refine_steps=3,
        effective_forcefield="UFF",
    )

    np.testing.assert_array_equal(component.coordinates, np.full((2, 3), 3.0))
    assert state["build"] == 1
    assert len(perturb_inputs) == 2
    assert all(np.array_equal(frame, np.ones((2, 3))) for frame in perturb_inputs)
    assert diagnostics.accepted_candidates == 0
    assert len(diagnostics.rejected_candidates) == 3
    assert diagnostics.ligand_untangling[0].final_piercing_count == 1


def test_select_ring_opening_edge_uses_watch_order(monkeypatch):
    component = _DummyComponent()

    def atom(index):
        return SimpleNamespace(idx=index)

    first = SimpleNamespace(atom1=atom(2), atom2=atom(4))
    second = SimpleNamespace(atom1=atom(1), atom2=atom(3))
    probe = SimpleNamespace(atom1=atom(5), atom2=atom(6))
    component.bonds = (first, second, probe)
    watch = (
        repair._WatchedRingPiercing(
            repair._BondRingPairKey((0, 2, 4), (5, 6)),
            ((2, 4),),
        ),
        repair._WatchedRingPiercing(
            repair._BondRingPairKey((0, 1, 3), (5, 6)),
            ((1, 3),),
        ),
    )

    monkeypatch.setattr(geo, "segment_from_bond", lambda bond: bond)
    monkeypatch.setattr(geo, "segment_segment_distance", lambda *args: 1.0)

    assert repair._select_ring_opening_edge(component, watch) is first


def test_worker_boundary_serializes_an_exception(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("cannot convert dative bond")

    class Connection:
        def __init__(self):
            self.result = None
            self.closed = False

        def send(self, result):
            self.result = result

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(workers, "_build_ligand_proxies", fail)
    workers._build_ligand_proxies_worker(
        object(),
        connection,
        1,
        1,
        1,
        1,
        "UFF",
        None,
    )
    assert connection.closed
    assert connection.result.status == "error"
    assert connection.result.error_type == "ValueError"
    assert "cannot convert dative bond" in connection.result.error_message


@pytest.mark.parametrize(
    "failure",
    (
        ff.ForceFieldSetupError("deliberate optimizer failure"),
        ff.GeometryQualityError(SimpleNamespace(passed=False)),
    ),
    ids=("optimizer", "geometry-gate"),
)
def test_complexes_build_final_failure_does_not_modify_caller(
    monkeypatch,
    failure,
):
    molecule = read_mol("[Zn](N)(N)", "smi")
    original = _molecule_state(molecule)
    diagnostics = ff.ComplexBuildDiagnostics(1, 1, (), 0.0)

    def built_working_copy(source, **options):
        working = source.copy()
        working.coordinates = working.coordinates + 7.0
        working.atoms[1].formal_charge = 1
        return workflows._PreparedComplex(
            mol=working,
            diagnostics=diagnostics,
            trajectory=ff.ForceFieldTrajectory.from_molecule(working),
        )

    def fail_final_stage(working, **options):
        working.coordinates = working.coordinates - 3.0
        working.remove_bonds([working.bonds[0]])
        raise failure

    monkeypatch.setattr(
        workflows,
        "_prepare_complex_working_mol",
        built_working_copy,
    )
    monkeypatch.setattr(workflows, "_optimize_working_mol", fail_final_stage)

    with pytest.raises(type(failure)):
        ff.complexes_build(
            molecule,
            epochs=1,
            steps_per_epoch=1,
            candidate_count=1,
            max_attempts=1,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            add_hydrogens=False,
        )

    _assert_molecule_state_unchanged(molecule, original)


def test_coordination_geometry_hook_is_explicitly_unimplemented():
    molecule = read_mol("[Zn](N)", "smi")

    with pytest.raises(NotImplementedError, match="reserved but not implemented"):
        ff.prepare_coordination_geometry(molecule, strategy="octahedral")


def test_coordination_geometry_hook_rejects_noncomplex_inputs():
    molecule = read_mol("CCO", "smi")

    with pytest.raises(ValueError, match="explicit metal-ligand bond"):
        ff.prepare_coordination_geometry(molecule, strategy="octahedral")


@pytest.mark.parametrize(
    "options",
    (
        {"max_attempts": 0},
        {"candidate_warmup_steps": 0},
        {"candidate_score_steps": 0},
        {"best_candidate_refine_steps": 0},
        {"timeout": 0.0},
    ),
)
def test_complex_builder_rejects_invalid_control_parameters(options):
    molecule = read_mol("[Zn](N)", "smi")

    with pytest.raises(ValueError):
        ff.build_complex3d(molecule, **options)


def test_complex_forcefield_rejects_unknown_name():
    molecule = read_mol("[Zn](N)", "smi")

    with pytest.raises(ValueError, match="Unsupported force field"):
        ff.build_complex3d(molecule, forcefield="typo")


def test_coordination_environment_preserves_a_bidentate_group():
    molecule = read_mol("NCCN.[Zn]", "smi")
    molecule.add_bond(molecule.atoms[4], molecule.atoms[0])
    molecule.add_bond(molecule.atoms[4], molecule.atoms[3])

    (environment,) = ff.collect_coordination_environments(molecule)

    assert environment.metal_idx == 4
    assert environment.coordination_number == 2
    assert environment.donor_indices == (0, 3)
    assert environment.chelate_groups == ((0, 3),)


@pytest.mark.parametrize(
    ("smiles", "expected_atomic_number", "expected_coordination"),
    (
        ("Cl[Zn]Cl", 30, 2),
        ("[Zn](N)(N)(N)N", 30, 4),
        ("[Pt](Cl)(Cl)(Cl)Cl", 78, 4),
        ("[Eu](N)(N)(N)(N)(N)N", 63, 6),
    ),
)
def test_coordination_environment_counts_explicit_donors(
    smiles,
    expected_atomic_number,
    expected_coordination,
):
    molecule = read_mol(smiles, "smi")

    (environment,) = ff.collect_coordination_environments(molecule)

    assert environment.metal_atomic_number == expected_atomic_number
    assert environment.coordination_number == expected_coordination
    assert len(environment.donor_indices) == expected_coordination


def test_coordination_environments_preserve_a_bridging_donor():
    molecule = read_mol("N.[Zn].[Zn]", "smi")
    molecule.add_bond(molecule.atoms[1], molecule.atoms[0])
    molecule.add_bond(molecule.atoms[2], molecule.atoms[0])

    environments = ff.collect_coordination_environments(molecule)

    assert len(environments) == 2
    assert all(environment.donor_indices == (0,) for environment in environments)
    assert all(environment.chelate_groups == ((0,),) for environment in environments)


def test_small_zinc_build_uses_default_inactive_coordination_hook(monkeypatch):
    molecule = read_mol("[Zn](N)(N)", "smi")
    monkeypatch.setattr(
        ff,
        "prepare_coordination_geometry",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the reserved hook must remain inactive by default")
        ),
    )

    report = ff.build_complex3d(
        molecule,
        candidate_count=1,
        max_attempts=3,
        candidate_warmup_steps=2,
        candidate_score_steps=2,
        best_candidate_refine_steps=2,
        timeout=10.0,
    )

    assert report.effective_forcefield == "UFF"
    assert len(molecule.hydrogens) == 6
    assert len(molecule.c_bonds) == 2


def test_seeded_complex_proxy_build_preserves_finite_complete_structure():
    molecule = read_mol("[Zn](N)(N)", "smi")

    ff.build_complex3d(
        molecule,
        candidate_count=1,
        max_attempts=3,
        candidate_warmup_steps=2,
        candidate_score_steps=2,
        best_candidate_refine_steps=2,
        timeout=10.0,
        seed=37,
    )

    assert molecule.coordinates.shape == (9, 3)
    assert np.all(np.isfinite(molecule.coordinates))
    assert len(molecule.c_bonds) == 2
