import multiprocessing as mp
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo import forcefields as ff


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


def _malformed_worker(connection):
    connection.send("not a BuildWorkerResult")
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


class _StubbornProcess:
    def __init__(self):
        self.started = False
        self.terminated = False
        self.killed = False

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started and not self.killed

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        return None


class _DummyAtom:
    def __init__(self, atom_id):
        self.id = atom_id
        self.coordinates = np.zeros(3)


class _DummyComponent:
    has_metal = False

    def __init__(self):
        self.coordinates = np.zeros((2, 3))
        self.atoms = [_DummyAtom(0), _DummyAtom(1)]
        self.hidden = []

    def recover_hided_covalent_bonds(self, clear_conformers=False):
        return None

    def hide_bonds(self, *bonds, clear_conformers=False):
        self.hidden.extend(bonds)

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
    result = ff._receive_worker_result(
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
        ff._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=5.0,
        )
    assert caught.value.error_type == "ValueError"
    assert caught.value.error_message == "dative conversion failed"
    assert caught.value.worker_traceback == "worker traceback"


def test_timeout_terminates_and_joins_worker():
    process, receive_connection, send_connection = _pipe_process(_blocking_worker)
    with pytest.raises(ff.ComplexBuildTimeoutError):
        ff._receive_worker_result(
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
    monkeypatch.setattr(ff.mp, "get_context", lambda method: fork_context)
    monkeypatch.setattr(ff, "_run_complexes_build", worker)

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
        ff._receive_worker_result(
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
        ff._receive_worker_result(
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
        ff._receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=5.0,
        )
    assert caught.value.error_type == "WorkerProtocolError"


def test_seed_is_forwarded_to_worker_without_mutating_parent_environment(monkeypatch):
    monkeypatch.setenv("OB_RANDOM_SEED", "parent")
    process, receive_connection, send_connection = _pipe_process(_send_seed_worker)

    result = ff._receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=5.0,
        seed=37,
    )

    assert result.coordinates[0, 0] == 37.0
    assert os.environ["OB_RANDOM_SEED"] == "parent"


def test_unseeded_worker_start_uses_the_seed_environment_lock(monkeypatch):
    class CountingLock:
        entered = 0

        def __enter__(self):
            self.entered += 1

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    lock = CountingLock()
    monkeypatch.setattr(ff, "_SEED_ENVIRONMENT_LOCK", lock)
    process, receive_connection, send_connection = _pipe_process(_send_small_worker)

    ff._receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=5.0,
    )

    assert lock.entered == 1


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
        result = ff._receive_worker_result(
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
        ff._validated_worker_coordinates(result, expected_atom_count=2)

    assert caught.value.error_type == "WorkerProtocolError"


def test_candidate_attempts_are_bounded_and_use_geometry_module(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    calls = {"build": 0, "find": 0, "closest": 0}

    def fake_build(current):
        calls["build"] += 1

    monkeypatch.setattr(ff, "ob_build", fake_build)
    monkeypatch.setattr(
        ff,
        "_single_ob_optimization",
        lambda *args, **kwargs: ff._CandidateOptimizationResult(1.0, "kJ/mol", False),
    )
    monkeypatch.setattr(
        ff.geo,
        "capture_topology",
        lambda mol, **options: object(),
    )

    def intersections(*args, **kwargs):
        calls["find"] += 1
        return (("ring", "probe"),)

    def closest(*args, **kwargs):
        calls["closest"] += 1
        return SimpleNamespace(a1idx=0, a2idx=1)

    monkeypatch.setattr(ff.geo, "find_bond_ring_intersections", intersections)
    monkeypatch.setattr(ff.geo, "closest_ring_edge_to_bond", closest)

    with pytest.raises(ff.ComplexBuildError) as caught:
        ff._build_ligand_proxies(
            molecule,
            candidate_count=1,
            max_attempts=3,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            effective_forcefield="UFF",
        )

    assert calls == {"build": 3, "find": 3, "closest": 3}
    assert caught.value.diagnostics.attempt_count == 3
    assert caught.value.diagnostics.accepted_candidates == 0
    assert len(caught.value.diagnostics.rejected_candidates) == 3


def test_builder_failures_consume_the_attempt_budget(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    calls = 0

    def fail_build(current):
        nonlocal calls
        calls += 1
        raise ff.ForceFieldError("builder failed")

    monkeypatch.setattr(ff, "ob_build", fail_build)
    monkeypatch.setattr(
        ff.geo,
        "capture_topology",
        lambda mol, **options: object(),
    )

    with pytest.raises(ff.ComplexBuildError) as caught:
        ff._build_ligand_proxies(
            molecule,
            candidate_count=1,
            max_attempts=3,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            effective_forcefield="UFF",
        )

    assert calls == 3
    assert caught.value.diagnostics.attempt_count == 3
    assert len(caught.value.diagnostics.rejected_candidates) == 3


def test_refined_candidate_is_checked_before_coordinates_are_accepted(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    quality_calls = 0

    monkeypatch.setattr(ff, "ob_build", lambda current: None)
    monkeypatch.setattr(
        ff,
        "_single_ob_optimization",
        lambda *args, **kwargs: ff._CandidateOptimizationResult(1.0, "kJ/mol", False),
    )
    monkeypatch.setattr(
        ff.geo,
        "capture_topology",
        lambda mol, **options: object(),
    )
    monkeypatch.setattr(
        ff.geo, "find_bond_ring_intersections", lambda *args, **kwargs: ()
    )

    def quality(*args, **kwargs):
        nonlocal quality_calls
        quality_calls += 1
        return SimpleNamespace(passed=quality_calls == 1)

    monkeypatch.setattr(ff.geo, "evaluate_geometry_quality", quality)

    with pytest.raises(ff.ComplexBuildError, match="refinement failed") as caught:
        ff._build_ligand_proxies(
            molecule,
            candidate_count=1,
            max_attempts=1,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            effective_forcefield="UFF",
        )

    assert quality_calls == 2
    assert caught.value.diagnostics.rejected_candidates[-1].reason == (
        "refined candidate geometry gate"
    )


def test_refinement_tries_the_next_scored_candidate(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    state = {"build": 0, "refining": False}
    refined_markers = []

    def build(current):
        state["build"] += 1
        current.coordinates = np.full((2, 3), float(state["build"]))

    def optimize(current, forcefield, steps):
        marker = float(current.coordinates[0, 0])
        if steps == 3:
            state["refining"] = True
            refined_markers.append(marker)
        return ff._CandidateOptimizationResult(marker, "kJ/mol", False)

    def quality(current, **options):
        if not state["refining"]:
            return SimpleNamespace(passed=True)
        state["refining"] = False
        return SimpleNamespace(passed=float(current.coordinates[0, 0]) == 2.0)

    monkeypatch.setattr(ff, "ob_build", build)
    monkeypatch.setattr(ff, "_single_ob_optimization", optimize)
    monkeypatch.setattr(
        ff.geo,
        "capture_topology",
        lambda mol, **options: object(),
    )
    monkeypatch.setattr(
        ff.geo, "find_bond_ring_intersections", lambda *args, **kwargs: ()
    )
    monkeypatch.setattr(ff.geo, "evaluate_geometry_quality", quality)

    _, diagnostics = ff._build_ligand_proxies(
        molecule,
        candidate_count=2,
        max_attempts=2,
        candidate_warmup_steps=1,
        candidate_score_steps=2,
        best_candidate_refine_steps=3,
        effective_forcefield="UFF",
    )

    assert refined_markers == [1.0, 2.0]
    assert diagnostics.rejected_candidates[-1] == ff.CandidateRejection(
        component_index=0,
        attempt=1,
        reason="refined candidate geometry gate",
    )


def test_intersected_ring_edges_are_hidden_in_stable_endpoint_order(monkeypatch):
    component = _DummyComponent()
    molecule = _DummyComplex(component)
    first = SimpleNamespace(a1idx=4, a2idx=2)
    second = SimpleNamespace(a1idx=3, a2idx=1)

    monkeypatch.setattr(ff, "ob_build", lambda current: None)
    monkeypatch.setattr(
        ff,
        "_single_ob_optimization",
        lambda *args, **kwargs: ff._CandidateOptimizationResult(1.0, "kJ/mol", False),
    )
    monkeypatch.setattr(
        ff.geo,
        "capture_topology",
        lambda mol, **options: object(),
    )
    monkeypatch.setattr(
        ff.geo,
        "find_bond_ring_intersections",
        lambda *args, **kwargs: (("first", "probe"), ("second", "probe")),
    )
    monkeypatch.setattr(
        ff.geo,
        "closest_ring_edge_to_bond",
        lambda ring, bond: first if ring == "first" else second,
    )

    with pytest.raises(ff.ComplexBuildError):
        ff._build_ligand_proxies(
            molecule,
            candidate_count=1,
            max_attempts=1,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            effective_forcefield="UFF",
        )

    assert component.hidden == [second, first]


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
    monkeypatch.setattr(ff, "_build_ligand_proxies", fail)
    ff._run_complexes_build(
        object(),
        connection,
        1,
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
        return working, diagnostics

    def fail_final_stage(working, **options):
        working.coordinates = working.coordinates - 3.0
        working.remove_bonds([working.bonds[0]])
        raise failure

    monkeypatch.setattr(ff, "_build_complex_working", built_working_copy)
    monkeypatch.setattr(ff, "_run_optimizer_on_working", fail_final_stage)

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
    with pytest.raises(NotImplementedError, match="reserved but not implemented"):
        ff.prepare_coordination_geometry(SimpleNamespace(), strategy="octahedral")


@pytest.mark.parametrize(
    "options",
    (
        {"candidate_count": 0},
        {"candidate_count": 2, "max_attempts": 1},
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
