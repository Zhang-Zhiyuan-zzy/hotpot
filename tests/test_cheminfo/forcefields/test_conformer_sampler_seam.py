from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.conformers import (
    ConformerBatch,
    ConformerSamplingError,
    ConformerSamplingRequest,
    OpenBabelConformerAdapter,
    RDKitConformerAdapter,
)
from hotpot.cheminfo.forcefields import backend, ligand, workflows, workers
from hotpot.cheminfo.forcefields.contracts import (
    BuildWorkerResult,
    ComplexBuildDiagnostics,
    ComplexBuildError,
    RingUntanglingReport,
)
from hotpot.cheminfo.rdWrappers import (
    ConformerBatch as RDKitConformerBatch,
    ConformerRecord,
    GenerationStatus,
)
from hotpot.cheminfo.core_utils import read_mol


@dataclass
class _RecordingSampler:
    calls: list[tuple[int, ConformerSamplingRequest]] = field(default_factory=list)

    def sample(self, molecule, request):
        self.calls.append((len(molecule.atoms), request))
        value = float(len(self.calls))
        coordinates = np.full((1, len(molecule.atoms), 3), value)
        return ConformerBatch(coordinates, (0,))


class _Connection:
    def __init__(self):
        self.result = None
        self.closed = False

    def send(self, result):
        self.result = result

    def close(self):
        self.closed = True


class _SpawnProcess:
    def __init__(self, target, args):
        self.target = target
        self.args = args


class _SpawnContext:
    def __init__(self):
        self.process = None

    def Pipe(self, duplex):
        assert duplex is False
        return object(), object()

    def Process(self, *, target, args):
        self.process = _SpawnProcess(target, args)
        return self.process


def _accepted_report():
    return SimpleNamespace(
        state=geo.PiercingState.DOES_NOT_PIERCE,
        piercings=(),
        undetermined_pair_count=0,
        excluded_ring_count=0,
        max_ring_size=16,
    )


def _patch_accepted_downstream(monkeypatch, trace):
    report = _accepted_report()
    untangling_report = RingUntanglingReport(
        attempt_limit=4,
        attempts_completed=0,
        initial_piercing_count=0,
        final_piercing_count=0,
        minimum_piercing_count=0,
        resolved=True,
    )

    monkeypatch.setattr(ligand, "capture_topology", lambda *args, **kwargs: object())

    def optimize(molecule, forcefield, steps):
        trace.append(("optimize", forcefield, steps))
        return backend._CandidateOptimizationResult(1.0, "kJ/mol", False)

    def scan(molecule, *, ring_scope):
        trace.append(("scan", ring_scope))
        return report

    def record(*args, **kwargs):
        trace.append(("record", kwargs["stage"]))

    def untangle(*args, **kwargs):
        trace.append(
            (
                "untangle",
                kwargs["attempt_limit"],
                kwargs["short_steps"],
                kwargs["settling_steps"],
                kwargs["perturb_sigma"],
            )
        )
        return SimpleNamespace(
            checkpoint_report=report,
            energy=1.0,
            report=untangling_report,
        )

    def accept(*args, **kwargs):
        trace.append(("accept", kwargs["forcefield_stage"]))
        return SimpleNamespace(passed=True, failures=())

    monkeypatch.setattr(ligand, "_single_ob_optimization", optimize)
    monkeypatch.setattr(ligand, "_scan_ring_checkpoint", scan)
    monkeypatch.setattr(ligand, "_record_ring_checkpoint", record)
    monkeypatch.setattr(ligand, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        accept,
    )


def test_adapters_are_pickle_safe_and_prepare_passes_one_into_spawn(monkeypatch):
    for adapter in (OpenBabelConformerAdapter(), RDKitConformerAdapter()):
        assert type(pickle.loads(pickle.dumps(adapter))) is type(adapter)

    molecule = read_mol("[Zn].N", "smi")
    context = _SpawnContext()
    sampler = RDKitConformerAdapter()
    diagnostics = ComplexBuildDiagnostics(1, 1, (), 0.0)
    monkeypatch.setattr(
        workflows.mp,
        "get_context",
        lambda method: context if method == "spawn" else None,
    )

    def receive(process, *args, **kwargs):
        return BuildWorkerResult(
            status="ok",
            coordinates=process.args[0].coordinates,
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(workflows, "_receive_worker_result", receive)
    prepared = workflows._prepare_complex_working_mol(
        molecule,
        effective_forcefield="UFF",
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        timeout=1.0,
        add_hydrogens=False,
        seed=17,
        coordination_geometry=None,
        sampler=sampler,
    )

    assert prepared.diagnostics is diagnostics
    assert context.process is not None
    assert context.process.target is workers._build_ligand_proxies_worker
    assert context.process.args[-1] is sampler
    assert isinstance(pickle.loads(pickle.dumps(context.process.args[-1])), RDKitConformerAdapter)


def test_provider_runs_once_per_component_and_downstream_budgets_are_unchanged(
    monkeypatch,
):
    molecule = read_mol("[Zn].N.CC", "smi")
    sampler = _RecordingSampler()
    trace = []
    _patch_accepted_downstream(monkeypatch, trace)

    _, diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=7,
        candidate_warmup_steps=11,
        candidate_score_steps=22,
        best_candidate_refine_steps=33,
        effective_forcefield="UFF",
        ligand_untangling_attempts=4,
        perturb_sigma=0.25,
        seed=19,
        sampler=sampler,
    )

    assert sorted(atom_count for atom_count, _ in sampler.calls) == [1, 2]
    assert all(
        request == ConformerSamplingRequest(conformer_count=1, seed=19)
        for _, request in sampler.calls
    )
    assert diagnostics.attempt_count == 2
    assert diagnostics.accepted_candidates == 2
    assert trace == [
        ("optimize", "UFF", 11),
        ("scan", "ligand_skeleton"),
        ("record", ligand.TrajectoryStage.LIGAND_BUILD),
        ("untangle", 4, 11, 22, 0.25),
        ("accept", "candidate"),
        ("optimize", "UFF", 33),
        ("scan", "ligand_skeleton"),
        ("record", ligand.TrajectoryStage.LIGAND_BUILD),
        ("accept", "candidate"),
    ] * 2


def test_default_openbabel_path_matches_explicit_adapter_and_seeds_worker_once(
    monkeypatch,
):
    molecule = read_mol("[Zn].CC", "smi")
    trace = []
    _patch_accepted_downstream(monkeypatch, trace)
    build_calls = []

    def build(component):
        build_calls.append(len(component.atoms))
        component.coordinates = np.arange(
            len(component.atoms) * 3,
            dtype=float,
        ).reshape(-1, 3)

    monkeypatch.setattr(ligand, "_ob_build", build)
    default_coordinates, default_diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=2,
        best_candidate_refine_steps=3,
        effective_forcefield="UFF",
        seed=23,
    )
    default_trace = tuple(trace)

    trace.clear()
    build_calls.clear()
    explicit_coordinates, explicit_diagnostics = ligand._build_ligand_proxies(
        molecule,
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=2,
        best_candidate_refine_steps=3,
        effective_forcefield="UFF",
        seed=23,
        sampler=OpenBabelConformerAdapter(_builder=build),
    )

    assert build_calls == [2]
    np.testing.assert_array_equal(default_coordinates, explicit_coordinates)
    assert tuple(trace) == default_trace
    assert default_diagnostics.attempt_count == explicit_diagnostics.attempt_count
    assert (
        default_diagnostics.accepted_candidates
        == explicit_diagnostics.accepted_candidates
    )
    assert (
        default_diagnostics.rejected_candidates
        == explicit_diagnostics.rejected_candidates
    )

    seed_calls = []
    connection = _Connection()
    monkeypatch.setattr(
        workers._backend,
        "_seed_openbabel_random",
        lambda seed: seed_calls.append(seed),
    )
    monkeypatch.setattr(
        workers,
        "_build_ligand_proxies",
        lambda *args, **kwargs: (
            np.zeros((1, 3)),
            ComplexBuildDiagnostics(1, 1, (), 0.0),
        ),
    )
    workers._build_ligand_proxies_worker(
        object(),
        connection,
        1,
        1,
        1,
        1,
        "UFF",
        23,
        sampler=OpenBabelConformerAdapter(),
    )
    assert seed_calls == [23]
    assert connection.result.status == "ok"


def test_rdkit_adapter_requests_one_and_selects_the_lowest_stable_ordinal(
    monkeypatch,
):
    from hotpot.cheminfo import rdWrappers

    molecule = read_mol("CC", "smi")
    original = molecule.coordinates.copy()
    calls = []
    coordinates = np.stack(
        (
            np.full((len(molecule.atoms), 3), 9.0),
            np.full((len(molecule.atoms), 3), 2.0),
        )
    )

    def generate(current, *, options, constraints=None):
        calls.append((current, options, constraints))
        return RDKitConformerBatch(
            GenerationStatus.PARTIAL,
            coordinates,
            (ConformerRecord(9, 90), ConformerRecord(2, 20)),
            "rdkit-etkdg",
            "2026.09.1",
            None,
        )

    monkeypatch.setattr(rdWrappers, "generate_conformers", generate)
    sampled = RDKitConformerAdapter().sample(
        molecule,
        ConformerSamplingRequest(conformer_count=32, seed=29),
    )

    assert len(calls) == 1
    assert calls[0][0] is molecule
    assert calls[0][1].conformer_count == 1
    assert calls[0][1].seed == 29
    assert calls[0][2] is None
    assert sampled.ordinals == (2,)
    np.testing.assert_array_equal(sampled.coordinates, coordinates[1:2])
    np.testing.assert_array_equal(molecule.coordinates, original)


def test_rdkit_failure_is_explicit_and_never_falls_back_to_openbabel(monkeypatch):
    from hotpot.cheminfo import rdWrappers

    molecule = read_mol("[Zn].CC", "smi")
    fallback_calls = []

    monkeypatch.setattr(
        rdWrappers,
        "generate_conformers",
        lambda *args, **kwargs: RDKitConformerBatch(
            GenerationStatus.FAILED,
            np.empty((0, 2, 3)),
            (),
            "rdkit-etkdg",
            "2026.09.1",
            "embedding failed",
        ),
    )
    monkeypatch.setattr(
        ligand,
        "_ob_build",
        lambda component: fallback_calls.append(component),
    )

    with pytest.raises(ComplexBuildError, match="conformer provider") as caught:
        ligand._build_ligand_proxies(
            molecule,
            max_attempts=5,
            candidate_warmup_steps=1,
            candidate_score_steps=1,
            best_candidate_refine_steps=1,
            effective_forcefield="UFF",
            sampler=RDKitConformerAdapter(),
        )

    assert isinstance(caught.value.__cause__, ConformerSamplingError)
    assert "FAILED: embedding failed" in str(caught.value.__cause__)
    assert caught.value.diagnostics.attempt_count == 1
    assert fallback_calls == []


def test_public_forcefield_api_keeps_candidate_count_reserved(monkeypatch):
    marker = object()
    captured = {}

    def private_workflow(*args, **kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(workflows, "_complexes_build_workflow", private_workflow)
    assert workflows.complexes_build(object(), candidate_count=37) is marker
    assert "candidate_count" not in captured
    assert "sampler" not in captured
