"""Characterize force-field topology scans before the three-stage refactor.

These tests intentionally record the current scan boundaries.  Assertions marked
as current behavior are expected to change when the corresponding refactor step
moves topology checks to explicit stage checkpoints.
"""

from types import SimpleNamespace

import numpy as np

from hotpot import read_mol
from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.forcefields import backend as ob_backend
from hotpot.cheminfo.forcefields import ligand
from hotpot.cheminfo.forcefields import optimizer as optimizer_impl
from hotpot.cheminfo.forcefields import repair
from hotpot.cheminfo.forcefields import utils as ff
from hotpot.cheminfo.forcefields import workflows
from hotpot.cheminfo.forcefields.trajectory import (
    ForceFieldTrajectory,
    TrajectoryStart,
)


def _empty_screening_report(*, ring_scope="full_graph"):
    return geo.BondRingScreeningReport(
        actionable_findings=(),
        ring_scope=ring_scope,
        max_ring_size=16,
        selected_ring_count=0,
        excluded_ring_count=0,
        candidate_pair_count=0,
        aabb_separated_pair_count=0,
        exact_pair_count=0,
        piercing_pair_count=0,
        does_not_pierce_pair_count=0,
        undetermined_pair_count=0,
        scan_complete=True,
    )


def _untangling_result(*, attempt_limit=1):
    checkpoint_report = _empty_screening_report(ring_scope="ligand_skeleton")
    return repair._RingUntanglingResult(
        report=ff.RingUntanglingReport(
            attempt_limit=attempt_limit,
            attempts_completed=0,
            initial_piercing_count=0,
            final_piercing_count=0,
            minimum_piercing_count=0,
            resolved=True,
        ),
        energy=0.0,
        checkpoint_report=checkpoint_report,
    )


def _forcefield_run_report(*, epochs_completed=1):
    return ff.ForceFieldRunReport(
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        setup_succeeded=True,
        converged=True,
        epochs_completed=epochs_completed,
        steps_submitted=1,
        initialization_steps=0,
        steps_completed=None,
        final_energy=0.0,
        best_energy=0.0,
        energy_unit="kJ/mol",
        rms_gradient=0.0,
        max_gradient=0.0,
        exploded=False,
    )


def test_current_stage1_runs_untangling_then_a_post_refinement_scan(monkeypatch):
    """Stage 1 currently adds one explicit scan after the untangling helper."""
    mol = read_mol("[Zn](N)", "smi")
    calls = []

    monkeypatch.setattr(ligand, "_ob_build", lambda current: None)
    monkeypatch.setattr(
        ligand,
        "_single_ob_optimization",
        lambda *args, **kwargs: ob_backend._CandidateOptimizationResult(
            0.0,
            "kJ/mol",
            False,
        ),
    )
    monkeypatch.setattr(ligand, "capture_topology", lambda *args, **kwargs: object())

    def untangle(*args, **kwargs):
        calls.append(("untangle", kwargs["checkpoint_report"].ring_scope))
        return _untangling_result(attempt_limit=kwargs["attempt_limit"])

    def scan(*args, **kwargs):
        calls.append(("checkpoint", kwargs["ring_scope"]))
        return _empty_screening_report(ring_scope=kwargs["ring_scope"])

    monkeypatch.setattr(ligand, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(ligand, "_scan_ring_checkpoint", scan)
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance",
        lambda *args, **kwargs: ff.ForceFieldValidationReport(
            level="basic",
            passed=True,
            checks=(),
        ),
    )

    ligand._build_ligand_proxies(
        mol,
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
    )

    assert calls == [
        ("checkpoint", "ligand_skeleton"),
        ("untangle", "ligand_skeleton"),
        ("checkpoint", "ligand_skeleton"),
    ]


def test_current_stage2_scans_each_restored_candidate_then_the_full_graph(
    monkeypatch,
):
    """Stage 2 currently mutates each candidate before a ligand-ring scan."""
    mol = read_mol("[Zn](N)", "smi")
    candidate_topologies = []
    terminal_scans = []

    def candidate_scan(current, bond):
        candidate_topologies.append(
            tuple(sorted(repair._bond_key(candidate) for candidate in current.bonds))
        )
        assert bond in current.bonds
        return _empty_screening_report(ring_scope="ligand_skeleton")

    def terminal_scan(current):
        terminal_scans.append(
            tuple(sorted(repair._bond_key(bond) for bond in current.bonds))
        )
        return _empty_screening_report(ring_scope="full_graph")

    monkeypatch.setattr(
        repair,
        "_screen_coordination_bond_relations",
        candidate_scan,
    )
    monkeypatch.setattr(
        repair,
        "_scan_full_graph_bond_ring_relations",
        terminal_scan,
    )
    monkeypatch.setattr(
        repair,
        "_single_ob_optimization",
        lambda *args, **kwargs: ob_backend._CandidateOptimizationResult(
            0.0,
            "kJ/mol",
            False,
        ),
    )

    result = repair._restore_coordination_bonds_incrementally(
        mol,
        "UFF",
        attempt_limit=1,
        relaxation_steps=1,
        perturb_sigma=0.0,
        rng=np.random.default_rng(7),
    )

    assert len(candidate_topologies) == 1
    assert candidate_topologies[0] == ((0, 1),)
    assert terminal_scans == [((0, 1),)]
    assert result.report.restored_without_forcing == 1


def test_current_stage3_uses_ligand_scope_before_and_after_optimizer(monkeypatch):
    """Stage 3 keeps topology scans outside the numerical optimizer."""
    mol = read_mol("[Zn](N)", "smi")
    calls = []
    run_report = _forcefield_run_report()

    def untangle(*args, **kwargs):
        calls.append(("untangle", kwargs["checkpoint_report"].ring_scope))
        return _untangling_result(attempt_limit=kwargs["attempt_limit"])

    def optimize(*args, **kwargs):
        calls.append(("optimizer",))
        return run_report

    def scan(*args, **kwargs):
        calls.append(("checkpoint", kwargs["ring_scope"]))
        return _empty_screening_report(ring_scope=kwargs["ring_scope"])

    def accept(*args, **kwargs):
        calls.append(("terminal_acceptance", kwargs["forcefield_stage"]))
        return ff.ForceFieldValidationReport(
            level="basic",
            passed=True,
            checks=(),
        )

    monkeypatch.setattr(workflows, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(workflows, "_optimize_working_mol", optimize)
    monkeypatch.setattr(workflows, "_scan_ring_checkpoint", scan)
    monkeypatch.setattr(workflows, "evaluate_structure_acceptance", accept)
    monkeypatch.setattr(
        workflows,
        "_combine_forcefield_run_reports",
        lambda reports: reports[-1],
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        mol,
        start=TrajectoryStart.FINAL_OPTIMIZATION,
    )

    workflows._optimize_complex_working_mol(
        mol,
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        algorithm="conjugate",
        epochs=1,
        steps_per_epoch=1,
        complex_untangling_attempts=1,
        quality_level="basic",
        topology_reference=object(),
        quality_thresholds=None,
        seed=7,
        perturb_interval=None,
        perturb_sigma=0.0,
        retain_epoch_history=False,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
        trajectory=trajectory,
    )

    assert calls == [
        ("checkpoint", "ligand_skeleton"),
        ("untangle", "ligand_skeleton"),
        ("optimizer",),
        ("checkpoint", "ligand_skeleton"),
        ("terminal_acceptance", "final"),
    ]


class _EpochBackend:
    def __init__(self, frames):
        self.frames = frames
        self.index = -1
        self.obmol = None

    def Setup(self, obmol, constraints):
        self.obmol = obmol
        return True

    def EnableCutOff(self, enabled):
        return None

    def ConjugateGradientsInitialize(self, steps, tolerance):
        return None

    def ConjugateGradientsTakeNSteps(self, steps):
        self.index += 1
        return True

    def GetCoordinates(self, obmol):
        if self.index >= 0:
            obmol.coordinates = self.frames[self.index].copy()

    def Energy(self, calc_grad=True):
        return float(self.index + 1)

    def GetGradient(self, atom):
        return SimpleNamespace(GetX=lambda: 0.0, GetY=lambda: 0.0, GetZ=lambda: 0.0)

    def DetectExplosion(self):
        return False

    def GetUnit(self):
        return "kJ/mol"


def test_numerical_optimizer_does_not_evaluate_acceptance_or_topology_each_epoch(
    monkeypatch,
):
    """The numerical loop leaves scientific acceptance to workflow checkpoints."""
    mol = read_mol("CC", "smi")
    frames = [
        np.full_like(mol.coordinates, 1.0),
        np.full_like(mol.coordinates, 2.0),
        np.full_like(mol.coordinates, 3.0),
    ]
    backend = _EpochBackend(frames)
    obmol = SimpleNamespace(coordinates=np.asarray(mol.coordinates).copy())
    topology_calls = []

    monkeypatch.setattr(optimizer_impl, "_get_forcefield", lambda name: backend)
    monkeypatch.setattr(ob_backend, "_make_constraints", lambda current: object())
    monkeypatch.setattr(
        optimizer_impl.ob,
        "OBMolAtomIter",
        lambda current: (object(),) * len(mol.atoms),
    )
    monkeypatch.setattr(
        optimizer_impl,
        "mol2obmol",
        lambda current: (obmol, {index: index + 1 for index in range(len(mol.atoms))}),
    )
    monkeypatch.setattr(
        optimizer_impl,
        "extract_obmol_coordinates",
        lambda current: np.asarray(current.coordinates).copy(),
    )

    def scan(*args, **kwargs):
        topology_calls.append(kwargs["ring_scope"])
        return geo.PiercingState.DOES_NOT_PIERCE

    monkeypatch.setattr(geo, "determine_bond_ring_piercing_state", scan)
    optimizer = optimizer_impl._OpenBabelOptimizer(
        "UFF",
        "UFF",
        algorithm="conjugate",
        epochs=3,
        steps_per_epoch=2,
        perturb_interval=None,
        perturb_sigma=0.0,
        retain_epoch_history=False,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
        seed=7,
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        mol,
        start=TrajectoryStart.FINAL_OPTIMIZATION,
    )

    report = optimizer.optimize(
        mol,
        trajectory=trajectory,
    )

    assert report.epochs_completed == 3
    assert not hasattr(optimizer_impl, "evaluate_structure_acceptance")
    assert topology_calls == []
