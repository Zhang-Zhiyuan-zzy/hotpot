"""Characterize force-field topology scans during the three-stage refactor.

These tests intentionally record the current scan boundaries.  Assertions marked
as current behavior are expected to change when the corresponding refactor step
moves topology checks to explicit stage checkpoints.
"""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from hotpot import read_mol
from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.forcefields import backend as ob_backend
from hotpot.cheminfo.forcefields import ff, ligand, repair
from hotpot.cheminfo.forcefields import optimizer as optimizer_impl
from hotpot.cheminfo.forcefields.trajectory import (
    ForceFieldTrajectory,
    RingFrameEvidence,
    TrajectoryEvent,
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


def _piercing_screening_report(*, ring_scope="ligand_skeleton"):
    finding = SimpleNamespace(
        relation=SimpleNamespace(state=geo.PiercingState.PIERCES),
    )
    return geo.BondRingScreeningReport(
        actionable_findings=(finding,),
        ring_scope=ring_scope,
        max_ring_size=16,
        selected_ring_count=1,
        excluded_ring_count=0,
        candidate_pair_count=1,
        aabb_separated_pair_count=0,
        exact_pair_count=1,
        piercing_pair_count=1,
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


def test_stage1_reuses_each_checkpoint_report_for_acceptance(monkeypatch):
    """Stage 1 acceptance consumes the exact report returned by its checkpoint."""
    mol = read_mol("[Zn](N)", "smi")
    calls = []
    scanned_reports = []
    accepted_reports = []
    trajectories = []
    candidate_terminal_report = _empty_screening_report(
        ring_scope="ligand_skeleton"
    )

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
        checkpoint_report = kwargs["checkpoint_report"]
        calls.append(("untangle", checkpoint_report.ring_scope))
        assert checkpoint_report is scanned_reports[-1]
        result = _untangling_result(attempt_limit=kwargs["attempt_limit"])
        return repair._RingUntanglingResult(
            report=result.report,
            energy=result.energy,
            checkpoint_report=candidate_terminal_report,
        )

    def scan(*args, **kwargs):
        calls.append(("checkpoint", kwargs["ring_scope"]))
        report = _empty_screening_report(ring_scope=kwargs["ring_scope"])
        scanned_reports.append(report)
        return report

    def accept(*args, **kwargs):
        accepted_reports.append(kwargs["bond_ring_report"])
        return ff.ForceFieldValidationReport(
            level="basic",
            passed=True,
            checks=(),
        )

    monkeypatch.setattr(ligand, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(ligand, "_scan_ring_checkpoint", scan)
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        accept,
    )

    ligand._build_ligand_proxies(
        mol,
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
        trajectory_attempts=trajectories,
    )

    assert calls == [
        ("checkpoint", "ligand_skeleton"),
        ("untangle", "ligand_skeleton"),
        ("checkpoint", "ligand_skeleton"),
    ]
    assert accepted_reports[0] is candidate_terminal_report
    assert accepted_reports[1] is scanned_reports[1]
    checkpoint_frames = tuple(
        frame
        for frame in trajectories[0]
        if frame.event is TrajectoryEvent.TOPOLOGY_CHECKPOINT
    )
    assert len(checkpoint_frames) == len(scanned_reports) == 2
    assert all(
        isinstance(frame.evidence, RingFrameEvidence)
        and frame.evidence.ring_scope == "ligand_skeleton"
        and frame.evidence.scan_complete is True
        for frame in checkpoint_frames
    )


def test_stage1_basic_gate_still_rejects_confirmed_piercing(monkeypatch):
    mol = read_mol("[Zn](N)", "smi")
    piercing_report = _piercing_screening_report()

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
    monkeypatch.setattr(
        ligand,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: _empty_screening_report(
            ring_scope="ligand_skeleton"
        ),
    )
    monkeypatch.setattr(
        ligand,
        "_untangle_ring_piercings",
        lambda *args, **kwargs: repair._RingUntanglingResult(
            report=ff.RingUntanglingReport(
                attempt_limit=kwargs["attempt_limit"],
                attempts_completed=0,
                initial_piercing_count=1,
                final_piercing_count=1,
                minimum_piercing_count=1,
                resolved=False,
            ),
            energy=0.0,
            checkpoint_report=piercing_report,
        ),
    )
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        lambda *args, **kwargs: ff.ForceFieldValidationReport(
            level="basic",
            passed=True,
            checks=(),
        ),
    )

    _, diagnostics = ligand._build_ligand_proxies(
        mol,
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
    )

    assert diagnostics.accepted_candidates == 0
    assert diagnostics.ligand_untangling[0].final_piercing_count == 1


def test_stage1_refinement_piercing_reenters_untangling(monkeypatch):
    mol = read_mol("[Zn](N)", "smi")
    entry_report = _empty_screening_report(ring_scope="ligand_skeleton")
    refined_report = _piercing_screening_report(ring_scope="ligand_skeleton")
    candidate_terminal = _empty_screening_report(ring_scope="ligand_skeleton")
    refined_terminal = _empty_screening_report(ring_scope="ligand_skeleton")
    scan_reports = iter((entry_report, refined_report))
    untangling_inputs = []
    accepted_reports = []

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
    monkeypatch.setattr(
        ligand,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: next(scan_reports),
    )

    def untangle(*args, **kwargs):
        checkpoint_report = kwargs["checkpoint_report"]
        untangling_inputs.append(checkpoint_report)
        terminal_report = (
            candidate_terminal
            if checkpoint_report is entry_report
            else refined_terminal
        )
        return repair._RingUntanglingResult(
            report=ff.RingUntanglingReport(
                attempt_limit=kwargs["attempt_limit"],
                attempts_completed=0,
                initial_piercing_count=len(checkpoint_report.piercings),
                final_piercing_count=0,
                minimum_piercing_count=0,
                resolved=True,
            ),
            energy=0.0,
            checkpoint_report=terminal_report,
        )

    def accept(*args, **kwargs):
        accepted_reports.append(kwargs["bond_ring_report"])
        return ff.ForceFieldValidationReport(
            level="basic",
            passed=True,
            checks=(),
        )

    monkeypatch.setattr(ligand, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(
        ligand,
        "evaluate_structure_acceptance_at_checkpoint",
        accept,
    )

    _, diagnostics = ligand._build_ligand_proxies(
        mol,
        max_attempts=1,
        candidate_warmup_steps=1,
        candidate_score_steps=1,
        best_candidate_refine_steps=1,
        effective_forcefield="UFF",
    )

    assert diagnostics.accepted_candidates == 1
    assert untangling_inputs == [entry_report, refined_report]
    assert accepted_reports == [candidate_terminal, refined_terminal]


def test_coordination_restoration_report_has_stage2_mechanical_facts_only():
    assert tuple(field.name for field in fields(ff.CoordinationBondRestorationReport)) == (
        "attempt_limit",
        "attempts_completed",
        "bond_count",
        "metal_relocation_attempt_count",
        "relocated_metal_indices",
        "infeasible_metal_indices",
        "forced_bond_keys",
        "rejected_piercing_trial_count",
        "undetermined_trial_count",
        "excluded_ring_observation_count",
        "warning_messages",
    )


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
    native_frames = tuple(
        SimpleNamespace(
            coordinates=coordinates,
            energy=float(epoch_index + 1),
            rms_gradient=0.0,
            max_gradient=0.0,
            exploded=False,
            converged=False,
            epoch_index=epoch_index,
            energy_change=None,
            max_displacement=None,
        )
        for epoch_index, coordinates in enumerate(frames)
    )
    native_calls = []
    topology_calls = []

    def native_optimize(current, forcefield, **options):
        native_calls.append((current, forcefield, options))
        return SimpleNamespace(
            coordinates=frames[-1],
            frames=native_frames,
            selected_frame_index=2,
            best_epoch=2,
            final_energy=3.0,
            best_energy=3.0,
            rms_gradient=0.0,
            max_gradient=0.0,
            exploded=False,
            converged=False,
            epochs_completed=3,
            steps_submitted=6,
            initialization_steps=1,
            selected_segment_epochs_completed=3,
            energy_unit="kJ/mol",
            backend_energy_unit="kJ/mol",
            termination_reason="budget_exhausted",
            terminal_converged=False,
            energy_changes=(),
            max_displacements=(),
            epoch_energies=(),
        )

    monkeypatch.setattr(optimizer_impl, "_native_optimize", native_optimize)

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
    assert len(native_calls) == 1
    assert native_calls[0][0] is mol
    assert native_calls[0][1] == "UFF"
    assert native_calls[0][2]["epochs"] == 3
    assert not hasattr(optimizer_impl, "evaluate_structure_acceptance")
    assert topology_calls == []
