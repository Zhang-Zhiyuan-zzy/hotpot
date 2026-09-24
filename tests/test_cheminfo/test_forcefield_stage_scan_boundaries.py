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
from hotpot.cheminfo.forcefields import ligand
from hotpot.cheminfo.forcefields import optimizer as optimizer_impl
from hotpot.cheminfo.forcefields import repair
from hotpot.cheminfo.forcefields import utils as ff
from hotpot.cheminfo.forcefields import workflows
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


def test_stage2_screens_hidden_candidate_against_full_graph_without_terminal_scan(
    monkeypatch,
):
    """Stage 2 screens the hypothetical bond before changing graph topology."""
    mol = read_mol("[Zn](N)", "smi")
    candidate_topologies = []

    def candidate_scan(bond, workspace):
        candidate_topologies.append(
            tuple(sorted(repair._bond_key(candidate) for candidate in mol.bonds))
        )
        assert bond not in mol.bonds
        assert workspace.plan.ring_scope == "full_graph"
        return _empty_screening_report(ring_scope="full_graph")

    monkeypatch.setattr(
        repair,
        "_screen_coordination_bond_relations",
        candidate_scan,
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
    assert candidate_topologies[0] == ()
    assert result.report.forced_bond_keys == ()
    assert result.report.rejected_piercing_trial_count == 0


def test_stage2_candidate_screen_uses_full_graph_and_ring_size_limit(monkeypatch):
    mol = read_mol("[Zn](N)", "smi")
    candidate = mol.bonds[0]
    mol.hide_bonds(candidate, clear_conformers=False)
    plans = []
    workspaces = []
    segments = []

    def prepare_plan(current, **kwargs):
        plans.append((current, kwargs))
        return SimpleNamespace(ring_scope=kwargs["ring_scope"])

    def prepare_frame(plan):
        workspaces.append(plan)
        return SimpleNamespace(plan=plan)

    def screen(bonds, workspace, **kwargs):
        segments.append((tuple(bonds), workspace, kwargs))
        return _empty_screening_report(ring_scope=workspace.plan.ring_scope)

    monkeypatch.setattr(
        repair.geo,
        "prepare_bond_ring_screening_plan",
        prepare_plan,
    )
    monkeypatch.setattr(repair.geo, "prepare_bond_ring_frame", prepare_frame)
    monkeypatch.setattr(
        repair.geo,
        "screen_segments_against_ring_workspace",
        screen,
    )

    workspace = repair._prepare_coordination_screening_workspace(mol)
    report = repair._screen_coordination_bond_relations(candidate, workspace)

    assert candidate not in mol.bonds
    assert plans == [(
        mol,
        {
            "ring_scope": "full_graph",
            "max_ring_size": 16,
            "bonds": (),
        },
    )]
    assert len(workspaces) == 1
    assert workspaces[0].ring_scope == "full_graph"
    assert len(segments) == 1
    bond_geometries, workspace, options = segments[0]
    assert workspace.plan.ring_scope == "full_graph"
    assert options == {"stop_after_confirmed": True}
    assert tuple(geometry.bond for geometry in bond_geometries) == (candidate,)
    assert tuple(geometry.key for geometry in bond_geometries) == (
        repair._bond_key(candidate),
    )
    assert report.ring_scope == "full_graph"


def test_stage2_reuses_one_workspace_per_unchanged_candidate_batch(monkeypatch):
    mol = read_mol("[Zn](N)(N)", "smi")
    candidate_keys = tuple(sorted(repair._bond_key(bond) for bond in mol.bonds))
    plans = []
    frames = []
    screening_calls = []

    def prepare_plan(current, **kwargs):
        plan = SimpleNamespace(index=len(plans), mol=current, kwargs=kwargs)
        plans.append(plan)
        return plan

    def prepare_frame(plan):
        workspace = SimpleNamespace(index=len(frames), plan=plan)
        frames.append(workspace)
        return workspace

    def screen(bond, workspace):
        screening_calls.append((repair._bond_key(bond), workspace.index))
        return SimpleNamespace(bond=bond, workspace=workspace)

    def relation_counts(report, bond):
        assert report.bond is bond
        first_candidate_is_initially_blocked = (
            repair._bond_key(bond) == candidate_keys[0]
            and report.workspace.index == 0
        )
        return repair._CoordinationRelationCounts(
            piercing=int(first_candidate_is_initially_blocked),
            undetermined=0,
            excluded_rings=0,
        )

    monkeypatch.setattr(
        repair.geo,
        "prepare_bond_ring_screening_plan",
        prepare_plan,
    )
    monkeypatch.setattr(repair.geo, "prepare_bond_ring_frame", prepare_frame)
    monkeypatch.setattr(repair, "_screen_coordination_bond_relations", screen)
    monkeypatch.setattr(
        repair,
        "_candidate_coordination_relation_counts",
        relation_counts,
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

    assert len(plans) == 2
    assert len(frames) == 2
    assert all(
        plan.kwargs == {
            "ring_scope": "full_graph",
            "max_ring_size": 16,
            "bonds": (),
        }
        for plan in plans
    )
    assert screening_calls == [
        (candidate_keys[0], 0),
        (candidate_keys[1], 0),
        (candidate_keys[0], 1),
    ]
    assert result.report.forced_bond_keys == ()


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


def test_stage3_reuses_full_graph_terminal_checkpoint_for_acceptance(monkeypatch):
    """Stage 3 scans full graph only at its explicit boundary checkpoints."""
    mol = read_mol("[Zn](N)", "smi")
    calls = []
    scanned_reports = []
    run_report = _forcefield_run_report()

    def untangle(*args, **kwargs):
        calls.append(("untangle", kwargs["checkpoint_report"].ring_scope))
        return _untangling_result(attempt_limit=kwargs["attempt_limit"])

    def optimize(*args, **kwargs):
        calls.append(("optimizer",))
        return run_report

    def scan(*args, **kwargs):
        calls.append(("checkpoint", kwargs["ring_scope"]))
        report = _empty_screening_report(ring_scope=kwargs["ring_scope"])
        scanned_reports.append(report)
        return report

    def accept(*args, **kwargs):
        calls.append((
            "terminal_acceptance",
            kwargs["forcefield_stage"],
            kwargs["bond_ring_report"],
        ))
        return ff.ForceFieldValidationReport(
            level="basic",
            passed=True,
            checks=(),
        )

    monkeypatch.setattr(workflows, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(workflows, "_optimize_working_mol", optimize)
    monkeypatch.setattr(workflows, "_scan_ring_checkpoint", scan)
    monkeypatch.setattr(
        workflows,
        "evaluate_structure_acceptance_at_checkpoint",
        accept,
    )
    monkeypatch.setattr(
        workflows,
        "_combine_forcefield_run_reports",
        lambda reports: reports[-1],
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        mol,
        start=TrajectoryStart.COMPLEX_UNTANGLING,
    )

    result = workflows._optimize_complex_working_mol(
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
        ("checkpoint", "full_graph"),
        ("optimizer",),
        ("checkpoint", "full_graph"),
        ("terminal_acceptance", "final", scanned_reports[-1]),
    ]
    assert result.untangling.attempts_completed == 0
    assert result.untangling.initial_piercing_count == 0
    checkpoint_frames = tuple(
        frame
        for frame in trajectory
        if frame.event is TrajectoryEvent.TOPOLOGY_CHECKPOINT
    )
    assert len(checkpoint_frames) == len(scanned_reports) == 2
    assert all(
        isinstance(frame.evidence, RingFrameEvidence)
        and frame.evidence.ring_scope == "full_graph"
        and frame.evidence.scan_complete is True
        for frame in checkpoint_frames
    )


def test_stage3_passes_entry_checkpoint_directly_to_repair(monkeypatch):
    mol = read_mol("[Zn](N)", "smi")
    entry_report = _piercing_screening_report(ring_scope="full_graph")
    repaired_report = _empty_screening_report(ring_scope="full_graph")
    terminal_report = _empty_screening_report(ring_scope="full_graph")
    scan_reports = iter((entry_report, terminal_report))
    repaired_inputs = []
    accepted_reports = []

    def untangle(*args, **kwargs):
        repaired_inputs.append(kwargs["checkpoint_report"])
        return repair._RingUntanglingResult(
            report=ff.RingUntanglingReport(
                attempt_limit=kwargs["attempt_limit"],
                attempts_completed=1,
                initial_piercing_count=1,
                final_piercing_count=0,
                minimum_piercing_count=0,
                resolved=True,
            ),
            energy=0.0,
            checkpoint_report=repaired_report,
        )

    monkeypatch.setattr(
        workflows,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: next(scan_reports),
    )
    monkeypatch.setattr(workflows, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(
        workflows,
        "_optimize_working_mol",
        lambda *args, **kwargs: _forcefield_run_report(),
    )
    monkeypatch.setattr(
        workflows,
        "_combine_forcefield_run_reports",
        lambda reports: reports[-1],
    )
    monkeypatch.setattr(
        workflows,
        "evaluate_structure_acceptance_at_checkpoint",
        lambda *args, **kwargs: (
            accepted_reports.append(kwargs["bond_ring_report"])
            or ff.ForceFieldValidationReport(
                level="basic",
                passed=True,
                checks=(),
            )
        ),
    )

    workflows._optimize_complex_working_mol(
        mol,
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        algorithm="conjugate",
        epochs=1,
        steps_per_epoch=1,
        complex_untangling_attempts=2,
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
        trajectory=ForceFieldTrajectory.from_molecule(
            mol,
            start=TrajectoryStart.FINAL_OPTIMIZATION,
        ),
    )

    assert repaired_inputs == [entry_report]
    assert accepted_reports == [terminal_report]


def test_stage3_does_not_optimize_an_unresolved_entry_piercing(monkeypatch):
    mol = read_mol("[Zn](N)", "smi")
    checkpoint_report = _piercing_screening_report(ring_scope="full_graph")
    accepted_reports = []

    def fail_optimize(*args, **kwargs):
        raise AssertionError("topology-blocked coordinates reached optimizer")

    monkeypatch.setattr(
        workflows,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: checkpoint_report,
    )
    monkeypatch.setattr(
        workflows,
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
            energy=float("nan"),
            checkpoint_report=checkpoint_report,
        ),
    )
    monkeypatch.setattr(
        workflows,
        "_optimize_working_mol",
        fail_optimize,
    )
    monkeypatch.setattr(workflows.warnings, "warn", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        workflows,
        "evaluate_structure_acceptance_at_checkpoint",
        lambda *args, **kwargs: (
            accepted_reports.append(kwargs["bond_ring_report"])
            or ff.ForceFieldValidationReport(
                level="basic",
                passed=False,
                checks=(),
            )
        ),
    )

    result = workflows._optimize_complex_working_mol(
        mol,
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        algorithm="conjugate",
        epochs=1,
        steps_per_epoch=1,
        complex_untangling_attempts=2,
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
        trajectory=ForceFieldTrajectory.from_molecule(
            mol,
            start=TrajectoryStart.FINAL_OPTIMIZATION,
        ),
    )

    assert result.epochs_completed == 0
    assert result.termination_reason == "topology_blocked"
    assert result.untangling.final_piercing_count == 1
    assert accepted_reports == [checkpoint_report]


def test_stage3_invalidates_numerical_evidence_after_unresolved_repair(
    monkeypatch,
):
    """A repaired return frame must not inherit an earlier frame's numbers."""
    mol = read_mol("[Zn](N)", "smi")
    optimized_coordinates = np.full_like(mol.coordinates, 1.0)
    repaired_coordinates = np.full_like(mol.coordinates, 2.0)
    clear_report = _empty_screening_report(ring_scope="full_graph")
    piercing_report = _piercing_screening_report(ring_scope="full_graph")
    scan_reports = iter((clear_report, piercing_report))
    accepted_evidence = []

    numerical_report = ff.ForceFieldRunReport(
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        setup_succeeded=True,
        converged=True,
        epochs_completed=3,
        steps_submitted=30,
        initialization_steps=1,
        steps_completed=None,
        final_energy=12.0,
        best_energy=10.0,
        energy_unit="kJ/mol",
        rms_gradient=1.5,
        max_gradient=2.5,
        exploded=False,
        best_epoch=1,
        selected_segment_epochs_completed=2,
        epoch_energies=(12.0, 10.0, 11.0),
    )

    def optimize(working_mol, *args, **kwargs):
        working_mol.coordinates = optimized_coordinates
        return numerical_report

    def untangle(working_mol, *args, **kwargs):
        working_mol.coordinates = repaired_coordinates
        return repair._RingUntanglingResult(
            report=ff.RingUntanglingReport(
                attempt_limit=kwargs["attempt_limit"],
                attempts_completed=1,
                initial_piercing_count=1,
                final_piercing_count=1,
                minimum_piercing_count=1,
                resolved=False,
            ),
            energy=float("nan"),
            checkpoint_report=piercing_report,
        )

    monkeypatch.setattr(
        workflows,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: next(scan_reports),
    )
    monkeypatch.setattr(workflows, "_optimize_working_mol", optimize)
    monkeypatch.setattr(workflows, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(workflows.warnings, "warn", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        workflows,
        "evaluate_structure_acceptance_at_checkpoint",
        lambda *args, **kwargs: (
            accepted_evidence.append(kwargs["forcefield_report"])
            or ff.ForceFieldValidationReport(
                level="basic",
                passed=False,
                checks=(),
            )
        ),
    )

    result = workflows._optimize_complex_working_mol(
        mol,
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        algorithm="conjugate",
        epochs=3,
        steps_per_epoch=10,
        complex_untangling_attempts=1,
        quality_level="basic",
        topology_reference=object(),
        quality_thresholds=None,
        seed=7,
        perturb_interval=None,
        perturb_sigma=0.0,
        retain_epoch_history=True,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
        trajectory=ForceFieldTrajectory.from_molecule(
            mol,
            start=TrajectoryStart.FINAL_OPTIMIZATION,
        ),
    )

    assert np.array_equal(mol.coordinates, repaired_coordinates)
    assert result.epochs_completed == numerical_report.epochs_completed
    assert result.steps_submitted == numerical_report.steps_submitted
    assert result.epoch_energies == numerical_report.epoch_energies
    assert result.best_epoch == -1
    assert result.selected_segment_epochs_completed == 0
    assert result.termination_reason == "topology_blocked"
    assert result.setup_succeeded is False
    assert np.isnan(result.final_energy)
    assert np.isnan(result.best_energy)
    assert np.isnan(result.rms_gradient)
    assert np.isnan(result.max_gradient)
    assert len(accepted_evidence) == 1
    assert accepted_evidence[0]["setup_succeeded"] is False
    assert np.isnan(accepted_evidence[0]["final_energy"])
    assert np.isnan(accepted_evidence[0]["rms_gradient"])
    assert np.isnan(accepted_evidence[0]["max_gradient"])


def test_stage3_stabilizes_coordinates_changed_by_an_earlier_repair(
    monkeypatch,
):
    """Repair history is compared with the last numerically evaluated frame."""
    mol = read_mol("[Zn](N)", "smi")
    first_optimized_coordinates = np.full_like(mol.coordinates, 1.0)
    repaired_coordinates = np.full_like(mol.coordinates, 2.0)
    final_optimized_coordinates = np.full_like(mol.coordinates, 3.0)
    clear_report = _empty_screening_report(ring_scope="full_graph")
    piercing_report = _piercing_screening_report(ring_scope="full_graph")
    scan_reports = iter((clear_report, piercing_report, clear_report))
    repair_reports = iter((piercing_report, clear_report))
    initial_energies = []
    optimize_calls = 0

    def optimize(working_mol, *args, **kwargs):
        nonlocal optimize_calls
        optimize_calls += 1
        working_mol.coordinates = (
            first_optimized_coordinates
            if optimize_calls == 1
            else final_optimized_coordinates
        )
        return _forcefield_run_report()

    def untangle(working_mol, *args, **kwargs):
        initial_energies.append(kwargs["initial_energy"])
        if len(initial_energies) == 1:
            working_mol.coordinates = repaired_coordinates
        checkpoint_report = next(repair_reports)
        return repair._RingUntanglingResult(
            report=ff.RingUntanglingReport(
                attempt_limit=kwargs["attempt_limit"],
                attempts_completed=1,
                initial_piercing_count=1,
                final_piercing_count=(
                    1
                    if checkpoint_report.state is geo.PiercingState.PIERCES
                    else 0
                ),
                minimum_piercing_count=(
                    1
                    if checkpoint_report.state is geo.PiercingState.PIERCES
                    else 0
                ),
                resolved=(
                    checkpoint_report.state is not geo.PiercingState.PIERCES
                ),
            ),
            energy=float("nan"),
            checkpoint_report=checkpoint_report,
        )

    monkeypatch.setattr(
        workflows,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: next(scan_reports),
    )
    monkeypatch.setattr(workflows, "_optimize_working_mol", optimize)
    monkeypatch.setattr(workflows, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(
        workflows,
        "evaluate_structure_acceptance_at_checkpoint",
        lambda *args, **kwargs: ff.ForceFieldValidationReport(
            level="basic",
            passed=True,
            checks=(),
        ),
    )

    result = workflows._optimize_complex_working_mol(
        mol,
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        algorithm="conjugate",
        epochs=1,
        steps_per_epoch=1,
        complex_untangling_attempts=2,
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
        trajectory=ForceFieldTrajectory.from_molecule(
            mol,
            start=TrajectoryStart.FINAL_OPTIMIZATION,
        ),
    )

    assert optimize_calls == 2
    assert initial_energies[0] == 0.0
    assert np.isnan(initial_energies[1])
    assert np.array_equal(mol.coordinates, final_optimized_coordinates)
    assert result.epochs_completed == 2
    assert result.best_epoch == 1


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
