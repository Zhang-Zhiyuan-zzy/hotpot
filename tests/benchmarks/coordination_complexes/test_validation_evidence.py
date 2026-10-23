"""Contracts for translating live force-field reports into Plan-17 gates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.forcefields import ff

from .validation_cohort import _graph_document, canonical_digest
from .validation_evidence import (
    evaluate_route,
    semantic_evidence,
    write_selected_structure,
)
from .validation_providers import component_sha256, coordinate_sha256
from .validation_routes import PROVIDER_STAGE_SEQUENCE, RouteExecution


def _complex() -> Molecule:
    molecule = Molecule()
    molecule.create_atom(atomic_number=7, id=11)
    molecule.create_atom(atomic_number=95, formal_charge=3, id=12)
    molecule.add_bond(0, 1, 1.0)
    molecule.coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (2.4, 0.0, 0.0)),
        dtype=np.float64,
    )
    return molecule


def _case(molecule: Molecule) -> dict[str, object]:
    graph = _graph_document(molecule)
    coordination = {
        "metal": {
            "row": 1,
            "atom_id": 12,
            "atomic_number": 95,
            "symbol": "Am",
        },
        "donor_rows": [0],
    }
    return {
        "index": 1,
        "status": "cbond",
        "graph": graph,
        "coordination": coordination,
        "semantic_graph_sha256": canonical_digest(
            {"graph": graph, "coordination": coordination}
        ),
    }


def _optimization_evidence(*, finite_gradients: bool = True):
    return ff.OptimizationFrameEvidence(
        converged=False,
        exploded=False,
        finite_coordinates=True,
        finite_energy=True,
        finite_gradients=finite_gradients,
        rms_gradient_kj_mol_angstrom=(0.2 if finite_gradients else None),
        max_gradient_kj_mol_angstrom=(0.4 if finite_gradients else None),
    )


def _quality() -> ff.ForceFieldValidationReport:
    return ff.ForceFieldValidationReport(
        level="standard",
        passed=True,
        checks=(ff.AcceptanceCheck("finite_coordinates", True),),
    )


def _execution(
    molecule: Molecule,
    *,
    first_frame_finite_gradients: bool = True,
    include_later_frame: bool = False,
    include_recovery_action: bool = False,
    include_ligand_proposal: bool = True,
    include_coordination_stage: bool = True,
) -> RouteExecution:
    ligand = Molecule()
    ligand.create_atom(atomic_number=7, id=11)
    ligand.coordinates = np.asarray(((0.0, 0.0, 0.0),), dtype=np.float64)
    ligand_trajectory = ff.ForceFieldTrajectory.from_molecule(
        ligand,
        start=ff.TrajectoryStart.LIGAND_BUILD,
    )
    ligand_trajectory.record_molecule(
        ligand,
        stage=ff.TrajectoryStage.LIGAND_BUILD,
        event=ff.TrajectoryEvent.INITIAL,
        component_index=0,
        attempt=1,
    )
    ligand_trajectory.record_molecule(
        ligand,
        stage=ff.TrajectoryStage.LIGAND_BUILD,
        event=ff.TrajectoryEvent.BUILD_COMPLETE,
        component_index=0,
        attempt=1,
    )

    main = ff.ForceFieldTrajectory.from_molecule(
        molecule,
        start=ff.TrajectoryStart.LIGAND_BUILD,
    )
    if include_coordination_stage:
        main.record_molecule(
            molecule,
            stage=ff.TrajectoryStage.COORDINATION_RESTORATION,
            event=ff.TrajectoryEvent.COORDINATION_READY,
            evidence=ff.CoordinationFrameEvidence(None, accepted=True),
        )
    first = main.record_molecule(
        molecule,
        stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
        event=ff.TrajectoryEvent.EPOCH_COMPLETE,
        energy_kj_mol=-1.0,
        step=0,
        evidence=_optimization_evidence(finite_gradients=first_frame_finite_gradients),
    )
    selected = first
    if include_later_frame:
        if include_recovery_action:
            main.record_molecule(
                molecule,
                stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
                event=ff.TrajectoryEvent.PERTURBED,
            )
        selected = main.record_molecule(
            molecule,
            stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
            event=ff.TrajectoryEvent.EPOCH_COMPLETE,
            energy_kj_mol=-2.0,
            step=1,
            evidence=_optimization_evidence(),
        )
    main.select(selected.index)
    main.set_terminal(selected.index)
    archive = ff.ForceFieldTrajectoryArchive(
        main=main,
        ligand_build_attempts=(ligand_trajectory,) if include_ligand_proposal else (),
    )
    quality = _quality()
    optimization = ff.ForceFieldRunReport(
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        setup_succeeded=True,
        converged=False,
        epochs_completed=2 if include_later_frame else 1,
        steps_submitted=20,
        initialization_steps=1,
        steps_completed=None,
        final_energy=-2.0 if include_later_frame else -1.0,
        best_energy=-2.0 if include_later_frame else -1.0,
        energy_unit="kJ/mol",
        rms_gradient=0.2,
        max_gradient=0.4,
        exploded=False,
        quality_report=quality,
        trajectory=archive,
        termination_reason="budget_exhausted",
    )
    report = ff.ComplexBuildReport(
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        build=ff.ComplexBuildDiagnostics(
            attempt_count=1,
            accepted_candidates=1,
            rejected_candidates=(),
            elapsed_seconds=0.1,
        ),
        optimization=optimization,
        quality_report=quality,
        trajectory=archive,
    )
    return RouteExecution(
        route="F1",
        molecule=molecule,
        report=report,
        proposal=None,
        proposal_manifest=(),
        stage_sequence=PROVIDER_STAGE_SEQUENCE,
        proposal_seconds=0.01,
        forcefield_seconds=0.10,
        elapsed_seconds=0.11,
    )


def test_successful_report_translates_to_all_ten_passing_gates() -> None:
    molecule = _complex()
    _, result = evaluate_route(_case(molecule), _execution(molecule))

    assert result.passed
    assert tuple(gate.gate_id for gate in result.gates) == tuple(
        f"A{index:02d}" for index in range(1, 11)
    )
    assert all(gate.passed for gate in result.gates)
    assert "not_strictly_converged" in result.diagnostics
    assert "terminal_reason:budget_exhausted" in result.diagnostics


def test_added_hydrogen_rows_do_not_change_original_graph_semantics() -> None:
    molecule = _complex()
    case = _case(molecule)
    molecule.create_atom(atomic_number=1, id=13)
    molecule.add_bond(0, 2, 1.0)
    molecule.coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (2.4, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )

    evidence = semantic_evidence(case, molecule)

    assert evidence.equal
    assert evidence.source_digest == evidence.result_digest


def test_unavailable_gradient_requires_explicit_successful_recovery() -> None:
    molecule = _complex()
    case = _case(molecule)
    execution = _execution(
        molecule,
        first_frame_finite_gradients=False,
        include_later_frame=True,
    )
    recovered_execution = _execution(
        molecule,
        first_frame_finite_gradients=False,
        include_later_frame=True,
        include_recovery_action=True,
    )

    _, failed = evaluate_route(case, execution)
    _, recovered = evaluate_route(case, recovered_execution)

    assert not failed.decision("A06").passed
    assert failed.decision("A06").diagnostics == ("recovery_missing:gradient:main:1",)
    assert recovered.decision("A06").passed
    assert "recovered_unavailable_gradient:gradient:main:1" in (recovered.diagnostics)


def test_f2_manifest_must_match_the_replayed_build_coordinates() -> None:
    molecule = _complex()
    ligand = Molecule()
    ligand.create_atom(atomic_number=7, id=11)
    execution = replace(
        _execution(molecule),
        route="F2",
        proposal_manifest=(
            {
                "component_sha256": component_sha256(ligand),
                "coordinate_sha256": coordinate_sha256(
                    np.asarray(((9.0, 9.0, 9.0),), dtype=np.float64)
                ),
                "atom_count": 1,
                "ordinal": 0,
            },
        ),
    )

    _, result = evaluate_route(_case(molecule), execution)

    assert not result.decision("A01").passed
    assert "proposal_mapping_changed" in result.decision("A01").diagnostics


def test_selected_trajectory_frame_must_match_the_committed_coordinates() -> None:
    molecule = _complex()
    case = _case(molecule)
    execution = _execution(molecule)
    molecule.coordinates = molecule.coordinates + np.asarray((0.5, 0.0, 0.0))

    _, result = evaluate_route(case, execution)

    assert not result.decision("A03").passed
    assert "selected_frame_committed_coordinates_mismatch" in (
        result.decision("A03").diagnostics
    )


def test_missing_selected_trajectory_frame_cannot_be_synthesized() -> None:
    molecule = _complex()
    execution = _execution(molecule)
    execution.report.trajectory.main._selected_index = None

    _, result = evaluate_route(_case(molecule), execution)

    assert not result.decision("A03").passed
    assert "selected_frame_count:0" in result.decision("A03").diagnostics


def test_missing_provider_component_fails_a01_and_stage_trace() -> None:
    molecule = _complex()
    execution = _execution(molecule, include_ligand_proposal=False)

    _, result = evaluate_route(_case(molecule), execution)

    assert not result.decision("A01").passed
    assert not result.decision("A10").passed
    assert "stage_not_completed:provider:failed" in (result.decision("A10").diagnostics)


def test_missing_observed_coordination_stage_fails_a10() -> None:
    molecule = _complex()
    execution = _execution(molecule, include_coordination_stage=False)

    _, result = evaluate_route(_case(molecule), execution)

    assert not result.decision("A10").passed
    assert "stage_not_completed:coordination_restoration:failed" in (
        result.decision("A10").diagnostics
    )


def test_ordinary_bond_row_and_endpoint_serialization_are_not_semantics() -> None:
    molecule = _complex()
    case = _case(molecule)
    case["graph"]["bonds"][0]["row"] = 41
    case["graph"]["bonds"][0]["begin"] = 1
    case["graph"]["bonds"][0]["end"] = 0
    case["semantic_graph_sha256"] = canonical_digest(
        {"graph": case["graph"], "coordination": case["coordination"]}
    )

    evidence = semantic_evidence(case, molecule)

    assert evidence.equal
    assert evidence.source_digest == evidence.result_digest


def test_semantic_mutation_fails_a09() -> None:
    molecule = _complex()
    case = _case(molecule)
    molecule.atoms[0].formal_charge = 1

    _, result = evaluate_route(case, _execution(molecule))

    assert not result.decision("A09").passed
    assert result.decision("A09").diagnostics == (
        "semantic_digest_changed",
        "graph_semantics_changed",
    )


def test_failed_route_cannot_write_a_selected_structure(tmp_path) -> None:
    molecule = _complex()
    execution = _execution(molecule)
    _, result = evaluate_route(_case(molecule), execution)
    failed = replace(result, passed=False)
    output = tmp_path / "selected.sdf"

    try:
        write_selected_structure(execution, failed, output)
    except ValueError as error:
        assert str(error) == "a failed route has no accepted selected structure"
    else:
        raise AssertionError("failed acceptance wrote a selected structure")
    assert not output.exists()
