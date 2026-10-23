"""Translate completed Plan-17 routes into immutable acceptance evidence."""

from __future__ import annotations

import hashlib
import json
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from hotpot.cheminfo.forcefields.trajectory import (
    OptimizationFrameEvidence as HotpotOptimizationFrameEvidence,
)
from hotpot.cheminfo.forcefields.working_copy import _copy_molecule_metadata

from .validation_acceptance import (
    AcceptanceEvidence,
    AcceptanceResult,
    CoordinateProposalEvidence,
    NumericEvidence,
    OptimizationFrameEvidence,
    QualityEvidence,
    RecoveryEvidence,
    SemanticEvidence,
    StageRecord,
    StageTraceEvidence,
    UnavailableGradientEvent,
    evaluate_acceptance,
)
from .validation_cohort import _graph_document, canonical_digest
from .validation_providers import coordinate_sha256
from .validation_routes import RouteExecution


RECOVERY_EVENT_ACTIONS = {
    "perturbed": "trajectory_perturbation",
    "rolled_back": "trajectory_rollback",
    "bond_rollback": "coordination_bond_rollback",
}
ESTABLISHED_RECOVERY_ACTIONS = tuple(RECOVERY_EVENT_ACTIONS.values())


@dataclass(frozen=True)
class _ExpectedComponent:
    component_index: int
    atom_identity: tuple[tuple[int, int], ...]
    topology: tuple[tuple[int, int, float, str], ...]
    atom_count: int


@dataclass(frozen=True)
class _ArchiveFact:
    frame_id: str
    branch: str
    branch_sequence: int
    sequence: int
    stage: str
    event: str
    frame: object


def _optimization_report(execution: RouteExecution):
    report = execution.report
    optimization = getattr(report, "optimization", report)
    if optimization is None:
        raise ValueError("route report omitted force-field optimization")
    quality = getattr(report, "quality_report", None)
    if quality is None:
        quality = getattr(optimization, "quality_report", None)
    if quality is None:
        raise ValueError("route report omitted terminal quality evidence")
    archive = getattr(optimization, "trajectory", None)
    if archive is None:
        archive = getattr(report, "trajectory", None)
    if archive is None:
        raise ValueError("route report omitted trajectory evidence")
    return optimization, quality, archive


def _expected_components(molecule: object) -> dict[int, _ExpectedComponent]:
    detached = copy(molecule)
    _copy_molecule_metadata(molecule, detached)
    detached.hide_metal_ligand_bonds(clear_conformers=False)
    expected = {}
    for component_index, component in enumerate(detached.components):
        if component.has_metal:
            continue
        atom_rows = {id(atom): row for row, atom in enumerate(component.atoms)}
        expected[component_index] = _ExpectedComponent(
            component_index=component_index,
            atom_identity=tuple(
                (
                    int(atom.atomic_number),
                    int(atom.formal_charge),
                )
                for atom in component.atoms
            ),
            topology=tuple(
                sorted(
                    (
                        min(atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)]),
                        max(atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)]),
                        float(bond.bond_order),
                        str(bond.bond_kind.value),
                    )
                    for bond in component.bonds
                )
            ),
            atom_count=len(component.atoms),
        )
    return expected


def _manifest_by_component(
    frozen_manifest: Sequence[Mapping[str, object]],
) -> tuple[dict[str, Mapping[str, object]], bool]:
    entries: dict[str, Mapping[str, object]] = {}
    valid = True
    for item in frozen_manifest:
        component = str(item.get("component_sha256", ""))
        coordinate = str(item.get("coordinate_sha256", ""))
        if not component or not coordinate or component in entries:
            valid = False
            continue
        entries[component] = item
    return entries, valid


def _provider_proposal(
    archive: object,
    frozen_manifest: Sequence[Mapping[str, object]],
    molecule: object,
) -> CoordinateProposalEvidence:
    expected = _expected_components(molecule)
    observed: dict[int, tuple[object, object]] = {}
    for trajectory in archive.ligand_build_attempts:
        build_frame = next(
            (
                frame
                for frame in trajectory.frames
                if frame.event.value == "build_complete"
            ),
            None,
        )
        if build_frame is None:
            continue
        component_index = int(build_frame.component_index or 0)
        observed.setdefault(component_index, (trajectory, build_frame))

    mapping_preserved = set(observed) == set(expected)
    coordinates_finite = bool(expected)
    atom_row_count = 0
    proposal_digests: dict[int, str] = {}
    for component_index, (trajectory, frame) in observed.items():
        rows = trajectory.coordinates(frame.index)
        atom_row_count += int(rows.shape[0])
        expected_component = expected.get(component_index)
        observed_identity = tuple(
            (atom.atomic_number, atom.formal_charge)
            for atom in trajectory.atoms
        )
        if expected_component is None:
            mapping_preserved = False
        else:
            mapping_preserved = mapping_preserved and (
                observed_identity == expected_component.atom_identity
            )
            mapping_preserved = mapping_preserved and (
                rows.shape == (expected_component.atom_count, 3)
            )
            observed_topology = tuple(
                (
                    bond.atom_indices[0],
                    bond.atom_indices[1],
                    bond.bond_order,
                    bond.bond_kind,
                )
                for bond in trajectory.topology(frame.index).bonds
            )
            mapping_preserved = mapping_preserved and (
                observed_topology == expected_component.topology
            )
        coordinates_finite = coordinates_finite and bool(
            rows.ndim == 2 and rows.shape[1:] == (3,) and np.all(np.isfinite(rows))
        )
        if rows.ndim == 2 and rows.shape[1:] == (3,):
            proposal_digests[component_index] = coordinate_sha256(rows)

    if frozen_manifest:
        manifest, manifest_valid = _manifest_by_component(frozen_manifest)
        mapping_preserved = mapping_preserved and manifest_valid
        mapping_preserved = mapping_preserved and len(manifest) == len(expected)
        ordered_components = tuple(sorted(expected.items()))
        for (component_index, expected_component), item in zip(
            ordered_components,
            frozen_manifest,
        ):
            mapping_preserved = mapping_preserved and (
                int(item.get("atom_count", -1)) == expected_component.atom_count
            )
            mapping_preserved = mapping_preserved and (
                proposal_digests.get(component_index) == item.get("coordinate_sha256")
            )

    expected_atom_count = sum(component.atom_count for component in expected.values())
    returned = bool(expected) and set(observed) == set(expected)
    return CoordinateProposalEvidence(
        returned=returned,
        coordinate_shape=(atom_row_count, 3) if observed else None,
        atom_row_count=atom_row_count,
        expected_atom_row_count=expected_atom_count,
        mapping_preserved=mapping_preserved,
        coordinates_finite=coordinates_finite,
        status="returned" if returned else "missing",
    )


def _whole_complex_proposal(
    execution: RouteExecution,
) -> CoordinateProposalEvidence:
    proposal = execution.proposal
    if proposal is None:
        return CoordinateProposalEvidence(
            returned=False,
            coordinate_shape=None,
            atom_row_count=0,
            expected_atom_row_count=len(execution.molecule.atoms),
            mapping_preserved=False,
            coordinates_finite=False,
            status="missing",
        )
    atom_count = int(proposal.atom_count)
    return CoordinateProposalEvidence(
        returned=True,
        coordinate_shape=(atom_count, 3),
        atom_row_count=atom_count,
        expected_atom_row_count=len(execution.molecule.atoms),
        mapping_preserved=bool(proposal.mapping_preserved),
        coordinates_finite=True,
        status="returned",
    )


def _project_stereo(stereo: Mapping[str, object], atom_count: int) -> dict[str, object]:
    records = []
    for record in stereo["records"]:
        anchor_rows = tuple(record["anchor"]["atom_rows"])
        carrier_rows = tuple(
            carrier["atom_row"]
            for carrier in record["carriers"]
            if carrier["atom_row"] is not None
        )
        if all(row < atom_count for row in anchor_rows + carrier_rows):
            records.append(record)
    return {
        "schema_version": stereo["schema_version"],
        "records": records,
        "atom_transport": [
            fact
            for fact in stereo["atom_transport"]
            if int(fact["atom_row"]) < atom_count
        ],
        "bond_transport": [
            fact
            for fact in stereo["bond_transport"]
            if all(int(row) < atom_count for row in fact["atom_rows"])
        ],
    }


def _project_original_graph(molecule: object, atom_count: int) -> dict[str, object]:
    graph = _graph_document(molecule)
    return {
        "atoms": graph["atoms"][:atom_count],
        "bonds": [
            bond
            for bond in graph["bonds"]
            if int(bond["begin"]) < atom_count and int(bond["end"]) < atom_count
        ],
        "stereo": _project_stereo(graph["stereo"], atom_count),
    }


def _sort_json_records(records: Sequence[Mapping[str, object]]) -> list[object]:
    return sorted(
        records,
        key=lambda value: json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _semantic_graph(graph: Mapping[str, object]) -> dict[str, object]:
    bonds = []
    for source in graph["bonds"]:
        bond = {key: value for key, value in source.items() if key != "row"}
        if bond["bond_kind"] != "dative":
            bond["begin"], bond["end"] = sorted((int(bond["begin"]), int(bond["end"])))
        bonds.append(bond)
    stereo = graph["stereo"]
    return {
        "atoms": list(graph["atoms"]),
        "bonds": _sort_json_records(bonds),
        "stereo": {
            "schema_version": stereo["schema_version"],
            "records": _sort_json_records(stereo["records"]),
            "atom_transport": _sort_json_records(stereo["atom_transport"]),
            "bond_transport": _sort_json_records(stereo["bond_transport"]),
        },
    }


def semantic_evidence(case: Mapping[str, object], molecule: object) -> SemanticEvidence:
    """Compare business graph semantics while allowing added hydrogen rows."""
    source_graph = case["graph"]
    result_graph = _project_original_graph(molecule, len(source_graph["atoms"]))
    source_semantics = _semantic_graph(source_graph)
    result_semantics = _semantic_graph(result_graph)
    source_document = {
        "graph": source_semantics,
        "coordination": case["coordination"],
    }
    result_document = {
        "graph": result_semantics,
        "coordination": case["coordination"],
    }
    source_digest = canonical_digest(source_document)
    result_digest = canonical_digest(result_document)
    differences = []
    if result_semantics["atoms"] != source_semantics["atoms"]:
        differences.append("atoms")
    if result_semantics["bonds"] != source_semantics["bonds"]:
        differences.append("bonds")
    if result_semantics["stereo"] != source_semantics["stereo"]:
        differences.append("stereo")
    raw_source_digest = canonical_digest(
        {"graph": source_graph, "coordination": case["coordination"]}
    )
    if raw_source_digest != case["semantic_graph_sha256"]:
        differences.append("source_digest")
    return SemanticEvidence(
        equal=not differences,
        source_digest=source_digest,
        result_digest=result_digest,
        differences=tuple(differences),
    )


def _normalized_stage(stage: object) -> str:
    value = stage.value
    if value == "ligand_build":
        return "ligand_build"
    if value == "coordination_restoration":
        return "coordination_restoration"
    return "optimization"


def _archive_facts(archive: object) -> tuple[_ArchiveFact, ...]:
    branches = [
        (f"ligand:{index}", trajectory)
        for index, trajectory in enumerate(archive.ligand_build_attempts)
    ]
    branches.extend(
        (f"preliminary:{index}", trajectory)
        for index, trajectory in enumerate(archive.preliminary_attempts)
    )
    branches.append(("main", archive.main))
    facts = []
    sequence = 0
    for branch, trajectory in branches:
        for branch_sequence, frame in enumerate(trajectory.frames):
            facts.append(
                _ArchiveFact(
                    frame_id=f"{branch}:{frame.index}",
                    branch=branch,
                    branch_sequence=branch_sequence,
                    sequence=sequence,
                    stage=_normalized_stage(frame.stage),
                    event=frame.event.value,
                    frame=frame,
                )
            )
            sequence += 1
    return tuple(facts)


def _numeric_from_flag(
    value: Optional[float], *, finite: bool, diagnostic: str
) -> NumericEvidence:
    if not finite:
        return NumericEvidence.from_value(None, diagnostic=diagnostic)
    return NumericEvidence.from_value(value)


def _same_number(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return False
    return bool(
        np.isfinite(left)
        and np.isfinite(right)
        and np.isclose(left, right, rtol=1.0e-12, atol=1.0e-12)
    )


def _selected_frame(
    fact: _ArchiveFact,
    archive: object,
    optimization: object,
) -> OptimizationFrameEvidence:
    frame = fact.frame
    evidence = frame.evidence
    archived_coordinates = archive.main.coordinates(frame.index)
    evidence_coordinates_finite = (
        True
        if not isinstance(evidence, HotpotOptimizationFrameEvidence)
        else bool(evidence.finite_coordinates)
    )
    coordinates_finite = bool(
        evidence_coordinates_finite and np.all(np.isfinite(archived_coordinates))
    )

    if isinstance(evidence, HotpotOptimizationFrameEvidence):
        energy_consistent = bool(
            evidence.finite_energy
            and _same_number(frame.energy_kj_mol, optimization.best_energy)
        )
        gradients_consistent = bool(
            evidence.finite_gradients
            and _same_number(
                evidence.rms_gradient_kj_mol_angstrom,
                optimization.rms_gradient,
            )
            and _same_number(
                evidence.max_gradient_kj_mol_angstrom,
                optimization.max_gradient,
            )
        )
        exploded = bool(evidence.exploded or optimization.exploded)
    else:
        energy_consistent = _same_number(
            frame.energy_kj_mol,
            optimization.best_energy,
        )
        gradients_consistent = bool(
            np.isfinite(optimization.rms_gradient)
            and np.isfinite(optimization.max_gradient)
        )
        exploded = bool(optimization.exploded)

    return OptimizationFrameEvidence(
        frame_id=fact.frame_id,
        sequence=fact.sequence,
        stage=fact.stage,
        event=fact.event,
        role="selected",
        coordinates_finite=coordinates_finite,
        energy=_numeric_from_flag(
            optimization.best_energy,
            finite=energy_consistent,
            diagnostic="selected_report_trajectory_energy_mismatch",
        ),
        gradients_evaluated=True,
        rms_gradient=_numeric_from_flag(
            optimization.rms_gradient,
            finite=gradients_consistent,
            diagnostic="selected_report_trajectory_gradient_mismatch",
        ),
        max_gradient=_numeric_from_flag(
            optimization.max_gradient,
            finite=gradients_consistent,
            diagnostic="selected_report_trajectory_gradient_mismatch",
        ),
        exploded=exploded,
        optimization_progressed=True,
    )


def _optimization_frames(
    archive: object,
    optimization: object,
    facts: Sequence[_ArchiveFact],
) -> list[OptimizationFrameEvidence]:
    selected_index = archive.main.selected_index
    selected_id = None if selected_index is None else f"main:{selected_index}"
    frames = []
    selected_fact = None
    for fact in facts:
        if fact.frame_id == selected_id:
            selected_fact = fact
            continue
        evidence = fact.frame.evidence
        if not isinstance(evidence, HotpotOptimizationFrameEvidence):
            continue
        frames.append(
            OptimizationFrameEvidence(
                frame_id=fact.frame_id,
                sequence=fact.sequence,
                stage=fact.stage,
                event=fact.event,
                role="exploratory",
                coordinates_finite=bool(evidence.finite_coordinates),
                energy=_numeric_from_flag(
                    fact.frame.energy_kj_mol,
                    finite=bool(evidence.finite_energy),
                    diagnostic="nonfinite_energy",
                ),
                gradients_evaluated=True,
                rms_gradient=_numeric_from_flag(
                    evidence.rms_gradient_kj_mol_angstrom,
                    finite=bool(evidence.finite_gradients),
                    diagnostic="nonfinite_gradients",
                ),
                max_gradient=_numeric_from_flag(
                    evidence.max_gradient_kj_mol_angstrom,
                    finite=bool(evidence.finite_gradients),
                    diagnostic="nonfinite_gradients",
                ),
                exploded=bool(evidence.exploded),
                optimization_progressed=(
                    fact.event in {"optimized", "settled", "epoch_complete", "terminal"}
                ),
            )
        )
    if selected_fact is not None:
        frames.append(_selected_frame(selected_fact, archive, optimization))
    frames.sort(key=lambda frame: frame.sequence)
    return frames


def _gradient_events(
    frames: Sequence[OptimizationFrameEvidence],
    facts: Sequence[_ArchiveFact],
) -> tuple[UnavailableGradientEvent, ...]:
    facts_by_id = {fact.frame_id: fact for fact in facts}
    events = []
    for frame in frames:
        if not frame.gradients_evaluated or (
            frame.rms_gradient.finite and frame.max_gradient.finite
        ):
            continue
        unavailable_fact = facts_by_id[frame.frame_id]
        recovery = None
        for action_fact in facts:
            if (
                action_fact.branch != unavailable_fact.branch
                or action_fact.stage != unavailable_fact.stage
                or action_fact.branch_sequence <= unavailable_fact.branch_sequence
                or action_fact.event not in RECOVERY_EVENT_ACTIONS
            ):
                continue
            later_frames = [
                candidate
                for candidate in frames
                if candidate.frame_id in facts_by_id
                and facts_by_id[candidate.frame_id].branch == unavailable_fact.branch
                and candidate.stage == unavailable_fact.stage
                and facts_by_id[candidate.frame_id].branch_sequence
                > action_fact.branch_sequence
            ]
            finite = next(
                (
                    candidate
                    for candidate in later_frames
                    if candidate.gradients_evaluated
                    and candidate.rms_gradient.finite
                    and candidate.max_gradient.finite
                ),
                None,
            )
            resumed = next(
                (
                    candidate
                    for candidate in later_frames
                    if candidate.optimization_progressed
                    and finite is not None
                    and candidate.sequence >= finite.sequence
                ),
                None,
            )
            if finite is None or resumed is None:
                continue
            recovery = RecoveryEvidence(
                action_name=RECOVERY_EVENT_ACTIONS[action_fact.event],
                attempted=True,
                finite_gradient_frame_id=finite.frame_id,
                optimization_resumed=True,
                resumed_frame_id=resumed.frame_id,
            )
            break
        events.append(
            UnavailableGradientEvent(
                event_id=f"gradient:{frame.frame_id}",
                frame_id=frame.frame_id,
                stage=frame.stage,
                recovery=recovery,
            )
        )
    return tuple(events)


def _stage_records(
    execution: RouteExecution,
    optimization: object,
    quality: object,
    archive: object,
    proposal: CoordinateProposalEvidence,
    facts: Sequence[_ArchiveFact],
    gradient_events: Sequence[UnavailableGradientEvent],
) -> tuple[StageRecord, ...]:
    fact_stages = {fact.stage for fact in facts}
    fact_events = {fact.event for fact in facts}
    selected_present = archive.main.selected_frame is not None
    recovery_actions_by_stage: dict[str, list[str]] = {}
    for event in gradient_events:
        if event.recovery is not None:
            recovery_actions_by_stage.setdefault(event.stage, []).append(
                event.recovery.action_name
            )

    if execution.route in {"O0", "F1", "F2"}:
        build = getattr(execution.report, "build", None)
        completed = {
            "provider": proposal.returned and proposal.mapping_preserved,
            "ligand_build": (
                bool(archive.ligand_build_attempts)
                and proposal.returned
                and "ligand_build" in fact_stages
            ),
            "placement": build is not None and bool(archive.main.frames),
            "coordination_restoration": "coordination_restoration" in fact_stages,
            "optimization": bool(optimization.setup_succeeded and selected_present),
            "acceptance": bool(quality.checks),
        }
    else:
        completed = {
            "whole_complex_provider": proposal.returned and proposal.mapping_preserved,
            "complex_entry": bool(
                execution.proposal is not None and archive.main.frames
            ),
            "topology_checkpoint": "topology_checkpoint" in fact_events,
            "optimization": bool(optimization.setup_succeeded and selected_present),
            "acceptance": bool(quality.checks),
        }

    return tuple(
        StageRecord(
            name=stage,
            status="completed" if completed.get(stage, False) else "failed",
            recovery_actions=tuple(recovery_actions_by_stage.get(stage, ())),
        )
        for stage in execution.stage_sequence
    )


def build_acceptance_evidence(
    case: Mapping[str, object], execution: RouteExecution
) -> AcceptanceEvidence:
    """Translate one completed route using only archived production facts."""
    optimization, quality, archive = _optimization_report(execution)
    proposal = (
        _provider_proposal(
            archive,
            execution.proposal_manifest,
            execution.molecule,
        )
        if execution.route in {"O0", "F1", "F2"}
        else _whole_complex_proposal(execution)
    )
    facts = _archive_facts(archive)
    frames = _optimization_frames(
        archive,
        optimization,
        facts,
    )
    gradient_events = _gradient_events(frames, facts)
    stage_records = _stage_records(
        execution,
        optimization,
        quality,
        archive,
        proposal,
        facts,
        gradient_events,
    )
    quality_checks = tuple(check.name for check in quality.checks)
    selected_frame = archive.main.selected_frame
    selected_frame_consistent = bool(
        selected_frame is not None
        and archive.main.coordinates(selected_frame.index).shape
        == execution.molecule.coordinates.shape
        and np.array_equal(
            archive.main.coordinates(selected_frame.index),
            execution.molecule.coordinates,
        )
    )
    return AcceptanceEvidence(
        case_id=f"extractant-{int(case['index']):03d}",
        route=execution.route,
        proposal=proposal,
        setup_succeeded=bool(optimization.setup_succeeded),
        setup_exception=None,
        frames=tuple(frames),
        committed_coordinates_finite=bool(
            np.all(np.isfinite(execution.molecule.coordinates))
        ),
        unavailable_gradient_events=gradient_events,
        established_recovery_actions=ESTABLISHED_RECOVERY_ACTIONS,
        quality=QualityEvidence(
            level=quality.level,
            passed=bool(quality.passed),
            check_names=quality_checks,
            failure_names=tuple(check.name for check in quality.failures),
            warning_names=tuple(check.name for check in quality.warnings),
        ),
        semantics=semantic_evidence(case, execution.molecule),
        stage_trace=StageTraceEvidence(
            expected_stages=execution.stage_sequence,
            records=stage_records,
            provider_fallback_used=False,
        ),
        terminal_reason=optimization.termination_reason,
        converged=bool(optimization.converged),
        selected_frame_consistent=selected_frame_consistent,
    )


def evaluate_route(
    case: Mapping[str, object], execution: RouteExecution
) -> tuple[AcceptanceEvidence, AcceptanceResult]:
    evidence = build_acceptance_evidence(case, execution)
    return evidence, evaluate_acceptance(evidence)


def write_selected_structure(
    execution: RouteExecution,
    acceptance: AcceptanceResult,
    path: Path,
) -> dict[str, object]:
    """Write only an automatically accepted selected force-field structure."""
    if not acceptance.passed:
        raise ValueError("a failed route has no accepted selected structure")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    execution.molecule.write(path, overwrite=True, write_single=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "format": path.suffix.lstrip(".").lower(),
        "atom_row_mapping": list(range(len(execution.molecule.atoms))),
    }


__all__ = (
    "ESTABLISHED_RECOVERY_ACTIONS",
    "build_acceptance_evidence",
    "evaluate_route",
    "semantic_evidence",
    "write_selected_structure",
)
