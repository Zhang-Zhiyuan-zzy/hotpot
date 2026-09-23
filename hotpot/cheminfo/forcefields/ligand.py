"""Ligand-proxy construction, candidate selection, and refinement."""

from __future__ import annotations

import time
from copy import copy
from dataclasses import dataclass, replace
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from .. import geometry as geo
from .acceptance import (
    _bond_ring_acceptance_checks,
    _format_geometry_checks,
    _has_unreturnable_frame_failure,
    evaluate_structure_acceptance,
)
from .backend import _ob_build, _single_ob_optimization
from .contracts import (
    CandidateRejection,
    ComplexBuildDiagnostics,
    ComplexBuildError,
    ForceFieldError,
    RingUntanglingReport,
)
from .coordinates import _copy_coordinates
from .repair import (
    _piercing_count,
    _scan_confirmed_ring_piercings,
    _untangle_ring_piercings,
)
from .topology import capture_topology
from .trajectory import (
    ForceFieldFrame,
    ForceFieldTrajectory,
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)
from .working_copy import _copy_molecule_metadata


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ()


@dataclass(frozen=True)
class _LigandCandidate:
    coordinates: np.ndarray
    energy: float
    attempt: int
    untangling: RingUntanglingReport
    trajectory: Optional[ForceFieldTrajectory] = None
def _ligand_candidate_sort_key(
    candidate: _LigandCandidate,
) -> Tuple[int, bool, float, int]:
    """Rank usable fallback starts by piercing count and finite energy."""
    finite_energy = bool(np.isfinite(candidate.energy))
    return (
        candidate.untangling.final_piercing_count,
        not finite_energy,
        candidate.energy if finite_energy else float("inf"),
        candidate.attempt,
    )


def _build_ligand_proxies(
    mol: "Molecule",
    *,
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    effective_forcefield: str,
    ligand_untangling_attempts: int = 20,
    perturb_sigma: float = 0.5,
    seed: Optional[int] = None,
    trajectory_attempts: Optional[list[ForceFieldTrajectory]] = None,
) -> Tuple[np.ndarray, ComplexBuildDiagnostics]:
    started = time.monotonic()
    clone_mol = copy(mol)
    _copy_molecule_metadata(mol, clone_mol)
    clone_mol.hide_metal_ligand_bonds(clear_conformers=False)
    rng = np.random.default_rng(seed)
    total_attempts = 0
    total_accepted = 0
    rejections: list[CandidateRejection] = []
    warning_messages: list[str] = []
    selected_untangling_reports: list[RingUntanglingReport] = []

    for component_index, component_mol in enumerate(clone_mol.components):
        if component_mol.has_metal:
            continue

        component_reference = capture_topology(
            component_mol,
            allow_added_hydrogens=False,
        )
        accepted_candidate: Optional[_LigandCandidate] = None
        fallback_candidates: list[_LigandCandidate] = []
        component_attempts = 0
        while accepted_candidate is None and component_attempts < max_attempts:
            component_attempts += 1
            total_attempts += 1
            attempt_trajectory: Optional[ForceFieldTrajectory] = None
            if trajectory_attempts is not None:
                attempt_trajectory = ForceFieldTrajectory.from_molecule(
                    component_mol,
                    start=TrajectoryStart.LIGAND_BUILD,
                )
                trajectory_attempts.append(attempt_trajectory)
                attempt_trajectory.record_molecule(
                    component_mol,
                    stage=TrajectoryStage.LIGAND_BUILD,
                    event=TrajectoryEvent.INITIAL,
                    component_index=component_index,
                    attempt=component_attempts,
                )
            try:
                _ob_build(component_mol)
                if attempt_trajectory is not None:
                    attempt_trajectory.record_molecule(
                        component_mol,
                        stage=TrajectoryStage.LIGAND_BUILD,
                        event=TrajectoryEvent.BUILD_COMPLETE,
                        component_index=component_index,
                        attempt=component_attempts,
                    )
                warmed = _single_ob_optimization(
                    component_mol,
                    effective_forcefield,
                    candidate_warmup_steps,
                )
                if attempt_trajectory is not None:
                    attempt_trajectory.record_molecule(
                        component_mol,
                        stage=TrajectoryStage.LIGAND_BUILD,
                        event=TrajectoryEvent.WARMUP_COMPLETE,
                        energy_kj_mol=float(warmed.energy),
                        component_index=component_index,
                        attempt=component_attempts,
                    )
                untangling = _untangle_ring_piercings(
                    component_mol,
                    effective_forcefield,
                    attempt_limit=ligand_untangling_attempts,
                    short_steps=candidate_warmup_steps,
                    settling_steps=candidate_score_steps,
                    perturb_sigma=perturb_sigma,
                    rng=rng,
                    ring_scope="ligand_skeleton",
                    initial_energy=float(warmed.energy),
                    trajectory=attempt_trajectory,
                    trajectory_stage=TrajectoryStage.LIGAND_BUILD,
                )
            except ForceFieldError as exc:
                if attempt_trajectory is not None:
                    terminal_frame = attempt_trajectory.record_molecule(
                        component_mol,
                        stage=TrajectoryStage.LIGAND_BUILD,
                        event=TrajectoryEvent.TERMINAL,
                        component_index=component_index,
                        attempt=component_attempts,
                    )
                    attempt_trajectory.select(terminal_frame.index)
                rejections.append(
                    CandidateRejection(component_index, component_attempts, str(exc))
                )
                continue

            candidate_quality = evaluate_structure_acceptance(
                component_mol,
                level="basic",
                topology_reference=component_reference,
                forcefield_report={
                    "setup_succeeded": True,
                    "final_energy": untangling.energy,
                    "energy_unit": "kJ/mol",
                    "exploded": False,
                },
                forcefield_stage="candidate",
            )
            candidate = _LigandCandidate(
                coordinates=_copy_coordinates(component_mol.coordinates),
                energy=float(untangling.energy),
                attempt=component_attempts,
                untangling=untangling.report,
                trajectory=attempt_trajectory,
            )
            if not _has_unreturnable_frame_failure(candidate_quality):
                fallback_candidates.append(candidate)
            if not candidate_quality.passed:
                rejections.append(
                    CandidateRejection(
                        component_index,
                        component_attempts,
                        _format_geometry_checks(
                            "candidate geometry gate",
                            tuple(candidate_quality.failures),
                        ),
                        tuple(candidate_quality.failures),
                    )
                )
                continue

            accepted_candidate = candidate
            total_accepted += 1

        if accepted_candidate is None and not fallback_candidates:
            diagnostics = ComplexBuildDiagnostics(
                attempt_count=total_attempts,
                accepted_candidates=total_accepted,
                rejected_candidates=tuple(rejections),
                elapsed_seconds=time.monotonic() - started,
                warning_messages=tuple(warning_messages),
                ligand_untangling=tuple(selected_untangling_reports),
            )
            raise ComplexBuildError(
                f"Component {component_index} produced no usable candidate "
                f"after {component_attempts} attempts",
                diagnostics,
            )

        if accepted_candidate is None:
            selected_candidate = min(
                fallback_candidates,
                key=_ligand_candidate_sort_key,
            )
            component_mol.coordinates = selected_candidate.coordinates
            warning_messages.append(
                f"Component {component_index}: no candidate passed the basic "
                f"geometry gate after {component_attempts} attempts; retaining "
                "the usable attempted geometry with the lowest confirmed "
                "bond-ring piercing count and energy as the next-stage start"
            )
        else:
            selected_candidate = accepted_candidate
            component_mol.coordinates = selected_candidate.coordinates
            try:
                refined = _single_ob_optimization(
                    component_mol,
                    effective_forcefield,
                    best_candidate_refine_steps,
                )
            except ForceFieldError as exc:
                rejections.append(
                    CandidateRejection(
                        component_index,
                        selected_candidate.attempt,
                        f"refined candidate: {exc}",
                    )
                )
                component_mol.coordinates = selected_candidate.coordinates
                if selected_candidate.trajectory is not None:
                    rollback_frame = selected_candidate.trajectory.record_molecule(
                        component_mol,
                        stage=TrajectoryStage.LIGAND_BUILD,
                        event=TrajectoryEvent.ROLLED_BACK,
                        component_index=component_index,
                        attempt=selected_candidate.attempt,
                    )
                    selected_candidate.trajectory.select(rollback_frame.index)
                warning_messages.append(
                    f"Component {component_index}: long refinement failed; "
                    "retaining the medium-optimized candidate"
                )
            else:
                refined_frame: Optional[ForceFieldFrame] = None
                if selected_candidate.trajectory is not None:
                    refined_frame = selected_candidate.trajectory.record_molecule(
                        component_mol,
                        stage=TrajectoryStage.LIGAND_BUILD,
                        event=TrajectoryEvent.OPTIMIZED,
                        energy_kj_mol=float(refined.energy),
                        component_index=component_index,
                        attempt=selected_candidate.attempt,
                    )
                refined_state, refined_report = _scan_confirmed_ring_piercings(
                    component_mol,
                    ring_scope="ligand_skeleton",
                )
                refined_piercing_count = _piercing_count(refined_report)
                refined_quality = evaluate_structure_acceptance(
                    component_mol,
                    level="basic",
                    topology_reference=component_reference,
                    forcefield_report={
                        "setup_succeeded": True,
                        "final_energy": refined.energy,
                        "energy_unit": refined.energy_unit,
                        "exploded": refined.exploded,
                    },
                    forcefield_stage="candidate",
                )
                if not refined_quality.passed:
                    intersection_failures = (
                        _bond_ring_acceptance_checks(
                            component_mol,
                            refined_report,
                        )
                        if refined_report is not None
                        else ()
                    )
                    failures = (
                        tuple(intersection_failures)
                        + tuple(refined_quality.failures)
                    )
                    rejections.append(CandidateRejection(
                        component_index,
                        selected_candidate.attempt,
                        _format_geometry_checks(
                            "refined candidate geometry gate",
                            failures,
                        ),
                        failures,
                    ))
                    component_mol.coordinates = selected_candidate.coordinates
                    if selected_candidate.trajectory is not None:
                        rollback_frame = (
                            selected_candidate.trajectory.record_molecule(
                                component_mol,
                                stage=TrajectoryStage.LIGAND_BUILD,
                                event=TrajectoryEvent.ROLLED_BACK,
                                component_index=component_index,
                                attempt=selected_candidate.attempt,
                            )
                        )
                        selected_candidate.trajectory.select(rollback_frame.index)
                    warning_messages.append(
                        f"Component {component_index}: long refinement failed "
                        "the basic geometry gate; retaining the "
                        "medium-optimized candidate"
                    )
                elif (
                    refined_piercing_count
                    <= selected_candidate.untangling.final_piercing_count
                ):
                    selected_candidate = _LigandCandidate(
                        coordinates=_copy_coordinates(component_mol.coordinates),
                        energy=float(refined.energy),
                        attempt=selected_candidate.attempt,
                        untangling=replace(
                            selected_candidate.untangling,
                            final_piercing_count=refined_piercing_count,
                            minimum_piercing_count=min(
                                selected_candidate.untangling.minimum_piercing_count,
                                refined_piercing_count,
                            ),
                            resolved=(
                                refined_state is not geo.PiercingState.PIERCES
                            ),
                        ),
                        trajectory=selected_candidate.trajectory,
                    )
                    if (
                        refined_frame is not None
                        and selected_candidate.trajectory is not None
                    ):
                        selected_candidate.trajectory.select(refined_frame.index)
                else:
                    component_mol.coordinates = selected_candidate.coordinates
                    if selected_candidate.trajectory is not None:
                        rollback_frame = (
                            selected_candidate.trajectory.record_molecule(
                                component_mol,
                                stage=TrajectoryStage.LIGAND_BUILD,
                                event=TrajectoryEvent.ROLLED_BACK,
                                component_index=component_index,
                                attempt=selected_candidate.attempt,
                            )
                        )
                        selected_candidate.trajectory.select(rollback_frame.index)
                    warning_messages.append(
                        f"Component {component_index}: long refinement increased "
                        "the confirmed piercing count; retaining the "
                        "pre-refinement closed-topology frame"
                    )

        warning_messages.extend(
            f"Component {component_index}: {message}"
            for message in selected_candidate.untangling.warning_messages
        )
        selected_untangling_reports.append(selected_candidate.untangling)
        component_mol.coordinates = selected_candidate.coordinates
        clone_mol.update_atoms_attrs_from_id_dict(
            {
                atom.id: {"coordinates": atom.coordinates}
                for atom in component_mol.atoms
            }
        )

    diagnostics = ComplexBuildDiagnostics(
        attempt_count=total_attempts,
        accepted_candidates=total_accepted,
        rejected_candidates=tuple(rejections),
        elapsed_seconds=time.monotonic() - started,
        warning_messages=tuple(warning_messages),
        ligand_untangling=tuple(selected_untangling_reports),
    )
    return clone_mol.coordinates, diagnostics
