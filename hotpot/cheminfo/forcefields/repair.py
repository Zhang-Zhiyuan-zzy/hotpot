"""Ring-piercing repair and incremental coordination-bond restoration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

from .. import geometry as geo
from .backend import _single_ob_optimization
from .coordination import _relocate_unbound_metal
from .contracts import CoordinationBondRestorationReport, RingUntanglingReport
from .coordinates import _copy_coordinates, _perturbed_coordinates
from .settings import _BOND_RING_MAX_SIZE
from .topology import _bond_key
from .trajectory import (
    CoordinationFrameEvidence,
    ForceFieldTrajectory,
    RingFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
)


if TYPE_CHECKING:
    from ..core import Atom, Bond, Molecule, Ring


__all__ = ()


@dataclass(frozen=True)
class _RingUntanglingResult:
    report: RingUntanglingReport
    energy: float
    checkpoint_report: "geo.BondRingScreeningReport[Ring, Bond]"


@dataclass(frozen=True)
class _CoordinationRestorationResult:
    report: CoordinationBondRestorationReport


@dataclass(frozen=True)
class _CoordinationRelationCounts:
    piercing: int
    undetermined: int
    excluded_rings: int


@dataclass(frozen=True)
class _CoordinationTrialStatistics:
    rejected_piercing_trial_count: int
    undetermined_trial_count: int
    excluded_ring_observation_count: int
    piercing_bond_keys: Tuple[Tuple[int, int], ...] = ()


@dataclass(frozen=True)
class _BlockedMetalCenter:
    metal: "Atom"
    pending_bonds: Tuple["Bond", ...]


@dataclass(frozen=True, order=True)
class _BondRingPairKey:
    ring_atom_indices: Tuple[int, ...]
    bond_atom_indices: Tuple[int, int]


@dataclass(frozen=True)
class _WatchedRingPiercing:
    key: _BondRingPairKey
    opening_edge_keys: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class _RingPiercingWatchResult:
    state: geo.PiercingState
    piercings: Tuple[_WatchedRingPiercing, ...]


def _piercing_count(
    report: Optional[Union[
        "geo.BondRingScanReport[Ring, Bond]",
        "geo.BondRingScreeningReport[Ring, Bond]",
    ]],
) -> int:
    """Return the confirmed piercing count of an optional geometry scan."""
    return 0 if report is None else len(report.piercings)


def _unique_messages(messages: Sequence[str]) -> Tuple[str, ...]:
    """Deduplicate messages while preserving their first-seen order."""
    return tuple(dict.fromkeys(messages))


def _select_ring_opening_edge(
    mol: "Molecule",
    piercings: Sequence[_WatchedRingPiercing],
) -> Optional["Bond"]:
    """Choose the nearest eligible edge for the first repairable piercing."""
    bonds_by_key = {_bond_key(bond): bond for bond in mol.bonds}
    for piercing in piercings:
        target_bond = bonds_by_key.get(piercing.key.bond_atom_indices)
        eligible_edges = tuple(
            bonds_by_key[key]
            for key in piercing.opening_edge_keys
            if key in bonds_by_key
        )
        if target_bond is None or not eligible_edges:
            continue
        target_segment = geo.segment_from_bond(target_bond)

        def edge_distance(edge: "Bond") -> Tuple[float, Tuple[int, int]]:
            return (
                geo.segment_segment_distance(
                    geo.segment_from_bond(edge),
                    target_segment,
                ),
                _bond_key(edge),
            )

        return min(eligible_edges, key=edge_distance)
    return None


def _ring_edge_memberships(
    mol: "Molecule",
    ring_scope: geo.RingScope,
) -> dict[Tuple[int, int], int]:
    """Count edge memberships across every ring in the selected scope."""
    memberships: dict[Tuple[int, int], int] = {}
    for ring in mol.rings_for_scope(ring_scope):
        for edge in ring.bonds:
            key = _bond_key(edge)
            memberships[key] = memberships.get(key, 0) + 1
    return memberships


def _ring_piercing_watch(
    mol: "Molecule",
    report: "geo.BondRingScreeningReport[Ring, Bond]",
) -> Tuple[_WatchedRingPiercing, ...]:
    """Freeze stable graph keys for the currently confirmed piercings."""
    memberships = _ring_edge_memberships(mol, report.ring_scope)
    watched = []
    seen = set()
    for finding in report.piercings:
        key = _BondRingPairKey(
            finding.target.ring.key,
            finding.target.bond.key,
        )
        if key in seen:
            continue
        seen.add(key)
        ring = finding.target.ring.ring
        contains_metal = any(atom.is_metal for atom in ring.atoms)
        if contains_metal:
            eligible_edges = (
                edge
                for edge in ring.bonds
                if edge.is_metal_ligand_bond
            )
        else:
            eligible_edges = (
                edge
                for edge in ring.bonds
                if float(edge.bond_order) == 1.0
                and memberships.get(_bond_key(edge), 0) == 1
            )
        opening_edge_keys = tuple(sorted(
            _bond_key(edge)
            for edge in eligible_edges
        ))
        watched.append(_WatchedRingPiercing(key, opening_edge_keys))
    return tuple(watched)


def _scan_ring_piercing_watch(
    mol: "Molecule",
    watch: Sequence[_WatchedRingPiercing],
) -> Optional[_RingPiercingWatchResult]:
    """Recheck only stable ring--bond pairs from the current watch batch."""
    atoms_by_index = {int(atom.idx): atom for atom in mol.atoms}
    bonds_by_key = {_bond_key(bond): bond for bond in mol.bonds}
    watched_by_ring: dict[Tuple[int, ...], list[_WatchedRingPiercing]] = {}
    for piercing in watch:
        missing_ring_atom = any(
            index not in atoms_by_index
            for index in piercing.key.ring_atom_indices
        )
        if (
            missing_ring_atom
            or piercing.key.bond_atom_indices not in bonds_by_key
        ):
            return None
        watched_by_ring.setdefault(
            piercing.key.ring_atom_indices,
            [],
        ).append(piercing)

    aggregate = geo.PiercingState.DOES_NOT_PIERCE
    confirmed = []
    for ring_atom_indices, ring_watch in watched_by_ring.items():
        cycle = geo.Cycle(tuple(
            geo.point_from_atom(atoms_by_index[index])
            for index in ring_atom_indices
        ))
        segments = tuple(
            geo.segment_from_bond(
                bonds_by_key[item.key.bond_atom_indices]
            )
            for item in ring_watch
        )
        for item, screening in zip(
            ring_watch,
            geo.iter_segment_cycle_screenings(segments, cycle),
        ):
            if screening.state is geo.PiercingState.PIERCES:
                aggregate = geo.PiercingState.PIERCES
                confirmed.append(item)
            elif (
                screening.state is geo.PiercingState.UNDETERMINED
                and aggregate is geo.PiercingState.DOES_NOT_PIERCE
            ):
                aggregate = geo.PiercingState.UNDETERMINED
    return _RingPiercingWatchResult(aggregate, tuple(confirmed))


def _scan_ring_checkpoint(
    mol: "Molecule",
    *,
    ring_scope: geo.RingScope,
) -> "geo.BondRingScreeningReport[Ring, Bond]":
    """Return one complete full-scope bond--ring checkpoint report."""
    return geo.screen_bond_ring_relations(
        mol,
        ring_scope=ring_scope,
        max_ring_size=_BOND_RING_MAX_SIZE,
    )


def _ring_frame_evidence(
    state: geo.PiercingState,
    report: Optional[Union[
        "geo.BondRingScanReport[Ring, Bond]",
        "geo.BondRingScreeningReport[Ring, Bond]",
    ]],
    *,
    confirmed_piercing_count: Optional[int] = None,
) -> RingFrameEvidence:
    """Describe one observed ring state without triggering another scan."""
    if report is None:
        uncertain_relation_count = (
            None if state is geo.PiercingState.UNDETERMINED else 0
        )
    else:
        uncertain_relation_count = len(report.undetermined)
    return RingFrameEvidence(
        confirmed_piercing_count=(
            _piercing_count(report)
            if confirmed_piercing_count is None
            else confirmed_piercing_count
        ),
        uncertain_relation_count=uncertain_relation_count,
    )


def _untangle_ring_piercings(
    mol: "Molecule",
    effective_forcefield: str,
    *,
    attempt_limit: int,
    short_steps: int,
    settling_steps: int,
    perturb_sigma: float,
    rng: np.random.Generator,
    checkpoint_report: "geo.BondRingScreeningReport[Ring, Bond]",
    initial_energy: float = float("nan"),
    trajectory: Optional[ForceFieldTrajectory] = None,
    trajectory_stage: TrajectoryStage = TrajectoryStage.COMPLEX_UNTANGLING,
) -> _RingUntanglingResult:
    """Repair confirmed ring piercing without rebuilding the molecular graph.

    The caller supplies the entry checkpoint.  This routine builds its initial
    watch from that exact report and performs no duplicate entry scan.  Full
    checkpoints are recomputed only when the watch clears or becomes invalid,
    after settling, and when the attempt budget requires closed-topology
    selection.

    One covalent ring edge is opened per attempt.  The open structure is
    perturbed and relaxed, then the exact bond object is restored before the
    next geometric observation.  A shared trajectory records both the open
    and closed topology revisions without taking over workflow control.
    """
    records_trajectory = (
        trajectory is not None
        and trajectory.records(trajectory_stage)
    )
    ring_scope = checkpoint_report.ring_scope

    def record_ring_frame(
        event: TrajectoryEvent,
        *,
        energy: Optional[float] = None,
        state: Optional[geo.PiercingState] = None,
        report: Optional[Union[
            "geo.BondRingScanReport[Ring, Bond]",
            "geo.BondRingScreeningReport[Ring, Bond]",
        ]] = None,
        confirmed_piercing_count: Optional[int] = None,
        attempt: Optional[int] = None,
    ) -> Optional[int]:
        if not records_trajectory or trajectory is None:
            return None
        frame = trajectory.record_molecule(
            mol,
            stage=trajectory_stage,
            event=event,
            energy_kj_mol=energy,
            attempt=attempt,
            evidence=(
                None
                if state is None
                else _ring_frame_evidence(
                    state,
                    report,
                    confirmed_piercing_count=confirmed_piercing_count,
                )
            ),
        )
        return frame.index

    report = checkpoint_report
    state = report.state
    initial_count = _piercing_count(report)
    current_count = initial_count
    minimum_count = initial_count
    best_coordinates = _copy_coordinates(mol.coordinates)
    best_state = state
    best_report = report
    best_energy = float(initial_energy)
    best_trace_energy = (
        best_energy if np.isfinite(best_energy) else None
    )
    warning_messages = []
    attempts_completed = 0
    settled = False
    unresolved_reason: Optional[str] = None
    watch = (
        _ring_piercing_watch(mol, report)
        if state is geo.PiercingState.PIERCES
        else ()
    )
    current_piercings = watch
    watch_best_coordinates = _copy_coordinates(mol.coordinates)
    watch_best_energy = best_energy
    watch_minimum_count = len(watch)
    record_ring_frame(
        TrajectoryEvent.INITIAL,
        energy=best_trace_energy,
        state=state,
        report=report,
        attempt=0,
    )

    while True:
        if state is not geo.PiercingState.PIERCES:
            if state is geo.PiercingState.UNDETERMINED:
                warning_messages.append(
                    "A bond-ring relation remained mathematically undetermined"
                )
            if settled or settling_steps == 0:
                break
            optimized = _single_ob_optimization(
                mol,
                effective_forcefield,
                settling_steps,
            )
            settled = True
            report = _scan_ring_checkpoint(
                mol,
                ring_scope=ring_scope,
            )
            state = report.state
            current_count = _piercing_count(report)
            watch = (
                _ring_piercing_watch(mol, report)
                if state is geo.PiercingState.PIERCES
                else ()
            )
            current_piercings = watch
            watch_best_coordinates = _copy_coordinates(mol.coordinates)
            watch_best_energy = float(optimized.energy)
            watch_minimum_count = len(watch)
            if current_count <= minimum_count:
                minimum_count = current_count
                best_coordinates = _copy_coordinates(mol.coordinates)
                best_state = state
                best_report = report
                best_energy = float(optimized.energy)
                best_trace_energy = float(optimized.energy)
            record_ring_frame(
                TrajectoryEvent.SETTLED,
                energy=float(optimized.energy),
                state=state,
                report=report,
                attempt=attempts_completed,
            )
            continue

        if attempts_completed >= attempt_limit:
            unresolved_reason = (
                f"Confirmed bond-ring piercing remains after {attempt_limit} "
                "untangling attempts; retaining the closed-topology frame with "
                "the lowest piercing count"
            )
            break

        ring_edge = _select_ring_opening_edge(mol, current_piercings)
        if ring_edge is None:
            unresolved_reason = (
                "Confirmed bond-ring piercing has no eligible ring-opening edge; "
                "retaining the closed-topology frame with the lowest piercing count"
            )
            break

        attempts_completed += 1
        mol.hide_bonds(ring_edge, clear_conformers=False)
        record_ring_frame(
            TrajectoryEvent.RING_OPENED,
            attempt=attempts_completed,
        )
        optimized_successfully = False
        try:
            mol.coordinates = _perturbed_coordinates(
                mol.coordinates,
                sigma=perturb_sigma,
                rng=rng,
            )
            record_ring_frame(
                TrajectoryEvent.PERTURBED,
                attempt=attempts_completed,
            )
            optimized = _single_ob_optimization(
                mol,
                effective_forcefield,
                short_steps,
            )
            optimized_successfully = True
            record_ring_frame(
                TrajectoryEvent.OPTIMIZED,
                energy=float(optimized.energy),
                attempt=attempts_completed,
            )
        finally:
            mol.restore_bonds(ring_edge, clear_conformers=False)
            if not optimized_successfully:
                record_ring_frame(
                    TrajectoryEvent.RING_CLOSED,
                    attempt=attempts_completed,
                )

        watched_result = _scan_ring_piercing_watch(mol, watch)
        if (
            watched_result is None
            or watched_result.state is not geo.PiercingState.PIERCES
        ):
            report = _scan_ring_checkpoint(
                mol,
                ring_scope=ring_scope,
            )
            state = report.state
            current_count = _piercing_count(report)
            watch = (
                _ring_piercing_watch(mol, report)
                if state is geo.PiercingState.PIERCES
                else ()
            )
            current_piercings = watch
            watch_best_coordinates = _copy_coordinates(mol.coordinates)
            watch_best_energy = float("nan")
            watch_minimum_count = len(watch)
            record_ring_frame(
                TrajectoryEvent.RING_CLOSED,
                state=state,
                report=report,
                attempt=attempts_completed,
            )
            if current_count <= minimum_count:
                minimum_count = current_count
                best_coordinates = _copy_coordinates(mol.coordinates)
                best_state = state
                best_report = report
                best_energy = float("nan")
                best_trace_energy = None
        else:
            state = geo.PiercingState.PIERCES
            report = None
            current_piercings = watched_result.piercings
            if len(current_piercings) <= watch_minimum_count:
                watch_minimum_count = len(current_piercings)
                watch_best_coordinates = _copy_coordinates(mol.coordinates)
                watch_best_energy = float("nan")
            record_ring_frame(
                TrajectoryEvent.RING_CLOSED,
                attempt=attempts_completed,
            )
        settled = False

    if unresolved_reason is not None:
        mol.coordinates = watch_best_coordinates
        candidate_report = (
            checkpoint_report
            if attempts_completed == 0
            else _scan_ring_checkpoint(
                mol,
                ring_scope=ring_scope,
            )
        )
        candidate_state = candidate_report.state
        candidate_count = _piercing_count(candidate_report)
        if candidate_count <= minimum_count:
            current_count = candidate_count
            minimum_count = candidate_count
            state = candidate_state
            report = candidate_report
            best_coordinates = _copy_coordinates(mol.coordinates)
            best_state = state
            best_report = report
            best_energy = watch_best_energy
        else:
            mol.coordinates = best_coordinates
            current_count = minimum_count
            state = best_state
            report = best_report
        best_trace_energy = (
            best_energy if np.isfinite(best_energy) else None
        )
        record_ring_frame(
            TrajectoryEvent.ROLLED_BACK,
            energy=best_trace_energy,
            state=state,
            report=report,
            confirmed_piercing_count=current_count,
            attempt=attempts_completed,
        )
        if settling_steps:
            retained_coordinates = best_coordinates.copy()
            retained_energy = best_energy
            retained_trace_energy = best_trace_energy
            retained_count = minimum_count
            retained_state = state
            retained_report = report
            optimized = _single_ob_optimization(
                mol,
                effective_forcefield,
                settling_steps,
            )
            settled_report = _scan_ring_checkpoint(
                mol,
                ring_scope=ring_scope,
            )
            settled_state = settled_report.state
            settled_count = _piercing_count(settled_report)
            record_ring_frame(
                TrajectoryEvent.SETTLED,
                energy=float(optimized.energy),
                state=settled_state,
                report=settled_report,
                attempt=attempts_completed,
            )
            if settled_count <= retained_count:
                state = settled_state
                report = settled_report
                current_count = settled_count
                minimum_count = settled_count
                best_coordinates = _copy_coordinates(mol.coordinates)
                best_energy = float(optimized.energy)
                best_trace_energy = float(optimized.energy)
            else:
                mol.coordinates = retained_coordinates
                state = retained_state
                report = retained_report
                current_count = retained_count
                best_energy = retained_energy
                best_trace_energy = retained_trace_energy
                record_ring_frame(
                    TrajectoryEvent.ROLLED_BACK,
                    energy=best_trace_energy,
                    state=state,
                    confirmed_piercing_count=retained_count,
                    attempt=attempts_completed,
                )
        if state is geo.PiercingState.PIERCES:
            warning_messages.append(unresolved_reason)
        elif state is geo.PiercingState.UNDETERMINED:
            warning_messages.append(
                "A bond-ring relation remained mathematically undetermined"
            )
    resolved = current_count == 0
    terminal_index = record_ring_frame(
        TrajectoryEvent.TERMINAL,
        energy=best_trace_energy,
        state=state,
        report=report,
        confirmed_piercing_count=current_count,
        attempt=attempts_completed,
    )
    if terminal_index is not None and trajectory is not None:
        trajectory.select(terminal_index)

    return _RingUntanglingResult(
        report=RingUntanglingReport(
            attempt_limit=attempt_limit,
            attempts_completed=attempts_completed,
            initial_piercing_count=initial_count,
            final_piercing_count=current_count,
            minimum_piercing_count=minimum_count,
            resolved=resolved,
            warning_messages=_unique_messages(warning_messages),
        ),
        energy=best_energy,
        checkpoint_report=report,
    )


def _screen_coordination_bond_relations(
    mol: "Molecule",
    coordination_bond: "Bond",
) -> "geo.BondRingScreeningReport[Ring, Bond]":
    """Screen one hidden coordination candidate against the pre-addition graph."""
    return geo.screen_bonds_against_rings(
        mol,
        (coordination_bond,),
        ring_scope="full_graph",
        max_ring_size=_BOND_RING_MAX_SIZE,
    )


def _candidate_coordination_relation_counts(
    report: "geo.BondRingScreeningReport[Ring, Bond]",
    coordination_bond: "Bond",
) -> _CoordinationRelationCounts:
    """Count actionable relations for one proposed coordination bond."""
    candidate_key = _bond_key(coordination_bond)
    piercing_count = 0
    undetermined_count = 0
    for finding in report.actionable_findings:
        if finding.target.bond.key != candidate_key:
            continue
        if finding.relation.state is geo.PiercingState.PIERCES:
            piercing_count += 1
        elif finding.relation.state is geo.PiercingState.UNDETERMINED:
            undetermined_count += 1
    return _CoordinationRelationCounts(
        piercing=piercing_count,
        undetermined=undetermined_count,
        excluded_rings=report.excluded_ring_count,
    )


def _coordination_metal(coordination_bond: "Bond") -> "Atom":
    """Return the metal endpoint of one metal--ligand bond."""
    return (
        coordination_bond.atom1
        if coordination_bond.atom1.is_metal
        else coordination_bond.atom2
    )


def _active_coordination_metal_indices(mol: "Molecule") -> set[int]:
    """Return metal centers already anchored by an active coordination bond."""
    return {
        _coordination_metal(bond).idx
        for bond in mol.bonds
        if bond.is_metal_ligand_bond
    }


def _blocked_unbound_metal_centers(
    mol: "Molecule",
    pending_bonds: Sequence["Bond"],
    piercing_bond_keys: Sequence[Tuple[int, int]],
    attempted_metal_indices: set[int],
) -> Tuple[_BlockedMetalCenter, ...]:
    """Group fully blocked pending paths of not-yet-anchored metal centers."""
    piercing_keys = set(piercing_bond_keys)
    pending_by_metal: dict[int, list["Bond"]] = {}
    metals_by_index: dict[int, "Atom"] = {}
    for bond in pending_bonds:
        metal = _coordination_metal(bond)
        metals_by_index[metal.idx] = metal
        pending_by_metal.setdefault(metal.idx, []).append(bond)

    active_metal_indices = _active_coordination_metal_indices(mol)
    return tuple(
        _BlockedMetalCenter(
            metals_by_index[metal_idx],
            tuple(sorted(bonds, key=_bond_key)),
        )
        for metal_idx, bonds in sorted(pending_by_metal.items())
        if metal_idx not in active_metal_indices
        and metal_idx not in attempted_metal_indices
        and all(_bond_key(bond) in piercing_keys for bond in bonds)
    )


def _restore_next_nonpiercing_coordination_bond(
    mol: "Molecule",
    pending_bonds: list["Bond"],
    *,
    trajectory: Optional[ForceFieldTrajectory] = None,
    attempt: Optional[int] = None,
) -> Tuple[Optional["Bond"], Tuple[str, ...], _CoordinationTrialStatistics]:
    """Restore the first hidden bond whose hypothetical segment does not pierce."""
    records_trajectory = (
        trajectory is not None
        and trajectory.records(TrajectoryStage.COORDINATION_RESTORATION)
    )

    def record_candidate(
        event: TrajectoryEvent,
        bond: "Bond",
        *,
        accepted: Optional[bool],
        pending_count: int,
        relation_counts: Optional[_CoordinationRelationCounts] = None,
    ) -> None:
        if not records_trajectory or trajectory is None:
            return
        trajectory.record_molecule(
            mol,
            stage=TrajectoryStage.COORDINATION_RESTORATION,
            event=event,
            attempt=attempt,
            evidence=CoordinationFrameEvidence(
                bond_atom_indices=_bond_key(bond),
                accepted=accepted,
                pending_bond_count=pending_count,
                piercing_relation_count=(
                    0 if relation_counts is None else relation_counts.piercing
                ),
                undetermined_relation_count=(
                    0 if relation_counts is None else relation_counts.undetermined
                ),
                excluded_ring_count=(
                    0 if relation_counts is None else relation_counts.excluded_rings
                ),
            ),
        )

    warning_messages = []
    rejected_piercing_trials = 0
    undetermined_trials = 0
    excluded_ring_observations = 0
    piercing_bond_keys = []
    for bond in tuple(pending_bonds):
        record_candidate(
            TrajectoryEvent.BOND_TRIAL,
            bond,
            accepted=None,
            pending_count=len(pending_bonds),
        )
        relation_counts = _candidate_coordination_relation_counts(
            _screen_coordination_bond_relations(mol, bond),
            bond,
        )
        rejected_piercing_trials += int(relation_counts.piercing > 0)
        if relation_counts.piercing:
            piercing_bond_keys.append(_bond_key(bond))
        undetermined_trials += int(relation_counts.undetermined > 0)
        excluded_ring_observations += relation_counts.excluded_rings
        keep_restored = relation_counts.piercing == 0
        if keep_restored:
            mol.restore_bonds(bond, clear_conformers=False)
        record_candidate(
            (
                TrajectoryEvent.BOND_ACCEPTED
                if keep_restored
                else TrajectoryEvent.BOND_REJECTED
            ),
            bond,
            accepted=keep_restored,
            pending_count=(
                len(pending_bonds) - 1
                if keep_restored
                else len(pending_bonds)
            ),
            relation_counts=relation_counts,
        )
        if relation_counts.undetermined:
            warning_messages.append(
                f"Hypothetical coordination bond {_bond_key(bond)} has "
                f"{relation_counts.undetermined} mathematically undetermined "
                "ring relation(s)"
            )
        if relation_counts.excluded_rings:
            warning_messages.append(
                f"Coordination bond {_bond_key(bond)} was checked while "
                f"{relation_counts.excluded_rings} ring(s) larger than "
                f"{_BOND_RING_MAX_SIZE} atoms were excluded"
            )
        if relation_counts.piercing:
            continue
        pending_bonds.remove(bond)
        return (
            bond,
            tuple(warning_messages),
            _CoordinationTrialStatistics(
                rejected_piercing_trial_count=rejected_piercing_trials,
                undetermined_trial_count=undetermined_trials,
                excluded_ring_observation_count=excluded_ring_observations,
                piercing_bond_keys=tuple(piercing_bond_keys),
            ),
        )
    return (
        None,
        tuple(warning_messages),
        _CoordinationTrialStatistics(
            rejected_piercing_trial_count=rejected_piercing_trials,
            undetermined_trial_count=undetermined_trials,
            excluded_ring_observation_count=excluded_ring_observations,
            piercing_bond_keys=tuple(piercing_bond_keys),
        ),
    )


def _restore_coordination_bonds_incrementally(
    mol: "Molecule",
    effective_forcefield: str,
    *,
    attempt_limit: int,
    relaxation_steps: int,
    perturb_sigma: float,
    rng: np.random.Generator,
    trajectory: Optional[ForceFieldTrajectory] = None,
) -> _CoordinationRestorationResult:
    """Restore original metal--ligand bonds through explicit bounded rounds."""
    records_trajectory = (
        trajectory is not None
        and trajectory.records(TrajectoryStage.COORDINATION_RESTORATION)
    )

    def record_restoration_frame(
        event: TrajectoryEvent,
        *,
        energy: Optional[float] = None,
        evidence: Optional[CoordinationFrameEvidence] = None,
        attempt: Optional[int] = None,
    ) -> Optional[int]:
        if not records_trajectory or trajectory is None:
            return None
        frame = trajectory.record_molecule(
            mol,
            stage=TrajectoryStage.COORDINATION_RESTORATION,
            event=event,
            energy_kj_mol=energy,
            attempt=attempt,
            evidence=evidence,
        )
        return frame.index

    coordination_bonds = tuple(sorted(
        (bond for bond in mol.bonds if bond.is_metal_ligand_bond),
        key=_bond_key,
    ))
    if not coordination_bonds:
        record_restoration_frame(TrajectoryEvent.COORDINATION_READY)
        terminal_index = record_restoration_frame(
            TrajectoryEvent.TERMINAL,
            evidence=CoordinationFrameEvidence(
                bond_atom_indices=None,
                accepted=True,
            ),
        )
        if terminal_index is not None and trajectory is not None:
            trajectory.select(terminal_index)
        return _CoordinationRestorationResult(
            report=CoordinationBondRestorationReport(
                attempt_limit=attempt_limit,
                attempts_completed=0,
                bond_count=0,
                metal_relocation_attempt_count=0,
                relocated_metal_indices=(),
                infeasible_metal_indices=(),
                forced_bond_keys=(),
                rejected_piercing_trial_count=0,
                undetermined_trial_count=0,
                excluded_ring_observation_count=0,
            ),
        )

    mol.hide_bonds(*coordination_bonds, clear_conformers=False)
    pending_bonds = list(coordination_bonds)
    record_restoration_frame(
        TrajectoryEvent.COORDINATION_READY,
        evidence=CoordinationFrameEvidence(
            bond_atom_indices=None,
            accepted=True,
            pending_bond_count=len(pending_bonds),
        ),
    )
    warning_messages: list[str] = []
    stalled_attempts = 0
    last_energy: Optional[float] = None
    rejected_piercing_trial_count = 0
    undetermined_trial_count = 0
    excluded_ring_observation_count = 0
    metal_relocation_attempt_count = 0
    relocation_attempted_metal_indices: set[int] = set()
    relocated_metal_indices: set[int] = set()
    infeasible_metal_indices: set[int] = set()

    while pending_bonds:
        restored_bond, relation_warnings, relation_observations = (
            _restore_next_nonpiercing_coordination_bond(
                mol,
                pending_bonds,
                trajectory=trajectory,
                attempt=stalled_attempts,
            )
        )
        warning_messages.extend(relation_warnings)
        rejected_piercing_trial_count += (
            relation_observations.rejected_piercing_trial_count
        )
        undetermined_trial_count += relation_observations.undetermined_trial_count
        excluded_ring_observation_count += (
            relation_observations.excluded_ring_observation_count
        )
        if restored_bond is None:
            blocked_centers = _blocked_unbound_metal_centers(
                mol,
                pending_bonds,
                relation_observations.piercing_bond_keys,
                relocation_attempted_metal_indices,
            )
            if blocked_centers:
                center = blocked_centers[0]
                record_restoration_frame(
                    TrajectoryEvent.METAL_RELOCATION_TRIAL,
                    evidence=CoordinationFrameEvidence(
                        bond_atom_indices=None,
                        accepted=None,
                        pending_bond_count=len(pending_bonds),
                        metal_atom_index=center.metal.idx,
                    ),
                    attempt=stalled_attempts,
                )
                relocation = _relocate_unbound_metal(
                    mol,
                    center.metal,
                    center.pending_bonds,
                )
                metal_relocation_attempt_count += 1
                relocation_attempted_metal_indices.add(relocation.metal_idx)
                if relocation.moved:
                    relocated_metal_indices.add(relocation.metal_idx)
                else:
                    infeasible_metal_indices.add(relocation.metal_idx)
                    warning_messages.append(
                        "metal-only placement infeasible for unbound metal "
                        f"atom {relocation.metal_idx} after evaluating "
                        f"{relocation.candidates_evaluated} candidate(s)"
                    )
                record_restoration_frame(
                    (
                        TrajectoryEvent.METAL_RELOCATED
                        if relocation.moved
                        else TrajectoryEvent.METAL_RELOCATION_FAILED
                    ),
                    evidence=CoordinationFrameEvidence(
                        bond_atom_indices=None,
                        accepted=relocation.moved,
                        pending_bond_count=len(pending_bonds),
                        metal_atom_index=relocation.metal_idx,
                        relocation_status=relocation.status,
                        relocation_candidates_evaluated=(
                            relocation.candidates_evaluated
                        ),
                        safe_donor_atom_indices=relocation.safe_donor_indices,
                        minimum_normalized_clearance=(
                            relocation.minimum_normalized_clearance
                        ),
                        coordination_distance_deviation=(
                            relocation.coordination_distance_deviation
                        ),
                    ),
                    attempt=stalled_attempts,
                )
                continue
            active_metal_indices = _active_coordination_metal_indices(mol)
            if (
                relocation_attempted_metal_indices
                and not any(
                    _coordination_metal(bond).idx in active_metal_indices
                    for bond in pending_bonds
                )
            ):
                break
            if stalled_attempts >= attempt_limit:
                break
            if stalled_attempts:
                mol.coordinates = _perturbed_coordinates(
                    mol.coordinates,
                    sigma=perturb_sigma,
                    rng=rng,
                )
                record_restoration_frame(
                    TrajectoryEvent.PERTURBED,
                    attempt=stalled_attempts,
                )
            stalled_attempts += 1
        optimized = _single_ob_optimization(
            mol,
            effective_forcefield,
            relaxation_steps,
        )
        last_energy = float(optimized.energy)
        record_restoration_frame(
            TrajectoryEvent.OPTIMIZED,
            energy=last_energy,
            attempt=stalled_attempts,
        )

    forced_bond_keys = tuple(_bond_key(bond) for bond in pending_bonds)
    if pending_bonds:
        for forced_index, bond in enumerate(pending_bonds):
            mol.restore_bonds(bond, clear_conformers=False)
            record_restoration_frame(
                TrajectoryEvent.BOND_FORCED,
                evidence=CoordinationFrameEvidence(
                    bond_atom_indices=_bond_key(bond),
                    accepted=False,
                    pending_bond_count=(
                        len(pending_bonds) - forced_index - 1
                    ),
                    forced=True,
                ),
                attempt=stalled_attempts,
            )
        last_energy = None
        warning_messages.append(
            f"Forced restoration of {len(pending_bonds)} coordination bond(s) "
            "after Stage 2 could not expose a nonpiercing path "
            f"({stalled_attempts} stalled relaxation attempt(s), "
            f"{metal_relocation_attempt_count} metal relocation attempt(s))"
        )

    report = CoordinationBondRestorationReport(
        attempt_limit=attempt_limit,
        attempts_completed=stalled_attempts,
        bond_count=len(coordination_bonds),
        metal_relocation_attempt_count=metal_relocation_attempt_count,
        relocated_metal_indices=tuple(sorted(relocated_metal_indices)),
        infeasible_metal_indices=tuple(sorted(infeasible_metal_indices)),
        forced_bond_keys=forced_bond_keys,
        rejected_piercing_trial_count=rejected_piercing_trial_count,
        undetermined_trial_count=undetermined_trial_count,
        excluded_ring_observation_count=excluded_ring_observation_count,
        warning_messages=_unique_messages(warning_messages),
    )
    terminal_index = record_restoration_frame(
        TrajectoryEvent.TERMINAL,
        energy=last_energy,
        evidence=CoordinationFrameEvidence(
            bond_atom_indices=None,
            accepted=not forced_bond_keys,
            pending_bond_count=0,
            forced=bool(forced_bond_keys),
        ),
        attempt=stalled_attempts,
    )
    if terminal_index is not None and trajectory is not None:
        trajectory.select(terminal_index)
    return _CoordinationRestorationResult(
        report=report,
    )
