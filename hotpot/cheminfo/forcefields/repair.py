"""Ring-piercing repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np

from .. import geometry as geo
from .backend import _single_ob_optimization
from .contracts import RingUntanglingReport
from .coordinates import _copy_coordinates, _perturbed_coordinates
from .settings import _BOND_RING_MAX_SIZE
from .topology import _bond_key
from .trajectory import (
    ForceFieldTrajectory,
    RingFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
)

if TYPE_CHECKING:
    from ..core import Bond, Molecule, Ring


__all__ = ()


@dataclass(frozen=True)
class _RingUntanglingResult:
    report: RingUntanglingReport
    energy: float
    checkpoint_report: "geo.BondRingScreeningReport[Ring, Bond]"


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


@dataclass(frozen=True)
class _WatchedRingRepairObservation:
    watch_result: Optional[_RingPiercingWatchResult]


@dataclass(frozen=True)
class _RingTrajectoryRecorder:
    mol: "Molecule"
    trajectory: Optional[ForceFieldTrajectory]
    stage: TrajectoryStage
    enabled: bool

    def record(
        self,
        event: TrajectoryEvent,
        *,
        energy: Optional[float] = None,
        state: Optional[geo.PiercingState] = None,
        report: Optional["geo.BondRingScreeningReport[Ring, Bond]"] = None,
        confirmed_piercing_count: Optional[int] = None,
        attempt: Optional[int] = None,
    ) -> Optional[int]:
        """Record one ring-repair frame without making workflow decisions."""
        if not self.enabled or self.trajectory is None:
            return None
        frame = self.trajectory.record_molecule(
            self.mol,
            stage=self.stage,
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

    def record_checkpoint(
        self,
        checkpoint_report: "geo.BondRingScreeningReport[Ring, Bond]",
        *,
        energy: Optional[float] = None,
        attempt: Optional[int] = None,
    ) -> Optional[int]:
        """Record an already-computed full-scope checkpoint report."""
        if not self.enabled:
            return None
        return _record_ring_checkpoint(
            self.mol,
            checkpoint_report,
            trajectory=self.trajectory,
            stage=self.stage,
            energy=energy,
            attempt=attempt,
        )

    def select(self, frame_index: Optional[int]) -> None:
        """Select a recorded frame when trajectory capture is enabled."""
        if frame_index is not None and self.trajectory is not None:
            self.trajectory.select(frame_index)


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
    report: Optional["geo.BondRingScreeningReport[Ring, Bond]"],
    *,
    confirmed_piercing_count: Optional[int] = None,
) -> RingFrameEvidence:
    """Describe one observed ring state without triggering another scan."""
    if report is None:
        uncertain_relation_count = (
            None if state is geo.PiercingState.UNDETERMINED else 0
        )
        observed_piercing_count = (
            0
            if confirmed_piercing_count is None
            else confirmed_piercing_count
        )
    else:
        uncertain_relation_count = report.undetermined_pair_count
        observed_piercing_count = (
            report.piercing_pair_count
            if confirmed_piercing_count is None
            else confirmed_piercing_count
        )
    return RingFrameEvidence(
        confirmed_piercing_count=observed_piercing_count,
        uncertain_relation_count=uncertain_relation_count,
        ring_scope=None if report is None else report.ring_scope,
        max_ring_size=None if report is None else report.max_ring_size,
        selected_ring_count=(
            None if report is None else report.selected_ring_count
        ),
        excluded_ring_count=(
            None if report is None else report.excluded_ring_count
        ),
        candidate_pair_count=(
            None if report is None else report.candidate_pair_count
        ),
        aabb_separated_pair_count=(
            None if report is None else report.aabb_separated_pair_count
        ),
        exact_pair_count=None if report is None else report.exact_pair_count,
        does_not_pierce_pair_count=(
            None if report is None else report.does_not_pierce_pair_count
        ),
        scan_complete=None if report is None else report.scan_complete,
    )


def _record_ring_checkpoint(
    mol: "Molecule",
    checkpoint_report: "geo.BondRingScreeningReport[Ring, Bond]",
    *,
    trajectory: Optional[ForceFieldTrajectory],
    stage: TrajectoryStage,
    energy: Optional[float] = None,
    component_index: Optional[int] = None,
    attempt: Optional[int] = None,
) -> Optional[int]:
    """Record one existing full-scope report without recomputing geometry."""
    if trajectory is None or not trajectory.records(stage):
        return None
    frame = trajectory.record_molecule(
        mol,
        stage=stage,
        event=TrajectoryEvent.TOPOLOGY_CHECKPOINT,
        energy_kj_mol=energy,
        component_index=component_index,
        attempt=attempt,
        evidence=_ring_frame_evidence(
            checkpoint_report.state,
            checkpoint_report,
        ),
    )
    return frame.index


def _repair_watched_ring_piercings_once(
    mol: "Molecule",
    effective_forcefield: str,
    *,
    current_piercings: Tuple[_WatchedRingPiercing, ...],
    watch_batch: Tuple[_WatchedRingPiercing, ...],
    attempt: int,
    short_steps: int,
    perturb_sigma: float,
    rng: np.random.Generator,
    trajectory_recorder: _RingTrajectoryRecorder,
) -> Optional[_WatchedRingRepairObservation]:
    """Repair once and observe only the fixed ring--bond watch batch."""
    ring_edge = _select_ring_opening_edge(mol, current_piercings)
    if ring_edge is None:
        return None

    mol.hide_bonds(ring_edge, clear_conformers=False)
    trajectory_recorder.record(
        TrajectoryEvent.RING_OPENED,
        attempt=attempt,
    )
    optimized_successfully = False
    try:
        mol.coordinates = _perturbed_coordinates(
            mol.coordinates,
            sigma=perturb_sigma,
            rng=rng,
        )
        trajectory_recorder.record(
            TrajectoryEvent.PERTURBED,
            attempt=attempt,
        )
        optimized = _single_ob_optimization(
            mol,
            effective_forcefield,
            short_steps,
        )
        optimized_successfully = True
        trajectory_recorder.record(
            TrajectoryEvent.OPTIMIZED,
            energy=float(optimized.energy),
            attempt=attempt,
        )
    finally:
        mol.restore_bonds(ring_edge, clear_conformers=False)
        if not optimized_successfully:
            trajectory_recorder.record(
                TrajectoryEvent.RING_CLOSED,
                attempt=attempt,
            )

    return _WatchedRingRepairObservation(
        watch_result=_scan_ring_piercing_watch(mol, watch_batch),
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
    trajectory_recorder = _RingTrajectoryRecorder(
        mol=mol,
        trajectory=trajectory,
        stage=trajectory_stage,
        enabled=(
            trajectory is not None
            and trajectory.records(trajectory_stage)
        ),
    )
    return _resolve_ring_piercings(
        mol,
        effective_forcefield,
        attempt_limit=attempt_limit,
        short_steps=short_steps,
        settling_steps=settling_steps,
        perturb_sigma=perturb_sigma,
        rng=rng,
        checkpoint_report=checkpoint_report,
        initial_energy=initial_energy,
        trajectory_recorder=trajectory_recorder,
    )


def _resolve_ring_piercings(
    mol: "Molecule",
    effective_forcefield: str,
    *,
    attempt_limit: int,
    short_steps: int,
    settling_steps: int,
    perturb_sigma: float,
    rng: np.random.Generator,
    checkpoint_report: "geo.BondRingScreeningReport[Ring, Bond]",
    initial_energy: float,
    trajectory_recorder: _RingTrajectoryRecorder,
) -> _RingUntanglingResult:
    """Control targeted watch batches and their full checkpoint transitions."""
    ring_scope = checkpoint_report.ring_scope

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
            trajectory_recorder.record_checkpoint(
                report,
                energy=float(optimized.energy),
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

        repair_observation = _repair_watched_ring_piercings_once(
            mol,
            effective_forcefield,
            current_piercings=current_piercings,
            watch_batch=watch,
            attempt=attempts_completed + 1,
            short_steps=short_steps,
            perturb_sigma=perturb_sigma,
            rng=rng,
            trajectory_recorder=trajectory_recorder,
        )
        if repair_observation is None:
            unresolved_reason = (
                "Confirmed bond-ring piercing has no eligible ring-opening edge; "
                "retaining the closed-topology frame with the lowest piercing count"
            )
            break

        attempts_completed += 1
        watched_result = repair_observation.watch_result
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
            trajectory_recorder.record_checkpoint(
                report,
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
            trajectory_recorder.record(
                TrajectoryEvent.RING_CLOSED,
                attempt=attempts_completed,
            )
        settled = False

    if unresolved_reason is not None:
        mol.coordinates = watch_best_coordinates
        if attempts_completed == 0:
            candidate_report = checkpoint_report
        else:
            candidate_report = _scan_ring_checkpoint(
                mol,
                ring_scope=ring_scope,
            )
            trajectory_recorder.record_checkpoint(
                candidate_report,
                energy=(
                    watch_best_energy
                    if np.isfinite(watch_best_energy)
                    else None
                ),
                attempt=attempts_completed,
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
        trajectory_recorder.record(
            TrajectoryEvent.ROLLED_BACK,
            energy=best_trace_energy,
            attempt=attempts_completed,
        )
        if state is geo.PiercingState.PIERCES:
            warning_messages.append(unresolved_reason)
        elif state is geo.PiercingState.UNDETERMINED:
            warning_messages.append(
                "A bond-ring relation remained mathematically undetermined"
            )
    resolved = current_count == 0
    terminal_index = trajectory_recorder.record(
        TrajectoryEvent.TERMINAL,
        energy=best_trace_energy,
        attempt=attempts_completed,
    )
    trajectory_recorder.select(terminal_index)

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
