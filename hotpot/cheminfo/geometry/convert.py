"""Typed adapters from Hotpot chemical objects to factual geometry objects.

This module is the only geometry layer that understands the structural shape
of Hotpot atoms, bonds, rings, and molecules.  It intentionally uses narrow
protocols instead of importing :mod:`hotpot.cheminfo.core` at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import (
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

from . import native as _native
from .object import Cycle, Point, Segment
from .relation import (
    ClosestCycleEdge,
    PiercingState,
    PointPairDistance,
    SegmentCycleRelation,
    determine_segment_cycle_relation,
)
from .settings import DEFAULT_GEOMETRY_SETTINGS, GeometrySettings


__all__ = (
    "PairScope",
    "RingScope",
    "AtomGeometry",
    "AtomPairTarget",
    "BondGeometry",
    "RingGeometry",
    "BondRingTarget",
    "AtomPairDistance",
    "BondRingFinding",
    "RingEdgeDistance",
    "BondRingScanReport",
    "BondRingScreeningReport",
    "BondRingScreeningPlan",
    "BondRingFrameWorkspace",
    "point_from_atom",
    "segment_from_bond",
    "cycle_from_ring",
    "iter_atom_geometries",
    "iter_atom_pair_targets",
    "iter_ring_geometries",
    "iter_bond_ring_targets",
    "measure_atom_pair_distances",
    "determine_bond_ring_relation",
    "iter_bond_ring_findings",
    "scan_bond_ring_relations",
    "prepare_bond_ring_screening_plan",
    "prepare_bond_ring_frame",
    "screen_bond_ring_workspace",
    "screen_segments_against_ring_workspace",
    "screen_bonds_against_rings",
    "screen_bond_ring_relations",
    "determine_bond_ring_piercing_state",
)


PairScope = Literal["all", "bonded", "nonbonded"]
RingScope = Literal["full_graph", "ligand_skeleton"]
BondKey = Tuple[int, int]
RingKey = Tuple[int, ...]


# Source-object protocols.  They describe only what conversion needs and keep
# Core out of this module's runtime import graph.


class _AtomLike(Protocol):
    @property
    def coordinates(self) -> Sequence[float]: ...

    @property
    def idx(self) -> int: ...


_AtomSource_co = TypeVar("_AtomSource_co", bound=_AtomLike, covariant=True)


class _BondLike(Protocol[_AtomSource_co]):
    @property
    def atom1(self) -> _AtomSource_co: ...

    @property
    def atom2(self) -> _AtomSource_co: ...


class _RingLike(Protocol[_AtomSource_co]):
    @property
    def atoms(self) -> Sequence[_AtomSource_co]: ...


_BondSource_co = TypeVar(
    "_BondSource_co",
    bound=_BondLike[_AtomLike],
    covariant=True,
)
_RingSource_co = TypeVar(
    "_RingSource_co",
    bound=_RingLike[_AtomLike],
    covariant=True,
)


class _StructureLike(Protocol[_AtomSource_co, _BondSource_co]):
    @property
    def atoms(self) -> Iterable[_AtomSource_co]: ...

    @property
    def bonds(self) -> Iterable[_BondSource_co]: ...


class _MoleculeLike(
    _StructureLike[_AtomSource_co, _BondSource_co],
    Protocol[_AtomSource_co, _BondSource_co, _RingSource_co],
):
    def rings_for_scope(
        self,
        ring_scope: RingScope,
        *,
        max_size: Optional[int] = None,
        max_cycles: Optional[int] = None,
    ) -> Sequence[_RingSource_co]: ...


AtomSourceT = TypeVar("AtomSourceT", bound=_AtomLike)
BondSourceT = TypeVar("BondSourceT", bound=_BondLike[_AtomLike])
RingSourceT = TypeVar("RingSourceT", bound=_RingLike[_AtomLike])


# Immutable source mappings.


@dataclass(frozen=True)
class AtomGeometry(Generic[AtomSourceT]):
    atom: AtomSourceT
    point: Point
    key: int


@dataclass(frozen=True)
class AtomPairTarget(Generic[AtomSourceT]):
    first: AtomGeometry[AtomSourceT]
    second: AtomGeometry[AtomSourceT]
    bonded: bool


@dataclass(frozen=True)
class BondGeometry(Generic[BondSourceT]):
    bond: BondSourceT
    segment: Segment
    key: Tuple[int, int]


@dataclass(frozen=True)
class RingGeometry(Generic[RingSourceT]):
    ring: RingSourceT
    cycle: Cycle
    key: Tuple[int, ...]


@dataclass(frozen=True)
class BondRingTarget(Generic[RingSourceT, BondSourceT]):
    ring: RingGeometry[RingSourceT]
    bond: BondGeometry[BondSourceT]


@dataclass(frozen=True)
class AtomPairDistance(Generic[AtomSourceT]):
    target: AtomPairTarget[AtomSourceT]
    measurement: PointPairDistance


@dataclass(frozen=True)
class BondRingFinding(Generic[RingSourceT, BondSourceT]):
    target: BondRingTarget[RingSourceT, BondSourceT]
    relation: SegmentCycleRelation


@dataclass(frozen=True)
class RingEdgeDistance(Generic[BondSourceT]):
    bond: BondSourceT
    measurement: ClosestCycleEdge


@dataclass(frozen=True)
class BondRingScanReport(Generic[RingSourceT, BondSourceT]):
    findings: Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]
    ring_scope: RingScope
    max_ring_size: int
    selected_ring_count: int
    excluded_ring_count: int
    piercing_pair_count: int
    does_not_pierce_pair_count: int
    undetermined_pair_count: int

    @property
    def candidate_pair_count(self) -> int:
        """Return the number of fully evaluated ring--bond candidates."""
        return len(self.findings)

    @property
    def scan_complete(self) -> bool:
        """Return whether every selected candidate surface family is complete."""
        return all(
            finding.relation.surface_evidence.enumeration_complete
            for finding in self.findings
        )

    @property
    def piercings(
            self,
    ) -> Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]:
        """Return only confirmed piercing findings."""
        return tuple(
            finding
            for finding in self.findings
            if finding.relation.state is PiercingState.PIERCES
        )

    @property
    def undetermined(
            self,
    ) -> Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]:
        """Return findings whose geometric relation is unresolved."""
        return tuple(
            finding
            for finding in self.findings
            if finding.relation.state is PiercingState.UNDETERMINED
        )


@dataclass(frozen=True)
class BondRingScreeningReport(Generic[RingSourceT, BondSourceT]):
    """Sparse all-pair screening report for bond--ring piercing states."""

    actionable_findings: Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]
    ring_scope: RingScope
    max_ring_size: int
    selected_ring_count: int
    excluded_ring_count: int
    candidate_pair_count: int
    aabb_separated_pair_count: int
    exact_pair_count: int
    piercing_pair_count: int
    does_not_pierce_pair_count: int
    undetermined_pair_count: int
    scan_complete: bool

    @property
    def state(self) -> PiercingState:
        """Return the aggregate three-state result for the screened scope."""
        if self.piercing_pair_count:
            return PiercingState.PIERCES
        if self.undetermined_pair_count:
            return PiercingState.UNDETERMINED
        return PiercingState.DOES_NOT_PIERCE

    @property
    def piercings(
            self,
    ) -> Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]:
        """Return confirmed piercing findings retained by the screen."""
        return tuple(
            finding
            for finding in self.actionable_findings
            if finding.relation.state is PiercingState.PIERCES
        )

    @property
    def undetermined(
            self,
    ) -> Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]:
        """Return mathematically unresolved findings retained by the screen."""
        return tuple(
            finding
            for finding in self.actionable_findings
            if finding.relation.state is PiercingState.UNDETERMINED
        )


@dataclass(frozen=True)
class BondRingScreeningPlan(Generic[RingSourceT, BondSourceT]):
    """Coordinate-free ring--bond screening plan for one graph topology.

    Source-object references are retained so a later frame can read current
    coordinates.  The plan itself stores only topology-derived keys and pair
    selections; rebuild it after changing molecular connectivity.
    """

    ring_scope: RingScope
    max_ring_size: int
    settings: GeometrySettings
    rings: Tuple[RingSourceT, ...]
    bonds: Tuple[BondSourceT, ...]
    ring_atom_keys: Tuple[RingKey, ...]
    ring_edge_keys: Tuple[FrozenSet[BondKey], ...]
    bond_keys: Tuple[BondKey, ...]
    candidate_bond_keys_by_ring: Tuple[Tuple[BondKey, ...], ...]
    excluded_ring_count: int


@dataclass(frozen=True)
class _PreparedRingFrame(Generic[RingSourceT]):
    ring: RingGeometry[RingSourceT]
    edge_keys: FrozenSet[BondKey]
    cycle_index: int


@dataclass(frozen=True, eq=False)
class BondRingFrameWorkspace(Generic[RingSourceT, BondSourceT]):
    """Immutable coordinate snapshot prepared from a screening plan.

    All predicates consume the snapshotted ``Cycle`` and ``Segment`` objects,
    not live source coordinates.  Rebuild the frame after moving any atom.
    """

    plan: BondRingScreeningPlan[RingSourceT, BondSourceT]
    coordinate_keys: Tuple[int, ...]
    coordinates: np.ndarray
    rings: Tuple[_PreparedRingFrame[RingSourceT], ...]
    bonds: Tuple[BondGeometry[BondSourceT], ...]

    def __post_init__(self) -> None:
        coordinate_snapshot = np.asarray(self.coordinates, dtype=np.float64)
        if (
            coordinate_snapshot.flags.writeable
            or not coordinate_snapshot.flags.owndata
            or not coordinate_snapshot.flags.c_contiguous
        ):
            coordinate_snapshot = coordinate_snapshot.copy()
            coordinate_snapshot.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinate_snapshot)
        coordinate_position = {
            key: position for position, key in enumerate(self.coordinate_keys)
        }
        cycle_indices: List[int] = []
        cycle_offsets = [0]
        for ring_frame in self.rings:
            cycle_indices.extend(
                coordinate_position[_atom_key(atom)]
                for atom in ring_frame.ring.ring.atoms
            )
            cycle_offsets.append(len(cycle_indices))
        object.__setattr__(
            self,
            "_native_cycle_batch",
            _native._prepare_cycles(
                coordinate_snapshot,
                cycle_indices,
                cycle_offsets,
                self.plan.settings,
            ),
        )

    @property
    def _prepared_cycles(self) -> _native.PreparedCycleBatch:
        return object.__getattribute__(self, "_native_cycle_batch")

# Stable source keys and chemical graph selection.


def _atom_key(atom: AtomSourceT) -> int:
    return int(atom.idx)


def _validate_pair_scope(pair_scope: PairScope) -> None:
    if pair_scope not in ("all", "bonded", "nonbonded"):
        raise ValueError(f"Unsupported atom-pair scope: {pair_scope!r}")


def _bond_key(bond: BondSourceT) -> Tuple[int, int]:
    first, second = sorted((_atom_key(bond.atom1), _atom_key(bond.atom2)))
    return first, second


def _ring_key(ring: RingSourceT) -> Tuple[int, ...]:
    ordered = tuple(_atom_key(atom) for atom in ring.atoms)
    reverse = tuple(reversed(ordered))
    rotations = tuple(
        sequence[offset:] + sequence[:offset]
        for sequence in (ordered, reverse)
        for offset in range(len(sequence))
    )
    return min(rotations)


def _ring_edge_keys(ring: RingSourceT) -> Tuple[Tuple[int, int], ...]:
    atom_keys = tuple(_atom_key(atom) for atom in ring.atoms)
    return tuple(
        (
            min(atom_keys[index], atom_keys[(index + 1) % len(atom_keys)]),
            max(atom_keys[index], atom_keys[(index + 1) % len(atom_keys)]),
        )
        for index in range(len(atom_keys))
    )


def _selected_rings(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        ring_scope: RingScope,
        max_ring_size: int,
) -> Tuple[Tuple[RingSourceT, ...], int]:
    rings = tuple(sorted(mol.rings_for_scope(ring_scope), key=_ring_key))
    selected = tuple(
        ring for ring in rings if len(_ring_key(ring)) <= max_ring_size
    )
    return selected, len(rings) - len(selected)


def _iter_bond_ring_targets_from_rings(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        rings: Sequence[RingSourceT],
) -> Iterator[BondRingTarget[RingSourceT, BondSourceT]]:
    bonds = tuple(sorted(mol.bonds, key=_bond_key))
    for ring in rings:
        ring_geometry = RingGeometry(
            ring=ring,
            cycle=cycle_from_ring(ring),
            key=_ring_key(ring),
        )
        yield from _iter_bond_ring_targets_for_ring(ring_geometry, bonds)


def _iter_bond_ring_targets_for_ring(
    ring: RingGeometry[RingSourceT],
    bonds: Sequence[BondSourceT],
) -> Iterator[BondRingTarget[RingSourceT, BondSourceT]]:
    ring_edge_keys = frozenset(_ring_edge_keys(ring.ring))
    for bond in bonds:
        key = _bond_key(bond)
        if key not in ring_edge_keys:
            yield BondRingTarget(
                ring=ring,
                bond=BondGeometry(
                    bond=bond,
                    segment=segment_from_bond(bond),
                    key=key,
                ),
            )


def _selected_workspace_rings(
    workspace: BondRingFrameWorkspace[RingSourceT, BondSourceT],
    ring_keys: Optional[Iterable[RingKey]],
) -> Tuple[_PreparedRingFrame[RingSourceT], ...]:
    if ring_keys is None:
        return workspace.rings
    selected_keys = frozenset(tuple(key) for key in ring_keys)
    return tuple(
        ring_frame
        for ring_frame in workspace.rings
        if ring_frame.ring.key in selected_keys
    )


def _selected_workspace_bonds(
    bonds: Iterable[BondGeometry[BondSourceT]],
    bond_keys: Optional[Iterable[BondKey]],
) -> Tuple[BondGeometry[BondSourceT], ...]:
    selected_keys = (
        None
        if bond_keys is None
        else frozenset(tuple(sorted(key)) for key in bond_keys)
    )
    return tuple(
        sorted(
            (
                bond
                for bond in bonds
                if selected_keys is None or bond.key in selected_keys
            ),
            key=lambda bond: bond.key,
        )
    )


def _workspace_candidates(
    segments: Iterable[BondGeometry[BondSourceT]],
    workspace: BondRingFrameWorkspace[RingSourceT, BondSourceT],
    *,
    bond_keys: Optional[Iterable[BondKey]],
    ring_keys: Optional[Iterable[RingKey]],
) -> Tuple[
    Tuple[_PreparedRingFrame[RingSourceT], ...],
    Tuple[BondGeometry[BondSourceT], ...],
    Tuple[BondRingTarget[RingSourceT, BondSourceT], ...],
    Tuple[Tuple[int, int], ...],
]:
    """Select sources and pack native pair indices in canonical ring order."""

    selected_rings = _selected_workspace_rings(workspace, ring_keys)
    selected_bonds = _selected_workspace_bonds(segments, bond_keys)
    targets = []
    candidate_pairs = []
    for ring_frame in selected_rings:
        for segment_index, bond in enumerate(selected_bonds):
            if bond.key in ring_frame.edge_keys:
                continue
            targets.append(BondRingTarget(ring=ring_frame.ring, bond=bond))
            candidate_pairs.append((segment_index, ring_frame.cycle_index))
    return (
        selected_rings,
        selected_bonds,
        tuple(targets),
        tuple(candidate_pairs),
    )


def _dense_workspace_findings(
    workspace: BondRingFrameWorkspace[RingSourceT, BondSourceT],
) -> Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]:
    """Evaluate every planned ring--bond pair in one exact native call."""

    _, selected_bonds, targets, candidate_pairs = _workspace_candidates(
        workspace.bonds,
        workspace,
        bond_keys=None,
        ring_keys=None,
    )
    native_relations = _native._determine_segment_cycle_relations(
        workspace._prepared_cycles,
        (bond.segment for bond in selected_bonds),
        candidate_pairs,
    )
    return tuple(
        BondRingFinding(
            target=target,
            relation=_native._segment_cycle_relation_result(
                native_relation,
                target.ring.cycle,
                workspace.plan.settings,
            ),
        )
        for target, native_relation in zip(targets, native_relations)
    )


def _screen_segment_geometries(
    segments: Iterable[BondGeometry[BondSourceT]],
    workspace: BondRingFrameWorkspace[RingSourceT, BondSourceT],
    *,
    bond_keys: Optional[Iterable[BondKey]],
    ring_keys: Optional[Iterable[RingKey]],
    stop_after_confirmed: bool,
) -> BondRingScreeningReport[RingSourceT, BondSourceT]:
    selected_rings, selected_bonds, targets, candidate_pairs = (
        _workspace_candidates(
            segments,
            workspace,
            bond_keys=bond_keys,
            ring_keys=ring_keys,
        )
    )
    batch = _native._screen_segments(
        workspace._prepared_cycles,
        (bond.segment for bond in selected_bonds),
        candidate_pairs,
        _native.DetailLevel.ACTIONABLE,
        stop_after_confirmed=stop_after_confirmed,
    )
    actionable_findings = tuple(
        BondRingFinding(
            target=targets[position],
            relation=_native._segment_cycle_relation_result(
                native_relation,
                targets[position].ring.cycle,
                workspace.plan.settings,
            ),
        )
        for position, native_relation in zip(
            batch.relation_positions,
            batch.relations,
        )
    )

    return BondRingScreeningReport(
        actionable_findings=actionable_findings,
        ring_scope=workspace.plan.ring_scope,
        max_ring_size=workspace.plan.max_ring_size,
        selected_ring_count=len(selected_rings),
        excluded_ring_count=workspace.plan.excluded_ring_count,
        candidate_pair_count=batch.evaluated_pair_count,
        aabb_separated_pair_count=batch.aabb_separated_pair_count,
        exact_pair_count=batch.exact_pair_count,
        piercing_pair_count=batch.piercing_pair_count,
        does_not_pierce_pair_count=batch.does_not_pierce_pair_count,
        undetermined_pair_count=batch.undetermined_pair_count,
        scan_complete=(
            batch.scan_complete
            and all(batch.surface_complete)
        ),
    )


# Public conversion and aggregation interfaces.  These functions never mutate
# source objects, their coordinates, conformers, or Core ring caches.


def point_from_atom(atom: AtomSourceT) -> Point:
    """Convert one atom position to an immutable point."""
    return Point.from_coordinates(atom.coordinates)


def segment_from_bond(bond: BondSourceT) -> Segment:
    """Convert one chemical bond to a finite geometric segment."""
    return Segment(point_from_atom(bond.atom1), point_from_atom(bond.atom2))


def cycle_from_ring(ring: RingSourceT) -> Cycle:
    """Convert an already perceived, ordered ring boundary to a cycle.

    This adapter copies coordinates only; it performs no ring perception.
    """
    return Cycle(tuple(point_from_atom(atom) for atom in ring.atoms))


def iter_atom_geometries(
        structure: _StructureLike[AtomSourceT, BondSourceT],
) -> Iterator[AtomGeometry[AtomSourceT]]:
    """Yield atoms with their immutable point and stable molecular index."""
    for atom in structure.atoms:
        yield AtomGeometry(atom=atom, point=point_from_atom(atom), key=_atom_key(atom))


def iter_atom_pair_targets(
        structure: _StructureLike[AtomSourceT, BondSourceT],
        pair_scope: PairScope,
) -> Iterator[AtomPairTarget[AtomSourceT]]:
    """Yield stable atom pairs selected by their chemical graph relation."""
    _validate_pair_scope(pair_scope)
    atom_geometries = tuple(iter_atom_geometries(structure))
    bonded_keys = frozenset(_bond_key(bond) for bond in structure.bonds)
    for first, second in combinations(atom_geometries, 2):
        bonded = tuple(sorted((first.key, second.key))) in bonded_keys
        if pair_scope == "all" or (pair_scope == "bonded") == bonded:
            yield AtomPairTarget(first=first, second=second, bonded=bonded)


def iter_ring_geometries(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        *,
        ring_scope: RingScope,
        max_ring_size: int,
) -> Iterator[RingGeometry[RingSourceT]]:
    """Yield selected Relevant Cycles in canonical key order."""
    rings, _ = _selected_rings(mol, ring_scope, max_ring_size)
    for ring in rings:
        yield RingGeometry(
            ring=ring,
            cycle=cycle_from_ring(ring),
            key=_ring_key(ring),
        )


def iter_bond_ring_targets(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        *,
        ring_scope: RingScope,
        max_ring_size: int,
) -> Iterator[BondRingTarget[RingSourceT, BondSourceT]]:
    """Yield every selected Ring x Bond pair except the ring's own edges."""
    rings, _ = _selected_rings(mol, ring_scope, max_ring_size)
    yield from _iter_bond_ring_targets_from_rings(mol, rings)


def measure_atom_pair_distances(
        structure: _StructureLike[AtomSourceT, BondSourceT],
        pair_scope: PairScope,
) -> Tuple[AtomPairDistance[AtomSourceT], ...]:
    """Measure selected atom pairs while retaining their chemical sources."""
    _validate_pair_scope(pair_scope)
    atom_geometries = tuple(iter_atom_geometries(structure))
    position_by_key = {
        atom_geometry.key: position
        for position, atom_geometry in enumerate(atom_geometries)
    }
    bonded_keys = frozenset(_bond_key(bond) for bond in structure.bonds)
    targets = tuple(
        AtomPairTarget(
            first=first,
            second=second,
            bonded=(tuple(sorted((first.key, second.key))) in bonded_keys),
        )
        for first, second in combinations(atom_geometries, 2)
        if (
            pair_scope == "all"
            or (pair_scope == "bonded")
            == (tuple(sorted((first.key, second.key))) in bonded_keys)
        )
    )
    pair_indices = tuple(
        (
            position_by_key[target.first.key],
            position_by_key[target.second.key],
        )
        for target in targets
    )
    measurements = _native._selected_point_pair_distances(
        tuple(atom_geometry.point for atom_geometry in atom_geometries),
        pair_indices,
    )
    return tuple(
        AtomPairDistance(target=target, measurement=measurement)
        for target, measurement in zip(targets, measurements)
    )


def determine_bond_ring_relation(
        ring: RingSourceT,
        bond: BondSourceT,
        *,
        settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingFinding[RingSourceT, BondSourceT]:
    """Determine one bond-ring relation and retain both source objects."""
    target = BondRingTarget(
        ring=RingGeometry(
            ring=ring,
            cycle=cycle_from_ring(ring),
            key=_ring_key(ring),
        ),
        bond=BondGeometry(
            bond=bond,
            segment=segment_from_bond(bond),
            key=_bond_key(bond),
        ),
    )
    return BondRingFinding(
        target=target,
        relation=determine_segment_cycle_relation(
            target.bond.segment,
            target.ring.cycle,
            settings=settings,
        ),
    )


def iter_bond_ring_findings(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        *,
        ring_scope: RingScope,
        max_ring_size: int,
        settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> Iterator[BondRingFinding[RingSourceT, BondSourceT]]:
    """Evaluate one packed frame and yield findings in canonical pair order."""

    plan = prepare_bond_ring_screening_plan(
        mol,
        ring_scope=ring_scope,
        max_ring_size=max_ring_size,
        settings=settings,
    )
    yield from _dense_workspace_findings(prepare_bond_ring_frame(plan))


def scan_bond_ring_relations(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        *,
        ring_scope: RingScope,
        max_ring_size: int,
        settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingScanReport[RingSourceT, BondSourceT]:
    """Return a dense factual report for the declared ring selection scope."""
    plan = prepare_bond_ring_screening_plan(
        mol,
        ring_scope=ring_scope,
        max_ring_size=max_ring_size,
        settings=settings,
    )
    findings = _dense_workspace_findings(prepare_bond_ring_frame(plan))
    piercing_pair_count = sum(
        finding.relation.state is PiercingState.PIERCES
        for finding in findings
    )
    does_not_pierce_pair_count = sum(
        finding.relation.state is PiercingState.DOES_NOT_PIERCE
        for finding in findings
    )
    undetermined_pair_count = sum(
        finding.relation.state is PiercingState.UNDETERMINED
        for finding in findings
    )
    return BondRingScanReport(
        findings=findings,
        ring_scope=ring_scope,
        max_ring_size=max_ring_size,
        selected_ring_count=len(plan.rings),
        excluded_ring_count=plan.excluded_ring_count,
        piercing_pair_count=piercing_pair_count,
        does_not_pierce_pair_count=does_not_pierce_pair_count,
        undetermined_pair_count=undetermined_pair_count,
    )


def prepare_bond_ring_screening_plan(
    mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
    *,
    ring_scope: RingScope,
    max_ring_size: int,
    bonds: Optional[Iterable[BondSourceT]] = None,
    settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingScreeningPlan[RingSourceT, BondSourceT]:
    """Capture coordinate-free ring and bond selections for one topology."""
    selected_rings, excluded_ring_count = _selected_rings(
        mol,
        ring_scope,
        max_ring_size,
    )
    selected_bonds = tuple(sorted(
        mol.bonds if bonds is None else bonds,
        key=_bond_key,
    ))
    ring_atom_keys = tuple(_ring_key(ring) for ring in selected_rings)
    ring_edge_keys = tuple(
        frozenset(_ring_edge_keys(ring)) for ring in selected_rings
    )
    bond_keys = tuple(_bond_key(bond) for bond in selected_bonds)
    return BondRingScreeningPlan(
        ring_scope=ring_scope,
        max_ring_size=max_ring_size,
        settings=settings,
        rings=selected_rings,
        bonds=selected_bonds,
        ring_atom_keys=ring_atom_keys,
        ring_edge_keys=ring_edge_keys,
        bond_keys=bond_keys,
        candidate_bond_keys_by_ring=tuple(
            tuple(key for key in bond_keys if key not in edge_keys)
            for edge_keys in ring_edge_keys
        ),
        excluded_ring_count=excluded_ring_count,
    )


def prepare_bond_ring_frame(
    plan: BondRingScreeningPlan[RingSourceT, BondSourceT],
) -> BondRingFrameWorkspace[RingSourceT, BondSourceT]:
    """Snapshot coordinates and prepare cycle facts for one screening frame."""
    atom_by_key: Dict[int, _AtomLike] = {}
    for ring in plan.rings:
        for atom in ring.atoms:
            atom_by_key.setdefault(_atom_key(atom), atom)
    for bond in plan.bonds:
        atom_by_key.setdefault(_atom_key(bond.atom1), bond.atom1)
        atom_by_key.setdefault(_atom_key(bond.atom2), bond.atom2)
    coordinate_keys = tuple(sorted(atom_by_key))
    coordinates = np.array(
        [atom_by_key[key].coordinates for key in coordinate_keys],
        dtype=np.float64,
        copy=True,
    )
    coordinates.setflags(write=False)
    coordinate_position = {
        key: position for position, key in enumerate(coordinate_keys)
    }
    ring_frames: List[_PreparedRingFrame[RingSourceT]] = []
    for cycle_index, (ring, ring_key, edge_keys) in enumerate(zip(
        plan.rings,
        plan.ring_atom_keys,
        plan.ring_edge_keys,
    )):
        ordered_positions = tuple(
            coordinate_position[_atom_key(atom)] for atom in ring.atoms
        )
        cycle = Cycle(tuple(
            Point(tuple(float(value) for value in coordinates[position]))
            for position in ordered_positions
        ))
        ring_frames.append(_PreparedRingFrame(
            ring=RingGeometry(
                ring=ring,
                cycle=cycle,
                key=ring_key,
            ),
            edge_keys=edge_keys,
            cycle_index=cycle_index,
        ))
    bond_frames = tuple(
        BondGeometry(
            bond=bond,
            segment=Segment(
                Point(tuple(
                    float(value)
                    for value in coordinates[coordinate_position[_atom_key(bond.atom1)]]
                )),
                Point(tuple(
                    float(value)
                    for value in coordinates[coordinate_position[_atom_key(bond.atom2)]]
                )),
            ),
            key=key,
        )
        for bond, key in zip(plan.bonds, plan.bond_keys)
    )
    return BondRingFrameWorkspace(
        plan=plan,
        coordinate_keys=coordinate_keys,
        coordinates=coordinates,
        rings=tuple(ring_frames),
        bonds=bond_frames,
    )


def screen_segments_against_ring_workspace(
    segments: Iterable[BondGeometry[BondSourceT]],
    workspace: BondRingFrameWorkspace[RingSourceT, BondSourceT],
    *,
    bond_keys: Optional[Iterable[BondKey]] = None,
    ring_keys: Optional[Iterable[RingKey]] = None,
    stop_after_confirmed: bool = False,
) -> BondRingScreeningReport[RingSourceT, BondSourceT]:
    """Screen immutable keyed segments against a prepared ring frame.

    With ``stop_after_confirmed=True``, ``scan_complete`` is false when the
    first confirmed piercing leaves any selected pair unevaluated.
    """
    return _screen_segment_geometries(
        segments,
        workspace,
        bond_keys=bond_keys,
        ring_keys=ring_keys,
        stop_after_confirmed=stop_after_confirmed,
    )


def screen_bond_ring_workspace(
    workspace: BondRingFrameWorkspace[RingSourceT, BondSourceT],
    *,
    bond_keys: Optional[Iterable[BondKey]] = None,
    ring_keys: Optional[Iterable[RingKey]] = None,
    stop_after_confirmed: bool = False,
) -> BondRingScreeningReport[RingSourceT, BondSourceT]:
    """Screen the workspace's snapshotted bonds against its rings.

    With ``stop_after_confirmed=True``, ``scan_complete`` is false when the
    first confirmed piercing leaves any selected pair unevaluated.
    """
    return screen_segments_against_ring_workspace(
        workspace.bonds,
        workspace,
        bond_keys=bond_keys,
        ring_keys=ring_keys,
        stop_after_confirmed=stop_after_confirmed,
    )


def screen_bonds_against_rings(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        bonds: Iterable[BondSourceT],
        *,
        ring_scope: RingScope,
        max_ring_size: int,
        settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingScreeningReport[RingSourceT, BondSourceT]:
    """Screen explicit bonds against selected rings with strict AABB culling."""
    plan = prepare_bond_ring_screening_plan(
        mol,
        ring_scope=ring_scope,
        max_ring_size=max_ring_size,
        bonds=bonds,
        settings=settings,
    )
    return screen_bond_ring_workspace(prepare_bond_ring_frame(plan))


def screen_bond_ring_relations(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        *,
        ring_scope: RingScope,
        max_ring_size: int,
        settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingScreeningReport[RingSourceT, BondSourceT]:
    """Screen all molecular bonds against the selected ring scope."""
    return screen_bonds_against_rings(
        mol,
        mol.bonds,
        ring_scope=ring_scope,
        max_ring_size=max_ring_size,
        settings=settings,
    )


def determine_bond_ring_piercing_state(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        *,
        ring_scope: RingScope,
        max_ring_size: int,
        settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PiercingState:
    """Return an aggregate native state, stopping at the first piercing."""

    workspace = prepare_bond_ring_frame(prepare_bond_ring_screening_plan(
        mol,
        ring_scope=ring_scope,
        max_ring_size=max_ring_size,
        settings=settings,
    ))
    _, bonds, _, candidate_pairs = _workspace_candidates(
        workspace.bonds,
        workspace,
        bond_keys=None,
        ring_keys=None,
    )
    batch = _native._screen_segments(
        workspace._prepared_cycles,
        (bond.segment for bond in bonds),
        candidate_pairs,
        _native.DetailLevel.STATE_ONLY,
        stop_after_confirmed=True,
    )
    if batch.piercing_pair_count:
        return PiercingState.PIERCES
    if batch.undetermined_pair_count:
        return PiercingState.UNDETERMINED
    return PiercingState.DOES_NOT_PIERCE
