"""Typed adapters from Hotpot chemical objects to factual geometry objects.

This module is the only geometry layer that understands the structural shape
of Hotpot atoms, bonds, rings, and molecules.  It intentionally uses narrow
protocols instead of importing :mod:`hotpot.cheminfo.core` at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, tee
from typing import (
    Generic,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

from .object import Cycle, Point, Segment
from .relation import (
    ClosestCycleEdge,
    PiercingState,
    PointPairDistance,
    SegmentCycleRelation,
    determine_segment_cycle_relation,
    iter_segment_cycle_relations,
    point_pair_distances,
)
from .settings import DEFAULT_GEOMETRY_SETTINGS, GeometrySettings


__all__ = (
    "PairScope",
    "RingScope",
    "RingFamily",
    "AtomGeometry",
    "AtomPairTarget",
    "BondGeometry",
    "RingGeometry",
    "BondRingTarget",
    "AtomPairDistance",
    "BondRingFinding",
    "RingEdgeDistance",
    "BondRingScanReport",
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
    "determine_bond_ring_piercing_state",
)


PairScope = Literal["all", "bonded", "nonbonded"]
RingScope = Literal["full_graph", "ligand_skeleton"]


class RingFamily(str, Enum):
    """Algorithm family used by Core to select molecular rings."""

    NETWORKX_CYCLE_BASIS = "networkx_cycle_basis"
    RELEVANT_CYCLES = "relevant_cycles"


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
    source: AtomSourceT
    point: Point
    key: int


@dataclass(frozen=True)
class AtomPairTarget(Generic[AtomSourceT]):
    first: AtomGeometry[AtomSourceT]
    second: AtomGeometry[AtomSourceT]
    bonded: bool


@dataclass(frozen=True)
class BondGeometry(Generic[BondSourceT]):
    source: BondSourceT
    segment: Segment
    key: Tuple[int, int]


@dataclass(frozen=True)
class RingGeometry(Generic[RingSourceT]):
    source: RingSourceT
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
    source_bond: BondSourceT
    measurement: ClosestCycleEdge


@dataclass(frozen=True)
class BondRingScanReport(Generic[RingSourceT, BondSourceT]):
    findings: Tuple[BondRingFinding[RingSourceT, BondSourceT], ...]
    ring_scope: RingScope
    ring_family: RingFamily
    max_ring_size: int
    selected_ring_count: int
    excluded_ring_count: Optional[int]
    candidate_pair_count: int
    evaluated_pair_count: int
    piercing_pair_count: int
    does_not_pierce_pair_count: int
    undetermined_pair_count: int
    scan_complete: bool

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
) -> Tuple[Tuple[RingSourceT, ...], Optional[int]]:
    selected = tuple(
        sorted(
            mol.rings_for_scope(ring_scope, max_size=max_ring_size),
            key=_ring_key,
        )
    )
    return selected, None


def _iter_bond_ring_targets_from_rings(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        rings: Sequence[RingSourceT],
) -> Iterator[BondRingTarget[RingSourceT, BondSourceT]]:
    bonds = tuple(sorted(mol.bonds, key=_bond_key))
    for ring in rings:
        ring_geometry = RingGeometry(
            source=ring,
            cycle=cycle_from_ring(ring),
            key=_ring_key(ring),
        )
        yield from _iter_bond_ring_targets_for_ring(ring_geometry, bonds)


def _iter_bond_ring_targets_for_ring(
    ring: RingGeometry[RingSourceT],
    bonds: Sequence[BondSourceT],
) -> Iterator[BondRingTarget[RingSourceT, BondSourceT]]:
    ring_edge_keys = frozenset(_ring_edge_keys(ring.source))
    for bond in bonds:
        key = _bond_key(bond)
        if key not in ring_edge_keys:
            yield BondRingTarget(
                ring=ring,
                bond=BondGeometry(
                    source=bond,
                    segment=segment_from_bond(bond),
                    key=key,
                ),
            )


def _iter_bond_ring_findings_from_rings(
    mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
    rings: Sequence[RingSourceT],
    settings: GeometrySettings,
) -> Iterator[BondRingFinding[RingSourceT, BondSourceT]]:
    bonds = tuple(sorted(mol.bonds, key=_bond_key))
    for ring in rings:
        ring_geometry = RingGeometry(
            source=ring,
            cycle=cycle_from_ring(ring),
            key=_ring_key(ring),
        )
        target_source = _iter_bond_ring_targets_for_ring(ring_geometry, bonds)
        finding_targets, relation_targets = tee(target_source)
        relations = iter_segment_cycle_relations(
            (target.bond.segment for target in relation_targets),
            ring_geometry.cycle,
            settings,
        )
        for target, relation in zip(finding_targets, relations):
            yield BondRingFinding(target=target, relation=relation)


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
        yield AtomGeometry(source=atom, point=point_from_atom(atom), key=_atom_key(atom))


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
            source=ring,
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
    measurement_by_positions = {
        (measurement.first_index, measurement.second_index): measurement
        for measurement in point_pair_distances(
            tuple(atom_geometry.point for atom_geometry in atom_geometries)
        )
    }
    return tuple(
        AtomPairDistance(
            target=target,
            measurement=measurement_by_positions[
                (
                    position_by_key[target.first.key],
                    position_by_key[target.second.key],
                )
            ],
        )
        for target in iter_atom_pair_targets(structure, pair_scope)
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
            source=ring,
            cycle=cycle_from_ring(ring),
            key=_ring_key(ring),
        ),
        bond=BondGeometry(
            source=bond,
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
    """Lazily evaluate selected bond-ring pairs in canonical key order."""
    rings, _ = _selected_rings(mol, ring_scope, max_ring_size)
    yield from _iter_bond_ring_findings_from_rings(mol, rings, settings)


def scan_bond_ring_relations(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        *,
        ring_scope: RingScope,
        max_ring_size: int,
        settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> BondRingScanReport[RingSourceT, BondSourceT]:
    """Return a dense factual report for the declared ring selection scope."""
    selected_rings, excluded_ring_count = _selected_rings(
        mol,
        ring_scope,
        max_ring_size,
    )
    findings = tuple(
        _iter_bond_ring_findings_from_rings(mol, selected_rings, settings)
    )
    candidate_pair_count = len(findings)
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
    evaluated_pair_count = len(findings)
    return BondRingScanReport(
        findings=findings,
        ring_scope=ring_scope,
        ring_family=RingFamily.RELEVANT_CYCLES,
        max_ring_size=max_ring_size,
        selected_ring_count=len(selected_rings),
        excluded_ring_count=excluded_ring_count,
        candidate_pair_count=candidate_pair_count,
        evaluated_pair_count=evaluated_pair_count,
        piercing_pair_count=piercing_pair_count,
        does_not_pierce_pair_count=does_not_pierce_pair_count,
        undetermined_pair_count=undetermined_pair_count,
        scan_complete=(
            evaluated_pair_count == candidate_pair_count
            and all(
                finding.relation.surface_evidence.enumeration_complete
                for finding in findings
            )
        ),
    )


def determine_bond_ring_piercing_state(
        mol: _MoleculeLike[AtomSourceT, BondSourceT, RingSourceT],
        *,
        ring_scope: RingScope,
        max_ring_size: int,
        settings: GeometrySettings = DEFAULT_GEOMETRY_SETTINGS,
) -> PiercingState:
    """Return a lazy aggregate, stopping at the first confirmed piercing."""
    aggregate = PiercingState.DOES_NOT_PIERCE
    for finding in iter_bond_ring_findings(
            mol,
            ring_scope=ring_scope,
            max_ring_size=max_ring_size,
            settings=settings,
    ):
        if finding.relation.state is PiercingState.PIERCES:
            return PiercingState.PIERCES
        if finding.relation.state is PiercingState.UNDETERMINED:
            aggregate = PiercingState.UNDETERMINED
    return aggregate
