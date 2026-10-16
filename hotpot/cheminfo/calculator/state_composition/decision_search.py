"""Private policy-aware contracts for bounded scientific-state search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Tuple

from .contracts import (
    CandidateFactor,
    CoupledScientificStateCandidate,
    FactorCandidateValue,
    FrontierBound,
    ScientificStateIdentity,
)

__all__ = []


class _DecisionRelation(Enum):
    """Partial-order relation between two policy decision records."""

    DOMINATES = "dominates"
    DOMINATED = "dominated"
    EQUIVALENT = "equivalent"
    INCOMPARABLE = "incomparable"


@dataclass(frozen=True)
class _AuthorityVector:
    """Authority coordinate for every operation-relevant quantity."""

    coordinates: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class _PreferenceKey:
    """Versioned rule ranks followed by assumption-severity coordinates."""

    rank_values: Tuple[float, ...]
    assumption_severity: Tuple[int, ...]


@dataclass(frozen=True)
class _DecisionEquivalenceKey:
    """Operation-relevant state values that canonical ties may not cross."""

    net_charge: int
    oxidation_state_class: str
    molecular_two_s: Optional[int]


@dataclass(frozen=True)
class _DecisionRecord:
    """One hard-valid path represented in the policy decision order."""

    state_identity: ScientificStateIdentity
    authority: _AuthorityVector
    preference: _PreferenceKey
    equivalence: _DecisionEquivalenceKey


@dataclass(frozen=True)
class _DecisionBound:
    """Provable optimistic envelope for one unexpanded search subtree."""

    authority_upper: _AuthorityVector
    rank_upper: Tuple[float, ...]
    assumption_severity_lower: Tuple[int, ...]
    equivalence: Optional[_DecisionEquivalenceKey]
    canonical_identity_lower_bound: Optional[str]


class _DecisionSearchCompiler(Protocol):
    """Compile one exact policy context into path and frontier decisions."""

    @property
    def identity(self) -> str:
        """Return the immutable compiler/profile artifact identity."""

    @property
    def frontier_schema_version(self) -> str:
        """Return the exact schema used to encode optimistic bounds."""

    def record_for_path(
        self,
        path: CoupledScientificStateCandidate,
        factors: Tuple[CandidateFactor, ...],
    ) -> _DecisionRecord:
        """Return the policy decision record for one hard-valid path."""

    def optimistic_bound_for_prefix(
        self,
        selected_values: Tuple[FactorCandidateValue, ...],
        remaining_factors: Tuple[CandidateFactor, ...],
    ) -> Optional[_DecisionBound]:
        """Return a proven bound, or ``None`` when the branch is unboundable."""

    def compare_records(
        self,
        left: _DecisionRecord,
        right: _DecisionRecord,
    ) -> _DecisionRelation:
        """Compare two hard-valid paths without imposing a false total order."""

    def record_strictly_beats_bound(
        self,
        record: _DecisionRecord,
        bound: _DecisionBound,
    ) -> bool:
        """Return whether one selected record provably beats a whole branch."""

    def aggregate_bounds(
        self,
        bounds: Tuple[_DecisionBound, ...],
    ) -> Optional[_DecisionBound]:
        """Return one safe envelope, or ``None`` if aggregation is unprovable."""

    def bound_order_key(self, bound: _DecisionBound) -> Tuple[str, ...]:
        """Return a deterministic best-first queue key for one bound."""

    def encode_bound(self, bound: _DecisionBound) -> Tuple[str, ...]:
        """Encode one bound under ``frontier_schema_version``."""

    def decode_bound(
        self,
        bound: FrontierBound,
    ) -> Optional[_DecisionBound]:
        """Decode one compatible serialized bound, or return ``None``."""


def _authority_relation(
    left: _AuthorityVector,
    right: _AuthorityVector,
) -> _DecisionRelation:
    """Compare equal-axis authority vectors by Pareto dominance."""
    left_coordinates = dict(left.coordinates)
    right_coordinates = dict(right.coordinates)
    if (
        len(left_coordinates) != len(left.coordinates)
        or len(right_coordinates) != len(right.coordinates)
    ):
        raise ValueError("Authority-vector coordinates must be unique")
    if left_coordinates.keys() != right_coordinates.keys():
        raise ValueError("Authority vectors must describe the same quantities")
    left_values = tuple(
        left_coordinates[key] for key in sorted(left_coordinates)
    )
    right_values = tuple(
        right_coordinates[key] for key in sorted(right_coordinates)
    )
    left_at_least = all(
        left_value >= right_value
        for left_value, right_value in zip(left_values, right_values)
    )
    right_at_least = all(
        right_value >= left_value
        for left_value, right_value in zip(left_values, right_values)
    )
    if left_at_least and right_at_least:
        return _DecisionRelation.EQUIVALENT
    if left_at_least:
        return _DecisionRelation.DOMINATES
    if right_at_least:
        return _DecisionRelation.DOMINATED
    return _DecisionRelation.INCOMPARABLE


def _non_dominated_records(
    records: Tuple[_DecisionRecord, ...],
    compiler: _DecisionSearchCompiler,
) -> Tuple[_DecisionRecord, ...]:
    return tuple(
        record
        for record in records
        if not any(
            compiler.compare_records(other, record)
            is _DecisionRelation.DOMINATES
            for other in records
            if other.state_identity != record.state_identity
        )
    )


def _select_decision_record(
    paths: Tuple[CoupledScientificStateCandidate, ...],
    factors: Tuple[CandidateFactor, ...],
    compiler: _DecisionSearchCompiler,
) -> Optional[_DecisionRecord]:
    records = tuple(
        compiler.record_for_path(path, factors) for path in paths
    )
    non_dominated = _non_dominated_records(records, compiler)
    if not non_dominated:
        return None
    equivalence = non_dominated[0].equivalence
    if any(record.equivalence != equivalence for record in non_dominated[1:]):
        return None
    return min(
        non_dominated,
        key=lambda record: record.state_identity.digest,
    )


def _record_beats_frontier(
    record: _DecisionRecord,
    frontier_bounds: Tuple[Optional[_DecisionBound], ...],
    compiler: _DecisionSearchCompiler,
) -> bool:
    return all(
        bound is not None
        and compiler.record_strictly_beats_bound(record, bound)
        for bound in frontier_bounds
    )
