"""Decision primitives frozen by the coordination-screening profile."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..charge.contracts import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
)
from ..contracts import (
    ScientificOperationRequirements,
    ScientificQuantity,
    StateValueAuthority,
)
from ..electronic_state.contracts import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
)
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    OxidationStateCandidateEvidence,
)
from ..state_composition.contracts import (
    CandidateFactor,
    CoupledScientificStateCandidate,
    DerivedStateValue,
    FactorCandidateValue,
    FrontierBound,
    StateCandidate,
)
from ..state_composition.decision_search import (
    _AuthorityVector,
    _DecisionBound,
    _DecisionEquivalenceKey,
    _DecisionRecord,
    _DecisionRelation,
    _DecisionSearchCompiler,
    _PreferenceKey,
    _authority_relation,
)
from .contracts import RankAdjustment
from .profiles import _ScientificStateProfileArtifact

__all__ = []


_AUTHORITY_RANK = {
    StateValueAuthority.UNRESOLVED: 0,
    StateValueAuthority.DERIVED_CONSTRAINT: 1,
    StateValueAuthority.DOMAIN_RULE: 2,
    StateValueAuthority.EXACT_IDENTITY_TRUSTED_SOURCE: 3,
    StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE: 4,
    StateValueAuthority.VERIFIED_EXPLICIT: 5,
}


@dataclass(frozen=True)
class _AssumptionSeverityPolicy:
    """Versioned categories ordered from most to least severe."""

    identity: str
    ordered_categories: Tuple[str, ...]

    def vector(
        self,
        assumption_ids: Tuple[str, ...],
    ) -> Optional[Tuple[int, ...]]:
        categories = tuple(
            assumption_id.partition(":")[0]
            for assumption_id in assumption_ids
        )
        known = frozenset(self.ordered_categories)
        if any(not category or category not in known for category in categories):
            return None
        return tuple(
            categories.count(category)
            for category in self.ordered_categories
        )


def _assumption_severity_policy(
    profile: _ScientificStateProfileArtifact,
) -> _AssumptionSeverityPolicy:
    ordered = tuple(
        category
        for category, _ in sorted(
            profile.assumption_severity,
            key=lambda item: (-item[1], item[0]),
        )
    )
    encoded = json.dumps(
        {
            "profile_digest": profile.content_digest,
            "ordered_categories": ordered,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _AssumptionSeverityPolicy(
        hashlib.sha256(encoded).hexdigest(),
        ordered,
    )


def _candidate_quantity(candidate: StateCandidate) -> ScientificQuantity:
    if isinstance(candidate, DerivedStateValue):
        return candidate.quantity
    if isinstance(candidate, NetChargeCandidateEvidence):
        return ScientificQuantity.NET_CHARGE
    if isinstance(candidate, LewisFormalChargeCandidate):
        return ScientificQuantity.LEWIS_FORMAL_CHARGE
    if isinstance(candidate, CoordinationCountingCandidate):
        return ScientificQuantity.COORDINATION_COUNTING
    if isinstance(candidate, OxidationStateCandidateEvidence):
        return ScientificQuantity.OXIDATION_STATE
    if isinstance(
        candidate,
        (SpinContributionCandidate, ElectronicStateCandidate),
    ):
        return ScientificQuantity.ELECTRONIC_STATE
    raise TypeError("Unsupported scientific-state candidate")


def _candidate_index(
    factors: Tuple[CandidateFactor, ...],
) -> Dict[str, Tuple[ScientificQuantity, str, StateCandidate]]:
    return {
        value.candidate.provenance.identity: (
            factor.factor_key.quantity,
            factor.factor_key.subject_identity,
            value.candidate,
        )
        for factor in factors
        for value in factor.values
    }


def _authority_coordinate(
    quantity: ScientificQuantity,
    evidence_ids: Tuple[str, ...],
    candidates: Dict[str, Tuple[ScientificQuantity, str, StateCandidate]],
    derived_values: Tuple[DerivedStateValue, ...],
) -> int:
    subject_ranks: Dict[str, int] = {}
    for evidence_id in evidence_ids:
        indexed = candidates.get(evidence_id)
        if indexed is None or indexed[0] is not quantity:
            continue
        _, subject, candidate = indexed
        rank = _AUTHORITY_RANK[candidate.provenance.authority]
        subject_ranks[subject] = max(subject_ranks.get(subject, 0), rank)
    if quantity is ScientificQuantity.NET_CHARGE:
        for candidate in derived_values:
            if candidate.quantity is quantity:
                subject_ranks["molecule"] = max(
                    subject_ranks.get("molecule", 0),
                    _AUTHORITY_RANK[StateValueAuthority.DERIVED_CONSTRAINT],
                )
    return min(subject_ranks.values()) if subject_ranks else 0


def _oxidation_equivalence(
    path: CoupledScientificStateCandidate,
    candidates: Dict[str, Tuple[ScientificQuantity, str, StateCandidate]],
) -> str:
    identity = path.oxidation_state_assignment_id
    if identity is None:
        return "unresolved"
    indexed = candidates.get(identity)
    if indexed is None or not isinstance(
        indexed[2],
        OxidationStateCandidateEvidence,
    ):
        return "unresolved"
    candidate = indexed[2]
    return json.dumps(
        (candidate.convention.value, candidate.atom_states),
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _molecular_two_s(
    path: CoupledScientificStateCandidate,
    candidates: Dict[str, Tuple[ScientificQuantity, str, StateCandidate]],
) -> Optional[int]:
    identity = path.electronic_state_candidate_identity
    if identity is None:
        return None
    indexed = candidates.get(identity)
    if indexed is None or not isinstance(indexed[2], ElectronicStateCandidate):
        return None
    return indexed[2].two_s


def _sortable_float(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return (~bits & ((1 << 64) - 1)) if bits >> 63 else bits ^ (1 << 63)


class _CoordinationScreeningDecisionCompiler(_DecisionSearchCompiler):
    """Profile-frozen authority, preference and equivalence compiler."""

    def __init__(
        self,
        profile: _ScientificStateProfileArtifact,
        requirements: ScientificOperationRequirements,
        rank_adjustments: Tuple[RankAdjustment, ...],
    ) -> None:
        dimensions = frozenset(profile.decision_rank_dimensions)
        if any(
            adjustment.rank_dimension not in dimensions
            or not math.isfinite(adjustment.value)
            for adjustment in rank_adjustments
        ):
            raise ValueError("Rank adjustment is outside the profile schema")
        self._profile = profile
        self._requirements = requirements
        self._rank_adjustments = tuple(
            sorted(
                rank_adjustments,
                key=lambda item: (
                    item.rank_dimension,
                    item.target_identity,
                    item.value,
                    item.reason,
                ),
            )
        )
        self._severity = _assumption_severity_policy(profile)
        payload = {
            "profile": profile.content_digest,
            "requirements": (
                requirements.operation_id,
                tuple(
                    sorted(
                        quantity.value
                        for quantity in requirements.required_resolved_quantities
                    )
                ),
            ),
            "rank_adjustments": tuple(
                (
                    item.target_identity,
                    item.rank_dimension,
                    item.value,
                    item.reason,
                )
                for item in self._rank_adjustments
            ),
            "severity": self._severity.identity,
        }
        self._identity = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def frontier_schema_version(self) -> str:
        return self._profile.frontier_schema_version

    def _authority_quantities(self) -> Tuple[ScientificQuantity, ...]:
        quantities = set(self._requirements.required_resolved_quantities)
        if quantities.intersection(
            {
                ScientificQuantity.NET_CHARGE,
                ScientificQuantity.OXIDATION_STATE,
            }
        ):
            quantities.update(
                {
                    ScientificQuantity.LEWIS_FORMAL_CHARGE,
                    ScientificQuantity.COORDINATION_COUNTING,
                    ScientificQuantity.OXIDATION_STATE,
                }
            )
        return tuple(sorted(quantities, key=lambda quantity: quantity.value))

    def _rank_values(self, candidate_ids: frozenset[str]) -> Tuple[float, ...]:
        return tuple(
            sum(
                adjustment.value
                for adjustment in self._rank_adjustments
                if adjustment.rank_dimension == dimension
                and adjustment.target_identity in candidate_ids
            )
            for dimension in self._profile.decision_rank_dimensions
        )

    def record_for_path(
        self,
        path: CoupledScientificStateCandidate,
        factors: Tuple[CandidateFactor, ...],
    ) -> _DecisionRecord:
        candidates = _candidate_index(factors)
        quantities = self._authority_quantities()
        authority = _AuthorityVector(
            tuple(
                (
                    quantity.value,
                    _authority_coordinate(
                        quantity,
                        path.evidence_ids,
                        candidates,
                        path.derived_values,
                    ),
                )
                for quantity in quantities
            )
        )
        severity = self._severity.vector(path.assumption_ids)
        return _DecisionRecord(
            path.identity,
            authority,
            _PreferenceKey(
                self._rank_values(frozenset(path.evidence_ids)),
                severity
                if severity is not None
                else tuple(0 for _ in self._severity.ordered_categories),
            ),
            _DecisionEquivalenceKey(
                path.net_charge.value,
                _oxidation_equivalence(path, candidates),
                _molecular_two_s(path, candidates),
            ),
            certifiable=severity is not None,
        )

    def optimistic_bound_for_prefix(
        self,
        selected_values: Tuple[FactorCandidateValue, ...],
        remaining_factors: Tuple[CandidateFactor, ...],
    ) -> Optional[_DecisionBound]:
        selected_candidates = tuple(
            value.candidate for value in selected_values
        )
        selected_ids = frozenset(
            candidate.provenance.identity
            for candidate in selected_candidates
        )
        selected_assumptions = tuple(
            assumption
            for candidate in selected_candidates
            for assumption in candidate.provenance.assumption_ids
        )
        severity_lower = self._severity.vector(selected_assumptions)
        if severity_lower is None:
            return None
        quantities = self._authority_quantities()
        possible_candidates = selected_candidates + tuple(
            value.candidate
            for factor in remaining_factors
            for value in factor.values
        )
        authority_upper = _AuthorityVector(
            tuple(
                (
                    quantity.value,
                    max(
                        (
                            _AUTHORITY_RANK[candidate.provenance.authority]
                            for candidate in possible_candidates
                            if _candidate_quantity(candidate) is quantity
                        ),
                        default=0,
                    ),
                )
                for quantity in quantities
            )
        )
        rank_upper = tuple(
            sum(
                adjustment.value
                for adjustment in self._rank_adjustments
                if adjustment.rank_dimension == dimension
                and adjustment.target_identity in selected_ids
            )
            + sum(
                max(
                    0.0,
                    *tuple(
                        sum(
                            adjustment.value
                            for adjustment in self._rank_adjustments
                            if adjustment.rank_dimension == dimension
                            and adjustment.target_identity
                            == value.candidate.provenance.identity
                        )
                        for value in factor.values
                    ),
                )
                for factor in remaining_factors
            )
            for dimension in self._profile.decision_rank_dimensions
        )
        return _DecisionBound(
            authority_upper,
            rank_upper,
            severity_lower,
            None,
            None,
        )

    def compare_records(
        self,
        left: _DecisionRecord,
        right: _DecisionRecord,
    ) -> _DecisionRelation:
        authority = _authority_relation(left.authority, right.authority)
        if authority in (
            _DecisionRelation.DOMINATES,
            _DecisionRelation.DOMINATED,
        ):
            return authority
        if left.preference.rank_values > right.preference.rank_values:
            return _DecisionRelation.DOMINATES
        if left.preference.rank_values < right.preference.rank_values:
            return _DecisionRelation.DOMINATED
        if (
            left.preference.assumption_severity
            < right.preference.assumption_severity
        ):
            return _DecisionRelation.DOMINATES
        if (
            left.preference.assumption_severity
            > right.preference.assumption_severity
        ):
            return _DecisionRelation.DOMINATED
        if (
            authority is _DecisionRelation.EQUIVALENT
            and left.equivalence == right.equivalence
        ):
            return _DecisionRelation.EQUIVALENT
        return _DecisionRelation.INCOMPARABLE

    def record_strictly_beats_bound(
        self,
        record: _DecisionRecord,
        bound: _DecisionBound,
    ) -> bool:
        if not record.certifiable:
            return False
        authority = _authority_relation(record.authority, bound.authority_upper)
        if authority is _DecisionRelation.DOMINATES:
            return True
        if authority is not _DecisionRelation.EQUIVALENT:
            return False
        if record.preference.rank_values > bound.rank_upper:
            return True
        if record.preference.rank_values < bound.rank_upper:
            return False
        if (
            record.preference.assumption_severity
            < bound.assumption_severity_lower
        ):
            return True
        if (
            record.preference.assumption_severity
            > bound.assumption_severity_lower
        ):
            return False
        return (
            bound.equivalence == record.equivalence
            and bound.canonical_identity_lower_bound is not None
            and record.state_identity.digest
            < bound.canonical_identity_lower_bound
        )

    def aggregate_bounds(
        self,
        bounds: Tuple[_DecisionBound, ...],
    ) -> Optional[_DecisionBound]:
        if not bounds:
            return None
        authority_axes = tuple(
            axis for axis, _ in bounds[0].authority_upper.coordinates
        )
        if any(
            tuple(axis for axis, _ in bound.authority_upper.coordinates)
            != authority_axes
            or len(bound.rank_upper) != len(bounds[0].rank_upper)
            or len(bound.assumption_severity_lower)
            != len(bounds[0].assumption_severity_lower)
            for bound in bounds[1:]
        ):
            return None
        authority = _AuthorityVector(
            tuple(
                (
                    axis,
                    max(
                        dict(bound.authority_upper.coordinates)[axis]
                        for bound in bounds
                    ),
                )
                for axis in authority_axes
            )
        )
        equivalence = bounds[0].equivalence
        if equivalence is None or any(
            bound.equivalence != equivalence for bound in bounds[1:]
        ):
            equivalence = None
        canonical_ids = tuple(
            bound.canonical_identity_lower_bound for bound in bounds
        )
        return _DecisionBound(
            authority,
            tuple(max(values) for values in zip(*(b.rank_upper for b in bounds))),
            tuple(
                min(values)
                for values in zip(
                    *(b.assumption_severity_lower for b in bounds)
                )
            ),
            equivalence,
            (
                min(identity for identity in canonical_ids if identity is not None)
                if all(identity is not None for identity in canonical_ids)
                else None
            ),
        )

    def bound_order_key(self, bound: _DecisionBound) -> Tuple[str, ...]:
        authority = tuple(
            f"{9 - value:02d}"
            for _, value in bound.authority_upper.coordinates
        )
        ranks = tuple(
            f"{((1 << 64) - 1) - _sortable_float(value):016x}"
            for value in bound.rank_upper
        )
        severity = tuple(
            f"{value:012d}" for value in bound.assumption_severity_lower
        )
        return authority + ranks + severity

    def encode_bound(self, bound: _DecisionBound) -> Tuple[str, ...]:
        payload = {
            "authority": bound.authority_upper.coordinates,
            "rank": bound.rank_upper,
            "assumption": bound.assumption_severity_lower,
            "equivalence": (
                None
                if bound.equivalence is None
                else (
                    bound.equivalence.net_charge,
                    bound.equivalence.oxidation_state_class,
                    bound.equivalence.molecular_two_s,
                )
            ),
            "canonical": bound.canonical_identity_lower_bound,
        }
        return (
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    def decode_bound(
        self,
        bound: FrontierBound,
    ) -> Optional[_DecisionBound]:
        if (
            bound.schema_version != self.frontier_schema_version
            or len(bound.optimistic_key) != 1
        ):
            return None
        try:
            payload = json.loads(bound.optimistic_key[0])
        except json.JSONDecodeError:
            return None
        expected = {
            "authority",
            "rank",
            "assumption",
            "equivalence",
            "canonical",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            return None
        equivalence_payload = payload["equivalence"]
        equivalence = (
            None
            if equivalence_payload is None
            else _DecisionEquivalenceKey(*equivalence_payload)
        )
        return _DecisionBound(
            _AuthorityVector(
                tuple(
                    (str(axis), int(value))
                    for axis, value in payload["authority"]
                )
            ),
            tuple(float(value) for value in payload["rank"]),
            tuple(int(value) for value in payload["assumption"]),
            equivalence,
            payload["canonical"],
        )
