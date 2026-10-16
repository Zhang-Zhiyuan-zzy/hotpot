"""Budgeted expansion and hard validation of coupled scientific paths."""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple, Union

from ..charge.contracts import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
)
from ..contracts import (
    CandidateProvenance,
    CoverageStatus,
    MolecularFacts,
    ScientificAdmissibility,
    ScientificQuantity,
    StateValueAuthority,
)
from ..electronic_state.contracts import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
    SpinContributionScope,
)
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    OxidationStateCandidateEvidence,
)
from ._artifacts import _FRONTIER_SCHEMA
from .constraints import (
    _constraint_candidate_ids,
    _molecular_subject_identity,
    _unconditional_constraint,
    _validate_coupled_candidate_sets,
)
from .coupling import (
    _coupled_two_s_values,
    _local_spin_subjects_form_partition,
)
from .derive import _derive_charge_values
from .decision_search import (
    _DecisionBound,
    _DecisionSearchCompiler,
    _record_beats_frontier,
    _select_decision_record,
)
from .contracts import (
    BranchAssignment,
    CandidateConstraint,
    CandidateDomainScope,
    CandidateFactor,
    CandidateSearchStatus,
    CoupledScientificStateCandidate,
    DerivedStateValue,
    FactorCandidateValue,
    FactorKey,
    FrontierBound,
    GuardedCandidateConstraint,
    MaterializationStatus,
    RejectedPathSummary,
    ScientificStateIdentity,
    StateValueOrigin,
    StateValueRef,
)
from .validation import _candidate_matches_factor

__all__ = []


_HARD_CONSTRAINT_ID = "hard-scientific-consistency-v1"
_UNBOUNDED_FRONTIER_KEY = ("unbounded",)
_MOLECULAR_SPIN_COUPLING_SOURCE = (
    "state-composition:molecular-spin-coupling-v1"
)
_PARTIAL_OXIDATION_MERGE_SOURCE = (
    "state-composition:partial-oxidation-merge-v1"
)


_ChargeValue = Union[NetChargeCandidateEvidence, DerivedStateValue]


@dataclass(frozen=True)
class _SelectedValues:
    charge: _ChargeValue
    lewis: Tuple[LewisFormalChargeCandidate, ...]
    counting: Tuple[CoordinationCountingCandidate, ...]
    oxidation_state: Optional[OxidationStateCandidateEvidence]
    spin_contributions: Tuple[SpinContributionCandidate, ...]
    electronic_state: Optional[ElectronicStateCandidate]
    derived_values: Tuple[DerivedStateValue, ...]
    candidate_ids: Tuple[str, ...]
    branch_assignments: Tuple[BranchAssignment, ...]


@dataclass(frozen=True)
class _SearchNode:
    selected_values: Tuple[FactorCandidateValue, ...]
    bound: Optional[_DecisionBound]
    branch_identity: str
    queue_key: Tuple[str, ...]


@dataclass(frozen=True)
class _FrontierEntry:
    serialized: FrontierBound
    decision_bound: Optional[_DecisionBound]


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_factors(factors: Tuple[CandidateFactor, ...]) -> None:
    factor_keys = tuple(factor.factor_key for factor in factors)
    if len(set(factor_keys)) != len(factor_keys):
        raise ValueError("Candidate factor keys must be unique after merging")

    candidate_factor_by_identity: Dict[str, FactorKey] = {}
    for factor in factors:
        for value in factor.values:
            candidate = value.candidate
            if not _candidate_matches_factor(factor.factor_key, candidate):
                raise ValueError(
                    "Candidate type does not match its scientific factor"
                )
            identity = candidate.provenance.identity
            incumbent = candidate_factor_by_identity.get(identity)
            if incumbent is not None and incumbent != factor.factor_key:
                raise ValueError(
                    "One candidate identity cannot belong to multiple factors"
                )
            candidate_factor_by_identity[identity] = factor.factor_key


def _prune_factor_domains(
    factors: Tuple[CandidateFactor, ...],
    guarded_constraints: Tuple[GuardedCandidateConstraint, ...],
) -> Tuple[
    Tuple[Tuple[FactorCandidateValue, ...], ...],
    Tuple[RejectedPathSummary, ...],
]:
    domains = [factor.values for factor in factors]
    candidate_locations = {
        value.candidate.provenance.identity: factor_index
        for factor_index, factor in enumerate(factors)
        for value in factor.values
    }
    required_by_factor: Dict[int, str] = {}
    excluded_singletons = set()
    early_rejections = []
    for guarded in guarded_constraints:
        if not _unconditional_constraint(guarded):
            continue
        constraint = guarded.constraint
        candidate_ids = _constraint_candidate_ids(constraint)
        if constraint.constraint_type == "requires_all":
            for candidate_id in candidate_ids:
                factor_index = candidate_locations.get(candidate_id)
                if factor_index is None:
                    early_rejections.append(
                        RejectedPathSummary(
                            (candidate_id,),
                            constraint.identity,
                            ("CONSTRAINT.REQUIRES_ALL",),
                        )
                    )
                    return (
                        tuple(() for _ in factors),
                        tuple(early_rejections),
                    )
                incumbent = required_by_factor.get(factor_index)
                if incumbent is not None and incumbent != candidate_id:
                    early_rejections.append(
                        RejectedPathSummary(
                            (incumbent, candidate_id),
                            constraint.identity,
                            ("CONSTRAINT.REQUIRES_ALL",),
                        )
                    )
                    return (
                        tuple(() for _ in factors),
                        tuple(early_rejections),
                    )
                required_by_factor[factor_index] = candidate_id
        elif constraint.constraint_type == "excludes_all" and len(
            candidate_ids
        ) == 1:
            excluded_singletons.add(candidate_ids[0])

    for factor_index, required_id in required_by_factor.items():
        domains[factor_index] = tuple(
            value
            for value in domains[factor_index]
            if value.candidate.provenance.identity == required_id
        )
    return (
        tuple(
            tuple(
                value
                for value in domain
                if value.candidate.provenance.identity
                not in excluded_singletons
            )
            for domain in domains
        ),
        tuple(early_rejections),
    )


def _branch_conflict(
    selected_values: Tuple[FactorCandidateValue, ...],
) -> bool:
    selected_options: Dict[str, str] = {}
    for value in selected_values:
        for assignment in value.branch_assignments:
            incumbent = selected_options.get(assignment.group_id)
            if incumbent is not None and incumbent != assignment.option_id:
                return True
            selected_options[assignment.group_id] = assignment.option_id
    return False


def _guard_matches(
    guarded: GuardedCandidateConstraint,
    selected_ids: Tuple[str, ...],
    selected_assignments: Tuple[BranchAssignment, ...],
) -> bool:
    return (
        set(guarded.guard.all_candidate_ids).issubset(selected_ids)
        and set(guarded.guard.branch_assignments).issubset(
            selected_assignments
        )
    )


def _constraint_rejection(
    constraint: CandidateConstraint,
    selected_ids: Tuple[str, ...],
) -> Optional[str]:
    candidate_ids = set(_constraint_candidate_ids(constraint))
    selected = set(selected_ids)
    if constraint.constraint_type == "requires_all":
        return (
            None
            if candidate_ids.issubset(selected)
            else "CONSTRAINT.REQUIRES_ALL"
        )
    if constraint.constraint_type == "excludes_all":
        return (
            "CONSTRAINT.EXCLUDES_ALL"
            if candidate_ids.issubset(selected)
            else None
        )
    return "CONSTRAINT.UNSUPPORTED_TYPE"


def _active_constraint_rejection(
    guarded_constraints: Tuple[GuardedCandidateConstraint, ...],
    selected_ids: Tuple[str, ...],
    selected_assignments: Tuple[BranchAssignment, ...],
) -> Optional[Tuple[str, str]]:
    for guarded in guarded_constraints:
        if not _guard_matches(
            guarded,
            selected_ids,
            selected_assignments,
        ):
            continue
        reason = _constraint_rejection(guarded.constraint, selected_ids)
        if reason is not None:
            return guarded.constraint.identity, reason
    return None


def _prefix_constraint_rejection(
    guarded_constraints: Tuple[GuardedCandidateConstraint, ...],
    selected_values: Tuple[FactorCandidateValue, ...],
    candidate_locations: Dict[str, int],
) -> Optional[Tuple[str, str]]:
    selected_ids = _candidate_ids(selected_values)
    selected_assignments = tuple(
        assignment
        for value in selected_values
        for assignment in value.branch_assignments
    )
    selected_depth = len(selected_values)
    for guarded in guarded_constraints:
        if not _guard_matches(
            guarded,
            selected_ids,
            selected_assignments,
        ):
            continue
        constraint = guarded.constraint
        candidate_ids = _constraint_candidate_ids(constraint)
        if constraint.constraint_type == "excludes_all":
            if set(candidate_ids).issubset(selected_ids):
                return constraint.identity, "CONSTRAINT.EXCLUDES_ALL"
            continue
        if constraint.constraint_type != "requires_all":
            continue
        for candidate_id in candidate_ids:
            factor_index = candidate_locations.get(candidate_id)
            if factor_index is None or (
                factor_index < selected_depth
                and candidate_id not in selected_ids
            ):
                return constraint.identity, "CONSTRAINT.REQUIRES_ALL"
    return None


def _merge_oxidation_states(
    facts: MolecularFacts,
    candidates: Tuple[OxidationStateCandidateEvidence, ...],
) -> Tuple[Optional[OxidationStateCandidateEvidence], Optional[str]]:
    if not candidates:
        return None, None
    if len(candidates) == 1:
        return candidates[0], None
    conventions = {candidate.convention for candidate in candidates}
    if len(conventions) != 1:
        return None, "HARD-005.COUNTING_CONVENTION_MISMATCH"
    atom_count = len(facts.atomic_numbers)
    if any(len(candidate.atom_states) != atom_count for candidate in candidates):
        return None, "HARD-004.OS_COVERAGE_INVALID"
    merged_states = []
    for atom_index in range(atom_count):
        values = {
            candidate.atom_states[atom_index]
            for candidate in candidates
            if candidate.atom_states[atom_index] is not None
        }
        if len(values) > 1:
            return None, "HARD-004.OS_PARTIAL_CONFLICT"
        merged_states.append(next(iter(values)) if values else None)
    parent_ids = tuple(
        sorted(candidate.provenance.identity for candidate in candidates)
    )
    convention = candidates[0].convention
    identity = _digest(
        {
            "source": _PARTIAL_OXIDATION_MERGE_SOURCE,
            "facts": facts.identity.chemical_identity_digest,
            "parents": parent_ids,
            "convention": convention.value,
            "atom_states": merged_states,
        }
    )
    counting_references = {
        candidate.coordination_counting_candidate_identity
        for candidate in candidates
        if candidate.coordination_counting_candidate_identity is not None
    }
    has_implicit_hydrogen = any(
        explicit_count == 0 and implicit_count not in (None, 0)
        for explicit_count, implicit_count in zip(
            facts.hydrogen_inventory.explicit_counts,
            facts.hydrogen_inventory.implicit_counts,
        )
    )
    declared_complete = any(
        candidate.coverage is CoverageStatus.COMPLETE
        for candidate in candidates
    )
    all_atoms_covered = all(state is not None for state in merged_states)
    coverage = (
        CoverageStatus.COMPLETE
        if declared_complete or (all_atoms_covered and not has_implicit_hydrogen)
        else CoverageStatus.PARTIAL
    )
    return (
        OxidationStateCandidateEvidence(
            provenance=CandidateProvenance(
                identity=identity,
                authority=StateValueAuthority.DERIVED_CONSTRAINT,
                source_identity=_PARTIAL_OXIDATION_MERGE_SOURCE,
                evidence_ids=parent_ids,
                assumption_ids=tuple(
                    sorted(
                        {
                            assumption
                            for candidate in candidates
                            for assumption in candidate.provenance.assumption_ids
                        }
                    )
                ),
            ),
            convention=convention,
            atom_states=tuple(merged_states),
            coverage=coverage,
            coordination_counting_candidate_identity=(
                next(iter(counting_references))
                if len(counting_references) == 1
                else None
            ),
        ),
        None,
    )


def _classify_selected_values(
    facts: MolecularFacts,
    selected_values: Tuple[FactorCandidateValue, ...],
) -> Tuple[Optional[_SelectedValues], Optional[str]]:
    candidates = tuple(value.candidate for value in selected_values)
    result_charges = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, NetChargeCandidateEvidence)
    )
    supplied_derived = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, DerivedStateValue)
        and candidate.quantity is ScientificQuantity.NET_CHARGE
    )
    unsupported_derived = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, DerivedStateValue)
        and candidate.quantity is not ScientificQuantity.NET_CHARGE
    )
    oxidation_states = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, OxidationStateCandidateEvidence)
    )
    electronic_states = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, ElectronicStateCandidate)
    )
    spin_contributions = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, SpinContributionCandidate)
    )
    molecular_spin_contributions = tuple(
        candidate
        for candidate in spin_contributions
        if candidate.scope is SpinContributionScope.MOLECULAR
    )
    if unsupported_derived:
        return None, "COMPOSITION.STRUCTURED_DERIVATION_UNSUPPORTED"
    if len(electronic_states) > 1:
        return None, "COMPOSITION.ELECTRONIC_STATE_CARDINALITY"
    if len(molecular_spin_contributions) > 1:
        return None, "COMPOSITION.MOLECULAR_SPIN_CARDINALITY"
    if molecular_spin_contributions and (
        molecular_spin_contributions[0].subject_identity
        != _molecular_subject_identity(facts)
    ):
        return None, "COMPOSITION.MOLECULAR_SPIN_SUBJECT_MISMATCH"

    base_candidate_ids = tuple(
        candidate.provenance.identity for candidate in candidates
    )
    if any(
        not set(candidate.parent_candidate_ids).issubset(base_candidate_ids)
        for candidate in spin_contributions
    ):
        return None, "COMPOSITION.SPIN_CONTRIBUTION_PARENT_MISSING"
    counting = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, CoordinationCountingCandidate)
    )
    oxidation_state, oxidation_rejection = _merge_oxidation_states(
        facts,
        oxidation_states,
    )
    if oxidation_rejection is not None:
        return None, oxidation_rejection
    computed_derived = _derive_charge_values(
        facts,
        counting,
        oxidation_state,
    )
    derived_by_identity = {
        candidate.provenance.identity: candidate
        for candidate in supplied_derived + computed_derived
    }
    derived_values = tuple(
        derived_by_identity[identity]
        for identity in sorted(derived_by_identity)
    )
    if any(
        not set(candidate.parent_candidate_ids).issubset(base_candidate_ids)
        for candidate in derived_values
    ):
        return None, "COMPOSITION.DERIVED_PARENT_MISSING"
    if len(result_charges) > 1:
        return None, "COMPOSITION.NET_CHARGE_CARDINALITY"
    derived_charge_values = {candidate.value for candidate in derived_values}
    if len(derived_charge_values) > 1:
        return None, "HARD-002.DERIVED_CHARGE_CONFLICT"
    if result_charges and derived_charge_values and (
        result_charges[0].charge not in derived_charge_values
    ):
        return None, "HARD-002.DERIVED_CHARGE_CONFLICT"
    if result_charges:
        charge: _ChargeValue = result_charges[0]
    elif derived_values:
        charge = derived_values[0]
    else:
        return None, "COMPOSITION.NET_CHARGE_CARDINALITY"
    electronic_state = electronic_states[0] if electronic_states else None
    if molecular_spin_contributions:
        molecular_contribution = molecular_spin_contributions[0]
        if electronic_state is not None and (
            electronic_state.two_s != molecular_contribution.two_s
            or electronic_state.pure_spin != molecular_contribution.pure_spin
        ):
            return None, "COMPOSITION.MOLECULAR_SPIN_CONFLICT"
        if electronic_state is None:
            electronic_state = _couple_molecular_spin_contribution(
                facts,
                charge,
                oxidation_state,
                molecular_contribution,
            )
    local_spin_contributions = tuple(
        candidate
        for candidate in spin_contributions
        if candidate.scope is not SpinContributionScope.MOLECULAR
    )
    if (
        electronic_state is not None
        and electronic_state.pure_spin
        and local_spin_contributions
        and all(candidate.pure_spin for candidate in local_spin_contributions)
        and _local_spin_subjects_form_partition(
            facts,
            local_spin_contributions,
        )
        and electronic_state.two_s
        not in _coupled_two_s_values(
            tuple(
                candidate.two_s for candidate in local_spin_contributions
            )
        )
    ):
        return None, "COMPOSITION.LOCAL_SPIN_COUPLING_CONFLICT"
    candidate_ids = base_candidate_ids + tuple(
        candidate.provenance.identity
        for candidate in computed_derived
        if candidate.provenance.identity not in base_candidate_ids
    )
    return (
        _SelectedValues(
            charge=charge,
            lewis=tuple(
                candidate
                for candidate in candidates
                if isinstance(candidate, LewisFormalChargeCandidate)
            ),
            counting=counting,
            oxidation_state=oxidation_state,
            spin_contributions=spin_contributions,
            electronic_state=electronic_state,
            derived_values=derived_values,
            candidate_ids=candidate_ids,
            branch_assignments=tuple(
                assignment
                for value in selected_values
                for assignment in value.branch_assignments
            ),
        ),
        None,
    )


def _charge_value(charge: _ChargeValue) -> int:
    if isinstance(charge, NetChargeCandidateEvidence):
        return charge.charge
    return charge.value


def _couple_molecular_spin_contribution(
    facts: MolecularFacts,
    charge: _ChargeValue,
    oxidation_state: Optional[OxidationStateCandidateEvidence],
    contribution: SpinContributionCandidate,
) -> ElectronicStateCandidate:
    charge_identity = charge.provenance.identity
    oxidation_identity = (
        oxidation_state.provenance.identity
        if oxidation_state is not None
        else None
    )
    identity = _digest(
        {
            "source": _MOLECULAR_SPIN_COUPLING_SOURCE,
            "facts": facts.identity.chemical_identity_digest,
            "charge": charge_identity,
            "oxidation_state": oxidation_identity,
            "contribution": contribution.provenance.identity,
        }
    )
    evidence_ids = tuple(
        sorted(
            {
                charge_identity,
                contribution.provenance.identity,
                *contribution.parent_candidate_ids,
                *((oxidation_identity,) if oxidation_identity else ()),
            }
        )
    )
    return ElectronicStateCandidate(
        provenance=CandidateProvenance(
            identity=identity,
            authority=StateValueAuthority.DERIVED_CONSTRAINT,
            source_identity=_MOLECULAR_SPIN_COUPLING_SOURCE,
            evidence_ids=evidence_ids,
            assumption_ids=contribution.provenance.assumption_ids,
        ),
        net_charge_candidate_identity=charge_identity,
        oxidation_state_candidate_identity=oxidation_identity,
        electron_count=facts.nuclear_charge_sum - _charge_value(charge),
        two_s=contribution.two_s,
        multiplicity=(
            contribution.two_s + 1 if contribution.pure_spin else None
        ),
        pure_spin=contribution.pure_spin,
    )


def _charge_evidence(selected: _SelectedValues) -> NetChargeCandidateEvidence:
    if isinstance(selected.charge, NetChargeCandidateEvidence):
        return selected.charge
    return NetChargeCandidateEvidence(
        selected.charge.provenance,
        selected.charge.value,
    )


def _electronic_result_ids(
    selected: _SelectedValues,
    electronic_result_by_candidate: Dict[str, Tuple[str, ...]],
    unresolved_electronic_result_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    if selected.electronic_state is None:
        return unresolved_electronic_result_ids
    result_ids = electronic_result_by_candidate.get(
        selected.electronic_state.provenance.identity,
        (),
    )
    if result_ids:
        return result_ids
    molecular_contribution_ids = tuple(
        candidate.provenance.identity
        for candidate in selected.spin_contributions
        if candidate.scope is SpinContributionScope.MOLECULAR
    )
    return tuple(
        sorted(
            {
                result_id
                for candidate_id in molecular_contribution_ids
                for result_id in electronic_result_by_candidate.get(
                    candidate_id,
                    (),
                )
            }
        )
    )


def _path_identity(
    facts: MolecularFacts,
    selected: _SelectedValues,
    charge_origin_identity: str,
    electronic_result_id: str,
) -> ScientificStateIdentity:
    return ScientificStateIdentity(
        _digest(
            {
                "facts": facts.identity.chemical_identity_digest,
                "candidate_ids": selected.candidate_ids,
                "charge_origin_identity": charge_origin_identity,
                "electronic_result_id": electronic_result_id,
            }
        )
    )


def _coupled_path(
    facts: MolecularFacts,
    selected: _SelectedValues,
    charge_result_by_candidate: Dict[str, Tuple[str, ...]],
    electronic_result_id: str,
) -> Optional[CoupledScientificStateCandidate]:
    charge = selected.charge
    if isinstance(charge, DerivedStateValue):
        charge_ref = StateValueRef(
            ScientificQuantity.NET_CHARGE,
            charge.value,
            StateValueOrigin.DERIVED_CONSTRAINT,
            charge.provenance.identity,
        )
    else:
        result_ids = charge_result_by_candidate.get(
            charge.provenance.identity,
            (),
        )
        if not result_ids:
            return None
        charge_ref = StateValueRef(
            ScientificQuantity.NET_CHARGE,
            charge.charge,
            StateValueOrigin.RESULT,
            result_ids[0],
        )
    selected_candidates = (
        (
            (selected.charge,)
            if isinstance(selected.charge, NetChargeCandidateEvidence)
            else ()
        )
        + selected.lewis
        + selected.counting
        + ((selected.oxidation_state,) if selected.oxidation_state else ())
        + selected.spin_contributions
        + ((selected.electronic_state,) if selected.electronic_state else ())
        + selected.derived_values
    )
    return CoupledScientificStateCandidate(
        identity=_path_identity(
            facts,
            selected,
            charge_ref.origin_identity,
            electronic_result_id,
        ),
        net_charge=charge_ref,
        component_assignment_ids=tuple(
            candidate.provenance.identity for candidate in selected.lewis
        ),
        coordination_assignment_ids=tuple(
            candidate.provenance.identity for candidate in selected.counting
        ),
        oxidation_state_assignment_id=(
            selected.oxidation_state.provenance.identity
            if selected.oxidation_state is not None
            else None
        ),
        electronic_state_result_id=electronic_result_id,
        electronic_state_candidate_identity=(
            selected.electronic_state.provenance.identity
            if selected.electronic_state is not None
            else None
        ),
        derived_values=selected.derived_values,
        evidence_ids=selected.candidate_ids,
        assumption_ids=tuple(
            sorted(
                {
                    assumption
                    for candidate in selected_candidates
                    for assumption in candidate.provenance.assumption_ids
                }
            )
        ),
    )


def _candidate_ids(
    selected_values: Tuple[FactorCandidateValue, ...],
) -> Tuple[str, ...]:
    return tuple(
        value.candidate.provenance.identity for value in selected_values
    )


def _search_node(
    selected_values: Tuple[FactorCandidateValue, ...],
    factors: Tuple[CandidateFactor, ...],
    decision_search: Optional[_DecisionSearchCompiler],
) -> _SearchNode:
    depth = len(selected_values)
    candidate_ids = _candidate_ids(selected_values)
    branch_identity = _digest(
        {"depth": depth, "candidate_ids": candidate_ids}
    )
    bound = (
        None
        if decision_search is None
        else decision_search.optimistic_bound_for_prefix(
            selected_values,
            factors[depth:],
        )
    )
    policy_key = (
        ("0:unboundable",)
        if bound is None
        else ("1:bounded",) + decision_search.bound_order_key(bound)
    )
    return _SearchNode(
        selected_values=selected_values,
        bound=bound,
        branch_identity=branch_identity,
        queue_key=policy_key + candidate_ids + (branch_identity,),
    )


def _push_search_node(
    heap: list[Tuple[Tuple[str, ...], str, _SearchNode]],
    node: _SearchNode,
) -> None:
    heapq.heappush(heap, (node.queue_key, node.branch_identity, node))


def _frontier_for_node(
    node: _SearchNode,
    decision_search: Optional[_DecisionSearchCompiler],
) -> FrontierBound:
    if node.bound is None or decision_search is None:
        optimistic_key = _UNBOUNDED_FRONTIER_KEY
    else:
        optimistic_key = _encode_decision_bound(decision_search, node.bound)
    return FrontierBound(
        branch_identity=node.branch_identity,
        schema_version=_FRONTIER_SCHEMA,
        optimistic_key=optimistic_key,
    )


def _encode_decision_bound(
    decision_search: _DecisionSearchCompiler,
    bound: _DecisionBound,
) -> Tuple[str, ...]:
    optimistic_key = decision_search.encode_bound(bound)
    if (
        not optimistic_key
        or optimistic_key == _UNBOUNDED_FRONTIER_KEY
        or any(not isinstance(value, str) for value in optimistic_key)
    ):
        raise ValueError(
            "Decision-search bounds must use non-empty typed encodings"
        )
    return optimistic_key


def _decoded_frontier_bound(
    serialized: FrontierBound,
    decision_search: Optional[_DecisionSearchCompiler],
) -> Optional[_DecisionBound]:
    if serialized.schema_version != _FRONTIER_SCHEMA:
        raise ValueError("Candidate frontier uses an unsupported schema")
    if decision_search is None:
        return None
    decoded = decision_search.decode_bound(serialized)
    if decoded is not None and (
        _encode_decision_bound(decision_search, decoded)
        != serialized.optimistic_key
    ):
        raise ValueError("Decoded decision bound does not reproduce its key")
    return decoded


def _aggregate_frontier_entries(
    entries: Tuple[_FrontierEntry, ...],
    decision_search: Optional[_DecisionSearchCompiler],
) -> _FrontierEntry:
    branch_identity = _digest(
        {
            "aggregate": tuple(
                entry.serialized.branch_identity for entry in entries
            )
        }
    )
    decision_bound = None
    if decision_search is not None and all(
        entry.decision_bound is not None for entry in entries
    ):
        decision_bound = decision_search.aggregate_bounds(
            tuple(
                entry.decision_bound
                for entry in entries
                if entry.decision_bound is not None
            )
        )
    optimistic_key = (
        _UNBOUNDED_FRONTIER_KEY
        if decision_bound is None or decision_search is None
        else _encode_decision_bound(decision_search, decision_bound)
    )
    return _FrontierEntry(
        FrontierBound(
            branch_identity=branch_identity,
            schema_version=_FRONTIER_SCHEMA,
            optimistic_key=optimistic_key,
        ),
        decision_bound,
    )


def _compact_frontier_entries(
    entries: Tuple[_FrontierEntry, ...],
    max_frontier_paths: int,
    decision_search: Optional[_DecisionSearchCompiler],
) -> Tuple[_FrontierEntry, ...]:
    if len(entries) <= max_frontier_paths:
        return entries
    explicit_count = max_frontier_paths - 1
    return entries[:explicit_count] + (
        _aggregate_frontier_entries(
            entries[explicit_count:],
            decision_search,
        ),
    )


def _candidate_search_status(
    paths: Tuple[CoupledScientificStateCandidate, ...],
    factors: Tuple[CandidateFactor, ...],
    frontier: Tuple[_FrontierEntry, ...],
    truncated: bool,
    decision_search: Optional[_DecisionSearchCompiler],
) -> CandidateSearchStatus:
    if decision_search is None:
        factors_are_certifiable = all(
            factor.domain_scope is not CandidateDomainScope.OPEN
            and factor.materialization_status
            is MaterializationStatus.COMPLETE
            for factor in factors
        )
        if truncated:
            return CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION
        if (
            len(paths) == 1
            and factors_are_certifiable
            and paths[0].electronic_state_candidate_identity is not None
        ):
            return CandidateSearchStatus.DECISION_CERTIFIED
        return CandidateSearchStatus.NO_CERTIFIED_DECISION

    selected_record = _select_decision_record(
        paths,
        factors,
        decision_search,
    )
    certifiable_factors = all(
        factor.domain_scope is not CandidateDomainScope.OPEN
        for factor in factors
    )
    frontier_certified = (
        selected_record is not None
        and certifiable_factors
        and _record_beats_frontier(
            selected_record,
            tuple(entry.decision_bound for entry in frontier),
            decision_search,
        )
    )
    if truncated:
        return (
            CandidateSearchStatus.TRUNCATED_BUT_DECISION_CERTIFIED
            if frontier_certified
            else CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION
        )
    return (
        CandidateSearchStatus.DECISION_CERTIFIED
        if selected_record is not None and certifiable_factors
        else CandidateSearchStatus.NO_CERTIFIED_DECISION
    )


def _expand_scientific_paths(
    facts: MolecularFacts,
    factors: Tuple[CandidateFactor, ...],
    guarded_constraints: Tuple[GuardedCandidateConstraint, ...],
    charge_result_by_candidate: Dict[str, Tuple[str, ...]],
    electronic_result_by_candidate: Dict[str, Tuple[str, ...]],
    unresolved_electronic_result_ids: Tuple[str, ...],
    *,
    max_expanded_paths: int,
    max_frontier_paths: int,
    decision_search: Optional[_DecisionSearchCompiler] = None,
) -> Tuple[
    Tuple[CoupledScientificStateCandidate, ...],
    Tuple[RejectedPathSummary, ...],
    Tuple[FrontierBound, ...],
    CandidateSearchStatus,
]:
    if (
        decision_search is not None
        and decision_search.frontier_schema_version != _FRONTIER_SCHEMA
    ):
        raise ValueError("Decision-search frontier schema is not supported")
    factors = tuple(
        sorted(
            factors,
            key=lambda factor: (
                factor.factor_key.quantity.value,
                factor.factor_key.subject_identity,
            ),
        )
    )
    _validate_factors(factors)
    factor_keys = {factor.factor_key for factor in factors}
    for guarded in guarded_constraints:
        if not set(guarded.constraint.factor_keys).issubset(factor_keys):
            raise ValueError("Candidate constraint references an unknown factor")

    domains, early_rejections = _prune_factor_domains(
        factors,
        guarded_constraints,
    )
    domains = tuple(
        tuple(
            sorted(
                domain,
                key=lambda value: value.candidate.provenance.identity,
            )
        )
        for domain in domains
    )
    if not factors or any(not domain for domain in domains):
        return (
            (),
            early_rejections,
            (),
            CandidateSearchStatus.NOT_EVALUATED,
        )

    search_factors = tuple(
        replace(factor, values=domain)
        for factor, domain in zip(factors, domains)
    )
    candidate_locations = {
        value.candidate.provenance.identity: factor_index
        for factor_index, factor in enumerate(search_factors)
        for value in factor.values
    }
    search_heap: list[Tuple[Tuple[str, ...], str, _SearchNode]] = []
    _push_search_node(
        search_heap,
        _search_node((), search_factors, decision_search),
    )
    paths: Dict[str, CoupledScientificStateCandidate] = {}
    rejections = list(early_rejections)
    attempted = 0
    while search_heap and attempted < max_expanded_paths:
        _, _, node = heapq.heappop(search_heap)
        depth = len(node.selected_values)
        if depth < len(domains):
            for value in domains[depth]:
                selected_values = node.selected_values + (value,)
                prefix_rejection = _prefix_constraint_rejection(
                    guarded_constraints,
                    selected_values,
                    candidate_locations,
                )
                if prefix_rejection is not None:
                    constraint_identity, reason = prefix_rejection
                    rejections.append(
                        RejectedPathSummary(
                            _candidate_ids(selected_values),
                            constraint_identity,
                            (reason,),
                        )
                    )
                    continue
                _push_search_node(
                    search_heap,
                    _search_node(
                        selected_values,
                        search_factors,
                        decision_search,
                    ),
                )
            continue

        selected_values = node.selected_values
        attempted += 1
        selected, classification_rejection = _classify_selected_values(
            facts,
            selected_values,
        )
        selected_ids = _candidate_ids(selected_values)
        if classification_rejection is not None:
            rejections.append(
                RejectedPathSummary(
                    selected_ids,
                    _HARD_CONSTRAINT_ID,
                    (classification_rejection,),
                )
            )
            continue
        if _branch_conflict(selected_values):
            rejections.append(
                RejectedPathSummary(
                    selected_ids,
                    "branch-assignment-v1",
                    ("COMPOSITION.BRANCH_CONFLICT",),
                )
            )
            continue
        constraint_rejection = _active_constraint_rejection(
            guarded_constraints,
            selected.candidate_ids,
            selected.branch_assignments,
        )
        if constraint_rejection is not None:
            constraint_identity, reason = constraint_rejection
            rejections.append(
                RejectedPathSummary(
                    selected.candidate_ids,
                    constraint_identity,
                    (reason,),
                )
            )
            continue
        electronic_result_ids = _electronic_result_ids(
            selected,
            electronic_result_by_candidate,
            unresolved_electronic_result_ids,
        )
        if not electronic_result_ids:
            rejections.append(
                RejectedPathSummary(
                    selected.candidate_ids,
                    _HARD_CONSTRAINT_ID,
                    ("COMPOSITION.ELECTRONIC_RESULT_ENVELOPE_MISSING",),
                )
            )
            continue
        charge_evidence = _charge_evidence(selected)
        assessment = _validate_coupled_candidate_sets(
            facts,
            charge_evidence,
            lewis_candidates=selected.lewis,
            counting_candidates=selected.counting,
            oxidation_state=selected.oxidation_state,
            electronic_state=selected.electronic_state,
        )
        if assessment.admissibility is not ScientificAdmissibility.ADMISSIBLE:
            rejections.append(
                RejectedPathSummary(
                    selected.candidate_ids,
                    _HARD_CONSTRAINT_ID,
                    tuple(finding.code for finding in assessment.findings),
                )
            )
            continue
        for electronic_result_id in electronic_result_ids:
            path = _coupled_path(
                facts,
                selected,
                charge_result_by_candidate,
                electronic_result_id,
            )
            if path is None:
                rejections.append(
                    RejectedPathSummary(
                        selected.candidate_ids,
                        _HARD_CONSTRAINT_ID,
                        ("COMPOSITION.CHARGE_RESULT_ENVELOPE_MISSING",),
                    )
                )
                continue
            paths[path.identity.digest] = path

    factor_truncation = any(
        factor.materialization_status
        is MaterializationStatus.RESOURCE_TRUNCATED
        for factor in factors
    )
    truncated = bool(search_heap) or factor_truncation
    frontier_entries = [
        _FrontierEntry(
            _frontier_for_node(node, decision_search),
            node.bound,
        )
        for _, _, node in sorted(search_heap)
    ]
    for factor in factors:
        summary = factor.unexpanded_domain_summary
        if summary is None:
            continue
        serialized_bound = summary.frontier_bound or FrontierBound(
            branch_identity=summary.continuation_identity or factor.identity,
            schema_version=_FRONTIER_SCHEMA,
            optimistic_key=_UNBOUNDED_FRONTIER_KEY,
        )
        decision_bound = (
            _decoded_frontier_bound(serialized_bound, decision_search)
            if summary.frontier_bound is not None
            else None
        )
        frontier_entries.append(
            _FrontierEntry(serialized_bound, decision_bound)
        )
    compact_frontier = (
        _compact_frontier_entries(
            tuple(frontier_entries),
            max_frontier_paths,
            decision_search,
        )
        if max_frontier_paths > 0
        else ()
    )
    frontier = tuple(entry.serialized for entry in compact_frontier)

    ordered_paths = tuple(paths[identity] for identity in sorted(paths))
    status = _candidate_search_status(
        ordered_paths,
        factors,
        compact_frontier,
        truncated,
        decision_search,
    )
    return ordered_paths, tuple(rejections), frontier, status
