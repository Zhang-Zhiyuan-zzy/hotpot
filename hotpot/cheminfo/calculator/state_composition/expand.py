"""Budgeted expansion and hard validation of coupled scientific paths."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple, Union

from ..charge.contracts import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
)
from ..contracts import (
    MolecularFacts,
    ScientificAdmissibility,
    ScientificQuantity,
)
from ..electronic_state.contracts import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
)
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    OxidationStateCandidateEvidence,
)
from ._artifacts import _FRONTIER_SCHEMA
from .constraints import (
    _constraint_candidate_ids,
    _unconditional_constraint,
    _validate_coupled_candidate_sets,
)
from .derive import _derive_charge_values
from .decision_search import _DecisionSearchCompiler
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
    StateCandidate,
    StateValueOrigin,
    StateValueRef,
)

__all__ = []


_HARD_CONSTRAINT_ID = "hard-scientific-consistency-v1"
_UNBOUNDED_FRONTIER_KEY = ("unbounded",)


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


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_matches_factor(
    factor_key: FactorKey,
    candidate: StateCandidate,
) -> bool:
    if isinstance(candidate, DerivedStateValue):
        return candidate.quantity is factor_key.quantity
    candidate_types = {
        ScientificQuantity.NET_CHARGE: (NetChargeCandidateEvidence,),
        ScientificQuantity.LEWIS_FORMAL_CHARGE: (
            LewisFormalChargeCandidate,
        ),
        ScientificQuantity.COORDINATION_COUNTING: (
            CoordinationCountingCandidate,
        ),
        ScientificQuantity.OXIDATION_STATE: (
            OxidationStateCandidateEvidence,
        ),
        ScientificQuantity.ELECTRONIC_STATE: (
            SpinContributionCandidate,
            ElectronicStateCandidate,
        ),
    }
    return isinstance(candidate, candidate_types[factor_key.quantity])


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
    if unsupported_derived:
        return None, "COMPOSITION.STRUCTURED_DERIVATION_UNSUPPORTED"
    if len(oxidation_states) > 1:
        return None, "COMPOSITION.OXIDATION_STATE_CARDINALITY"
    if len(electronic_states) > 1:
        return None, "COMPOSITION.ELECTRONIC_STATE_CARDINALITY"

    base_candidate_ids = tuple(
        candidate.provenance.identity for candidate in candidates
    )
    counting = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, CoordinationCountingCandidate)
    )
    oxidation_state = oxidation_states[0] if oxidation_states else None
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
            spin_contributions=tuple(
                candidate
                for candidate in candidates
                if isinstance(candidate, SpinContributionCandidate)
            ),
            electronic_state=(
                electronic_states[0] if electronic_states else None
            ),
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
    return electronic_result_by_candidate.get(
        selected.electronic_state.provenance.identity,
        (),
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


def _iter_selected_values(
    domains: Tuple[Tuple[FactorCandidateValue, ...], ...],
) -> Iterator[Tuple[FactorCandidateValue, ...]]:
    if not domains or any(not domain for domain in domains):
        return
    yield from itertools.product(*domains)


def _frontier_for_selection(
    selected_values: Tuple[FactorCandidateValue, ...],
) -> FrontierBound:
    candidate_ids = tuple(
        value.candidate.provenance.identity for value in selected_values
    )
    return FrontierBound(
        branch_identity=_digest(candidate_ids),
        schema_version=_FRONTIER_SCHEMA,
        optimistic_key=_UNBOUNDED_FRONTIER_KEY,
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
    del decision_search
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
    if not factors or any(not domain for domain in domains):
        return (
            (),
            early_rejections,
            (),
            CandidateSearchStatus.NOT_EVALUATED,
        )

    iterator = _iter_selected_values(domains)
    paths: Dict[str, CoupledScientificStateCandidate] = {}
    rejections = list(early_rejections)
    attempted = 0
    truncated = False
    first_unexpanded = None
    for selected_values in iterator:
        if attempted >= max_expanded_paths:
            truncated = True
            first_unexpanded = selected_values
            break
        attempted += 1
        selected, classification_rejection = _classify_selected_values(
            facts,
            selected_values,
        )
        selected_ids = tuple(
            value.candidate.provenance.identity for value in selected_values
        )
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
    truncated = truncated or factor_truncation
    frontier = []
    if first_unexpanded is not None and max_frontier_paths > 0:
        frontier.append(_frontier_for_selection(first_unexpanded))
        for selected_values in itertools.islice(
            iterator,
            max_frontier_paths - 1,
        ):
            frontier.append(_frontier_for_selection(selected_values))
    for factor in factors:
        if len(frontier) >= max_frontier_paths:
            break
        summary = factor.unexpanded_domain_summary
        if summary is None:
            continue
        frontier.append(
            summary.frontier_bound
            or FrontierBound(
                branch_identity=summary.continuation_identity
                or factor.identity,
                schema_version=_FRONTIER_SCHEMA,
                optimistic_key=_UNBOUNDED_FRONTIER_KEY,
            )
        )

    ordered_paths = tuple(paths[identity] for identity in sorted(paths))
    factors_are_certifiable = all(
        factor.domain_scope is not CandidateDomainScope.OPEN
        and factor.materialization_status is MaterializationStatus.COMPLETE
        for factor in factors
    )
    if truncated:
        status = CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION
    elif (
        len(ordered_paths) == 1
        and factors_are_certifiable
        and ordered_paths[0].electronic_state_candidate_identity is not None
    ):
        status = CandidateSearchStatus.DECISION_CERTIFIED
    else:
        status = CandidateSearchStatus.NO_CERTIFIED_DECISION
    return ordered_paths, tuple(rejections), tuple(frontier), status
