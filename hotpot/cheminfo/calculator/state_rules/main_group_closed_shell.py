"""OPTIONAL-001 supported main-group closed-shell prior."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Dict, Tuple

from ..charge.contracts import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
)
from ..contracts import (
    CandidateProvenance,
    ScientificQuantity,
    StateValueAuthority,
)
from ..electronic_state.contracts import (
    SpinContributionCandidate,
    SpinContributionScope,
)
from ..state_composition.constraints import _molecular_subject_identity
from ..state_composition.contracts import (
    CandidateConstraint,
    CandidateDomainScope,
    CandidateDomainSummary,
    CandidateProposal,
    ConstraintGuard,
    FactorKey,
    GuardedCandidateConstraint,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
    StateCandidate,
)
from .base import InferenceRule
from .contracts import (
    EvidenceItem,
    ResultType,
    RuleApplicability,
    RuleContext,
    RuleKind,
    RuleOutcome,
    RulePhase,
    RuleSpecification,
    RuleValue,
)

__all__ = ["MainGroupClosedShellRule"]


_RULE_ID = "OPTIONAL-001"
_RULE_VERSION = "1.0.0"
_REGISTRY_SCHEMA = "hotpot-main-group-closed-shell-registry-v1"
_REGISTRY_RESOURCE = "data/rules/main_group_closed_shell_v1.json"
_DOMAIN_DEFINITION = "supported-main-group-closed-shell-v1"
_CONSTRAINT_SCHEMA = "candidate-constraint-v1"
_ASSUMPTION_ID = "domain_template:optional-001"


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _ShellDomain:
    atomic_number: int
    allowed_electrons: Tuple[int, ...]


@dataclass(frozen=True)
class _ClosedShellRegistry:
    version: str
    source_identity: str
    source_locator: str
    content_digest: str
    shell_domains: Tuple[_ShellDomain, ...]
    excluded_neutral_homonuclear_diatomics: frozenset[int]


@lru_cache(maxsize=1)
def _closed_shell_registry() -> _ClosedShellRegistry:
    resource = resources.files(__package__).joinpath(_REGISTRY_RESOURCE)
    encoded = resource.read_bytes()
    payload = json.loads(encoded.decode("utf-8"))
    expected = {
        "schema_version",
        "registry_version",
        "source_identity",
        "source_locator",
        "shell_domains",
        "excluded_neutral_homonuclear_diatomics",
    }
    if set(payload) != expected or payload["schema_version"] != _REGISTRY_SCHEMA:
        raise ValueError("Closed-shell registry does not match its schema")
    shell_domains = tuple(
        _ShellDomain(
            item["atomic_number"],
            tuple(item["allowed_electrons"]),
        )
        for item in payload["shell_domains"]
    )
    atomic_numbers = tuple(domain.atomic_number for domain in shell_domains)
    excluded = frozenset(
        payload["excluded_neutral_homonuclear_diatomics"]
    )
    if (
        len(set(atomic_numbers)) != len(atomic_numbers)
        or any(
            type(domain.atomic_number) is not int
            or not domain.allowed_electrons
            or any(
                type(value) is not int or value <= 0
                for value in domain.allowed_electrons
            )
            for domain in shell_domains
        )
        or any(type(atomic_number) is not int for atomic_number in excluded)
    ):
        raise ValueError("Closed-shell registry contains invalid records")
    return _ClosedShellRegistry(
        version=payload["registry_version"],
        source_identity=payload["source_identity"],
        source_locator=payload["source_locator"],
        content_digest=hashlib.sha256(encoded).hexdigest(),
        shell_domains=shell_domains,
        excluded_neutral_homonuclear_diatomics=excluded,
    )


def _candidate_index(
    context: RuleContext,
) -> Dict[str, Tuple[FactorKey, StateCandidate]]:
    if context.candidate_space is None:
        return {}
    return {
        value.candidate.provenance.identity: (
            factor.factor_key,
            value.candidate,
        )
        for factor in context.candidate_space.factors
        for value in factor.values
    }


def _shell_electron_counts(
    context: RuleContext,
    candidate: LewisFormalChargeCandidate,
) -> Tuple[float, ...]:
    bond_valences = [0.0] * len(context.facts.atomic_numbers)
    covalent_bonds = tuple(
        bond
        for bond in context.facts.semantic_bonds
        if bond.kind == "covalent"
    )
    for bond, order in zip(covalent_bonds, candidate.bond_orders):
        begin, end = bond.atom_indices
        bond_valences[begin] += order
        bond_valences[end] += order
    for index, atomic_number in enumerate(context.facts.atomic_numbers):
        if atomic_number == 1:
            continue
        explicit_count = context.facts.hydrogen_inventory.explicit_counts[index]
        implicit_count = context.facts.hydrogen_inventory.implicit_counts[index]
        if explicit_count == 0 and implicit_count is not None:
            bond_valences[index] += implicit_count
    return tuple(
        nonbonding + 2.0 * bond_valence
        for nonbonding, bond_valence in zip(
            candidate.nonbonding_electrons,
            bond_valences,
        )
    )


def _excluded_neutral_homonuclear_diatomic(
    context: RuleContext,
    charge: NetChargeCandidateEvidence,
) -> bool:
    atomic_numbers = context.facts.atomic_numbers
    return (
        charge.charge == 0
        and len(atomic_numbers) == 2
        and atomic_numbers[0] == atomic_numbers[1]
        and atomic_numbers[0]
        in _closed_shell_registry().excluded_neutral_homonuclear_diatomics
        and len(context.facts.covalent_components) == 1
    )


def _in_closed_shell_domain(
    context: RuleContext,
    charge: NetChargeCandidateEvidence,
    candidate: LewisFormalChargeCandidate,
) -> bool:
    registry = _closed_shell_registry()
    allowed_shells = {
        domain.atomic_number: domain.allowed_electrons
        for domain in registry.shell_domains
    }
    if (
        len(context.facts.covalent_components) != 1
        or any(bond.kind == "coordination" for bond in context.facts.semantic_bonds)
        or any(
            atomic_number not in allowed_shells
            for atomic_number in context.facts.atomic_numbers
        )
        or any(candidate.radical_electrons)
        or (context.facts.nuclear_charge_sum - charge.charge) % 2
        or _excluded_neutral_homonuclear_diatomic(context, charge)
    ):
        return False
    return all(
        any(math.isclose(shell_count, allowed) for allowed in allowed_shells[z])
        for z, shell_count in zip(
            context.facts.atomic_numbers,
            _shell_electron_counts(context, candidate),
        )
    )


def _matching_candidates(
    context: RuleContext,
) -> Tuple[
    Tuple[
        FactorKey,
        FactorKey,
        LewisFormalChargeCandidate,
        NetChargeCandidateEvidence,
    ],
    ...,
]:
    if context.candidate_space is None:
        return ()
    indexed = _candidate_index(context)
    matches = []
    for factor in context.candidate_space.factors:
        if factor.factor_key.quantity is not ScientificQuantity.LEWIS_FORMAL_CHARGE:
            continue
        for value in factor.values:
            candidate = value.candidate
            if not isinstance(candidate, LewisFormalChargeCandidate):
                continue
            parent = indexed.get(candidate.net_charge_candidate_identity)
            if parent is None or not isinstance(
                parent[1],
                NetChargeCandidateEvidence,
            ):
                continue
            charge_factor, charge = parent
            if _in_closed_shell_domain(context, charge, candidate):
                matches.append(
                    (factor.factor_key, charge_factor, candidate, charge)
                )
    return tuple(matches)


def _domain_summary(
    context: RuleContext,
    omitted_count: int,
) -> CandidateDomainSummary:
    return CandidateDomainSummary(
        remaining_count_upper_bound=omitted_count,
        continuation_identity=_digest(
            {
                "rule": _RULE_ID,
                "facts": context.facts.identity.chemical_identity_digest,
                "offset": (
                    context.candidate_budget.max_materialized_values_per_factor
                ),
            }
        ),
        frontier_bound=None,
    )


class MainGroupClosedShellRule(InferenceRule):
    """Propose molecular singlet contributions in the reviewed domain."""

    specification = RuleSpecification(
        rule_id=_RULE_ID,
        rule_version=_RULE_VERSION,
        kind=RuleKind.EMPIRICAL,
        phase=RulePhase.ELECTRONIC_STATE_CANDIDATE,
        requires=(),
        consumes=(
            ResultType.NET_CHARGE_RESULT,
            ResultType.LEWIS_FORMAL_CHARGE_RESULT,
        ),
        produces=(
            ResultType.SPIN_CONTRIBUTION,
            ResultType.CANDIDATE_PROPOSAL,
            ResultType.EVIDENCE,
        ),
        applicability_schema=_DOMAIN_DEFINITION,
    )

    def applicable(self, context: RuleContext) -> RuleApplicability:
        matches = _matching_candidates(context)
        return RuleApplicability(
            bool(matches),
            () if matches else ("CLOSED_SHELL_DOMAIN_UNAVAILABLE",),
        )

    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        if configuration:
            raise ValueError("OPTIONAL-001 does not accept configuration")
        registry = _closed_shell_registry()
        matches = _matching_candidates(context)
        spin_factor = FactorKey(
            ScientificQuantity.ELECTRONIC_STATE,
            _molecular_subject_identity(context.facts),
        )
        values = []
        constraints = []
        for lewis_factor, charge_factor, lewis, charge in matches:
            identity = _digest(
                {
                    "rule": _RULE_ID,
                    "version": _RULE_VERSION,
                    "registry": registry.content_digest,
                    "facts": context.facts.identity.chemical_identity_digest,
                    "lewis": lewis.provenance.identity,
                    "charge": charge.provenance.identity,
                }
            )
            contribution = SpinContributionCandidate(
                provenance=CandidateProvenance(
                    identity=identity,
                    authority=StateValueAuthority.DOMAIN_RULE,
                    source_identity=registry.source_identity,
                    evidence_ids=(
                        registry.content_digest,
                        lewis.provenance.identity,
                        charge.provenance.identity,
                    ),
                    assumption_ids=(_ASSUMPTION_ID,),
                ),
                scope=SpinContributionScope.MOLECULAR,
                subject_identity=spin_factor.subject_identity,
                parent_candidate_ids=(
                    lewis.provenance.identity,
                    charge.provenance.identity,
                ),
                two_s=0,
                pure_spin=True,
            )
            values.append(ProposedCandidateValue(contribution))
            constraint = CandidateConstraint(
                identity=_digest(
                    {
                        "spin": identity,
                        "lewis": lewis.provenance.identity,
                        "charge": charge.provenance.identity,
                    }
                ),
                constraint_type="requires_all",
                schema_version=_CONSTRAINT_SCHEMA,
                factor_keys=(spin_factor, lewis_factor, charge_factor),
                parameters=(
                    (
                        "candidate_ids",
                        (
                            identity,
                            lewis.provenance.identity,
                            charge.provenance.identity,
                        ),
                    ),
                ),
            )
            constraints.append(
                GuardedCandidateConstraint(
                    ConstraintGuard(all_candidate_ids=(identity,)),
                    constraint,
                )
            )
        ordered = tuple(
            sorted(values, key=lambda value: value.candidate.provenance.identity)
        )
        limit = context.candidate_budget.max_materialized_values_per_factor
        retained = ordered[:limit]
        retained_ids = {
            value.candidate.provenance.identity for value in retained
        }
        omitted = len(ordered) - len(retained)
        proposal = CandidateProposal(
            proposal_id=_digest(
                {
                    "rule": _RULE_ID,
                    "version": _RULE_VERSION,
                    "facts": context.facts.identity.chemical_identity_digest,
                    "parents": tuple(
                        sorted(
                            lewis.provenance.identity
                            for _, _, lewis, _ in matches
                        )
                    ),
                }
            ),
            factors=(
                ProposedCandidateFactor(
                    factor_key=spin_factor,
                    domain_definition_id=(
                        f"{_DOMAIN_DEFINITION}:{registry.content_digest}"
                    ),
                    materialized_candidates=retained,
                    domain_scope=CandidateDomainScope.DOMAIN_BOUNDED,
                    materialization_status=(
                        MaterializationStatus.RESOURCE_TRUNCATED
                        if omitted
                        else MaterializationStatus.COMPLETE
                    ),
                    unexpanded_domain_summary=(
                        _domain_summary(context, omitted) if omitted else None
                    ),
                ),
            ),
            constraints=tuple(
                constraint
                for constraint in constraints
                if constraint.guard.all_candidate_ids[0] in retained_ids
            ),
        )
        evidence = EvidenceItem(
            identity=registry.content_digest,
            authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            source_identity=registry.source_identity,
            claim="Reviewed ordinary main-group closed-shell domain",
            locator=registry.source_locator,
        )
        return RuleOutcome(
            rule_id=_RULE_ID,
            rule_version=_RULE_VERSION,
            applicability=RuleApplicability(True),
            candidate_proposal=proposal,
            evidence=(evidence,),
            rank_adjustments=(),
            assumptions=(_ASSUMPTION_ID,),
            warnings=(),
        )
