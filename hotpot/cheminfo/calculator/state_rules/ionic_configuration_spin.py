"""OPTIONAL-009 ionic-configuration local spin candidates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Dict, Tuple

from ..contracts import CandidateProvenance, ScientificQuantity, StateValueAuthority
from ..electronic_state.contracts import (
    SpinContributionCandidate,
    SpinContributionScope,
)
from ..oxidation_state.contracts import OxidationStateCandidateEvidence
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

__all__ = ["IonicConfigurationSpinRule"]


_RULE_ID = "OPTIONAL-009"
_RULE_VERSION = "1.0.0"
_REGISTRY_SCHEMA = "hotpot-ionic-configuration-registry-v1"
_REGISTRY_RESOURCE = "data/rules/ionic_configurations_v1.json"
_DOMAIN_DEFINITION = "reviewed-ionic-configuration-spin-v1"
_CONSTRAINT_SCHEMA = "candidate-constraint-v1"


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _IonicConfiguration:
    atomic_number: int
    oxidation_state: int
    shell: str
    occupancy: int

    @property
    def two_s_candidates(self) -> Tuple[int, ...]:
        capacity = 10 if self.shell == "d" else 14
        maximum = min(self.occupancy, capacity - self.occupancy)
        return tuple(range(maximum, -1, -2))


@dataclass(frozen=True)
class _IonicConfigurationRegistry:
    version: str
    source_identity: str
    source_locator: str
    content_digest: str
    records: Tuple[_IonicConfiguration, ...]


@lru_cache(maxsize=1)
def _configuration_registry() -> _IonicConfigurationRegistry:
    resource = resources.files(__package__).joinpath(_REGISTRY_RESOURCE)
    encoded = resource.read_bytes()
    payload = json.loads(encoded.decode("utf-8"))
    expected = {
        "schema_version",
        "registry_version",
        "source_identity",
        "source_locator",
        "records",
    }
    if set(payload) != expected or payload["schema_version"] != _REGISTRY_SCHEMA:
        raise ValueError("Ionic-configuration registry does not match its schema")
    records = tuple(
        _IonicConfiguration(
            item["atomic_number"],
            item["oxidation_state"],
            item["shell"],
            item["occupancy"],
        )
        for item in payload["records"]
    )
    keys = tuple(
        (record.atomic_number, record.oxidation_state) for record in records
    )
    if (
        len(set(keys)) != len(keys)
        or any(
            record.shell not in {"d", "f"}
            or type(record.occupancy) is not int
            or not 0
            <= record.occupancy
            <= (10 if record.shell == "d" else 14)
            for record in records
        )
    ):
        raise ValueError("Ionic-configuration registry contains invalid records")
    return _IonicConfigurationRegistry(
        version=payload["registry_version"],
        source_identity=payload["source_identity"],
        source_locator=payload["source_locator"],
        content_digest=hashlib.sha256(encoded).hexdigest(),
        records=records,
    )


def _registered_oxidation_candidates(
    context: RuleContext,
) -> Tuple[
    Tuple[
        FactorKey,
        OxidationStateCandidateEvidence,
        Tuple[Tuple[int, _IonicConfiguration], ...],
    ],
    ...,
]:
    if context.candidate_space is None:
        return ()
    registry = {
        (record.atomic_number, record.oxidation_state): record
        for record in _configuration_registry().records
    }
    matches = []
    for factor in context.candidate_space.factors:
        if factor.factor_key.quantity is not ScientificQuantity.OXIDATION_STATE:
            continue
        for value in factor.values:
            candidate = value.candidate
            if not isinstance(candidate, OxidationStateCandidateEvidence):
                continue
            configurations = tuple(
                (atom_index, registry[key])
                for atom_index, oxidation_state in enumerate(
                    candidate.atom_states
                )
                if oxidation_state is not None
                for key in (
                    (
                        context.facts.atomic_numbers[atom_index],
                        oxidation_state,
                    ),
                )
                if key in registry
            )
            if configurations:
                matches.append((factor.factor_key, candidate, configurations))
    return tuple(matches)


def _factor_summary(
    context: RuleContext,
    subject_identity: str,
    omitted_count: int,
) -> CandidateDomainSummary:
    return CandidateDomainSummary(
        remaining_count_upper_bound=omitted_count,
        continuation_identity=_digest(
            {
                "rule": _RULE_ID,
                "facts": context.facts.identity.chemical_identity_digest,
                "subject": subject_identity,
                "offset": context.candidate_budget.max_materialized_values_per_factor,
            }
        ),
        frontier_bound=None,
    )


class IonicConfigurationSpinRule(InferenceRule):
    """Generate local spin-only domains from reviewed ionic configurations."""

    specification = RuleSpecification(
        rule_id=_RULE_ID,
        rule_version=_RULE_VERSION,
        kind=RuleKind.EMPIRICAL,
        phase=RulePhase.ELECTRONIC_STATE_CANDIDATE,
        requires=(),
        consumes=(ResultType.OXIDATION_STATE_RESULT,),
        produces=(ResultType.CANDIDATE_PROPOSAL, ResultType.EVIDENCE),
        applicability_schema=_DOMAIN_DEFINITION,
    )

    def applicable(self, context: RuleContext) -> RuleApplicability:
        matches = _registered_oxidation_candidates(context)
        return RuleApplicability(
            bool(matches),
            () if matches else ("IONIC_CONFIGURATION_UNAVAILABLE",),
        )

    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        if configuration:
            raise ValueError("OPTIONAL-009 does not accept configuration")
        registry = _configuration_registry()
        matches = _registered_oxidation_candidates(context)
        values_by_factor: Dict[FactorKey, list[ProposedCandidateValue]] = {}
        constraints = []
        for oxidation_factor, oxidation_candidate, configurations in matches:
            for atom_index, ionic_configuration in configurations:
                subject = f"atom:{context.facts.atom_map[atom_index]}"
                spin_factor = FactorKey(
                    ScientificQuantity.ELECTRONIC_STATE,
                    subject,
                )
                for two_s in ionic_configuration.two_s_candidates:
                    identity = _digest(
                        {
                            "rule": _RULE_ID,
                            "version": _RULE_VERSION,
                            "registry": registry.content_digest,
                            "facts": context.facts.identity.chemical_identity_digest,
                            "oxidation_candidate": (
                                oxidation_candidate.provenance.identity
                            ),
                            "atom_index": atom_index,
                            "two_s": two_s,
                        }
                    )
                    contribution = SpinContributionCandidate(
                        provenance=CandidateProvenance(
                            identity=identity,
                            authority=StateValueAuthority.DOMAIN_RULE,
                            source_identity=registry.source_identity,
                            evidence_ids=(
                                registry.content_digest,
                                oxidation_candidate.provenance.identity,
                            ),
                            assumption_ids=(
                                "domain_template:optional-009",
                            ),
                        ),
                        scope=SpinContributionScope.ATOM,
                        subject_identity=subject,
                        parent_candidate_ids=(
                            oxidation_candidate.provenance.identity,
                        ),
                        two_s=two_s,
                        pure_spin=True,
                    )
                    values_by_factor.setdefault(spin_factor, []).append(
                        ProposedCandidateValue(contribution)
                    )
                    constraint = CandidateConstraint(
                        identity=_digest(
                            {
                                "spin": identity,
                                "oxidation": (
                                    oxidation_candidate.provenance.identity
                                ),
                            }
                        ),
                        constraint_type="requires_all",
                        schema_version=_CONSTRAINT_SCHEMA,
                        factor_keys=(spin_factor, oxidation_factor),
                        parameters=(
                            (
                                "candidate_ids",
                                (
                                    identity,
                                    oxidation_candidate.provenance.identity,
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
        limit = context.candidate_budget.max_materialized_values_per_factor
        factors = []
        retained_ids = set()
        for factor_key in sorted(
            values_by_factor,
            key=lambda key: key.subject_identity,
        ):
            ordered = tuple(
                sorted(
                    values_by_factor[factor_key],
                    key=lambda value: value.candidate.provenance.identity,
                )
            )
            retained = ordered[:limit]
            retained_ids.update(
                value.candidate.provenance.identity for value in retained
            )
            omitted = len(ordered) - len(retained)
            factors.append(
                ProposedCandidateFactor(
                    factor_key=factor_key,
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
                        _factor_summary(
                            context,
                            factor_key.subject_identity,
                            omitted,
                        )
                        if omitted
                        else None
                    ),
                )
            )
        retained_constraints = tuple(
            constraint
            for constraint in constraints
            if constraint.guard.all_candidate_ids[0] in retained_ids
        )
        proposal = CandidateProposal(
            proposal_id=_digest(
                {
                    "rule": _RULE_ID,
                    "version": _RULE_VERSION,
                    "facts": context.facts.identity.chemical_identity_digest,
                    "parents": tuple(
                        candidate.provenance.identity
                        for _, candidate, _ in matches
                    ),
                }
            ),
            factors=tuple(factors),
            constraints=retained_constraints,
        )
        evidence = EvidenceItem(
            identity=registry.content_digest,
            authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            source_identity=registry.source_identity,
            claim="Reviewed ionic configurations and spin-only domains",
            locator=registry.source_locator,
        )
        return RuleOutcome(
            rule_id=_RULE_ID,
            rule_version=_RULE_VERSION,
            applicability=RuleApplicability(True),
            candidate_proposal=proposal,
            evidence=(evidence,),
            rank_adjustments=(),
            assumptions=("domain_template:optional-009",),
            warnings=(),
        )
