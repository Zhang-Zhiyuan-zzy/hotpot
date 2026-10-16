"""Shared mechanics for reviewed local oxidation-state priors."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from importlib import resources
from typing import Dict, Tuple

from ..contracts import (
    CandidateProvenance,
    CoverageStatus,
    ScientificQuantity,
    StateValueAuthority,
)
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    CountingConvention,
    OxidationStateCandidateEvidence,
)
from ..state_composition.contracts import (
    CandidateDomainScope,
    CandidateDomainSummary,
    CandidateProposal,
    FactorKey,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
)
from .base import InferenceRule
from .contracts import (
    EvidenceItem,
    RankAdjustment,
    ResultType,
    RuleApplicability,
    RuleContext,
    RuleKind,
    RuleOutcome,
    RulePhase,
    RuleSpecification,
    RuleValue,
)

__all__ = []


@dataclass(frozen=True)
class _ElementOxidationPrior:
    atomic_number: int
    candidate_states: Tuple[int, ...]
    preferred_state: int


@dataclass(frozen=True)
class _OxidationPriorArtifact:
    rule_id: str
    rule_version: str
    source_identity: str
    source_locator: str
    content_digest: str
    applicability_schema: str
    elements: Tuple[_ElementOxidationPrior, ...]


def _load_oxidation_prior_artifact(
    resource_name: str,
    expected_rule_id: str,
) -> _OxidationPriorArtifact:
    resource = resources.files(__package__).joinpath(resource_name)
    encoded = resource.read_bytes()
    payload = json.loads(encoded.decode("utf-8"))
    expected_fields = {
        "schema_version",
        "rule_id",
        "rule_version",
        "source_identity",
        "source_locator",
        "applicability_schema",
        "elements",
    }
    if (
        set(payload) != expected_fields
        or payload["schema_version"] != "hotpot-oxidation-prior-v1"
        or payload["rule_id"] != expected_rule_id
    ):
        raise ValueError("Oxidation-prior artifact does not match its schema")
    elements = tuple(
        _ElementOxidationPrior(
            atomic_number=item["atomic_number"],
            candidate_states=tuple(item["candidate_states"]),
            preferred_state=item["preferred_state"],
        )
        for item in payload["elements"]
    )
    if (
        not elements
        or len({entry.atomic_number for entry in elements}) != len(elements)
        or any(
            type(entry.atomic_number) is not int
            or entry.atomic_number <= 0
            or not entry.candidate_states
            or len(set(entry.candidate_states)) != len(entry.candidate_states)
            or any(type(state) is not int for state in entry.candidate_states)
            or entry.preferred_state not in entry.candidate_states
            for entry in elements
        )
    ):
        raise ValueError("Oxidation-prior artifact contains invalid domains")
    return _OxidationPriorArtifact(
        rule_id=payload["rule_id"],
        rule_version=payload["rule_version"],
        source_identity=payload["source_identity"],
        source_locator=payload["source_locator"],
        content_digest=hashlib.sha256(encoded).hexdigest(),
        applicability_schema=payload["applicability_schema"],
        elements=elements,
    )


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coordination_neighbours(context: RuleContext) -> Dict[int, set[int]]:
    neighbours: Dict[int, set[int]] = {}
    for bond in context.facts.semantic_bonds:
        if bond.kind != "coordination":
            continue
        begin, end = bond.atom_indices
        neighbours.setdefault(begin, set()).add(end)
        neighbours.setdefault(end, set()).add(begin)
    return neighbours


def _component_by_atom(context: RuleContext) -> Dict[int, str]:
    return {
        atom_index: component.identity
        for component in context.facts.covalent_components
        for atom_index in component.atom_indices
    }


def _counted_component_ids(context: RuleContext) -> frozenset[str]:
    candidate_space = context.candidate_space
    if candidate_space is None:
        return frozenset()
    return frozenset(
        fragment.component_identity
        for factor in candidate_space.factors
        if factor.factor_key.quantity
        is ScientificQuantity.COORDINATION_COUNTING
        for value in factor.values
        if isinstance(value.candidate, CoordinationCountingCandidate)
        and value.candidate.convention is CountingConvention.IONIC
        for fragment in value.candidate.fragments
        if fragment.ionic_bookkeeping_charge is not None
    )


def _applicable_centers(
    context: RuleContext,
    artifact: _OxidationPriorArtifact,
) -> Tuple[int, ...]:
    domains = {entry.atomic_number: entry for entry in artifact.elements}
    neighbours = _coordination_neighbours(context)
    component_by_atom = _component_by_atom(context)
    counted_components = _counted_component_ids(context)
    centers = []
    for atom_index, atomic_number in enumerate(context.facts.atomic_numbers):
        if atomic_number not in domains or atom_index not in neighbours:
            continue
        ligand_components = {
            component_by_atom[neighbour]
            for neighbour in neighbours[atom_index]
            if neighbour in component_by_atom
            and component_by_atom[neighbour] != component_by_atom[atom_index]
        }
        if ligand_components and ligand_components.issubset(counted_components):
            centers.append(atom_index)
    return tuple(centers)


class _OxidationStatePriorRule(InferenceRule):
    """Factorized candidate/ranking implementation shared by OS priors."""

    def __init__(self, artifact: _OxidationPriorArtifact) -> None:
        self._artifact = artifact
        self.specification = RuleSpecification(
            rule_id=artifact.rule_id,
            rule_version=artifact.rule_version,
            kind=RuleKind.EMPIRICAL,
            phase=RulePhase.OXIDATION_STATE_CANDIDATE,
            requires=(),
            consumes=(ResultType.COORDINATION_COUNTING_RESULT,),
            produces=(
                ResultType.CANDIDATE_PROPOSAL,
                ResultType.EVIDENCE,
                ResultType.RANK_ADJUSTMENT,
            ),
            applicability_schema=artifact.applicability_schema,
        )

    def applicable(self, context: RuleContext) -> RuleApplicability:
        centers = _applicable_centers(context, self._artifact)
        return RuleApplicability(
            bool(centers),
            (
                ()
                if centers
                else ("OXIDATION_PRIOR_LOCAL_COUNTING_UNAVAILABLE",)
            ),
        )

    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        if configuration:
            raise ValueError(
                f"{self._artifact.rule_id} does not accept configuration"
            )
        centers = _applicable_centers(context, self._artifact)
        domains = {
            entry.atomic_number: entry for entry in self._artifact.elements
        }
        state_domains = tuple(
            domains[context.facts.atomic_numbers[index]].candidate_states
            for index in centers
        )
        state_combinations = itertools.product(*state_domains)
        limit = context.candidate_budget.max_materialized_values_per_factor
        materialized = tuple(itertools.islice(state_combinations, limit + 1))
        truncated = len(materialized) > limit
        retained = materialized[:limit]
        candidates = []
        rank_adjustments = []
        for states in retained:
            atom_states = [None] * len(context.facts.atomic_numbers)
            for atom_index, state in zip(centers, states):
                atom_states[atom_index] = state
            identity = _digest(
                {
                    "rule": self._artifact.rule_id,
                    "version": self._artifact.rule_version,
                    "artifact": self._artifact.content_digest,
                    "facts": context.facts.identity.chemical_identity_digest,
                    "atom_states": atom_states,
                }
            )
            candidate = OxidationStateCandidateEvidence(
                provenance=CandidateProvenance(
                    identity=identity,
                    authority=StateValueAuthority.DOMAIN_RULE,
                    source_identity=self._artifact.source_identity,
                    evidence_ids=(self._artifact.content_digest,),
                    assumption_ids=(
                        f"domain_template:{self._artifact.rule_id.lower()}",
                    ),
                ),
                convention=CountingConvention.IONIC,
                atom_states=tuple(atom_states),
                coverage=CoverageStatus.PARTIAL,
            )
            candidates.append(ProposedCandidateValue(candidate))
            preferred = all(
                state
                == domains[
                    context.facts.atomic_numbers[atom_index]
                ].preferred_state
                for atom_index, state in zip(centers, states)
            )
            if preferred:
                rank_adjustments.append(
                    RankAdjustment(
                        target_identity=identity,
                        rank_dimension="oxidation_state_commonality",
                        value=1.0,
                        reason=(
                            f"{self._artifact.rule_id} reviewed common-state prior"
                        ),
                    )
                )
        subject = f"molecule:{context.facts.identity.chemical_identity_digest}"
        summary = (
            CandidateDomainSummary(
                remaining_count_upper_bound=(
                    None
                    if not state_domains
                    else _integer_product(
                        tuple(len(domain) for domain in state_domains)
                    )
                    - limit
                ),
                continuation_identity=_digest(
                    {
                        "rule": self._artifact.rule_id,
                        "facts": context.facts.identity.chemical_identity_digest,
                        "offset": limit,
                    }
                ),
                frontier_bound=None,
            )
            if truncated
            else None
        )
        factor = ProposedCandidateFactor(
            factor_key=FactorKey(
                ScientificQuantity.OXIDATION_STATE,
                subject,
            ),
            domain_definition_id=(
                f"{self._artifact.applicability_schema}:"
                f"{self._artifact.content_digest}"
            ),
            materialized_candidates=tuple(candidates),
            domain_scope=CandidateDomainScope.DOMAIN_BOUNDED,
            materialization_status=(
                MaterializationStatus.RESOURCE_TRUNCATED
                if truncated
                else MaterializationStatus.COMPLETE
            ),
            unexpanded_domain_summary=summary,
        )
        proposal = CandidateProposal(
            proposal_id=_digest(
                {
                    "rule": self._artifact.rule_id,
                    "version": self._artifact.rule_version,
                    "facts": context.facts.identity.chemical_identity_digest,
                    "centers": centers,
                }
            ),
            factors=(factor,),
            constraints=(),
        )
        evidence = EvidenceItem(
            identity=self._artifact.content_digest,
            authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            source_identity=self._artifact.source_identity,
            claim=f"Reviewed domain for {self._artifact.rule_id}",
            locator=self._artifact.source_locator,
        )
        return RuleOutcome(
            rule_id=self._artifact.rule_id,
            rule_version=self._artifact.rule_version,
            applicability=RuleApplicability(True),
            candidate_proposal=proposal,
            evidence=(evidence,),
            rank_adjustments=tuple(rank_adjustments),
            assumptions=(
                f"domain_template:{self._artifact.rule_id.lower()}",
            ),
            warnings=(),
        )


def _integer_product(values: Tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result
