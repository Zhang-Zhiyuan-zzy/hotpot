"""OPTIONAL-011 localized f-block maximum-spin execution prior."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Dict, Tuple

from ..contracts import ScientificQuantity, StateValueAuthority
from ..electronic_state.contracts import SpinContributionCandidate
from ..oxidation_state.contracts import OxidationStateCandidateEvidence
from .base import InferenceRule
from .contracts import (
    EvidenceItem,
    RankAdjustment,
    ResultType,
    RuleApplicability,
    RuleContext,
    RuleDependency,
    RuleKind,
    RuleOutcome,
    RulePhase,
    RuleSpecification,
    RuleValue,
)

__all__ = ["FBlockHundPriorRule"]


_RULE_ID = "OPTIONAL-011"
_RULE_VERSION = "1.0.0"
_REGISTRY_SCHEMA = "hotpot-f-block-hund-prior-v1"
_REGISTRY_RESOURCE = "data/rules/f_block_hund_prior_v1.json"
_APPLICABILITY_SCHEMA = "localized-f-block-hund-prior-v1"
_IONIC_SPIN_ASSUMPTION = "domain_template:optional-009"


@dataclass(frozen=True)
class _HundPriorRegistry:
    version: str
    source_identity: str
    source_locator: str
    content_digest: str
    applicable_ions: frozenset[Tuple[int, int]]


@lru_cache(maxsize=1)
def _hund_registry() -> _HundPriorRegistry:
    resource = resources.files(__package__).joinpath(_REGISTRY_RESOURCE)
    encoded = resource.read_bytes()
    payload = json.loads(encoded.decode("utf-8"))
    expected = {
        "schema_version",
        "registry_version",
        "source_identity",
        "source_locator",
        "applicable_ions",
    }
    if set(payload) != expected or payload["schema_version"] != _REGISTRY_SCHEMA:
        raise ValueError("f-block Hund registry does not match its schema")
    applicable_ions = frozenset(
        (item[0], item[1]) for item in payload["applicable_ions"]
    )
    if len(applicable_ions) != len(payload["applicable_ions"]) or any(
        type(atomic_number) is not int
        or type(oxidation_state) is not int
        for atomic_number, oxidation_state in applicable_ions
    ):
        raise ValueError("f-block Hund registry contains invalid ion keys")
    return _HundPriorRegistry(
        version=payload["registry_version"],
        source_identity=payload["source_identity"],
        source_locator=payload["source_locator"],
        content_digest=hashlib.sha256(encoded).hexdigest(),
        applicable_ions=applicable_ions,
    )


def _rankable_groups(
    context: RuleContext,
) -> Tuple[Tuple[SpinContributionCandidate, ...], ...]:
    if context.candidate_space is None:
        return ()
    candidates = {
        value.candidate.provenance.identity: value.candidate
        for factor in context.candidate_space.factors
        for value in factor.values
    }
    atom_index_by_subject = {
        f"atom:{atom_map}": index
        for index, atom_map in enumerate(context.facts.atom_map)
    }
    groups: Dict[
        Tuple[str, str],
        list[SpinContributionCandidate],
    ] = {}
    for factor in context.candidate_space.factors:
        if factor.factor_key.quantity is not ScientificQuantity.ELECTRONIC_STATE:
            continue
        for value in factor.values:
            contribution = value.candidate
            if (
                not isinstance(contribution, SpinContributionCandidate)
                or _IONIC_SPIN_ASSUMPTION
                not in contribution.provenance.assumption_ids
                or len(contribution.parent_candidate_ids) != 1
            ):
                continue
            atom_index = atom_index_by_subject.get(contribution.subject_identity)
            parent_identity = contribution.parent_candidate_ids[0]
            parent = candidates.get(parent_identity)
            if (
                atom_index is None
                or not isinstance(parent, OxidationStateCandidateEvidence)
                or atom_index >= len(parent.atom_states)
                or parent.atom_states[atom_index] is None
                or (
                    context.facts.atomic_numbers[atom_index],
                    parent.atom_states[atom_index],
                )
                not in _hund_registry().applicable_ions
            ):
                continue
            groups.setdefault(
                (contribution.subject_identity, parent_identity),
                [],
            ).append(contribution)
    return tuple(
        tuple(
            sorted(
                group,
                key=lambda candidate: (
                    candidate.two_s,
                    candidate.provenance.identity,
                ),
            )
        )
        for _, group in sorted(groups.items())
    )


class FBlockHundPriorRule(InferenceRule):
    """Rank maximum local spin in reviewed localized f-block ion domains."""

    specification = RuleSpecification(
        rule_id=_RULE_ID,
        rule_version=_RULE_VERSION,
        kind=RuleKind.EMPIRICAL,
        phase=RulePhase.ELECTRONIC_STATE_RANKING,
        requires=(RuleDependency("OPTIONAL-009", ">=1,<2"),),
        consumes=(ResultType.SPIN_CONTRIBUTION,),
        produces=(ResultType.EVIDENCE, ResultType.RANK_ADJUSTMENT),
        applicability_schema=_APPLICABILITY_SCHEMA,
    )

    def applicable(self, context: RuleContext) -> RuleApplicability:
        groups = _rankable_groups(context)
        return RuleApplicability(
            bool(groups),
            () if groups else ("F_BLOCK_HUND_DOMAIN_UNAVAILABLE",),
        )

    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        if configuration:
            raise ValueError("OPTIONAL-011 does not accept configuration")
        registry = _hund_registry()
        adjustments = tuple(
            RankAdjustment(
                target_identity=max(
                    group,
                    key=lambda candidate: candidate.two_s,
                ).provenance.identity,
                rank_dimension="f_block_hund",
                value=1.0,
                reason="Reviewed localized f-block maximum-spin execution prior",
                assumption_ids=("execution_prior:optional-011",),
            )
            for group in _rankable_groups(context)
        )
        evidence = EvidenceItem(
            identity=registry.content_digest,
            authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            source_identity=registry.source_identity,
            claim="Localized f-block maximum-spin execution-prior domain",
            locator=registry.source_locator,
        )
        return RuleOutcome(
            rule_id=_RULE_ID,
            rule_version=_RULE_VERSION,
            applicability=RuleApplicability(True),
            candidate_proposal=None,
            evidence=(evidence,),
            rank_adjustments=adjustments,
            assumptions=("execution_prior:optional-011",),
            warnings=("SPIN_ORBIT_COUPLING_NOT_MODELLED",),
        )
