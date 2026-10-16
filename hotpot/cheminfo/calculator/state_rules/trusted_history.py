"""Exact-identity scientific history rule without backend context."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..charge.contracts import NetChargeCandidateEvidence
from ..contracts import (
    CandidateProvenance,
    CoverageStatus,
    ScientificQuantity,
    StateValueAuthority,
)
from ..electronic_state.contracts import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
    SpinContributionScope,
)
from ..oxidation_state.contracts import (
    CountingConvention,
    OxidationStateCandidateEvidence,
)
from ..state_composition.contracts import (
    BranchAssignment,
    CandidateDomainScope,
    CandidateProposal,
    FactorKey,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
    StateCandidate,
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

__all__ = ["TrustedScientificStateRecord", "TrustedHistoryRule"]


_RULE_ID = "OPTIONAL-014"
_RULE_VERSION = "1.0.0"
_DOMAIN_DEFINITION = "exact-identity-trusted-history-v1"
_TRUSTED_AUTHORITIES = frozenset(
    {
        StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
        StateValueAuthority.EXACT_IDENTITY_TRUSTED_SOURCE,
    }
)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrustedScientificStateRecord:
    """One backend-independent state record for an exact molecular identity."""

    identity: str
    facts_identity_digest: str
    atom_map: Tuple[int, ...]
    source_identity: str
    source_digest: str
    authority: StateValueAuthority
    net_charge: Optional[int] = None
    oxidation_states: Tuple[Optional[int], ...] = ()
    counting_convention: CountingConvention = CountingConvention.IONIC
    two_s: Optional[int] = None
    multiplicity: Optional[int] = None
    pure_spin: bool = True
    assumption_ids: Tuple[str, ...] = ()
    locator: Optional[str] = None

    def __post_init__(self) -> None:
        if self.authority not in _TRUSTED_AUTHORITIES:
            raise ValueError(
                "Trusted scientific history requires a verified source authority"
            )
        if not self.identity or not self.source_identity or not self.source_digest:
            raise ValueError("Trusted scientific history requires source identity")


def _provenance(
    record: TrustedScientificStateRecord,
    quantity: ScientificQuantity,
) -> CandidateProvenance:
    return CandidateProvenance(
        identity=_digest(
            {
                "record": record.identity,
                "source_digest": record.source_digest,
                "quantity": quantity.value,
            }
        ),
        authority=record.authority,
        source_identity=record.source_identity,
        evidence_ids=(record.identity, record.source_digest),
        assumption_ids=record.assumption_ids,
    )


def _record_candidates(
    context: RuleContext,
    record: TrustedScientificStateRecord,
) -> Tuple[Tuple[FactorKey, StateCandidate], ...]:
    subject = f"molecule:{context.facts.identity.chemical_identity_digest}"
    charge = (
        NetChargeCandidateEvidence(
            _provenance(record, ScientificQuantity.NET_CHARGE),
            record.net_charge,
        )
        if record.net_charge is not None
        else None
    )
    oxidation = (
        OxidationStateCandidateEvidence(
            provenance=_provenance(
                record,
                ScientificQuantity.OXIDATION_STATE,
            ),
            convention=record.counting_convention,
            atom_states=record.oxidation_states,
            coverage=(
                CoverageStatus.COMPLETE
                if record.oxidation_states
                and all(state is not None for state in record.oxidation_states)
                else CoverageStatus.PARTIAL
            ),
        )
        if record.oxidation_states
        else None
    )
    electronic_provenance = _provenance(
        record,
        ScientificQuantity.ELECTRONIC_STATE,
    )
    if record.two_s is None:
        electronic: Optional[StateCandidate] = None
    elif charge is None:
        electronic = SpinContributionCandidate(
            provenance=electronic_provenance,
            scope=SpinContributionScope.MOLECULAR,
            subject_identity=subject,
            parent_candidate_ids=(record.identity,),
            two_s=record.two_s,
            pure_spin=record.pure_spin,
        )
    else:
        electronic = ElectronicStateCandidate(
            provenance=electronic_provenance,
            net_charge_candidate_identity=charge.provenance.identity,
            oxidation_state_candidate_identity=(
                oxidation.provenance.identity
                if oxidation is not None
                else None
            ),
            electron_count=(
                context.facts.nuclear_charge_sum - charge.charge
            ),
            two_s=record.two_s,
            multiplicity=record.multiplicity,
            pure_spin=record.pure_spin,
        )
    candidates = (
        (ScientificQuantity.NET_CHARGE, charge),
        (ScientificQuantity.OXIDATION_STATE, oxidation),
        (ScientificQuantity.ELECTRONIC_STATE, electronic),
    )
    return tuple(
        (FactorKey(quantity, subject), candidate)
        for quantity, candidate in candidates
        if candidate is not None
    )


class TrustedHistoryRule(InferenceRule):
    """Propose exact-identity scientific records as correlated candidates."""

    specification = RuleSpecification(
        rule_id=_RULE_ID,
        rule_version=_RULE_VERSION,
        kind=RuleKind.EMPIRICAL,
        phase=RulePhase.SOURCE_EVIDENCE,
        requires=(),
        consumes=(),
        produces=(
            ResultType.CANDIDATE_PROPOSAL,
            ResultType.EVIDENCE,
            ResultType.RANK_ADJUSTMENT,
        ),
        applicability_schema=_DOMAIN_DEFINITION,
    )

    def __init__(
        self,
        records: Tuple[TrustedScientificStateRecord, ...] = (),
    ) -> None:
        self._records = records

    def _matching_records(
        self,
        context: RuleContext,
    ) -> Tuple[TrustedScientificStateRecord, ...]:
        return tuple(
            record
            for record in self._records
            if record.facts_identity_digest
            == context.facts.identity.chemical_identity_digest
            and record.atom_map == context.facts.atom_map
        )

    def applicable(self, context: RuleContext) -> RuleApplicability:
        records = self._matching_records(context)
        return RuleApplicability(
            bool(records),
            (
                ()
                if records
                else ("EXACT_IDENTITY_HISTORY_UNAVAILABLE",)
            ),
        )

    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        del configuration
        records = self._matching_records(context)
        values_by_factor: Dict[
            FactorKey,
            list[ProposedCandidateValue],
        ] = {}
        rank_adjustments = []
        for record in records:
            branch = BranchAssignment(
                f"trusted-history:{context.facts.identity.chemical_identity_digest}",
                record.identity,
            )
            for factor_key, candidate in _record_candidates(context, record):
                values_by_factor.setdefault(factor_key, []).append(
                    ProposedCandidateValue(candidate, (branch,))
                )
                rank_adjustments.append(
                    RankAdjustment(
                        candidate.provenance.identity,
                        "exact_identity_history",
                        1.0,
                        f"Exact-identity scientific record {record.identity}",
                    )
                )
        proposal = CandidateProposal(
            proposal_id=_digest(
                {
                    "rule": _RULE_ID,
                    "version": _RULE_VERSION,
                    "facts": context.facts.identity.chemical_identity_digest,
                    "records": tuple(record.identity for record in records),
                }
            ),
            factors=tuple(
                ProposedCandidateFactor(
                    factor_key=factor_key,
                    domain_definition_id=_DOMAIN_DEFINITION,
                    materialized_candidates=tuple(values_by_factor[factor_key]),
                    domain_scope=CandidateDomainScope.DOMAIN_BOUNDED,
                    materialization_status=MaterializationStatus.COMPLETE,
                    unexpanded_domain_summary=None,
                )
                for factor_key in sorted(
                    values_by_factor,
                    key=lambda key: (
                        key.quantity.value,
                        key.subject_identity,
                    ),
                )
            ),
            constraints=(),
        )
        evidence = tuple(
            EvidenceItem(
                identity=record.identity,
                authority=record.authority,
                source_identity=record.source_identity,
                claim="Exact-identity backend-independent scientific state",
                locator=record.locator,
            )
            for record in records
        )
        return RuleOutcome(
            rule_id=_RULE_ID,
            rule_version=_RULE_VERSION,
            applicability=RuleApplicability(True),
            candidate_proposal=proposal,
            evidence=evidence,
            rank_adjustments=tuple(rank_adjustments),
            assumptions=tuple(
                sorted(
                    {
                        assumption
                        for record in records
                        for assumption in record.assumption_ids
                    }
                )
            ),
            warnings=(),
        )
