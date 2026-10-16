"""Independent molecular net-charge resolution."""

from __future__ import annotations

import hashlib
import json
from typing import Tuple

from ..contracts import (
    CandidateProvenance,
    ChemicalConfidence,
    MolecularFacts,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificFinding,
    StateValueAuthority,
)
from ..legality import (
    _merge_hard_rule_assessments,
    _validate_charge_conservation,
    _validate_facts_identity,
)
from ..state_resolver import StateRequest
from .contracts import NetChargeCandidateEvidence, NetChargeResult

__all__ = ["resolve_net_charge"]


# Stable identities and result summaries.


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _explicit_candidate(facts: MolecularFacts, charge: int) -> NetChargeCandidateEvidence:
    identity = _digest(
        {
            "facts": facts.identity.chemical_identity_digest,
            "source": "StateRequest.net_charge",
            "charge": charge,
        }
    )
    return NetChargeCandidateEvidence(
        provenance=CandidateProvenance(
            identity=identity,
            authority=StateValueAuthority.VERIFIED_EXPLICIT,
            source_identity="StateRequest.net_charge",
        ),
        charge=charge,
    )


def _confidence(
    candidates: Tuple[NetChargeCandidateEvidence, ...],
) -> ChemicalConfidence:
    authorities = {candidate.provenance.authority for candidate in candidates}
    if authorities & {
        StateValueAuthority.VERIFIED_EXPLICIT,
        StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
    }:
        return ChemicalConfidence.AUTHORITATIVE
    if StateValueAuthority.EXACT_IDENTITY_TRUSTED_SOURCE in authorities:
        return ChemicalConfidence.HIGH
    if candidates:
        return ChemicalConfidence.MEDIUM
    return ChemicalConfidence.UNRESOLVED


def _result_identity(
    facts: MolecularFacts,
    candidates: Tuple[NetChargeCandidateEvidence, ...],
    resolution: ResolutionStatus,
) -> str:
    return _digest(
        {
            "facts": facts.identity.chemical_identity_digest,
            "candidate_ids": tuple(
                candidate.provenance.identity for candidate in candidates
            ),
            "resolution": resolution.value,
        }
    )


def _request_result(facts: MolecularFacts, request: StateRequest) -> NetChargeResult:
    requested_candidates = request.charge_candidates
    explicit_candidate = (
        _explicit_candidate(facts, request.net_charge)
        if request.net_charge is not None
        else None
    )
    if explicit_candidate is not None and requested_candidates:
        matching_candidates = tuple(
            candidate
            for candidate in requested_candidates
            if candidate.charge == request.net_charge
        )
        if not matching_candidates:
            candidates = (explicit_candidate,) + requested_candidates
            resolution = ResolutionStatus.CONFLICT
            findings = (
                ScientificFinding(
                    code="HARD-007.NET_CHARGE_EMPTY_INTERSECTION",
                    message=(
                        "Explicit net charge is excluded by the authoritative "
                        "request candidate domain."
                    ),
                    related_identities=tuple(
                        candidate.provenance.identity for candidate in candidates
                    ),
                ),
            )
            return NetChargeResult(
                identity=_result_identity(facts, candidates, resolution),
                facts_identity=facts.identity,
                candidates=candidates,
                admissibility=ScientificAdmissibility.INADMISSIBLE,
                resolution=resolution,
                confidence=_confidence(candidates),
                findings=findings,
            )
        candidates = (explicit_candidate,) + matching_candidates
    elif explicit_candidate is not None:
        candidates = (explicit_candidate,)
    else:
        candidates = requested_candidates

    assessments = tuple(
        _validate_charge_conservation(
            facts,
            candidate,
            atom_evidence=request.atom_formal_charges,
            component_evidence=request.component_charges,
        )
        for candidate in candidates
    )
    legal_candidates = tuple(
        candidate
        for candidate, assessment in zip(candidates, assessments)
        if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE
    )
    findings = tuple(
        finding for assessment in assessments for finding in assessment.findings
    )

    if not candidates:
        admissibility = ScientificAdmissibility.NOT_EVALUATED
        resolution = ResolutionStatus.UNRESOLVED
    elif not legal_candidates:
        merged = _merge_hard_rule_assessments(assessments)
        admissibility = merged.admissibility
        resolution = merged.terminal_status or ResolutionStatus.UNRESOLVED
    else:
        admissibility = ScientificAdmissibility.ADMISSIBLE
        resolution = (
            ResolutionStatus.UNIQUE
            if len({candidate.charge for candidate in legal_candidates}) == 1
            else ResolutionStatus.AMBIGUOUS
        )

    return NetChargeResult(
        identity=_result_identity(facts, legal_candidates, resolution),
        facts_identity=facts.identity,
        candidates=legal_candidates,
        admissibility=admissibility,
        resolution=resolution,
        confidence=_confidence(legal_candidates),
        findings=findings,
    )


def _revalidate_provided_result(
    facts: MolecularFacts,
    result: NetChargeResult,
) -> NetChargeResult:
    identity_assessment = _validate_facts_identity(
        facts.identity,
        result.facts_identity,
        related_identities=(result.identity,),
    )
    candidate_assessments = tuple(
        _validate_charge_conservation(facts, candidate)
        for candidate in result.candidates
    )
    assessment = _merge_hard_rule_assessments(
        (identity_assessment,) + candidate_assessments
    )
    if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE:
        return result
    return NetChargeResult(
        identity=result.identity,
        facts_identity=result.facts_identity,
        candidates=result.candidates,
        admissibility=assessment.admissibility,
        resolution=assessment.terminal_status or ResolutionStatus.UNRESOLVED,
        confidence=result.confidence,
        findings=result.findings + assessment.findings,
    )


# Public independent resolver.


def resolve_net_charge(
    facts: MolecularFacts,
    request: StateRequest = StateRequest(),
    *,
    provided: Tuple[NetChargeResult, ...] = (),
) -> Tuple[NetChargeResult, ...]:
    """Resolve explicit and provided net-charge evidence without Lewis inference."""
    request_result = _request_result(facts, request)
    return (request_result,) + tuple(
        _revalidate_provided_result(facts, result) for result in provided
    )
