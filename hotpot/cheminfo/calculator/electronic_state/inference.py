"""Independent resolution of attributable molecular electronic states."""

from __future__ import annotations

import hashlib
import json
from typing import Optional, Tuple

from ..charge.contracts import NetChargeCandidateEvidence, NetChargeResult
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
    _validate_electronic_state_candidate,
    _validate_facts_identity,
)
from ..oxidation_state.contracts import OxidationStateInferenceResult
from ..state_resolver import StateRequest
from .contracts import (
    ElectronicStateCandidate,
    ElectronicStateCandidateEvidence,
    ElectronicStateInferenceResult,
)

__all__ = ["infer_electronic_states"]


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _explicit_evidence(
    facts: MolecularFacts,
    request: StateRequest,
) -> Optional[ElectronicStateCandidateEvidence]:
    if request.two_s is None and request.multiplicity is None:
        return None
    two_s = (
        request.two_s
        if request.two_s is not None
        else request.multiplicity - 1
    )
    multiplicity = (
        request.multiplicity
        if request.multiplicity is not None
        else two_s + 1
    )
    identity = _digest(
        {
            "facts": facts.identity.chemical_identity_digest,
            "source": "StateRequest.electronic_state",
            "two_s": two_s,
            "multiplicity": multiplicity,
        }
    )
    return ElectronicStateCandidateEvidence(
        provenance=CandidateProvenance(
            identity=identity,
            authority=StateValueAuthority.VERIFIED_EXPLICIT,
            source_identity="StateRequest.electronic_state",
        ),
        two_s=two_s,
        multiplicity=multiplicity,
        pure_spin=True,
    )


def _request_evidence(
    facts: MolecularFacts,
    request: StateRequest,
) -> Tuple[
    Tuple[ElectronicStateCandidateEvidence, ...],
    Tuple[ScientificFinding, ...],
]:
    explicit = _explicit_evidence(facts, request)
    requested = request.electronic_candidates
    if (
        request.two_s is not None
        and request.multiplicity is not None
        and request.multiplicity != request.two_s + 1
    ):
        return (), (
            ScientificFinding(
                code="HARD-007.EXPLICIT_SPIN_CONFLICT",
                message="Explicit two_s and multiplicity describe different pure-spin states.",
                related_identities=(explicit.provenance.identity,),
            ),
        )
    if explicit is None:
        return requested, ()
    if not requested:
        return (explicit,), ()

    matching = tuple(
        candidate
        for candidate in requested
        if candidate.two_s == explicit.two_s
        and (
            candidate.multiplicity is None
            or candidate.multiplicity == explicit.multiplicity
        )
    )
    if matching:
        return (explicit,) + matching, ()
    return (), (
        ScientificFinding(
            code="HARD-007.ELECTRONIC_STATE_EMPTY_INTERSECTION",
            message=(
                "Explicit molecular spin is excluded by the authoritative "
                "request candidate domain."
            ),
            related_identities=(
                explicit.provenance.identity,
                *tuple(
                    candidate.provenance.identity for candidate in requested
                ),
            ),
        ),
    )


def _candidate(
    facts: MolecularFacts,
    charge: NetChargeCandidateEvidence,
    evidence: ElectronicStateCandidateEvidence,
) -> ElectronicStateCandidate:
    identity = _digest(
        {
            "facts": facts.identity.chemical_identity_digest,
            "charge": charge.provenance.identity,
            "evidence": evidence.provenance.identity,
        }
    )
    return ElectronicStateCandidate(
        provenance=CandidateProvenance(
            identity=identity,
            authority=evidence.provenance.authority,
            source_identity=evidence.provenance.source_identity,
            evidence_ids=(evidence.provenance.identity,),
            assumption_ids=evidence.provenance.assumption_ids,
        ),
        net_charge_candidate_identity=charge.provenance.identity,
        oxidation_state_candidate_identity=None,
        electron_count=facts.nuclear_charge_sum - charge.charge,
        two_s=evidence.two_s,
        multiplicity=evidence.multiplicity,
        pure_spin=evidence.pure_spin,
    )


def _confidence(
    candidates: Tuple[ElectronicStateCandidate, ...],
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


def _request_result(
    facts: MolecularFacts,
    request: StateRequest,
    net_charge_results: Tuple[NetChargeResult, ...],
) -> ElectronicStateInferenceResult:
    evidence, request_findings = _request_evidence(facts, request)
    charge_candidates = tuple(
        candidate
        for result in net_charge_results
        if result.facts_identity == facts.identity
        and result.admissibility is ScientificAdmissibility.ADMISSIBLE
        for candidate in result.candidates
    )
    candidates = tuple(
        _candidate(facts, charge, item)
        for charge in charge_candidates
        for item in evidence
    )
    charge_by_identity = {
        charge.provenance.identity: charge for charge in charge_candidates
    }
    assessments = tuple(
        _validate_electronic_state_candidate(
            facts,
            charge_by_identity[candidate.net_charge_candidate_identity],
            candidate,
        )
        for candidate in candidates
    )
    legal_candidates = tuple(
        candidate
        for candidate, assessment in zip(candidates, assessments)
        if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE
    )
    findings = request_findings + tuple(
        finding for assessment in assessments for finding in assessment.findings
    )

    if request_findings:
        admissibility = ScientificAdmissibility.INADMISSIBLE
        resolution = ResolutionStatus.CONFLICT
    elif not evidence or not charge_candidates:
        admissibility = ScientificAdmissibility.NOT_EVALUATED
        resolution = ResolutionStatus.UNRESOLVED
    elif not legal_candidates:
        assessment = _merge_hard_rule_assessments(assessments)
        admissibility = assessment.admissibility
        resolution = assessment.terminal_status or ResolutionStatus.UNRESOLVED
    else:
        admissibility = ScientificAdmissibility.ADMISSIBLE
        states = {
            (
                candidate.net_charge_candidate_identity,
                candidate.two_s,
                candidate.multiplicity,
                candidate.pure_spin,
            )
            for candidate in legal_candidates
        }
        resolution = (
            ResolutionStatus.UNIQUE
            if len(states) == 1
            else ResolutionStatus.AMBIGUOUS
        )

    identity = _digest(
        {
            "facts": facts.identity.chemical_identity_digest,
            "candidate_ids": tuple(
                candidate.provenance.identity for candidate in legal_candidates
            ),
            "resolution": resolution.value,
        }
    )
    return ElectronicStateInferenceResult(
        identity=identity,
        facts_identity=facts.identity,
        candidates=legal_candidates,
        admissibility=admissibility,
        resolution=resolution,
        confidence=_confidence(legal_candidates),
        findings=findings,
    )


def _revalidate_provided_result(
    facts: MolecularFacts,
    result: ElectronicStateInferenceResult,
    charge_candidates: Tuple[NetChargeCandidateEvidence, ...],
) -> ElectronicStateInferenceResult:
    charge_by_identity = {
        candidate.provenance.identity: candidate
        for candidate in charge_candidates
    }
    assessments = [
        _validate_facts_identity(
            facts.identity,
            result.facts_identity,
            related_identities=(result.identity,),
        )
    ]
    for candidate in result.candidates:
        charge = charge_by_identity.get(candidate.net_charge_candidate_identity)
        if charge is None:
            findings = result.findings + (
                ScientificFinding(
                    code="HARD-007.NET_CHARGE_REFERENCE_MISSING",
                    message=(
                        "Electronic-state evidence references an unavailable "
                        "net-charge candidate."
                    ),
                    related_identities=(candidate.provenance.identity,),
                ),
            )
            return ElectronicStateInferenceResult(
                result.identity,
                result.facts_identity,
                result.candidates,
                ScientificAdmissibility.INADMISSIBLE,
                ResolutionStatus.CONFLICT,
                result.confidence,
                findings,
            )
        assessments.append(
            _validate_electronic_state_candidate(facts, charge, candidate)
        )
    assessment = _merge_hard_rule_assessments(tuple(assessments))
    if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE:
        return result
    return ElectronicStateInferenceResult(
        result.identity,
        result.facts_identity,
        result.candidates,
        assessment.admissibility,
        assessment.terminal_status or ResolutionStatus.UNRESOLVED,
        result.confidence,
        result.findings + assessment.findings,
    )


def infer_electronic_states(
    facts: MolecularFacts,
    request: StateRequest = StateRequest(),
    *,
    net_charge_results: Tuple[NetChargeResult, ...],
    oxidation_state_results: Tuple[OxidationStateInferenceResult, ...] = (),
    provided: Tuple[ElectronicStateInferenceResult, ...] = (),
) -> Tuple[ElectronicStateInferenceResult, ...]:
    """Resolve explicit/source molecular spin without applying defaults."""
    del oxidation_state_results
    request_result = _request_result(facts, request, net_charge_results)
    charge_candidates = tuple(
        candidate
        for result in net_charge_results
        if result.facts_identity == facts.identity
        for candidate in result.candidates
    )
    return (request_result,) + tuple(
        _revalidate_provided_result(facts, result, charge_candidates)
        for result in provided
    )
