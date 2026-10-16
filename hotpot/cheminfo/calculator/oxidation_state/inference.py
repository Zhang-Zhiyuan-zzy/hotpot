"""Independent resolution of attributable oxidation-state evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Tuple

from ..contracts import (
    ChemicalConfidence,
    MolecularFacts,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificFinding,
    StateValueAuthority,
)
from .. import legality
from ..state_resolver import StateRequest
from .contracts import (
    CoordinationCountingCandidate,
    CoordinationElectronCountingResult,
    OxidationStateCandidateEvidence,
    OxidationStateInferenceResult,
)

__all__ = ["infer_oxidation_states"]


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _confidence(
    candidates: Tuple[OxidationStateCandidateEvidence, ...],
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


def _candidate_matches_overrides(
    candidate: OxidationStateCandidateEvidence,
    request: StateRequest,
) -> bool:
    return all(
        override.atom_index >= len(candidate.atom_states)
        or candidate.atom_states[override.atom_index] is None
        or candidate.atom_states[override.atom_index] == override.oxidation_state
        for override in request.oxidation_overrides
    )


def _request_result(
    facts: MolecularFacts,
    request: StateRequest,
    counting_candidates: Tuple[CoordinationCountingCandidate, ...],
) -> OxidationStateInferenceResult:
    requested_candidates = request.oxidation_candidates
    compatible_candidates = tuple(
        candidate
        for candidate in requested_candidates
        if _candidate_matches_overrides(candidate, request)
    )
    assessments = tuple(
        legality._merge_hard_rule_assessments(
            (
                legality._validate_oxidation_state_evidence(facts, candidate),
                legality._validate_counting_reference(candidate, counting_candidates),
            )
        )
        for candidate in compatible_candidates
    )
    legal_candidates = tuple(
        candidate
        for candidate, assessment in zip(compatible_candidates, assessments)
        if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE
    )
    findings = tuple(
        finding for assessment in assessments for finding in assessment.findings
    )

    override_assessment = legality._validate_oxidation_overrides(
        facts,
        request.oxidation_overrides,
        legal_candidates,
    )
    if requested_candidates and not compatible_candidates:
        admissibility = ScientificAdmissibility.INADMISSIBLE
        resolution = ResolutionStatus.CONFLICT
        findings += (
            ScientificFinding(
                code="HARD-007.OS_OVERRIDE_EMPTY_INTERSECTION",
                message=(
                    "No oxidation-state candidate is compatible with all "
                    "explicit overrides."
                ),
                related_identities=tuple(
                    candidate.provenance.identity
                    for candidate in requested_candidates
                ),
            ),
        )
    elif not requested_candidates and not request.oxidation_overrides:
        admissibility = ScientificAdmissibility.NOT_EVALUATED
        resolution = ResolutionStatus.UNRESOLVED
    elif not requested_candidates:
        findings += override_assessment.findings
        admissibility = override_assessment.admissibility
        resolution = (
            override_assessment.terminal_status
            or ResolutionStatus.UNRESOLVED
        )
    elif not legal_candidates:
        assessment = legality._merge_hard_rule_assessments(assessments)
        admissibility = assessment.admissibility
        resolution = assessment.terminal_status or ResolutionStatus.UNRESOLVED
    elif override_assessment.admissibility is ScientificAdmissibility.INADMISSIBLE:
        findings += override_assessment.findings
        admissibility = override_assessment.admissibility
        resolution = override_assessment.terminal_status or ResolutionStatus.CONFLICT
    else:
        findings += override_assessment.findings
        all_states = {
            (candidate.convention, candidate.atom_states)
            for candidate in legal_candidates
        }
        complete_states = {
            state
            for state in all_states
            if all(value is not None for value in state[1])
        }
        admissibility = ScientificAdmissibility.ADMISSIBLE
        resolution = (
            ResolutionStatus.UNIQUE
            if len(all_states) == len(complete_states) == 1
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
    return OxidationStateInferenceResult(
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
    result: OxidationStateInferenceResult,
    counting_candidates: Tuple[CoordinationCountingCandidate, ...],
) -> OxidationStateInferenceResult:
    assessments = (
        legality._validate_facts_identity(
            facts.identity,
            result.facts_identity,
            related_identities=(result.identity,),
        ),
        *tuple(
            legality._merge_hard_rule_assessments(
                (
                    legality._validate_oxidation_state_evidence(facts, candidate),
                    legality._validate_counting_reference(
                        candidate,
                        counting_candidates,
                    ),
                )
            )
            for candidate in result.candidates
        ),
    )
    assessment = legality._merge_hard_rule_assessments(assessments)
    if assessment.admissibility is ScientificAdmissibility.ADMISSIBLE:
        return result
    return OxidationStateInferenceResult(
        result.identity,
        result.facts_identity,
        result.candidates,
        assessment.admissibility,
        assessment.terminal_status or ResolutionStatus.UNRESOLVED,
        result.confidence,
        result.findings + assessment.findings,
    )


def infer_oxidation_states(
    facts: MolecularFacts,
    request: StateRequest = StateRequest(),
    *,
    counting_results: Tuple[CoordinationElectronCountingResult, ...] = (),
    provided: Tuple[OxidationStateInferenceResult, ...] = (),
) -> Tuple[OxidationStateInferenceResult, ...]:
    """Resolve explicit/source oxidation-state evidence without defaults."""
    counting_candidates = tuple(
        candidate
        for result in counting_results
        if result.facts_identity == facts.identity
        for candidate in result.candidates
        if legality._validate_counting_candidate(candidate).admissibility
        is ScientificAdmissibility.ADMISSIBLE
    )
    request_result = _request_result(facts, request, counting_candidates)
    return (request_result,) + tuple(
        _revalidate_provided_result(facts, result, counting_candidates)
        for result in provided
    )
