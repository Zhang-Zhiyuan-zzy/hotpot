"""Convention-labelled coordination electron-counting operation."""

from __future__ import annotations

import hashlib
import json
from typing import Tuple

from .. import legality
from ..charge.contracts import (
    LewisFormalChargeCandidate,
    LewisFormalChargeResult,
)
from ..contracts import (
    CandidateProvenance,
    ChemicalConfidence,
    MolecularFacts,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificFinding,
    StateValueAuthority,
)
from .contracts import (
    CoordinationCountingCandidate,
    CoordinationElectronCountingResult,
    CoordinationFragmentAssignment,
    CountingConvention,
)

__all__ = ["infer_coordination_electron_counting"]


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unresolved_result(
    facts: MolecularFacts,
    convention: CountingConvention,
    code: str,
    message: str,
    related_identities: Tuple[str, ...] = (),
) -> CoordinationElectronCountingResult:
    identity = _digest(
        {
            "facts": facts.identity.chemical_identity_digest,
            "convention": convention.value,
            "status": code,
            "related": related_identities,
        }
    )
    return CoordinationElectronCountingResult(
        identity,
        facts.identity,
        (),
        ScientificAdmissibility.NOT_EVALUATED,
        ResolutionStatus.UNRESOLVED,
        ChemicalConfidence.UNRESOLVED,
        (ScientificFinding(code, message, related_identities),),
    )


def _ionic_candidate(
    facts: MolecularFacts,
    lewis_candidate: LewisFormalChargeCandidate,
) -> CoordinationCountingCandidate:
    fragments = tuple(
        CoordinationFragmentAssignment(
            component_identity=component_identity,
            ionic_bookkeeping_charge=charge,
            lxz_class=None,
            donated_electrons=None,
        )
        for component_identity, charge in lewis_candidate.component_charges
    )
    identity = _digest(
        {
            "facts": facts.identity.chemical_identity_digest,
            "convention": CountingConvention.IONIC.value,
            "lewis_candidate": lewis_candidate.provenance.identity,
            "fragments": tuple(
                (
                    fragment.component_identity,
                    fragment.ionic_bookkeeping_charge,
                )
                for fragment in fragments
            ),
        }
    )
    return CoordinationCountingCandidate(
        provenance=CandidateProvenance(
            identity=identity,
            authority=StateValueAuthority.DERIVED_CONSTRAINT,
            source_identity="coordination-counting:ionic-v1",
            evidence_ids=(lewis_candidate.provenance.identity,),
        ),
        convention=CountingConvention.IONIC,
        fragments=fragments,
    )


def _ionic_result(
    facts: MolecularFacts,
    lewis_result: LewisFormalChargeResult,
) -> CoordinationElectronCountingResult:
    if lewis_result.facts_identity != facts.identity:
        return _unresolved_result(
            facts,
            CountingConvention.IONIC,
            "HARD-001.FACTS_IDENTITY_MISMATCH",
            "Lewis result is bound to different molecular facts.",
            (lewis_result.identity,),
        )
    if lewis_result.admissibility is not ScientificAdmissibility.ADMISSIBLE:
        return _unresolved_result(
            facts,
            CountingConvention.IONIC,
            "HARD-005.LEWIS_RESULT_UNAVAILABLE",
            "Ionic coordination counting requires a hard-valid Lewis result.",
            (lewis_result.identity,),
        )

    candidates = tuple(
        _ionic_candidate(facts, candidate)
        for candidate in lewis_result.candidates
    )
    legal_candidates = tuple(
        candidate
        for candidate in candidates
        if legality._validate_counting_candidate(candidate).admissibility
        is ScientificAdmissibility.ADMISSIBLE
    )
    identity = _digest(
        {
            "facts": facts.identity.chemical_identity_digest,
            "convention": CountingConvention.IONIC.value,
            "lewis_result": lewis_result.identity,
            "candidate_ids": tuple(
                candidate.provenance.identity for candidate in legal_candidates
            ),
        }
    )
    return CoordinationElectronCountingResult(
        identity=identity,
        facts_identity=facts.identity,
        candidates=legal_candidates,
        admissibility=(
            ScientificAdmissibility.ADMISSIBLE
            if legal_candidates
            else ScientificAdmissibility.NOT_EVALUATED
        ),
        resolution=(
            ResolutionStatus.UNIQUE
            if len(legal_candidates) == 1
            else ResolutionStatus.AMBIGUOUS
            if legal_candidates
            else ResolutionStatus.UNRESOLVED
        ),
        confidence=(
            ChemicalConfidence.MEDIUM
            if legal_candidates
            else ChemicalConfidence.UNRESOLVED
        ),
        findings=(),
    )


def infer_coordination_electron_counting(
    facts: MolecularFacts,
    *,
    lewis_results: Tuple[LewisFormalChargeResult, ...] = (),
    conventions: Tuple[CountingConvention, ...] = (),
) -> Tuple[CoordinationElectronCountingResult, ...]:
    """Translate supported evidence under explicitly requested conventions."""
    results = []
    for convention in conventions:
        if convention is not CountingConvention.IONIC:
            results.append(
                _unresolved_result(
                    facts,
                    convention,
                    "COUNTING.CONVENTION_REQUIRES_RULE_EVIDENCE",
                    (
                        "Donor-return and CBC assignments require explicit "
                        "motif/rule evidence."
                    ),
                )
            )
            continue
        if not lewis_results:
            results.append(
                _unresolved_result(
                    facts,
                    convention,
                    "COUNTING.LEWIS_EVIDENCE_MISSING",
                    "Ionic counting requires attributable Lewis evidence.",
                )
            )
            continue
        results.extend(
            _ionic_result(facts, lewis_result)
            for lewis_result in lewis_results
        )
    return tuple(results)
