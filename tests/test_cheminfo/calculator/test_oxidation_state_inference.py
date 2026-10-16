"""Independent explicit oxidation-state inference tests."""

from typing import Optional, Tuple

import hotpot

from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    CoverageStatus,
    ResolutionStatus,
    ScientificAdmissibility,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CountingConvention,
    OxidationStateCandidateEvidence,
    OxidationStateOverride,
    infer_oxidation_states,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)


def _facts(smiles: str):
    return extract_molecular_facts(hotpot.read_mol(smiles, "smi"))


def _candidate(
    identity: str,
    states: Tuple[Optional[int], ...],
    coverage: CoverageStatus = CoverageStatus.COMPLETE,
) -> OxidationStateCandidateEvidence:
    return OxidationStateCandidateEvidence(
        provenance=CandidateProvenance(
            identity=identity,
            authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            source_identity="oxidation-state-test",
        ),
        convention=CountingConvention.IONIC,
        atom_states=states,
        coverage=coverage,
    )


def test_strict_inference_does_not_activate_element_registry() -> None:
    facts = _facts("[Eu]")

    result = infer_oxidation_states(facts)[0]

    assert result.candidates == ()
    assert result.admissibility is ScientificAdmissibility.NOT_EVALUATED
    assert result.resolution is ResolutionStatus.UNRESOLVED


def test_complete_source_candidate_is_preserved() -> None:
    facts = _facts("[Na+].[Cl-]")
    request = StateRequest(
        oxidation_candidates=(_candidate("ionic-salt", (1, -1)),),
    )

    result = infer_oxidation_states(facts, request)[0]

    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert result.resolution is ResolutionStatus.UNIQUE
    assert result.candidates[0].atom_states == (1, -1)


def test_override_filters_compatible_candidate_domain() -> None:
    facts = _facts("[Eu]")
    request = StateRequest(
        oxidation_overrides=(OxidationStateOverride(0, 3),),
        oxidation_candidates=(
            _candidate("eu-ii", (2,)),
            _candidate("eu-iii", (3,)),
        ),
    )

    result = infer_oxidation_states(facts, request)[0]

    assert result.resolution is ResolutionStatus.UNIQUE
    assert tuple(candidate.provenance.identity for candidate in result.candidates) == (
        "eu-iii",
    )


def test_override_empty_intersection_is_conflict() -> None:
    facts = _facts("[Eu]")
    request = StateRequest(
        oxidation_overrides=(OxidationStateOverride(0, 4),),
        oxidation_candidates=(
            _candidate("eu-ii", (2,)),
            _candidate("eu-iii", (3,)),
        ),
    )

    result = infer_oxidation_states(facts, request)[0]

    assert result.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert result.resolution is ResolutionStatus.CONFLICT
    assert result.findings[0].code == "HARD-007.OS_OVERRIDE_EMPTY_INTERSECTION"


def test_partial_candidate_remains_ambiguous() -> None:
    facts = _facts("[Eu]O")
    request = StateRequest(
        oxidation_candidates=(
            _candidate("partial", (3, None), CoverageStatus.PARTIAL),
        ),
    )

    result = infer_oxidation_states(facts, request)[0]

    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert result.resolution is ResolutionStatus.AMBIGUOUS


def test_invalid_override_indices_return_typed_results() -> None:
    facts = _facts("[Eu]")

    for atom_index in (-2, 1):
        result = infer_oxidation_states(
            facts,
            StateRequest(
                oxidation_overrides=(
                    OxidationStateOverride(atom_index, 3),
                ),
                oxidation_candidates=(_candidate("eu-iii", (3,)),),
            ),
        )[0]

        assert result.admissibility is ScientificAdmissibility.INADMISSIBLE
        assert result.findings[0].code == "HARD-004.OS_OVERRIDE_INVALID"


def test_conflicting_duplicate_overrides_return_conflict() -> None:
    facts = _facts("[Eu]")
    request = StateRequest(
        oxidation_overrides=(
            OxidationStateOverride(0, 2),
            OxidationStateOverride(0, 3),
        ),
        oxidation_candidates=(_candidate("eu-iii", (3,)),),
    )

    result = infer_oxidation_states(facts, request)[0]

    assert result.resolution is ResolutionStatus.CONFLICT
    assert result.findings[0].code == "HARD-007.OS_OVERRIDE_CONFLICT"
