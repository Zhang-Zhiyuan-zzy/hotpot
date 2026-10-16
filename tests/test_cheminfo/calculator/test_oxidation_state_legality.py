"""HARD-004 and HARD-005 scientific legality tests."""

from typing import Optional, Tuple

import hotpot

from hotpot.cheminfo.calculator.charge import NetChargeCandidateEvidence
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    CoverageStatus,
    ResolutionStatus,
    ScientificAdmissibility,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.legality import (
    _validate_counting_candidate,
    _validate_counting_reference,
    _validate_oxidation_overrides,
    _validate_oxidation_state_evidence,
    _validate_oxidation_state_conservation,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CoordinationCountingCandidate,
    CoordinationFragmentAssignment,
    CountingConvention,
    OxidationStateCandidateEvidence,
    OxidationStateOverride,
)
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts


def _provenance(identity: str) -> CandidateProvenance:
    return CandidateProvenance(
        identity=identity,
        authority=StateValueAuthority.VERIFIED_EXPLICIT,
        source_identity="oxidation-legality-test",
    )


def _charge(value: int) -> NetChargeCandidateEvidence:
    return NetChargeCandidateEvidence(_provenance(f"q:{value}"), value)


def _oxidation_candidate(
    identity: str,
    states: Tuple[Optional[int], ...],
    coverage: CoverageStatus,
    convention: CountingConvention = CountingConvention.IONIC,
    counting_identity: Optional[str] = None,
) -> OxidationStateCandidateEvidence:
    return OxidationStateCandidateEvidence(
        provenance=_provenance(identity),
        convention=convention,
        atom_states=states,
        coverage=coverage,
        coordination_counting_candidate_identity=counting_identity,
    )


def test_complete_oxidation_states_obey_global_charge_conservation() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe].[Cl-].[Cl-]", "smi"))
    candidate = _oxidation_candidate(
        "fe-ii",
        (2, -1, -1),
        CoverageStatus.COMPLETE,
    )

    passed = _validate_oxidation_state_conservation(facts, _charge(0), candidate)
    failed = _validate_oxidation_state_conservation(facts, _charge(1), candidate)

    assert passed.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert failed.terminal_status is ResolutionStatus.CONFLICT
    assert failed.findings[0].code == "HARD-004.OS_SUM_MISMATCH"


def test_partial_oxidation_states_are_not_forced_to_sum() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe].[Cl-].[Cl-]", "smi"))
    partial = _oxidation_candidate(
        "partial",
        (2, None, None),
        CoverageStatus.PARTIAL,
    )

    assessment = _validate_oxidation_state_conservation(
        facts,
        _charge(3),
        partial,
    )

    assert assessment.admissibility is ScientificAdmissibility.ADMISSIBLE


def test_complete_oxidation_state_cannot_omit_implicit_hydrogen() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("O", "smi"))
    candidate = _oxidation_candidate(
        "oxygen-only",
        (-2,),
        CoverageStatus.COMPLETE,
    )

    assessment = _validate_oxidation_state_evidence(facts, candidate)

    assert assessment.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert assessment.findings[0].code == (
        "HARD-004.OS_IMPLICIT_HYDROGEN_UNCOVERED"
    )


def test_counting_candidate_rejects_mixed_convention_fields() -> None:
    mixed = CoordinationCountingCandidate(
        provenance=_provenance("mixed"),
        convention=CountingConvention.CBC,
        fragments=(
            CoordinationFragmentAssignment(
                component_identity="ligand",
                ionic_bookkeeping_charge=-1,
                lxz_class="X",
                donated_electrons=1,
            ),
        ),
    )

    assessment = _validate_counting_candidate(mixed)

    assert assessment.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert assessment.findings[0].code == "HARD-005.MIXED_COUNTING_FIELDS"


def test_alternative_counting_conventions_are_not_a_conflict() -> None:
    ionic = CoordinationCountingCandidate(
        provenance=_provenance("ionic"),
        convention=CountingConvention.IONIC,
        fragments=(
            CoordinationFragmentAssignment("ligand", -1, None, 2),
        ),
    )
    cbc = CoordinationCountingCandidate(
        provenance=_provenance("cbc"),
        convention=CountingConvention.CBC,
        fragments=(
            CoordinationFragmentAssignment("ligand", None, "X", 1),
        ),
    )

    assert (
        _validate_counting_candidate(ionic).admissibility
        is ScientificAdmissibility.ADMISSIBLE
    )
    assert (
        _validate_counting_candidate(cbc).admissibility
        is ScientificAdmissibility.ADMISSIBLE
    )


def test_counting_reference_must_exist_and_use_the_same_convention() -> None:
    ionic = CoordinationCountingCandidate(
        provenance=_provenance("ionic"),
        convention=CountingConvention.IONIC,
        fragments=(),
    )
    oxidation = _oxidation_candidate(
        "os",
        (3,),
        CoverageStatus.COMPLETE,
        convention=CountingConvention.CBC,
        counting_identity="ionic",
    )

    assessment = _validate_counting_reference(oxidation, (ionic,))

    assert assessment.findings[0].code == "HARD-005.COUNTING_CONVENTION_MISMATCH"


def test_oxidation_overrides_require_one_complete_extension() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe].[Fe]", "smi"))
    candidates = (
        _oxidation_candidate("mixed-a", (2, 3), CoverageStatus.COMPLETE),
        _oxidation_candidate("mixed-b", (3, 2), CoverageStatus.COMPLETE),
    )

    passed = _validate_oxidation_overrides(
        facts,
        (OxidationStateOverride(atom_index=0, oxidation_state=3),),
        candidates,
    )
    failed = _validate_oxidation_overrides(
        facts,
        (OxidationStateOverride(atom_index=0, oxidation_state=4),),
        candidates,
    )

    assert passed.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert failed.terminal_status is ResolutionStatus.CONFLICT


def test_override_extension_is_not_evaluated_without_complete_candidates() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe]", "smi"))

    assessment = _validate_oxidation_overrides(
        facts,
        (OxidationStateOverride(atom_index=0, oxidation_state=3),),
        (),
    )

    assert assessment.admissibility is ScientificAdmissibility.NOT_EVALUATED
    assert assessment.terminal_status is None
