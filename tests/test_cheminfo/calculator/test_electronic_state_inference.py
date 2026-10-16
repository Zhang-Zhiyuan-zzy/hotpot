"""Independent explicit molecular electronic-state inference tests."""

from dataclasses import replace
from typing import Optional

import hotpot

from hotpot.cheminfo.calculator.charge import resolve_net_charge
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    ChemicalConfidence,
    CoverageStatus,
    MolecularFacts,
    ResolutionStatus,
    ScientificAdmissibility,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateCandidate,
    ElectronicStateCandidateEvidence,
    ElectronicStateInferenceResult,
    infer_electronic_states,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CountingConvention,
    OxidationStateCandidateEvidence,
    OxidationStateInferenceResult,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)


def _facts(smiles: str):
    return extract_molecular_facts(hotpot.read_mol(smiles, "smi"))


def _evidence(identity: str, two_s: int) -> ElectronicStateCandidateEvidence:
    return ElectronicStateCandidateEvidence(
        provenance=CandidateProvenance(
            identity=identity,
            authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            source_identity="electronic-state-test",
        ),
        two_s=two_s,
        multiplicity=two_s + 1,
        pure_spin=True,
    )


def _provided_electronic_result(
    facts: MolecularFacts,
    charge_identity: str,
    oxidation_identity: Optional[str] = None,
) -> ElectronicStateInferenceResult:
    candidate = ElectronicStateCandidate(
        provenance=_evidence("provided-spin", 1).provenance,
        net_charge_candidate_identity=charge_identity,
        oxidation_state_candidate_identity=oxidation_identity,
        electron_count=7,
        two_s=1,
        multiplicity=2,
        pure_spin=True,
    )
    return ElectronicStateInferenceResult(
        "provided-result",
        facts.identity,
        (candidate,),
        ScientificAdmissibility.ADMISSIBLE,
        ResolutionStatus.UNIQUE,
        ChemicalConfidence.AUTHORITATIVE,
        (),
    )


def test_strict_inference_does_not_invent_spin() -> None:
    facts = _facts("C")
    charge_results = resolve_net_charge(facts, StateRequest(net_charge=0))

    result = infer_electronic_states(
        facts,
        net_charge_results=charge_results,
    )[0]

    assert result.candidates == ()
    assert result.admissibility is ScientificAdmissibility.NOT_EVALUATED
    assert result.resolution is ResolutionStatus.UNRESOLVED


def test_explicit_two_s_is_bound_to_charge_and_hard_validated() -> None:
    facts = _facts("[N]")
    request = StateRequest(net_charge=0, two_s=1)
    charge_results = resolve_net_charge(facts, request)

    result = infer_electronic_states(
        facts,
        request,
        net_charge_results=charge_results,
    )[0]

    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert result.resolution is ResolutionStatus.UNIQUE
    assert result.candidates[0].electron_count == 7
    assert result.candidates[0].two_s == 1
    assert result.candidates[0].multiplicity == 2


def test_multiplicity_only_is_an_explicit_pure_spin_constraint() -> None:
    facts = _facts("[O]")
    request = StateRequest(net_charge=0, multiplicity=3)

    result = infer_electronic_states(
        facts,
        request,
        net_charge_results=resolve_net_charge(facts, request),
    )[0]

    assert result.candidates[0].two_s == 2
    assert result.candidates[0].multiplicity == 3


def test_conflicting_explicit_spin_values_are_reported() -> None:
    facts = _facts("[N]")
    request = StateRequest(net_charge=0, two_s=1, multiplicity=4)

    result = infer_electronic_states(
        facts,
        request,
        net_charge_results=resolve_net_charge(facts, request),
    )[0]

    assert result.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert result.resolution is ResolutionStatus.CONFLICT
    assert result.findings[0].code == "HARD-007.EXPLICIT_SPIN_CONFLICT"


def test_candidate_domain_preserves_spin_ambiguity() -> None:
    facts = _facts("C")
    request = StateRequest(
        net_charge=0,
        electronic_candidates=(_evidence("singlet", 0), _evidence("triplet", 2)),
    )

    result = infer_electronic_states(
        facts,
        request,
        net_charge_results=resolve_net_charge(facts, request),
    )[0]

    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert result.resolution is ResolutionStatus.AMBIGUOUS
    assert {candidate.two_s for candidate in result.candidates} == {0, 2}


def test_illegal_spin_is_rejected_instead_of_falling_back() -> None:
    facts = _facts("C")
    request = StateRequest(net_charge=0, two_s=1)

    result = infer_electronic_states(
        facts,
        request,
        net_charge_results=resolve_net_charge(facts, request),
    )[0]

    assert result.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert result.candidates == ()
    assert result.findings[0].code == "HARD-006.PARITY_MISMATCH"


def test_provided_spin_cannot_use_inadmissible_charge_envelope() -> None:
    facts = _facts("[N]")
    charge_result = resolve_net_charge(facts, StateRequest(net_charge=0))[0]
    provided = _provided_electronic_result(
        facts,
        charge_result.candidates[0].provenance.identity,
    )

    results = infer_electronic_states(
        facts,
        net_charge_results=(
            replace(
                charge_result,
                admissibility=ScientificAdmissibility.INADMISSIBLE,
            ),
        ),
        provided=(provided,),
    )

    assert results[1].admissibility is ScientificAdmissibility.INADMISSIBLE
    assert results[1].findings[-1].code == (
        "HARD-007.NET_CHARGE_REFERENCE_MISSING"
    )


def test_provided_spin_requires_live_oxidation_candidate_reference() -> None:
    facts = _facts("[N]")
    charge_result = resolve_net_charge(facts, StateRequest(net_charge=0))[0]
    charge_identity = charge_result.candidates[0].provenance.identity
    provided = _provided_electronic_result(facts, charge_identity, "os:missing")

    results = infer_electronic_states(
        facts,
        net_charge_results=(charge_result,),
        provided=(provided,),
    )

    assert results[1].admissibility is ScientificAdmissibility.INADMISSIBLE
    assert results[1].findings[-1].code == (
        "HARD-007.OXIDATION_STATE_REFERENCE_MISSING"
    )


def test_provided_spin_accepts_hard_valid_oxidation_reference() -> None:
    facts = _facts("[N]")
    charge_result = resolve_net_charge(facts, StateRequest(net_charge=0))[0]
    charge_identity = charge_result.candidates[0].provenance.identity
    oxidation_candidate = OxidationStateCandidateEvidence(
        provenance=CandidateProvenance(
            "os:nitrogen",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        convention=CountingConvention.IONIC,
        atom_states=(0,),
        coverage=CoverageStatus.COMPLETE,
    )
    oxidation_result = OxidationStateInferenceResult(
        "os-result",
        facts.identity,
        (oxidation_candidate,),
        ScientificAdmissibility.ADMISSIBLE,
        ResolutionStatus.UNIQUE,
        ChemicalConfidence.AUTHORITATIVE,
        (),
    )
    provided = _provided_electronic_result(
        facts,
        charge_identity,
        oxidation_candidate.provenance.identity,
    )

    results = infer_electronic_states(
        facts,
        net_charge_results=(charge_result,),
        oxidation_state_results=(oxidation_result,),
        provided=(provided,),
    )

    assert results[1] == provided
