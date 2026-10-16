"""Contracts separating molecular scientific spin from backend UHF."""

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    ChemicalConfidence,
    MolecularFactsIdentity,
    ResolutionStatus,
    ScientificAdmissibility,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateCandidate,
    ElectronicStateCandidateEvidence,
    ElectronicStateInferenceResult,
    NeutralAtomReference,
    SpinContributionCandidate,
    SpinContributionScope,
)


def _provenance(identity: str) -> CandidateProvenance:
    return CandidateProvenance(
        identity=identity,
        authority=StateValueAuthority.VERIFIED_EXPLICIT,
        source_identity="spin-test-source",
    )


def test_local_spin_contribution_is_not_a_molecular_state() -> None:
    contribution = SpinContributionCandidate(
        provenance=_provenance("eu-local-spin"),
        scope=SpinContributionScope.ATOM,
        subject_identity="atom:0",
        parent_candidate_ids=("eu-iii",),
        two_s=6,
        pure_spin=True,
    )
    molecular = ElectronicStateCandidate(
        provenance=_provenance("molecular-spin"),
        net_charge_candidate_identity="q0",
        oxidation_state_candidate_identity="eu-iii",
        electron_count=100,
        two_s=6,
        multiplicity=7,
        pure_spin=True,
    )

    assert contribution.scope is SpinContributionScope.ATOM
    assert contribution.subject_identity == "atom:0"
    assert molecular.net_charge_candidate_identity == "q0"


def test_scientific_state_contracts_are_frozen_and_serializable() -> None:
    evidence = ElectronicStateCandidateEvidence(
        provenance=_provenance("explicit-spin"),
        two_s=1,
        multiplicity=2,
        pure_spin=True,
    )
    candidate = ElectronicStateCandidate(
        provenance=evidence.provenance,
        net_charge_candidate_identity="q0",
        oxidation_state_candidate_identity=None,
        electron_count=9,
        two_s=evidence.two_s,
        multiplicity=evidence.multiplicity,
        pure_spin=evidence.pure_spin,
    )
    result = ElectronicStateInferenceResult(
        identity="electronic-result",
        facts_identity=MolecularFactsIdentity("facts"),
        candidates=(candidate,),
        admissibility=ScientificAdmissibility.ADMISSIBLE,
        resolution=ResolutionStatus.UNIQUE,
        confidence=ChemicalConfidence.AUTHORITATIVE,
        findings=(),
    )
    reference = NeutralAtomReference(
        atomic_number=8,
        electron_count=8,
        ground_configuration="1s2 2s2 2p4",
        two_s=2,
        source_identity="nist-asd",
        registry_version="test-v1",
        content_digest="digest",
    )

    with pytest.raises(FrozenInstanceError):
        result.identity = "changed"
    encoded = json.dumps(
        {"result": asdict(result), "reference": asdict(reference)},
        sort_keys=True,
        default=lambda item: item.value,
    )
    assert '"two_s": 1' in encoded
    assert '"ground_configuration": "1s2 2s2 2p4"' in encoded
