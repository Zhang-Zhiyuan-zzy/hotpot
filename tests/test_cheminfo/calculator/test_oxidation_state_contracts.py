"""Contracts for convention-labelled oxidation-state evidence."""

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    ChemicalConfidence,
    CoverageStatus,
    MolecularFactsIdentity,
    ResolutionStatus,
    ScientificAdmissibility,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CoordinationCountingCandidate,
    CoordinationElectronCountingResult,
    CoordinationFragmentAssignment,
    CountingConvention,
    OxidationStateCandidateEvidence,
    OxidationStateInferenceResult,
)


def _provenance(identity: str) -> CandidateProvenance:
    return CandidateProvenance(
        identity=identity,
        authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
        source_identity="oxidation-test-source",
    )


def test_counting_convention_is_carried_by_every_assignment() -> None:
    fragment = CoordinationFragmentAssignment(
        component_identity="ligand",
        ionic_bookkeeping_charge=-1,
        lxz_class=None,
        donated_electrons=2,
    )
    ionic = CoordinationCountingCandidate(
        _provenance("ionic"),
        convention=CountingConvention.IONIC,
        fragments=(fragment,),
    )
    cbc = CoordinationCountingCandidate(
        _provenance("cbc"),
        convention=CountingConvention.CBC,
        fragments=(fragment,),
    )

    assert ionic.convention is CountingConvention.IONIC
    assert cbc.convention is CountingConvention.CBC
    assert ionic != cbc


def test_oxidation_results_are_frozen_and_serializable() -> None:
    facts_identity = MolecularFactsIdentity("facts")
    counting = CoordinationElectronCountingResult(
        identity="counting-result",
        facts_identity=facts_identity,
        candidates=(),
        admissibility=ScientificAdmissibility.NOT_EVALUATED,
        resolution=ResolutionStatus.UNRESOLVED,
        confidence=ChemicalConfidence.UNRESOLVED,
        findings=(),
    )
    candidate = OxidationStateCandidateEvidence(
        provenance=_provenance("eu-iii"),
        convention=CountingConvention.IONIC,
        atom_states=(3, None),
        coverage=CoverageStatus.PARTIAL,
        coordination_counting_candidate_identity="ionic-counting",
    )
    result = OxidationStateInferenceResult(
        identity="oxidation-result",
        facts_identity=facts_identity,
        candidates=(candidate,),
        admissibility=ScientificAdmissibility.ADMISSIBLE,
        resolution=ResolutionStatus.UNIQUE,
        confidence=ChemicalConfidence.AUTHORITATIVE,
        findings=(),
    )

    with pytest.raises(FrozenInstanceError):
        result.identity = "changed"
    encoded = json.dumps(
        {"counting": asdict(counting), "oxidation": asdict(result)},
        sort_keys=True,
        default=lambda item: item.value,
    )
    assert '"convention": "ionic"' in encoded
    assert '"atom_states": [3, null]' in encoded
