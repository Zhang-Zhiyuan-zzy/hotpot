"""Input contracts for scientific-state orchestration."""

from dataclasses import FrozenInstanceError

import pytest

from hotpot.cheminfo.calculator.charge import NetChargeCandidateEvidence
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateCandidateEvidence,
)
from hotpot.cheminfo.calculator.state_resolver import (
    ProvidedScientificResults,
    StateRequest,
)


def _provenance(identity: str) -> CandidateProvenance:
    return CandidateProvenance(
        identity=identity,
        authority=StateValueAuthority.VERIFIED_EXPLICIT,
        source_identity="request-test",
    )


def test_empty_request_preserves_every_quantity_as_unspecified() -> None:
    request = StateRequest()

    assert request.net_charge is None
    assert request.two_s is None
    assert request.multiplicity is None
    assert request.charge_candidates == ()
    assert request.oxidation_candidates == ()
    assert request.electronic_candidates == ()


def test_scalar_and_candidate_domain_are_independent_request_fields() -> None:
    q0 = NetChargeCandidateEvidence(_provenance("q0"), charge=0)
    q1 = NetChargeCandidateEvidence(_provenance("q1"), charge=1)
    spin = ElectronicStateCandidateEvidence(
        _provenance("doublet"),
        two_s=1,
        multiplicity=2,
        pure_spin=True,
    )
    request = StateRequest(
        net_charge=1,
        charge_candidates=(q0, q1),
        electronic_candidates=(spin,),
    )

    assert request.net_charge == 1
    assert tuple(candidate.charge for candidate in request.charge_candidates) == (
        0,
        1,
    )
    assert request.electronic_candidates == (spin,)
    with pytest.raises(FrozenInstanceError):
        request.net_charge = 0


def test_precomputed_results_default_to_empty_independent_collections() -> None:
    first = ProvidedScientificResults()
    second = ProvidedScientificResults()

    assert first == second
    assert first.net_charge_results == ()
    assert first.electronic_state_results == ()
