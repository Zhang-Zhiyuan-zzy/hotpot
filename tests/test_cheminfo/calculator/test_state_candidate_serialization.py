"""Lossless tagged serialization tests for every state candidate type."""

import json
from typing import get_args

import pytest

from hotpot.cheminfo.calculator.charge import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
)
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    CoverageStatus,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
    SpinContributionScope,
)
from hotpot.cheminfo.calculator.oxidation_state import (
    CoordinationCountingCandidate,
    CoordinationFragmentAssignment,
    CountingConvention,
    OxidationStateCandidateEvidence,
)
from hotpot.cheminfo.calculator.state_composition import (
    DerivedStateValue,
    StateCandidate,
)
from hotpot.cheminfo.calculator.state_composition._serialization import (
    _SUPPORTED_STATE_CANDIDATE_TYPES,
    _state_candidate_from_payload,
    _state_candidate_to_payload,
)


def _provenance(identity: str) -> CandidateProvenance:
    return CandidateProvenance(
        identity,
        StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
        "serialization-test",
        ("evidence",),
        ("assumption",),
    )


@pytest.mark.parametrize(
    "candidate",
    (
        NetChargeCandidateEvidence(_provenance("charge"), 1),
        LewisFormalChargeCandidate(
            _provenance("lewis"),
            "charge",
            (0, -1),
            (1.0,),
            (2, 6),
            (0, 0),
            (("component", -1),),
        ),
        CoordinationCountingCandidate(
            _provenance("counting"),
            CountingConvention.IONIC,
            (
                CoordinationFragmentAssignment(
                    "component",
                    -1,
                    None,
                    2,
                ),
            ),
        ),
        OxidationStateCandidateEvidence(
            _provenance("oxidation"),
            CountingConvention.IONIC,
            (3, None),
            CoverageStatus.PARTIAL,
            "counting",
        ),
        SpinContributionCandidate(
            _provenance("spin-contribution"),
            SpinContributionScope.ATOM,
            "atom:0",
            ("oxidation",),
            5,
            True,
        ),
        ElectronicStateCandidate(
            _provenance("electronic"),
            "charge",
            "oxidation",
            24,
            2,
            3,
            True,
        ),
        DerivedStateValue(
            _provenance("derived"),
            ScientificQuantity.NET_CHARGE,
            2,
            "HARD-002.test-v1",
            ("oxidation", "counting"),
            CountingConvention.IONIC,
        ),
    ),
)
def test_state_candidate_payload_round_trip(candidate) -> None:
    json_payload = json.loads(
        json.dumps(_state_candidate_to_payload(candidate), allow_nan=False)
    )
    restored = _state_candidate_from_payload(
        json_payload
    )

    assert restored == candidate
    assert type(restored) is type(candidate)


def test_candidate_codec_matches_union_contract() -> None:
    assert frozenset(_SUPPORTED_STATE_CANDIDATE_TYPES) == frozenset(
        get_args(StateCandidate)
    )


def test_optional_candidate_fields_round_trip() -> None:
    candidates = (
        CoordinationCountingCandidate(
            _provenance("counting-optionals"),
            CountingConvention.CBC,
            (
                CoordinationFragmentAssignment(
                    "component",
                    None,
                    "L2X",
                    None,
                ),
            ),
        ),
        OxidationStateCandidateEvidence(
            _provenance("oxidation-optionals"),
            CountingConvention.IONIC,
            (2,),
            CoverageStatus.COMPLETE,
            None,
        ),
        ElectronicStateCandidate(
            _provenance("electronic-optionals"),
            "charge",
            None,
            25,
            1,
            None,
            False,
        ),
    )

    for candidate in candidates:
        payload = json.loads(
            json.dumps(_state_candidate_to_payload(candidate), allow_nan=False)
        )
        assert _state_candidate_from_payload(payload) == candidate


def test_unknown_candidate_tag_is_rejected() -> None:
    payload = _state_candidate_to_payload(
        NetChargeCandidateEvidence(_provenance("charge"), 0)
    )
    payload["type"] = "unknown"

    with pytest.raises(ValueError, match="Unknown scientific-state"):
        _state_candidate_from_payload(payload)


def test_candidate_schema_drift_is_rejected() -> None:
    payload = _state_candidate_to_payload(
        NetChargeCandidateEvidence(_provenance("charge"), 0)
    )
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="schema mismatch"):
        _state_candidate_from_payload(payload)


def test_non_finite_bond_order_is_rejected() -> None:
    payload = _state_candidate_to_payload(
        LewisFormalChargeCandidate(
            _provenance("lewis"),
            "charge",
            (0, 0),
            (1.0,),
            (0, 0),
            (0, 0),
            (("component", 0),),
        )
    )
    payload["bond_orders"] = [float("nan")]

    with pytest.raises(ValueError, match="must be finite"):
        _state_candidate_from_payload(payload)
