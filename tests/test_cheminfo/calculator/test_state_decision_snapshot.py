"""Replay tests for candidate-space scientific decisions."""

import json

import hotpot
import pytest

from hotpot.cheminfo.calculator import resolve_scientific_state
from hotpot.cheminfo.calculator.contracts import (
    ScientificOperationRequirements,
    ScientificQuantity,
)
from hotpot.cheminfo.calculator.state_composition._decision_snapshot import (
    _decision_snapshot_from_json,
    _decision_snapshot_to_json,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)


def _selected_bundle():
    facts = extract_molecular_facts(hotpot.read_mol("CCO", "smi"))
    requirements = ScientificOperationRequirements(
        "snapshot-test",
        frozenset(
            (
                ScientificQuantity.NET_CHARGE,
                ScientificQuantity.ELECTRONIC_STATE,
            )
        ),
        frozenset(),
    )
    return resolve_scientific_state(
        facts,
        StateRequest(net_charge=0, two_s=0),
        requirements=requirements,
    )


def test_decision_snapshot_round_trip_preserves_selected_identity() -> None:
    bundle = _selected_bundle()

    serialized = _decision_snapshot_to_json(
        bundle.candidate_space,
        bundle.scientific_resolution,
        bundle.execution_decision,
    )
    candidate_space, resolution, decision = _decision_snapshot_from_json(
        serialized
    )

    assert candidate_space == bundle.candidate_space
    assert resolution == bundle.scientific_resolution
    assert decision == bundle.execution_decision
    assert decision.selected_state_identity is not None
    assert _decision_snapshot_to_json(
        candidate_space,
        resolution,
        decision,
    ) == serialized


def test_decision_snapshot_rejects_unknown_selected_identity() -> None:
    bundle = _selected_bundle()
    payload = json.loads(
        _decision_snapshot_to_json(
            bundle.candidate_space,
            bundle.scientific_resolution,
            bundle.execution_decision,
        )
    )
    payload["execution_decision"]["selected_state_identity"] = "unknown"

    with pytest.raises(ValueError, match="absent from candidate space"):
        _decision_snapshot_from_json(json.dumps(payload))


def test_decision_snapshot_rejects_resolution_drift() -> None:
    bundle = _selected_bundle()
    payload = json.loads(
        _decision_snapshot_to_json(
            bundle.candidate_space,
            bundle.scientific_resolution,
            bundle.execution_decision,
        )
    )
    payload["scientific_resolution"]["status"] = "unresolved"

    with pytest.raises(ValueError, match="does not match"):
        _decision_snapshot_from_json(json.dumps(payload))
