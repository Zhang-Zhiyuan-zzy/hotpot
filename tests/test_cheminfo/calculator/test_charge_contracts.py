"""Public contracts for independent charge quantities."""

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from hotpot.cheminfo.calculator.charge import (
    AtomFormalChargeEvidence,
    ComponentChargeEvidence,
    LewisFormalChargeCandidate,
    LewisFormalChargeResult,
    NetChargeCandidateEvidence,
    NetChargeResult,
)
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    ChemicalConfidence,
    CoverageStatus,
    MolecularFactsIdentity,
    ResolutionStatus,
    ScientificAdmissibility,
    StateValueAuthority,
)


def _provenance(identity: str) -> CandidateProvenance:
    return CandidateProvenance(
        identity=identity,
        authority=StateValueAuthority.VERIFIED_EXPLICIT,
        source_identity="test-source",
    )


def test_net_charge_and_formal_charge_are_distinct_contracts() -> None:
    net_candidate = NetChargeCandidateEvidence(_provenance("q0"), charge=0)
    atom_evidence = AtomFormalChargeEvidence(
        _provenance("atom-formal"),
        charges=(0, 1, -1),
        coverage=CoverageStatus.COMPLETE,
    )
    component_evidence = ComponentChargeEvidence(
        _provenance("component-charge"),
        component_charges=(("component-0", 0),),
        coverage=CoverageStatus.COMPLETE,
    )

    assert net_candidate.charge == 0
    assert atom_evidence.charges == (0, 1, -1)
    assert component_evidence.component_charges == (("component-0", 0),)


def test_charge_result_contracts_are_frozen_and_json_serializable() -> None:
    facts_identity = MolecularFactsIdentity("facts")
    net_candidate = NetChargeCandidateEvidence(_provenance("q-1"), charge=-1)
    net_result = NetChargeResult(
        identity="net-result",
        facts_identity=facts_identity,
        candidates=(net_candidate,),
        admissibility=ScientificAdmissibility.ADMISSIBLE,
        resolution=ResolutionStatus.UNIQUE,
        confidence=ChemicalConfidence.AUTHORITATIVE,
        findings=(),
    )
    lewis_candidate = LewisFormalChargeCandidate(
        provenance=_provenance("lewis-1"),
        net_charge_candidate_identity="q-1",
        atom_formal_charges=(0, -1),
        bond_orders=(1.0,),
        nonbonding_electrons=(0, 6),
        radical_electrons=(0, 0),
        component_charges=(("component-0", -1),),
    )
    lewis_result = LewisFormalChargeResult(
        identity="lewis-result",
        facts_identity=facts_identity,
        candidates=(lewis_candidate,),
        admissibility=ScientificAdmissibility.ADMISSIBLE,
        resolution=ResolutionStatus.UNIQUE,
        confidence=ChemicalConfidence.HIGH,
        findings=(),
    )

    with pytest.raises(FrozenInstanceError):
        net_result.identity = "changed"
    encoded = json.dumps(
        {"net": asdict(net_result), "lewis": asdict(lewis_result)},
        sort_keys=True,
        default=lambda item: item.value,
    )
    assert '"charge": -1' in encoded
    assert '"atom_formal_charges": [0, -1]' in encoded
