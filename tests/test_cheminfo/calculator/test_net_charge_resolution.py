"""Independent explicit/source net-charge resolution tests."""

import hotpot

from hotpot.cheminfo.calculator.charge import (
    AtomFormalChargeEvidence,
    NetChargeCandidateEvidence,
    NetChargeResult,
    resolve_net_charge,
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
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)


def _candidate(identity: str, charge: int) -> NetChargeCandidateEvidence:
    return NetChargeCandidateEvidence(
        provenance=CandidateProvenance(
            identity=identity,
            authority=StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            source_identity="charge-resolution-test",
        ),
        charge=charge,
    )


def test_empty_request_remains_unresolved() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe]", "smi"))

    result = resolve_net_charge(facts)[0]

    assert result.candidates == ()
    assert result.admissibility is ScientificAdmissibility.NOT_EVALUATED
    assert result.resolution is ResolutionStatus.UNRESOLVED


def test_explicit_charge_resolves_without_lewis_or_metal_defaults() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe]", "smi"))

    result = resolve_net_charge(facts, StateRequest(net_charge=2))[0]

    assert tuple(candidate.charge for candidate in result.candidates) == (2,)
    assert result.resolution is ResolutionStatus.UNIQUE
    assert result.confidence is ChemicalConfidence.AUTHORITATIVE


def test_multiple_allowed_candidates_are_ambiguous_not_conflicting() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe]", "smi"))
    request = StateRequest(
        charge_candidates=(_candidate("q2", 2), _candidate("q3", 3))
    )

    result = resolve_net_charge(facts, request)[0]

    assert result.resolution is ResolutionStatus.AMBIGUOUS
    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE


def test_scalar_intersects_or_conflicts_with_candidate_domain() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe]", "smi"))
    domain = (_candidate("q2", 2), _candidate("q3", 3))

    matched = resolve_net_charge(
        facts,
        StateRequest(net_charge=2, charge_candidates=domain),
    )[0]
    conflicted = resolve_net_charge(
        facts,
        StateRequest(net_charge=1, charge_candidates=domain),
    )[0]

    assert {candidate.charge for candidate in matched.candidates} == {2}
    assert matched.resolution is ResolutionStatus.UNIQUE
    assert conflicted.resolution is ResolutionStatus.CONFLICT
    assert conflicted.admissibility is ScientificAdmissibility.INADMISSIBLE


def test_complete_formal_charge_evidence_constrains_explicit_net_charge() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[NH4+]", "smi"))
    atom_evidence = AtomFormalChargeEvidence(
        provenance=CandidateProvenance(
            identity="atom-formal",
            authority=StateValueAuthority.VERIFIED_EXPLICIT,
            source_identity="request",
        ),
        charges=(1,),
        coverage=CoverageStatus.COMPLETE,
    )

    result = resolve_net_charge(
        facts,
        StateRequest(net_charge=0, atom_formal_charges=atom_evidence),
    )[0]

    assert result.resolution is ResolutionStatus.CONFLICT
    assert result.candidates == ()


def test_provided_result_with_wrong_facts_identity_is_rejected() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("C", "smi"))
    provided = NetChargeResult(
        identity="provided",
        facts_identity=MolecularFactsIdentity("other"),
        candidates=(_candidate("q0", 0),),
        admissibility=ScientificAdmissibility.ADMISSIBLE,
        resolution=ResolutionStatus.UNIQUE,
        confidence=ChemicalConfidence.AUTHORITATIVE,
        findings=(),
    )

    result = resolve_net_charge(facts, provided=(provided,))[1]

    assert result.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert result.findings[0].code == "HARD-001.FACTS_IDENTITY_MISMATCH"
