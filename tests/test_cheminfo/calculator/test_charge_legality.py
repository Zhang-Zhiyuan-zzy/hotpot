"""HARD-001 and HARD-002 scientific legality tests."""

import hotpot

from hotpot.cheminfo.calculator.charge import (
    AtomFormalChargeEvidence,
    ComponentChargeEvidence,
    NetChargeCandidateEvidence,
)
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    CoverageStatus,
    MolecularFactsIdentity,
    ResolutionStatus,
    ScientificAdmissibility,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.legality import (
    _validate_charge_conservation,
    _validate_facts_identity,
)
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts


def _provenance(identity: str) -> CandidateProvenance:
    return CandidateProvenance(
        identity=identity,
        authority=StateValueAuthority.VERIFIED_EXPLICIT,
        source_identity="legality-test",
    )


def _charge(value: int) -> NetChargeCandidateEvidence:
    return NetChargeCandidateEvidence(_provenance(f"q:{value}"), value)


def test_exact_identity_passes_and_mismatch_is_typed() -> None:
    expected = MolecularFactsIdentity("expected")

    passed = _validate_facts_identity(expected, expected)
    failed = _validate_facts_identity(expected, MolecularFactsIdentity("other"))

    assert passed.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert failed.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert failed.findings[0].code == "HARD-001.FACTS_IDENTITY_MISMATCH"


def test_explicit_net_charge_is_legal_without_lewis_evidence() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Fe]", "smi"))

    assessment = _validate_charge_conservation(facts, _charge(2))

    assert assessment.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert assessment.findings == ()


def test_complete_atom_and_component_charge_sums_are_enforced() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Na+].[Cl-]", "smi"))
    component_ids = tuple(component.identity for component in facts.source_components)
    atom_evidence = AtomFormalChargeEvidence(
        _provenance("atom-charges"),
        charges=(1, -1),
        coverage=CoverageStatus.COMPLETE,
    )
    component_evidence = ComponentChargeEvidence(
        _provenance("component-charges"),
        component_charges=((component_ids[0], 1), (component_ids[1], -1)),
        coverage=CoverageStatus.COMPLETE,
    )

    passed = _validate_charge_conservation(
        facts,
        _charge(0),
        atom_evidence=atom_evidence,
        component_evidence=component_evidence,
    )
    failed = _validate_charge_conservation(
        facts,
        _charge(1),
        atom_evidence=atom_evidence,
        component_evidence=component_evidence,
    )

    assert passed.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert failed.terminal_status is ResolutionStatus.CONFLICT
    assert {finding.code for finding in failed.findings} == {
        "HARD-002.ATOM_CHARGE_SUM_MISMATCH",
        "HARD-002.COMPONENT_CHARGE_SUM_MISMATCH",
    }


def test_partial_charge_evidence_does_not_assert_a_complete_sum() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("CCO", "smi"))
    partial = AtomFormalChargeEvidence(
        _provenance("partial"),
        charges=(1, None, None),
        coverage=CoverageStatus.PARTIAL,
    )

    assessment = _validate_charge_conservation(
        facts,
        _charge(0),
        atom_evidence=partial,
    )

    assert assessment.admissibility is ScientificAdmissibility.ADMISSIBLE


def test_complete_component_coverage_requires_exact_component_ids() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Na+].[Cl-]", "smi"))
    incomplete = ComponentChargeEvidence(
        _provenance("components"),
        component_charges=((facts.source_components[0].identity, 0),),
        coverage=CoverageStatus.COMPLETE,
    )

    assessment = _validate_charge_conservation(
        facts,
        _charge(0),
        component_evidence=incomplete,
    )

    assert assessment.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert assessment.terminal_status is None
    assert assessment.findings[0].code == "HARD-002.COMPONENT_COVERAGE_INVALID"
