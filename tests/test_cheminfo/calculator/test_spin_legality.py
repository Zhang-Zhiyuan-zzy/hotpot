"""HARD-006 backend-independent molecular spin legality tests."""

import hotpot
import pytest

from hotpot.cheminfo.calculator.charge import NetChargeCandidateEvidence
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    ScientificAdmissibility,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import ElectronicStateCandidate
from hotpot.cheminfo.calculator.legality import (
    _validate_electronic_state_candidate,
    _validate_molecular_spin,
)
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts


def _provenance(identity: str) -> CandidateProvenance:
    return CandidateProvenance(
        identity=identity,
        authority=StateValueAuthority.VERIFIED_EXPLICIT,
        source_identity="spin-legality-test",
    )


def _charge(value: int) -> NetChargeCandidateEvidence:
    return NetChargeCandidateEvidence(_provenance(f"q:{value}"), value)


@pytest.mark.parametrize(
    ("smiles", "charge", "two_s", "multiplicity"),
    (
        ("C", 0, 0, 1),
        ("[NH4+]", 1, 0, 1),
        ("[N]", 0, 1, 2),
        ("[O-]", -1, 1, 2),
    ),
)
def test_electron_parity_accepts_legal_molecular_spin(
    smiles: str,
    charge: int,
    two_s: int,
    multiplicity: int,
) -> None:
    facts = extract_molecular_facts(hotpot.read_mol(smiles, "smi"))
    electron_count = facts.nuclear_charge_sum - charge

    assessment = _validate_molecular_spin(
        facts,
        _charge(charge),
        electron_count=electron_count,
        two_s=two_s,
        multiplicity=multiplicity,
        pure_spin=True,
    )

    assert assessment.admissibility is ScientificAdmissibility.ADMISSIBLE


@pytest.mark.parametrize(
    ("charge", "electron_count", "two_s", "multiplicity", "code"),
    (
        (20, -10, 0, 1, "HARD-006.NEGATIVE_ELECTRON_COUNT"),
        (0, 9, 0, 1, "HARD-006.ELECTRON_COUNT_MISMATCH"),
        (0, 10, -1, 0, "HARD-006.TWO_S_OUT_OF_RANGE"),
        (0, 10, 11, 12, "HARD-006.TWO_S_OUT_OF_RANGE"),
        (0, 10, 1, 2, "HARD-006.PARITY_MISMATCH"),
        (0, 10, 2, 2, "HARD-006.MULTIPLICITY_MISMATCH"),
    ),
)
def test_illegal_spin_states_return_reason_coded_findings(
    charge: int,
    electron_count: int,
    two_s: int,
    multiplicity: int,
    code: str,
) -> None:
    facts = extract_molecular_facts(hotpot.read_mol("C", "smi"))

    assessment = _validate_molecular_spin(
        facts,
        _charge(charge),
        electron_count=electron_count,
        two_s=two_s,
        multiplicity=multiplicity,
        pure_spin=True,
    )

    assert assessment.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert assessment.findings[0].code == code


def test_non_pure_spin_candidate_does_not_invent_multiplicity() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[N]", "smi"))

    assessment = _validate_molecular_spin(
        facts,
        _charge(0),
        electron_count=7,
        two_s=1,
        multiplicity=None,
        pure_spin=False,
    )

    assert assessment.admissibility is ScientificAdmissibility.ADMISSIBLE


def test_electronic_candidate_must_reference_the_same_charge_candidate() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("C", "smi"))
    candidate = ElectronicStateCandidate(
        provenance=_provenance("singlet"),
        net_charge_candidate_identity="q:other",
        oxidation_state_candidate_identity=None,
        electron_count=10,
        two_s=0,
        multiplicity=1,
        pure_spin=True,
    )

    assessment = _validate_electronic_state_candidate(
        facts,
        _charge(0),
        candidate,
    )

    assert assessment.findings[0].code == "HARD-007.NET_CHARGE_REFERENCE_MISMATCH"
