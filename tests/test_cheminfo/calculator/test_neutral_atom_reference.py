"""PROVIDER-003 isolated-neutral-atom registry tests."""

import json
from importlib.resources import files

import hotpot
import pytest

from hotpot.cheminfo.calculator.charge import NetChargeCandidateEvidence
from hotpot.cheminfo.calculator.contracts import (
    CandidateProvenance,
    ScientificAdmissibility,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import neutral_atom_reference
from hotpot.cheminfo.calculator.legality import _validate_molecular_spin
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts


@pytest.mark.parametrize(
    ("atomic_number", "configuration", "two_s"),
    (
        (6, "1s^22s^22p^2", 2),
        (24, "[Ar]3d^54s", 6),
        (26, "[Ar]3d^64s^2", 4),
        (63, "[Xe]4f^76s^2", 7),
        (64, "[Xe]4f^75d6s^2", 8),
        (92, "[Rn]5f^36d7s^2", 4),
        (95, "[Rn]5f^77s^2", 7),
    ),
)
def test_representative_references_match_reviewed_snapshot(
    atomic_number: int,
    configuration: str,
    two_s: int,
) -> None:
    reference = neutral_atom_reference(atomic_number)

    assert reference is not None
    assert reference.electron_count == atomic_number
    assert reference.ground_configuration == configuration
    assert reference.two_s == two_s
    assert reference.registry_version == "nist-asd-2024-accessed-2026-10-16"
    assert len(reference.content_digest) == 64


@pytest.mark.parametrize("atomic_number", (82, 106, 107, 108, 109, 118))
def test_unreviewed_spin_only_reference_is_explicitly_unavailable(
    atomic_number: int,
) -> None:
    assert neutral_atom_reference(atomic_number) is None


@pytest.mark.parametrize("atomic_number", (0, 119))
def test_atomic_number_must_be_in_periodic_table(atomic_number: int) -> None:
    with pytest.raises(ValueError):
        neutral_atom_reference(atomic_number)


def test_registry_covers_every_atomic_number() -> None:
    results = tuple(neutral_atom_reference(z) for z in range(1, 119))

    assert len(results) == 118
    assert sum(reference is not None for reference in results) == 104


def test_registry_records_exact_source_snapshot() -> None:
    resource = files(
        "hotpot.cheminfo.calculator.electronic_state"
    ).joinpath("data", "neutral_atom_references.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))

    assert tuple(record["atomic_number"] for record in payload["records"]) == (
        tuple(range(1, 119))
    )
    assert payload["source"]["source_snapshot_sha256"] == (
        "e5840c9673ce08e44f96edc9bb382159c07448749e2a6f668d576c6a1b60c40b"
    )
    assert "H-Ds%20I" in payload["source"]["url"]


def test_atomic_reference_passes_molecular_spin_legality_before_projection() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[C]", "smi"))
    reference = neutral_atom_reference(6)
    assert reference is not None
    charge = NetChargeCandidateEvidence(
        CandidateProvenance(
            "carbon-neutral",
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "PROVIDER-003",
        ),
        0,
    )

    assessment = _validate_molecular_spin(
        facts,
        charge,
        electron_count=reference.electron_count,
        two_s=reference.two_s,
        multiplicity=reference.two_s + 1,
        pure_spin=True,
    )

    assert assessment.admissibility is ScientificAdmissibility.ADMISSIBLE
