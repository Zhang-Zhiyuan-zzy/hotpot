from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from hotpot.ds.detail import (
    DatasetSchemaError,
    DatasetStore,
    QualityGateError,
    assess_bundle,
)
from hotpot.ds.models import (
    STANDARD_VERSION,
    Asset,
    ClaimMembership,
    ClaimResolution,
    Dataset,
    DatasetBundle,
    DatasetRecord,
    DatasetRelease,
    Entity,
    EvidenceItem,
    Observation,
    ObservationEvidence,
    RecordEntity,
    Source,
    Structure,
    StructureAsset,
)
from hotpot.ds.properties import builtin_property_types
from hotpot.ds.quality import QualityCertificate, QualityCheck


def _structure_only_bundle(root: Path) -> DatasetBundle:
    coordinates = root / "one.xyz"
    coordinates.write_text("1\nFe ion\nFe 0 0 0\n", encoding="utf-8")
    digest = hashlib.sha256(coordinates.read_bytes()).hexdigest()
    return DatasetBundle(
        standard_version=STANDARD_VERSION,
        datasets=(Dataset("structures", "Structures", "Structure-only collection"),),
        releases=(
            DatasetRelease(
                "structures.v1",
                "structures",
                "1.0",
                "B0_collection",
                "CC0-1.0",
                "open",
                {"created_by": "test"},
                {},
            ),
        ),
        subsets=(),
        records=(DatasetRecord("structures:r1", "structures.v1", "r1", {}),),
        memberships=(),
        observation_memberships=(),
        claim_memberships=(),
        entities=(Entity("entity:one", "molecular_structure", None, {}),),
        external_identifiers=(),
        record_entities=(RecordEntity("structures:r1", "entity:one", "structure"),),
        entity_relations=(),
        assets=(Asset("asset:one", "one.xyz", "chemical/x-xyz", digest, "open", {}),),
        structures=(Structure("structure:one", "entity:one", digest, 1, 2, {}),),
        structure_assets=(StructureAsset("structure:one", "asset:one", "coordinates"),),
        descriptors=(),
        property_types=(),
        observations=(),
        observation_participants=(),
        sources=(),
        evidence=(),
        observation_evidence=(),
        claims=(),
        claim_evidence=(),
        claim_resolutions=(),
    )


def test_structure_only_data_can_be_certified_stored_and_read(tmp_path: Path) -> None:
    bundle = _structure_only_bundle(tmp_path)
    certificate = assess_bundle(bundle, tmp_path, profile="admission")

    assert certificate.passed
    store = DatasetStore(tmp_path / "datasets.sqlite3")
    result = store.ingest(bundle, certificate, tmp_path)

    assert result.records == 1
    assert store.get_structure("structure:one")["formal_charge"] == 2
    assert list(store.iter_observations()) == []
    assert store.integrity_issues() == []


def test_release_level_observation_does_not_require_a_subset(tmp_path: Path) -> None:
    base = _structure_only_bundle(tmp_path)
    observation = Observation(
        "observation:release-only",
        "structures:r1",
        "entity:one",
        "oxidation_state",
        2,
        None,
        {},
        {},
        "active",
        "E2_explicit_reported",
        {"reported_by": "test source"},
        {},
    )
    bundle = replace(
        base,
        property_types=tuple(
            row for row in builtin_property_types() if row.property_type_id == "oxidation_state"
        ),
        observations=(observation,),
        sources=(
            Source("source:test", "publication", "Test source", None, "CC0-1.0", "open", {}),
        ),
        evidence=(
            EvidenceItem(
                "evidence:test", "source:test", "reported_value", "page 1", None, "Fe(II)", {}
            ),
        ),
        observation_evidence=(
            ObservationEvidence("observation:release-only", "evidence:test", "supports"),
        ),
    )

    certificate = assess_bundle(bundle, tmp_path)
    assert certificate.passed
    store = DatasetStore(tmp_path / "release-only.sqlite3")
    store.ingest(bundle, certificate, tmp_path)
    assert store.get_observation(observation.observation_id)["subsets"] == []


def test_storage_rejects_non_admission_certificate(tmp_path: Path) -> None:
    bundle = _structure_only_bundle(tmp_path)
    benchmark_certificate = assess_bundle(bundle, tmp_path, profile="benchmark")

    with pytest.raises(QualityGateError, match="admission certificate"):
        DatasetStore(tmp_path / "datasets.sqlite3").ingest(
            bundle, benchmark_certificate, tmp_path
        )


def test_certificate_is_invalid_after_an_asset_changes(tmp_path: Path) -> None:
    bundle = _structure_only_bundle(tmp_path)
    certificate = assess_bundle(bundle, tmp_path, profile="admission")
    (tmp_path / "one.xyz").write_text("1\nchanged\nFe 1 0 0\n", encoding="utf-8")

    with pytest.raises(QualityGateError, match="asset changed|asset_integrity"):
        DatasetStore(tmp_path / "datasets.sqlite3").ingest(bundle, certificate, tmp_path)


def test_audit_detects_asset_change_after_storage(tmp_path: Path) -> None:
    bundle = _structure_only_bundle(tmp_path)
    store = DatasetStore(tmp_path / "datasets.sqlite3")
    store.ingest(bundle, assess_bundle(bundle, tmp_path), tmp_path)

    (tmp_path / "one.xyz").write_text("1\nchanged later\nFe 2 0 0\n", encoding="utf-8")

    assert any("asset digest mismatch" in issue for issue in store.integrity_issues())


def test_remote_asset_must_be_mirrored_before_admission(tmp_path: Path) -> None:
    bundle = replace(
        _structure_only_bundle(tmp_path),
        assets=(
            replace(
                _structure_only_bundle(tmp_path).assets[0],
                uri="https://example.invalid/one.xyz",
            ),
        ),
    )
    certificate = assess_bundle(bundle, tmp_path)

    integrity = next(check for check in certificate.checks if check.code == "asset_integrity")
    assert not integrity.passed
    assert "cannot be content-verified" in integrity.issues[0]


def test_incompatible_store_is_not_silently_relabelled(tmp_path: Path) -> None:
    path = tmp_path / "datasets.sqlite3"
    store = DatasetStore(path)
    store.initialize()
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE schema_metadata SET value='5' WHERE key='schema_version'"
        )

    with pytest.raises(DatasetSchemaError, match="rebuild"):
        store.initialize()
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()[0] == "5"


def test_self_issued_passing_certificate_cannot_bypass_fresh_assessment(
    tmp_path: Path,
) -> None:
    bundle = replace(
        _structure_only_bundle(tmp_path),
        record_entities=(RecordEntity("structures:r1", "missing-entity", "structure"),),
    )
    forged = QualityCertificate.issue(
        profile="admission",
        subject_digest=bundle.digest,
        checks=(QualityCheck("forged", True, 1, 0, ()),),
        assessed_levels={},
        counts={},
    )

    with pytest.raises(QualityGateError):
        DatasetStore(tmp_path / "datasets.sqlite3").ingest(bundle, forged, tmp_path)


@pytest.mark.parametrize(
    "relations",
    [
        {"claim_memberships": (ClaimMembership("missing-subset", "missing-claim"),)},
        {
            "claim_resolutions": (
                ClaimResolution(
                    "missing-claim", "missing-observation", "accepted_as_observation", {}
                ),
            )
        },
    ],
)
def test_claim_relationships_are_checked_before_storage(
    tmp_path: Path, relations: dict
) -> None:
    bundle = replace(_structure_only_bundle(tmp_path), **relations)
    certificate = assess_bundle(bundle, tmp_path, profile="admission")
    referential = next(
        check for check in certificate.checks if check.code == "referential_integrity"
    )

    assert not referential.passed
    assert referential.issue_count > 0
    with pytest.raises(QualityGateError):
        DatasetStore(tmp_path / "datasets.sqlite3").ingest(bundle, certificate, tmp_path)

