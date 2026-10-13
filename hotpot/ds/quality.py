"""Admission and quality-readiness assessment for generic datasets."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

from .levels import EvidenceLevel, QualityLevel, highest_satisfied_level, level_rank
from .models import DatasetBundle, canonical_json, load_bundle
from .properties import DEFAULT_PROPERTY_REGISTRY, PropertyRegistry, builtin_property_types


ASSESSOR_VERSION = "hotpot-ds-quality/current"
CERTIFICATE_VERSION = "hotpot-ds-certificate/current"
CERTIFICATE_ATTESTATION = "self_hashed_integrity_gate"
CERTIFICATE_ISSUER = "hotpot.ds automated local assessor"


class QualityGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class QualityCheck:
    code: str
    passed: bool
    checked_count: int
    issue_count: int
    issues: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["issues"] = list(self.issues)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QualityCheck":
        if set(value) != set(cls.__dataclass_fields__):
            raise QualityGateError("quality check fields do not match the standard")
        return cls(
            code=value["code"],
            passed=value["passed"],
            checked_count=value["checked_count"],
            issue_count=value["issue_count"],
            issues=tuple(value["issues"]),
        )


@dataclass(frozen=True)
class QualityCertificate:
    certificate_id: str
    certificate_version: str
    assessor_version: str
    attestation_type: str
    issuer: str
    profile: str
    issued_at: str
    subject_digest: str
    passed: bool
    quality_score: float
    assessed_levels: Dict[str, str]
    counts: Dict[str, int]
    checks: Tuple[QualityCheck, ...]
    certificate_sha256: str

    def _unsigned(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "certificate_version": self.certificate_version,
            "assessor_version": self.assessor_version,
            "attestation_type": self.attestation_type,
            "issuer": self.issuer,
            "profile": self.profile,
            "issued_at": self.issued_at,
            "subject_digest": self.subject_digest,
            "passed": self.passed,
            "quality_score": self.quality_score,
            "assessed_levels": self.assessed_levels,
            "counts": self.counts,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._unsigned(), "certificate_sha256": self.certificate_sha256}

    def verify_integrity(self) -> None:
        digest = hashlib.sha256(canonical_json(self._unsigned()).encode()).hexdigest()
        if digest != self.certificate_sha256:
            raise QualityGateError("quality certificate integrity check failed")

    @classmethod
    def issue(
        cls,
        profile: str,
        subject_digest: str,
        checks: Iterable[QualityCheck],
        assessed_levels: Mapping[str, str],
        counts: Mapping[str, int],
    ) -> "QualityCertificate":
        check_tuple = tuple(checks)
        passed = all(check.passed for check in check_tuple)
        score = round(100 * sum(check.passed for check in check_tuple) / len(check_tuple), 2)
        fields = {
            "certificate_id": str(uuid.uuid4()),
            "certificate_version": CERTIFICATE_VERSION,
            "assessor_version": ASSESSOR_VERSION,
            "attestation_type": CERTIFICATE_ATTESTATION,
            "issuer": CERTIFICATE_ISSUER,
            "profile": profile,
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "subject_digest": subject_digest,
            "passed": passed,
            "quality_score": score,
            "assessed_levels": dict(assessed_levels),
            "counts": dict(counts),
            "checks": check_tuple,
        }
        unsigned = {**fields, "checks": [check.to_dict() for check in check_tuple]}
        digest = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
        return cls(**fields, certificate_sha256=digest)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QualityCertificate":
        if set(value) != set(cls.__dataclass_fields__):
            raise QualityGateError("certificate fields do not match the standard")
        fields = dict(value)
        fields["checks"] = tuple(QualityCheck.from_dict(row) for row in fields["checks"])
        certificate = cls(**fields)
        certificate.verify_integrity()
        return certificate


def _check(code: str, checked: int, issues: List[str]) -> QualityCheck:
    return QualityCheck(code, not issues, checked, len(issues), tuple(issues[:25]))


def _duplicates(values: Iterable[Any]) -> List[Any]:
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates, key=repr)


def _parent_cycle_issues(rows: Iterable[Any], id_field: str, parent_field: str) -> List[str]:
    parents = {
        getattr(row, id_field): getattr(row, parent_field)
        for row in rows
        if getattr(row, parent_field) is not None
    }
    issues = []
    for start in parents:
        seen = set()
        current = start
        while current in parents:
            if current in seen:
                issues.append(f"parent cycle contains {current}")
                break
            seen.add(current)
            current = parents[current]
    return sorted(set(issues))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assess_bundle(
    bundle: DatasetBundle,
    asset_root: Union[str, Path],
    profile: str = "admission",
    property_registry: PropertyRegistry = DEFAULT_PROPERTY_REGISTRY,
) -> QualityCertificate:
    if profile not in {"admission", "benchmark", "export"}:
        raise ValueError(f"unsupported assessment profile: {profile}")
    root = Path(asset_root).resolve()
    checks = [QualityCheck("record_standard", True, 1, 0, ())]

    identities = {
        "dataset": [row.dataset_id for row in bundle.datasets],
        "release": [row.release_id for row in bundle.releases],
        "subset": [row.subset_id for row in bundle.subsets],
        "record": [row.record_id for row in bundle.records],
        "entity": [row.entity_id for row in bundle.entities],
        "asset": [row.asset_id for row in bundle.assets],
        "structure": [row.structure_id for row in bundle.structures],
        "descriptor": [row.descriptor_id for row in bundle.descriptors],
        "property_type": [row.property_type_id for row in bundle.property_types],
        "observation": [row.observation_id for row in bundle.observations],
        "claim": [row.claim_id for row in bundle.claims],
        "source": [row.source_id for row in bundle.sources],
        "evidence": [row.evidence_id for row in bundle.evidence],
    }
    uniqueness_issues = [
        f"duplicate {kind}: {identity}"
        for kind, values in identities.items()
        for identity in _duplicates(values)
    ]
    checks.append(_check("identity_uniqueness", sum(map(len, identities.values())), uniqueness_issues))

    relation_keys = {
        "dataset_release_version": [
            (row.dataset_id, row.version) for row in bundle.releases
        ],
        "release_local_record": [
            (row.release_id, row.local_id) for row in bundle.records
        ],
        "coordinate_hash": [
            row.coordinate_hash for row in bundle.structures if row.coordinate_hash is not None
        ],
        "membership": [(row.subset_id, row.record_id) for row in bundle.memberships],
        "observation_membership": [
            (row.subset_id, row.observation_id)
            for row in bundle.observation_memberships
        ],
        "claim_membership": [
            (row.subset_id, row.claim_id) for row in bundle.claim_memberships
        ],
        "external_identifier": [
            (row.entity_id, row.namespace, row.value) for row in bundle.external_identifiers
        ],
        "record_entity": [
            (row.record_id, row.entity_id, row.role) for row in bundle.record_entities
        ],
        "entity_relation": [
            (row.subject_entity_id, row.predicate, row.object_entity_id)
            for row in bundle.entity_relations
        ],
        "structure_asset": [
            (row.structure_id, row.asset_id, row.role) for row in bundle.structure_assets
        ],
        "observation_participant": [
            (row.observation_id, row.entity_id, row.role)
            for row in bundle.observation_participants
        ],
        "observation_evidence": [
            (row.observation_id, row.evidence_id, row.role)
            for row in bundle.observation_evidence
        ],
        "claim_evidence": [
            (row.claim_id, row.evidence_id, row.role) for row in bundle.claim_evidence
        ],
        "claim_resolution": [
            (row.claim_id, row.observation_id, row.resolution_type)
            for row in bundle.claim_resolutions
        ],
    }
    relation_issues = [
        f"duplicate {kind}: {identity!r}"
        for kind, values in relation_keys.items()
        for identity in _duplicates(values)
    ]
    checks.append(
        _check("relation_uniqueness", sum(map(len, relation_keys.values())), relation_issues)
    )

    cardinality_issues = []
    if not bundle.datasets:
        cardinality_issues.append("bundle has no dataset")
    if not bundle.releases:
        cardinality_issues.append("bundle has no release")
    if not bundle.records:
        cardinality_issues.append("bundle has no record")
    releases_by_dataset = {row.dataset_id for row in bundle.releases}
    records_by_release = {row.release_id for row in bundle.records}
    for dataset in bundle.datasets:
        if dataset.dataset_id not in releases_by_dataset:
            cardinality_issues.append(f"dataset {dataset.dataset_id} has no release")
    for release in bundle.releases:
        if release.release_id not in records_by_release:
            cardinality_issues.append(f"release {release.release_id} has no record")
    checks.append(
        _check(
            "minimum_cardinality",
            len(bundle.datasets) + len(bundle.releases) + len(bundle.records),
            cardinality_issues,
        )
    )

    ids = {kind: set(values) for kind, values in identities.items()}
    reference_issues = []
    for row in bundle.releases:
        if row.dataset_id not in ids["dataset"]:
            reference_issues.append(f"release {row.release_id} has missing dataset")
    subset_release = {row.subset_id: row.release_id for row in bundle.subsets}
    record_release = {row.record_id: row.release_id for row in bundle.records}
    observation_record = {
        row.observation_id: row.record_id for row in bundle.observations
    }
    claim_record = {row.claim_id: row.record_id for row in bundle.claims}
    observations_by_id = {
        row.observation_id: row for row in bundle.observations
    }
    claims_by_id = {row.claim_id: row for row in bundle.claims}
    for row in bundle.subsets:
        if row.release_id not in ids["release"]:
            reference_issues.append(f"subset {row.subset_id} has missing release")
        if row.parent_subset_id is not None and row.parent_subset_id not in ids["subset"]:
            reference_issues.append(f"subset {row.subset_id} has missing parent")
        elif (
            row.parent_subset_id is not None
            and subset_release.get(row.parent_subset_id) != row.release_id
        ):
            reference_issues.append(f"subset {row.subset_id} parent belongs to another release")
    for row in bundle.records:
        if row.release_id not in ids["release"]:
            reference_issues.append(f"record {row.record_id} has missing release")
    for row in bundle.memberships:
        if row.subset_id not in ids["subset"] or row.record_id not in ids["record"]:
            reference_issues.append(f"invalid membership: {row}")
        elif subset_release[row.subset_id] != record_release[row.record_id]:
            reference_issues.append(f"membership crosses releases: {row}")
    record_memberships = {
        (row.subset_id, row.record_id) for row in bundle.memberships
    }
    for row in bundle.observation_memberships:
        record_id = observation_record.get(row.observation_id)
        if row.subset_id not in ids["subset"] or record_id is None:
            reference_issues.append(f"invalid observation membership: {row}")
        elif subset_release[row.subset_id] != record_release[record_id]:
            reference_issues.append(f"observation membership crosses releases: {row}")
        elif (row.subset_id, record_id) not in record_memberships:
            reference_issues.append(
                f"observation membership lacks matching record membership: {row}"
            )
    for row in bundle.claim_memberships:
        record_id = claim_record.get(row.claim_id)
        if row.subset_id not in ids["subset"] or record_id is None:
            reference_issues.append(f"invalid claim membership: {row}")
        elif subset_release[row.subset_id] != record_release[record_id]:
            reference_issues.append(f"claim membership crosses releases: {row}")
        elif (row.subset_id, record_id) not in record_memberships:
            reference_issues.append(
                f"claim membership lacks matching record membership: {row}"
            )
    for row in bundle.entities:
        if row.parent_entity_id is not None and row.parent_entity_id not in ids["entity"]:
            reference_issues.append(f"entity {row.entity_id} has missing parent")
    for row in bundle.external_identifiers:
        if row.entity_id not in ids["entity"]:
            reference_issues.append(f"external identifier has missing entity {row.entity_id}")
    for row in bundle.record_entities:
        if row.record_id not in ids["record"] or row.entity_id not in ids["entity"]:
            reference_issues.append(f"invalid record entity: {row}")
    for row in bundle.entity_relations:
        if row.subject_entity_id not in ids["entity"] or row.object_entity_id not in ids["entity"]:
            reference_issues.append(f"invalid entity relation: {row}")
    for row in bundle.structures:
        if row.entity_id not in ids["entity"]:
            reference_issues.append(f"structure {row.structure_id} has missing entity")
    for row in bundle.structure_assets:
        if row.structure_id not in ids["structure"] or row.asset_id not in ids["asset"]:
            reference_issues.append(f"invalid structure asset: {row}")
    for row in bundle.descriptors:
        if row.entity_id not in ids["entity"]:
            reference_issues.append(f"descriptor {row.descriptor_id} has missing entity")
    for row in bundle.observations:
        if row.record_id not in ids["record"]:
            reference_issues.append(f"observation {row.observation_id} has missing record")
        if row.subject_entity_id not in ids["entity"]:
            reference_issues.append(f"observation {row.observation_id} has missing subject")
        if row.property_type_id not in ids["property_type"]:
            reference_issues.append(f"observation {row.observation_id} has missing property type")
    for row in bundle.observation_participants:
        if row.observation_id not in ids["observation"] or row.entity_id not in ids["entity"]:
            reference_issues.append(f"invalid observation participant: {row}")
    for row in bundle.evidence:
        if row.source_id not in ids["source"]:
            reference_issues.append(f"evidence {row.evidence_id} has missing source")
    for row in bundle.observation_evidence:
        if row.observation_id not in ids["observation"] or row.evidence_id not in ids["evidence"]:
            reference_issues.append(f"invalid observation evidence: {row}")
    for row in bundle.claims:
        if row.record_id not in ids["record"]:
            reference_issues.append(f"claim {row.claim_id} has missing record")
        if row.subject_entity_id is not None and row.subject_entity_id not in ids["entity"]:
            reference_issues.append(f"claim {row.claim_id} has missing subject")
        if row.property_type_id is not None and row.property_type_id not in ids["property_type"]:
            reference_issues.append(f"claim {row.claim_id} has missing property type")
    for row in bundle.claim_evidence:
        if row.claim_id not in ids["claim"] or row.evidence_id not in ids["evidence"]:
            reference_issues.append(f"invalid claim evidence: {row}")
    for row in bundle.claim_resolutions:
        claim_record_id = claim_record.get(row.claim_id)
        observation_record_id = observation_record.get(row.observation_id)
        if claim_record_id is None or observation_record_id is None:
            reference_issues.append(f"invalid claim resolution: {row}")
        elif record_release[claim_record_id] != record_release[observation_record_id]:
            reference_issues.append(f"claim resolution crosses releases: {row}")
        else:
            claim = claims_by_id[row.claim_id]
            observation = observations_by_id[row.observation_id]
            if (
                claim.property_type_id is not None
                and claim.property_type_id != observation.property_type_id
            ):
                reference_issues.append(
                    f"claim resolution crosses property types: {row}"
                )
            if (
                claim.subject_entity_id is not None
                and claim.subject_entity_id != observation.subject_entity_id
                and not row.metadata.get("subject_mapping_explanation")
            ):
                reference_issues.append(
                    f"claim resolution changes subject without explanation: {row}"
                )
            if (
                claim_record_id != observation_record_id
                and not row.metadata.get("cross_record_reason")
            ):
                reference_issues.append(
                    f"claim resolution crosses records without explanation: {row}"
                )
    reference_issues.extend(
        f"subset {issue}" for issue in _parent_cycle_issues(bundle.subsets, "subset_id", "parent_subset_id")
    )
    reference_issues.extend(
        f"entity {issue}" for issue in _parent_cycle_issues(bundle.entities, "entity_id", "parent_entity_id")
    )
    checks.append(_check("referential_integrity", sum(map(len, identities.values())), reference_issues))

    ownership_issues = []
    linked_entities = {(row.record_id, row.entity_id) for row in bundle.record_entities}
    records_with_entities = {row.record_id for row in bundle.record_entities}
    entities_with_records = {row.entity_id for row in bundle.record_entities}
    for record in bundle.records:
        if record.record_id not in records_with_entities:
            ownership_issues.append(f"record {record.record_id} has no linked entity")
    for observation in bundle.observations:
        if (observation.record_id, observation.subject_entity_id) not in linked_entities:
            ownership_issues.append(
                f"observation {observation.observation_id} subject is not linked to its record"
            )
    for claim in bundle.claims:
        if (
            claim.subject_entity_id is not None
            and (claim.record_id, claim.subject_entity_id) not in linked_entities
        ):
            ownership_issues.append(f"claim {claim.claim_id} subject is not linked to its record")
    for descriptor in bundle.descriptors:
        if descriptor.entity_id not in entities_with_records:
            ownership_issues.append(
                f"descriptor {descriptor.descriptor_id} entity is not linked to a record"
            )
    checks.append(
        _check(
            "record_entity_ownership",
            len(bundle.records) + len(bundle.observations) + len(bundle.claims) + len(bundle.descriptors),
            ownership_issues,
        )
    )

    provenance_issues = []
    for release in bundle.releases:
        if not release.provenance:
            provenance_issues.append(f"release {release.release_id} lacks provenance")
        if release.license.lower() in {"unknown", "unspecified"}:
            provenance_issues.append(f"release {release.release_id} lacks license declaration")
    for source in bundle.sources:
        if source.license.lower() in {"unknown", "unspecified"}:
            provenance_issues.append(f"source {source.source_id} lacks license declaration")
    checks.append(_check("provenance_and_license", len(bundle.releases) + len(bundle.sources), provenance_issues))

    property_types = {row.property_type_id: row for row in bundle.property_types}
    canonical_types = {
        row.property_type_id: row for row in builtin_property_types()
    }
    participant_roles: Dict[str, set] = {}
    for participant in bundle.observation_participants:
        participant_roles.setdefault(participant.observation_id, set()).add(participant.role)
    property_issues = []
    for property_type in bundle.property_types:
        canonical = canonical_types.get(property_type.property_type_id)
        if canonical is not None and property_type != canonical:
            property_issues.append(
                f"{property_type.property_type_id}: reserved property definition "
                "does not match the canonical schema"
            )
    for observation in bundle.observations:
        definition = property_types.get(observation.property_type_id)
        if definition is None:
            continue
        property_issues.extend(
            f"{observation.observation_id}: {issue}"
            for issue in property_registry.validate(observation, definition)
        )
        required_roles = set(definition.metadata.get("required_participant_roles", []))
        missing_roles = required_roles - participant_roles.get(observation.observation_id, set())
        if missing_roles:
            property_issues.append(
                f"{observation.observation_id}: missing participant roles {sorted(missing_roles)}"
            )
    checks.append(
        _check(
            "property_semantics",
            len(bundle.property_types) + len(bundle.observations),
            property_issues,
        )
    )

    evidence_issues = []
    observation_evidence = {row.observation_id for row in bundle.observation_evidence}
    claim_evidence = {row.claim_id for row in bundle.claim_evidence}
    linked_evidence = {row.evidence_id for row in bundle.observation_evidence}
    linked_evidence.update(row.evidence_id for row in bundle.claim_evidence)
    for observation in bundle.observations:
        if (
            EvidenceLevel(observation.evidence_level) != EvidenceLevel.E0
            and observation.observation_id not in observation_evidence
        ):
            evidence_issues.append(
                f"observation {observation.observation_id} has an evidence grade without a link"
            )
    for claim in bundle.claims:
        if claim.claim_id not in claim_evidence:
            evidence_issues.append(f"claim {claim.claim_id} has no evidence link")
    for evidence in bundle.evidence:
        if evidence.evidence_id not in linked_evidence:
            evidence_issues.append(f"evidence {evidence.evidence_id} is not linked to a claim or observation")
    resolutions_by_claim = {row.claim_id for row in bundle.claim_resolutions}
    for claim in bundle.claims:
        has_resolution = claim.claim_id in resolutions_by_claim
        if claim.status == "resolved" and not has_resolution:
            evidence_issues.append(
                f"resolved claim {claim.claim_id} has no observation resolution"
            )
        elif claim.status != "resolved" and has_resolution:
            evidence_issues.append(
                f"claim {claim.claim_id} has a resolution but status is {claim.status}"
            )
    checks.append(
        _check(
            "evidence_linkage",
            len(bundle.observations) + len(bundle.claims),
            evidence_issues,
        )
    )

    asset_issues = []
    assets = {row.asset_id: row for row in bundle.assets}
    for asset in bundle.assets:
        if "://" in asset.uri:
            asset_issues.append(
                f"remote asset {asset.asset_id} cannot be content-verified; "
                "mirror it under asset_root before admission"
            )
            continue
        path = (root / asset.uri).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            asset_issues.append(f"asset {asset.asset_id} escapes asset_root")
            continue
        if not path.is_file():
            asset_issues.append(f"asset {asset.asset_id} is missing: {asset.uri}")
        elif asset.content_sha256 is None:
            asset_issues.append(f"local asset {asset.asset_id} lacks SHA-256")
        elif _sha256(path) != asset.content_sha256:
            asset_issues.append(f"asset {asset.asset_id} SHA-256 mismatch")
    for observation in bundle.observations:
        definition = property_types.get(observation.property_type_id)
        if definition and definition.value_kind == "series_asset" and observation.value not in assets:
            asset_issues.append(f"observation {observation.observation_id} has missing series asset")
    checks.append(_check("asset_integrity", len(bundle.assets), asset_issues))

    assessed_levels = {
        release.release_id: highest_satisfied_level(release.quality).value
        for release in bundle.releases
    }
    assessed_levels.update(
        {
            subset.subset_id: highest_satisfied_level(subset.quality).value
            for subset in bundle.subsets
        }
    )
    quality_issues = []
    for item in (*bundle.releases, *bundle.subsets):
        identifier = getattr(item, "subset_id", None) or item.release_id
        declared = QualityLevel(item.quality_level)
        assessed = QualityLevel(assessed_levels[identifier])
        if level_rank(declared) > level_rank(assessed):
            quality_issues.append(
                f"{identifier}: declared {declared.value}, assessed {assessed.value}"
            )
    checks.append(
        _check(
            "quality_level_support",
            len(bundle.releases) + len(bundle.subsets),
            quality_issues,
        )
    )

    if profile == "benchmark":
        benchmark_specific_issues = []
        for subset in bundle.subsets:
            if QualityLevel(subset.quality_level) == QualityLevel.B0:
                continue
            member_count = sum(
                row.subset_id == subset.subset_id
                for rows in (
                    bundle.memberships,
                    bundle.observation_memberships,
                    bundle.claim_memberships,
                )
                for row in rows
            )
            if member_count == 0:
                benchmark_specific_issues.append(f"{subset.subset_id}: benchmark subset is empty")
            if "task_definition" not in subset.quality:
                benchmark_specific_issues.append(f"{subset.subset_id}: task_definition is missing")
        checks.append(
            _check(
                "benchmark_readiness",
                len(bundle.subsets),
                benchmark_specific_issues,
            )
        )

    if profile == "export":
        export_issues = [
            f"asset {asset.asset_id} is not redistributable"
            for asset in bundle.assets
            if asset.access_policy not in {"open", "redistributable"}
        ]
        export_issues.extend(
            f"source {source.source_id} is not redistributable"
            for source in bundle.sources
            if source.access_policy not in {"open", "redistributable"}
        )
        checks.append(_check("export_policy", len(bundle.assets) + len(bundle.sources), export_issues))

    return QualityCertificate.issue(
        profile=profile,
        subject_digest=bundle.digest,
        checks=checks,
        assessed_levels=assessed_levels,
        counts={field: len(getattr(bundle, field)) for field in bundle.__dataclass_fields__ if field != "standard_version"},
    )


def assess_file(
    path: Union[str, Path],
    asset_root: Union[str, Path],
    profile: str = "admission",
) -> QualityCertificate:
    try:
        bundle = load_bundle(path)
    except Exception as error:
        source = Path(path)
        raw_digest = (
            _sha256(source)
            if source.is_file()
            else hashlib.sha256(f"missing:{source}".encode()).hexdigest()
        )
        return QualityCertificate.issue(
            profile=profile,
            subject_digest=raw_digest,
            checks=(QualityCheck("record_standard", False, 1, 1, (f"{type(error).__name__}: {error}",)),),
            assessed_levels={},
            counts={},
        )
    return assess_bundle(bundle, asset_root, profile)


def verify_admission(
    bundle: DatasetBundle,
    certificate: QualityCertificate,
    asset_root: Union[str, Path],
) -> None:
    certificate.verify_integrity()
    if certificate.certificate_version != CERTIFICATE_VERSION:
        raise QualityGateError("unsupported certificate version")
    if certificate.assessor_version != ASSESSOR_VERSION:
        raise QualityGateError("unsupported assessor version")
    if certificate.attestation_type != CERTIFICATE_ATTESTATION:
        raise QualityGateError("unsupported certificate attestation type")
    if certificate.profile != "admission" or not certificate.passed:
        raise QualityGateError("a passing admission certificate is required")
    if certificate.subject_digest != bundle.digest:
        raise QualityGateError("certificate does not match this dataset bundle")
    fresh = assess_bundle(bundle, asset_root, profile="admission")
    failed_codes = [check.code for check in fresh.checks if not check.passed]
    if failed_codes:
        raise QualityGateError(
            "fresh admission assessment failed: " + ", ".join(failed_codes)
        )
    if (
        certificate.passed != fresh.passed
        or certificate.quality_score != fresh.quality_score
        or certificate.assessed_levels != fresh.assessed_levels
        or certificate.counts != fresh.counts
        or certificate.checks != fresh.checks
    ):
        raise QualityGateError(
            "certificate findings do not match a fresh admission assessment"
        )


def load_certificate(path: Union[str, Path]) -> QualityCertificate:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    return QualityCertificate.from_dict(value)


def write_certificate(certificate: QualityCertificate, path: Union[str, Path]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(certificate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
