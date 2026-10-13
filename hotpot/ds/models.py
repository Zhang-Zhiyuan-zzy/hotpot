"""Current property-agnostic records for the Hotpot dataset store."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

from .levels import EvidenceLevel, QualityLevel


STANDARD_VERSION = "hotpot-dataset/current"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
VALUE_KINDS = frozenset({"integer", "real", "category", "text", "json", "series_asset"})
OBSERVATION_STATUSES = frozenset({"active", "superseded", "retracted"})
CLAIM_STATUSES = frozenset({"unresolved", "resolved", "rejected"})
CLAIM_RESOLUTION_TYPES = frozenset(
    {
        "accepted_as_observation",
        "partially_resolved_by",
        "contradicted_by_observation",
    }
)


class DatasetStandardError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _keys(value: Mapping[str, Any], expected: Iterable[str], context: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise DatasetStandardError(
            f"{context} fields differ from standard; "
            f"missing={sorted(expected_set - actual)}, extra={sorted(actual - expected_set)}"
        )


def _id(value: str, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise DatasetStandardError(f"{field} is not a portable identifier")
    return value


def _optional_id(value: Optional[str], field: str) -> Optional[str]:
    return None if value is None else _id(value, field)


def _digest(value: Optional[str], field: str) -> Optional[str]:
    if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
        raise DatasetStandardError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _json_object(value: Dict[str, Any], field: str) -> None:
    if not isinstance(value, dict):
        raise DatasetStandardError(f"{field} must be a JSON object")
    canonical_json(value)


class _Record:
    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        for key, item in tuple(value.items()):
            if isinstance(item, Enum):
                value[key] = item.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        _keys(value, cls.__dataclass_fields__, cls.__name__)
        return cls(**dict(value))


@dataclass(frozen=True)
class Dataset(_Record):
    dataset_id: str
    title: str
    description: str

    def __post_init__(self) -> None:
        _id(self.dataset_id, "dataset_id")
        if not self.title:
            raise DatasetStandardError("dataset title is required")


@dataclass(frozen=True)
class DatasetRelease(_Record):
    release_id: str
    dataset_id: str
    version: str
    quality_level: str
    license: str
    access_policy: str
    provenance: Dict[str, Any]
    quality: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.release_id, "release_id")
        _id(self.dataset_id, "dataset_id")
        _id(self.version, "version")
        QualityLevel(self.quality_level)
        if not self.license or not self.access_policy:
            raise DatasetStandardError("release license and access_policy are required")
        _json_object(self.provenance, "provenance")
        _json_object(self.quality, "quality")


@dataclass(frozen=True)
class DatasetSubset(_Record):
    subset_id: str
    release_id: str
    parent_subset_id: Optional[str]
    name: str
    quality_level: str
    selection: Dict[str, Any]
    quality: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.subset_id, "subset_id")
        _id(self.release_id, "release_id")
        _optional_id(self.parent_subset_id, "parent_subset_id")
        QualityLevel(self.quality_level)
        _json_object(self.selection, "selection")
        _json_object(self.quality, "quality")


@dataclass(frozen=True)
class DatasetRecord(_Record):
    record_id: str
    release_id: str
    local_id: str
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.record_id, "record_id")
        _id(self.release_id, "release_id")
        if not self.local_id:
            raise DatasetStandardError("local_id is required")
        _json_object(self.metadata, "record metadata")


@dataclass(frozen=True)
class SubsetMembership(_Record):
    subset_id: str
    record_id: str

    def __post_init__(self) -> None:
        _id(self.subset_id, "subset_id")
        _id(self.record_id, "record_id")


@dataclass(frozen=True)
class ObservationMembership(_Record):
    subset_id: str
    observation_id: str

    def __post_init__(self) -> None:
        _id(self.subset_id, "subset_id")
        _id(self.observation_id, "observation_id")


@dataclass(frozen=True)
class ClaimMembership(_Record):
    subset_id: str
    claim_id: str

    def __post_init__(self) -> None:
        _id(self.subset_id, "subset_id")
        _id(self.claim_id, "claim_id")


@dataclass(frozen=True)
class Entity(_Record):
    entity_id: str
    entity_kind: str
    parent_entity_id: Optional[str]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.entity_id, "entity_id")
        _id(self.entity_kind, "entity_kind")
        _optional_id(self.parent_entity_id, "parent_entity_id")
        _json_object(self.metadata, "entity metadata")


@dataclass(frozen=True)
class ExternalIdentifier(_Record):
    entity_id: str
    namespace: str
    value: str
    version: Optional[str]

    def __post_init__(self) -> None:
        _id(self.entity_id, "entity_id")
        _id(self.namespace, "namespace")
        if not self.value:
            raise DatasetStandardError("external identifier value is required")


@dataclass(frozen=True)
class RecordEntity(_Record):
    record_id: str
    entity_id: str
    role: str

    def __post_init__(self) -> None:
        _id(self.record_id, "record_id")
        _id(self.entity_id, "entity_id")
        _id(self.role, "record entity role")


@dataclass(frozen=True)
class EntityRelation(_Record):
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.subject_entity_id, "subject_entity_id")
        _id(self.predicate, "predicate")
        _id(self.object_entity_id, "object_entity_id")
        _json_object(self.metadata, "relation metadata")


@dataclass(frozen=True)
class Asset(_Record):
    asset_id: str
    uri: str
    media_type: str
    content_sha256: Optional[str]
    access_policy: str
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.asset_id, "asset_id")
        if not self.uri or not self.media_type or not self.access_policy:
            raise DatasetStandardError("asset uri, media_type and access_policy are required")
        _digest(self.content_sha256, "content_sha256")
        if not ("://" in self.uri):
            path = PurePosixPath(self.uri)
            if path.is_absolute() or ".." in path.parts:
                raise DatasetStandardError("local asset URI must be a safe relative POSIX path")
        _json_object(self.metadata, "asset metadata")


@dataclass(frozen=True)
class Structure(_Record):
    structure_id: str
    entity_id: str
    coordinate_hash: Optional[str]
    atom_count: Optional[int]
    formal_charge: Optional[int]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.structure_id, "structure_id")
        _id(self.entity_id, "entity_id")
        _digest(self.coordinate_hash, "coordinate_hash")
        if self.atom_count is not None and self.atom_count <= 0:
            raise DatasetStandardError("atom_count must be positive")
        _json_object(self.metadata, "structure metadata")


@dataclass(frozen=True)
class StructureAsset(_Record):
    structure_id: str
    asset_id: str
    role: str

    def __post_init__(self) -> None:
        _id(self.structure_id, "structure_id")
        _id(self.asset_id, "asset_id")
        _id(self.role, "structure asset role")


@dataclass(frozen=True)
class ComputedDescriptor(_Record):
    descriptor_id: str
    entity_id: str
    descriptor_type: str
    value: Any
    unit: Optional[str]
    method: Dict[str, Any]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.descriptor_id, "descriptor_id")
        _id(self.entity_id, "descriptor entity_id")
        _id(self.descriptor_type, "descriptor_type")
        canonical_json(self.value)
        _json_object(self.method, "descriptor method")
        _json_object(self.metadata, "descriptor metadata")


@dataclass(frozen=True)
class PropertyType(_Record):
    property_type_id: str
    name: str
    value_kind: str
    canonical_unit: Optional[str]
    schema_version: str
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.property_type_id, "property_type_id")
        if self.value_kind not in VALUE_KINDS:
            raise DatasetStandardError(f"unsupported value_kind: {self.value_kind}")
        _id(self.schema_version, "property schema_version")
        _json_object(self.metadata, "property metadata")


@dataclass(frozen=True)
class Observation(_Record):
    observation_id: str
    record_id: str
    subject_entity_id: str
    property_type_id: str
    value: Any
    unit: Optional[str]
    uncertainty: Dict[str, Any]
    conditions: Dict[str, Any]
    status: str
    evidence_level: str
    method: Dict[str, Any]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.observation_id, "observation_id")
        _id(self.record_id, "record_id")
        _id(self.subject_entity_id, "subject_entity_id")
        _id(self.property_type_id, "property_type_id")
        if self.status not in OBSERVATION_STATUSES:
            raise DatasetStandardError(f"unsupported observation status: {self.status}")
        EvidenceLevel(self.evidence_level)
        canonical_json(self.value)
        _json_object(self.uncertainty, "uncertainty")
        _json_object(self.conditions, "conditions")
        _json_object(self.method, "method")
        _json_object(self.metadata, "observation metadata")


@dataclass(frozen=True)
class Source(_Record):
    source_id: str
    source_type: str
    citation: str
    doi: Optional[str]
    license: str
    access_policy: str
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.source_id, "source_id")
        _id(self.source_type, "source_type")
        if not self.citation or not self.license or not self.access_policy:
            raise DatasetStandardError("source citation, license and access_policy are required")
        _json_object(self.metadata, "source metadata")


@dataclass(frozen=True)
class EvidenceItem(_Record):
    evidence_id: str
    source_id: str
    evidence_type: str
    locator: str
    quote: Optional[str]
    matched_text: Optional[str]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.evidence_id, "evidence_id")
        _id(self.source_id, "source_id")
        _id(self.evidence_type, "evidence_type")
        if not self.locator:
            raise DatasetStandardError("evidence locator is required")
        _json_object(self.metadata, "evidence metadata")


@dataclass(frozen=True)
class ObservationEvidence(_Record):
    observation_id: str
    evidence_id: str
    role: str

    def __post_init__(self) -> None:
        _id(self.observation_id, "observation_id")
        _id(self.evidence_id, "evidence_id")
        _id(self.role, "evidence role")


@dataclass(frozen=True)
class ObservationParticipant(_Record):
    """An entity participating in a measured or calculated property.

    This relation keeps solvent/phase/component roles queryable for stability
    constants, distribution ratios, separation factors, and similar data.
    """

    observation_id: str
    entity_id: str
    role: str
    stoichiometry: Optional[float]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.observation_id, "observation_id")
        _id(self.entity_id, "participant entity_id")
        _id(self.role, "participant role")
        if self.stoichiometry is not None and self.stoichiometry <= 0:
            raise DatasetStandardError("participant stoichiometry must be positive")
        _json_object(self.metadata, "participant metadata")


@dataclass(frozen=True)
class Claim(_Record):
    """A traceable source claim that is not safe to coerce into a value.

    Ambiguous CSD prose belongs here.  A claim may later be adjudicated into
    an :class:`Observation`, while the original wording remains preserved.
    """

    claim_id: str
    record_id: str
    subject_entity_id: Optional[str]
    property_type_id: Optional[str]
    text: str
    status: str
    evidence_level: str
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.claim_id, "claim_id")
        _id(self.record_id, "claim record_id")
        _optional_id(self.subject_entity_id, "claim subject_entity_id")
        _optional_id(self.property_type_id, "claim property_type_id")
        if not self.text:
            raise DatasetStandardError("claim text is required")
        if self.status not in CLAIM_STATUSES:
            raise DatasetStandardError(f"unsupported claim status: {self.status}")
        EvidenceLevel(self.evidence_level)
        _json_object(self.metadata, "claim metadata")


@dataclass(frozen=True)
class ClaimEvidence(_Record):
    claim_id: str
    evidence_id: str
    role: str

    def __post_init__(self) -> None:
        _id(self.claim_id, "claim_id")
        _id(self.evidence_id, "evidence_id")
        _id(self.role, "claim evidence role")


@dataclass(frozen=True)
class ClaimResolution(_Record):
    claim_id: str
    observation_id: str
    resolution_type: str
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _id(self.claim_id, "claim_id")
        _id(self.observation_id, "observation_id")
        if self.resolution_type not in CLAIM_RESOLUTION_TYPES:
            raise DatasetStandardError(
                f"unsupported claim resolution type: {self.resolution_type}"
            )
        _json_object(self.metadata, "claim resolution metadata")


@dataclass(frozen=True)
class DatasetBundle:
    standard_version: str
    datasets: Tuple[Dataset, ...]
    releases: Tuple[DatasetRelease, ...]
    subsets: Tuple[DatasetSubset, ...]
    records: Tuple[DatasetRecord, ...]
    memberships: Tuple[SubsetMembership, ...]
    observation_memberships: Tuple[ObservationMembership, ...]
    claim_memberships: Tuple[ClaimMembership, ...]
    entities: Tuple[Entity, ...]
    external_identifiers: Tuple[ExternalIdentifier, ...]
    record_entities: Tuple[RecordEntity, ...]
    entity_relations: Tuple[EntityRelation, ...]
    assets: Tuple[Asset, ...]
    structures: Tuple[Structure, ...]
    structure_assets: Tuple[StructureAsset, ...]
    descriptors: Tuple[ComputedDescriptor, ...]
    property_types: Tuple[PropertyType, ...]
    observations: Tuple[Observation, ...]
    observation_participants: Tuple[ObservationParticipant, ...]
    sources: Tuple[Source, ...]
    evidence: Tuple[EvidenceItem, ...]
    observation_evidence: Tuple[ObservationEvidence, ...]
    claims: Tuple[Claim, ...]
    claim_evidence: Tuple[ClaimEvidence, ...]
    claim_resolutions: Tuple[ClaimResolution, ...]

    def __post_init__(self) -> None:
        if self.standard_version != STANDARD_VERSION:
            raise DatasetStandardError(f"unsupported standard version: {self.standard_version}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            field: (
                self.standard_version if field == "standard_version"
                else [row.to_dict() for row in getattr(self, field)]
            )
            for field in self.__dataclass_fields__
        }

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"{")
        for index, field in enumerate(sorted(self.__dataclass_fields__)):
            if index:
                digest.update(b",")
            digest.update(canonical_json(field).encode())
            digest.update(b":")
            if field == "standard_version":
                digest.update(canonical_json(self.standard_version).encode())
                continue
            digest.update(b"[")
            for row_index, row in enumerate(getattr(self, field)):
                if row_index:
                    digest.update(b",")
                digest.update(canonical_json(row.to_dict()).encode())
            digest.update(b"]")
        digest.update(b"}")
        return digest.hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetBundle":
        _keys(value, cls.__dataclass_fields__, "DatasetBundle")
        record_types = {
            "datasets": Dataset,
            "releases": DatasetRelease,
            "subsets": DatasetSubset,
            "records": DatasetRecord,
            "memberships": SubsetMembership,
            "observation_memberships": ObservationMembership,
            "claim_memberships": ClaimMembership,
            "entities": Entity,
            "external_identifiers": ExternalIdentifier,
            "record_entities": RecordEntity,
            "entity_relations": EntityRelation,
            "assets": Asset,
            "structures": Structure,
            "structure_assets": StructureAsset,
            "descriptors": ComputedDescriptor,
            "property_types": PropertyType,
            "observations": Observation,
            "observation_participants": ObservationParticipant,
            "sources": Source,
            "evidence": EvidenceItem,
            "observation_evidence": ObservationEvidence,
            "claims": Claim,
            "claim_evidence": ClaimEvidence,
            "claim_resolutions": ClaimResolution,
        }
        return cls(
            standard_version=value["standard_version"],
            **{
                field: tuple(record_type.from_dict(row) for row in value[field])
                for field, record_type in record_types.items()
            },
        )


def load_bundle(path: Union[str, Path]) -> DatasetBundle:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DatasetStandardError("dataset bundle root must be an object")
    return DatasetBundle.from_dict(value)


def write_bundle(bundle: DatasetBundle, path: Union[str, Path]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write("{")
        for index, field in enumerate(sorted(bundle.__dataclass_fields__)):
            if index:
                handle.write(",")
            handle.write(canonical_json(field))
            handle.write(":")
            if field == "standard_version":
                handle.write(canonical_json(bundle.standard_version))
                continue
            handle.write("[")
            for row_index, row in enumerate(getattr(bundle, field)):
                if row_index:
                    handle.write(",")
                handle.write(canonical_json(row.to_dict()))
            handle.write("]")
        handle.write("}\n")
