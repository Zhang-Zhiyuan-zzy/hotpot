"""Public submission fences for folder and compact SQLite datasets.

The public formats intentionally describe data, rather than exposing the
normalized storage schema.  Both formats are converted to ``DatasetBundle``
before quality assessment and ingestion.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from ..models import (
    STANDARD_VERSION,
    Asset,
    Dataset,
    DatasetBundle,
    DatasetRecord,
    DatasetRelease,
    Entity,
    EvidenceItem,
    ExternalIdentifier,
    Observation,
    ObservationEvidence,
    PropertyType,
    RecordEntity,
    Source,
    Structure,
    StructureAsset,
    load_bundle,
)
from ..properties import builtin_property_types


SUBMISSION_VERSION = "hotpot-ds-submission/current"
SQLITE_REQUIRED_TABLES = frozenset({"dataset", "data", "structures", "labels"})


class SubmissionFenceError(ValueError):
    """The submitted source does not satisfy the public input contract."""


@dataclass(frozen=True)
class Submission:
    """A normalized, assessment-ready submission."""

    bundle: DatasetBundle
    asset_root: Path
    source: Optional[Path]
    manifest: Optional[Mapping[str, Any]] = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable(value: Any, prefix: str = "id") -> str:
    text = re.sub(r"[^A-Za-z0-9_.:/-]+", "_", str(value)).strip("_")
    if not text or not text[0].isalnum():
        text = f"{prefix}.{text}"
    return text


def _json_object(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SubmissionFenceError(f"{context} must be a JSON object")
    return dict(value)


def _json_array(value: Any, context: str) -> list:
    if not isinstance(value, list):
        raise SubmissionFenceError(f"{context} must be a JSON array")
    return value


def _loads(value: Any, context: str, default: Any) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise SubmissionFenceError(f"{context} is not valid JSON: {error}") from error


def _safe_relative_path(value: str, context: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SubmissionFenceError(f"{context} must be a safe relative POSIX path")
    return path.as_posix()


def _media_type(fmt: str) -> str:
    return {
        "smi": "chemical/x-daylight-smiles",
        "smiles": "chemical/x-daylight-smiles",
        "mol": "chemical/x-mdl-molfile",
        "sdf": "chemical/x-mdl-sdfile",
        "mol2": "chemical/x-mol2",
        "xyz": "chemical/x-xyz",
        "cif": "chemical/x-cif",
        "pdb": "chemical/x-pdb",
    }.get(fmt.lower(), "application/octet-stream")


def _value_kind(values: Sequence[Any]) -> str:
    if values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "integer"
    if values and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return "real"
    if values and all(isinstance(value, str) for value in values):
        return "text"
    return "json"


def _dataset_fields(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    dataset = _json_object(manifest.get("dataset"), "dataset")
    required = {
        "name",
        "version",
        "title",
        "description",
        "license",
        "access_policy",
        "provenance",
    }
    missing = sorted(required - set(dataset))
    if missing:
        raise SubmissionFenceError(f"dataset is missing required fields: {missing}")
    if not isinstance(dataset["provenance"], dict) or not dataset["provenance"]:
        raise SubmissionFenceError("dataset.provenance must be a non-empty JSON object")
    return dataset


def _normalise_label_entries(labels: Mapping[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for name, raw in labels.items():
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if isinstance(value, dict) and "value" in value:
                entry = dict(value)
            else:
                entry = {"value": value}
            yield str(name), entry


def _property_definitions(
    manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Tuple[PropertyType, ...]:
    values: Dict[str, list] = {}
    for record in records:
        labels = _json_object(record.get("labels", {}), "record.labels")
        for name, entry in _normalise_label_entries(labels):
            values.setdefault(name, []).append(entry["value"])

    builtins = {item.property_type_id: item for item in builtin_property_types()}
    declared = manifest.get("property_types", {})
    if not isinstance(declared, dict):
        raise SubmissionFenceError("property_types must be an object keyed by label name")

    definitions = []
    for name in values:
        if name in builtins:
            definitions.append(builtins[name])
            continue
        definition = declared.get(name, {})
        if not isinstance(definition, dict):
            raise SubmissionFenceError(f"property_types.{name} must be an object")
        definitions.append(
            PropertyType(
                property_type_id=_portable(name, "property"),
                name=str(definition.get("name", name.replace("_", " ").title())),
                value_kind=str(definition.get("value_kind", _value_kind(values[name]))),
                canonical_unit=definition.get("canonical_unit"),
                schema_version=str(definition.get("schema_version", "current")),
                metadata=_json_object(definition.get("metadata", {}), f"property_types.{name}.metadata"),
            )
        )
    return tuple(definitions)


def _structure_groups(raw: Any, root: Path, context: str) -> list:
    """Normalize the concise and multi-structure public representations."""

    if isinstance(raw, dict):
        raw_groups = [
            {
                "role": "structure",
                "primary": True,
                "representations": raw,
            }
        ] if raw else []
    elif isinstance(raw, list):
        raw_groups = raw
    else:
        raise SubmissionFenceError(f"{context} must be an object or an array")

    groups = []
    for group_index, raw_group in enumerate(raw_groups):
        group = _json_object(raw_group, f"{context}[{group_index}]")
        if "role" not in group or "representations" not in group:
            raise SubmissionFenceError(
                f"{context}[{group_index}] requires role and representations"
            )
        representations = _json_object(
            group["representations"], f"{context}[{group_index}].representations"
        )
        if not representations:
            raise SubmissionFenceError(
                f"{context}[{group_index}].representations cannot be empty"
            )
        entries = []
        for fmt, raw_representation in sorted(representations.items()):
            details = (
                raw_representation
                if isinstance(raw_representation, dict)
                else {"path": raw_representation}
            )
            details = _json_object(
                details, f"{context}[{group_index}].representations.{fmt}"
            )
            if "path" not in details:
                raise SubmissionFenceError(
                    f"{context}[{group_index}].representations.{fmt}.path is required"
                )
            uri = _safe_relative_path(
                str(details["path"]),
                f"{context}[{group_index}].representations.{fmt}.path",
            )
            path = (root / uri).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise SubmissionFenceError(
                    f"structure path escapes the asset root: {uri}"
                ) from error
            if not path.is_file():
                raise SubmissionFenceError(f"structure file does not exist: {uri}")
            entries.append((str(fmt).lower(), uri, details, _sha256(path)))
        groups.append(
            {
                "role": _portable(group["role"], "structure_role"),
                "primary": bool(group.get("primary", False)),
                "representations": entries,
                "atom_count": group.get("atom_count"),
                "formal_charge": group.get("formal_charge"),
                "entity_kind": group.get("entity_kind"),
                "metadata": _json_object(
                    group.get("metadata", {}), f"{context}[{group_index}].metadata"
                ),
            }
        )
    return groups


def _manifest_to_bundle(manifest: Mapping[str, Any], root: Path) -> DatasetBundle:
    if not isinstance(manifest, Mapping):
        raise SubmissionFenceError("submission manifest root must be an object")
    if manifest.get("schema_version") != SUBMISSION_VERSION:
        raise SubmissionFenceError(
            f"schema_version must be {SUBMISSION_VERSION!r}"
        )
    dataset = _dataset_fields(manifest)
    records = _json_array(manifest.get("records"), "records")
    if not all(isinstance(record, dict) for record in records):
        raise SubmissionFenceError("every records item must be an object")

    dataset_id = _portable(dataset["name"], "dataset")
    version = _portable(dataset["version"], "version")
    release_id = f"{dataset_id}.v{version}"
    quality_level = str(dataset.get("quality_level", "B0_collection"))
    release_quality = _json_object(dataset.get("quality", {}), "dataset.quality")
    for field in ("core_labels", "core_info"):
        if field in dataset:
            release_quality[field] = list(dataset[field])

    property_types = _property_definitions(manifest, records)
    property_ids = {definition.name: definition.property_type_id for definition in property_types}
    property_ids.update(
        {definition.property_type_id: definition.property_type_id for definition in property_types}
    )

    sources = []
    source_rows = manifest.get("sources", {})
    if not isinstance(source_rows, dict):
        raise SubmissionFenceError("sources must be an object keyed by source id")
    for raw_id, raw in source_rows.items():
        source = _json_object(raw, f"sources.{raw_id}")
        sources.append(
            Source(
                source_id=_portable(raw_id, "source"),
                source_type=str(source.get("source_type", "publication")),
                citation=str(source.get("citation", "")),
                doi=source.get("doi"),
                license=str(source.get("license", dataset["license"])),
                access_policy=str(source.get("access_policy", dataset["access_policy"])),
                metadata=_json_object(source.get("metadata", {}), f"source {raw_id} metadata"),
            )
        )
    source_ids = {source.source_id for source in sources}

    bundle_records = []
    entities = []
    external_identifiers = []
    record_entities = []
    assets = []
    structures = []
    structure_assets = []
    observations = []
    evidence = []
    observation_evidence = []
    seen_assets: Dict[str, Asset] = {}
    seen_entities: Dict[str, Entity] = {}
    seen_structures: Dict[str, Structure] = {}

    for record_index, record in enumerate(records):
        missing = sorted({"id", "structures", "labels"} - set(record))
        if missing:
            raise SubmissionFenceError(
                f"records[{record_index}] is missing required fields: {missing}"
            )
        local_id = str(record["id"])
        if not local_id:
            raise SubmissionFenceError(f"records[{record_index}].id cannot be empty")
        local_token = _portable(local_id, "record")
        record_id = f"{release_id}.record.{local_token}"
        metadata = _json_object(record.get("metadata", {}), f"records[{record_index}].metadata")
        bundle_records.append(DatasetRecord(record_id, release_id, local_id, metadata))

        structure_groups = _structure_groups(
            record["structures"], root, f"records[{record_index}].structures"
        )
        subject_entity_id = None
        for group_index, group in enumerate(structure_groups):
            representation_entries = group["representations"]
            first_format, _, first_details, primary_digest = representation_entries[0]
            entity_id = f"entity.structure.{primary_digest}"
            entity_kind = str(
                group["entity_kind"]
                or first_details.get(
                    "entity_kind",
                    "crystal_structure"
                    if any(fmt == "cif" for fmt, *_ in representation_entries)
                    else "molecular_structure",
                )
            )
            entity = Entity(entity_id, entity_kind, None, {})
            previous_entity = seen_entities.get(entity_id)
            if previous_entity is not None and previous_entity != entity:
                raise SubmissionFenceError(f"conflicting shared entity: {entity_id}")
            seen_entities.setdefault(entity_id, entity)
            relation_role = (
                f"primary.{group['role']}" if group["primary"] else group["role"]
            )
            relation = RecordEntity(record_id, entity_id, relation_role)
            if relation not in record_entities:
                record_entities.append(relation)
            if subject_entity_id is None or group["primary"]:
                subject_entity_id = entity_id

            structure_id = f"structure.{primary_digest}"
            structure = Structure(
                structure_id,
                entity_id,
                None,
                group["atom_count"]
                if group["atom_count"] is not None
                else first_details.get("atom_count"),
                group["formal_charge"]
                if group["formal_charge"] is not None
                else first_details.get("formal_charge"),
                group["metadata"],
            )
            previous_structure = seen_structures.get(structure_id)
            if previous_structure is not None and previous_structure != structure:
                raise SubmissionFenceError(f"conflicting shared structure: {structure_id}")
            seen_structures.setdefault(structure_id, structure)
            for fmt, uri, details, digest in representation_entries:
                uri_digest = hashlib.sha256(uri.encode()).hexdigest()[:12]
                asset_id = f"asset.{fmt}.{digest}.{uri_digest}"
                asset = Asset(
                    asset_id,
                    uri,
                    str(details.get("media_type", _media_type(fmt))),
                    digest,
                    str(details.get("access_policy", dataset["access_policy"])),
                    _json_object(details.get("asset_metadata", {}), "structure asset metadata"),
                )
                previous_asset = seen_assets.get(asset_id)
                if previous_asset is not None and previous_asset != asset:
                    raise SubmissionFenceError(
                        f"one shared asset has conflicting paths or metadata: {asset_id}"
                    )
                seen_assets.setdefault(asset_id, asset)
                link = StructureAsset(structure_id, asset_id, fmt)
                if link not in structure_assets:
                    structure_assets.append(link)

        if subject_entity_id is None:
            subject_entity_id = f"entity.{dataset_id}.{local_token}"
            entity = Entity(
                subject_entity_id, str(record.get("entity_kind", "data_subject")), None, {}
            )
            previous_entity = seen_entities.get(subject_entity_id)
            if previous_entity is not None and previous_entity != entity:
                raise SubmissionFenceError(f"conflicting shared entity: {subject_entity_id}")
            seen_entities.setdefault(subject_entity_id, entity)
            record_entities.append(RecordEntity(record_id, subject_entity_id, "subject"))

        identifiers = _json_object(
            record.get("identifiers", {}), f"records[{record_index}].identifiers"
        )
        for namespace, raw_values in identifiers.items():
            identifier_values = raw_values if isinstance(raw_values, list) else [raw_values]
            for value in identifier_values:
                external_identifiers.append(
                    ExternalIdentifier(
                        subject_entity_id, _portable(namespace, "identifier"), str(value), None
                    )
                )

        labels = _json_object(record["labels"], f"records[{record_index}].labels")
        label_counts: Dict[str, int] = {}
        for label_name, entry in _normalise_label_entries(labels):
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
            property_id = property_ids.get(label_name, _portable(label_name, "property"))
            observation_id = (
                f"{record_id}.label.{_portable(label_name, 'label')}.{label_counts[label_name]}"
            )
            evidence_level = str(entry.get("evidence_level", "E0_insufficient"))
            observations.append(
                Observation(
                    observation_id,
                    record_id,
                    subject_entity_id,
                    property_id,
                    entry["value"],
                    entry.get("unit"),
                    _json_object(entry.get("uncertainty", {}), "label uncertainty"),
                    _json_object(entry.get("conditions", {}), "label conditions"),
                    str(entry.get("status", "active")),
                    evidence_level,
                    _json_object(entry.get("method", {}), "label method"),
                    _json_object(entry.get("metadata", {}), "label metadata"),
                )
            )
            raw_evidence = entry.get("evidence", [])
            if isinstance(raw_evidence, dict):
                raw_evidence = [raw_evidence]
            if not isinstance(raw_evidence, list):
                raise SubmissionFenceError("label evidence must be an object or array")
            for evidence_index, raw in enumerate(raw_evidence, 1):
                item = _json_object(raw, "label evidence item")
                source_id = _portable(item.get("source"), "source")
                if source_id not in source_ids:
                    raise SubmissionFenceError(
                        f"label evidence refers to unknown source: {item.get('source')!r}"
                    )
                evidence_id = f"{observation_id}.evidence.{evidence_index}"
                evidence.append(
                    EvidenceItem(
                        evidence_id,
                        source_id,
                        str(item.get("evidence_type", "reported_value")),
                        str(item.get("locator", "")),
                        item.get("quote"),
                        item.get("matched_text"),
                        _json_object(item.get("metadata", {}), "evidence metadata"),
                    )
                )
                observation_evidence.append(
                    ObservationEvidence(
                        observation_id, evidence_id, str(item.get("role", "supports"))
                    )
                )

    assets.extend(seen_assets.values())
    entities.extend(seen_entities.values())
    structures.extend(seen_structures.values())
    return DatasetBundle(
        standard_version=STANDARD_VERSION,
        datasets=(
            Dataset(dataset_id, str(dataset["title"]), str(dataset["description"])),
        ),
        releases=(
            DatasetRelease(
                release_id,
                dataset_id,
                version,
                quality_level,
                str(dataset["license"]),
                str(dataset["access_policy"]),
                dict(dataset["provenance"]),
                release_quality,
            ),
        ),
        subsets=(),
        records=tuple(bundle_records),
        memberships=(),
        observation_memberships=(),
        claim_memberships=(),
        entities=tuple(entities),
        external_identifiers=tuple(external_identifiers),
        record_entities=tuple(record_entities),
        entity_relations=(),
        assets=tuple(assets),
        structures=tuple(structures),
        structure_assets=tuple(structure_assets),
        descriptors=(),
        property_types=property_types,
        observations=tuple(observations),
        observation_participants=(),
        sources=tuple(sources),
        evidence=tuple(evidence),
        observation_evidence=tuple(observation_evidence),
        claims=(),
        claim_evidence=(),
        claim_resolutions=(),
    )


def _table_columns(connection: sqlite3.Connection, table: str) -> set:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _require_columns(
    connection: sqlite3.Connection, table: str, required: Iterable[str]
) -> None:
    missing = sorted(set(required) - _table_columns(connection, table))
    if missing:
        raise SubmissionFenceError(f"SQLite table {table!r} is missing columns: {missing}")


def _sqlite_to_manifest(path: Path) -> Mapping[str, Any]:
    try:
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        missing_tables = sorted(SQLITE_REQUIRED_TABLES - tables)
        if missing_tables:
            raise SubmissionFenceError(
                f"SQLite submission is missing required tables: {missing_tables}"
            )
        _require_columns(
            connection,
            "dataset",
            {
                "name",
                "version",
                "title",
                "description",
                "license",
                "access_policy",
                "provenance_json",
            },
        )
        _require_columns(connection, "data", {"id", "metadata_json", "identifiers_json"})
        _require_columns(connection, "structures", {"data_id", "format", "path"})
        _require_columns(connection, "labels", {"data_id", "name", "value_json"})

        dataset_rows = list(connection.execute("SELECT * FROM dataset"))
        if len(dataset_rows) != 1:
            raise SubmissionFenceError("SQLite dataset table must contain exactly one row")
        raw_dataset = dict(dataset_rows[0])
        dataset = {
            "name": raw_dataset["name"],
            "version": raw_dataset["version"],
            "title": raw_dataset["title"],
            "description": raw_dataset["description"],
            "license": raw_dataset["license"],
            "access_policy": raw_dataset["access_policy"],
            "provenance": _loads(raw_dataset["provenance_json"], "dataset.provenance_json", {}),
            "quality": _loads(raw_dataset.get("quality_json"), "dataset.quality_json", {}),
        }
        for field in ("quality_level",):
            if raw_dataset.get(field) is not None:
                dataset[field] = raw_dataset[field]
        for field in ("core_labels", "core_info"):
            column = f"{field}_json"
            if raw_dataset.get(column) is not None:
                dataset[field] = _loads(raw_dataset[column], f"dataset.{column}", [])

        structure_columns = _table_columns(connection, "structures")
        label_columns = _table_columns(connection, "labels")
        records = []
        known_ids = set()
        for raw_record in connection.execute("SELECT * FROM data ORDER BY id"):
            record = dict(raw_record)
            data_id = str(record["id"])
            known_ids.add(data_id)
            grouped_structures: Dict[str, Dict[str, Any]] = {}
            raw_structures = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM structures WHERE data_id=?", (data_id,)
                )
            ]
            raw_structures.sort(
                key=lambda row: (
                    str(row.get("structure_id") or row.get("role") or "structure"),
                    str(row["format"]),
                    str(row["path"]),
                )
            )
            for structure in raw_structures:
                fmt = str(structure["format"])
                group_key = str(
                    structure.get("structure_id")
                    or structure.get("role")
                    or "structure"
                )
                role = str(structure.get("role") or group_key)
                primary = bool(structure.get("primary") or False)
                group = grouped_structures.setdefault(
                    group_key,
                    {
                        "role": role,
                        "primary": primary,
                        "representations": {},
                    },
                )
                if group["role"] != role or group["primary"] != primary:
                    raise SubmissionFenceError(
                        f"SQLite structure group {group_key!r} has conflicting role or primary values"
                    )
                if fmt in group["representations"]:
                    raise SubmissionFenceError(
                        f"SQLite structure group {group_key!r} has more than one {fmt!r} representation"
                    )
                details = {"path": structure["path"]}
                for field in (
                    "media_type",
                    "access_policy",
                    "entity_kind",
                ):
                    if field in structure_columns and structure[field] is not None:
                        details[field] = structure[field]
                if (
                    "asset_metadata_json" in structure_columns
                    and structure["asset_metadata_json"] is not None
                ):
                    details["asset_metadata"] = _loads(
                        structure["asset_metadata_json"],
                        "structures.asset_metadata_json",
                        {},
                    )
                group["representations"][fmt] = details
                for field in ("atom_count", "formal_charge", "entity_kind"):
                    if field in structure_columns and structure[field] is not None:
                        previous = group.get(field)
                        if previous is not None and previous != structure[field]:
                            raise SubmissionFenceError(
                                f"SQLite structure group {group_key!r} has conflicting {field}"
                            )
                        group[field] = structure[field]
                if (
                    "metadata_json" in structure_columns
                    and structure["metadata_json"] is not None
                ):
                    metadata = _loads(
                        structure["metadata_json"], "structures.metadata_json", {}
                    )
                    previous = group.get("metadata")
                    if previous is not None and previous != metadata:
                        raise SubmissionFenceError(
                            f"SQLite structure group {group_key!r} has conflicting metadata"
                        )
                    group["metadata"] = metadata
            structures = list(grouped_structures.values())

            labels: Dict[str, Any] = {}
            for raw_label in connection.execute(
                "SELECT * FROM labels WHERE data_id=? ORDER BY name,rowid", (data_id,)
            ):
                label = dict(raw_label)
                entry: Dict[str, Any] = {
                    "value": _loads(label["value_json"], "labels.value_json", None)
                }
                for field in ("unit", "evidence_level", "status"):
                    if field in label_columns and label[field] is not None:
                        entry[field] = label[field]
                for column, field, default in (
                    ("uncertainty_json", "uncertainty", {}),
                    ("conditions_json", "conditions", {}),
                    ("method_json", "method", {}),
                    ("metadata_json", "metadata", {}),
                    ("evidence_json", "evidence", []),
                ):
                    if column in label_columns and label[column] is not None:
                        entry[field] = _loads(label[column], f"labels.{column}", default)
                name = str(label["name"])
                if name in labels:
                    if not isinstance(labels[name], list):
                        labels[name] = [labels[name]]
                    labels[name].append(entry)
                else:
                    labels[name] = entry
            records.append(
                {
                    "id": data_id,
                    "structures": structures,
                    "labels": labels,
                    "metadata": _loads(record["metadata_json"], "data.metadata_json", {}),
                    "identifiers": _loads(
                        record["identifiers_json"], "data.identifiers_json", {}
                    ),
                }
            )

        dangling_structures = connection.execute(
            "SELECT DISTINCT data_id FROM structures WHERE data_id NOT IN (SELECT id FROM data)"
        ).fetchall()
        dangling_labels = connection.execute(
            "SELECT DISTINCT data_id FROM labels WHERE data_id NOT IN (SELECT id FROM data)"
        ).fetchall()
        if dangling_structures or dangling_labels:
            raise SubmissionFenceError("SQLite structures or labels contain unknown data_id values")

        sources = {}
        if "sources" in tables:
            _require_columns(
                connection,
                "sources",
                {"id", "source_type", "citation", "license", "access_policy"},
            )
            for row in connection.execute("SELECT * FROM sources ORDER BY id"):
                raw = dict(row)
                sources[str(raw.pop("id"))] = {
                    "source_type": raw["source_type"],
                    "citation": raw["citation"],
                    "doi": raw.get("doi"),
                    "license": raw["license"],
                    "access_policy": raw["access_policy"],
                    "metadata": _loads(raw.get("metadata_json"), "sources.metadata_json", {}),
                }
        property_types = {}
        if "property_types" in tables:
            _require_columns(
                connection,
                "property_types",
                {"id", "name", "value_kind", "schema_version"},
            )
            for row in connection.execute("SELECT * FROM property_types ORDER BY id"):
                raw = dict(row)
                property_types[str(raw["id"])] = {
                    "name": raw["name"],
                    "value_kind": raw["value_kind"],
                    "canonical_unit": raw.get("canonical_unit"),
                    "schema_version": raw["schema_version"],
                    "metadata": _loads(
                        raw.get("metadata_json"), "property_types.metadata_json", {}
                    ),
                }
        return {
            "schema_version": SUBMISSION_VERSION,
            "dataset": dataset,
            "records": records,
            "sources": sources,
            "property_types": property_types,
        }
    except sqlite3.DatabaseError as error:
        raise SubmissionFenceError(f"cannot read SQLite submission: {error}") from error
    finally:
        if "connection" in locals():
            connection.close()


def load_submission(
    source: Union[DatasetBundle, Mapping[str, Any], str, Path],
    *,
    asset_root: Optional[Union[str, Path]] = None,
) -> Submission:
    """Load a backend bundle, public folder, manifest, or compact SQLite input."""

    if isinstance(source, DatasetBundle):
        root = Path(asset_root or ".").resolve()
        return Submission(source, root, None, None)
    if isinstance(source, Mapping):
        root = Path(asset_root or ".").resolve()
        manifest = json.loads(json.dumps(source))
        return Submission(_manifest_to_bundle(manifest, root), root, None, manifest)

    path = Path(source).resolve()
    if path.is_dir():
        manifest_path = path / "dataset.json"
        if not manifest_path.is_file():
            raise SubmissionFenceError(
                f"folder submissions require {manifest_path.name!r} at the folder root"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SubmissionFenceError(f"cannot read dataset.json: {error}") from error
        if not isinstance(manifest, dict):
            raise SubmissionFenceError("dataset.json root must be an object")
        root = Path(asset_root).resolve() if asset_root is not None else path
        return Submission(_manifest_to_bundle(manifest, root), root, path, manifest)
    if not path.is_file():
        raise SubmissionFenceError(f"submission does not exist: {path}")

    root = Path(asset_root).resolve() if asset_root is not None else path.parent
    if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
        manifest = _sqlite_to_manifest(path)
        return Submission(_manifest_to_bundle(manifest, root), root, path, manifest)
    if path.suffix.lower() == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SubmissionFenceError(f"cannot read JSON submission: {error}") from error
        if isinstance(raw, dict) and raw.get("standard_version") == STANDARD_VERSION:
            return Submission(load_bundle(path), root, path, None)
        if not isinstance(raw, dict):
            raise SubmissionFenceError("submission manifest root must be an object")
        return Submission(_manifest_to_bundle(raw, root), root, path, raw)
    raise SubmissionFenceError(
        "supported inputs are DatasetBundle, bundle JSON, a dataset.json folder, "
        "or a compact .sqlite/.sqlite3 database"
    )


__all__ = [
    "SQLITE_REQUIRED_TABLES",
    "SUBMISSION_VERSION",
    "Submission",
    "SubmissionFenceError",
    "load_submission",
]
