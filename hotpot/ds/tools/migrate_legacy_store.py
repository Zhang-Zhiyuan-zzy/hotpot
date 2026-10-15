"""One-time, provenance-preserving migration of the normalized legacy store.

This module is deliberately a maintenance tool, not a compatibility backend.
It opens the legacy SQLite database read-only, exports every release to a
current folder-backed :class:`~hotpot.ds.Bundle`, and only publishes a cutover
pointer after source/export/target parity has passed.  The original store is
never modified or queried by the current runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union

from ..gates import BenchmarkGate, EvidenceGate, QualityGate
from ..gates._persistence import deserialize_gate_configs
from ..storage import TargetStore, asset_inventory_digest


LEGACY_SCHEMA_VERSION = "1"
LEGACY_RECORD_STANDARD = "hotpot-dataset/current"
MIGRATION_SCHEMA = "hotpot-ds-legacy-migration/current"
CUTOVER_SCHEMA = "hotpot-ds-store-cutover/current"
BOOTSTRAP_SCHEMA = "hotpot-ds-migration-bootstrap/current"


class LegacyMigrationError(RuntimeError):
    """The legacy source or migration result violates the cutover contract."""


class LegacySchemaError(LegacyMigrationError):
    """The source is not the one supported normalized legacy schema."""


class MigrationParityError(LegacyMigrationError):
    """Source facts differ without an audited migration explanation."""


@dataclass(frozen=True)
class DatasetBootstrap:
    """Owner-approved policy required to migrate one legacy dataset.

    There is intentionally no inferred/default bootstrap.  The caller must
    make the identity, comparison, Gate, and incomplete-archive decisions
    explicitly before an old dataset becomes a current existing dataset.
    """

    ds_name: str
    core_labels: Tuple[str, ...]
    core_info: Tuple[str, ...]
    identity_schema: Mapping[str, Any]
    property_definitions: Mapping[str, Any]
    property_comparators: Mapping[str, Any]
    required_structure_roles: Tuple[str, ...]
    gates: Tuple[Union[EvidenceGate, QualityGate, BenchmarkGate], ...]
    owner: str
    decision: str
    include_incomplete: bool


def load_bootstraps(path: Union[str, Path]) -> Dict[str, DatasetBootstrap]:
    """Load the explicit owner/admin bootstrap file used by the migration CLI."""

    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if set(value) != {"schema", "datasets"} or value["schema"] != BOOTSTRAP_SCHEMA:
        raise LegacyMigrationError("migration bootstrap uses an unsupported schema")
    result = {}
    expected = {
        "core_labels",
        "core_info",
        "identity_schema",
        "property_definitions",
        "property_comparators",
        "required_structure_roles",
        "gates",
        "owner",
        "decision",
        "include_incomplete",
    }
    for ds_name, config in value["datasets"].items():
        if set(config) != expected:
            raise LegacyMigrationError(
                f"bootstrap fields differ for dataset {ds_name!r}"
            )
        result[ds_name] = DatasetBootstrap(
            ds_name=ds_name,
            core_labels=tuple(config["core_labels"]),
            core_info=tuple(config["core_info"]),
            identity_schema=dict(config["identity_schema"]),
            property_definitions=dict(config["property_definitions"]),
            property_comparators=dict(config["property_comparators"]),
            required_structure_roles=tuple(config["required_structure_roles"]),
            gates=deserialize_gate_configs(config["gates"]),
            owner=config["owner"],
            decision=config["decision"],
            include_incomplete=config["include_incomplete"],
        )
    return result


_REQUIRED_COLUMNS = {
    "schema_metadata": ("key", "value"),
    "datasets": ("dataset_id", "title", "description"),
    "dataset_policies": (
        "dataset_id", "revision", "config_json", "config_sha256", "updated_at",
    ),
    "dataset_releases": (
        "release_id", "dataset_id", "version", "release_serial", "quality_level",
        "core_labels_json", "license", "access_policy", "provenance_json",
        "quality_json", "ingestion_id",
    ),
    "subsets": (
        "subset_id", "release_id", "parent_subset_id", "name", "quality_level",
        "selection_json", "quality_json",
    ),
    "records": ("record_id", "release_id", "public_index", "local_id", "metadata_json"),
    "subset_memberships": ("subset_id", "record_id"),
    "entities": ("entity_id", "entity_kind", "parent_entity_id", "metadata_json"),
    "external_identifiers": ("entity_id", "namespace", "value", "version"),
    "record_entities": ("record_id", "entity_id", "role"),
    "entity_relations": (
        "subject_entity_id", "predicate", "object_entity_id", "metadata_json",
    ),
    "assets": (
        "asset_id", "uri", "media_type", "content_sha256", "access_policy",
        "metadata_json",
    ),
    "structures": (
        "structure_id", "entity_id", "coordinate_hash", "atom_count",
        "formal_charge", "metadata_json",
    ),
    "structure_assets": ("structure_id", "asset_id", "role"),
    "ingestion_assets": ("ingestion_id", "asset_id"),
    "computed_descriptors": (
        "descriptor_id", "entity_id", "descriptor_type", "value_json", "unit",
        "method_json", "metadata_json",
    ),
    "property_types": (
        "property_type_id", "name", "value_kind", "canonical_unit",
        "schema_version", "metadata_json",
    ),
    "observations": (
        "observation_id", "record_id", "subject_entity_id", "property_type_id",
        "value_json", "unit", "uncertainty_json", "conditions_json", "status",
        "evidence_level", "method_json", "metadata_json",
    ),
    "observation_subset_memberships": ("subset_id", "observation_id"),
    "observation_participants": (
        "observation_id", "entity_id", "role", "stoichiometry", "metadata_json",
    ),
    "sources": (
        "source_id", "source_type", "citation", "doi", "license",
        "access_policy", "metadata_json",
    ),
    "evidence_items": (
        "evidence_id", "source_id", "evidence_type", "locator", "quote",
        "matched_text", "metadata_json",
    ),
    "observation_evidence": ("observation_id", "evidence_id", "role"),
    "claims": (
        "claim_id", "record_id", "subject_entity_id", "property_type_id", "text",
        "status", "evidence_level", "metadata_json",
    ),
    "claim_subset_memberships": ("subset_id", "claim_id"),
    "claim_evidence": ("claim_id", "evidence_id", "role"),
    "claim_resolutions": (
        "claim_id", "observation_id", "resolution_type", "metadata_json",
    ),
}

_SCOPED_TABLES = tuple(name for name in _REQUIRED_COLUMNS if name != "schema_metadata")
_IDENTITY_TABLES = (
    "records",
    "entities",
    "external_identifiers",
    "record_entities",
    "entity_relations",
    "structures",
    "structure_assets",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stream_json_array_digest(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for value in values:
        if not first:
            digest.update(b",")
        digest.update(_canonical_json(value).encode("utf-8"))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _sha256_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = (_canonical_json(dict(value)) + "\n").encode("utf-8")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _row(connection: sqlite3.Connection, sql: str, parameters: Sequence[Any]) -> Dict[str, Any]:
    result = connection.execute(sql, parameters).fetchone()
    if result is None:
        raise LegacyMigrationError("required legacy row is missing")
    return dict(result)


def _case_child_id(case_id: str, kind: str, legacy_id: str) -> str:
    token = hashlib.sha256(legacy_id.encode("utf-8")).hexdigest()[:20]
    return f"{case_id}:{kind}:{token}"


def _identifier_namespace(value: str) -> str:
    return "CSD" if value == "CSD_identifier" else value


def _inventory_identity(table: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    row_value = dict(row)
    digest = _digest_json(row_value)
    return {
        "row_id": f"legacy-{table}-{digest[:24]}",
        "row_sha256": digest,
        "table": table,
        "row": row_value,
    }


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> Tuple[str, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    size = 0
    with path.open("wb") as stream:
        for value in values:
            line = (_canonical_json(dict(value)) + "\n").encode("utf-8")
            stream.write(line)
            digest.update(line)
            size += len(line)
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest(), size, count


def _safe_legacy_asset(root: Path, uri: str) -> Optional[Path]:
    if "://" in uri:
        return None
    pure = PurePosixPath(uri)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise LegacyMigrationError(f"unsafe legacy asset URI: {uri!r}")
    path = (root / Path(*pure.parts)).resolve(strict=False)
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise LegacyMigrationError(f"legacy asset URI escapes its root: {uri!r}") from error
    return path


class _RowInventory:
    """Disk-backed unique-row digest accumulator used by parity checks."""

    def __init__(self, directory: Path):
        descriptor, name = tempfile.mkstemp(prefix=".migration-inventory-", suffix=".sqlite3", dir=directory)
        os.close(descriptor)
        self.path = Path(name)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE rows(table_name TEXT NOT NULL,row_digest TEXT NOT NULL,"
            "PRIMARY KEY(table_name,row_digest))"
        )

    def add(self, table: str, row: Mapping[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO rows(table_name,row_digest) VALUES(?,?)",
            (table, _digest_json(dict(row))),
        )

    def summary(self) -> Dict[str, Dict[str, Any]]:
        self.connection.commit()
        result: Dict[str, Dict[str, Any]] = {}
        names = set(_SCOPED_TABLES)
        names.update(
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT table_name FROM rows"
            )
        )
        for table in sorted(names):
            digest = hashlib.sha256()
            count = 0
            for (row_digest,) in self.connection.execute(
                "SELECT row_digest FROM rows WHERE table_name=? ORDER BY row_digest",
                (table,),
            ):
                digest.update(bytes.fromhex(row_digest))
                count += 1
            result[table] = {"count": count, "row_set_sha256": digest.hexdigest()}
        return result

    def close(self) -> None:
        self.connection.close()
        self.path.unlink(missing_ok=True)


_SCOPE_CTE = """
WITH RECURSIVE
scoped_records(record_id) AS ({record_scope}),
scoped_observations(observation_id) AS (
  SELECT o.observation_id FROM observations o JOIN scoped_records r USING(record_id)
),
scoped_claims(claim_id) AS (
  SELECT c.claim_id FROM claims c JOIN scoped_records r USING(record_id)
),
base_entities(entity_id) AS (
  SELECT re.entity_id FROM record_entities re JOIN scoped_records r USING(record_id)
  UNION SELECT o.subject_entity_id FROM observations o
    JOIN scoped_observations so USING(observation_id) WHERE o.subject_entity_id IS NOT NULL
  UNION SELECT op.entity_id FROM observation_participants op
    JOIN scoped_observations so USING(observation_id)
  UNION SELECT c.subject_entity_id FROM claims c
    JOIN scoped_claims sc USING(claim_id) WHERE c.subject_entity_id IS NOT NULL
),
scoped_entities(entity_id) AS (
  SELECT entity_id FROM base_entities
  UNION
  SELECT e.parent_entity_id FROM entities e JOIN scoped_entities se
    ON e.entity_id=se.entity_id WHERE e.parent_entity_id IS NOT NULL
),
scoped_structures(structure_id) AS (
  SELECT s.structure_id FROM structures s JOIN scoped_entities se USING(entity_id)
),
scoped_evidence(evidence_id) AS (
  SELECT oe.evidence_id FROM observation_evidence oe
    JOIN scoped_observations so USING(observation_id)
  UNION SELECT ce.evidence_id FROM claim_evidence ce JOIN scoped_claims sc USING(claim_id)
),
scoped_sources(source_id) AS (
  SELECT ei.source_id FROM evidence_items ei JOIN scoped_evidence se USING(evidence_id)
),
scoped_properties(property_type_id) AS (
  SELECT o.property_type_id FROM observations o JOIN scoped_observations so USING(observation_id)
  UNION SELECT c.property_type_id FROM claims c JOIN scoped_claims sc USING(claim_id)
    WHERE c.property_type_id IS NOT NULL
),
scoped_subsets(subset_id) AS (
  SELECT sm.subset_id FROM subset_memberships sm JOIN scoped_records sr USING(record_id)
  UNION SELECT osm.subset_id FROM observation_subset_memberships osm
    JOIN scoped_observations so USING(observation_id)
  UNION SELECT csm.subset_id FROM claim_subset_memberships csm JOIN scoped_claims sc USING(claim_id)
),
scoped_assets(asset_id) AS (
  SELECT sa.asset_id FROM structure_assets sa JOIN scoped_structures ss USING(structure_id)
)
"""

_TABLE_FROM = {
    "records": "records t JOIN scoped_records x ON x.record_id=t.record_id",
    "subsets": "subsets t JOIN scoped_subsets x ON x.subset_id=t.subset_id",
    "subset_memberships": "subset_memberships t JOIN scoped_records x ON x.record_id=t.record_id",
    "entities": "entities t JOIN scoped_entities x ON x.entity_id=t.entity_id",
    "external_identifiers": "external_identifiers t JOIN scoped_entities x ON x.entity_id=t.entity_id",
    "record_entities": "record_entities t JOIN scoped_records x ON x.record_id=t.record_id",
    "entity_relations": (
        "entity_relations t WHERE t.subject_entity_id IN (SELECT entity_id FROM scoped_entities) "
        "OR t.object_entity_id IN (SELECT entity_id FROM scoped_entities)"
    ),
    "assets": "assets t JOIN scoped_assets x ON x.asset_id=t.asset_id",
    "structures": "structures t JOIN scoped_structures x ON x.structure_id=t.structure_id",
    "structure_assets": "structure_assets t JOIN scoped_structures x ON x.structure_id=t.structure_id",
    "computed_descriptors": "computed_descriptors t JOIN scoped_entities x ON x.entity_id=t.entity_id",
    "property_types": "property_types t JOIN scoped_properties x ON x.property_type_id=t.property_type_id",
    "observations": "observations t JOIN scoped_observations x ON x.observation_id=t.observation_id",
    "observation_subset_memberships": (
        "observation_subset_memberships t JOIN scoped_observations x ON x.observation_id=t.observation_id"
    ),
    "observation_participants": (
        "observation_participants t JOIN scoped_observations x ON x.observation_id=t.observation_id"
    ),
    "sources": "sources t JOIN scoped_sources x ON x.source_id=t.source_id",
    "evidence_items": "evidence_items t JOIN scoped_evidence x ON x.evidence_id=t.evidence_id",
    "observation_evidence": (
        "observation_evidence t JOIN scoped_observations x ON x.observation_id=t.observation_id"
    ),
    "claims": "claims t JOIN scoped_claims x ON x.claim_id=t.claim_id",
    "claim_subset_memberships": (
        "claim_subset_memberships t JOIN scoped_claims x ON x.claim_id=t.claim_id"
    ),
    "claim_evidence": "claim_evidence t JOIN scoped_claims x ON x.claim_id=t.claim_id",
    "claim_resolutions": "claim_resolutions t JOIN scoped_claims x ON x.claim_id=t.claim_id",
}


class LegacyNormalizedStore:
    """Strict read-only view of the removed normalized-store schema."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).expanduser().resolve()

    def connect(self) -> sqlite3.Connection:
        uri = self.path.as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        self._validate(connection)
        connection.execute("BEGIN")
        return connection

    @staticmethod
    def _validate(connection: sqlite3.Connection) -> None:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        missing = set(_REQUIRED_COLUMNS) - tables
        if missing:
            raise LegacySchemaError(
                "legacy store is missing tables: " + ", ".join(sorted(missing))
            )
        for table, required in _REQUIRED_COLUMNS.items():
            actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
            if actual != required:
                raise LegacySchemaError(
                    f"legacy table {table!r} columns differ from schema 1"
                )
        metadata = dict(connection.execute("SELECT key,value FROM schema_metadata"))
        if metadata != {
            "schema_version": LEGACY_SCHEMA_VERSION,
            "record_standard": LEGACY_RECORD_STANDARD,
        }:
            raise LegacySchemaError("legacy schema metadata is not the supported frozen version")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise LegacySchemaError("legacy store failed SQLite integrity_check")

    def releases(self, connection: sqlite3.Connection) -> Iterator[Dict[str, Any]]:
        rows = connection.execute(
            "SELECT d.dataset_id,d.title,d.description,r.*,p.revision AS policy_revision,"
            "p.config_json AS policy_json,p.config_sha256 AS policy_sha256 "
            "FROM dataset_releases r JOIN datasets d USING(dataset_id) "
            "JOIN dataset_policies p USING(dataset_id) "
            "ORDER BY d.dataset_id,r.release_serial,r.release_id"
        )
        for row in rows:
            yield dict(row)

    @staticmethod
    def records(connection: sqlite3.Connection, release_id: str) -> Iterator[Dict[str, Any]]:
        for row in connection.execute(
            "SELECT * FROM records WHERE release_id=? ORDER BY public_index,record_id",
            (release_id,),
        ):
            yield dict(row)

    @staticmethod
    def _scope_sql(table: str, record_scope: str) -> str:
        columns = _REQUIRED_COLUMNS[table]
        ordering = ",".join(f"t.{name}" for name in columns)
        cte = _SCOPE_CTE.format(record_scope=record_scope)
        if table == "datasets":
            return (
                cte + " SELECT DISTINCT t.* FROM datasets t JOIN dataset_releases r "
                "ON r.dataset_id=t.dataset_id JOIN scoped_records sr ON sr.record_id IN "
                "(SELECT record_id FROM records WHERE release_id=r.release_id) "
                f"ORDER BY {ordering}"
            )
        if table == "dataset_policies":
            return (
                cte + " SELECT DISTINCT t.* FROM dataset_policies t JOIN dataset_releases r "
                "ON r.dataset_id=t.dataset_id JOIN records q ON q.release_id=r.release_id "
                "JOIN scoped_records sr ON sr.record_id=q.record_id "
                f"ORDER BY {ordering}"
            )
        if table == "dataset_releases":
            return (
                cte + " SELECT DISTINCT t.* FROM dataset_releases t JOIN records q "
                "ON q.release_id=t.release_id JOIN scoped_records sr ON sr.record_id=q.record_id "
                f"ORDER BY {ordering}"
            )
        if table == "ingestion_assets":
            return (
                cte + " SELECT DISTINCT t.* FROM ingestion_assets t JOIN scoped_assets x "
                "ON x.asset_id=t.asset_id "
                f"ORDER BY {ordering}"
            )
        source = _TABLE_FROM[table]
        separator = " " if " WHERE " in source else ""
        return cte + f" SELECT DISTINCT t.* FROM {source}{separator} ORDER BY {ordering}"

    def iter_scoped_rows(
        self,
        connection: sqlite3.Connection,
        release_id: str,
        table: str,
    ) -> Iterator[Dict[str, Any]]:
        sql = self._scope_sql(
            table,
            "SELECT record_id FROM records WHERE release_id=?",
        )
        for row in connection.execute(sql, (release_id,)):
            yield dict(row)

    def rows_for_case(
        self,
        connection: sqlite3.Connection,
        record_id: str,
        tables: Iterable[str] = _SCOPED_TABLES,
    ) -> Dict[str, list[Dict[str, Any]]]:
        result = {}
        for table in tables:
            sql = self._scope_sql(table, "SELECT ?")
            result[table] = [dict(row) for row in connection.execute(sql, (record_id,))]
        return result

    @staticmethod
    def identifier_collisions(
        connection: sqlite3.Connection,
        release_id: str,
    ) -> set[Tuple[str, str]]:
        return {
            (row["namespace"], row["value"])
            for row in connection.execute(
                "SELECT x.namespace,x.value FROM external_identifiers x "
                "JOIN record_entities re USING(entity_id) JOIN records r USING(record_id) "
                "WHERE r.release_id=? GROUP BY x.namespace,x.value "
                "HAVING COUNT(DISTINCT r.record_id)>1",
                (release_id,),
            )
        }

    def release_inventory(
        self,
        connection: sqlite3.Connection,
        release_id: str,
        directory: Path,
    ) -> Dict[str, Dict[str, Any]]:
        inventory = _RowInventory(directory)
        try:
            for table in _SCOPED_TABLES:
                for row in self.iter_scoped_rows(connection, release_id, table):
                    inventory.add(table, row)
            return inventory.summary()
        finally:
            inventory.close()


def _sql_json_object(alias: str, table: str) -> str:
    arguments = []
    for column in _REQUIRED_COLUMNS[table]:
        arguments.extend((f"'{column}'", f'{alias}."{column}"'))
    return "json_object(" + ",".join(arguments) + ")"


class _ReleaseCaseRows:
    """One-pass, disk-backed association index for a large legacy release.

    Building this spool executes a bounded number of set-oriented legacy
    queries.  Export then performs one indexed lookup per Case instead of
    millions of repeated relationship queries against the live source.
    """

    def __init__(
        self,
        legacy_db: Path,
        release_id: str,
        directory: Path,
    ):
        descriptor, name = tempfile.mkstemp(
            prefix=".legacy-release-rows-",
            suffix=".sqlite3",
            dir=directory,
        )
        os.close(descriptor)
        self.path = Path(name)
        self.connection = sqlite3.connect(self.path, uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "ATTACH DATABASE ? AS legacy",
            (legacy_db.as_uri() + "?mode=ro",),
        )
        self._build(release_id)

    def _build(self, release_id: str) -> None:
        connection = self.connection
        connection.executescript(
            """
            CREATE TABLE case_rows(
              record_id TEXT NOT NULL,
              table_name TEXT NOT NULL,
              row_json TEXT NOT NULL,
              PRIMARY KEY(record_id,table_name,row_json)
            ) WITHOUT ROWID;
            CREATE TABLE case_entities(
              record_id TEXT NOT NULL,
              entity_id TEXT NOT NULL,
              PRIMARY KEY(record_id,entity_id)
            ) WITHOUT ROWID;
            CREATE TABLE case_observations(
              record_id TEXT NOT NULL,
              observation_id TEXT NOT NULL,
              PRIMARY KEY(record_id,observation_id)
            ) WITHOUT ROWID;
            CREATE TABLE case_claims(
              record_id TEXT NOT NULL,
              claim_id TEXT NOT NULL,
              PRIMARY KEY(record_id,claim_id)
            ) WITHOUT ROWID;
            CREATE TABLE case_structures(
              record_id TEXT NOT NULL,
              structure_id TEXT NOT NULL,
              PRIMARY KEY(record_id,structure_id)
            ) WITHOUT ROWID;
            CREATE TABLE case_assets(
              record_id TEXT NOT NULL,
              asset_id TEXT NOT NULL,
              PRIMARY KEY(record_id,asset_id)
            ) WITHOUT ROWID;
            CREATE TABLE case_evidence(
              record_id TEXT NOT NULL,
              evidence_id TEXT NOT NULL,
              PRIMARY KEY(record_id,evidence_id)
            ) WITHOUT ROWID;
            CREATE TABLE case_sources(
              record_id TEXT NOT NULL,
              source_id TEXT NOT NULL,
              PRIMARY KEY(record_id,source_id)
            ) WITHOUT ROWID;
            CREATE TABLE case_properties(
              record_id TEXT NOT NULL,
              property_type_id TEXT NOT NULL,
              PRIMARY KEY(record_id,property_type_id)
            ) WITHOUT ROWID;
            CREATE TABLE case_subsets(
              record_id TEXT NOT NULL,
              subset_id TEXT NOT NULL,
              PRIMARY KEY(record_id,subset_id)
            ) WITHOUT ROWID;
            """
        )
        parameters = (release_id,)
        connection.execute(
            "INSERT INTO case_observations "
            "SELECT o.record_id,o.observation_id FROM legacy.observations o "
            "JOIN legacy.records r USING(record_id) WHERE r.release_id=?",
            parameters,
        )
        connection.execute(
            "INSERT INTO case_claims "
            "SELECT c.record_id,c.claim_id FROM legacy.claims c "
            "JOIN legacy.records r USING(record_id) WHERE r.release_id=?",
            parameters,
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_entities "
            "SELECT re.record_id,re.entity_id FROM legacy.record_entities re "
            "JOIN legacy.records r USING(record_id) WHERE r.release_id=?",
            parameters,
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_entities "
            "SELECT co.record_id,o.subject_entity_id FROM case_observations co "
            "JOIN legacy.observations o USING(observation_id) "
            "WHERE o.subject_entity_id IS NOT NULL"
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_entities "
            "SELECT co.record_id,op.entity_id FROM case_observations co "
            "JOIN legacy.observation_participants op USING(observation_id)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_entities "
            "SELECT cc.record_id,c.subject_entity_id FROM case_claims cc "
            "JOIN legacy.claims c USING(claim_id) WHERE c.subject_entity_id IS NOT NULL"
        )
        while True:
            before = connection.total_changes
            connection.execute(
                "INSERT OR IGNORE INTO case_entities "
                "SELECT ce.record_id,e.parent_entity_id FROM case_entities ce "
                "JOIN legacy.entities e USING(entity_id) WHERE e.parent_entity_id IS NOT NULL"
            )
            if connection.total_changes == before:
                break
        connection.execute(
            "INSERT OR IGNORE INTO case_structures "
            "SELECT ce.record_id,s.structure_id FROM case_entities ce "
            "JOIN legacy.structures s USING(entity_id)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_assets "
            "SELECT cs.record_id,sa.asset_id FROM case_structures cs "
            "JOIN legacy.structure_assets sa USING(structure_id)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_evidence "
            "SELECT co.record_id,oe.evidence_id FROM case_observations co "
            "JOIN legacy.observation_evidence oe USING(observation_id)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_evidence "
            "SELECT cc.record_id,ce.evidence_id FROM case_claims cc "
            "JOIN legacy.claim_evidence ce USING(claim_id)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_sources "
            "SELECT ce.record_id,ei.source_id FROM case_evidence ce "
            "JOIN legacy.evidence_items ei USING(evidence_id)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_properties "
            "SELECT co.record_id,o.property_type_id FROM case_observations co "
            "JOIN legacy.observations o USING(observation_id)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_properties "
            "SELECT cc.record_id,c.property_type_id FROM case_claims cc "
            "JOIN legacy.claims c USING(claim_id) WHERE c.property_type_id IS NOT NULL"
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_subsets "
            "SELECT r.record_id,sm.subset_id FROM legacy.records r "
            "JOIN legacy.subset_memberships sm USING(record_id) WHERE r.release_id=?",
            parameters,
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_subsets "
            "SELECT co.record_id,osm.subset_id FROM case_observations co "
            "JOIN legacy.observation_subset_memberships osm USING(observation_id)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO case_subsets "
            "SELECT cc.record_id,csm.subset_id FROM case_claims cc "
            "JOIN legacy.claim_subset_memberships csm USING(claim_id)"
        )

        associations = {
            "datasets": (
                "legacy.datasets t JOIN legacy.dataset_releases dr USING(dataset_id) "
                "JOIN legacy.records r USING(release_id)",
                "r.record_id",
                "r.release_id=?",
            ),
            "dataset_policies": (
                "legacy.dataset_policies t JOIN legacy.dataset_releases dr USING(dataset_id) "
                "JOIN legacy.records r USING(release_id)",
                "r.record_id",
                "r.release_id=?",
            ),
            "dataset_releases": (
                "legacy.dataset_releases t JOIN legacy.records r USING(release_id)",
                "r.record_id",
                "r.release_id=?",
            ),
            "records": ("legacy.records t", "t.record_id", "t.release_id=?"),
            "subsets": (
                "legacy.subsets t JOIN case_subsets x USING(subset_id)",
                "x.record_id",
                None,
            ),
            "subset_memberships": (
                "legacy.subset_memberships t JOIN legacy.records r USING(record_id)",
                "r.record_id",
                "r.release_id=?",
            ),
            "entities": (
                "legacy.entities t JOIN case_entities x USING(entity_id)",
                "x.record_id",
                None,
            ),
            "external_identifiers": (
                "legacy.external_identifiers t JOIN case_entities x USING(entity_id)",
                "x.record_id",
                None,
            ),
            "record_entities": (
                "legacy.record_entities t JOIN legacy.records r USING(record_id)",
                "r.record_id",
                "r.release_id=?",
            ),
            "entity_relations": (
                "legacy.entity_relations t JOIN case_entities x "
                "ON x.entity_id=t.subject_entity_id OR x.entity_id=t.object_entity_id",
                "x.record_id",
                None,
            ),
            "assets": ("legacy.assets t JOIN case_assets x USING(asset_id)", "x.record_id", None),
            "structures": (
                "legacy.structures t JOIN case_structures x USING(structure_id)",
                "x.record_id",
                None,
            ),
            "structure_assets": (
                "legacy.structure_assets t JOIN case_structures x USING(structure_id)",
                "x.record_id",
                None,
            ),
            "ingestion_assets": (
                "legacy.ingestion_assets t JOIN case_assets x USING(asset_id) "
                "JOIN legacy.dataset_releases dr ON dr.ingestion_id=t.ingestion_id",
                "x.record_id",
                "dr.release_id=?",
            ),
            "computed_descriptors": (
                "legacy.computed_descriptors t JOIN case_entities x USING(entity_id)",
                "x.record_id",
                None,
            ),
            "property_types": (
                "legacy.property_types t JOIN case_properties x USING(property_type_id)",
                "x.record_id",
                None,
            ),
            "observations": (
                "legacy.observations t JOIN case_observations x USING(observation_id)",
                "x.record_id",
                None,
            ),
            "observation_subset_memberships": (
                "legacy.observation_subset_memberships t "
                "JOIN case_observations x USING(observation_id)",
                "x.record_id",
                None,
            ),
            "observation_participants": (
                "legacy.observation_participants t JOIN case_observations x USING(observation_id)",
                "x.record_id",
                None,
            ),
            "sources": ("legacy.sources t JOIN case_sources x USING(source_id)", "x.record_id", None),
            "evidence_items": (
                "legacy.evidence_items t JOIN case_evidence x USING(evidence_id)",
                "x.record_id",
                None,
            ),
            "observation_evidence": (
                "legacy.observation_evidence t JOIN case_observations x USING(observation_id)",
                "x.record_id",
                None,
            ),
            "claims": ("legacy.claims t JOIN case_claims x USING(claim_id)", "x.record_id", None),
            "claim_subset_memberships": (
                "legacy.claim_subset_memberships t JOIN case_claims x USING(claim_id)",
                "x.record_id",
                None,
            ),
            "claim_evidence": (
                "legacy.claim_evidence t JOIN case_claims x USING(claim_id)",
                "x.record_id",
                None,
            ),
            "claim_resolutions": (
                "legacy.claim_resolutions t JOIN case_claims x USING(claim_id)",
                "x.record_id",
                None,
            ),
        }
        for table, (source, record_expression, condition) in associations.items():
            where = "" if condition is None else f" WHERE {condition}"
            sql = (
                "INSERT OR IGNORE INTO case_rows(record_id,table_name,row_json) "
                f"SELECT DISTINCT {record_expression},?,{_sql_json_object('t', table)} "
                f"FROM {source}{where}"
            )
            values: Tuple[Any, ...] = (table,) if condition is None else (table, release_id)
            connection.execute(sql, values)
        connection.commit()
        connection.execute("DETACH DATABASE legacy")

    def rows_for_case(
        self,
        record_id: str,
        tables: Iterable[str] = _SCOPED_TABLES,
    ) -> Dict[str, list[Dict[str, Any]]]:
        selected = tuple(tables)
        result = {table: [] for table in selected}
        placeholders = ",".join("?" for _ in selected)
        for row in self.connection.execute(
            "SELECT table_name,row_json FROM case_rows WHERE record_id=? "
            f"AND table_name IN ({placeholders}) ORDER BY table_name,row_json",
            (record_id, *selected),
        ):
            result[row["table_name"]].append(json.loads(row["row_json"]))
        return result

    def release_inventory(self, directory: Path) -> Dict[str, Dict[str, Any]]:
        inventory = _RowInventory(directory)
        try:
            for row in self.connection.execute(
                "SELECT DISTINCT table_name,row_json FROM case_rows "
                "ORDER BY table_name,row_json"
            ):
                inventory.add(row["table_name"], json.loads(row["row_json"]))
            return inventory.summary()
        finally:
            inventory.close()

    def close(self) -> None:
        self.connection.close()
        self.path.unlink(missing_ok=True)


def _identity_inventory_rows(
    source: LegacyNormalizedStore,
    connection: sqlite3.Connection,
    release_id: str,
) -> Iterator[Dict[str, Any]]:
    for table in _IDENTITY_TABLES:
        for row in source.iter_scoped_rows(connection, release_id, table):
            yield _inventory_identity(table, row)


def _identity_migration_rows(
    source: LegacyNormalizedStore,
    connection: sqlite3.Connection,
    release_id: str,
    case_rows: Optional[_ReleaseCaseRows] = None,
) -> Iterator[Dict[str, Any]]:
    for record in source.records(connection, release_id):
        case_id = record["record_id"]
        rows = (
            source.rows_for_case(connection, case_id, _IDENTITY_TABLES)
            if case_rows is None
            else case_rows.rows_for_case(case_id, _IDENTITY_TABLES)
        )
        entity_map = {
            row["entity_id"]: _case_child_id(case_id, "entity", row["entity_id"])
            for row in rows["entities"]
        }
        structure_map = {
            row["structure_id"]: _case_child_id(case_id, "structure", row["structure_id"])
            for row in rows["structures"]
        }
        for table in _IDENTITY_TABLES:
            for row in rows[table]:
                inventory = _inventory_identity(table, row)
                old_ids: list[str] = []
                new_ids: list[str] = []
                outcome = "retained"
                reason = "the legacy relation remains scientifically operative"
                if table == "records":
                    old_ids = [row["record_id"]]
                    new_ids = [case_id]
                elif table == "entities":
                    old_ids = [row["entity_id"]]
                    new_ids = [entity_map[row["entity_id"]]]
                    outcome = "replaced"
                    reason = "legacy entity identity is retained as provenance; current subject identity is Case-scoped"
                elif table == "record_entities":
                    old_ids = [row["record_id"], row["entity_id"]]
                    new_ids = [case_id, entity_map[row["entity_id"]]]
                    outcome = "replaced"
                    reason = "the Case relation now targets a Case-scoped entity"
                elif table == "structures":
                    old_ids = [row["structure_id"], row["entity_id"]]
                    new_ids = [structure_map[row["structure_id"]], entity_map[row["entity_id"]]]
                    outcome = "replaced"
                    reason = "legacy coordinate/asset-derived IDs cannot establish Structure or sample identity"
                elif table == "structure_assets":
                    old_ids = [row["structure_id"], row["asset_id"]]
                    new_ids = [structure_map[row["structure_id"]], row["asset_id"]]
                    outcome = "replaced"
                    reason = "the byte asset is retained while its Structure target becomes Case-scoped"
                elif table == "external_identifiers":
                    namespace = _identifier_namespace(row["namespace"])
                    old_ids = [row["entity_id"], f"{row['namespace']}:{row['value']}"]
                    new_ids = [entity_map[row["entity_id"]], f"{namespace}:{row['value']}"]
                    outcome = "replaced"
                    reason = (
                        "the external identifier is retained on the Case-scoped entity; "
                        "legacy CSD_identifier is normalized to the current CSD namespace"
                    )
                elif table == "entity_relations":
                    old_ids = [row["subject_entity_id"], row["object_entity_id"]]
                    mapped = [
                        entity_map[item]
                        for item in old_ids
                        if item in entity_map
                    ]
                    new_ids = mapped
                    if len(mapped) == 2:
                        outcome = "replaced"
                        reason = "both endpoints were remapped within this Case"
                    else:
                        outcome = "unresolved"
                        reason = "the legacy relation crosses the Case identity boundary"
                value = {
                    "mapping_id": _digest_json(
                        {
                            "inventory": inventory["row_id"],
                            "case": case_id,
                            "new_ids": new_ids,
                        }
                    ),
                    "legacy_inventory_row_id": inventory["row_id"],
                    "legacy_inventory_row_sha256": inventory["row_sha256"],
                    "case_id": case_id,
                    "old_ids": old_ids,
                    "new_ids": new_ids,
                    "affected_relations": [table],
                    "outcome": outcome,
                    "comparison_basis": "legacy row plus Case ownership and audited external identifiers",
                    "reason": reason,
                }
                if table == "external_identifiers" and row["namespace"] == "CSD_identifier":
                    value["namespace_rewrite"] = {
                        "legacy": "CSD_identifier",
                        "current": "CSD",
                    }
                yield value


def _identity_mapping_is_valid(value: Mapping[str, Any]) -> bool:
    case_id = value["case_id"]
    table = value["affected_relations"][0]
    old_ids = value["old_ids"]
    new_ids = value["new_ids"]
    expected_mapping_id = _digest_json(
        {
            "inventory": value["legacy_inventory_row_id"],
            "case": case_id,
            "new_ids": new_ids,
        }
    )
    if value["mapping_id"] != expected_mapping_id:
        return False
    entity_prefix = f"{case_id}:entity:"
    structure_prefix = f"{case_id}:structure:"
    if table == "records":
        return new_ids == [case_id]
    if table == "entities":
        return len(new_ids) == 1 and new_ids[0].startswith(entity_prefix)
    if table == "record_entities":
        return (
            len(new_ids) == 2
            and new_ids[0] == case_id
            and new_ids[1].startswith(entity_prefix)
        )
    if table == "structures":
        return (
            len(new_ids) == 2
            and new_ids[0].startswith(structure_prefix)
            and new_ids[1].startswith(entity_prefix)
        )
    if table == "structure_assets":
        return (
            len(new_ids) == 2
            and new_ids[0].startswith(structure_prefix)
            and new_ids[1] == old_ids[1]
        )
    if table == "external_identifiers":
        return (
            len(new_ids) == 2
            and new_ids[0].startswith(entity_prefix)
            and not new_ids[1].startswith("CSD_identifier:")
        )
    if table == "entity_relations":
        if value["outcome"] == "unresolved":
            return len(new_ids) < 2
        return len(new_ids) == 2 and all(
            item.startswith(entity_prefix) for item in new_ids
        )
    return False


def _asset_descriptor(
    asset: Mapping[str, Any],
    legacy_asset_root: Path,
    bundle_root: Path,
    *,
    materialize: bool = True,
) -> Dict[str, Any]:
    metadata = _json_value(asset["metadata_json"]) or {}
    descriptor: Dict[str, Any] = {
        "format": str(metadata.get("format") or PurePosixPath(asset["uri"]).suffix.lstrip(".") or "bin"),
        "role": "coordinates",
        "media_type": asset["media_type"],
        "legacy_asset_id": asset["asset_id"],
        "access_policy": asset["access_policy"],
    }
    expected = asset["content_sha256"]
    if expected:
        descriptor["expected_sha256"] = expected
    source_path = _safe_legacy_asset(legacy_asset_root, asset["uri"])
    if source_path is None:
        descriptor["remote_not_materialized"] = True
        descriptor["uri"] = asset["uri"]
        return descriptor
    if not source_path.is_file():
        descriptor["path"] = f"assets/missing/{asset['asset_id']}"
        descriptor["source_missing"] = True
        descriptor["legacy_uri"] = asset["uri"]
        return descriptor
    actual, _ = _sha256_file(source_path)
    suffix = source_path.suffix.lower() or ".bin"
    relative = Path("assets") / "objects" / actual[:2] / f"{actual}{suffix}"
    if materialize:
        destination = bundle_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            found, _ = _sha256_file(destination)
            if found != actual:
                raise LegacyMigrationError(f"migration asset collision at {destination}")
        else:
            try:
                os.link(source_path, destination)
            except OSError:
                shutil.copy2(source_path, destination)
    descriptor["path"] = relative.as_posix()
    descriptor["observed_sha256"] = actual
    descriptor["legacy_uri"] = asset["uri"]
    return descriptor


def _entity_mapping(case_id: str, rows: Mapping[str, list[Dict[str, Any]]]) -> Dict[str, str]:
    return {
        row["entity_id"]: _case_child_id(case_id, "entity", row["entity_id"])
        for row in rows["entities"]
    }


def _normalize_identifiers(
    rows: Mapping[str, list[Dict[str, Any]]],
    _collisions: set[Tuple[str, str]],
) -> Dict[str, Any]:
    grouped: Dict[str, list[str]] = {}
    record_entities = {row["entity_id"] for row in rows["record_entities"]}
    for item in rows["external_identifiers"]:
        if item["entity_id"] not in record_entities:
            continue
        namespace = _identifier_namespace(item["namespace"])
        values = grouped.setdefault(namespace, [])
        if item["value"] not in values:
            values.append(item["value"])
    return {
        namespace: values[0] if len(values) == 1 else sorted(values)
        for namespace, values in sorted(grouped.items())
    }


def _normalize_entities(
    case_id: str,
    rows: Mapping[str, list[Dict[str, Any]]],
    entity_map: Mapping[str, str],
) -> list[Dict[str, Any]]:
    roles: Dict[str, list[str]] = {}
    for relation in rows["record_entities"]:
        roles.setdefault(relation["entity_id"], []).append(relation["role"])
    identifiers: Dict[str, list[Dict[str, Any]]] = {}
    for identifier in rows["external_identifiers"]:
        normalized = dict(identifier)
        normalized["entity_id"] = entity_map[identifier["entity_id"]]
        normalized["namespace"] = _identifier_namespace(identifier["namespace"])
        identifiers.setdefault(identifier["entity_id"], []).append(normalized)
    relations: Dict[str, list[Dict[str, Any]]] = {}
    for relation in rows["entity_relations"]:
        subject = relation["subject_entity_id"]
        object_ = relation["object_entity_id"]
        if subject not in entity_map or object_ not in entity_map:
            continue
        normalized = dict(relation)
        normalized["subject_entity_id"] = entity_map[subject]
        normalized["object_entity_id"] = entity_map[object_]
        relations.setdefault(subject, []).append(normalized)
    result = []
    for raw in rows["entities"]:
        old_id = raw["entity_id"]
        parent = raw["parent_entity_id"]
        result.append(
            {
                "entity_id": entity_map[old_id],
                "entity_kind": raw["entity_kind"],
                "parent_entity_id": entity_map.get(parent) if parent else None,
                "roles": sorted(set(roles.get(old_id, []))),
                "external_identifiers": identifiers.get(old_id, []),
                "relations": relations.get(old_id, []),
                "metadata": _json_value(raw["metadata_json"]) or {},
                "legacy_entity_id": old_id,
                "identity_status": "case_scoped_after_legacy_identity_audit",
                "case_id": case_id,
            }
        )
    return result


def _normalize_structures(
    case_id: str,
    rows: Mapping[str, list[Dict[str, Any]]],
    entity_map: Mapping[str, str],
    legacy_asset_root: Path,
    bundle_root: Path,
    *,
    materialize_assets: bool = True,
) -> list[Dict[str, Any]]:
    roles: Dict[str, list[str]] = {}
    for relation in rows["record_entities"]:
        roles.setdefault(relation["entity_id"], []).append(relation["role"])
    assets = {row["asset_id"]: row for row in rows["assets"]}
    by_structure: Dict[str, list[Dict[str, Any]]] = {}
    for relation in rows["structure_assets"]:
        asset = dict(assets[relation["asset_id"]])
        descriptor = _asset_descriptor(
            asset,
            legacy_asset_root,
            bundle_root,
            materialize=materialize_assets,
        )
        descriptor["role"] = relation["role"]
        by_structure.setdefault(relation["structure_id"], []).append(descriptor)
    result = []
    for raw in rows["structures"]:
        structure_id = _case_child_id(case_id, "structure", raw["structure_id"])
        entity_roles = sorted(set(roles.get(raw["entity_id"], [])))
        result.append(
            {
                "structure_id": structure_id,
                "entity_id": entity_map[raw["entity_id"]],
                "role": entity_roles[0] if entity_roles else "structure",
                "roles": entity_roles,
                "primary": len(result) == 0,
                "status": "active",
                "coordinate_hash": raw["coordinate_hash"],
                "atom_count": raw["atom_count"],
                "formal_charge": raw["formal_charge"],
                "metadata": _json_value(raw["metadata_json"]) or {},
                "legacy_structure_id": raw["structure_id"],
                "representations": by_structure.get(raw["structure_id"], []),
            }
        )
    return result


def _normalize_observations(
    rows: Mapping[str, list[Dict[str, Any]]],
    entity_map: Mapping[str, str],
) -> list[Dict[str, Any]]:
    participants: Dict[str, list[Dict[str, Any]]] = {}
    for raw in rows["observation_participants"]:
        participants.setdefault(raw["observation_id"], []).append(
            {
                "entity_id": entity_map[raw["entity_id"]],
                "role": raw["role"],
                "stoichiometry": raw["stoichiometry"],
                "metadata": _json_value(raw["metadata_json"]) or {},
            }
        )
    memberships: Dict[str, list[str]] = {}
    for raw in rows["observation_subset_memberships"]:
        memberships.setdefault(raw["observation_id"], []).append(raw["subset_id"])
    result = []
    for raw in rows["observations"]:
        result.append(
            {
                "observation_id": raw["observation_id"],
                "subject_entity_id": entity_map[raw["subject_entity_id"]],
                "property": raw["property_type_id"],
                "value": _json_value(raw["value_json"]),
                "unit": raw["unit"],
                "uncertainty": _json_value(raw["uncertainty_json"]) or {},
                "conditions": _json_value(raw["conditions_json"]) or {},
                "status": raw["status"],
                "method": _json_value(raw["method_json"]) or {},
                "metadata": _json_value(raw["metadata_json"]) or {},
                "participants": participants.get(raw["observation_id"], []),
                "legacy_subset_ids": sorted(memberships.get(raw["observation_id"], [])),
                "claimed_evidence_level": raw["evidence_level"],
            }
        )
    return result


def _normalize_claims(
    rows: Mapping[str, list[Dict[str, Any]]],
    entity_map: Mapping[str, str],
) -> list[Dict[str, Any]]:
    memberships: Dict[str, list[str]] = {}
    for raw in rows["claim_subset_memberships"]:
        memberships.setdefault(raw["claim_id"], []).append(raw["subset_id"])
    resolutions: Dict[str, list[Dict[str, Any]]] = {}
    for raw in rows["claim_resolutions"]:
        resolutions.setdefault(raw["claim_id"], []).append(raw)
    return [
        {
            "claim_id": raw["claim_id"],
            "subject_entity_id": entity_map.get(raw["subject_entity_id"]),
            "property": raw["property_type_id"],
            "text": raw["text"],
            "status": "active",
            "resolution_status": raw["status"],
            "metadata": _json_value(raw["metadata_json"]) or {},
            "legacy_subset_ids": sorted(memberships.get(raw["claim_id"], [])),
            "resolutions": resolutions.get(raw["claim_id"], []),
            "claimed_evidence_level": raw["evidence_level"],
        }
        for raw in rows["claims"]
    ]


def _normalize_evidence(rows: Mapping[str, list[Dict[str, Any]]]) -> list[Dict[str, Any]]:
    items = {row["evidence_id"]: row for row in rows["evidence_items"]}
    sources = {row["source_id"]: row for row in rows["sources"]}
    links: list[Tuple[str, str, str, str]] = []
    links.extend(
        (row["observation_id"], "observation", row["evidence_id"], row["role"])
        for row in rows["observation_evidence"]
    )
    links.extend(
        (row["claim_id"], "claim", row["evidence_id"], row["role"])
        for row in rows["claim_evidence"]
    )
    result = []
    for target, target_kind, evidence_id, role in sorted(links):
        item = items[evidence_id]
        source = sources[item["source_id"]]
        child_id = f"{evidence_id}:target:{hashlib.sha256(target.encode('utf-8')).hexdigest()[:16]}"
        result.append(
            {
                "evidence_id": child_id,
                "target": target,
                "target_kind": target_kind,
                "relation": "supports",
                "role": role,
                "evidence_type": item["evidence_type"],
                "locator": item["locator"],
                "quote": item["quote"],
                "matched_text": item["matched_text"],
                "source_id": source["source_id"],
                "source_type": source["source_type"],
                "citation": source["citation"],
                "doi": source["doi"],
                "scientific_work_id": source["doi"],
                "source_lineage_id": source["source_id"],
                "license": source["license"],
                "access_policy": source["access_policy"],
                "metadata": {
                    "evidence": _json_value(item["metadata_json"]) or {},
                    "source": _json_value(source["metadata_json"]) or {},
                },
                "legacy_evidence_id": evidence_id,
                "status": "active",
            }
        )
    return result


def _core_info(
    record: Mapping[str, Any],
    required: Sequence[str],
) -> Dict[str, Any]:
    metadata = _json_value(record["metadata_json"]) or {}
    source_fields = metadata.get("source_fields", {})
    result = {}
    for name in required:
        if name in metadata:
            result[name] = metadata[name]
        elif isinstance(source_fields, Mapping) and name in source_fields:
            result[name] = source_fields[name]
    return result


def _migration_control_descriptors(root: Path) -> list[Dict[str, Any]]:
    paths = (
        ("legacy-identity-relationships", root / "evidence" / "migration" / "legacy-identity-relationships.jsonl"),
        ("identity-migration", root / "evidence" / "migration" / "identity-migration.jsonl"),
        ("legacy-source-inventory", root / "evidence" / "migration" / "legacy-source-inventory.json"),
    )
    result = []
    for name, path in paths:
        digest, size = _sha256_file(path)
        result.append(
            {
                "category": "migration",
                "name": name,
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "size": size,
            }
        )
    return result


def _supplier_attestations(case_id: str, record: Mapping[str, Any], rows: Mapping[str, list[Dict[str, Any]]]) -> list[Dict[str, Any]]:
    metadata = _json_value(record["metadata_json"]) or {}
    subsets = [item["name"] for item in rows["subsets"]]
    grade = metadata.get("source_evidence_grade")
    asserted = bool(grade and str(grade).lower().startswith("gold")) or any(
        "gold" in str(item).lower() for item in subsets
    )
    if not asserted:
        return []
    return [
        {
            "attestation_id": f"{case_id}:legacy-supplier-benchmark-claim",
            "authority": "Other",
            "actor": "legacy-data-supplier",
            "decision": "approve",
            "reason": "Preserved supplier gold/benchmark claim; it does not bypass the current BenchmarkGate.",
            "status": "active",
            "metadata": {"source_evidence_grade": grade, "legacy_subsets": subsets},
        }
    ]


def _normalized_case(
    record: Mapping[str, Any],
    rows: Mapping[str, list[Dict[str, Any]]],
    bootstrap: DatasetBootstrap,
    legacy_asset_root: Path,
    bundle_root: Path,
    identifier_collisions: set[Tuple[str, str]],
    *,
    materialize_assets: bool = True,
) -> Dict[str, Any]:
    case_id = record["record_id"]
    entity_map = _entity_mapping(case_id, rows)
    release = rows["dataset_releases"][0]
    evidence = _normalize_evidence(rows)
    attestations = _supplier_attestations(case_id, record, rows)
    claimed_subsets = [item["name"] for item in rows["subsets"]]
    return {
        "id": case_id,
        "local_id": record["local_id"],
        "public_index": record["public_index"],
        "identifiers": _normalize_identifiers(rows, identifier_collisions),
        "metadata": {
            "legacy_record_metadata": _json_value(record["metadata_json"]) or {},
            "legacy_release_quality_level": release["quality_level"],
            "legacy_subsets": claimed_subsets,
            "migration_rows": rows,
            "migration_schema": MIGRATION_SCHEMA,
            "quarantined_entity_relations": [
                dict(relation)
                for relation in rows["entity_relations"]
                if relation["subject_entity_id"] not in entity_map
                or relation["object_entity_id"] not in entity_map
            ],
            "shared_identifier_relations": [
                {"namespace": _identifier_namespace(namespace), "value": value}
                for namespace, value in sorted(identifier_collisions)
                if any(
                    _identifier_namespace(item["namespace"])
                    == _identifier_namespace(namespace)
                    and item["value"] == value
                    for item in rows["external_identifiers"]
                )
            ],
        },
        "core_info": _core_info(record, bootstrap.core_info),
        "entities": _normalize_entities(case_id, rows, entity_map),
        "structures": _normalize_structures(
            case_id,
            rows,
            entity_map,
            legacy_asset_root,
            bundle_root,
            materialize_assets=materialize_assets,
        ),
        "observations": _normalize_observations(rows, entity_map),
        "claims": _normalize_claims(rows, entity_map),
        "evidence": evidence,
        "verifications": [],
        "attestations": attestations,
        "claimed_quality_level": release["quality_level"],
        "claimed_archive_class": claimed_subsets,
        "claimed_benchmark": bool(attestations),
    }


def export_legacy_release(
    legacy_db: Union[str, Path],
    legacy_asset_root: Union[str, Path],
    release_id: str,
    bundle_root: Union[str, Path],
    bootstrap: DatasetBootstrap,
) -> Dict[str, Any]:
    """Export one release without modifying the legacy database."""

    source = LegacyNormalizedStore(legacy_db)
    root = Path(bundle_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise LegacyMigrationError(f"Bundle export root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    assets_root = Path(legacy_asset_root).expanduser().resolve()
    connection = source.connect()
    case_rows: Optional[_ReleaseCaseRows] = None
    try:
        release = next(
            (item for item in source.releases(connection) if item["release_id"] == release_id),
            None,
        )
        if release is None:
            raise LegacyMigrationError(f"unknown legacy release: {release_id}")
        if release["dataset_id"] != bootstrap.ds_name:
            raise LegacyMigrationError("bootstrap dataset does not match legacy release")
        if tuple(_json_value(release["core_labels_json"])) != tuple(bootstrap.core_labels):
            raise LegacyMigrationError("owner bootstrap core_labels differ from the legacy release")
        case_rows = _ReleaseCaseRows(Path(legacy_db).resolve(), release_id, root)

        migration_dir = root / "evidence" / "migration"
        inventory_path = migration_dir / "legacy-identity-relationships.jsonl"
        mapping_path = migration_dir / "identity-migration.jsonl"
        inventory_sha, inventory_size, inventory_count = _write_jsonl(
            inventory_path,
            _identity_inventory_rows(source, connection, release_id),
        )
        mapping_sha, mapping_size, mapping_count = _write_jsonl(
            mapping_path,
            _identity_migration_rows(source, connection, release_id, case_rows),
        )
        source_inventory = case_rows.release_inventory(root)
        source_inventory_path = migration_dir / "legacy-source-inventory.json"
        _atomic_json(
            source_inventory_path,
            {
                "schema": MIGRATION_SCHEMA,
                "dataset": bootstrap.ds_name,
                "release": release_id,
                "tables": source_inventory,
            },
        )
        source_inventory_sha, _ = _sha256_file(source_inventory_path)
        identifier_collisions = source.identifier_collisions(connection, release_id)
        migration_controls = _migration_control_descriptors(root)

        manifest = {
            "schema": "hotpot-ds-bundle/current",
            "ds_name": bootstrap.ds_name,
            "dataset_id": bootstrap.ds_name,
            "release_id": release["release_id"],
            "release_serial": release["release_serial"],
            "version": release["version"],
            "title": release["title"],
            "description": release["description"],
            "license": release["license"],
            "access_policy": release["access_policy"],
            "provenance": {
                "legacy": {
                    "database_sha256": _sha256_file(Path(legacy_db).resolve())[0],
                    "release": release_id,
                    "release_provenance": _json_value(release["provenance_json"]) or {},
                    "release_quality": _json_value(release["quality_json"]) or {},
                    "policy_revision": release["policy_revision"],
                    "policy_json": _json_value(release["policy_json"]),
                    "policy_sha256": release["policy_sha256"],
                    "identity_inventory_sha256": inventory_sha,
                    "identity_migration_sha256": mapping_sha,
                    "source_inventory_sha256": source_inventory_sha,
                    "asset_root": str(assets_root),
                },
                "owner_bootstrap": {
                    "owner": bootstrap.owner,
                    "decision": bootstrap.decision,
                },
            },
            "core_labels": list(bootstrap.core_labels),
            "core_info": list(bootstrap.core_info),
            "identity_schema": dict(bootstrap.identity_schema),
            "property_definitions": dict(bootstrap.property_definitions),
            "property_comparators": dict(bootstrap.property_comparators),
            "required_structure_roles": list(bootstrap.required_structure_roles),
            "control_assets": migration_controls,
            "intake": {"format": "jsonl", "path": "cases.jsonl"},
        }
        _atomic_json(root / "bundle.json", manifest)

        cases_path = root / "cases.jsonl"
        count = 0
        digest = hashlib.sha256()
        with cases_path.open("wb") as stream:
            for record in source.records(connection, release_id):
                rows = case_rows.rows_for_case(record["record_id"])
                case = _normalized_case(
                    record,
                    rows,
                    bootstrap,
                    assets_root,
                    root,
                    identifier_collisions,
                )
                line = (_canonical_json(case) + "\n").encode("utf-8")
                stream.write(line)
                digest.update(line)
                count += 1
            stream.flush()
            os.fsync(stream.fileno())

        report = {
            "schema": MIGRATION_SCHEMA,
            "phase": "export",
            "dataset": bootstrap.ds_name,
            "release": release_id,
            "source_database": str(Path(legacy_db).resolve()),
            "bundle_root": str(root),
            "case_count": count,
            "cases_sha256": digest.hexdigest(),
            "source_asset_root": str(assets_root),
            "source_inventory": source_inventory,
            "identifier_collisions": [
                {"namespace": namespace, "value": value}
                for namespace, value in sorted(identifier_collisions)
            ],
            "identity_inventory": {
                "path": inventory_path.relative_to(root).as_posix(),
                "sha256": inventory_sha,
                "size": inventory_size,
                "rows": inventory_count,
            },
            "identity_migration": {
                "path": mapping_path.relative_to(root).as_posix(),
                "sha256": mapping_sha,
                "size": mapping_size,
                "rows": mapping_count,
            },
            "migration_controls": migration_controls,
        }
        _atomic_json(root / "migration-export.json", report)
        return report
    finally:
        if case_rows is not None:
            case_rows.close()
        connection.close()


def verify_bundle_export(
    legacy_db: Union[str, Path],
    release_id: str,
    bundle_root: Union[str, Path],
    legacy_asset_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Independently compare every source Case and operative exported fact."""

    database = Path(legacy_db).expanduser().resolve()
    source = LegacyNormalizedStore(database)
    root = Path(bundle_root).expanduser().resolve()
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    export_report = json.loads(
        (root / "migration-export.json").read_text(encoding="utf-8")
    )
    asset_root = Path(
        export_report["source_asset_root"]
        if legacy_asset_root is None
        else legacy_asset_root
    ).expanduser().resolve()
    database_digest_before, _ = _sha256_file(database)
    if manifest["provenance"]["legacy"]["database_sha256"] != database_digest_before:
        raise MigrationParityError("legacy database differs from the exported source snapshot")
    connection = source.connect()
    source_rows: Optional[_ReleaseCaseRows] = None
    try:
        release = next(
            (item for item in source.releases(connection) if item["release_id"] == release_id),
            None,
        )
        if release is None:
            raise LegacyMigrationError(f"unknown legacy release: {release_id}")
        source_rows = _ReleaseCaseRows(database, release_id, root)
        source_inventory = source_rows.release_inventory(root)
        identifier_collisions = source.identifier_collisions(connection, release_id)
        records = source.records(connection, release_id)
        actual = _RowInventory(root)
        case_ids = set()
        indexes = set()
        case_count = 0
        case_mismatches = []
        case_projection_digest = hashlib.sha256()
        asset_hash_cache: Dict[str, str] = {}
        asset_summary = {
            "present": 0,
            "source_missing": 0,
            "remote_not_materialized": 0,
            "declared_hash_mismatch": 0,
            "copy_integrity_failures": 0,
        }
        try:
            with (root / "cases.jsonl").open("r", encoding="utf-8") as stream:
                for record in records:
                    case_count += 1
                    line = stream.readline()
                    if not line:
                        case_mismatches.append(
                            {"case_id": record["record_id"], "reason": "missing_export_case"}
                        )
                        continue
                    case = json.loads(line)
                    rows = source_rows.rows_for_case(record["record_id"])
                    expected_case = _normalized_case(
                        record,
                        rows,
                        DatasetBootstrap(
                            ds_name=manifest["ds_name"],
                            core_labels=tuple(manifest["core_labels"]),
                            core_info=tuple(manifest["core_info"]),
                            identity_schema=manifest["identity_schema"],
                            property_definitions=manifest["property_definitions"],
                            property_comparators=manifest["property_comparators"],
                            required_structure_roles=tuple(manifest["required_structure_roles"]),
                            gates=(),
                            owner="parity",
                            decision="parity",
                            include_incomplete=False,
                        ),
                        asset_root,
                        root,
                        identifier_collisions,
                        materialize_assets=False,
                    )
                    case_ids.add(case.get("id"))
                    indexes.add(case.get("public_index"))
                    expected_digest = _digest_json(expected_case)
                    actual_digest = _digest_json(case)
                    case_projection_digest.update(bytes.fromhex(actual_digest))
                    if expected_digest != actual_digest and len(case_mismatches) < 25:
                        case_mismatches.append(
                            {
                                "case_id": record["record_id"],
                                "reason": "operative_case_projection_changed",
                                "source_sha256": expected_digest,
                                "export_sha256": actual_digest,
                            }
                        )
                    migration_rows = case.get("metadata", {}).get("migration_rows", {})
                    for table, table_rows in migration_rows.items():
                        if table in _SCOPED_TABLES:
                            for row in table_rows:
                                actual.add(table, row)
                    descriptors = []
                    for structure in case.get("structures", []):
                        descriptors.extend(structure.get("representations", []))
                    descriptors.extend(
                        item for item in case.get("evidence", []) if "path" in item
                    )
                    for descriptor in descriptors:
                        path_text = descriptor.get("path")
                        if not path_text:
                            asset_summary["remote_not_materialized"] += 1
                            continue
                        path = root / path_text
                        if not path.is_file():
                            if descriptor.get("source_missing") is True:
                                asset_summary["source_missing"] += 1
                            else:
                                asset_summary["copy_integrity_failures"] += 1
                            continue
                        if path_text not in asset_hash_cache:
                            asset_hash_cache[path_text] = _sha256_file(path)[0]
                        digest = asset_hash_cache[path_text]
                        if digest != descriptor.get("observed_sha256", digest):
                            asset_summary["copy_integrity_failures"] += 1
                        expected = descriptor.get("expected_sha256")
                        if expected is not None and digest != expected:
                            asset_summary["declared_hash_mismatch"] += 1
                        asset_summary["present"] += 1
                if stream.readline():
                    case_mismatches.append(
                        {"case_id": None, "reason": "unexpected_extra_export_case"}
                    )
            actual_inventory = actual.summary()
        finally:
            actual.close()

        inventory_path = root / "evidence" / "migration" / "legacy-identity-relationships.jsonl"
        migration_path = root / "evidence" / "migration" / "identity-migration.jsonl"
        coverage = _RowInventory(root)
        inventory_digest = hashlib.sha256()
        mapping_digest = hashlib.sha256()
        mapping_count = 0
        unresolved = 0
        identity_new_ids_valid = True
        try:
            for value in _identity_inventory_rows(source, connection, release_id):
                inventory_digest.update(
                    (_canonical_json(value) + "\n").encode("utf-8")
                )
                coverage.add("identity_inventory", {"id": value["row_id"]})
            for value in _identity_migration_rows(
                source, connection, release_id, source_rows
            ):
                mapping_digest.update(
                    (_canonical_json(value) + "\n").encode("utf-8")
                )
                coverage.add(
                    "identity_mapped",
                    {"id": value["legacy_inventory_row_id"]},
                )
                mapping_count += 1
                unresolved += value["outcome"] == "unresolved"
                identity_new_ids_valid = (
                    identity_new_ids_valid and _identity_mapping_is_valid(value)
                )
            coverage_summary = coverage.summary()
        finally:
            coverage.close()
        inventory_expected_sha = inventory_digest.hexdigest()
        mapping_expected_sha = mapping_digest.hexdigest()
        identity_file_mismatches = []
        if _sha256_file(inventory_path)[0] != inventory_expected_sha:
            identity_file_mismatches.append("legacy_identity_inventory")
        if _sha256_file(migration_path)[0] != mapping_expected_sha:
            identity_file_mismatches.append("identity_migration")

        identity_coverage_complete = (
            coverage_summary["identity_inventory"]
            == coverage_summary["identity_mapped"]
        )
    finally:
        if source_rows is not None:
            source_rows.close()
        connection.close()

    database_digest_after, _ = _sha256_file(database)
    stable_source_snapshot = database_digest_before == database_digest_after
    control_bindings_valid = (
        manifest.get("control_assets") == _migration_control_descriptors(root)
    )
    mismatches = {
        table: {"source": source_inventory[table], "export": actual_inventory[table]}
        for table in _SCOPED_TABLES
        if source_inventory[table] != actual_inventory[table]
    }
    contiguous = indexes == set(range(case_count))
    passed = (
        not mismatches
        and not case_mismatches
        and not identity_file_mismatches
        and identity_coverage_complete
        and identity_new_ids_valid
        and stable_source_snapshot
        and control_bindings_valid
        and len(case_ids) == case_count
        and contiguous
        and asset_summary["declared_hash_mismatch"] == 0
        and asset_summary["copy_integrity_failures"] == 0
    )
    result = {
        "schema": MIGRATION_SCHEMA,
        "phase": "source-export-parity",
        "dataset": manifest["ds_name"],
        "release": release_id,
        "passed": passed,
        "case_count": case_count,
        "unique_case_ids": len(case_ids),
        "persistent_indices_contiguous": contiguous,
        "stable_source_snapshot": stable_source_snapshot,
        "source_database_sha256": database_digest_after,
        "control_bindings_valid": control_bindings_valid,
        "case_projection_sha256": case_projection_digest.hexdigest(),
        "case_mismatches": case_mismatches,
        "source_inventory": source_inventory,
        "export_inventory": actual_inventory,
        "mismatches": mismatches,
        "asset_parity": asset_summary,
        "identity_migration_rows": mapping_count,
        "identity_indeterminate_rows": unresolved,
        "identity_inventory_coverage_complete": identity_coverage_complete,
        "identity_new_ids_valid": identity_new_ids_valid,
        "identity_file_mismatches": identity_file_mismatches,
    }
    _atomic_json(root / "migration-parity.json", result)
    if not passed:
        failures = list(sorted(mismatches))
        if asset_summary["declared_hash_mismatch"]:
            failures.append("declared asset hash mismatch")
        if asset_summary["copy_integrity_failures"]:
            failures.append("copied asset integrity failure")
        if case_mismatches:
            failures.append("per-Case operative fact mismatch")
        if (
            identity_file_mismatches
            or not identity_coverage_complete
            or not identity_new_ids_valid
        ):
            failures.append("identity migration coverage mismatch")
        if not stable_source_snapshot:
            failures.append("source database changed during parity")
        if not control_bindings_valid:
            failures.append("release control binding mismatch")
        if len(case_ids) != case_count:
            failures.append("duplicate Case identifier")
        if not contiguous:
            failures.append("non-contiguous public index")
        raise MigrationParityError(
            "legacy-to-Bundle parity failed: " + ", ".join(failures)
        )
    return result


def create_rollback_artifact(
    legacy_db: Union[str, Path],
    legacy_asset_root: Union[str, Path],
    rollback_root: Union[str, Path],
) -> Dict[str, Any]:
    """Create a consistent, immutable recovery copy before migration."""

    source = LegacyNormalizedStore(legacy_db)
    root = Path(rollback_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise LegacyMigrationError(f"rollback root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    backup = root / "legacy-store.sqlite3"
    source_connection = source.connect()
    destination = sqlite3.connect(backup)
    try:
        source_connection.backup(destination)
        destination.commit()
        if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise LegacyMigrationError("rollback database failed integrity_check")
    finally:
        destination.close()
        source_connection.close()

    legacy_assets = Path(legacy_asset_root).expanduser().resolve()
    snapshot_assets = root / "source-assets"
    asset_rows = []
    connection = LegacyNormalizedStore(backup).connect()
    try:
        for row in connection.execute("SELECT * FROM assets ORDER BY asset_id"):
            asset = dict(row)
            path = _safe_legacy_asset(legacy_assets, asset["uri"])
            item = {
                "asset_id": asset["asset_id"],
                "legacy_uri": asset["uri"],
                "declared_sha256": asset["content_sha256"],
                "state": "remote" if path is None else "missing",
                "sha256": None,
                "size": None,
                "rollback_path": None,
                "snapshot_path": None,
            }
            if path is not None and path.is_file():
                digest, size = _sha256_file(path)
                suffix = path.suffix.lower() or ".bin"
                relative = Path("assets") / "objects" / digest[:2] / f"{digest}{suffix}"
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(path, target)
                    os.chmod(target, 0o444)
                snapshot = _safe_legacy_asset(snapshot_assets, asset["uri"])
                if snapshot is None:
                    raise LegacyMigrationError("local rollback asset became remote")
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                if snapshot.exists():
                    snapshot_digest, snapshot_size = _sha256_file(snapshot)
                    if snapshot_digest != digest or snapshot_size != size:
                        raise LegacyMigrationError(
                            f"rollback source-path collision: {asset['uri']}"
                        )
                else:
                    try:
                        os.link(target, snapshot)
                    except OSError:
                        shutil.copy2(target, snapshot)
                        os.chmod(snapshot, 0o444)
                item.update(
                    {
                        "state": "present" if asset["content_sha256"] in (None, digest) else "hash_mismatch",
                        "sha256": digest,
                        "size": size,
                        "rollback_path": relative.as_posix(),
                        "snapshot_path": snapshot.relative_to(root).as_posix(),
                    }
                )
            asset_rows.append(item)
    finally:
        connection.close()
    asset_manifest = root / "legacy-assets.jsonl"
    asset_digest, asset_size, asset_count = _write_jsonl(asset_manifest, asset_rows)
    source_digest, source_size = _sha256_file(Path(legacy_db).resolve())
    backup_digest, backup_size = _sha256_file(backup)
    result = {
        "schema": MIGRATION_SCHEMA,
        "phase": "rollback-snapshot",
        "created_at": _utc_now(),
        "source_database": str(Path(legacy_db).resolve()),
        "source_database_sha256": source_digest,
        "source_database_size": source_size,
        "rollback_database": str(backup),
        "rollback_database_sha256": backup_digest,
        "rollback_database_size": backup_size,
        "asset_manifest": str(asset_manifest),
        "asset_manifest_sha256": asset_digest,
        "asset_manifest_size": asset_size,
        "asset_count": asset_count,
        "asset_failures": sum(item["state"] in {"missing", "hash_mismatch"} for item in asset_rows),
        "snapshot_asset_root": str(snapshot_assets),
    }
    _atomic_json(root / "rollback.json", result)
    os.chmod(backup, 0o444)
    os.chmod(asset_manifest, 0o444)
    os.chmod(root / "rollback.json", 0o444)
    return result


def verify_rollback_artifact(rollback_root: Union[str, Path]) -> Dict[str, Any]:
    """Verify the immutable database/asset snapshot used as migration input."""

    root = Path(rollback_root).expanduser().resolve()
    manifest_path = root / "rollback.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report.get("schema") != MIGRATION_SCHEMA or report.get("phase") != "rollback-snapshot":
        raise LegacyMigrationError("rollback manifest uses an unsupported schema")
    database = root / "legacy-store.sqlite3"
    database_digest, database_size = _sha256_file(database)
    database_connection = LegacyNormalizedStore(database).connect()
    database_connection.close()
    database_valid = (
        database_digest == report["rollback_database_sha256"]
        and database_size == report["rollback_database_size"]
    )
    asset_manifest = root / "legacy-assets.jsonl"
    asset_manifest_digest, asset_manifest_size = _sha256_file(asset_manifest)
    manifest_valid = (
        asset_manifest_digest == report["asset_manifest_sha256"]
        and asset_manifest_size == report["asset_manifest_size"]
    )
    asset_count = 0
    missing_negative_facts = 0
    undeclared_losses = 0
    declared_hash_mismatches = 0
    with asset_manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            asset_count += 1
            state = item["state"]
            if state == "missing":
                missing_negative_facts += 1
                if item["rollback_path"] is not None or item["snapshot_path"] is not None:
                    undeclared_losses += 1
                continue
            if state == "remote":
                if item["rollback_path"] is not None or item["snapshot_path"] is not None:
                    undeclared_losses += 1
                continue
            if state == "hash_mismatch":
                declared_hash_mismatches += 1
            if state not in {"present", "hash_mismatch"}:
                undeclared_losses += 1
                continue
            paths = (item["rollback_path"], item["snapshot_path"])
            for path_text in paths:
                if not isinstance(path_text, str):
                    undeclared_losses += 1
                    continue
                path = root / path_text
                if not path.is_file():
                    undeclared_losses += 1
                    continue
                digest, size = _sha256_file(path)
                if digest != item["sha256"] or size != item["size"]:
                    undeclared_losses += 1
    passed = (
        database_valid
        and manifest_valid
        and asset_count == report["asset_count"]
        and undeclared_losses == 0
        and declared_hash_mismatches == 0
    )
    result = {
        "schema": MIGRATION_SCHEMA,
        "phase": "rollback-verification",
        "passed": passed,
        "rollback_database_sha256": database_digest,
        "asset_manifest_sha256": asset_manifest_digest,
        "asset_count": asset_count,
        "missing_negative_facts": missing_negative_facts,
        "undeclared_losses": undeclared_losses,
        "declared_hash_mismatches": declared_hash_mismatches,
        "snapshot_database": str(database),
        "snapshot_asset_root": str(root / "source-assets"),
    }
    if not passed:
        raise MigrationParityError(
            "rollback snapshot failed integrity or declared-hash verification"
        )
    return result


def verify_target_parity(
    target_db: Union[str, Path],
    bundle_root: Union[str, Path],
    source_export_parity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind target Cases, facts, policy, certificate, generation, and assets."""

    from ..public.bundle import load_bundle
    from ..public.certificate import certificate_digest, load_certificate_envelope

    target = Path(target_db).expanduser().resolve()
    root = Path(bundle_root).expanduser().resolve()
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    store = TargetStore(target)
    release = store.load_release_metadata(manifest["ds_name"], manifest["release_id"])
    if release is None or release["receipt"] is None:
        raise MigrationParityError("migrated release has no authoritative target receipt")
    receipt = release["receipt"]
    bundle = load_bundle(root, manifest["ds_name"], db=target)
    bundle._authoritative_receipt()
    generation = bundle._frozen_generation()
    certificate_path = root / "certificate.json"
    certificate = load_certificate_envelope(certificate_path)["certificate"]
    status = bundle._state.status

    descriptor, parity_name = tempfile.mkstemp(
        prefix=".migration-target-parity-", suffix=".sqlite3", dir=root
    )
    os.close(descriptor)
    parity_path = Path(parity_name)
    parity_db = sqlite3.connect(parity_path)
    parity_db.execute(
        "CREATE TABLE source_cases(case_id TEXT PRIMARY KEY,public_index INTEGER NOT NULL,"
        "local_id TEXT NOT NULL,fact_digest TEXT NOT NULL,selected INTEGER NOT NULL)"
    )
    parity_db.execute(
        "CREATE TABLE target_cases(case_id TEXT PRIMARY KEY,public_index INTEGER NOT NULL,"
        "local_id TEXT NOT NULL,fact_digest TEXT NOT NULL)"
    )
    source_case_projection_digest = hashlib.sha256()
    try:
        assessments = generation.iter_cases()
        with (root / "cases.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                case = json.loads(line)
                case_digest = _digest_json(case)
                source_case_projection_digest.update(bytes.fromhex(case_digest))
                try:
                    assessment = next(assessments)
                except StopIteration as error:
                    raise MigrationParityError(
                        "assessment generation has fewer Cases than the source"
                    ) from error
                if assessment["case_id"] != case["id"]:
                    raise MigrationParityError(
                        "assessment generation Case order differs from the source"
                    )
                selected = int(
                    assessment["archive_class"] != "incomplete"
                    or receipt["include_incomplete"]
                )
                parity_db.execute(
                    "INSERT INTO source_cases VALUES(?,?,?,?,?)",
                    (
                        case["id"], case["public_index"], case["local_id"],
                        case_digest, selected,
                    ),
                )
        try:
            next(assessments)
        except StopIteration:
            pass
        else:
            raise MigrationParityError(
                "assessment generation has more Cases than the source"
            )
        target_connection = sqlite3.connect(target)
        try:
            target_rows = target_connection.execute(
                "SELECT case_id,public_index,local_id,source_payload_json FROM cases "
                "WHERE dataset_id=? AND release_id=? ORDER BY public_index,case_id",
                (manifest["ds_name"], manifest["release_id"]),
            )
            for case_id, public_index, local_id, source_payload_json in target_rows:
                parity_db.execute(
                    "INSERT INTO target_cases VALUES(?,?,?,?)",
                    (
                        case_id,
                        public_index,
                        local_id,
                        _digest_json(json.loads(source_payload_json)),
                    ),
                )
        finally:
            target_connection.close()
        parity_db.commit()
        expected_included_count = parity_db.execute(
            "SELECT COUNT(*) FROM source_cases WHERE selected=1"
        ).fetchone()[0]
        expected_excluded_count = parity_db.execute(
            "SELECT COUNT(*) FROM source_cases WHERE selected=0"
        ).fetchone()[0]
        included_digest = _stream_json_array_digest(
            row[0]
            for row in parity_db.execute(
                "SELECT case_id FROM source_cases WHERE selected=1 ORDER BY case_id"
            )
        )
        excluded_digest = _stream_json_array_digest(
            row[0]
            for row in parity_db.execute(
                "SELECT case_id FROM source_cases WHERE selected=0 ORDER BY case_id"
            )
        )
        target_case_digest = _stream_json_array_digest(
            row[0]
            for row in parity_db.execute(
                "SELECT case_id FROM target_cases ORDER BY case_id"
            )
        )
        fact_mismatches = [
            row[0]
            for row in parity_db.execute(
                "SELECT s.case_id FROM source_cases s LEFT JOIN target_cases t USING(case_id) "
                "WHERE s.selected=1 AND (t.case_id IS NULL OR s.public_index!=t.public_index "
                "OR s.local_id!=t.local_id OR s.fact_digest!=t.fact_digest) "
                "UNION SELECT t.case_id FROM target_cases t LEFT JOIN source_cases s USING(case_id) "
                "WHERE s.case_id IS NULL OR s.selected=0 ORDER BY 1 LIMIT 25"
            )
        ]
        source_fact_digest = _stream_json_array_digest(
            [row[0], row[1]]
            for row in parity_db.execute(
                "SELECT case_id,fact_digest FROM source_cases WHERE selected=1 ORDER BY case_id"
            )
        )
        target_fact_digest = _stream_json_array_digest(
            [row[0], row[1]]
            for row in parity_db.execute(
                "SELECT case_id,fact_digest FROM target_cases ORDER BY case_id"
            )
        )
    finally:
        parity_db.close()
        parity_path.unlink(missing_ok=True)
    persisted_policy = store.read_policy(manifest["ds_name"])

    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    try:
        asset_rows = list(
            connection.execute(
                "SELECT DISTINCT t.content_sha256,t.size,t.relative_path FROM target_assets t "
                "JOIN (SELECT content_sha256 FROM scientific_asset_refs WHERE dataset_id=? AND release_id=? "
                "UNION SELECT content_sha256 FROM structure_asset_refs WHERE dataset_id=? AND release_id=? "
                "UNION SELECT content_sha256 FROM release_control_asset_refs WHERE dataset_id=? AND release_id=?) r "
                "USING(content_sha256) ORDER BY t.content_sha256",
                (
                    manifest["ds_name"], manifest["release_id"],
                    manifest["ds_name"], manifest["release_id"],
                    manifest["ds_name"], manifest["release_id"],
                ),
            )
        )
        control_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT category,name,content_sha256 FROM release_control_asset_refs "
                "WHERE dataset_id=? AND release_id=? ORDER BY category,name",
                (manifest["ds_name"], manifest["release_id"]),
            )
        ]
    finally:
        connection.close()
    asset_failures = []
    for row in asset_rows:
        path = store.asset_root / row["relative_path"]
        if not path.is_file():
            asset_failures.append(row["content_sha256"])
            continue
        digest, size = _sha256_file(path)
        if digest != row["content_sha256"] or size != row["size"]:
            asset_failures.append(row["content_sha256"])
    target_asset_digest = asset_inventory_digest(
        {"sha256": row["content_sha256"], "size": row["size"]}
        for row in asset_rows
    )
    expected_migration_controls = sorted([
        {
            "category": item["category"],
            "name": item["name"],
            "content_sha256": item["sha256"],
        }
        for item in _migration_control_descriptors(root)
    ], key=lambda item: (item["category"], item["name"]))
    stored_migration_controls = [
        item for item in control_rows if item["category"] == "migration"
    ]

    expected = int(source_export_parity["case_count"])
    stored = sum(release["counts"].values())
    included = int(receipt["included_count"])
    excluded = int(receipt["excluded_count"])
    bindings_valid = bool(
        source_export_parity["case_projection_sha256"]
        == source_case_projection_digest.hexdigest()
        and certificate["source"]["digest"] == status["source_digest"]
        and certificate["policy"]["digest"] == receipt["policy_digest"]
        and certificate["generation"]["digest"] == receipt["generation_digest"]
        and receipt["certificate_digest"] == certificate_digest(certificate_path)
        and persisted_policy is not None
        and persisted_policy["digest"] == receipt["policy_digest"]
        and receipt["included_cases_digest"] == included_digest
        and receipt["excluded_cases_digest"] == excluded_digest
        and target_case_digest == included_digest
        and receipt["asset_inventory_digest"] == target_asset_digest
    )
    passed = bool(
        source_export_parity["passed"]
        and stored == included
        and included + excluded == expected
        and included == expected_included_count
        and excluded == expected_excluded_count
        and not fact_mismatches
        and not asset_failures
        and expected_migration_controls == stored_migration_controls
        and bindings_valid
    )
    result = {
        "schema": MIGRATION_SCHEMA,
        "phase": "target-parity",
        "dataset": manifest["ds_name"],
        "release": manifest["release_id"],
        "passed": passed,
        "source_case_count": expected,
        "target_case_count": stored,
        "certified_included_case_count": included,
        "certified_excluded_case_count": excluded,
        "included_case_ids_sha256": included_digest,
        "excluded_cases_digest": receipt["excluded_cases_digest"],
        "target_case_ids_sha256": target_case_digest,
        "source_case_facts_sha256": source_fact_digest,
        "target_case_facts_sha256": target_fact_digest,
        "case_fact_mismatches": fact_mismatches,
        "source_digest": certificate["source"]["digest"],
        "policy_digest": receipt["policy_digest"],
        "certificate_digest": receipt["certificate_digest"],
        "generation_digest": receipt["generation_digest"],
        "target_asset_inventory_digest": target_asset_digest,
        "target_asset_failures": asset_failures[:25],
        "migration_control_assets": stored_migration_controls,
        "bindings_valid": bindings_valid,
        "archive_counts": release["counts"],
        "receipt_id": receipt["receipt_id"],
        "target_store_uuid": receipt["store_uuid"],
        "target_database": str(target),
        "classification_changes_are_derived": True,
    }
    _atomic_json(root / "migration-target-parity.json", result)
    if not passed:
        raise MigrationParityError(
            "target release does not exactly bind source Cases, facts, controls, and certified state"
        )
    return result


def publish_cutover_pointer(
    pointer_path: Union[str, Path],
    target_db: Union[str, Path],
    rollback_root: Union[str, Path],
    parity_reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Publish an audited locator after all parity reports pass.

    This JSON is provenance for operators; the runtime does not treat it as an
    activation mechanism.  Use :func:`activate_store_generation` for the
    atomic database-plus-assets switch.
    """

    if not parity_reports or not all(report.get("passed") is True for report in parity_reports):
        raise MigrationParityError("cutover requires passing parity for every migrated release")
    target = Path(target_db).expanduser().resolve()
    metadata = TargetStore(target).metadata()
    rollback = Path(rollback_root).expanduser().resolve()
    rollback_report = rollback / "rollback.json"
    if not rollback_report.is_file():
        raise LegacyMigrationError("cutover requires an explicit rollback artifact")
    value = {
        "schema": CUTOVER_SCHEMA,
        "target_database": str(target),
        "target_assets": str(TargetStore(target).asset_root),
        "target_schema": metadata["schema_id"],
        "target_store_uuid": metadata["store_uuid"],
        "rollback_artifact": str(rollback),
        "rollback_manifest_sha256": _sha256_file(rollback_report)[0],
        "parity_sha256": _digest_json([dict(report) for report in parity_reports]),
        "published_at": _utc_now(),
    }
    _atomic_json(Path(pointer_path).expanduser().resolve(), value)
    return value


def activate_store_generation(
    active_link: Union[str, Path],
    target_db: Union[str, Path],
    rollback_root: Union[str, Path],
    parity_reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Atomically activate one directory containing a database and its assets.

    The configured runtime database must be ``ACTIVE_LINK/<database-name>``.
    Replacing the directory symlink makes the SQLite file and sibling
    ``<database-name>.assets`` tree visible as one generation.
    """

    if not parity_reports or not all(
        report.get("schema") == MIGRATION_SCHEMA
        and report.get("phase") == "target-parity"
        and report.get("passed") is True
        for report in parity_reports
    ):
        raise MigrationParityError("activation requires passing parity for every migrated release")
    target = Path(target_db).expanduser().resolve()
    store = TargetStore(target)
    metadata = store.metadata()
    store.asset_root.mkdir(parents=True, exist_ok=True)
    rollback = Path(rollback_root).expanduser().resolve()
    rollback_manifest = rollback / "rollback.json"
    rollback_verification = verify_rollback_artifact(rollback)
    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise MigrationParityError("target database failed integrity_check before activation")
        receipts = [
            dict(row)
            for row in connection.execute(
                "SELECT dataset_id,release_id,receipt_id FROM ingestion_receipts "
                "ORDER BY dataset_id,release_id"
            )
        ]
        assets = [
            dict(row)
            for row in connection.execute(
                "SELECT content_sha256,size,relative_path FROM target_assets "
                "ORDER BY content_sha256"
            )
        ]
    finally:
        connection.close()
    expected_receipts = sorted(
        (
            report["dataset"],
            report["release"],
            report["receipt_id"],
            report["target_store_uuid"],
            str(Path(report["target_database"]).resolve()),
        )
        for report in parity_reports
    )
    actual_receipts = [
        (
            item["dataset_id"], item["release_id"], item["receipt_id"],
            metadata["store_uuid"], str(target),
        )
        for item in receipts
    ]
    if expected_receipts != actual_receipts:
        raise MigrationParityError(
            "activation parity reports do not exactly cover target receipts"
        )
    target_asset_failures = []
    expected_asset_paths = {
        (store.asset_root / item["relative_path"]).resolve() for item in assets
    }
    for item in assets:
        path = store.asset_root / item["relative_path"]
        if not path.is_file():
            target_asset_failures.append(item["content_sha256"])
            continue
        digest, size = _sha256_file(path)
        if digest != item["content_sha256"] or size != item["size"]:
            target_asset_failures.append(item["content_sha256"])
    if target_asset_failures:
        raise MigrationParityError("target assets failed integrity verification before activation")
    actual_asset_paths = {
        path.resolve()
        for path in (store.asset_root / "objects" / "sha256").rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    if actual_asset_paths != expected_asset_paths:
        raise MigrationParityError(
            "target asset directory differs from the target database inventory"
        )
    target_database_sha256, target_database_size = _sha256_file(target)
    target_assets_sha256 = asset_inventory_digest(
        {"sha256": item["content_sha256"], "size": item["size"]}
        for item in assets
    )
    link = Path(active_link).expanduser().absolute()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() and not link.is_symlink():
        raise LegacyMigrationError("active store path must be absent or a directory symlink")
    generation = target.parent
    if (
        generation == link.parent
        or link.parent.is_relative_to(generation)
        or generation.is_relative_to(link)
    ):
        raise LegacyMigrationError("active link must live outside the target generation")
    previous = None
    if link.is_symlink():
        previous = str((link.parent / os.readlink(link)).resolve())
    activation = {
        "schema": CUTOVER_SCHEMA,
        "activation": "directory-symlink",
        "active_link": str(link),
        "active_database": str(link / target.name),
        "active_assets": str(link / store.asset_root.name),
        "generation": str(generation),
        "target_database": str(target),
        "target_store_uuid": metadata["store_uuid"],
        "target_schema": metadata["schema_id"],
        "previous_generation": previous,
        "rollback_artifact": str(rollback),
        "rollback_manifest_sha256": _sha256_file(rollback_manifest)[0],
        "rollback_verification": rollback_verification,
        "parity_sha256": _digest_json([dict(report) for report in parity_reports]),
        "target_database_sha256": target_database_sha256,
        "target_database_size": target_database_size,
        "target_assets_sha256": target_assets_sha256,
        "target_asset_count": len(assets),
        "activated_at": _utc_now(),
    }
    _atomic_json(generation / "cutover-activation.json", activation)
    temporary = link.with_name(f".{link.name}.{uuid.uuid4().hex}.tmp")
    try:
        relative = os.path.relpath(generation, link.parent)
        os.symlink(relative, temporary, target_is_directory=True)
        os.replace(temporary, link)
        descriptor = os.open(link.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.is_symlink():
            temporary.unlink()
    return activation


def migrate_legacy_store(
    legacy_db: Union[str, Path],
    legacy_asset_root: Union[str, Path],
    work_root: Union[str, Path],
    target_db: Union[str, Path],
    rollback_root: Union[str, Path],
    bootstraps: Mapping[str, DatasetBootstrap],
    *,
    cutover_pointer: Optional[Union[str, Path]] = None,
    active_store_link: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run the complete snapshot/export/assess/certify/ingest/parity workflow."""

    from ..public.bundle import Bundle

    work = Path(work_root).expanduser().resolve()
    target = Path(target_db).expanduser().resolve()
    if target == Path(legacy_db).expanduser().resolve():
        raise LegacyMigrationError("migration target must differ from the legacy store")
    if target.exists():
        raise LegacyMigrationError("migration target must be a new store")
    work.mkdir(parents=True, exist_ok=True)
    rollback = create_rollback_artifact(legacy_db, legacy_asset_root, rollback_root)
    rollback_verification = verify_rollback_artifact(rollback_root)
    snapshot_db = Path(rollback_verification["snapshot_database"])
    snapshot_assets = Path(rollback_verification["snapshot_asset_root"])
    source = LegacyNormalizedStore(snapshot_db)
    connection = source.connect()
    releases = list(source.releases(connection))
    release_counts = {
        release["release_id"]: connection.execute(
            "SELECT COUNT(*) FROM records WHERE release_id=?",
            (release["release_id"],),
        ).fetchone()[0]
        for release in releases
    }
    connection.close()
    releases.sort(
        key=lambda release: (
            release_counts[release["release_id"]],
            release["dataset_id"],
            release["release_serial"],
            release["release_id"],
        )
    )
    release_order = [
        {
            "position": index,
            "dataset": release["dataset_id"],
            "release": release["release_id"],
            "source_case_count": release_counts[release["release_id"]],
        }
        for index, release in enumerate(releases)
    ]
    _atomic_json(
        work / "migration-plan.json",
        {
            "schema": MIGRATION_SCHEMA,
            "phase": "migration-plan",
            "activation": "only_after_all_releases_pass",
            "release_order": release_order,
        },
    )
    dataset_names = {item["dataset_id"] for item in releases}
    if set(bootstraps) != dataset_names:
        raise LegacyMigrationError(
            "owner bootstrap set must exactly match the legacy datasets"
        )

    migrated = []
    target_parity_reports = []
    for release in releases:
        bootstrap = bootstraps[release["dataset_id"]]
        root = work / release["dataset_id"] / release["release_id"]
        export = export_legacy_release(
            snapshot_db,
            snapshot_assets,
            release["release_id"],
            root,
            bootstrap,
        )
        source_parity = verify_bundle_export(
            snapshot_db,
            release["release_id"],
            root,
            snapshot_assets,
        )
        established = target.is_file() and TargetStore(target).read_policy(bootstrap.ds_name) is not None
        bundle = Bundle(
            root,
            bootstrap.ds_name,
            db=target,
            gates=None if established else bootstrap.gates,
        )
        synchronization = bundle.synchronize()
        certificate = bundle.write_certificate()
        receipt = bundle.ingest(
            include_incomplete=bootstrap.include_incomplete,
            reason=(
                "Owner-approved one-time migration preserving incomplete legacy Cases."
                if bootstrap.include_incomplete
                else None
            ),
        )
        target_parity = verify_target_parity(target, root, source_parity)
        target_parity_reports.append(target_parity)
        migrated.append(
            {
                "dataset": bootstrap.ds_name,
                "release": release["release_id"],
                "bundle_root": str(root),
                "export": export,
                "source_parity": source_parity,
                "synchronization": synchronization,
                "certificate": str(certificate),
                "receipt": receipt,
                "target_parity": target_parity,
            }
        )

    cutover = None
    if cutover_pointer is not None:
        cutover = publish_cutover_pointer(
            cutover_pointer,
            target,
            rollback_root,
            target_parity_reports,
        )
    activation = None
    if active_store_link is not None:
        activation = activate_store_generation(
            active_store_link,
            target,
            rollback_root,
            target_parity_reports,
        )
    result = {
        "schema": MIGRATION_SCHEMA,
        "phase": "complete",
        "passed": all(item["passed"] for item in target_parity_reports),
        "legacy_database": str(Path(legacy_db).resolve()),
        "source_snapshot_database": str(snapshot_db),
        "target_database": str(target),
        "rollback": rollback,
        "rollback_verification": rollback_verification,
        "release_order": release_order,
        "releases": migrated,
        "cutover": cutover,
        "activation": activation,
    }
    _atomic_json(work / "migration-report.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Owner/admin one-time migration from the frozen normalized legacy "
            "store into a new governed Hotpot store."
        )
    )
    parser.add_argument("legacy_db", type=Path)
    parser.add_argument("legacy_asset_root", type=Path)
    parser.add_argument("work_root", type=Path)
    parser.add_argument("target_db", type=Path)
    parser.add_argument("rollback_root", type=Path)
    parser.add_argument(
        "--bootstrap",
        required=True,
        type=Path,
        help=f"owner-approved {BOOTSTRAP_SCHEMA} JSON file",
    )
    parser.add_argument(
        "--active-store-link",
        type=Path,
        help="directory symlink atomically activated after all parity checks pass",
    )
    parser.add_argument(
        "--cutover-report",
        type=Path,
        help="optional operator-facing JSON locator; this does not activate the runtime",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = migrate_legacy_store(
        args.legacy_db,
        args.legacy_asset_root,
        args.work_root,
        args.target_db,
        args.rollback_root,
        load_bootstraps(args.bootstrap),
        cutover_pointer=args.cutover_report,
        active_store_link=args.active_store_link,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "DatasetBootstrap",
    "LegacyMigrationError",
    "LegacySchemaError",
    "MigrationParityError",
    "activate_store_generation",
    "build_parser",
    "create_rollback_artifact",
    "export_legacy_release",
    "migrate_legacy_store",
    "load_bootstraps",
    "main",
    "publish_cutover_pointer",
    "verify_bundle_export",
    "verify_rollback_artifact",
    "verify_target_parity",
]


if __name__ == "__main__":
    raise SystemExit(main())
