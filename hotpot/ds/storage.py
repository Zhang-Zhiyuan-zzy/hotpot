"""Private current-schema storage backend for governed dataset releases.

This module deliberately does not expose the legacy :class:`DatasetStore`.
``TargetStore`` accepts a synchronized, certified snapshot from ``Bundle`` and
owns every byte required by an ingested release.

Snapshot contract
-----------------
``TargetStore.ingest`` accepts a mapping with exactly these logical sections::

    {
        "bundle": {
            "dataset": str,
            "release": str,
            "version": str,
            "release_serial": int | None,
            "title": str,
            "description": str,
            "metadata": JSON object,
        },
        "policy": {
            "revision": positive int,
            "digest": lowercase SHA-256,
            "value": JSON object,
        },
        "digests": {
            "bundle": lowercase SHA-256,
            "certificate": lowercase SHA-256,
            "generation": lowercase SHA-256,
            "case_content": lowercase SHA-256,
            "included_cases": lowercase SHA-256,
            "excluded_cases": lowercase SHA-256,
            "asset_inventory": lowercase SHA-256,
        },
        "cases": iterable,
    }

Each item in ``cases`` is a mapping with ``id``, ``public_index``,
``local_id``, ``facts``, ``assessment``, ``identifiers``, ``structures``, and
``assets``.
``facts`` is the canonical source payload. ``assessment`` contains
``archive_class`` (``ordinary``, ``benchmark``, or ``incomplete``), optional
``quality_level`` and ``evidence_level``, and optional ``benchmark``.
``identifiers`` is an iterable of ``{"namespace": str, "value": str}``.
``structures`` contains the active, valid molecular representations projected
by the read API. ``assets`` contains every positively materialized scientific
asset, including evidence, withdrawn history, and observed bytes that fail an
expected hash::

    {
        "collection": "structures" | "evidence",
        "declaration_id": str,
        "asset_index": int,
        "role": str,
        "status": str,
        "source_path": str | Path,
        "sha256": lowercase SHA-256,
        "size": non-negative int,
    }

Negative and remote asset declarations remain in ``facts`` and do not create
target objects. Case-set digests are SHA-256 digests of a canonical JSON array
of sorted Case IDs. The asset inventory digest uses a canonical JSON array of
unique ``{"sha256": ..., "size": ...}`` objects sorted by SHA-256. Helpers
``case_set_digest`` and ``asset_inventory_digest`` implement those algorithms.

Cases are normalized once into a relational temporary SQLite spool while
unique assets are copied and verified. ``case_content`` binds the persisted
canonical facts, assessment, identity, and identifiers in their supplied
order; the separately bound asset inventory covers materialized bytes. The
target transaction publishes the verified spool with set-based SQL, so neither
the Case collection nor asset bytes are fully materialized in memory and Case
JSON is not reparsed by Python during commit.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
import uuid
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union


TARGET_SCHEMA_ID = "hotpot-ds-store/compact-current"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARCHIVE_CLASSES = frozenset({"ordinary", "benchmark", "incomplete"})


SCHEMA = """
CREATE TABLE IF NOT EXISTS store_metadata(
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 store_uuid TEXT NOT NULL UNIQUE,
 schema_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets(
 dataset_id TEXT PRIMARY KEY,
 title TEXT NOT NULL,
 description TEXT NOT NULL,
 metadata_json TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS dataset_policies(
 dataset_id TEXT PRIMARY KEY REFERENCES datasets(dataset_id),
 revision INTEGER NOT NULL CHECK(revision>0),
 policy_json TEXT NOT NULL,
 policy_digest TEXT NOT NULL,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS releases(
 dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
 release_id TEXT NOT NULL,
 version TEXT NOT NULL,
 release_serial INTEGER NOT NULL CHECK(release_serial>0),
 metadata_json TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(dataset_id,release_id),
 UNIQUE(dataset_id,version),
 UNIQUE(dataset_id,release_serial)
);
CREATE TABLE IF NOT EXISTS cases(
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 case_id TEXT NOT NULL,
 public_index INTEGER NOT NULL CHECK(public_index>=0),
 local_id TEXT NOT NULL,
 source_payload_blob BLOB NOT NULL,
 source_payload_digest BLOB NOT NULL CHECK(length(source_payload_digest)=32),
 PRIMARY KEY(dataset_id,release_id,case_id),
 UNIQUE(dataset_id,release_id,public_index),
 UNIQUE(dataset_id,release_id,local_id),
 FOREIGN KEY(dataset_id,release_id) REFERENCES releases(dataset_id,release_id)
);
CREATE TABLE IF NOT EXISTS case_assessments(
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 case_id TEXT NOT NULL,
 quality_level TEXT,
 evidence_level TEXT,
 archive_class TEXT NOT NULL CHECK(archive_class IN ('ordinary','benchmark','incomplete')),
 benchmark INTEGER NOT NULL CHECK(benchmark IN (0,1)),
 benchmark_status TEXT,
 reasons_json TEXT NOT NULL CHECK(json_valid(reasons_json)),
 PRIMARY KEY(dataset_id,release_id,case_id),
 FOREIGN KEY(dataset_id,release_id,case_id)
   REFERENCES cases(dataset_id,release_id,case_id)
);
CREATE TABLE IF NOT EXISTS identifiers(
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 case_id TEXT NOT NULL,
 namespace TEXT NOT NULL,
 value TEXT NOT NULL,
 PRIMARY KEY(dataset_id,release_id,case_id,namespace,value),
 FOREIGN KEY(dataset_id,release_id,case_id)
   REFERENCES cases(dataset_id,release_id,case_id)
);
CREATE TABLE IF NOT EXISTS target_assets(
 content_sha256 TEXT PRIMARY KEY,
 size INTEGER NOT NULL CHECK(size>=0),
 relative_path TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS structure_asset_refs(
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 case_id TEXT NOT NULL,
 structure_id TEXT NOT NULL,
 format TEXT NOT NULL,
 role TEXT NOT NULL,
 is_primary INTEGER NOT NULL CHECK(is_primary IN (0,1)),
 content_sha256 TEXT NOT NULL REFERENCES target_assets(content_sha256),
 PRIMARY KEY(
   dataset_id,release_id,case_id,structure_id,format,role,content_sha256
 ),
 FOREIGN KEY(dataset_id,release_id,case_id)
   REFERENCES cases(dataset_id,release_id,case_id)
);
CREATE TABLE IF NOT EXISTS scientific_asset_refs(
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 case_id TEXT NOT NULL,
 collection TEXT NOT NULL CHECK(collection IN ('structures','evidence')),
 declaration_id TEXT NOT NULL,
 asset_index INTEGER NOT NULL CHECK(asset_index>=0),
 role TEXT NOT NULL,
 status TEXT NOT NULL,
 content_sha256 TEXT NOT NULL REFERENCES target_assets(content_sha256),
 PRIMARY KEY(
   dataset_id,release_id,case_id,collection,declaration_id,asset_index
 ),
 FOREIGN KEY(dataset_id,release_id,case_id)
   REFERENCES cases(dataset_id,release_id,case_id)
);
CREATE TABLE IF NOT EXISTS release_control_asset_refs(
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 category TEXT NOT NULL,
 name TEXT NOT NULL,
 content_sha256 TEXT NOT NULL REFERENCES target_assets(content_sha256),
 PRIMARY KEY(dataset_id,release_id,category,name),
 FOREIGN KEY(dataset_id,release_id) REFERENCES releases(dataset_id,release_id)
);
CREATE TABLE IF NOT EXISTS ingestion_receipts(
 receipt_id TEXT PRIMARY KEY,
 store_uuid TEXT NOT NULL,
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 bundle_digest TEXT NOT NULL,
 certificate_digest TEXT NOT NULL,
 policy_digest TEXT NOT NULL,
 generation_digest TEXT NOT NULL,
 case_content_digest TEXT NOT NULL,
 included_cases_digest TEXT NOT NULL,
 excluded_cases_digest TEXT NOT NULL,
 asset_inventory_digest TEXT NOT NULL,
 include_incomplete INTEGER NOT NULL CHECK(include_incomplete IN (0,1)),
 reason TEXT,
 included_count INTEGER NOT NULL,
 excluded_count INTEGER NOT NULL,
 binding_json TEXT NOT NULL,
 committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(store_uuid,dataset_id,release_id),
 FOREIGN KEY(dataset_id,release_id) REFERENCES releases(dataset_id,release_id)
);
CREATE INDEX IF NOT EXISTS ix_releases_dataset ON releases(dataset_id);
CREATE INDEX IF NOT EXISTS ix_cases_release ON cases(dataset_id,release_id,public_index);
CREATE INDEX IF NOT EXISTS ix_assessments_levels
 ON case_assessments(
   dataset_id,release_id,archive_class,quality_level,evidence_level
 );
CREATE INDEX IF NOT EXISTS ix_identifiers_lookup
 ON identifiers(dataset_id,namespace,value);
CREATE INDEX IF NOT EXISTS ix_structure_refs_asset
 ON structure_asset_refs(content_sha256);
CREATE INDEX IF NOT EXISTS ix_scientific_refs_asset
 ON scientific_asset_refs(content_sha256);
CREATE INDEX IF NOT EXISTS ix_control_refs_asset
 ON release_control_asset_refs(content_sha256);
"""


class TargetStoreError(RuntimeError):
    """Base error for the current target store."""


class TargetSchemaError(TargetStoreError):
    """The target is not an empty or current-schema Hotpot store."""


class SnapshotError(TargetStoreError, ValueError):
    """The certified ingestion snapshot is internally inconsistent."""


class PolicyMismatchError(TargetStoreError):
    """A release attempted to change an established dataset policy."""


class ReceiptConflictError(TargetStoreError):
    """A release identity is already bound to different certified content."""


class AssetIntegrityError(TargetStoreError):
    """A source or target-owned asset does not match its declaration."""


@dataclass(frozen=True)
class TargetIngestResult:
    """Stable projection of the authoritative target-store receipt."""

    receipt_id: str
    store_uuid: str
    dataset: str
    release: str
    release_serial: int
    included_count: int
    excluded_count: int
    committed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


_RECONSTRUCTED_FACT_FIELDS = frozenset(
    {"id", "public_index", "local_id", "identifiers"}
)
_DERIVED_MIGRATION_METADATA = frozenset(
    {
        "migration_rows",
        "migration_schema",
        "quarantined_entity_relations",
        "shared_identifier_relations",
    }
)
_FINAL_ASSESSMENT_FIELDS = (
    "quality_level",
    "evidence_level",
    "archive_class",
    "benchmark",
    "benchmark_status",
    "reasons",
)


def _compact_case_facts(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the permanent scientific-fact projection for one Case."""

    facts = {
        key: item
        for key, item in value.items()
        if key not in _RECONSTRUCTED_FACT_FIELDS
    }
    metadata = facts.get("metadata")
    if isinstance(metadata, Mapping):
        retained = {
            key: item
            for key, item in metadata.items()
            if key not in _DERIVED_MIGRATION_METADATA
        }
        facts["metadata"] = retained
    return facts


def _compact_assessment(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return final labels and concise decision reasons, never Gate traces."""

    return {
        key: value[key]
        for key in _FINAL_ASSESSMENT_FIELDS
        if key in value
    }


def _encode_facts(value: Mapping[str, Any]) -> tuple[bytes, bytes]:
    canonical = _canonical_json(value).encode("utf-8")
    return zlib.compress(canonical, level=6), hashlib.sha256(canonical).digest()


def _decode_facts(payload: bytes, expected_digest: bytes) -> Dict[str, Any]:
    try:
        canonical = zlib.decompress(payload)
        value = json.loads(canonical)
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetStoreError("stored Case facts are corrupt") from exc
    if hashlib.sha256(canonical).digest() != expected_digest:
        raise TargetStoreError("stored Case fact digest does not match")
    if not isinstance(value, dict) or _canonical_json(value).encode("utf-8") != canonical:
        raise TargetStoreError("stored Case facts are not canonical JSON")
    return value


def _restored_facts(
    facts: Mapping[str, Any],
    *,
    case_id: str,
    public_index: int,
    local_id: str,
    identifiers: Iterable[Mapping[str, str]],
) -> Dict[str, Any]:
    value = dict(facts)
    value.update(
        {
            "id": case_id,
            "public_index": public_index,
            "local_id": local_id,
        }
    )
    grouped: Dict[str, list[str]] = {}
    for item in identifiers:
        grouped.setdefault(str(item["namespace"]), []).append(str(item["value"]))
    value["identifiers"] = {
        namespace: values[0] if len(values) == 1 else values
        for namespace, values in grouped.items()
    }
    return value


def _assessment_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    value = {
        "quality_level": row["quality_level"],
        "evidence_level": row["evidence_level"],
        "archive_class": row["archive_class"],
        "benchmark": bool(row["benchmark"]),
    }
    if row["benchmark_status"] is not None:
        value["benchmark_status"] = row["benchmark_status"]
    reasons = json.loads(row["reasons_json"])
    if reasons:
        value["reasons"] = reasons
    return value


def case_set_digest(case_ids: Iterable[str]) -> str:
    """Return the canonical ordered-set digest used by ingestion receipts."""
    return _json_digest(sorted(set(case_ids)))


def asset_inventory_digest(assets: Iterable[Mapping[str, Any]]) -> str:
    """Return the canonical digest of unique target-owned asset objects."""
    unique = {(str(item["sha256"]), int(item["size"])) for item in assets}
    inventory = [
        {"sha256": digest, "size": size}
        for digest, size in sorted(unique)
    ]
    return _json_digest(inventory)


def case_content_digest(cases: Iterable[Mapping[str, Any]]) -> str:
    """Bind canonical facts and assessments in the supplied Case order."""

    return TargetStore._stream_array_digest(
        TargetStore._case_content_projection(TargetStore._normalize_case(case))
        for case in cases
    )


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SnapshotError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> Tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AssetIntegrityError(
            f"asset cannot be opened as a regular file: {path}"
        ) from exc

    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AssetIntegrityError(f"asset is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise AssetIntegrityError(f"asset changed while it was hashed: {path}") from exc
    if not stat.S_ISREG(current.st_mode) or any(
        getattr(before, field) != getattr(after, field)
        or getattr(after, field) != getattr(current, field)
        for field in stable_fields
    ):
        raise AssetIntegrityError(f"asset changed while it was hashed: {path}")
    return digest.hexdigest(), size


class TargetStore:
    """Current-schema SQLite target with target-owned content-addressed assets."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).expanduser().resolve()
        self.asset_root = Path(str(self.path) + ".assets")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _tables(connection: sqlite3.Connection) -> set:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }

    @staticmethod
    def _metadata(connection: sqlite3.Connection) -> Dict[str, str]:
        row = connection.execute(
            "SELECT store_uuid,schema_id FROM store_metadata WHERE singleton=1"
        ).fetchone()
        if row is None or row["schema_id"] != TARGET_SCHEMA_ID:
            raise TargetSchemaError("target store does not use the current Hotpot schema")
        receipt_columns = {
            item[1]
            for item in connection.execute("PRAGMA table_info(ingestion_receipts)")
        }
        if "case_content_digest" not in receipt_columns:
            raise TargetSchemaError(
                "target store layout is not current; rebuild it with this checkout"
            )
        case_columns = {
            item[1] for item in connection.execute("PRAGMA table_info(cases)")
        }
        assessment_columns = {
            item[1]
            for item in connection.execute("PRAGMA table_info(case_assessments)")
        }
        if (
            {"source_payload_blob", "source_payload_digest"} - case_columns
            or {"benchmark_status", "reasons_json"} - assessment_columns
            or "source_payload_json" in case_columns
            or "assessment_json" in assessment_columns
        ):
            raise TargetSchemaError(
                "target store layout is not compact-current; rebuild it"
            )
        return {"store_uuid": row["store_uuid"], "schema_id": row["schema_id"]}

    def initialize(self) -> Dict[str, str]:
        """Create an empty current store or verify an existing current store."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            tables = self._tables(connection)
            if tables and "store_metadata" not in tables:
                raise TargetSchemaError(
                    "target contains a different schema; rebuild it for the current version"
                )
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    connection.execute(statement)
            row = connection.execute(
                "SELECT store_uuid,schema_id FROM store_metadata WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO store_metadata(singleton,store_uuid,schema_id) VALUES(1,?,?)",
                    (str(uuid.uuid4()), TARGET_SCHEMA_ID),
                )
            elif row["schema_id"] != TARGET_SCHEMA_ID:
                raise TargetSchemaError("target store schema identifier is not current")
            metadata = self._metadata(connection)
            connection.commit()
            return metadata
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def metadata(self) -> Dict[str, str]:
        with self._connect() as connection:
            return self._metadata(connection)

    def read_policy(self, dataset: str) -> Optional[Dict[str, Any]]:
        """Return the immutable persisted policy for ``dataset`` if present."""
        with self._connect() as connection:
            self._metadata(connection)
            row = connection.execute(
                "SELECT revision,policy_json,policy_digest FROM dataset_policies "
                "WHERE dataset_id=?",
                (dataset,),
            ).fetchone()
        if row is None:
            return None
        return self._policy_mapping(row)

    @staticmethod
    def _policy_mapping(row: sqlite3.Row) -> Dict[str, Any]:
        value = json.loads(row["policy_json"])
        digest = row["policy_digest"]
        if _json_digest(value) != digest:
            raise TargetStoreError("persisted dataset policy digest does not match")
        from .public._implementation import (
            PolicyProtocolError,
            validate_frozen_runtime_policy,
        )

        try:
            validate_frozen_runtime_policy(value["runtime_policy"])
        except (KeyError, PolicyProtocolError) as exc:
            raise TargetStoreError("persisted dataset runtime policy is invalid") from exc
        return {
            "revision": row["revision"],
            "digest": digest,
            "value": value,
        }

    def inspect_dataset(self, dataset: str) -> Optional[Dict[str, Any]]:
        """Return dataset, policy, release, and archive-count metadata."""
        with self._connect() as connection:
            self._metadata(connection)
            dataset_row = connection.execute(
                "SELECT dataset_id,title,description,metadata_json,created_at "
                "FROM datasets WHERE dataset_id=?",
                (dataset,),
            ).fetchone()
            if dataset_row is None:
                return None
            policy_row = connection.execute(
                "SELECT revision,policy_json,policy_digest FROM dataset_policies "
                "WHERE dataset_id=?",
                (dataset,),
            ).fetchone()
            release_rows = connection.execute(
                "SELECT release_id,version,release_serial,metadata_json,created_at "
                "FROM releases WHERE dataset_id=? ORDER BY release_serial",
                (dataset,),
            ).fetchall()
            releases = []
            for release in release_rows:
                counts = self._archive_counts(connection, dataset, release["release_id"])
                releases.append(
                    {
                        "release": release["release_id"],
                        "version": release["version"],
                        "release_serial": release["release_serial"],
                        "metadata": json.loads(release["metadata_json"]),
                        "created_at": release["created_at"],
                        "counts": counts,
                    }
                )
        return {
            "dataset": dataset_row["dataset_id"],
            "title": dataset_row["title"],
            "description": dataset_row["description"],
            "metadata": json.loads(dataset_row["metadata_json"]),
            "created_at": dataset_row["created_at"],
            "policy": self._policy_mapping(policy_row),
            "releases": releases,
        }

    @staticmethod
    def _archive_counts(
        connection: sqlite3.Connection, dataset: str, release: str
    ) -> Dict[str, int]:
        counts = {item: 0 for item in _ARCHIVE_CLASSES}
        for row in connection.execute(
            "SELECT archive_class,COUNT(*) AS count FROM case_assessments "
            "WHERE dataset_id=? AND release_id=? GROUP BY archive_class",
            (dataset, release),
        ):
            counts[row["archive_class"]] = row["count"]
        return counts

    def load_release_metadata(
        self, dataset: str, release: str
    ) -> Optional[Dict[str, Any]]:
        """Load immutable release metadata and its authoritative receipt."""
        with self._connect() as connection:
            metadata = self._metadata(connection)
            row = connection.execute(
                "SELECT release_id,version,release_serial,metadata_json,created_at "
                "FROM releases WHERE dataset_id=? AND release_id=?",
                (dataset, release),
            ).fetchone()
            if row is None:
                return None
            receipt = connection.execute(
                "SELECT * FROM ingestion_receipts WHERE store_uuid=? "
                "AND dataset_id=? AND release_id=?",
                (metadata["store_uuid"], dataset, release),
            ).fetchone()
            counts = self._archive_counts(connection, dataset, release)
        return {
            "dataset": dataset,
            "release": row["release_id"],
            "version": row["version"],
            "release_serial": row["release_serial"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "counts": counts,
            "receipt": self._receipt_mapping(receipt) if receipt is not None else None,
        }

    @staticmethod
    def _receipt_mapping(receipt: sqlite3.Row) -> Dict[str, Any]:
        binding = json.loads(receipt["binding_json"])
        from .public._implementation import (
            PolicyProtocolError,
            validate_frozen_runtime_policy,
        )

        try:
            validate_frozen_runtime_policy(binding["runtime_policy"])
        except (KeyError, PolicyProtocolError) as exc:
            raise TargetStoreError("ingestion receipt runtime policy is invalid") from exc
        return {
            "receipt_id": receipt["receipt_id"],
            "store_uuid": receipt["store_uuid"],
            "dataset": receipt["dataset_id"],
            "release": receipt["release_id"],
            "bundle_digest": receipt["bundle_digest"],
            "certificate_digest": receipt["certificate_digest"],
            "policy_digest": receipt["policy_digest"],
            "generation_digest": receipt["generation_digest"],
            "case_content_digest": receipt["case_content_digest"],
            "included_cases_digest": receipt["included_cases_digest"],
            "excluded_cases_digest": receipt["excluded_cases_digest"],
            "asset_inventory_digest": receipt["asset_inventory_digest"],
            "include_incomplete": bool(receipt["include_incomplete"]),
            "reason": receipt["reason"],
            "included_count": receipt["included_count"],
            "excluded_count": receipt["excluded_count"],
            "committed_at": receipt["committed_at"],
            "binding": binding,
        }

    def iter_cases(
        self,
        dataset: str,
        release: str,
        *,
        include_incomplete: bool = False,
        archive_class: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate stored Cases; default reads exclude the incomplete partition."""
        if archive_class is not None and archive_class not in _ARCHIVE_CLASSES:
            raise ValueError(f"unknown archive class: {archive_class}")
        connection = self._connect()
        try:
            self._metadata(connection)
            clauses = ["c.dataset_id=?", "c.release_id=?"]
            parameters: list = [dataset, release]
            if archive_class is not None:
                clauses.append("a.archive_class=?")
                parameters.append(archive_class)
            elif not include_incomplete:
                clauses.append("a.archive_class!='incomplete'")
            rows = connection.execute(
                "SELECT c.case_id,c.public_index,c.local_id,c.source_payload_blob,"
                "c.source_payload_digest,"
                "a.quality_level,a.evidence_level,a.archive_class,a.benchmark,"
                "a.benchmark_status,a.reasons_json FROM cases c JOIN case_assessments a "
                "USING(dataset_id,release_id,case_id) WHERE "
                + " AND ".join(clauses)
                + " ORDER BY c.public_index",
                parameters,
            )
            for row in rows:
                identifiers = [
                    {"namespace": item["namespace"], "value": item["value"]}
                    for item in connection.execute(
                        "SELECT namespace,value FROM identifiers WHERE dataset_id=? "
                        "AND release_id=? AND case_id=? ORDER BY namespace,value",
                        (dataset, release, row["case_id"]),
                    )
                ]
                structures = []
                for item in connection.execute(
                    "SELECT r.structure_id,r.format,r.role,r.is_primary,r.content_sha256,"
                    "t.size,t.relative_path FROM structure_asset_refs r "
                    "JOIN target_assets t USING(content_sha256) WHERE r.dataset_id=? "
                    "AND r.release_id=? AND r.case_id=? "
                    "ORDER BY r.structure_id,r.format,r.role,r.content_sha256",
                    (dataset, release, row["case_id"]),
                ):
                    structures.append(
                        {
                            "structure_id": item["structure_id"],
                            "format": item["format"],
                            "role": item["role"],
                            "primary": bool(item["is_primary"]),
                            "sha256": item["content_sha256"],
                            "size": item["size"],
                            "path": self.asset_root / item["relative_path"],
                        }
                    )
                yield {
                    "id": row["case_id"],
                    "public_index": row["public_index"],
                    "local_id": row["local_id"],
                    "facts": _restored_facts(
                        _decode_facts(
                            row["source_payload_blob"],
                            row["source_payload_digest"],
                        ),
                        case_id=row["case_id"],
                        public_index=row["public_index"],
                        local_id=row["local_id"],
                        identifiers=identifiers,
                    ),
                    "assessment": _assessment_from_row(row),
                    "identifiers": identifiers,
                    "structures": structures,
                }
        finally:
            connection.close()

    def iter_case_headers(
        self,
        dataset: str,
        release: str,
        *,
        archive_classes: Optional[Iterable[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream lightweight Case headers for a public Dataset view."""

        classes = (
            ("ordinary", "benchmark")
            if archive_classes is None
            else tuple(dict.fromkeys(archive_classes))
        )
        if not classes or any(item not in _ARCHIVE_CLASSES for item in classes):
            raise ValueError("archive_classes must contain current archive classes")
        connection = self._connect()
        try:
            self._metadata(connection)
            placeholders = ",".join("?" for _ in classes)
            rows = connection.execute(
                "SELECT c.case_id,c.public_index,c.local_id,a.quality_level,"
                "a.evidence_level,a.archive_class,a.benchmark "
                "FROM cases c JOIN case_assessments a "
                "USING(dataset_id,release_id,case_id) WHERE c.dataset_id=? "
                "AND c.release_id=? AND a.archive_class IN ("
                + placeholders
                + ") ORDER BY c.public_index",
                (dataset, release, *classes),
            )
            for row in rows:
                yield {
                    "id": row["case_id"],
                    "public_index": row["public_index"],
                    "local_id": row["local_id"],
                    "quality_level": row["quality_level"],
                    "evidence_level": row["evidence_level"],
                    "archive_class": row["archive_class"],
                    "benchmark": bool(row["benchmark"]),
                }
        finally:
            connection.close()

    def identifier_namespaces(
        self,
        dataset: str,
        release: str,
        *,
        archive_classes: Iterable[str],
        visible_case_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[str, ...]:
        """Return identifier namespaces visible through one Dataset view."""

        classes = tuple(dict.fromkeys(archive_classes))
        if not classes or any(item not in _ARCHIVE_CLASSES for item in classes):
            raise ValueError("archive_classes must contain current archive classes")
        visible = None if visible_case_ids is None else set(visible_case_ids)
        connection = self._connect()
        try:
            self._metadata(connection)
            placeholders = ",".join("?" for _ in classes)
            if visible is None:
                candidates = connection.execute(
                    "SELECT DISTINCT namespace FROM identifiers "
                    "INDEXED BY ix_identifiers_lookup WHERE dataset_id=? "
                    "AND release_id=? ORDER BY namespace",
                    (dataset, release),
                )
                names = []
                for candidate in candidates:
                    found = connection.execute(
                        "SELECT 1 FROM identifiers i "
                        "INDEXED BY ix_identifiers_lookup JOIN case_assessments a "
                        "USING(dataset_id,release_id,case_id) WHERE i.dataset_id=? "
                        "AND i.release_id=? AND i.namespace=? "
                        "AND a.archive_class IN ("
                        + placeholders
                        + ") LIMIT 1",
                        (dataset, release, candidate["namespace"], *classes),
                    ).fetchone()
                    if found is not None:
                        names.append(candidate["namespace"])
                return tuple(names)
            rows = connection.execute(
                "SELECT DISTINCT i.case_id,i.namespace FROM identifiers i "
                "JOIN case_assessments a USING(dataset_id,release_id,case_id) "
                "WHERE i.dataset_id=? AND i.release_id=? AND a.archive_class IN ("
                + placeholders
                + ")",
                (dataset, release, *classes),
            )
            return tuple(sorted({
                row["namespace"] for row in rows if row["case_id"] in visible
            }))
        finally:
            connection.close()

    def find_case_ids_by_identifier(
        self,
        dataset: str,
        release: str,
        namespace: str,
        value: str,
        *,
        archive_classes: Iterable[str],
    ) -> Tuple[str, ...]:
        """Find Case IDs through the indexed identifier table."""

        classes = tuple(dict.fromkeys(archive_classes))
        if not classes or any(item not in _ARCHIVE_CLASSES for item in classes):
            raise ValueError("archive_classes must contain current archive classes")
        connection = self._connect()
        try:
            self._metadata(connection)
            placeholders = ",".join("?" for _ in classes)
            rows = connection.execute(
                "SELECT i.case_id FROM identifiers i "
                "INDEXED BY ix_identifiers_lookup JOIN case_assessments a "
                "USING(dataset_id,release_id,case_id) WHERE i.dataset_id=? "
                "AND i.release_id=? AND i.namespace=? AND i.value=? "
                "AND a.archive_class IN ("
                + placeholders
                + ")",
                (dataset, release, namespace, value, *classes),
            )
            return tuple(row["case_id"] for row in rows)
        finally:
            connection.close()

    def get_case(
        self, dataset: str, release: str, case_id: str
    ) -> Dict[str, Any]:
        """Load one stored Case and its target-owned structure references."""

        connection = self._connect()
        try:
            self._metadata(connection)
            row = connection.execute(
                "SELECT c.case_id,c.public_index,c.local_id,c.source_payload_blob,"
                "c.source_payload_digest,a.quality_level,a.evidence_level,"
                "a.archive_class,a.benchmark,a.benchmark_status,a.reasons_json "
                "FROM cases c JOIN case_assessments a "
                "USING(dataset_id,release_id,case_id) WHERE c.dataset_id=? "
                "AND c.release_id=? AND c.case_id=?",
                (dataset, release, case_id),
            ).fetchone()
            if row is None:
                raise KeyError(case_id)
            identifiers = [
                {"namespace": item["namespace"], "value": item["value"]}
                for item in connection.execute(
                    "SELECT namespace,value FROM identifiers WHERE dataset_id=? "
                    "AND release_id=? AND case_id=? ORDER BY namespace,value",
                    (dataset, release, case_id),
                )
            ]
            structures = [
                {
                    "structure_id": item["structure_id"],
                    "format": item["format"],
                    "role": item["role"],
                    "primary": bool(item["is_primary"]),
                    "sha256": item["content_sha256"],
                    "size": item["size"],
                    "path": self.asset_root / item["relative_path"],
                }
                for item in connection.execute(
                    "SELECT r.structure_id,r.format,r.role,r.is_primary,"
                    "r.content_sha256,t.size,t.relative_path "
                    "FROM structure_asset_refs r JOIN target_assets t "
                    "USING(content_sha256) WHERE r.dataset_id=? AND r.release_id=? "
                    "AND r.case_id=? ORDER BY r.structure_id,r.format,r.role",
                    (dataset, release, case_id),
                )
            ]
            return {
                "id": row["case_id"],
                "public_index": row["public_index"],
                "local_id": row["local_id"],
                "facts": _restored_facts(
                    _decode_facts(
                        row["source_payload_blob"], row["source_payload_digest"]
                    ),
                    case_id=row["case_id"],
                    public_index=row["public_index"],
                    local_id=row["local_id"],
                    identifiers=identifiers,
                ),
                "assessment": _assessment_from_row(row),
                "identifiers": identifiers,
                "structures": structures,
            }
        finally:
            connection.close()

    def ingest(
        self,
        snapshot: Mapping[str, Any],
        *,
        include_incomplete: bool = False,
        reason: Optional[str] = None,
        _precommit_check: Optional[Callable[[], None]] = None,
    ) -> TargetIngestResult:
        """Ingest one certified release and return its stable receipt projection."""
        if include_incomplete and (not isinstance(reason, str) or not reason.strip()):
            raise SnapshotError("include_incomplete=True requires a non-empty reason")
        reason = reason.strip() if include_incomplete else None
        header, policy, digests = self._snapshot_header(snapshot)
        target_context = self._target_context(snapshot)
        metadata = (
            self.metadata()
            if target_context["store_state"] == "existing"
            else self.initialize()
        )
        self._assert_target_context(metadata, target_context)
        binding = self._receipt_binding(
            metadata["store_uuid"],
            header,
            policy,
            digests,
            target_context,
            include_incomplete,
            reason,
        )

        retry = self._retry_existing_receipt(
            header,
            policy,
            target_context,
            binding,
            _precommit_check,
        )
        if retry is not None:
            return retry

        self.asset_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".hotpot-target-ingest-", dir=str(self.path.parent)
        ) as temporary:
            spool_path = Path(temporary) / "cases.sqlite3"
            counts = self._spool_cases(
                spool_path,
                snapshot["cases"],
                control_assets=snapshot.get("control_assets", ()),
                include_incomplete=include_incomplete,
            )
            self._verify_spool_digests(spool_path, digests, include_incomplete)
            return self._commit_spool(
                spool_path,
                header,
                policy,
                digests,
                target_context,
                binding,
                counts,
                _precommit_check,
            )

    def _target_context(self, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        context = dict(snapshot["target_context"])
        expected = {
            "canonical_path",
            "store_state",
            "store_uuid",
            "schema_id",
            "dataset_state",
            "dataset_policy_revision",
            "dataset_policy_digest",
        }
        if set(context) != expected:
            raise SnapshotError("target_context fields do not match the current contract")
        if Path(context["canonical_path"]).resolve() != self.path:
            raise SnapshotError("target_context canonical path does not match TargetStore")
        if context["schema_id"] != TARGET_SCHEMA_ID:
            raise SnapshotError("target_context schema is not current")
        if context["store_state"] not in {"absent", "existing"}:
            raise SnapshotError("target_context store_state is invalid")
        if context["dataset_state"] not in {"absent", "present"}:
            raise SnapshotError("target_context dataset_state is invalid")
        if context["store_state"] == "absent" and context["store_uuid"] is not None:
            raise SnapshotError("an absent target store cannot declare a UUID")
        if context["dataset_state"] == "absent" and any(
            context[field] is not None
            for field in ("dataset_policy_revision", "dataset_policy_digest")
        ):
            raise SnapshotError("an absent target dataset cannot declare a policy")
        if context["dataset_state"] == "present":
            if not isinstance(context["dataset_policy_revision"], int):
                raise SnapshotError("present target dataset requires a policy revision")
            _require_digest(
                context["dataset_policy_digest"],
                "target_context.dataset_policy_digest",
            )
        return context

    @staticmethod
    def _assert_target_context(
        metadata: Mapping[str, str], context: Mapping[str, Any]
    ) -> None:
        expected_uuid = context["store_uuid"]
        if expected_uuid is not None and metadata["store_uuid"] != expected_uuid:
            raise ReceiptConflictError("certified target store UUID does not match")

    @staticmethod
    def _snapshot_header(
        snapshot: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
        bundle = snapshot["bundle"]
        raw_policy = snapshot["policy"]
        raw_digests = snapshot["digests"]
        requested_serial = bundle.get("release_serial")
        if requested_serial is not None and (
            not isinstance(requested_serial, int) or requested_serial <= 0
        ):
            raise SnapshotError("bundle.release_serial must be a positive integer or null")
        header = {
            "dataset": str(bundle["dataset"]),
            "release": str(bundle["release"]),
            "version": str(bundle["version"]),
            "release_serial": requested_serial,
            "title": str(bundle.get("title", bundle["dataset"])),
            "description": str(bundle.get("description", "")),
            "metadata": dict(bundle.get("metadata", {})),
        }
        policy_value = dict(raw_policy["value"])
        from .public._implementation import (
            PolicyProtocolError,
            require_current_runtime_policy,
        )

        try:
            require_current_runtime_policy(policy_value["runtime_policy"])
        except PolicyProtocolError as exc:
            raise SnapshotError(
                "policy runtime is not current"
            ) from exc
        policy_digest = _require_digest(raw_policy["digest"], "policy.digest")
        if _json_digest(policy_value) != policy_digest:
            raise SnapshotError("policy.digest does not match policy.value")
        revision = raw_policy["revision"]
        if not isinstance(revision, int) or revision <= 0:
            raise SnapshotError("policy.revision must be a positive integer")
        policy = {"revision": revision, "digest": policy_digest, "value": policy_value}
        digests = {
            name: _require_digest(raw_digests[name], f"digests.{name}")
            for name in (
                "bundle",
                "certificate",
                "generation",
                "case_content",
                "included_cases",
                "excluded_cases",
                "asset_inventory",
            )
        }
        _canonical_json(header["metadata"])
        return header, policy, digests

    @staticmethod
    def _receipt_binding(
        store_uuid: str,
        header: Mapping[str, Any],
        policy: Mapping[str, Any],
        digests: Mapping[str, str],
        target_context: Mapping[str, Any],
        include_incomplete: bool,
        reason: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "store_uuid": store_uuid,
            "dataset": header["dataset"],
            "release": header["release"],
            "version": header["version"],
            "requested_release_serial": header["release_serial"],
            "target_context": dict(target_context),
            "policy_revision": policy["revision"],
            "policy_digest": policy["digest"],
            "runtime_policy": dict(policy["value"]["runtime_policy"]),
            "bundle_digest": digests["bundle"],
            "certificate_digest": digests["certificate"],
            "generation_digest": digests["generation"],
            "case_content_digest": digests["case_content"],
            "included_cases_digest": digests["included_cases"],
            "excluded_cases_digest": digests["excluded_cases"],
            "asset_inventory_digest": digests["asset_inventory"],
            "include_incomplete": bool(include_incomplete),
            "reason": reason,
        }

    def _read_receipt(self, dataset: str, release: str) -> Optional[sqlite3.Row]:
        with self._connect() as connection:
            metadata = self._metadata(connection)
            return self._receipt_row(
                connection, metadata["store_uuid"], dataset, release
            )

    @staticmethod
    def _receipt_row(
        connection: sqlite3.Connection,
        store_uuid: str,
        dataset: str,
        release: str,
    ) -> Optional[sqlite3.Row]:
        return connection.execute(
            "SELECT r.*,d.release_serial FROM ingestion_receipts r "
            "JOIN releases d USING(dataset_id,release_id) WHERE r.store_uuid=? "
            "AND r.dataset_id=? AND r.release_id=?",
            (store_uuid, dataset, release),
        ).fetchone()

    def _retry_existing_receipt(
        self,
        header: Mapping[str, Any],
        policy: Mapping[str, Any],
        target_context: Mapping[str, Any],
        binding: Mapping[str, Any],
        precommit_check: Optional[Callable[[], None]],
    ) -> Optional[TargetIngestResult]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            metadata = self._metadata(connection)
            self._assert_target_context(metadata, target_context)
            if metadata["store_uuid"] != binding["store_uuid"]:
                raise ReceiptConflictError(
                    "target store identity changed before receipt retry"
                )
            self._assert_dataset_precondition(
                connection, header["dataset"], target_context, policy
            )
            existing = self._receipt_row(
                connection,
                metadata["store_uuid"],
                header["dataset"],
                header["release"],
            )
            if existing is None:
                connection.rollback()
                return None
            self._assert_receipt(existing, binding)
            if precommit_check is not None:
                precommit_check()
            receipt = self._receipt_row(
                connection,
                metadata["store_uuid"],
                header["dataset"],
                header["release"],
            )
            if receipt is None:
                raise ReceiptConflictError(
                    "dataset release disappeared during receipt retry"
                )
            self._assert_receipt(receipt, binding)
            result = self._result_from_receipt(receipt)
            connection.rollback()
            return result
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _assert_receipt(receipt: sqlite3.Row, binding: Mapping[str, Any]) -> None:
        if receipt["binding_json"] != _canonical_json(binding):
            raise ReceiptConflictError(
                "dataset release is already bound to a different ingestion receipt"
            )

    @staticmethod
    def _result_from_receipt(receipt: sqlite3.Row) -> TargetIngestResult:
        return TargetIngestResult(
            receipt_id=receipt["receipt_id"],
            store_uuid=receipt["store_uuid"],
            dataset=receipt["dataset_id"],
            release=receipt["release_id"],
            release_serial=receipt["release_serial"],
            included_count=receipt["included_count"],
            excluded_count=receipt["excluded_count"],
            committed_at=receipt["committed_at"],
        )

    def _spool_cases(
        self,
        spool_path: Path,
        cases: Iterable[Mapping[str, Any]],
        *,
        control_assets: Iterable[Mapping[str, Any]],
        include_incomplete: bool,
    ) -> Dict[str, int]:
        spool = sqlite3.connect(str(spool_path))
        try:
            spool.executescript(
                """
                CREATE TABLE spool_cases(
                  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                  case_id TEXT NOT NULL UNIQUE,
                  public_index INTEGER NOT NULL UNIQUE,
                  local_id TEXT NOT NULL UNIQUE,
                  quality_level TEXT,
                  evidence_level TEXT,
                  archive_class TEXT NOT NULL,
                  benchmark INTEGER NOT NULL,
                  benchmark_status TEXT,
                  reasons_json TEXT NOT NULL CHECK(json_valid(reasons_json)),
                  facts_blob BLOB NOT NULL,
                  facts_digest BLOB NOT NULL CHECK(length(facts_digest)=32),
                  selected INTEGER NOT NULL,
                  content_json TEXT NOT NULL CHECK(json_valid(content_json))
                );
                CREATE TABLE spool_identifiers(
                  case_id TEXT NOT NULL,
                  identifier_order INTEGER NOT NULL,
                  namespace TEXT NOT NULL,
                  value TEXT NOT NULL,
                  PRIMARY KEY(case_id,identifier_order)
                );
                CREATE TABLE spool_assets(
                  sha256 TEXT PRIMARY KEY,
                  size INTEGER NOT NULL
                );
                CREATE TABLE spool_asset_sources(
                  source_path TEXT PRIMARY KEY,
                  sha256 TEXT NOT NULL,
                  size INTEGER NOT NULL
                );
                CREATE TABLE spool_structures(
                  case_id TEXT NOT NULL,
                  structure_order INTEGER NOT NULL,
                  structure_id TEXT NOT NULL,
                  format TEXT NOT NULL,
                  role TEXT NOT NULL,
                  is_primary INTEGER NOT NULL,
                  sha256 TEXT NOT NULL,
                  PRIMARY KEY(case_id,structure_order)
                );
                CREATE TABLE spool_scientific_asset_refs(
                  case_id TEXT NOT NULL,
                  reference_order INTEGER NOT NULL,
                  collection TEXT NOT NULL,
                  declaration_id TEXT NOT NULL,
                  asset_index INTEGER NOT NULL,
                  role TEXT NOT NULL,
                  status TEXT NOT NULL,
                  sha256 TEXT NOT NULL,
                  PRIMARY KEY(case_id,reference_order)
                );
                CREATE TABLE spool_control_assets(
                  category TEXT NOT NULL,
                  name TEXT NOT NULL,
                  sha256 TEXT NOT NULL,
                  PRIMARY KEY(category,name)
                );
                """
            )
            for item in control_assets:
                asset = self._normalize_control_asset(item)
                self._spool_asset(spool, asset)
                spool.execute(
                    "INSERT INTO spool_control_assets(category,name,sha256) "
                    "VALUES(?,?,?)",
                    (asset["category"], asset["name"], asset["sha256"]),
                )
            included = 0
            excluded = 0
            case_rows = []
            identifier_rows = []
            structure_rows = []
            scientific_rows = []

            def flush_rows() -> None:
                spool.executemany(
                    "INSERT INTO spool_cases(case_id,public_index,local_id,"
                    "quality_level,evidence_level,archive_class,benchmark,"
                    "benchmark_status,reasons_json,facts_blob,facts_digest,selected,"
                    "content_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    case_rows,
                )
                spool.executemany(
                    "INSERT INTO spool_identifiers(case_id,identifier_order,"
                    "namespace,value) VALUES(?,?,?,?)",
                    identifier_rows,
                )
                spool.executemany(
                    "INSERT INTO spool_structures(case_id,structure_order,"
                    "structure_id,format,role,is_primary,sha256) "
                    "VALUES(?,?,?,?,?,?,?)",
                    structure_rows,
                )
                spool.executemany(
                    "INSERT INTO spool_scientific_asset_refs(case_id,reference_order,"
                    "collection,declaration_id,asset_index,role,status,sha256) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    scientific_rows,
                )
                case_rows.clear()
                identifier_rows.clear()
                structure_rows.clear()
                scientific_rows.clear()

            for case in cases:
                normalized = self._normalize_case(case)
                archive_class = normalized["assessment"]["archive_class"]
                selected = archive_class != "incomplete" or include_incomplete
                if selected:
                    included += 1
                    for descriptors in (
                        normalized["structures"],
                        normalized["assets"],
                    ):
                        for asset in descriptors:
                            self._spool_asset(spool, asset)
                else:
                    excluded += 1
                content_json = _canonical_json(
                    self._case_content_projection(normalized)
                )
                facts_blob, facts_digest = _encode_facts(normalized["facts"])
                assessment = normalized["assessment"]
                case_rows.append(
                    (
                        normalized["id"],
                        normalized["public_index"],
                        normalized["local_id"],
                        assessment.get("quality_level"),
                        assessment.get("evidence_level"),
                        archive_class,
                        int(assessment["benchmark"]),
                        assessment.get("benchmark_status"),
                        _canonical_json(assessment.get("reasons", [])),
                        sqlite3.Binary(facts_blob),
                        sqlite3.Binary(facts_digest),
                        int(selected),
                        content_json,
                    )
                )
                if selected:
                    identifier_rows.extend(
                        (
                            normalized["id"],
                            index,
                            identifier["namespace"],
                            identifier["value"],
                        )
                        for index, identifier in enumerate(normalized["identifiers"])
                    )
                    structure_rows.extend(
                        (
                            (
                                normalized["id"],
                                index,
                                structure["structure_id"],
                                structure["format"],
                                structure["role"],
                                int(structure["primary"]),
                                structure["sha256"],
                            )
                            for index, structure in enumerate(normalized["structures"])
                        )
                    )
                    scientific_rows.extend(
                        (
                            (
                                normalized["id"],
                                index,
                                asset["collection"],
                                asset["declaration_id"],
                                asset["asset_index"],
                                asset["role"],
                                asset["status"],
                                asset["sha256"],
                            )
                            for index, asset in enumerate(normalized["assets"])
                        )
                    )
                if len(case_rows) == 1000:
                    flush_rows()
            flush_rows()
            spool.commit()
            return {"included": included, "excluded": excluded}
        except sqlite3.IntegrityError as exc:
            raise SnapshotError(f"duplicate Case identity or index: {exc}") from exc
        finally:
            spool.close()

    def _spool_asset(
        self, spool: sqlite3.Connection, descriptor: Mapping[str, Any]
    ) -> None:
        source_inserted = spool.execute(
            "INSERT OR IGNORE INTO spool_asset_sources(source_path,sha256,size) "
            "VALUES(?,?,?)",
            (
                descriptor["source_path"],
                descriptor["sha256"],
                descriptor["size"],
            ),
        ).rowcount
        if not source_inserted:
            source_claim = spool.execute(
                "SELECT sha256,size FROM spool_asset_sources WHERE source_path=?",
                (descriptor["source_path"],),
            ).fetchone()
            if source_claim != (descriptor["sha256"], descriptor["size"]):
                raise AssetIntegrityError(
                    "one source asset path was declared with conflicting content"
                )

        content_inserted = spool.execute(
            "INSERT OR IGNORE INTO spool_assets(sha256,size) VALUES(?,?)",
            (descriptor["sha256"], descriptor["size"]),
        ).rowcount
        if not content_inserted:
            existing_size = spool.execute(
                "SELECT size FROM spool_assets WHERE sha256=?",
                (descriptor["sha256"],),
            ).fetchone()[0]
            if existing_size != descriptor["size"]:
                raise AssetIntegrityError(
                    "one content digest was declared with different sizes"
                )

        if source_inserted:
            if content_inserted:
                self._store_asset(descriptor)
            else:
                self._verify_asset_source(descriptor)

    @staticmethod
    def _normalize_control_asset(item: Mapping[str, Any]) -> Dict[str, Any]:
        digest = _require_digest(item["sha256"], "control_asset.sha256")
        size = item["size"]
        if not isinstance(size, int) or size < 0:
            raise SnapshotError("control_asset.size must be a non-negative integer")
        return {
            "category": str(item["category"]),
            "name": str(item["name"]),
            "source_path": str(Path(item["source_path"]).expanduser().resolve()),
            "sha256": digest,
            "size": size,
        }

    @staticmethod
    def _normalize_case(case: Mapping[str, Any]) -> Dict[str, Any]:
        assessment = dict(case["assessment"])
        archive_class = assessment["archive_class"]
        if archive_class not in _ARCHIVE_CLASSES:
            raise SnapshotError(f"unknown archive class: {archive_class}")
        benchmark = bool(assessment.get("benchmark", archive_class == "benchmark"))
        if benchmark != (archive_class == "benchmark"):
            raise SnapshotError("benchmark assessment must match archive_class")
        assessment["benchmark"] = benchmark
        identifiers = [
            {"namespace": namespace, "value": value}
            for namespace, value in sorted(
                {
                    (str(item["namespace"]), str(item["value"]))
                    for item in case.get("identifiers", ())
                }
            )
        ]
        structures = []
        for item in case.get("structures", ()):
            digest = _require_digest(item["sha256"], "structure.sha256")
            size = item["size"]
            if not isinstance(size, int) or size < 0:
                raise SnapshotError("structure.size must be a non-negative integer")
            structures.append(
                {
                    "structure_id": str(item["structure_id"]),
                    "format": str(item["format"]),
                    "role": str(item["role"]),
                    "primary": bool(item.get("primary", False)),
                    "source_path": str(Path(item["source_path"]).expanduser().resolve()),
                    "sha256": digest,
                    "size": size,
                }
            )
        assets = []
        for item in case.get("assets", ()):
            digest = _require_digest(item["sha256"], "asset.sha256")
            size = item["size"]
            asset_index = item["asset_index"]
            collection = str(item["collection"])
            if collection not in {"structures", "evidence"}:
                raise SnapshotError("asset.collection must be structures or evidence")
            if not isinstance(asset_index, int) or asset_index < 0:
                raise SnapshotError("asset.asset_index must be a non-negative integer")
            if not isinstance(size, int) or size < 0:
                raise SnapshotError("asset.size must be a non-negative integer")
            assets.append(
                {
                    "collection": collection,
                    "declaration_id": str(item["declaration_id"]),
                    "asset_index": asset_index,
                    "role": str(item.get("role", "asset")),
                    "status": str(item.get("status", "active")),
                    "source_path": str(Path(item["source_path"]).expanduser().resolve()),
                    "sha256": digest,
                    "size": size,
                }
            )
        normalized = {
            "id": str(case["id"]),
            "public_index": int(case["public_index"]),
            "local_id": str(case["local_id"]),
            "facts": _compact_case_facts(dict(case["facts"])),
            "assessment": _compact_assessment(assessment),
            "identifiers": identifiers,
            "structures": structures,
            "assets": assets,
        }
        if normalized["public_index"] < 0:
            raise SnapshotError("Case public_index must be non-negative")
        _canonical_json(normalized)
        return normalized

    @staticmethod
    def _verify_asset_source(descriptor: Mapping[str, Any]) -> None:
        source = Path(descriptor["source_path"])
        expected_digest = descriptor["sha256"]
        expected_size = descriptor["size"]
        if not source.is_file():
            raise AssetIntegrityError(f"materialized source asset is missing: {source}")
        source_digest, source_size = _sha256_file(source)
        if source_digest != expected_digest or source_size != expected_size:
            raise AssetIntegrityError(f"source asset does not match its declaration: {source}")

    def _store_asset(self, descriptor: Mapping[str, Any]) -> None:
        source = Path(descriptor["source_path"])
        expected_digest = descriptor["sha256"]
        expected_size = descriptor["size"]
        if not source.is_file():
            raise AssetIntegrityError(f"materialized source asset is missing: {source}")

        parent = self.asset_root / "objects" / "sha256" / expected_digest[:2]
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / expected_digest
        lock_path = parent / f".{expected_digest}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                target_status = target.lstat()
            except FileNotFoundError:
                target_status = None
            if target_status is not None:
                if not stat.S_ISREG(target_status.st_mode):
                    raise AssetIntegrityError(
                        f"target-owned asset is not a regular file: {target}"
                    )
                # This source path still has to be proved independently even
                # though another ingest already materialized the same digest.
                self._verify_asset_source(descriptor)
                target_digest, target_size = _sha256_file(target)
                if target_digest != expected_digest or target_size != expected_size:
                    raise AssetIntegrityError(
                        f"target-owned asset is corrupt: {target}"
                    )
                return
            temporary = parent / f".{expected_digest}.{uuid.uuid4().hex}.tmp"
            try:
                digest = hashlib.sha256()
                size = 0
                with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
                    source_before = os.fstat(input_stream.fileno())
                    for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                        output_stream.write(block)
                        digest.update(block)
                        size += len(block)
                    source_after = os.fstat(input_stream.fileno())
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
                if digest.hexdigest() != expected_digest or size != expected_size:
                    raise AssetIntegrityError(
                        f"source asset changed while it was copied: {source}"
                    )
                try:
                    source_at_publish = source.stat()
                except OSError as exc:
                    raise AssetIntegrityError(
                        f"source asset changed while it was copied: {source}"
                    ) from exc
                stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                if any(
                    getattr(source_before, field) != getattr(source_after, field)
                    or getattr(source_after, field) != getattr(source_at_publish, field)
                    for field in stable_fields
                ):
                    raise AssetIntegrityError(
                        f"source asset changed while it was copied: {source}"
                    )
                os.replace(str(temporary), str(target))
                directory_fd = os.open(str(parent), os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary.exists():
                    temporary.unlink()

    @staticmethod
    def _spool_case_digest(
        spool: sqlite3.Connection, selected: int
    ) -> str:
        values = (
            row[0]
            for row in spool.execute(
                "SELECT case_id FROM spool_cases WHERE selected=? ORDER BY case_id",
                (selected,),
            )
        )
        return TargetStore._stream_array_digest(values)

    @staticmethod
    def _spool_asset_digest(spool: sqlite3.Connection) -> str:
        values = (
            {"sha256": row[0], "size": row[1]}
            for row in spool.execute(
                "SELECT sha256,size FROM spool_assets ORDER BY sha256"
            )
        )
        return TargetStore._stream_array_digest(values)

    @staticmethod
    def _spool_content_digest(spool: sqlite3.Connection) -> str:
        values = (
            row[0]
            for row in spool.execute(
                "SELECT content_json FROM spool_cases ORDER BY sequence"
            )
        )
        return TargetStore._stream_canonical_array_digest(values)

    @staticmethod
    def _case_content_projection(case: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "id": case["id"],
            "public_index": case["public_index"],
            "local_id": case["local_id"],
            "facts": case["facts"],
            "assessment": case["assessment"],
            "identifiers": case["identifiers"],
        }

    @staticmethod
    def _stream_array_digest(values: Iterable[Any]) -> str:
        return TargetStore._stream_canonical_array_digest(
            _canonical_json(value) for value in values
        )

    @staticmethod
    def _stream_canonical_array_digest(values: Iterable[str]) -> str:
        digest = hashlib.sha256()
        digest.update(b"[")
        first = True
        for value in values:
            if not first:
                digest.update(b",")
            digest.update(value.encode("utf-8"))
            first = False
        digest.update(b"]")
        return digest.hexdigest()

    def _verify_spool_digests(
        self,
        spool_path: Path,
        digests: Mapping[str, str],
        include_incomplete: bool,
    ) -> None:
        with sqlite3.connect(str(spool_path)) as spool:
            included = self._spool_case_digest(spool, 1)
            excluded = self._spool_case_digest(spool, 0)
            assets = self._spool_asset_digest(spool)
            content = self._spool_content_digest(spool)
        if content != digests["case_content"]:
            raise SnapshotError("Case content digest does not match the snapshot")
        if included != digests["included_cases"]:
            raise SnapshotError("included Case-set digest does not match the snapshot")
        if excluded != digests["excluded_cases"]:
            raise SnapshotError("excluded Case-set digest does not match the snapshot")
        if assets != digests["asset_inventory"]:
            raise SnapshotError("target asset inventory digest does not match the snapshot")
        if include_incomplete and excluded != case_set_digest(()):
            raise SnapshotError("forced incomplete archival cannot retain excluded Cases")

    def _commit_spool(
        self,
        spool_path: Path,
        header: Mapping[str, Any],
        policy: Mapping[str, Any],
        digests: Mapping[str, str],
        target_context: Mapping[str, Any],
        binding: Mapping[str, Any],
        counts: Mapping[str, int],
        precommit_check: Optional[Callable[[], None]],
    ) -> TargetIngestResult:
        connection = self._connect()
        try:
            connection.execute("ATTACH DATABASE ? AS ingest_spool", (str(spool_path),))
            connection.execute("BEGIN IMMEDIATE")
            metadata = self._metadata(connection)
            self._assert_target_context(metadata, target_context)
            if metadata["store_uuid"] != binding["store_uuid"]:
                raise ReceiptConflictError("target store identity changed before commit")
            self._assert_dataset_precondition(
                connection, header["dataset"], target_context, policy
            )
            # Bundle supplies a compare-and-swap guard.  It is intentionally
            # evaluated after acquiring the target write transaction and before
            # either an idempotent receipt is accepted or new rows are written.
            # A failure aborts publication while leaving the immutable spool and
            # target-owned CAS objects unreferenced.
            if precommit_check is not None:
                precommit_check()
            receipt = connection.execute(
                "SELECT r.*,d.release_serial FROM ingestion_receipts r "
                "JOIN releases d USING(dataset_id,release_id) WHERE r.store_uuid=? "
                "AND r.dataset_id=? AND r.release_id=?",
                (metadata["store_uuid"], header["dataset"], header["release"]),
            ).fetchone()
            if receipt is not None:
                self._assert_receipt(receipt, binding)
                connection.rollback()
                return self._result_from_receipt(receipt)

            self._establish_dataset(connection, header, policy)
            release_serial = self._insert_release(connection, header)
            self._publish_spool(connection, header)
            receipt_id = "receipt-" + _json_digest(binding)
            connection.execute(
                "INSERT INTO ingestion_receipts("
                "receipt_id,store_uuid,dataset_id,release_id,bundle_digest,"
                "certificate_digest,policy_digest,generation_digest,"
                "case_content_digest,included_cases_digest,excluded_cases_digest,"
                "asset_inventory_digest,include_incomplete,reason,included_count,"
                "excluded_count,binding_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    receipt_id,
                    metadata["store_uuid"],
                    header["dataset"],
                    header["release"],
                    digests["bundle"],
                    digests["certificate"],
                    policy["digest"],
                    digests["generation"],
                    digests["case_content"],
                    digests["included_cases"],
                    digests["excluded_cases"],
                    digests["asset_inventory"],
                    int(binding["include_incomplete"]),
                    binding["reason"],
                    counts["included"],
                    counts["excluded"],
                    _canonical_json(binding),
                ),
            )
            connection.commit()
            stored = self._read_receipt(header["dataset"], header["release"])
            if stored is None:
                raise TargetStoreError("committed ingestion receipt cannot be reloaded")
            if stored["release_serial"] != release_serial:
                raise TargetStoreError("committed release serial changed unexpectedly")
            return self._result_from_receipt(stored)
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise ReceiptConflictError(f"release identity conflicts with target store: {exc}") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _assert_dataset_precondition(
        connection: sqlite3.Connection,
        dataset: str,
        context: Mapping[str, Any],
        policy: Mapping[str, Any],
    ) -> None:
        persisted = connection.execute(
            "SELECT revision,policy_json,policy_digest FROM dataset_policies "
            "WHERE dataset_id=?",
            (dataset,),
        ).fetchone()
        if context["dataset_state"] == "present":
            if persisted is None:
                raise PolicyMismatchError("certified target dataset is missing")
            if (
                persisted["revision"] != context["dataset_policy_revision"]
                or persisted["policy_digest"] != context["dataset_policy_digest"]
            ):
                raise PolicyMismatchError(
                    "certified target dataset policy precondition changed"
                )
        if persisted is not None and (
            persisted["revision"] != policy["revision"]
            or persisted["policy_json"] != _canonical_json(policy["value"])
            or persisted["policy_digest"] != policy["digest"]
        ):
            raise PolicyMismatchError(
                "concurrently established dataset policy differs from the snapshot"
            )

    @staticmethod
    def _establish_dataset(
        connection: sqlite3.Connection,
        header: Mapping[str, Any],
        policy: Mapping[str, Any],
    ) -> None:
        existing = connection.execute(
            "SELECT dataset_id FROM datasets WHERE dataset_id=?",
            (header["dataset"],),
        ).fetchone()
        policy_json = _canonical_json(policy["value"])
        if existing is None:
            connection.execute(
                "INSERT INTO datasets(dataset_id,title,description,metadata_json) "
                "VALUES(?,?,?,?)",
                (
                    header["dataset"],
                    header["title"],
                    header["description"],
                    _canonical_json(header["metadata"]),
                ),
            )
            connection.execute(
                "INSERT INTO dataset_policies(dataset_id,revision,policy_json,policy_digest) "
                "VALUES(?,?,?,?)",
                (
                    header["dataset"],
                    policy["revision"],
                    policy_json,
                    policy["digest"],
                ),
            )
            return
        persisted = connection.execute(
            "SELECT revision,policy_json,policy_digest FROM dataset_policies "
            "WHERE dataset_id=?",
            (header["dataset"],),
        ).fetchone()
        if persisted is None or (
            persisted["revision"] != policy["revision"]
            or persisted["policy_json"] != policy_json
            or persisted["policy_digest"] != policy["digest"]
        ):
            raise PolicyMismatchError(
                "existing dataset policy differs from the certified snapshot"
            )

    @staticmethod
    def _insert_release(
        connection: sqlite3.Connection, header: Mapping[str, Any]
    ) -> int:
        serial = header["release_serial"]
        if serial is None:
            serial = connection.execute(
                "SELECT COALESCE(MAX(release_serial),0)+1 FROM releases WHERE dataset_id=?",
                (header["dataset"],),
            ).fetchone()[0]
        connection.execute(
            "INSERT INTO releases(dataset_id,release_id,version,release_serial,metadata_json) "
            "VALUES(?,?,?,?,?)",
            (
                header["dataset"],
                header["release"],
                header["version"],
                serial,
                _canonical_json(header["metadata"]),
            ),
        )
        return serial

    @staticmethod
    def _publish_spool(
        connection: sqlite3.Connection,
        header: Mapping[str, Any],
    ) -> None:
        dataset = header["dataset"]
        release = header["release"]
        connection.execute(
            "INSERT OR IGNORE INTO target_assets(content_sha256,size,relative_path) "
            "SELECT sha256,size,'objects/sha256/' || substr(sha256,1,2) || '/' || "
            "sha256 FROM ingest_spool.spool_assets"
        )
        conflict = connection.execute(
            "SELECT s.sha256 FROM ingest_spool.spool_assets s "
            "JOIN target_assets t ON t.content_sha256=s.sha256 "
            "WHERE t.size!=s.size OR t.relative_path!=("
            "'objects/sha256/' || substr(s.sha256,1,2) || '/' || s.sha256) LIMIT 1"
        ).fetchone()
        if conflict is not None:
            raise AssetIntegrityError(
                "target asset row conflicts with its content digest"
            )

        connection.execute(
            "INSERT INTO cases(dataset_id,release_id,case_id,public_index,local_id,"
            "source_payload_blob,source_payload_digest) SELECT ?,?,case_id,"
            "public_index,local_id,facts_blob,facts_digest "
            "FROM ingest_spool.spool_cases "
            "WHERE selected=1 ORDER BY sequence",
            (dataset, release),
        )
        connection.execute(
            "INSERT INTO case_assessments(dataset_id,release_id,case_id,quality_level,"
            "evidence_level,archive_class,benchmark,benchmark_status,reasons_json) "
            "SELECT ?,?,case_id,quality_level,evidence_level,archive_class,benchmark,"
            "benchmark_status,reasons_json "
            "FROM ingest_spool.spool_cases WHERE selected=1 ORDER BY sequence",
            (dataset, release),
        )
        connection.execute(
            "INSERT INTO identifiers(dataset_id,release_id,case_id,namespace,value) "
            "SELECT ?,?,i.case_id,i.namespace,i.value "
            "FROM ingest_spool.spool_identifiers i "
            "JOIN ingest_spool.spool_cases c USING(case_id) WHERE c.selected=1 "
            "ORDER BY c.sequence,i.identifier_order",
            (dataset, release),
        )
        connection.execute(
            "INSERT INTO structure_asset_refs(dataset_id,release_id,case_id,"
            "structure_id,format,role,is_primary,content_sha256) "
            "SELECT ?,?,case_id,structure_id,format,role,is_primary,sha256 "
            "FROM ingest_spool.spool_structures ORDER BY case_id,structure_order",
            (dataset, release),
        )
        connection.execute(
            "INSERT INTO scientific_asset_refs(dataset_id,release_id,case_id,"
            "collection,declaration_id,asset_index,role,status,content_sha256) "
            "SELECT ?,?,case_id,collection,declaration_id,asset_index,role,status,"
            "sha256 "
            "FROM ingest_spool.spool_scientific_asset_refs "
            "ORDER BY case_id,reference_order",
            (dataset, release),
        )
        connection.execute(
            "INSERT INTO release_control_asset_refs(dataset_id,release_id,category,"
            "name,content_sha256) SELECT ?,?,category,name,sha256 "
            "FROM ingest_spool.spool_control_assets ORDER BY category,name",
            (dataset, release),
        )


__all__ = [
    "TARGET_SCHEMA_ID",
    "AssetIntegrityError",
    "PolicyMismatchError",
    "ReceiptConflictError",
    "SnapshotError",
    "TargetIngestResult",
    "TargetSchemaError",
    "TargetStore",
    "TargetStoreError",
    "asset_inventory_digest",
    "case_content_digest",
    "case_set_digest",
]
