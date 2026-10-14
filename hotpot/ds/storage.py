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
            "included_cases": lowercase SHA-256,
            "excluded_cases": lowercase SHA-256,
            "asset_inventory": lowercase SHA-256,
        },
        "cases": iterable,
    }

Each item in ``cases`` is a mapping with ``id``, ``public_index``,
``local_id``, ``facts``, ``assessment``, ``identifiers``, and ``structures``.
``facts`` is the canonical source payload. ``assessment`` contains
``archive_class`` (``ordinary``, ``benchmark``, or ``incomplete``), optional
``quality_level`` and ``evidence_level``, and optional ``benchmark``.
``identifiers`` is an iterable of ``{"namespace": str, "value": str}``.
``structures`` is an iterable of positive, materialized asset descriptors::

    {
        "structure_id": str,
        "format": str,
        "role": str,
        "primary": bool,
        "source_path": str | Path,
        "sha256": lowercase SHA-256,
        "size": non-negative int,
    }

Negative and remote asset declarations remain in ``facts`` and do not create
target objects. Case-set digests are SHA-256 digests of a canonical JSON array
of sorted Case IDs. The asset inventory digest uses a canonical JSON array of
unique ``{"sha256": ..., "size": ...}`` objects sorted by SHA-256. Helpers
``case_set_digest`` and ``asset_inventory_digest`` implement those algorithms.

Cases are spooled to a temporary SQLite file while assets are copied and
verified. The target transaction then streams that spool, so neither the Case
collection nor asset bytes are fully materialized in memory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union


TARGET_SCHEMA_ID = "hotpot-ds-store/current"

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
 source_payload_json TEXT NOT NULL,
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
 assessment_json TEXT NOT NULL,
 PRIMARY KEY(dataset_id,release_id,case_id),
 FOREIGN KEY(dataset_id,release_id,case_id)
   REFERENCES cases(dataset_id,release_id,case_id)
);
CREATE TABLE IF NOT EXISTS ordinary_case_index(
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 case_id TEXT NOT NULL,
 PRIMARY KEY(dataset_id,release_id,case_id),
 FOREIGN KEY(dataset_id,release_id,case_id)
   REFERENCES cases(dataset_id,release_id,case_id)
);
CREATE TABLE IF NOT EXISTS benchmark_case_index(
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 case_id TEXT NOT NULL,
 PRIMARY KEY(dataset_id,release_id,case_id),
 FOREIGN KEY(dataset_id,release_id,case_id)
   REFERENCES cases(dataset_id,release_id,case_id)
);
CREATE TABLE IF NOT EXISTS incomplete_case_index(
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 case_id TEXT NOT NULL,
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
 UNIQUE(dataset_id,release_id,namespace,value),
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
CREATE TABLE IF NOT EXISTS ingestion_receipts(
 receipt_id TEXT PRIMARY KEY,
 store_uuid TEXT NOT NULL,
 dataset_id TEXT NOT NULL,
 release_id TEXT NOT NULL,
 bundle_digest TEXT NOT NULL,
 certificate_digest TEXT NOT NULL,
 policy_digest TEXT NOT NULL,
 generation_digest TEXT NOT NULL,
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
 ON case_assessments(dataset_id,release_id,quality_level,evidence_level);
CREATE INDEX IF NOT EXISTS ix_identifiers_lookup
 ON identifiers(dataset_id,namespace,value);
CREATE INDEX IF NOT EXISTS ix_structure_refs_asset
 ON structure_asset_refs(content_sha256);
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


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SnapshotError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
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
        return {
            "revision": row["revision"],
            "digest": row["policy_digest"],
            "value": json.loads(row["policy_json"]),
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
            "policy": {
                "revision": policy_row["revision"],
                "digest": policy_row["policy_digest"],
                "value": json.loads(policy_row["policy_json"]),
            },
            "releases": releases,
        }

    @staticmethod
    def _archive_counts(
        connection: sqlite3.Connection, dataset: str, release: str
    ) -> Dict[str, int]:
        counts = {}
        for archive_class in ("ordinary", "benchmark", "incomplete"):
            table = f"{archive_class}_case_index"
            counts[archive_class] = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE dataset_id=? AND release_id=?",
                (dataset, release),
            ).fetchone()[0]
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
        return {
            "receipt_id": receipt["receipt_id"],
            "store_uuid": receipt["store_uuid"],
            "dataset": receipt["dataset_id"],
            "release": receipt["release_id"],
            "bundle_digest": receipt["bundle_digest"],
            "certificate_digest": receipt["certificate_digest"],
            "policy_digest": receipt["policy_digest"],
            "generation_digest": receipt["generation_digest"],
            "included_cases_digest": receipt["included_cases_digest"],
            "excluded_cases_digest": receipt["excluded_cases_digest"],
            "asset_inventory_digest": receipt["asset_inventory_digest"],
            "include_incomplete": bool(receipt["include_incomplete"]),
            "reason": receipt["reason"],
            "included_count": receipt["included_count"],
            "excluded_count": receipt["excluded_count"],
            "committed_at": receipt["committed_at"],
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
                "SELECT c.case_id,c.public_index,c.local_id,c.source_payload_json,"
                "a.quality_level,a.evidence_level,a.archive_class,a.benchmark,"
                "a.assessment_json FROM cases c JOIN case_assessments a "
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
                    "facts": json.loads(row["source_payload_json"]),
                    "assessment": json.loads(row["assessment_json"]),
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
    ) -> TargetIngestResult:
        """Ingest one certified release and return its stable receipt projection."""
        if include_incomplete and (not isinstance(reason, str) or not reason.strip()):
            raise SnapshotError("include_incomplete=True requires a non-empty reason")
        reason = reason.strip() if include_incomplete else None
        header, policy, digests = self._snapshot_header(snapshot)
        metadata = self.initialize()
        binding = self._receipt_binding(
            metadata["store_uuid"], header, policy, digests, include_incomplete, reason
        )

        existing = self._read_receipt(header["dataset"], header["release"])
        if existing is not None:
            self._assert_receipt(existing, binding)
            return self._result_from_receipt(existing)

        self.asset_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".hotpot-target-ingest-", dir=str(self.path.parent)
        ) as temporary:
            spool_path = Path(temporary) / "cases.sqlite3"
            counts = self._spool_cases(
                spool_path,
                snapshot["cases"],
                include_incomplete=include_incomplete,
            )
            self._verify_spool_digests(spool_path, digests, include_incomplete)
            return self._commit_spool(
                spool_path,
                header,
                policy,
                digests,
                binding,
                counts,
            )

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
        include_incomplete: bool,
        reason: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "store_uuid": store_uuid,
            "dataset": header["dataset"],
            "release": header["release"],
            "version": header["version"],
            "requested_release_serial": header["release_serial"],
            "policy_revision": policy["revision"],
            "policy_digest": policy["digest"],
            "bundle_digest": digests["bundle"],
            "certificate_digest": digests["certificate"],
            "generation_digest": digests["generation"],
            "included_cases_digest": digests["included_cases"],
            "excluded_cases_digest": digests["excluded_cases"],
            "asset_inventory_digest": digests["asset_inventory"],
            "include_incomplete": bool(include_incomplete),
            "reason": reason,
        }

    def _read_receipt(self, dataset: str, release: str) -> Optional[sqlite3.Row]:
        with self._connect() as connection:
            metadata = self._metadata(connection)
            return connection.execute(
                "SELECT r.*,d.release_serial FROM ingestion_receipts r "
                "JOIN releases d USING(dataset_id,release_id) WHERE r.store_uuid=? "
                "AND r.dataset_id=? AND r.release_id=?",
                (metadata["store_uuid"], dataset, release),
            ).fetchone()

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
                  archive_class TEXT NOT NULL,
                  selected INTEGER NOT NULL,
                  case_json TEXT NOT NULL
                );
                CREATE TABLE spool_assets(
                  sha256 TEXT PRIMARY KEY,
                  size INTEGER NOT NULL
                );
                """
            )
            included = 0
            excluded = 0
            for case in cases:
                normalized = self._normalize_case(case)
                archive_class = normalized["assessment"]["archive_class"]
                selected = archive_class != "incomplete" or include_incomplete
                if selected:
                    included += 1
                    for structure in normalized["structures"]:
                        self._store_asset(structure)
                        existing = spool.execute(
                            "SELECT size FROM spool_assets WHERE sha256=?",
                            (structure["sha256"],),
                        ).fetchone()
                        if existing is not None and existing[0] != structure["size"]:
                            raise AssetIntegrityError(
                                "one content digest was declared with different sizes"
                            )
                        spool.execute(
                            "INSERT OR IGNORE INTO spool_assets(sha256,size) VALUES(?,?)",
                            (structure["sha256"], structure["size"]),
                        )
                else:
                    excluded += 1
                spool.execute(
                    "INSERT INTO spool_cases(case_id,public_index,local_id,archive_class,"
                    "selected,case_json) VALUES(?,?,?,?,?,?)",
                    (
                        normalized["id"],
                        normalized["public_index"],
                        normalized["local_id"],
                        archive_class,
                        int(selected),
                        _canonical_json(normalized),
                    ),
                )
            spool.commit()
            return {"included": included, "excluded": excluded}
        except sqlite3.IntegrityError as exc:
            raise SnapshotError(f"duplicate Case identity or index: {exc}") from exc
        finally:
            spool.close()

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
            {"namespace": str(item["namespace"]), "value": str(item["value"])}
            for item in case.get("identifiers", ())
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
        normalized = {
            "id": str(case["id"]),
            "public_index": int(case["public_index"]),
            "local_id": str(case["local_id"]),
            "facts": dict(case["facts"]),
            "assessment": assessment,
            "identifiers": identifiers,
            "structures": structures,
        }
        if normalized["public_index"] < 0:
            raise SnapshotError("Case public_index must be non-negative")
        _canonical_json(normalized)
        return normalized

    def _store_asset(self, descriptor: Mapping[str, Any]) -> None:
        source = Path(descriptor["source_path"])
        expected_digest = descriptor["sha256"]
        expected_size = descriptor["size"]
        if not source.is_file():
            raise AssetIntegrityError(f"materialized source asset is missing: {source}")
        source_digest, source_size = _sha256_file(source)
        if source_digest != expected_digest or source_size != expected_size:
            raise AssetIntegrityError(f"source asset does not match its declaration: {source}")

        parent = self.asset_root / "objects" / "sha256" / expected_digest[:2]
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / expected_digest
        lock_path = parent / f".{expected_digest}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if target.exists():
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
                    for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                        output_stream.write(block)
                        digest.update(block)
                        size += len(block)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
                if digest.hexdigest() != expected_digest or size != expected_size:
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
    def _stream_array_digest(values: Iterable[Any]) -> str:
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
        binding: Mapping[str, Any],
        counts: Mapping[str, int],
    ) -> TargetIngestResult:
        connection = self._connect()
        spool = sqlite3.connect(str(spool_path))
        spool.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            metadata = self._metadata(connection)
            if metadata["store_uuid"] != binding["store_uuid"]:
                raise ReceiptConflictError("target store identity changed before commit")
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
            for row in spool.execute(
                "SELECT case_json FROM spool_cases WHERE selected=1 ORDER BY sequence"
            ):
                self._insert_case(connection, header, json.loads(row["case_json"]))
            receipt_id = "receipt-" + _json_digest(binding)
            connection.execute(
                "INSERT INTO ingestion_receipts("
                "receipt_id,store_uuid,dataset_id,release_id,bundle_digest,"
                "certificate_digest,policy_digest,generation_digest,"
                "included_cases_digest,excluded_cases_digest,asset_inventory_digest,"
                "include_incomplete,reason,included_count,excluded_count,binding_json"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    receipt_id,
                    metadata["store_uuid"],
                    header["dataset"],
                    header["release"],
                    digests["bundle"],
                    digests["certificate"],
                    policy["digest"],
                    digests["generation"],
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
            spool.close()
            connection.close()

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

    def _insert_case(
        self,
        connection: sqlite3.Connection,
        header: Mapping[str, Any],
        case: Mapping[str, Any],
    ) -> None:
        assessment = case["assessment"]
        values = (header["dataset"], header["release"], case["id"])
        connection.execute(
            "INSERT INTO cases(dataset_id,release_id,case_id,public_index,local_id,"
            "source_payload_json) VALUES(?,?,?,?,?,?)",
            values
            + (
                case["public_index"],
                case["local_id"],
                _canonical_json(case["facts"]),
            ),
        )
        connection.execute(
            "INSERT INTO case_assessments(dataset_id,release_id,case_id,quality_level,"
            "evidence_level,archive_class,benchmark,assessment_json) "
            "VALUES(?,?,?,?,?,?,?,?)",
            values
            + (
                assessment.get("quality_level"),
                assessment.get("evidence_level"),
                assessment["archive_class"],
                int(assessment["benchmark"]),
                _canonical_json(assessment),
            ),
        )
        connection.execute(
            f"INSERT INTO {assessment['archive_class']}_case_index"
            "(dataset_id,release_id,case_id) VALUES(?,?,?)",
            values,
        )
        for identifier in case["identifiers"]:
            connection.execute(
                "INSERT INTO identifiers(dataset_id,release_id,case_id,namespace,value) "
                "VALUES(?,?,?,?,?)",
                values + (identifier["namespace"], identifier["value"]),
            )
        for structure in case["structures"]:
            relative = str(
                Path("objects")
                / "sha256"
                / structure["sha256"][:2]
                / structure["sha256"]
            )
            connection.execute(
                "INSERT OR IGNORE INTO target_assets(content_sha256,size,relative_path) "
                "VALUES(?,?,?)",
                (structure["sha256"], structure["size"], relative),
            )
            target = connection.execute(
                "SELECT size,relative_path FROM target_assets WHERE content_sha256=?",
                (structure["sha256"],),
            ).fetchone()
            if target["size"] != structure["size"] or target["relative_path"] != relative:
                raise AssetIntegrityError("target asset row conflicts with its content digest")
            connection.execute(
                "INSERT INTO structure_asset_refs(dataset_id,release_id,case_id,"
                "structure_id,format,role,is_primary,content_sha256) "
                "VALUES(?,?,?,?,?,?,?,?)",
                values
                + (
                    structure["structure_id"],
                    structure["format"],
                    structure["role"],
                    int(structure["primary"]),
                    structure["sha256"],
                ),
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
    "case_set_digest",
]
