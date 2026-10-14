"""Canonical, transactional source-fact state for one Bundle release.

This module is intentionally private.  Public ``Bundle`` and ``Case`` objects
delegate persistence to :class:`BundleState`; callers never need to know the
SQLite layout.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Optional, Union


BUNDLE_SCHEMA = "hotpot-ds-bundle/current"

_MANIFEST_REQUIRED = {
    "schema",
    "ds_name",
    "release_id",
    "version",
    "title",
    "description",
    "license",
    "access_policy",
    "provenance",
    "core_labels",
    "core_info",
    "intake",
    "identity_schema",
    "required_structure_roles",
}
_MANIFEST_ALLOWED = _MANIFEST_REQUIRED | {"dataset_id"}
_CASE_ALLOWED = {
    "id",
    "local_id",
    "public_index",
    "identifiers",
    "metadata",
    "core_info",
    "structures",
    "observations",
    "claims",
    "evidence",
    "verifications",
    "attestations",
    "claimed_quality_level",
    "claimed_evidence_level",
    "claimed_archive_class",
    "claimed_benchmark",
}
_CASE_DEFAULTS = {
    "identifiers": {},
    "metadata": {},
    "core_info": {},
    "structures": [],
    "observations": [],
    "claims": [],
    "evidence": [],
    "verifications": [],
    "attestations": [],
}
_OPERATIVE_FIELDS = {
    "quality_level",
    "evidence_level",
    "archive_class",
    "benchmark",
}
_ARRAY_FIELDS = (
    "structures",
    "observations",
    "claims",
    "evidence",
    "verifications",
    "attestations",
)
_ASSET_FIELDS = ("structures", "evidence")
_RECORD_ID_FIELDS = {
    "structures": "structure_id",
    "observations": "observation_id",
    "claims": "claim_id",
    "evidence": "evidence_id",
    "verifications": "verification_id",
    "attestations": "attestation_id",
}


class BundleStateError(ValueError):
    """Base error for an invalid or inconsistent canonical Bundle state."""


class BundleAlreadyBuiltError(BundleStateError):
    """Raised when construction is attempted for an already built Bundle."""


class BundleNotBuiltError(BundleStateError):
    """Raised when validation-only open cannot find the build marker."""


class ConcurrentStateError(BundleStateError):
    """Raised when a compare-and-swap publication observes changed inputs."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_text(value: Any) -> str:
    return _canonical_bytes(value).decode("utf-8")


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_text(value))


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BundleStateError(f"{name} must be a non-empty string")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise BundleStateError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise BundleStateError(f"{name} cannot contain duplicates")
    return list(value)


def _safe_relative_path(root: Path, value: Any, name: str) -> Path:
    path_text = _nonempty_string(value, name)
    if "\\" in path_text or "\x00" in path_text:
        raise BundleStateError(f"{name} must be a safe POSIX relative path")
    path = PurePosixPath(path_text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BundleStateError(f"{name} must be a safe POSIX relative path")
    root_resolved = root.resolve()
    resolved = (root / Path(*path.parts)).resolve(strict=False)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise BundleStateError(f"{name} escapes the Bundle root") from exc
    return root / Path(*path.parts)


def _read_json(path: Path, name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleStateError(f"cannot read valid {name}: {path}") from exc


def _validate_manifest(root: Path, value: Any, ds_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleStateError("bundle.json must contain a JSON object")
    missing = _MANIFEST_REQUIRED - value.keys()
    extra = value.keys() - _MANIFEST_ALLOWED
    if missing:
        raise BundleStateError(
            "bundle.json is missing required fields: " + ", ".join(sorted(missing))
        )
    if extra:
        raise BundleStateError(
            "bundle.json contains unsupported fields: " + ", ".join(sorted(extra))
        )
    if value["schema"] != BUNDLE_SCHEMA:
        raise BundleStateError(f"bundle.json.schema must be {BUNDLE_SCHEMA!r}")
    _nonempty_string(ds_name, "ds_name")
    if value["ds_name"] != ds_name:
        raise BundleStateError("bundle.json.ds_name does not match ds_name")
    dataset_id = value.get("dataset_id", ds_name)
    if dataset_id != ds_name:
        raise BundleStateError("bundle.json.dataset_id must equal ds_name")
    for field in (
        "release_id",
        "version",
        "title",
        "description",
        "license",
        "access_policy",
    ):
        _nonempty_string(value[field], f"bundle.json.{field}")
    if not isinstance(value["provenance"], dict):
        raise BundleStateError("bundle.json.provenance must be an object")
    if not isinstance(value["identity_schema"], dict):
        raise BundleStateError("bundle.json.identity_schema must be an object")
    _string_list(value["core_labels"], "bundle.json.core_labels")
    _string_list(value["core_info"], "bundle.json.core_info")
    _string_list(
        value["required_structure_roles"],
        "bundle.json.required_structure_roles",
    )
    intake = value["intake"]
    if not isinstance(intake, dict) or set(intake) != {"format", "path"}:
        raise BundleStateError("bundle.json.intake requires only format and path")
    if intake["format"] not in {"jsonl", "sqlite"}:
        raise BundleStateError("bundle.json.intake.format must be jsonl or sqlite")
    _safe_relative_path(root, intake["path"], "bundle.json.intake.path")
    normalized = _json_copy(value)
    normalized["dataset_id"] = dataset_id
    return normalized


def _validate_record_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise BundleStateError(f"case.{name} must be an array")
    result = []
    seen = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise BundleStateError(f"case.{name}[{index}] must be an object")
        record = _json_copy(item)
        identity_field = _RECORD_ID_FIELDS[name]
        record_id = record.get(identity_field)
        if not isinstance(record_id, str) or not record_id:
            raise BundleStateError(
                f"case.{name}[{index}].{identity_field} is required"
            )
        if record_id in seen:
            raise BundleStateError(f"case.{name} contains duplicate id {record_id!r}")
        seen.add(record_id)
        result.append(record)
    return result


def _validate_case(
    root: Path,
    value: Any,
    *,
    supplied_public_index: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleStateError("each case must be a JSON object")
    forbidden = _OPERATIVE_FIELDS & value.keys()
    if forbidden:
        field = sorted(forbidden)[0]
        raise BundleStateError(
            f"case.{field} is an operative assessment field and is not input"
        )
    extra = value.keys() - _CASE_ALLOWED
    if extra:
        raise BundleStateError(
            "case contains unsupported fields: " + ", ".join(sorted(extra))
        )
    case = _json_copy(value)
    _nonempty_string(case.get("id"), "case.id")
    _nonempty_string(case.get("local_id"), "case.local_id")
    declared_index = case.get("public_index", supplied_public_index)
    if supplied_public_index is not None and "public_index" in case:
        if case["public_index"] != supplied_public_index:
            raise BundleStateError(
                "case.public_index disagrees with the SQLite intake index"
            )
    if not isinstance(declared_index, int) or isinstance(declared_index, bool):
        raise BundleStateError("case.public_index must be a non-negative integer")
    if declared_index < 0:
        raise BundleStateError("case.public_index must be a non-negative integer")
    case["public_index"] = declared_index
    for name, default in _CASE_DEFAULTS.items():
        case.setdefault(name, copy.deepcopy(default))
    for name in ("identifiers", "metadata", "core_info"):
        if not isinstance(case[name], dict):
            raise BundleStateError(f"case.{name} must be an object")
    for name in _ARRAY_FIELDS:
        case[name] = _validate_record_list(case[name], name)
    for name in ("observations", "claims"):
        for record in case[name]:
            forbidden = _OPERATIVE_FIELDS & record.keys()
            if forbidden:
                field = sorted(forbidden)[0]
                raise BundleStateError(
                    f"case.{name}.{field} is an operative assessment field and is not input"
                )
    for name in _ASSET_FIELDS:
        for record in case[name]:
            for asset_path in _record_asset_paths(record):
                asset = _safe_relative_path(root, asset_path, f"case.{name}.path")
                if asset.exists() and not asset.is_file():
                    raise BundleStateError(
                        f"case.{name}.path must name a regular file when present"
                    )
    return case


def _record_asset_paths(record: Mapping[str, Any]) -> Iterator[str]:
    direct = record.get("path")
    if isinstance(direct, str):
        yield direct
    assets = record.get("assets", [])
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, Mapping) and isinstance(asset.get("path"), str):
                yield asset["path"]
    representations = record.get("representations", {})
    if isinstance(representations, Mapping):
        for representation in representations.values():
            if isinstance(representation, str):
                yield representation
            elif isinstance(representation, Mapping) and isinstance(
                representation.get("path"), str
            ):
                yield representation["path"]
    elif isinstance(representations, list):
        for representation in representations:
            if isinstance(representation, Mapping) and isinstance(
                representation.get("path"), str
            ):
                yield representation["path"]


def _frame(hasher: "hashlib._Hash", name: str, payload: bytes) -> None:
    label = name.encode("utf-8")
    hasher.update(len(label).to_bytes(8, "big"))
    hasher.update(label)
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _hash_file(hasher: "hashlib._Hash", path: Path) -> None:
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
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


class BundleState:
    """Private authority for canonical Case facts and Bundle revisions."""

    def __init__(self, root: Path, ds_name: str):
        self.root = root.resolve()
        self.ds_name = ds_name
        self.path = self.root / "cases" / "facts.sqlite3"

    @classmethod
    def build(
        cls,
        root: Union[os.PathLike[str], str],
        ds_name: str,
        policy_json: Mapping[str, Any],
        target_context: Mapping[str, Any],
    ) -> "BundleState":
        bundle_root = Path(root).resolve()
        manifest = _validate_manifest(
            bundle_root,
            _read_json(bundle_root / "bundle.json", "bundle.json"),
            ds_name,
        )
        state = cls(bundle_root, ds_name)
        policy = _json_copy(dict(policy_json))
        target = _json_copy(dict(target_context))
        policy_digest = _digest_json(policy)
        intake = _safe_relative_path(
            bundle_root,
            manifest["intake"]["path"],
            "bundle.json.intake.path",
        )
        if not intake.is_file():
            raise BundleStateError(f"intake artifact does not exist: {intake}")

        state.path.parent.mkdir(parents=True, exist_ok=True)
        lock_descriptor = os.open(
            state.path.parent / ".build.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        temporary = state.path.with_name(f".facts.{uuid.uuid4().hex}.tmp")
        event_id = f"event-{uuid.uuid4().hex}"
        created_at = _utc_now()
        connection: Optional[sqlite3.Connection] = None
        try:
            if state.path.exists():
                raise BundleAlreadyBuiltError(
                    "Bundle is already built; use validation-only load_bundle()"
                )
            connection = sqlite3.connect(temporary)
            connection.row_factory = sqlite3.Row
            state._initialize(connection)
            with connection:
                connection.execute(
                    "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                    ("manifest", _canonical_text(manifest)),
                )
                connection.execute(
                    "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                    ("target_context", _canonical_text(target)),
                )
                connection.execute(
                    "INSERT INTO policy(singleton, policy_json) VALUES (1, ?)",
                    (_canonical_text(policy),),
                )
                for case in state._iter_intake(manifest, intake):
                    connection.execute(
                        """
                        INSERT INTO cases(case_id, public_index, local_id, payload_json)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            case["id"],
                            case["public_index"],
                            case["local_id"],
                            _canonical_text(case),
                        ),
                    )
                source_digest = state._compute_source_digest(connection, manifest)
                connection.execute(
                    """
                    INSERT INTO control(
                        singleton, source_revision, source_digest,
                        policy_revision, policy_digest
                    ) VALUES (1, 0, ?, 0, ?)
                    """,
                    (source_digest, policy_digest),
                )
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, created_at, action, case_id, details_json,
                        source_revision, policy_revision
                    ) VALUES (?, ?, 'bundle_built', NULL, ?, 0, 0)
                    """,
                    (
                        event_id,
                        created_at,
                        _canonical_text(
                            {
                                "ds_name": ds_name,
                                "release_id": manifest["release_id"],
                            }
                        ),
                    ),
                )
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.close()
            connection = None
            state._write_policy_projection_value(policy)
            state._write_audit_projection_rows(
                [
                    {
                        "event_id": event_id,
                        "created_at": created_at,
                        "action": "bundle_built",
                        "case_id": None,
                        "details": {
                            "ds_name": ds_name,
                            "release_id": manifest["release_id"],
                        },
                        "source_revision": 0,
                        "policy_revision": 0,
                    }
                ]
            )
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, state.path)
            descriptor = os.open(state.path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except sqlite3.IntegrityError as exc:
            raise BundleStateError(f"duplicate case id, local id, or public_index: {exc}") from exc
        finally:
            if connection is not None:
                connection.close()
            if temporary.exists():
                temporary.unlink()
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        return cls.open(bundle_root, ds_name)

    @classmethod
    def open(
        cls, root: Union[os.PathLike[str], str], ds_name: str
    ) -> "BundleState":
        bundle_root = Path(root).resolve()
        current_manifest = _validate_manifest(
            bundle_root,
            _read_json(bundle_root / "bundle.json", "bundle.json"),
            ds_name,
        )
        state = cls(bundle_root, ds_name)
        if not state.path.is_file():
            raise BundleNotBuiltError(
                "Bundle is not built; construct it with Bundle(...) first"
            )
        try:
            with state.connect() as connection:
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise BundleStateError("facts.sqlite3 failed integrity_check")
                if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise BundleStateError("facts.sqlite3 is not the current schema")
                stored_manifest = json.loads(
                    connection.execute(
                        "SELECT value_json FROM metadata WHERE key='manifest'"
                    ).fetchone()[0]
                )
                _validate_manifest(bundle_root, stored_manifest, ds_name)
                if stored_manifest != current_manifest:
                    raise BundleStateError(
                        "bundle.json differs from the immutable built manifest"
                    )
                if connection.execute("SELECT COUNT(*) FROM control").fetchone()[0] != 1:
                    raise BundleStateError("facts.sqlite3 has no unique control record")
                for row in connection.execute(
                    "SELECT public_index, payload_json FROM cases ORDER BY public_index"
                ):
                    _validate_case(
                        bundle_root,
                        json.loads(row["payload_json"]),
                        supplied_public_index=row["public_index"],
                    )
        except (sqlite3.Error, TypeError, KeyError, json.JSONDecodeError) as exc:
            if isinstance(exc, BundleStateError):
                raise
            raise BundleStateError("facts.sqlite3 is not a valid current Bundle store") from exc
        return state

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        finally:
            connection.close()

    @property
    def manifest(self) -> Dict[str, Any]:
        return _validate_manifest(
            self.root,
            _read_json(self.root / "bundle.json", "bundle.json"),
            self.ds_name,
        )

    @property
    def policy(self) -> Dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT policy_json FROM policy WHERE singleton=1"
            ).fetchone()
        return json.loads(row[0])

    @property
    def target_context(self) -> Dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM metadata WHERE key='target_context'"
            ).fetchone()
        return json.loads(row[0])

    def __len__(self) -> int:
        with self.connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]

    def iter_cases(self) -> Iterator[Dict[str, Any]]:
        with self.connect() as connection:
            cursor = connection.execute(
                "SELECT payload_json FROM cases ORDER BY public_index, case_id"
            )
            for row in cursor:
                yield json.loads(row[0])

    def get_case(self, case_id: str) -> Dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM cases WHERE case_id=?", (case_id,)
            ).fetchone()
        if row is None:
            raise KeyError(case_id)
        return json.loads(row[0])

    @property
    def source_digest(self) -> str:
        manifest = self.manifest
        with self.connect() as connection:
            return self._compute_source_digest(connection, manifest)

    @property
    def policy_digest(self) -> str:
        return _digest_json(self.policy)

    @property
    def status(self) -> Dict[str, Any]:
        source_digest = self.source_digest
        policy_digest = self.policy_digest
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM control WHERE singleton=1").fetchone()
        status = dict(row)
        status["recorded_source_digest"] = status.pop("source_digest")
        status["recorded_policy_digest"] = status.pop("policy_digest")
        status["source_digest"] = source_digest
        status["policy_digest"] = policy_digest
        status["is_synchronized"] = bool(
            status["current_generation_id"]
            and status["source_revision"] == status["synchronized_source_revision"]
            and source_digest == status["recorded_source_digest"]
            and source_digest == status["synchronized_source_digest"]
            and status["policy_revision"] == status["synchronized_policy_revision"]
            and policy_digest == status["recorded_policy_digest"]
            and policy_digest == status["synchronized_policy_digest"]
        )
        status.pop("singleton", None)
        return _json_copy(status)

    def refresh_source_state(self) -> Dict[str, Any]:
        """Audit an out-of-band scientific asset change as a source revision."""

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                digest = self._compute_source_digest(connection, self.manifest)
                if digest == control["source_digest"]:
                    connection.rollback()
                    return {
                        "changed": False,
                        "source_revision": control["source_revision"],
                        "source_digest": digest,
                        "event_id": None,
                    }
                revision = control["source_revision"] + 1
                event_id = f"event-{uuid.uuid4().hex}"
                connection.execute(
                    "UPDATE control SET source_revision=?,source_digest=? "
                    "WHERE singleton=1",
                    (revision, digest),
                )
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, created_at, action, case_id, details_json,
                        source_revision, policy_revision
                    ) VALUES (?, ?, 'scientific_asset_state_changed', NULL, ?, ?, ?)
                    """,
                    (
                        event_id,
                        _utc_now(),
                        _canonical_text(
                            {
                                "previous_source_digest": control["source_digest"],
                                "current_source_digest": digest,
                            }
                        ),
                        revision,
                        control["policy_revision"],
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self.write_audit_projection()
        return {
            "changed": True,
            "source_revision": revision,
            "source_digest": digest,
            "event_id": event_id,
        }

    def mutate_case(
        self,
        case_id: str,
        new_payload: Mapping[str, Any],
        action: str,
        details: Mapping[str, Any],
    ) -> Dict[str, Any]:
        _nonempty_string(action, "action")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM cases WHERE case_id=?", (case_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(case_id)
                case = _validate_case(
                    self.root,
                    dict(new_payload),
                    supplied_public_index=row["public_index"],
                )
                old = json.loads(row["payload_json"])
                if case["id"] != case_id or case["local_id"] != row["local_id"]:
                    raise BundleStateError("case id and local_id are immutable")
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                revision = control["source_revision"] + 1
                connection.execute(
                    "UPDATE cases SET payload_json=? WHERE case_id=?",
                    (_canonical_text(case), case_id),
                )
                digest = self._compute_source_digest(connection, self.manifest)
                connection.execute(
                    "UPDATE control SET source_revision=?, source_digest=? WHERE singleton=1",
                    (revision, digest),
                )
                event_details = _json_copy(dict(details))
                event_details.update(
                    {
                        "old_payload_digest": _digest_json(old),
                        "new_payload_digest": _digest_json(case),
                    }
                )
                event_id = f"event-{uuid.uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, created_at, action, case_id, details_json,
                        source_revision, policy_revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        _utc_now(),
                        action,
                        case_id,
                        _canonical_text(event_details),
                        revision,
                        control["policy_revision"],
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self.write_audit_projection()
        return {
            "case_id": case_id,
            "event_id": event_id,
            "source_revision": revision,
            "source_digest": digest,
        }

    def mutate_policy(
        self,
        new_policy: Mapping[str, Any],
        *,
        action: str = "gate_policy_changed",
        details: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        _nonempty_string(action, "action")
        policy = _json_copy(dict(new_policy))
        digest = _digest_json(policy)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                revision = control["policy_revision"] + 1
                connection.execute(
                    "UPDATE policy SET policy_json=? WHERE singleton=1",
                    (_canonical_text(policy),),
                )
                connection.execute(
                    "UPDATE control SET policy_revision=?, policy_digest=? WHERE singleton=1",
                    (revision, digest),
                )
                event_id = f"event-{uuid.uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, created_at, action, case_id, details_json,
                        source_revision, policy_revision
                    ) VALUES (?, ?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        event_id,
                        _utc_now(),
                        action,
                        _canonical_text(dict(details or {})),
                        control["source_revision"],
                        revision,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self.write_policy_projection()
        self.write_audit_projection()
        return {
            "event_id": event_id,
            "policy_revision": revision,
            "policy_digest": digest,
        }

    def publish_generation(
        self,
        generation_id: str,
        generation_digest: str,
        *,
        expected_source_revision: int,
        expected_source_digest: str,
        expected_policy_revision: int,
        expected_policy_digest: str,
    ) -> Dict[str, Any]:
        _nonempty_string(generation_id, "generation_id")
        _nonempty_string(generation_digest, "generation_digest")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                current_source = self._compute_source_digest(connection, self.manifest)
                current_policy = _digest_json(
                    json.loads(
                        connection.execute(
                            "SELECT policy_json FROM policy WHERE singleton=1"
                        ).fetchone()[0]
                    )
                )
                actual = (
                    control["source_revision"],
                    current_source,
                    control["policy_revision"],
                    current_policy,
                )
                expected = (
                    expected_source_revision,
                    expected_source_digest,
                    expected_policy_revision,
                    expected_policy_digest,
                )
                if actual != expected:
                    raise ConcurrentStateError(
                        "Bundle inputs changed before generation publication"
                    )
                changed = connection.execute(
                    """
                    UPDATE control SET
                        synchronized_source_revision=?,
                        synchronized_source_digest=?,
                        synchronized_policy_revision=?,
                        synchronized_policy_digest=?,
                        current_generation_id=?,
                        current_generation_digest=?
                    WHERE singleton=1
                      AND source_revision=? AND source_digest=?
                      AND policy_revision=? AND policy_digest=?
                    """,
                    (
                        expected_source_revision,
                        expected_source_digest,
                        expected_policy_revision,
                        expected_policy_digest,
                        generation_id,
                        generation_digest,
                        expected_source_revision,
                        expected_source_digest,
                        expected_policy_revision,
                        expected_policy_digest,
                    ),
                ).rowcount
                if changed != 1:
                    raise ConcurrentStateError(
                        "Bundle inputs changed before generation publication"
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return {
            "generation_id": generation_id,
            "generation_digest": generation_digest,
            "source_revision": expected_source_revision,
            "policy_revision": expected_policy_revision,
        }

    def write_audit_projection(self) -> Path:
        with self.connect() as connection:
            rows = []
            for row in connection.execute(
                "SELECT * FROM audit_events ORDER BY rowid"
            ):
                rows.append(
                    {
                        "event_id": row["event_id"],
                        "created_at": row["created_at"],
                        "action": row["action"],
                        "case_id": row["case_id"],
                        "details": json.loads(row["details_json"]),
                        "source_revision": row["source_revision"],
                        "policy_revision": row["policy_revision"],
                    }
                )
        return self._write_audit_projection_rows(rows)

    def write_policy_projection(self) -> Path:
        return self._write_policy_projection_value(self.policy)

    def _write_audit_projection_rows(self, rows: list[dict[str, Any]]) -> Path:
        path = self.root / "audit" / "events.jsonl"
        content = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
        _atomic_write(path, content)
        return path

    def _write_policy_projection_value(self, policy: Mapping[str, Any]) -> Path:
        path = self.root / "policy" / "policy.json"
        _atomic_write(path, _canonical_bytes(policy) + b"\n")
        return path

    def _initialize(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            PRAGMA user_version=1;

            CREATE TABLE metadata(
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            ) STRICT;

            CREATE TABLE control(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                source_revision INTEGER NOT NULL CHECK(source_revision >= 0),
                source_digest TEXT NOT NULL,
                policy_revision INTEGER NOT NULL CHECK(policy_revision >= 0),
                policy_digest TEXT NOT NULL,
                synchronized_source_revision INTEGER,
                synchronized_source_digest TEXT,
                synchronized_policy_revision INTEGER,
                synchronized_policy_digest TEXT,
                current_generation_id TEXT,
                current_generation_digest TEXT
            ) STRICT;

            CREATE TABLE cases(
                case_id TEXT PRIMARY KEY,
                public_index INTEGER NOT NULL UNIQUE CHECK(public_index >= 0),
                local_id TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            ) STRICT;

            CREATE TABLE audit_events(
                event_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                action TEXT NOT NULL,
                case_id TEXT,
                details_json TEXT NOT NULL,
                source_revision INTEGER NOT NULL,
                policy_revision INTEGER NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(case_id)
            ) STRICT;

            CREATE TABLE policy(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                policy_json TEXT NOT NULL
            ) STRICT;
            """
        )

    def _iter_intake(
        self, manifest: Mapping[str, Any], intake: Path
    ) -> Iterator[Dict[str, Any]]:
        if manifest["intake"]["format"] == "jsonl":
            try:
                with intake.open("r", encoding="utf-8") as stream:
                    for index, line in enumerate(stream):
                        if not line.strip():
                            raise BundleStateError(
                                f"blank JSONL record at line {index + 1}"
                            )
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise BundleStateError(
                                f"invalid JSONL record at line {index + 1}"
                            ) from exc
                        yield _validate_case(
                            self.root,
                            value,
                            supplied_public_index=value.get("public_index", index)
                            if isinstance(value, dict)
                            else index,
                        )
            except UnicodeError as exc:
                raise BundleStateError("JSONL intake must be UTF-8") from exc
            return

        uri = intake.resolve().as_uri() + "?mode=ro"
        source = sqlite3.connect(uri, uri=True)
        source.row_factory = sqlite3.Row
        try:
            columns = {
                row["name"]: row
                for row in source.execute("PRAGMA table_info(cases)")
            }
            if set(columns) != {"case_id", "public_index", "payload_json"}:
                raise BundleStateError(
                    "SQLite intake requires cases(case_id, public_index, payload_json)"
                )
            if columns["case_id"]["pk"] != 1:
                raise BundleStateError("SQLite intake cases.case_id must be PRIMARY KEY")
            for row in source.execute(
                "SELECT case_id, public_index, payload_json FROM cases ORDER BY public_index"
            ):
                try:
                    value = json.loads(row["payload_json"])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise BundleStateError(
                        f"invalid payload_json for SQLite case {row['case_id']!r}"
                    ) from exc
                if value.get("id") != row["case_id"]:
                    raise BundleStateError(
                        "SQLite case_id must equal payload_json.id"
                    )
                yield _validate_case(
                    self.root,
                    value,
                    supplied_public_index=row["public_index"],
                )
        except sqlite3.Error as exc:
            raise BundleStateError("invalid SQLite intake") from exc
        finally:
            source.close()

    def _compute_source_digest(
        self, connection: sqlite3.Connection, manifest: Mapping[str, Any]
    ) -> str:
        hasher = hashlib.sha256()
        _frame(hasher, "schema", BUNDLE_SCHEMA.encode("utf-8"))
        _frame(hasher, "manifest", _canonical_bytes(manifest))
        assets: set[str] = set()
        for row in connection.execute(
            "SELECT case_id, public_index, payload_json FROM cases "
            "ORDER BY public_index, case_id"
        ):
            payload = json.loads(row["payload_json"])
            _frame(
                hasher,
                f"case:{row['public_index']}:{row['case_id']}",
                _canonical_bytes(payload),
            )
            for field in _ASSET_FIELDS:
                for record in payload.get(field, []):
                    assets.update(_record_asset_paths(record))
        for relative in sorted(assets):
            path = _safe_relative_path(self.root, relative, "asset path")
            if not path.exists():
                _frame(hasher, f"asset:{relative}", b"missing")
            elif not path.is_file():
                _frame(hasher, f"asset:{relative}", b"not-a-regular-file")
            else:
                asset_hasher = hashlib.sha256()
                _hash_file(asset_hasher, path)
                _frame(
                    hasher,
                    f"asset:{relative}",
                    b"present\0" + asset_hasher.digest(),
                )
        return hasher.hexdigest()


__all__ = [
    "BUNDLE_SCHEMA",
    "BundleAlreadyBuiltError",
    "BundleNotBuiltError",
    "BundleState",
    "BundleStateError",
    "ConcurrentStateError",
]
