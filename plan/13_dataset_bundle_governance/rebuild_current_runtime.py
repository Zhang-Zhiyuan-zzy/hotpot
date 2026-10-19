#!/usr/bin/env python3
"""Rebuild the Stage 13 releases with only the current Hotpot runtime.

This is an owner-operated, one-time rebuild tool.  It deliberately lives in
``plan/`` rather than the package: reading the previous runtime is migration
work, not a compatibility feature of :mod:`hotpot.ds`.

The default command is ``plan`` and is read-only.  ``run`` builds and verifies
the candidate, while ``activate`` is a separate, explicitly confirmed step::

    python plan/13_dataset_bundle_governance/rebuild_current_runtime.py plan
    python plan/13_dataset_bundle_governance/rebuild_current_runtime.py run
    python plan/13_dataset_bundle_governance/rebuild_current_runtime.py status
    python plan/13_dataset_bundle_governance/rebuild_current_runtime.py \
        activate --confirm-run-id stage13-compact-current
    python plan/13_dataset_bundle_governance/rebuild_current_runtime.py \
        rollback --confirm-activation-id activation-<id>

No source Bundle or source store is ever opened writable.  The old active
store is only the rollback anchor and a stable byte-hash input; its schema and
CAS layout are deliberately irrelevant.  Gate definitions come from each
Stage 13 migration Bundle's own persisted policy.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
import uuid
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


STATE_SCHEMA = "hotpot-ds-current-runtime-rebuild/current"
EVENT_SCHEMA = "hotpot-ds-current-runtime-rebuild-event/current"
ACTIVATION_SCHEMA = "hotpot-ds-store-cutover/current"
SOURCE_GENERATION = "stage13-efficiency-current"
RUN_ID = "stage13-compact-current"
INCOMPLETE_REASON = (
    "Owner-approved one-time migration preserving incomplete legacy Cases."
)
COPY_CHUNK = 4 * 1024 * 1024
FICLONE = 0x40049409
_SAFE_COMPONENT = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_CAS_LOCK_NAME = re.compile(r"\.([0-9a-f]{64})\.lock\Z")
_DURABLE_CONTROL_CATEGORIES = frozenset(
    {"certificate", "source", "evidence", "provenance", "checker"}
)
MAX_CANDIDATE_BYTES = 1024**3


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    release: str
    source_relative: str


DATASETS = (
    DatasetSpec("oxidation", "oxidation.v001", "oxidation/oxidation.v001"),
    DatasetSpec("spin", "spin.v001", "spin/spin.v001"),
    DatasetSpec(
        "csd-electronic-state",
        "csd-electronic-state.vreported-text-v1",
        "csd-electronic-state/csd-electronic-state.vreported-text-v1",
    ),
)


class RebuildError(RuntimeError):
    """The candidate cannot be proved safe enough to continue."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stream_json_array_digest(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for value in values:
        if not first:
            digest.update(b",")
        digest.update(_canonical_bytes(value))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Durably publish every regular file and directory in a candidate tree."""

    directories = [root]
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RebuildError(f"candidate tree contains a symlink: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(path)
        elif stat.S_ISREG(metadata.st_mode):
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            raise RebuildError(f"candidate tree contains a non-regular node: {path}")
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = _canonical_bytes(dict(value)) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _append_event(path: Path, action: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    event = {
        "schema": EVENT_SCHEMA,
        "event_id": f"event-{uuid.uuid4().hex}",
        "created_at": _utc_now(),
        "action": action,
        "details": details,
    }
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        remaining = memoryview(_canonical_bytes(event) + b"\n")
        while remaining:
            written = os.write(descriptor, remaining)
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not existed:
        _fsync_directory(path.parent)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RebuildError(f"expected a JSON object: {path}")
    return value


def _read_stable_json(
    path: Path,
) -> tuple[dict[str, Any], str, int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RebuildError(f"not a regular JSON file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stamp = _stat_stamp(after)
    if _stat_stamp(before) != stamp or _stat_stamp(path.stat(follow_symlinks=False)) != stamp:
        raise RebuildError(f"JSON file changed while reading: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RebuildError(f"expected a JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest(), len(payload), stamp


def _sha256_file(path: Path) -> tuple[str, int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RebuildError(f"not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(COPY_CHUNK), b""):
                digest.update(chunk)
                size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_stamp = _stat_stamp(before)
    after_stamp = _stat_stamp(after)
    try:
        live_stamp = _stat_stamp(path.stat(follow_symlinks=False))
    except FileNotFoundError as error:
        raise RebuildError(f"file disappeared while hashing: {path}") from error
    if before_stamp != after_stamp or after_stamp != live_stamp or size != after.st_size:
        raise RebuildError(f"file changed while hashing: {path}")
    return digest.hexdigest(), size, after_stamp


def _stat_stamp(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _safe_relative(value: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RebuildError(f"unsafe Bundle-relative path: {value!r}")
    return relative


def _safe_component(value: str, name: str) -> str:
    if value in {".", ".."} or _SAFE_COMPONENT.fullmatch(value) is None:
        raise RebuildError(
            f"{name} must be one lowercase filesystem component containing only "
            "letters, digits, '.', '_' or '-'"
        )
    return value


def _path_lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _assert_managed_path(
    anchor: Path,
    path: Path,
    *,
    leaf_kind: str,
) -> None:
    """Reject symlinks and unexpected nodes in an owner-managed namespace."""

    try:
        relative = path.relative_to(anchor)
    except ValueError as error:
        raise RebuildError(f"managed path escapes its anchor: {path}") from error
    current = anchor
    components = (Path("."), *relative.parts)
    for index, component in enumerate(components):
        if component != Path("."):
            current = current / component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise RebuildError(f"managed path traverses a symlink: {current}")
        is_leaf = index == len(components) - 1
        expected_directory = not is_leaf or leaf_kind == "directory"
        expected_regular = is_leaf and leaf_kind == "file"
        if expected_directory and not stat.S_ISDIR(metadata.st_mode):
            raise RebuildError(f"managed path component is not a directory: {current}")
        if expected_regular and not stat.S_ISREG(metadata.st_mode):
            raise RebuildError(f"managed file is not regular: {current}")


def _bundle_path(root: Path, relative: Path) -> Path:
    """Resolve a Bundle-relative path without accepting symlink traversal."""

    if root.is_symlink():
        raise RebuildError(f"Bundle root must not be a symlink: {root}")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise RebuildError(f"Bundle path traverses a symlink: {current}")
    return current


def _paths_in_record(record: Mapping[str, Any]) -> Iterator[str]:
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
        representations = representations.values()
    if isinstance(representations, Iterable) and not isinstance(
        representations, (str, bytes)
    ):
        for representation in representations:
            if isinstance(representation, str):
                yield representation
            elif isinstance(representation, Mapping) and isinstance(
                representation.get("path"), str
            ):
                yield representation["path"]


@contextlib.contextmanager
def _referenced_paths(intake: Path) -> Iterator[sqlite3.Connection]:
    descriptor, name = tempfile.mkstemp(prefix=".hotpot-rebuild-paths-", suffix=".sqlite3")
    os.close(descriptor)
    path = Path(name)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE paths(path TEXT PRIMARY KEY) WITHOUT ROWID")
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value INTEGER)")
        batch: list[tuple[str]] = []
        case_count = 0
        with intake.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RebuildError(
                        f"invalid intake JSON at {intake}:{line_number}"
                    ) from error
                if not isinstance(case, Mapping):
                    raise RebuildError(f"non-object intake row at {intake}:{line_number}")
                case_count += 1
                for field in ("structures", "evidence"):
                    records = case.get(field, [])
                    if not isinstance(records, list):
                        raise RebuildError(f"case.{field} is not an array at line {line_number}")
                    for record in records:
                        if isinstance(record, Mapping):
                            for value in _paths_in_record(record):
                                batch.append((str(_safe_relative(value)),))
                if len(batch) >= 4096:
                    connection.executemany("INSERT OR IGNORE INTO paths VALUES (?)", batch)
                    batch.clear()
        if batch:
            connection.executemany("INSERT OR IGNORE INTO paths VALUES (?)", batch)
        connection.execute("INSERT INTO metadata VALUES('case_count',?)", (case_count,))
        connection.commit()
        yield connection
    finally:
        connection.close()
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _copy_regular(source: Path, destination: Path) -> tuple[str, int]:
    """Copy one stable regular file, preferring CoW reflink, and verify bytes."""

    source_digest, source_size, source_stamp = _sha256_file(source)
    if destination.exists():
        destination_digest, destination_size, _ = _sha256_file(destination)
        if (destination_digest, destination_size) != (source_digest, source_size):
            raise RebuildError(f"partial output conflicts with source: {destination}")
        return source_digest, source_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        try:
            fcntl.ioctl(destination_descriptor, FICLONE, source_descriptor)
        except OSError as error:
            if error.errno not in {
                errno.EOPNOTSUPP,
                errno.ENOTTY,
                errno.EXDEV,
                errno.EINVAL,
                errno.ENOSYS,
            }:
                raise
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            while True:
                block = os.read(source_descriptor, COPY_CHUNK)
                if not block:
                    break
                view = memoryview(block)
                while view:
                    written = os.write(destination_descriptor, view)
                    view = view[written:]
        os.fsync(destination_descriptor)
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)
    try:
        if _stat_stamp(source.stat(follow_symlinks=False)) != source_stamp:
            raise RebuildError(f"source changed while copying: {source}")
        copied_digest, copied_size, _ = _sha256_file(temporary)
        if (copied_digest, copied_size) != (source_digest, source_size):
            raise RebuildError(f"copied bytes do not match source: {source}")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return source_digest, source_size


def _closure_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return _json_digest(list(entries))


def _copy_bundle_closure(source: Path, partial: Path) -> dict[str, Any]:
    manifest = _read_json(_bundle_path(source, Path("bundle.json")))
    intake_relative = _safe_relative(manifest["intake"]["path"])
    core = (Path("bundle.json"), intake_relative)
    entries: dict[str, dict[str, Any]] = {}
    for relative in core:
        digest, size = _copy_regular(
            _bundle_path(source, relative), _bundle_path(partial, relative)
        )
        entries[str(relative)] = {
            "path": str(relative), "state": "present", "sha256": digest, "size": size
        }

    controls = {}
    for declaration in manifest.get("control_assets", []):
        relative = _safe_relative(declaration["path"])
        identity = (declaration["category"], declaration["name"])
        if identity in controls:
            raise RebuildError(f"duplicate control-asset identity in {source}")
        controls[identity] = str(relative)
        digest, size = _copy_regular(
            _bundle_path(source, relative), _bundle_path(partial, relative)
        )
        if digest != declaration["sha256"] or size != declaration["size"]:
            raise RebuildError(f"declared control asset does not match: {source / relative}")
        entries[str(relative)] = {
            "path": str(relative), "state": "present", "sha256": digest, "size": size
        }

    with _referenced_paths(_bundle_path(partial, intake_relative)) as references:
        for (value,) in references.execute("SELECT path FROM paths ORDER BY path"):
            relative = _safe_relative(value)
            source_path = _bundle_path(source, relative)
            if source_path.exists():
                digest, size = _copy_regular(
                    source_path, _bundle_path(partial, relative)
                )
                entries[value] = {
                    "path": value, "state": "present", "sha256": digest, "size": size
                }
            else:
                entries[value] = {
                    "path": value, "state": "missing", "sha256": None, "size": None
                }
        case_count = references.execute(
            "SELECT value FROM metadata WHERE key='case_count'"
        ).fetchone()[0]
    ordered = [entries[key] for key in sorted(entries)]
    return {
        "schema": "hotpot-ds-intake-closure/current",
        "dataset": manifest["ds_name"],
        "release": manifest["release_id"],
        "case_count": case_count,
        "entries": ordered,
        "closure_digest": _closure_digest(ordered),
    }


def _observe_bundle_closure(root: Path, recorded: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_json(_bundle_path(root, Path("bundle.json")))
    intake = _bundle_path(root, _safe_relative(manifest["intake"]["path"]))
    declared = {entry["path"]: entry for entry in recorded["entries"]}
    discovered = {"bundle.json", str(_safe_relative(manifest["intake"]["path"]))}
    discovered.update(
        str(_safe_relative(item["path"])) for item in manifest.get("control_assets", [])
    )
    with _referenced_paths(intake) as references:
        discovered.update(row[0] for row in references.execute("SELECT path FROM paths"))
        case_count = references.execute(
            "SELECT value FROM metadata WHERE key='case_count'"
        ).fetchone()[0]
    if discovered != set(declared):
        raise RebuildError(f"intake closure membership changed: {root}")
    entries = []
    stamps: dict[str, tuple[int, ...]] = {}
    for relative in sorted(discovered):
        path = _bundle_path(root, _safe_relative(relative))
        if path.exists():
            digest, size, stamp = _sha256_file(path)
            stamps[relative] = stamp
            entry = {"path": relative, "state": "present", "sha256": digest, "size": size}
        else:
            entry = {"path": relative, "state": "missing", "sha256": None, "size": None}
        entries.append(entry)
    result = {
        "schema": recorded["schema"],
        "dataset": recorded["dataset"],
        "release": recorded["release"],
        "case_count": case_count,
        "entries": entries,
        "closure_digest": _closure_digest(entries),
    }
    if result != recorded:
        raise RebuildError(f"intake closure no longer matches recorded baseline: {root}")
    for relative, stamp in stamps.items():
        path = _bundle_path(root, _safe_relative(relative))
        try:
            current = _stat_stamp(path.stat(follow_symlinks=False))
        except FileNotFoundError as error:
            raise RebuildError(f"intake closure changed during proof: {path}") from error
        if current != stamp:
            raise RebuildError(f"intake closure changed during proof: {path}")
    return result


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=60.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _bundle_policy(root: Path, spec: DatasetSpec) -> dict[str, Any]:
    manifest = _read_json(_bundle_path(root, Path("bundle.json")))
    if (
        manifest.get("ds_name") != spec.name
        or manifest.get("release_id") != spec.release
    ):
        raise RebuildError(f"Bundle policy identity differs: {root}")
    projection = _bundle_path(root, Path("policy/policy.json"))
    value, projection_digest, projection_size, projection_stamp = _read_stable_json(
        projection
    )
    facts = _bundle_path(root, Path("cases/facts.sqlite3"))
    with _readonly_connection(facts) as connection:
        policy_row = connection.execute(
            "SELECT policy_json FROM policy WHERE singleton=1"
        ).fetchone()
        control_row = connection.execute(
            "SELECT policy_revision FROM control WHERE singleton=1"
        ).fetchone()
    if policy_row is None or control_row is None:
        raise RebuildError(f"source Bundle policy is missing: {spec.name}")
    stored_value = json.loads(policy_row[0])
    if stored_value != value:
        raise RebuildError(f"Bundle policy projection differs from facts: {spec.name}")
    if not isinstance(value, dict):
        raise RebuildError(f"source Bundle policy is invalid: {spec.name}")
    try:
        from hotpot.ds.gates._persistence import deserialize_gate_configs

        deserialize_gate_configs(value["gates"])
    except (KeyError, TypeError, ValueError) as error:
        raise RebuildError(
            f"source Bundle Gate policy is invalid: {spec.name}"
        ) from error
    if _stat_stamp(projection.stat(follow_symlinks=False)) != projection_stamp:
        raise RebuildError(f"Bundle policy changed while reading: {spec.name}")
    return {
        "revision": int(control_row[0]),
        "digest": _json_digest(value),
        "projection_sha256": projection_digest,
        "projection_size": projection_size,
        "value": value,
    }


def _source_bundle_policy(root: Path, spec: DatasetSpec) -> dict[str, Any]:
    """Read Gate authority from the migration Bundle, never the old target."""

    _assert_schema_v2_source(root, spec)
    return _bundle_policy(root, spec)


def _assert_schema_v2_source(root: Path, spec: DatasetSpec) -> None:
    """Bind this operator to the three completed Stage 13 migration roots."""

    manifest = _read_json(_bundle_path(root, Path("bundle.json")))
    if (
        manifest.get("schema") != "hotpot-ds-bundle/current"
        or manifest.get("ds_name") != spec.name
        or manifest.get("release_id") != spec.release
        or not isinstance(manifest.get("provenance", {}).get("legacy"), Mapping)
    ):
        raise RebuildError(f"source is not the expected legacy migration Bundle: {root}")
    facts = _bundle_path(root, Path("cases/facts.sqlite3"))
    with _readonly_connection(facts) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        stored = connection.execute(
            "SELECT value_json FROM metadata WHERE key='manifest'"
        ).fetchone()
    if user_version != 2 or tables != {
        "audit_events", "cases", "control", "metadata", "policy"
    }:
        raise RebuildError(f"source facts are not the expected schema-v2 layout: {facts}")
    stored_manifest = None if stored is None else json.loads(stored[0])
    for value in (manifest, stored_manifest):
        if isinstance(value, dict) and isinstance(value.get("control_assets"), list):
            value["control_assets"] = sorted(
                value["control_assets"], key=lambda item: (item["category"], item["name"])
            )
    if stored_manifest != manifest:
        raise RebuildError(f"source facts manifest differs from bundle.json: {root}")
    for filename, phase in (
        ("migration-export.json", "export"),
        ("migration-parity.json", "source-export-parity"),
    ):
        evidence = _read_json(_bundle_path(root, Path(filename)))
        if (
            evidence.get("schema") != "hotpot-ds-legacy-migration/current"
            or evidence.get("phase") != phase
            or evidence.get("dataset") != spec.name
            or evidence.get("release") != spec.release
            or (filename == "migration-parity.json" and evidence.get("passed") is not True)
        ):
            raise RebuildError(f"source migration proof is invalid: {root / filename}")


def _migration_proof_binding(
    root: Path, spec: DatasetSpec, closure: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and bind the complete legacy migration admission proof."""

    _assert_schema_v2_source(root, spec)
    manifest = _read_json(_bundle_path(root, Path("bundle.json")))
    export_path = _bundle_path(root, Path("migration-export.json"))
    parity_path = _bundle_path(root, Path("migration-parity.json"))
    export, export_digest, export_size, export_stamp = _read_stable_json(export_path)
    parity, parity_digest, parity_size, parity_stamp = _read_stable_json(parity_path)
    entries = {item["path"]: item for item in closure["entries"]}
    intake_path = str(_safe_relative(manifest["intake"]["path"]))
    intake = entries.get(intake_path)
    if (
        intake is None
        or intake["state"] != "present"
        or export.get("cases_sha256") != intake["sha256"]
        or export.get("case_count") != closure["case_count"]
        or parity.get("case_count") != closure["case_count"]
        or parity.get("unique_case_ids") != closure["case_count"]
    ):
        raise RebuildError(f"migration Case proof does not bind the intake: {root}")
    if (
        parity.get("stable_source_snapshot") is not True
        or parity.get("control_bindings_valid") is not True
        or parity.get("identity_inventory_coverage_complete") is not True
        or parity.get("identity_new_ids_valid") is not True
        or parity.get("persistent_indices_contiguous") is not True
        or parity.get("mismatches") != {}
        or parity.get("case_mismatches") != []
        or parity.get("identity_file_mismatches") != []
        or parity.get("source_inventory") != export.get("source_inventory")
        or parity.get("export_inventory") != export.get("source_inventory")
    ):
        raise RebuildError(f"migration parity content is not a passing proof: {root}")
    legacy = manifest["provenance"]["legacy"]
    if parity.get("source_database_sha256") != legacy.get("database_sha256"):
        raise RebuildError(f"migration proof has another legacy database: {root}")
    for name in ("identity_inventory", "identity_migration"):
        descriptor = export.get(name)
        if not isinstance(descriptor, Mapping):
            raise RebuildError(f"migration export is missing {name}: {root}")
        relative = str(_safe_relative(descriptor["path"]))
        observed = entries.get(relative)
        if observed is None or (
            observed["state"], observed["sha256"], observed["size"]
        ) != ("present", descriptor.get("sha256"), descriptor.get("size")):
            raise RebuildError(f"migration identity proof does not bind {relative}")

    facts = _bundle_path(root, Path("cases/facts.sqlite3"))
    with _readonly_connection(facts) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RebuildError(f"source facts failed SQLite integrity_check: {facts}")
        facts_case_count = connection.execute("SELECT count(*) FROM cases").fetchone()[0]
    if facts_case_count != closure["case_count"]:
        raise RebuildError(f"schema-v2 facts Case count differs from intake: {root}")

    facts_digest, facts_size, facts_stamp = _sha256_file(facts)
    files = {
        "facts": {"sha256": facts_digest, "size": facts_size},
        "migration_export": {"sha256": export_digest, "size": export_size},
        "migration_parity": {"sha256": parity_digest, "size": parity_size},
    }
    for path, stamp in (
        (facts, facts_stamp),
        (export_path, export_stamp),
        (parity_path, parity_stamp),
    ):
        if _stat_stamp(path.stat(follow_symlinks=False)) != stamp:
            raise RebuildError(f"legacy proof changed during aggregate binding: {path}")
    result = {
        "schema": "hotpot-ds-schema-v2-source-proof/current",
        "dataset": spec.name,
        "release": spec.release,
        "case_count": closure["case_count"],
        "cases_sha256": intake["sha256"],
        "files": files,
        "source_inventory_sha256": _json_digest(export["source_inventory"]),
    }
    result["digest"] = _json_digest(result)
    return result


def _without_runtime(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(dict(policy)))
    value.pop("runtime_policy", None)
    return value


def _bundle_projection_rows(
    root: Path, spec: DatasetSpec
) -> Iterator[dict[str, Any]]:
    """Stream the permanent projection of one current synchronized Bundle."""

    from hotpot.ds.public._bundle_state import BundleState
    from hotpot.ds.public._generation import AssessmentGeneration
    from hotpot.ds.public.bundle import (
        _FileDigestCache,
        _target_scientific_asset_descriptors,
        _target_structure_descriptors,
    )
    from hotpot.ds.storage import TargetStore, _restored_facts

    state = BundleState.open(root, spec.name)
    status = state.status
    generation_id = status.get("current_generation_id")
    generation_digest = status.get("current_generation_digest")
    if not generation_id or not generation_digest:
        raise RebuildError(f"Bundle has no synchronized assessment: {root}")
    generation = AssessmentGeneration.open(root, generation_id, generation_digest)
    facts = state.iter_cases()
    file_cache = _FileDigestCache()
    missing = object()
    try:
        with generation._reader() as reader:
            assessments = reader.iter_cases()
            for case, assessed in zip_longest(facts, assessments, fillvalue=missing):
                if case is missing or assessed is missing:
                    raise RebuildError(f"Bundle fact/assessment count differs: {root}")
                assessment_id, assessment = assessed
                if case["id"] != assessment_id:
                    raise RebuildError(f"Bundle fact/assessment order differs: {root}")
                structures = _target_structure_descriptors(
                    root, case, file_cache=file_cache
                )
                assets = _target_scientific_asset_descriptors(
                    root, case, file_cache=file_cache
                )
                normalized = TargetStore._normalize_case(
                    {
                        "id": case["id"],
                        "public_index": case["public_index"],
                        "local_id": case["local_id"],
                        "facts": case,
                        "assessment": assessment,
                        "identifiers": [
                            {"namespace": str(namespace), "value": str(value)}
                            for namespace, raw in case.get("identifiers", {}).items()
                            for value in (raw if isinstance(raw, list) else [raw])
                        ],
                        "structures": structures,
                        "assets": assets,
                    }
                )
                identifiers = normalized["identifiers"]
                public = {
                    "id": normalized["id"],
                    "public_index": normalized["public_index"],
                    "local_id": normalized["local_id"],
                    "facts": _restored_facts(
                        normalized["facts"],
                        case_id=normalized["id"],
                        public_index=normalized["public_index"],
                        local_id=normalized["local_id"],
                        identifiers=identifiers,
                    ),
                    "assessment": normalized["assessment"],
                    "identifiers": identifiers,
                    "structures": _structure_projection(normalized["structures"]),
                }
                metadata = case.get("metadata", {})
                yield {
                    "content": TargetStore._case_content_projection(normalized),
                    "public": public,
                    "omitted_migration_fields": sorted(
                        set(metadata)
                        & {
                            "migration_rows",
                            "migration_schema",
                            "quarantined_entity_relations",
                            "shared_identifier_relations",
                        }
                    ) if isinstance(metadata, Mapping) else [],
                    "omitted_assessment_fields": sorted(
                        set(assessment)
                        - {
                            "quality_level",
                            "evidence_level",
                            "archive_class",
                            "benchmark",
                            "benchmark_status",
                            "reasons",
                        }
                    ),
                    "scientific_assets": _scientific_asset_projection(
                        normalized["assets"]
                    ),
                }
    finally:
        close = getattr(facts, "close", None)
        if close is not None:
            close()


def _source_intake_projection_rows(
    root: Path, spec: DatasetSpec
) -> Iterator[dict[str, Any]]:
    """Stream compact facts from the immutable Stage 13 JSONL intake.

    The source Bundle is deliberately an older working-state schema.  This
    one-time rebuild reads its declared original intake rather than adding an
    old Bundle schema reader to the package runtime.
    """

    from hotpot.ds.public.bundle import (
        _FileDigestCache,
        _target_scientific_asset_descriptors,
        _target_structure_descriptors,
    )
    from hotpot.ds.public._bundle_state import _validate_case
    from hotpot.ds.storage import TargetStore, _restored_facts

    manifest = _read_json(root / "bundle.json")
    intake = manifest["intake"]
    if intake.get("format") != "jsonl":
        raise RebuildError(f"Stage 13 source intake is not JSONL: {root}")
    path = _bundle_path(root, _safe_relative(intake["path"]))
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        os.close(descriptor)
        raise RebuildError(f"source intake is not a regular file: {path}")
    file_cache = _FileDigestCache()
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RebuildError(
                        f"invalid source intake JSON at {path}:{line_number}"
                    ) from error
                if not isinstance(case, dict):
                    raise RebuildError(
                        f"source intake Case is not an object at {path}:{line_number}"
                    )
                case = _validate_case(
                    root,
                    case,
                    supplied_public_index=line_number - 1,
                    property_definitions=manifest.get("property_definitions", {}),
                    property_comparators=manifest.get("property_comparators", {}),
                )
                structures = _target_structure_descriptors(
                    root, case, file_cache=file_cache
                )
                assets = _target_scientific_asset_descriptors(
                    root, case, file_cache=file_cache
                )
                normalized = TargetStore._normalize_case(
                    {
                        "id": case["id"],
                        "public_index": case["public_index"],
                        "local_id": case["local_id"],
                        "facts": case,
                        "assessment": {
                            "archive_class": "ordinary",
                            "benchmark": False,
                        },
                        "identifiers": [
                            {"namespace": str(namespace), "value": str(value)}
                            for namespace, raw in case.get("identifiers", {}).items()
                            for value in (raw if isinstance(raw, list) else [raw])
                        ],
                        "structures": structures,
                        "assets": assets,
                    }
                )
                identifiers = normalized["identifiers"]
                metadata = case.get("metadata", {})
                yield {
                    "facts": _restored_facts(
                        normalized["facts"],
                        case_id=normalized["id"],
                        public_index=normalized["public_index"],
                        local_id=normalized["local_id"],
                        identifiers=identifiers,
                    ),
                    "identifiers": identifiers,
                    "structures": _structure_projection(normalized["structures"]),
                    "scientific_assets": _scientific_asset_projection(
                        normalized["assets"]
                    ),
                    "omitted_migration_fields": sorted(
                        set(metadata)
                        & {
                            "migration_rows",
                            "migration_schema",
                            "quarantined_entity_relations",
                            "shared_identifier_relations",
                        }
                    )
                    if isinstance(metadata, Mapping)
                    else [],
                }
        after = os.fstat(descriptor)
        if _stat_stamp(before) != _stat_stamp(after):
            raise RebuildError(f"source intake changed while reading: {path}")
    finally:
        os.close(descriptor)


def _source_assessment_rows(
    root: Path, spec: DatasetSpec
) -> Iterator[dict[str, Any]]:
    """Read the final scalar labels from the sealed schema-v2 source.

    This migration-only reader is intentionally kept outside ``hotpot.ds``;
    the package itself supports only the single current Bundle schema.
    """

    facts = _bundle_path(root, Path("cases/facts.sqlite3"))
    with _readonly_connection(facts) as connection:
        control = connection.execute(
            "SELECT current_generation_id,current_generation_digest "
            "FROM control WHERE singleton=1"
        ).fetchone()
    if control is None or not control[0] or not control[1]:
        raise RebuildError(f"source assessment generation is missing: {spec.name}")
    generation_id = str(control[0])
    generation_digest = str(control[1])
    generation_root = _bundle_path(
        root, Path("assessments") / "generations" / _safe_relative(generation_id)
    )
    manifest_path = _bundle_path(generation_root, Path("generation.json"))
    manifest, _, _, manifest_stamp = _read_stable_json(manifest_path)
    if (
        manifest.get("generation_id") != generation_id
        or manifest.get("generation_digest") != generation_digest
    ):
        raise RebuildError(f"source assessment generation binding differs: {spec.name}")
    logical_fields = (
        "schema",
        "source",
        "policy",
        "governance",
        "runtime_policy",
        "ordered_assessment_digest",
        "summary_digest",
        "files",
    )
    if any(field not in manifest for field in logical_fields):
        raise RebuildError(f"source assessment manifest is incomplete: {spec.name}")
    computed_digest = _json_digest(
        {field: manifest[field] for field in logical_fields}
    )
    if (
        computed_digest != generation_digest
        or generation_id != "generation-" + generation_digest[:24]
    ):
        raise RebuildError(
            f"source assessment logical digest differs: {spec.name}"
        )
    declared = {}
    for item in manifest.get("files", []):
        relative = _safe_relative(item["path"])
        path = _bundle_path(generation_root, relative)
        digest, size, stamp = _sha256_file(path)
        if (digest, size) != (item.get("sha256"), item.get("size")):
            raise RebuildError(f"source assessment artifact differs: {path}")
        declared[str(relative)] = (path, stamp)
    if "assessment.sqlite3" not in declared:
        raise RebuildError(f"source assessment database is undeclared: {spec.name}")
    assessment_path, assessment_stamp = declared["assessment.sqlite3"]
    with _readonly_connection(assessment_path) as connection:
        if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise RebuildError(
                f"source assessment is not the expected schema-v1 layout: {spec.name}"
            )
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RebuildError(f"source assessment failed integrity_check: {spec.name}")
        for row in connection.execute(
            "SELECT case_id,public_index,quality_level,evidence_level,"
            "archive_class,benchmark,assessment_json FROM case_assessments "
            "ORDER BY public_index,case_id"
        ):
            assessment = json.loads(row[6])
            expected = {
                "quality_level": row[2],
                "evidence_level": row[3],
                "archive_class": row[4],
                "benchmark": bool(row[5]),
            }
            if any(assessment.get(key) != value for key, value in expected.items()):
                raise RebuildError(
                    f"source assessment scalar projection differs: {spec.name}.{row[0]}"
                )
            yield {
                "id": str(row[0]),
                "public_index": int(row[1]),
                "assessment": expected,
            }
    if _stat_stamp(assessment_path.stat(follow_symlinks=False)) != assessment_stamp:
        raise RebuildError(f"source assessment changed while reading: {spec.name}")
    if _stat_stamp(manifest_path.stat(follow_symlinks=False)) != manifest_stamp:
        raise RebuildError(
            f"source assessment manifest changed while reading: {spec.name}"
        )


def _structure_projection(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "structure_id": str(item["structure_id"]),
                "format": str(item["format"]),
                "role": str(item["role"]),
                "primary": bool(item["primary"]),
                "sha256": str(item["sha256"]),
                "size": int(item["size"]),
            }
            for item in values
        ),
        key=lambda item: (
            item["structure_id"], item["format"], item["role"], item["sha256"]
        ),
    )


def _scientific_asset_projection(
    values: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "collection": str(item["collection"]),
                "declaration_id": str(item["declaration_id"]),
                "asset_index": int(item["asset_index"]),
                "role": str(item["role"]),
                "status": str(item["status"]),
                "sha256": str(item["sha256"]),
                "size": int(item["size"]),
            }
            for item in values
        ),
        key=lambda item: (
            item["collection"], item["declaration_id"], item["asset_index"]
        ),
    )


def _target_scientific_assets(
    database: Path, spec: DatasetSpec
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with _readonly_connection(database) as connection:
        rows = connection.execute(
            "SELECT r.case_id,r.collection,r.declaration_id,r.asset_index,r.role,"
            "r.status,r.content_sha256,t.size FROM scientific_asset_refs r "
            "JOIN target_assets t USING(content_sha256) WHERE r.dataset_id=? "
            "AND r.release_id=? ORDER BY r.case_id,r.collection,r.declaration_id,"
            "r.asset_index",
            (spec.name, spec.release),
        )
        for row in rows:
            result.setdefault(str(row[0]), []).append(
                {
                    "collection": str(row[1]),
                    "declaration_id": str(row[2]),
                    "asset_index": int(row[3]),
                    "role": str(row[4]),
                    "status": str(row[5]),
                    "sha256": str(row[6]),
                    "size": int(row[7]),
                }
            )
    return result


def _target_case_projection(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "public_index": case["public_index"],
        "local_id": case["local_id"],
        "facts": case["facts"],
        "assessment": case["assessment"],
        "identifiers": case["identifiers"],
        "structures": _structure_projection(case["structures"]),
    }


def _compare_durable_cases(
    source_root: Path,
    output_root: Path,
    store: Any,
    spec: DatasetSpec,
) -> dict[str, Any]:
    """Compare source, rebuilt Bundle, and public formal-store reads."""

    from hotpot.ds.storage import case_set_digest

    source = _source_intake_projection_rows(source_root, spec)
    source_assessments = _source_assessment_rows(source_root, spec)
    output = _bundle_projection_rows(output_root, spec)
    target = store.iter_cases(spec.name, spec.release, include_incomplete=True)
    missing = object()
    content_digest = hashlib.sha256()
    content_digest.update(b"[")
    first = True
    case_ids = []
    migration_fields: set[str] = set()
    assessment_fields: set[str] = set()
    target_assets = _target_scientific_assets(store.path, spec)
    count = 0
    for source_row, source_assessment, output_row, target_case in zip_longest(
        source, source_assessments, output, target, fillvalue=missing
    ):
        if missing in (source_row, source_assessment, output_row, target_case):
            raise RebuildError(f"durable Case count differs: {spec.name}")
        output_facts = output_row["public"]
        if (
            source_row["facts"] != output_facts["facts"]
            or source_row["identifiers"] != output_facts["identifiers"]
        ):
            raise RebuildError(
                f"retained source/output Case facts differ: {spec.name}."
                f"{output_row['public']['id']}"
            )
        if (
            source_assessment["id"] != output_row["public"]["id"]
            or source_assessment["public_index"]
            != output_row["public"]["public_index"]
            or source_assessment["assessment"]
            != {
                key: output_row["public"]["assessment"].get(key)
                for key in (
                    "quality_level",
                    "evidence_level",
                    "archive_class",
                    "benchmark",
                )
            }
        ):
            raise RebuildError(
                f"source/output final assessment differs: {spec.name}."
                f"{output_row['public']['id']}"
            )
        if source_row["structures"] != output_row["public"]["structures"]:
            raise RebuildError(
                f"source/output structure projection differs: {spec.name}."
                f"{output_row['public']['id']}"
            )
        if source_row["scientific_assets"] != output_row["scientific_assets"]:
            raise RebuildError(
                f"source/output scientific assets differ: {spec.name}."
                f"{output_row['public']['id']}"
            )
        case_id = str(output_row["public"]["id"])
        if output_row["scientific_assets"] != target_assets.pop(case_id, []):
            raise RebuildError(
                f"output/target scientific assets differ: {spec.name}.{case_id}"
            )
        target_public = _target_case_projection(target_case)
        if output_row["public"] != target_public:
            raise RebuildError(
                f"output/target durable Case differs: {spec.name}."
                f"{output_row['public']['id']}"
            )
        metadata = target_public["facts"].get("metadata", {})
        if isinstance(metadata, Mapping) and set(metadata) & {
            "migration_rows",
            "migration_schema",
            "quarantined_entity_relations",
            "shared_identifier_relations",
        }:
            raise RebuildError(f"target retained migration projection: {spec.name}")
        if set(target_public["assessment"]) - {
            "quality_level",
            "evidence_level",
            "archive_class",
            "benchmark",
            "benchmark_status",
            "reasons",
        }:
            raise RebuildError(f"target retained assessment traces: {spec.name}")
        migration_fields.update(source_row["omitted_migration_fields"])
        assessment_fields.update(output_row["omitted_assessment_fields"])
        payload = _canonical_bytes(output_row["content"])
        if not first:
            content_digest.update(b",")
        content_digest.update(payload)
        first = False
        case_ids.append(str(output_row["public"]["id"]))
        count += 1
    if target_assets:
        raise RebuildError(f"target contains orphan scientific Case refs: {spec.name}")
    content_digest.update(b"]")
    return {
        "count": count,
        "case_content_sha256": content_digest.hexdigest(),
        "case_set_sha256": case_set_digest(case_ids),
        "omitted_migration_fields": sorted(migration_fields),
        "omitted_assessment_fields": sorted(assessment_fields),
        "retained_facts_compared_in_full": True,
        "original_intake_compared_after_declared_compaction": True,
        "original_facts_compared_in_full": False,
        "source_final_assessments_compared": True,
        "public_target_reads_compared": True,
    }


def _release_asset_inventory_digest(database: Path, spec: DatasetSpec) -> str:
    from hotpot.ds.storage import asset_inventory_digest

    with _readonly_connection(database) as connection:
        rows = connection.execute(
            "WITH refs(content_sha256) AS ("
            "SELECT content_sha256 FROM structure_asset_refs WHERE dataset_id=? AND release_id=? "
            "UNION SELECT content_sha256 FROM scientific_asset_refs "
            "WHERE dataset_id=? AND release_id=? "
            "UNION SELECT content_sha256 FROM release_control_asset_refs "
            "WHERE dataset_id=? AND release_id=?"
            ") SELECT t.content_sha256,t.size FROM refs "
            "JOIN target_assets t USING(content_sha256) "
            "ORDER BY t.content_sha256",
            (spec.name, spec.release) * 3,
        )
        return asset_inventory_digest(
            {"sha256": row[0], "size": row[1]} for row in rows
        )


def _validate_release_control_assets(
    root: Path, database: Path, spec: DatasetSpec
) -> dict[str, Any]:
    from hotpot.ds.public._bundle_state import BundleState
    from hotpot.ds.public._generation import AssessmentGeneration
    from hotpot.ds.public.bundle import _target_control_asset_descriptors

    state = BundleState.open(root, spec.name)
    status = state.status
    generation = AssessmentGeneration.open(
        root,
        status["current_generation_id"],
        status["current_generation_digest"],
    )
    expected = [
        {
            "category": str(item["category"]),
            "name": str(item["name"]),
            "sha256": str(item["sha256"]),
            "size": int(item["size"]),
        }
        for item in _target_control_asset_descriptors(
            root, generation, state.manifest
        )
    ]
    with _readonly_connection(database) as connection:
        observed = [
            {
                "category": str(row[0]),
                "name": str(row[1]),
                "sha256": str(row[2]),
                "size": int(row[3]),
            }
            for row in connection.execute(
                "SELECT r.category,r.name,r.content_sha256,t.size "
                "FROM release_control_asset_refs r JOIN target_assets t "
                "USING(content_sha256) WHERE r.dataset_id=? AND r.release_id=? "
                "ORDER BY r.category,r.name",
                (spec.name, spec.release),
            )
        ]
    if expected != observed:
        raise RebuildError(f"candidate durable control assets differ: {spec.name}")
    required_provenance = {
        ("provenance", "legacy-source-inventory"),
        ("provenance", "migration-provenance"),
    }
    observed_identities = {
        (item["category"], item["name"]) for item in observed
    }
    if not required_provenance <= observed_identities:
        raise RebuildError(
            f"candidate omitted durable migration provenance: {spec.name}"
        )
    return {"count": len(observed), "sha256": _json_digest(observed)}


def _validate_target_receipt(
    store: Any, database: Path, spec: DatasetSpec, cases: Mapping[str, Any]
) -> dict[str, Any]:
    from hotpot.ds.storage import (
        persisted_formal_content_digest,
        persisted_included_case_content_digest,
    )

    release = store.load_release_metadata(spec.name, spec.release)
    receipt = None if release is None else release.get("receipt")
    if receipt is None:
        raise RebuildError(f"candidate receipt is missing: {spec.name}")
    with store._connect() as connection:
        included_content = persisted_included_case_content_digest(
            connection, spec.name, spec.release
        )
        formal_content = persisted_formal_content_digest(
            connection, spec.name, spec.release
        )
    expected = {
        "case_content_digest": cases["case_content_sha256"],
        "included_case_content_digest": included_content,
        "formal_content_digest": formal_content,
        "included_cases_digest": cases["case_set_sha256"],
        "excluded_cases_digest": _stream_json_array_digest(()),
        "asset_inventory_digest": _release_asset_inventory_digest(database, spec),
        "include_incomplete": True,
        "reason": INCOMPLETE_REASON,
        "included_count": cases["count"],
        "excluded_count": 0,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise RebuildError(
                f"candidate receipt disagrees with durable {field}: {spec.name}"
            )
    binding = receipt["binding"]
    for field in (
        "store_uuid", "bundle_digest", "certificate_digest", "policy_digest",
        "generation_digest", "case_content_digest",
        "included_case_content_digest", "formal_content_digest",
        "included_cases_digest",
        "excluded_cases_digest", "asset_inventory_digest", "include_incomplete", "reason",
    ):
        if binding.get(field) != receipt.get(field):
            raise RebuildError(
                f"candidate receipt binding disagrees with {field}: {spec.name}"
            )
    if binding.get("dataset") != spec.name or binding.get("release") != spec.release:
        raise RebuildError(f"candidate receipt binding identity is invalid: {spec.name}")
    if receipt["receipt_id"] != "receipt-" + _json_digest(binding):
        raise RebuildError(f"candidate receipt identity is invalid: {spec.name}")
    return {
        "included_count": receipt["included_count"],
        "excluded_count": receipt["excluded_count"],
        "binding_sha256": _json_digest(binding),
        "asset_inventory_sha256": receipt["asset_inventory_digest"],
    }


def _database_semantic_parity(
    source_bundles: Path,
    output_bundles: Path,
    target_db: Path,
    specs: Sequence[DatasetSpec],
) -> dict[str, Any]:
    from hotpot.ds.public._implementation import runtime_policy
    from hotpot.ds.storage import TARGET_SCHEMA_ID, TargetStore

    store = TargetStore(target_db)
    if store.metadata()["schema_id"] != TARGET_SCHEMA_ID:
        raise RebuildError("candidate is not the compact-current target schema")
    expected_scope = sorted((item.name, item.release) for item in specs)
    with _readonly_connection(target_db) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RebuildError("candidate store failed SQLite integrity_check")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RebuildError("candidate store failed SQLite foreign_key_check")
        scope = [
            tuple(row) for row in connection.execute(
                "SELECT dataset_id,release_id FROM releases ORDER BY dataset_id,release_id"
            )
        ]
        datasets = [
            str(row[0])
            for row in connection.execute(
                "SELECT dataset_id FROM datasets ORDER BY dataset_id"
            )
        ]
        policies = [
            str(row[0])
            for row in connection.execute(
                "SELECT dataset_id FROM dataset_policies ORDER BY dataset_id"
            )
        ]
        receipts = [
            tuple(row)
            for row in connection.execute(
                "SELECT dataset_id,release_id FROM ingestion_receipts "
                "ORDER BY dataset_id,release_id"
            )
        ]
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        case_columns = {row[1] for row in connection.execute("PRAGMA table_info(cases)")}
        assessment_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(case_assessments)")
        }
    if scope != expected_scope:
        raise RebuildError("candidate store scope is not exactly the expected releases")
    expected_datasets = sorted(item.name for item in specs)
    if (
        datasets != expected_datasets
        or policies != expected_datasets
        or receipts != expected_scope
    ):
        raise RebuildError("candidate dataset/policy/receipt scope is not exact")
    if "source_payload_json" in case_columns or "assessment_json" in assessment_columns:
        raise RebuildError("candidate retained superseded JSON trace columns")
    if tables & {"ordinary_case_index", "benchmark_case_index", "incomplete_case_index"}:
        raise RebuildError("candidate retained reproducible archive index tables")

    result: dict[str, Any] = {"datasets": {}}
    for spec in specs:
        source_root = source_bundles / spec.source_relative
        output_root = output_bundles / spec.name / spec.release
        source_policy = _source_bundle_policy(source_root, spec)
        output_policy = _bundle_policy(output_root, spec)
        target_policy = store.read_policy(spec.name)
        if target_policy is None:
            raise RebuildError(f"candidate policy is missing: {spec.name}")
        expected_policy = _without_runtime(source_policy["value"])
        if (
            _without_runtime(output_policy["value"]) != expected_policy
            or _without_runtime(target_policy["value"]) != expected_policy
        ):
            raise RebuildError(f"durable Gate policy differs: {spec.name}")
        if target_policy["value"].get("runtime_policy") != runtime_policy():
            raise RebuildError(f"candidate policy is not current: {spec.name}")

        manifest = _read_json(output_root / "bundle.json")
        dataset = store.inspect_dataset(spec.name)
        release = store.load_release_metadata(spec.name, spec.release)
        if dataset is None or release is None:
            raise RebuildError(f"candidate public metadata is missing: {spec.name}")
        expected_metadata = {
            "license": manifest["license"],
            "access_policy": manifest["access_policy"],
            "provenance": manifest["provenance"],
            "core_labels": manifest["core_labels"],
            "core_info": manifest["core_info"],
        }
        if (
            dataset["title"] != manifest["title"]
            or dataset["description"] != manifest["description"]
            or release["version"] != manifest["version"]
            or release["metadata"] != expected_metadata
        ):
            raise RebuildError(f"candidate release metadata differs: {spec.name}")

        cases = _compare_durable_cases(source_root, output_root, store, spec)
        result["datasets"][spec.name] = {
            "cases": cases,
            "policy_without_runtime": {
                "source_revision": source_policy["revision"],
                "sha256": _json_digest(expected_policy),
            },
            "control_assets": _validate_release_control_assets(
                output_root, target_db, spec
            ),
            "receipt": _validate_target_receipt(store, target_db, spec, cases),
        }
    result["semantic_digest"] = _json_digest(result["datasets"])
    return result


def _verify_cas(database: Path) -> dict[str, Any]:
    asset_root = Path(str(database) + ".assets")
    with _readonly_connection(database) as connection:
        orphan = connection.execute(
            """
            WITH refs(content_sha256) AS (
              SELECT content_sha256 FROM structure_asset_refs
              UNION
              SELECT content_sha256 FROM scientific_asset_refs
              UNION
              SELECT content_sha256 FROM release_control_asset_refs
            )
            SELECT target_assets.content_sha256
            FROM target_assets
            LEFT JOIN refs USING(content_sha256)
            WHERE refs.content_sha256 IS NULL
            LIMIT 1
            """
        ).fetchone()
        if orphan is not None:
            raise RebuildError(f"CAS database contains an unreferenced object: {orphan[0]}")
        control_categories = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT category FROM release_control_asset_refs"
            )
        }
        unsupported = control_categories - _DURABLE_CONTROL_CATEGORIES
        if unsupported:
            raise RebuildError(
                "CAS retained reproducible control categories: "
                + ", ".join(sorted(unsupported))
            )
        rows = connection.execute(
            "SELECT content_sha256,size,relative_path FROM target_assets ORDER BY relative_path"
        )
        digest = hashlib.sha256()
        expected_paths = set()
        observed_assets = set()
        count = 0
        byte_count = 0
        for content_sha256, size, relative_path in rows:
            relative = _safe_relative(relative_path)
            canonical_relative = (
                f"objects/sha256/{content_sha256[:2]}/{content_sha256}"
            )
            if relative_path != canonical_relative:
                raise RebuildError(
                    f"CAS object has a non-canonical path: {relative_path}"
                )
            path = asset_root / relative
            observed_sha256, observed_size, _ = _sha256_file(path)
            if observed_sha256 != content_sha256 or observed_size != size:
                raise RebuildError(f"CAS object does not match database: {path}")
            item = {"sha256": content_sha256, "size": size, "path": relative_path}
            encoded = _canonical_bytes(item)
            framed = len(encoded).to_bytes(8, "big") + encoded
            digest.update(framed)
            expected_paths.add(relative_path)
            observed_assets.add(content_sha256)
            count += 1
            byte_count += size
    actual_paths = set()
    lock_count = 0
    if asset_root.exists():
        for path in asset_root.rglob("*"):
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode):
                relative_path = path.relative_to(asset_root)
                lock_match = _CAS_LOCK_NAME.fullmatch(path.name)
                if lock_match is not None:
                    content_sha256 = lock_match.group(1)
                    expected_parent = Path("objects") / "sha256" / content_sha256[:2]
                    if (
                        relative_path.parent != expected_parent
                        or content_sha256 not in observed_assets
                        or metadata.st_size != 0
                    ):
                        raise RebuildError(f"CAS contains an invalid lock node: {path}")
                    lock_count += 1
                    continue
                actual_paths.add(relative_path.as_posix())
            elif not stat.S_ISDIR(metadata.st_mode):
                raise RebuildError(f"CAS contains a non-regular node: {path}")
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)[:3]
        extra = sorted(actual_paths - expected_paths)[:3]
        raise RebuildError(f"CAS database/filesystem mismatch; missing={missing}, extra={extra}")
    return {
        "count": count,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "lock_files": lock_count,
        "durable_only": True,
        "control_categories": sorted(control_categories),
    }


class Operator:
    def __init__(self, repository: Path, run_id: str, source_generation: str):
        self.repository = repository.resolve()
        self.run_id = _safe_component(run_id, "run ID")
        self.source_generation = _safe_component(
            source_generation, "source generation"
        )
        self.data_root = self.repository / "hotpot" / "dataset" / "data"
        self.store_root = self.data_root / "store"
        self.source_generation_root = (
            self.store_root / "generations" / self.source_generation
        )
        self.source_database = self.source_generation_root / "datasets.sqlite3"
        self.source_bundles = self.data_root / "bundles" / "stage13-migration"
        self.output_bundles = self.data_root / "bundles" / self.run_id
        self.output_generation = self.store_root / "generations" / self.run_id
        self.output_database = self.output_generation / "datasets.sqlite3"
        self.state_root = self.store_root / "rebuilds" / self.run_id
        self.state_path = self.state_root / "state.json"
        self.events_path = self.state_root / "events.jsonl"
        self.lock_path = self.store_root / ".rebuild.lock"

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        _assert_managed_path(
            self.repository, self.store_root, leaf_kind="directory"
        )
        self.store_root.mkdir(parents=True, exist_ok=True)
        _assert_managed_path(
            self.repository, self.store_root, leaf_kind="directory"
        )
        _assert_managed_path(self.repository, self.lock_path, leaf_kind="file")
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as error:
            raise RebuildError(f"cannot open regular rebuild lock: {self.lock_path}") from error
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise RebuildError(f"rebuild lock is not regular: {self.lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @contextlib.contextmanager
    def _publication_barrier(self) -> Iterator[None]:
        """Exclude cooperative Bundle and SQLite writers through link publication."""

        lock_descriptors: list[int] = []
        connections: list[sqlite3.Connection] = []
        roots = [
            *(self.source_bundles / spec.source_relative for spec in DATASETS),
            *(self._output_root(spec) for spec in DATASETS),
        ]
        try:
            for root in sorted(roots):
                lock_path = root / ".bundle.lock"
                descriptor = os.open(lock_path, os.O_RDWR)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                lock_descriptors.append(descriptor)
            for database in (self.source_database, self.output_database):
                connection = sqlite3.connect(str(database), timeout=60.0)
                connections.append(connection)
                connection.execute("PRAGMA busy_timeout=60000")
                connection.execute("BEGIN IMMEDIATE")
            yield
        finally:
            for connection in reversed(connections):
                with contextlib.suppress(sqlite3.Error):
                    connection.rollback()
                connection.close()
            for descriptor in reversed(lock_descriptors):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _assert_output_namespace(self) -> None:
        directories = [
            self.output_bundles,
            self.output_generation,
            Path(str(self.output_database) + ".assets"),
            self.state_root,
        ]
        directories.extend(self._output_root(spec) for spec in DATASETS)
        directories.extend(
            self._output_root(spec).with_name(spec.release + ".partial")
            for spec in DATASETS
        )
        files = [
            self.output_database,
            self.state_path,
            self.events_path,
            self.state_root / "parity.json",
            self.state_root / "activation-intent.json",
        ]
        files.extend(
            self.state_root / f"source-{spec.name}-closure.json"
            for spec in DATASETS
        )
        for path in directories:
            _assert_managed_path(self.repository, path, leaf_kind="directory")
        for path in files:
            _assert_managed_path(self.repository, path, leaf_kind="file")

    def plan(self) -> dict[str, Any]:
        self._assert_output_namespace()
        active = self.store_root / "active"
        if not active.is_symlink():
            raise RebuildError(f"active store is not a directory symlink: {active}")
        active_target = active.resolve()
        if active_target != self.source_generation_root.resolve():
            raise RebuildError(
                f"active store is {active_target}, expected {self.source_generation_root}"
            )
        if not self.source_database.is_file():
            raise RebuildError(f"source database is missing: {self.source_database}")
        datasets = []
        for spec in DATASETS:
            source = self.source_bundles / spec.source_relative
            _assert_schema_v2_source(source, spec)
            manifest = _read_json(source / "bundle.json")
            if manifest.get("ds_name") != spec.name or manifest.get("release_id") != spec.release:
                raise RebuildError(f"source Bundle identity mismatch: {source}")
            policy = _source_bundle_policy(source, spec)
            datasets.append(
                {
                    "dataset": spec.name,
                    "release": spec.release,
                    "source": str(source),
                    "candidate": str(self._output_root(spec)),
                    "source_policy_revision": policy["revision"],
                    "source_policy_without_runtime_sha256": _json_digest(
                        _without_runtime(policy["value"])
                    ),
                }
            )
        return {
            "schema": STATE_SCHEMA,
            "mode": "read-only-plan",
            "run_id": self.run_id,
            "source_generation": str(self.source_generation_root),
            "source_database": str(self.source_database),
            "candidate_generation": str(self.output_generation),
            "candidate_database": str(self.output_database),
            "state": str(self.state_path),
            "lock": str(self.lock_path),
            "datasets": datasets,
            "activation_is_separate": True,
        }

    def _output_root(self, spec: DatasetSpec) -> Path:
        return self.output_bundles / spec.name / spec.release

    def _new_state(self) -> dict[str, Any]:
        now = _utc_now()
        return {
            "schema": STATE_SCHEMA,
            "run_id": self.run_id,
            "created_at": now,
            "updated_at": now,
            "source_generation": str(self.source_generation_root.resolve()),
            "source_database": str(self.source_database.resolve()),
            "candidate_generation": str(self.output_generation.resolve()),
            "candidate_database": str(self.output_database.resolve()),
            "phase_order": [
                *(f"copy:{item.name}" for item in DATASETS),
                *(f"process:{item.name}" for item in DATASETS),
                "parity",
                "ready",
            ],
            "phases": {},
            "activation": {"status": "not-activated"},
        }

    def _load_state(self, *, create: bool = False) -> dict[str, Any]:
        self._assert_output_namespace()
        if not _path_lexists(self.state_path):
            if not create:
                raise RebuildError(f"rebuild state does not exist: {self.state_path}")
            conflicts = [
                path
                for path in (self.state_root, self.output_bundles, self.output_generation)
                if _path_lexists(path)
            ]
            conflicts.extend(
                self._output_root(spec).with_name(spec.release + ".partial")
                for spec in DATASETS
                if _path_lexists(
                    self._output_root(spec).with_name(spec.release + ".partial")
                )
            )
            if conflicts:
                raise RebuildError(
                    "candidate output exists without owned rebuild state: "
                    + ", ".join(str(path) for path in conflicts)
                )
            state = self._new_state()
            _atomic_json(self.state_path, state)
            _append_event(self.events_path, "rebuild-created", run_id=self.run_id)
            return state
        state = _read_json(self.state_path)
        expected = self._new_state()
        for field in (
            "schema", "run_id", "source_generation", "source_database",
            "candidate_generation", "candidate_database", "phase_order",
        ):
            if state.get(field) != expected[field]:
                raise RebuildError(f"persisted rebuild state has different {field}")
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _utc_now()
        _atomic_json(self.state_path, state)

    def _phase(
        self,
        state: dict[str, Any],
        name: str,
        execute: Callable[[], Mapping[str, Any]],
        validate: Callable[[Mapping[str, Any]], None],
    ) -> None:
        recorded = state["phases"].get(name)
        if recorded is not None and recorded.get("status") == "complete":
            validate(recorded["result"])
            _append_event(self.events_path, "phase-resume-validated", phase=name)
            return
        _append_event(self.events_path, "phase-started", phase=name)
        try:
            result = dict(execute())
        except BaseException as error:
            _append_event(
                self.events_path,
                "phase-failed",
                phase=name,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        state["phases"][name] = {
            "status": "complete", "completed_at": _utc_now(), "result": result
        }
        self._save_state(state)
        _append_event(self.events_path, "phase-completed", phase=name, result=result)

    def run(self) -> dict[str, Any]:
        with self.locked():
            self.plan()
            state = self._load_state(create=True)
            for spec in DATASETS:
                self._phase(
                    state,
                    f"copy:{spec.name}",
                    lambda spec=spec: self._copy_phase(spec),
                    lambda result, spec=spec: self._validate_copy_phase(spec, result),
                )
            for spec in DATASETS:
                self._phase(
                    state,
                    f"process:{spec.name}",
                    lambda spec=spec: self._process_phase(spec),
                    lambda result, spec=spec: self._validate_process_phase(spec, result),
                )
            self._phase(state, "parity", self._parity_phase, self._validate_parity_phase)
            self._phase(
                state,
                "ready",
                lambda: {"activation_ready": True, "verified_at": _utc_now()},
                lambda result: self._require(result.get("activation_ready") is True, "candidate is not ready"),
            )
            return self.status(state=state)

    def _copy_phase(self, spec: DatasetSpec) -> dict[str, Any]:
        source = self.source_bundles / spec.source_relative
        output = self._output_root(spec)
        partial = output.with_name(output.name + ".partial")
        artifact = self.state_root / f"source-{spec.name}-closure.json"
        if _path_lexists(output):
            if not artifact.is_file():
                raise RebuildError(f"candidate root exists without its closure proof: {output}")
            closure = _read_json(artifact)
            _observe_bundle_closure(output, closure)
        else:
            partial.mkdir(parents=True, exist_ok=True)
            closure = _copy_bundle_closure(source, partial)
            _atomic_json(artifact, closure)
            output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial, output)
            _fsync_directory(output.parent)
        legacy_proof = _migration_proof_binding(source, spec, closure)
        artifact_digest, artifact_size, _ = _sha256_file(artifact)
        result = {
            "source": str(source.resolve()),
            "candidate": str(output.resolve()),
            "closure_artifact": str(artifact.resolve()),
            "closure_artifact_sha256": artifact_digest,
            "closure_artifact_size": artifact_size,
            "closure_digest": closure["closure_digest"],
            "entry_count": len(closure["entries"]),
            "legacy_proof": legacy_proof,
        }
        self._validate_copy_phase(spec, result, revalidate_proof=False)
        return result

    def _validate_copy_phase(
        self,
        spec: DatasetSpec,
        result: Mapping[str, Any],
        *,
        revalidate_proof: bool = True,
    ) -> None:
        output = self._output_root(spec).resolve()
        artifact = Path(result["closure_artifact"])
        if Path(result["candidate"]) != output or not output.is_dir():
            raise RebuildError(f"candidate Bundle root is not final: {output}")
        digest, size, _ = _sha256_file(artifact)
        if (digest, size) != (
            result["closure_artifact_sha256"], result["closure_artifact_size"]
        ):
            raise RebuildError(f"closure artifact changed: {artifact}")
        closure = _read_json(artifact)
        if closure["closure_digest"] != result["closure_digest"]:
            raise RebuildError(f"closure digest binding changed: {artifact}")
        _observe_bundle_closure(self.source_bundles / spec.source_relative, closure)
        _observe_bundle_closure(output, closure)
        if revalidate_proof:
            current_proof = _migration_proof_binding(
                self.source_bundles / spec.source_relative, spec, closure
            )
            if current_proof != result["legacy_proof"]:
                raise RebuildError(f"legacy migration proof changed: {spec.name}")

    def _current_gates(self, spec: DatasetSpec) -> tuple[Any, ...]:
        from hotpot.ds.gates._persistence import deserialize_gate_configs

        policy = _source_bundle_policy(
            self.source_bundles / spec.source_relative, spec
        )
        return deserialize_gate_configs(policy["value"]["gates"])

    def _process_phase(self, spec: DatasetSpec) -> dict[str, Any]:
        from hotpot import ds

        root = self._output_root(spec)
        facts = root / "cases" / "facts.sqlite3"
        if facts.exists():
            bundle = ds.load_bundle(root, spec.name, db=self.output_database)
        else:
            bundle = ds.Bundle(
                root, spec.name, db=self.output_database, gates=self._current_gates(spec)
            )
        if not bundle.is_synchronized:
            bundle.synchronize()
        certificate = bundle.write_certificate()
        receipt = bundle.ingest(include_incomplete=True, reason=INCOMPLETE_REASON)
        status = bundle.status
        if (
            status["release_state"] != "sealed"
            or status["receipt_integrity"] != "valid"
            or status["certificate_state"] != "current"
            or not status["is_synchronized"]
        ):
            raise RebuildError(f"candidate Bundle is not sealed and current: {spec.name}")
        return {
            "bundle_root": str(root.resolve()),
            "certificate": str(certificate.resolve()),
            "certificate_sha256": _sha256_file(certificate)[0],
            "receipt_digest": receipt["receipt_digest"],
            "case_count": len(bundle),
            "generation_id": status["generation_id"],
            "generation_digest": status["generation_digest"],
        }

    def _validate_process_phase(self, spec: DatasetSpec, result: Mapping[str, Any]) -> None:
        from hotpot import ds

        root = self._output_root(spec)
        bundle = ds.load_bundle(root, spec.name, db=self.output_database)
        status = bundle.status
        if (
            status["release_state"] != "sealed"
            or status["receipt_integrity"] != "valid"
            or status["certificate_state"] != "current"
            or not status["is_synchronized"]
        ):
            raise RebuildError(f"candidate Bundle is not sealed and current: {spec.name}")
        if len(bundle) != result["case_count"]:
            raise RebuildError(f"candidate Bundle Case count changed: {spec.name}")
        if (
            status["generation_id"] != result["generation_id"]
            or status["generation_digest"] != result["generation_digest"]
        ):
            raise RebuildError(f"candidate generation changed: {spec.name}")
        certificate_digest, _, _ = _sha256_file(bundle.certificate_path)
        if certificate_digest != result["certificate_sha256"]:
            raise RebuildError(f"candidate certificate changed: {spec.name}")
        receipt_path = root / "ingestion-receipt.json"
        receipt = _read_json(receipt_path)
        if receipt.get("receipt_digest") != result["receipt_digest"]:
            raise RebuildError(f"candidate receipt changed: {spec.name}")

    def _parity_phase(self) -> dict[str, Any]:
        source_stamp = _stat_stamp(self.source_database.stat(follow_symlinks=False))
        target_stamp = _stat_stamp(self.output_database.stat(follow_symlinks=False))
        for spec in DATASETS:
            copy_result = self._load_state()["phases"][f"copy:{spec.name}"]["result"]
            closure = _read_json(Path(copy_result["closure_artifact"]))
            _observe_bundle_closure(self.source_bundles / spec.source_relative, closure)
            _observe_bundle_closure(self._output_root(spec), closure)
        semantic = _database_semantic_parity(
            self.source_bundles,
            self.output_bundles,
            self.output_database,
            DATASETS,
        )
        target_cas = _verify_cas(self.output_database)
        source_database_digest, source_database_size, final_stamp = _sha256_file(
            self.source_database
        )
        if source_stamp != final_stamp:
            raise RebuildError("source database changed during parity verification")
        target_database_digest, target_database_size, _ = _sha256_file(
            self.output_database
        )
        if target_stamp != _stat_stamp(
            self.output_database.stat(follow_symlinks=False)
        ):
            raise RebuildError("candidate database changed during parity verification")
        formal_bytes = target_database_size + target_cas["bytes"]
        if formal_bytes > MAX_CANDIDATE_BYTES:
            raise RebuildError(
                "candidate formal store exceeds the 1 GiB compact-release limit: "
                f"{formal_bytes} > {MAX_CANDIDATE_BYTES}"
            )
        result = {
            "semantic": semantic,
            "target_cas": target_cas,
            "source_anchor": {
                "sha256": source_database_digest, "size": source_database_size
            },
            "target_database": {
                "sha256": target_database_digest, "size": target_database_size
            },
            "compact_size": {
                "database_bytes": target_database_size,
                "cas_object_bytes": target_cas["bytes"],
                "formal_bytes": formal_bytes,
                "limit_bytes": MAX_CANDIDATE_BYTES,
                "within_limit": True,
            },
        }
        artifact = self.state_root / "parity.json"
        _atomic_json(artifact, result)
        result["artifact"] = str(artifact.resolve())
        result["artifact_sha256"] = _sha256_file(artifact)[0]
        return result

    def _validate_parity_phase(self, result: Mapping[str, Any]) -> None:
        artifact = Path(result["artifact"])
        if _sha256_file(artifact)[0] != result["artifact_sha256"]:
            raise RebuildError("parity artifact changed")
        artifact_value = _read_json(artifact)
        if artifact_value != {
            key: result[key]
            for key in (
                "semantic", "target_cas", "source_anchor", "target_database",
                "compact_size",
            )
        }:
            raise RebuildError("parity artifact no longer matches rebuild state")
        current = self._parity_observation()
        for key in (
            "semantic", "target_cas", "source_anchor", "target_database",
            "compact_size",
        ):
            if current[key] != result[key]:
                raise RebuildError(f"completed parity output no longer matches: {key}")

    def _parity_observation(self) -> dict[str, Any]:
        semantic = _database_semantic_parity(
            self.source_bundles,
            self.output_bundles,
            self.output_database,
            DATASETS,
        )
        target_cas = _verify_cas(self.output_database)
        source_digest, source_size, _ = _sha256_file(self.source_database)
        target_digest, target_size, _ = _sha256_file(self.output_database)
        formal_bytes = target_size + target_cas["bytes"]
        return {
            "semantic": semantic,
            "target_cas": target_cas,
            "source_anchor": {"sha256": source_digest, "size": source_size},
            "target_database": {"sha256": target_digest, "size": target_size},
            "compact_size": {
                "database_bytes": target_size,
                "cas_object_bytes": target_cas["bytes"],
                "formal_bytes": formal_bytes,
                "limit_bytes": MAX_CANDIDATE_BYTES,
                "within_limit": formal_bytes <= MAX_CANDIDATE_BYTES,
            },
        }

    def _preactivation_recheck(self, state: Mapping[str, Any]) -> dict[str, Any]:
        parity = state["phases"]["parity"]["result"]
        self._validate_parity_phase(parity)
        for spec in DATASETS:
            copied = state["phases"][f"copy:{spec.name}"]["result"]
            self._validate_copy_phase(spec, copied)
            self._validate_process_phase(
                spec, state["phases"][f"process:{spec.name}"]["result"]
            )
        return {
            "checked_at": _utc_now(),
            "parity_artifact_sha256": parity["artifact_sha256"],
            "semantic_digest": parity["semantic"]["semantic_digest"],
            "source_database_sha256": parity["source_anchor"]["sha256"],
            "target_database_sha256": parity["target_database"]["sha256"],
            "cas_sha256": parity["target_cas"]["sha256"],
        }

    def _flush_candidate_publication(self) -> None:
        for spec in DATASETS:
            _fsync_tree(self._output_root(spec))
        _fsync_tree(self.output_generation)
        _fsync_directory(self.output_bundles)
        _fsync_directory(self.output_generation.parent)

    def activate(self, confirmation: str) -> dict[str, Any]:
        if confirmation != self.run_id:
            raise RebuildError("--confirm-run-id must exactly match the candidate run ID")
        with self.locked():
            state = self._load_state()
            if state["phases"].get("ready", {}).get("status") != "complete":
                raise RebuildError("candidate has not completed the ready phase")
            with self._publication_barrier():
                active = self.store_root / "active"
                active_target = active.resolve() if active.is_symlink() else None
                candidate = self.output_generation.resolve()
                if active_target == candidate:
                    self._preactivation_recheck(state)
                    self._flush_candidate_publication()
                    return self._finalize_activation(state, recovered=True)
                if active_target != self.source_generation_root.resolve():
                    raise RebuildError(
                        f"active store changed outside this rebuild: {active_target}"
                    )
                pending = state.get("activation", {})
                if pending.get("status") == "prepared":
                    intent = _read_json(Path(pending["intent"]))
                    if intent.get("candidate_generation") != str(candidate):
                        raise RebuildError("pending activation names another candidate")
                    self._preactivation_recheck(state)
                    self._flush_candidate_publication()
                    self._replace_active_link(candidate, pending["activation_id"])
                    return self._finalize_activation(state, recovered=True)
                proof = self._preactivation_recheck(state)
                self._flush_candidate_publication()
                activation_id = f"activation-{uuid.uuid4().hex}"
                intent = {
                    "schema": ACTIVATION_SCHEMA,
                    "activation_id": activation_id,
                    "run_id": self.run_id,
                    "created_at": _utc_now(),
                    "status": "prepared",
                    "active_link": str(active),
                    "previous_generation": str(active_target),
                    "candidate_generation": str(candidate),
                    "proof": proof,
                }
                intent_path = self.state_root / "activation-intent.json"
                _atomic_json(intent_path, intent)
                state["activation"] = {
                    "status": "prepared",
                    "activation_id": activation_id,
                    "intent": str(intent_path.resolve()),
                    "previous_generation": str(active_target),
                    "candidate_generation": str(candidate),
                }
                self._save_state(state)
                _append_event(
                    self.events_path,
                    "activation-prepared",
                    activation_id=activation_id,
                )
                self._replace_active_link(candidate, activation_id)
                return self._finalize_activation(state, recovered=False)

    def _replace_active_link(self, target: Path, operation_id: str) -> None:
        active = self.store_root / "active"
        temporary = active.with_name(f".{active.name}.{operation_id}.tmp")
        relative_target = os.path.relpath(target, self.store_root)
        if temporary.is_symlink():
            if temporary.resolve() != target.resolve():
                raise RebuildError(f"owned publication temporary has another target: {temporary}")
            temporary.unlink()
        elif temporary.exists():
            raise RebuildError(f"owned publication temporary is not a symlink: {temporary}")
        os.symlink(relative_target, temporary)
        try:
            os.replace(temporary, active)
            _fsync_directory(self.store_root)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    def _finalize_activation(
        self, state: dict[str, Any], *, recovered: bool
    ) -> dict[str, Any]:
        active = self.store_root / "active"
        if not active.is_symlink() or active.resolve() != self.output_generation.resolve():
            raise RebuildError("candidate is not the active generation")
        activation = state.get("activation", {})
        if activation.get("status") == "activated":
            record_path = Path(activation["record"])
            if (
                not record_path.is_file()
                or _sha256_file(record_path)[0] != activation["record_sha256"]
            ):
                raise RebuildError("persisted activation record is missing or changed")
            return dict(activation)
        intent_path = Path(activation.get("intent", self.state_root / "activation-intent.json"))
        intent = _read_json(intent_path)
        if intent.get("candidate_generation") != str(self.output_generation.resolve()):
            raise RebuildError("activation intent does not name this candidate")
        record = {
            **intent,
            "status": "activated",
            "activated_at": _utc_now(),
            "recovered_after_atomic_swap": recovered,
            "rollback": {
                "method": "atomically replace the active directory symlink",
                "target": intent["previous_generation"],
                "requires_owner_verification": True,
            },
        }
        activation_path = self.output_generation / "cutover-activation.json"
        _atomic_json(activation_path, record)
        state["activation"] = {
            "status": "activated",
            "activation_id": record["activation_id"],
            "activated_at": record["activated_at"],
            "record": str(activation_path.resolve()),
            "record_sha256": _sha256_file(activation_path)[0],
            "previous_generation": record["previous_generation"],
            "candidate_generation": record["candidate_generation"],
        }
        self._save_state(state)
        _append_event(
            self.events_path,
            "activation-completed",
            activation_id=record["activation_id"],
            recovered=recovered,
        )
        return dict(state["activation"])

    def rollback(self, confirmation: str) -> dict[str, Any]:
        """Atomically restore the proved previous generation without data loss."""

        with self.locked():
            state = self._load_state()
            activation = state.get("activation", {})
            activation_id = activation.get("activation_id")
            if confirmation != activation_id:
                raise RebuildError(
                    "--confirm-activation-id must exactly match the active activation ID"
                )
            with self._publication_barrier():
                active = self.store_root / "active"
                active_target = active.resolve() if active.is_symlink() else None
                previous = self.source_generation_root.resolve()
                if (
                    activation.get("status") == "rollback-prepared"
                    and active_target == previous
                ):
                    self._preactivation_recheck(state)
                    return self._finalize_rollback(state, recovered=True)
                if (
                    activation.get("status") == "rollback-prepared"
                    and active_target == self.output_generation.resolve()
                ):
                    self._preactivation_recheck(state)
                    self._replace_active_link(previous, activation["rollback_id"])
                    return self._finalize_rollback(state, recovered=True)
                if activation.get("status") != "activated":
                    raise RebuildError("there is no activated candidate to roll back")
                if active_target != self.output_generation.resolve():
                    raise RebuildError(
                        f"active store changed outside this rebuild: {active_target}"
                    )
                # This rejects rollback if either generation acquired any
                # post-activation write, preventing silent loss of new data.
                proof = self._preactivation_recheck(state)
                rollback_id = f"rollback-{uuid.uuid4().hex}"
                intent = {
                    "schema": ACTIVATION_SCHEMA,
                    "rollback_id": rollback_id,
                    "activation_id": activation_id,
                    "run_id": self.run_id,
                    "created_at": _utc_now(),
                    "status": "prepared",
                    "active_link": str(active),
                    "candidate_generation": str(self.output_generation.resolve()),
                    "previous_generation": str(previous),
                    "proof": proof,
                }
                intent_path = self.state_root / "rollback-intent.json"
                _atomic_json(intent_path, intent)
                state["activation"] = {
                    **activation,
                    "status": "rollback-prepared",
                    "rollback_id": rollback_id,
                    "rollback_intent": str(intent_path.resolve()),
                }
                self._save_state(state)
                _append_event(
                    self.events_path,
                    "rollback-prepared",
                    activation_id=activation_id,
                    rollback_id=rollback_id,
                )
                self._replace_active_link(previous, rollback_id)
                return self._finalize_rollback(state, recovered=False)

    def _finalize_rollback(
        self, state: dict[str, Any], *, recovered: bool
    ) -> dict[str, Any]:
        active = self.store_root / "active"
        activation = state["activation"]
        if not active.is_symlink() or active.resolve() != self.source_generation_root.resolve():
            raise RebuildError("previous generation is not active")
        intent = _read_json(Path(activation["rollback_intent"]))
        if intent.get("activation_id") != activation.get("activation_id"):
            raise RebuildError("rollback intent does not match the activation")
        record = {
            **intent,
            "status": "rolled-back",
            "rolled_back_at": _utc_now(),
            "recovered_after_atomic_swap": recovered,
        }
        record_path = self.output_generation / "cutover-rollback.json"
        _atomic_json(record_path, record)
        state["activation"] = {
            **activation,
            "status": "rolled-back",
            "rolled_back_at": record["rolled_back_at"],
            "rollback_record": str(record_path.resolve()),
            "rollback_record_sha256": _sha256_file(record_path)[0],
        }
        self._save_state(state)
        _append_event(
            self.events_path,
            "rollback-completed",
            activation_id=record["activation_id"],
            rollback_id=record["rollback_id"],
            recovered=recovered,
        )
        return dict(state["activation"])

    def status(self, *, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if state is None:
            state = self._load_state()
        active = self.store_root / "active"
        active_target = str(active.resolve()) if active.is_symlink() else None
        return {
            "schema": STATE_SCHEMA,
            "run_id": self.run_id,
            "active_generation": active_target,
            "candidate_generation": str(self.output_generation.resolve()),
            "phases": {
                name: value.get("status")
                for name, value in state.get("phases", {}).items()
            },
            "activation": state.get("activation", {"status": "not-activated"}),
            "events": str(self.events_path),
        }

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise RebuildError(message)


def _repository_default() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build, prove, and separately activate a current-runtime Stage 13 store."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="plan",
        choices=("plan", "run", "status", "activate", "rollback"),
    )
    parser.add_argument("--repository", type=Path, default=_repository_default())
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--source-generation", default=SOURCE_GENERATION)
    parser.add_argument(
        "--confirm-run-id",
        help="required by activate; must exactly equal --run-id",
    )
    parser.add_argument(
        "--confirm-activation-id",
        help="required by rollback; must exactly equal the persisted activation ID",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    operator = Operator(
        arguments.repository, arguments.run_id, arguments.source_generation
    )
    try:
        if arguments.command == "plan":
            result = operator.plan()
        elif arguments.command == "run":
            result = operator.run()
        elif arguments.command == "status":
            result = operator.status()
        elif arguments.command == "activate":
            result = operator.activate(arguments.confirm_run_id or "")
        else:
            result = operator.rollback(arguments.confirm_activation_id or "")
    except Exception as error:
        print(
            json.dumps(
                {"ok": False, "error": type(error).__name__, "message": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"ok": True, "result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
