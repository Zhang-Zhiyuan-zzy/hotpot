"""Remove reproducible working data from one sealed, ingested Bundle."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

from .public.bundle import load_bundle
from .public.certificate import certificate_digest
from .storage import TargetStore, _decode_facts


_CLEANUP_SCHEMA = "hotpot-ds-bundle-cleanup/current"
_CLEANUP_FILE = "bundle-cleanup.json"
_SQLITE_SUFFIXES = ("", "-journal", "-shm", "-wal")
_DERIVED_TREES = (
    "assessments/generations",
    "assessments/cache",
    "assessments/caches",
    "assessments/temp",
    "assessments/tmp",
    "cases/cache",
    "cases/caches",
    "cases/temp",
    "cases/tmp",
)
_CURRENT_ASSESSMENT_PATHS = (
    "assessments/current",
    "assessments/current-generation",
    "assessments/current.json",
)
_PRESERVED_ROOTS = ("audit", "policy", "structures")


class BundleCleanupError(RuntimeError):
    """A sealed Bundle cannot be safely or completely cleaned."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _value_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _manifest_digest(document: Mapping[str, Any]) -> str:
    return _value_digest(
        {key: value for key, value in document.items() if key != "manifest_sha256"}
    )


def _hash_stream(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _hash_regular_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BundleCleanupError(f"cannot open regular cleanup-bound file: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BundleCleanupError(f"cleanup candidate is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest, size = _hash_stream(stream)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ):
            raise BundleCleanupError(f"cleanup candidate changed while read: {path}")
        return digest, size
    finally:
        os.close(descriptor)


def _path_payload(path: Path) -> tuple[str, int, str]:
    metadata = path.lstat()
    if stat.S_ISREG(metadata.st_mode):
        digest, size = _hash_regular_file(path)
        return digest, size, "file"
    if stat.S_ISLNK(metadata.st_mode):
        payload = os.readlink(path).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), len(payload), "symlink"
    raise BundleCleanupError(f"cleanup candidate has unsupported type: {path}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = _canonical_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_relative(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise BundleCleanupError("cleanup path must be a safe POSIX relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise BundleCleanupError("cleanup path must be a safe POSIX relative path")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise BundleCleanupError(f"cleanup path traverses a symbolic link: {value}")
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise BundleCleanupError(f"cleanup path escapes the Bundle root: {value}") from error
    return candidate


@contextmanager
def _bundle_lock(root: Path) -> Iterator[None]:
    path = root / ".bundle.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise BundleCleanupError("cannot open the Bundle lock safely") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BundleCleanupError("Bundle lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _generation_locks(root: Path) -> Iterator[None]:
    directory = root / "assessments" / "leases"
    with ExitStack() as stack:
        if directory.exists():
            if directory.is_symlink() or not directory.is_dir():
                raise BundleCleanupError("assessment lease path is not a directory")
            for path in sorted(directory.iterdir()):
                if path.is_symlink() or not path.is_file():
                    raise BundleCleanupError(f"assessment lease is not a regular file: {path}")
                descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
                stack.callback(os.close, descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                stack.callback(fcntl.flock, descriptor, fcntl.LOCK_UN)
        yield


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleCleanupError(f"cannot read valid {description}: {path}") from error
    if not isinstance(value, dict):
        raise BundleCleanupError(f"{description} must contain one JSON object")
    return value


def _target_audit(store: TargetStore) -> dict[str, Any]:
    if store.asset_root.is_symlink() or not store.asset_root.is_dir():
        raise BundleCleanupError("target asset root is not a regular directory")
    with store._connect() as connection:
        metadata = store._metadata(connection)
        integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            raise BundleCleanupError("target store failed SQLite integrity_check")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise BundleCleanupError("target store failed foreign_key_check")
        assets = connection.execute(
            "SELECT content_sha256,size,relative_path FROM target_assets "
            "ORDER BY content_sha256"
        ).fetchall()
        case_count = 0
        for row in connection.execute(
            "SELECT source_payload_blob,source_payload_digest FROM cases "
            "ORDER BY dataset_id,release_id,public_index,case_id"
        ):
            try:
                _decode_facts(
                    row["source_payload_blob"],
                    row["source_payload_digest"],
                )
            except Exception as error:
                raise BundleCleanupError(
                    "target store contains corrupt compressed Case facts"
                ) from error
            case_count += 1
    total = 0
    for row in assets:
        relative = f"objects/sha256/{row['content_sha256'][:2]}/{row['content_sha256']}"
        if row["relative_path"] != relative:
            raise BundleCleanupError("target-owned asset has a non-canonical path")
        path = store.asset_root / row["relative_path"]
        digest, size = _hash_regular_file(path)
        if digest != row["content_sha256"] or size != row["size"]:
            raise BundleCleanupError(f"target-owned asset failed integrity check: {path}")
        total += size
    return {
        "store_uuid": metadata["store_uuid"],
        "asset_count": len(assets),
        "asset_bytes": total,
        "case_count": case_count,
    }


def _local_receipt_binding(
    root: Path,
    release_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    receipt = release_metadata["receipt"]
    if not isinstance(receipt, Mapping):
        raise BundleCleanupError("target release has no authoritative ingestion receipt")
    projection = {
        "receipt_id": receipt["receipt_id"],
        "store_uuid": receipt["store_uuid"],
        "dataset": receipt["dataset"],
        "release": receipt["release"],
        "release_serial": release_metadata["release_serial"],
        "included_count": receipt["included_count"],
        "excluded_count": receipt["excluded_count"],
        "committed_at": receipt["committed_at"],
    }
    local = _read_json(root / "ingestion-receipt.json", "ingestion receipt")
    expected = {**projection, "receipt_digest": _value_digest(projection)}
    if local != expected:
        raise BundleCleanupError(
            "local ingestion receipt does not match the authoritative target receipt"
        )
    return dict(receipt), hashlib.sha256(
        (root / "ingestion-receipt.json").read_bytes()
    ).hexdigest()


def _walk_files(directory: Path) -> Iterator[Path]:
    if not directory.exists():
        return
    if directory.is_symlink() or not directory.is_dir():
        raise BundleCleanupError(f"derived-data tree is not a directory: {directory}")
    for current, directories, files in os.walk(directory, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise BundleCleanupError(f"derived-data tree contains a symlink: {path}")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise BundleCleanupError(f"derived-data tree contains an unsafe entry: {path}")
            yield path


def _sqlite_candidate(path: Path) -> bool:
    name = path.name
    return any(name.endswith(f".sqlite3{suffix}") for suffix in _SQLITE_SUFFIXES)


def _candidate_paths(
    root: Path,
    manifest: Mapping[str, Any],
    scientific_paths: set[str],
) -> dict[Path, str]:
    candidates: dict[Path, str] = {}
    preserved = {
        str(manifest["intake"]["path"]),
        *scientific_paths,
        "bundle.json",
        "certificate.json",
        "ingestion-receipt.json",
        _CLEANUP_FILE,
    }
    for item in manifest.get("control_assets", []):
        if item["category"] != "migration":
            preserved.add(str(item["path"]))
    for prefix in _PRESERVED_ROOTS:
        for path in _walk_files(root / prefix):
            preserved.add(path.relative_to(root).as_posix())

    cases = root / "cases"
    if cases.exists():
        if cases.is_symlink() or not cases.is_dir():
            raise BundleCleanupError("cases path is not a directory")
        for path in _walk_files(cases):
            relative = path.relative_to(root).as_posix()
            if (
                _sqlite_candidate(path)
                or path.name == ".build.lock"
                or (path.name.startswith(".facts.") and path.name.endswith(".tmp"))
            ):
                candidates[path] = "canonical-facts"

    for value in _DERIVED_TREES:
        for path in _walk_files(root / value):
            candidates[path] = "derived-assessment"
    leases = root / "assessments" / "leases"
    for path in _walk_files(leases):
        candidates[path] = "assessment-lease"
    for value in _CURRENT_ASSESSMENT_PATHS:
        path = root / value
        if path.exists() or path.is_symlink():
            if path.is_dir() and not path.is_symlink():
                raise BundleCleanupError(f"current-generation marker is a directory: {path}")
            candidates[path] = "current-generation-link"
    for item in manifest.get("control_assets", []):
        if item["category"] == "migration":
            path = _safe_relative(root, item["path"])
            candidates[path] = "migration-control"

    conflicts = sorted(
        path.relative_to(root).as_posix()
        for path in candidates
        if path.relative_to(root).as_posix() in preserved
    )
    if conflicts:
        raise BundleCleanupError(
            "derived cleanup paths overlap preserved source assets: "
            + ", ".join(conflicts)
        )
    return candidates


def _scientific_paths(bundle) -> set[str]:
    result: set[str] = set()
    for case in bundle._state.iter_cases():
        for collection in ("structures", "evidence"):
            for record in case.get(collection, []):
                for asset in record.get("assets", []):
                    path = asset.get("path")
                    if isinstance(path, str):
                        result.add(path)
                path = record.get("path")
                if isinstance(path, str):
                    result.add(path)
    return result


def _deletion_inventory(
    root: Path,
    candidates: Mapping[Path, str],
) -> list[dict[str, Any]]:
    inventory = []
    for path, category in sorted(
        candidates.items(), key=lambda item: item[0].relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        safe = _safe_relative(root, relative)
        if not safe.exists() and not safe.is_symlink():
            continue
        digest, size, kind = _path_payload(safe)
        inventory.append(
            {
                "path": relative,
                "sha256": digest,
                "bytes": size,
                "kind": kind,
                "category": category,
            }
        )
    return inventory


def _validate_cleanup_manifest(
    root: Path,
    document: Mapping[str, Any],
    dataset: str,
    database: Path,
) -> None:
    fields = {
        "schema",
        "state",
        "created_at",
        "completed_at",
        "bundle",
        "receipt",
        "target_audit",
        "deleted",
        "removed_count",
        "removed_bytes",
        "manifest_sha256",
    }
    if set(document) != fields or document.get("schema") != _CLEANUP_SCHEMA:
        raise BundleCleanupError("bundle-cleanup.json does not use the current schema")
    if document["state"] not in {"planned", "complete"}:
        raise BundleCleanupError("bundle-cleanup.json has an invalid state")
    if document["manifest_sha256"] != _manifest_digest(document):
        raise BundleCleanupError("bundle-cleanup.json digest does not match its content")
    locator = document["bundle"]
    if (
        not isinstance(locator, Mapping)
        or set(locator) != {"root", "dataset", "release_id", "database"}
        or locator.get("root") != str(root)
        or locator.get("dataset") != dataset
        or locator.get("database") != str(database)
    ):
        raise BundleCleanupError("bundle-cleanup.json names another Bundle or target")
    receipt = document["receipt"]
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "receipt_id",
        "receipt_sha256",
        "local_receipt_sha256",
        "certificate_sha256",
        "store_uuid",
    }:
        raise BundleCleanupError("bundle-cleanup.json receipt binding is invalid")
    for name in (
        "receipt_sha256",
        "local_receipt_sha256",
        "certificate_sha256",
    ):
        value = receipt[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise BundleCleanupError("bundle-cleanup.json receipt digest is invalid")
    target_audit = document["target_audit"]
    if not isinstance(target_audit, Mapping) or set(target_audit) != {
        "store_uuid", "asset_count", "asset_bytes", "case_count"
    }:
        raise BundleCleanupError("bundle-cleanup.json target audit is invalid")
    if target_audit["store_uuid"] != receipt["store_uuid"]:
        raise BundleCleanupError("bundle-cleanup.json target identity is inconsistent")
    if any(
        not isinstance(target_audit[name], int)
        or isinstance(target_audit[name], bool)
        or target_audit[name] < 0
        for name in ("asset_count", "asset_bytes", "case_count")
    ):
        raise BundleCleanupError("bundle-cleanup.json target counts are invalid")

    deleted = document["deleted"]
    if (
        not isinstance(deleted, list)
        or any(not isinstance(item, Mapping) for item in deleted)
        or not isinstance(document["removed_count"], int)
        or isinstance(document["removed_count"], bool)
        or document["removed_count"] != len(deleted)
    ):
        raise BundleCleanupError("bundle-cleanup.json deletion inventory is invalid")
    if (
        not isinstance(document["removed_bytes"], int)
        or isinstance(document["removed_bytes"], bool)
        or document["removed_bytes"] < 0
        or document["removed_bytes"] != sum(item.get("bytes", -1) for item in deleted)
    ):
        raise BundleCleanupError("bundle-cleanup.json byte count is invalid")
    bundle_manifest = _read_json(root / "bundle.json", "Bundle manifest")
    if (
        bundle_manifest.get("ds_name") != dataset
        or bundle_manifest.get("release_id") != locator["release_id"]
    ):
        raise BundleCleanupError("bundle-cleanup.json conflicts with bundle.json")
    migration_paths = {
        str(item["path"])
        for item in bundle_manifest.get("control_assets", [])
        if isinstance(item, Mapping) and item.get("category") == "migration"
    }
    intake_path = bundle_manifest.get("intake", {}).get("path")
    seen = set()
    for item in deleted:
        if not isinstance(item, Mapping) or set(item) != {
            "path", "sha256", "bytes", "kind", "category"
        }:
            raise BundleCleanupError("bundle-cleanup.json has an invalid deletion record")
        path = str(item["path"])
        _safe_relative(root, path)
        if path in seen:
            raise BundleCleanupError("bundle-cleanup.json repeats a deletion path")
        seen.add(path)
        digest = item["sha256"]
        size = item["bytes"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or item["kind"] not in {"file", "symlink"}
            or not _allowed_cleanup_record(path, str(item["category"]), migration_paths)
            or path == intake_path
        ):
            raise BundleCleanupError("bundle-cleanup.json has an unsafe deletion record")


def _under(path: str, directory: str) -> bool:
    value = PurePosixPath(path)
    root = PurePosixPath(directory)
    return value != root and root in value.parents


def _allowed_cleanup_record(
    path: str,
    category: str,
    migration_paths: set[str],
) -> bool:
    value = PurePosixPath(path)
    if category == "canonical-facts":
        return _under(path, "cases") and (
            _sqlite_candidate(Path(value.name))
            or value.name == ".build.lock"
            or (value.name.startswith(".facts.") and value.name.endswith(".tmp"))
        )
    if category == "derived-assessment":
        return any(_under(path, directory) for directory in _DERIVED_TREES)
    if category == "assessment-lease":
        return _under(path, "assessments/leases")
    if category == "current-generation-link":
        return path in _CURRENT_ASSESSMENT_PATHS
    if category == "migration-control":
        return path in migration_paths
    return False


def _unlink_inventory(root: Path, inventory: list[Mapping[str, Any]]) -> None:
    for item in inventory:
        path = _safe_relative(root, item["path"])
        if not path.exists() and not path.is_symlink():
            continue
        actual = _path_payload(path)
        expected = (item["sha256"], item["bytes"], item["kind"])
        if actual != expected:
            raise BundleCleanupError(
                f"cleanup candidate changed after validation: {item['path']}"
            )
        path.unlink()
    for parent in sorted(
        {(_safe_relative(root, item["path"])).parent for item in inventory},
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        current = parent
        while current != root:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def _result(document: Mapping[str, Any], *, already_clean: bool) -> dict[str, Any]:
    return {
        "bundle": dict(document["bundle"]),
        "cleanup_manifest": str(Path(document["bundle"]["root"]) / _CLEANUP_FILE),
        "already_clean": already_clean,
        "removed_count": document["removed_count"],
        "removed_bytes": document["removed_bytes"],
        "receipt_id": document["receipt"]["receipt_id"],
    }


def remove_bundle_intermediates(
    root: Path,
    dataset: str,
    database: Path,
) -> dict[str, Any]:
    """Delete only allowlisted reproducible data from one verified sealed Bundle."""

    bundle_root = root.expanduser().resolve()
    target = database.expanduser().resolve()
    if not bundle_root.is_dir():
        raise NotADirectoryError(bundle_root)
    cleanup_path = bundle_root / _CLEANUP_FILE
    with _bundle_lock(bundle_root):
        if cleanup_path.is_file():
            document = _read_json(cleanup_path, "Bundle cleanup manifest")
            _validate_cleanup_manifest(bundle_root, document, dataset, target)
            already_clean = document["state"] == "complete"
            with _generation_locks(bundle_root):
                _unlink_inventory(bundle_root, document["deleted"])
            if not already_clean:
                document = dict(document)
                document["state"] = "complete"
                document["completed_at"] = _utc_now()
                document["manifest_sha256"] = _manifest_digest(document)
                _atomic_json(cleanup_path, document)
            return _result(document, already_clean=already_clean)

        bundle = load_bundle(bundle_root, dataset, db=target)
        receipt = bundle._authoritative_receipt()
        if receipt is None:
            raise BundleCleanupError(
                "Bundle cleanup requires a successfully ingested sealed release"
            )
        release_id = str(bundle._state.manifest["release_id"])
        store = TargetStore(target)
        target_audit = _target_audit(store)
        release = store.load_release_metadata(dataset, release_id)
        if release is None:
            raise BundleCleanupError("authoritative target release disappeared during cleanup")
        authoritative, local_receipt_sha256 = _local_receipt_binding(bundle_root, release)
        if authoritative != receipt:
            raise BundleCleanupError("authoritative receipt changed during cleanup validation")
        certificate_sha256 = certificate_digest(bundle.certificate_path)
        if certificate_sha256 != receipt["certificate_digest"]:
            raise BundleCleanupError("certificate does not match the authoritative receipt")
        manifest = bundle._state.manifest
        candidates = _candidate_paths(
            bundle_root,
            manifest,
            _scientific_paths(bundle),
        )
        inventory = _deletion_inventory(bundle_root, candidates)
        locator = {
            "root": str(bundle_root),
            "dataset": dataset,
            "release_id": release_id,
            "database": str(target),
        }
        document: dict[str, Any] = {
            "schema": _CLEANUP_SCHEMA,
            "state": "planned",
            "created_at": _utc_now(),
            "completed_at": None,
            "bundle": locator,
            "receipt": {
                "receipt_id": receipt["receipt_id"],
                "receipt_sha256": _value_digest(receipt),
                "local_receipt_sha256": local_receipt_sha256,
                "certificate_sha256": certificate_sha256,
                "store_uuid": receipt["store_uuid"],
            },
            "target_audit": target_audit,
            "deleted": inventory,
            "removed_count": len(inventory),
            "removed_bytes": sum(item["bytes"] for item in inventory),
        }
        document["manifest_sha256"] = _manifest_digest(document)
        _atomic_json(cleanup_path, document)
        with _generation_locks(bundle_root):
            _unlink_inventory(bundle_root, inventory)
        document["state"] = "complete"
        document["completed_at"] = _utc_now()
        document["manifest_sha256"] = _manifest_digest(document)
        _atomic_json(cleanup_path, document)
        return _result(document, already_clean=False)


__all__ = ["BundleCleanupError", "remove_bundle_intermediates"]
