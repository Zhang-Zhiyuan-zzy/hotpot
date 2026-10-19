"""Verified replication of one current Hotpot dataset store.

This is an internal CLI support module.  Replication preserves the immutable
SQLite rows and content-addressed objects exactly; it does not rebuild,
recertify, or re-ingest a release.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from .public._implementation import (
    PolicyProtocolError,
    require_current_runtime_policy,
)
from .storage import (
    SCHEMA,
    TARGET_SCHEMA_ID,
    TargetStore,
    TargetStoreError,
    _canonical_json,
    _json_digest,
    case_set_digest,
    persisted_included_case_content_digest,
)


class StoreCopyError(RuntimeError):
    """An ingested store could not be verified or published unchanged."""


_BINDING_DIGEST_KEYS = {
    "bundle_digest": "bundle_digest",
    "certificate_digest": "certificate_digest",
    "policy_digest": "policy_digest",
    "generation_digest": "generation_digest",
    "case_content_digest": "case_content_digest",
    "included_case_content_digest": "included_case_content_digest",
    "included_cases_digest": "included_cases_digest",
    "excluded_cases_digest": "excluded_cases_digest",
    "asset_inventory_digest": "asset_inventory_digest",
}
_DIGEST_CHARS = frozenset("0123456789abcdef")


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _DIGEST_CHARS
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(path)


def _schema_inventory(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, str, str, str | None], ...]:
    return tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        )
    )


def _expected_schema_inventory() -> tuple[tuple[str, str, str, str | None], ...]:
    with sqlite3.connect(":memory:") as reference:
        reference.executescript(SCHEMA)
        return _schema_inventory(reference)


def _runtime_contract(package: Path) -> str:
    required = (
        package / "__init__.py",
        package / "__version__.py",
        package / "dataset" / "__init__.py",
        package / "cheminfo" / "__init__.py",
        package / "ds" / "__init__.py",
    )
    if any(not path.is_file() for path in required):
        raise StoreCopyError("target is not a complete importable Hotpot package")
    ds_root = package / "ds"
    files = [package / "__init__.py", package / "__version__.py"]
    files.extend(
        path
        for path in ds_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".sql"}
    )
    inventory = [
        {
            "path": path.relative_to(package).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(files)
    ]
    return _json_digest(inventory)


def _release_asset_digest(
    connection: sqlite3.Connection, dataset: str, release: str
) -> str:
    rows = connection.execute(
        "SELECT DISTINCT t.content_sha256,t.size FROM target_assets t JOIN ("
        "SELECT content_sha256 FROM structure_asset_refs "
        "WHERE dataset_id=? AND release_id=? UNION "
        "SELECT content_sha256 FROM scientific_asset_refs "
        "WHERE dataset_id=? AND release_id=? UNION "
        "SELECT content_sha256 FROM release_control_asset_refs "
        "WHERE dataset_id=? AND release_id=?"
        ") r USING(content_sha256) ORDER BY t.content_sha256",
        (dataset, release, dataset, release, dataset, release),
    )
    return TargetStore._stream_array_digest(
        {"sha256": row["content_sha256"], "size": row["size"]}
        for row in rows
    )


def _audit_receipt(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    store_uuid: str,
    policies: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        binding = json.loads(row["binding_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise StoreCopyError("ingestion receipt binding is invalid") from exc
    if not isinstance(binding, dict) or row["binding_json"] != _canonical_json(binding):
        raise StoreCopyError("ingestion receipt binding is not canonical JSON")

    dataset = row["dataset_id"]
    release = row["release_id"]
    policy = policies.get(dataset)
    if policy is None:
        raise StoreCopyError("ingestion receipt has no dataset policy")
    try:
        require_current_runtime_policy(binding["runtime_policy"])
    except (KeyError, PolicyProtocolError) as exc:
        raise StoreCopyError("ingestion receipt runtime binding is not current") from exc

    direct_fields = {
        "store_uuid": store_uuid,
        "dataset": dataset,
        "release": release,
        "version": row["version"],
        "policy_revision": policy["revision"],
        "policy_digest": policy["digest"],
        "include_incomplete": bool(row["include_incomplete"]),
        "reason": row["reason"],
    }
    if any(binding.get(key) != value for key, value in direct_fields.items()):
        raise StoreCopyError("ingestion receipt binding does not match stored rows")
    if binding.get("runtime_policy") != policy["value"].get("runtime_policy"):
        raise StoreCopyError("ingestion receipt and dataset policy runtimes differ")
    requested_serial = binding.get("requested_release_serial")
    if requested_serial is not None and requested_serial != row["release_serial"]:
        raise StoreCopyError("ingestion receipt release serial is inconsistent")

    context = binding.get("target_context")
    expected_context_fields = {
        "canonical_path",
        "store_state",
        "store_uuid",
        "schema_id",
        "dataset_state",
        "dataset_policy_revision",
        "dataset_policy_digest",
    }
    if (
        not isinstance(context, dict)
        or set(context) != expected_context_fields
        or context.get("schema_id") != TARGET_SCHEMA_ID
        or not isinstance(context.get("canonical_path"), str)
        or not context["canonical_path"]
    ):
        raise StoreCopyError("ingestion receipt target binding is invalid")

    for column, key in _BINDING_DIGEST_KEYS.items():
        value = row[column]
        if not _is_digest(value) or binding.get(key) != value:
            raise StoreCopyError("ingestion receipt digest binding is invalid")
    if row["store_uuid"] != store_uuid:
        raise StoreCopyError("ingestion receipt store UUID is inconsistent")
    if row["receipt_id"] != "receipt-" + _json_digest(binding):
        raise StoreCopyError("ingestion receipt identity is invalid")

    included_count = connection.execute(
        "SELECT COUNT(*) FROM cases WHERE dataset_id=? AND release_id=?",
        (dataset, release),
    ).fetchone()[0]
    if row["included_count"] != included_count:
        raise StoreCopyError("ingestion receipt Case count is inconsistent")
    included_digest = TargetStore._stream_array_digest(
        item[0]
        for item in connection.execute(
            "SELECT case_id FROM cases WHERE dataset_id=? AND release_id=? "
            "ORDER BY case_id",
            (dataset, release),
        )
    )
    if row["included_cases_digest"] != included_digest:
        raise StoreCopyError("ingestion receipt included-Case digest is invalid")
    try:
        persisted_content = persisted_included_case_content_digest(
            connection, dataset, release
        )
    except TargetStoreError as exc:
        raise StoreCopyError("persisted Case content audit failed") from exc
    if row["included_case_content_digest"] != persisted_content:
        raise StoreCopyError("ingestion receipt included-Case content is invalid")
    if row["excluded_count"] < 0:
        raise StoreCopyError("ingestion receipt excluded-Case count is invalid")
    if row["excluded_count"] == 0 and row["excluded_cases_digest"] != case_set_digest(()):
        raise StoreCopyError("ingestion receipt excluded-Case digest is invalid")
    if row["include_incomplete"] and row["excluded_count"]:
        raise StoreCopyError("forced incomplete archival retained excluded Cases")

    if row["asset_inventory_digest"] != _release_asset_digest(
        connection, dataset, release
    ):
        raise StoreCopyError("ingestion receipt asset inventory is invalid")


def _audit_database(database: Path) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, int]]:
    store = TargetStore(database)
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            raise StoreCopyError("SQLite integrity audit failed")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise StoreCopyError("SQLite foreign-key audit failed")
        actual_schema = _schema_inventory(connection)
        if actual_schema != _expected_schema_inventory():
            raise StoreCopyError("source store schema is not current")
        metadata = store._metadata(connection)

        datasets = {
            row[0] for row in connection.execute("SELECT dataset_id FROM datasets")
        }
        policies: dict[str, Mapping[str, Any]] = {}
        for row in connection.execute(
            "SELECT dataset_id,revision,policy_json,policy_digest "
            "FROM dataset_policies ORDER BY dataset_id"
        ):
            try:
                if row["policy_json"] != _canonical_json(
                    json.loads(row["policy_json"])
                ):
                    raise StoreCopyError("dataset policy is not canonical JSON")
                policy = store._policy_mapping(row)
                require_current_runtime_policy(policy["value"]["runtime_policy"])
            except (KeyError, json.JSONDecodeError, PolicyProtocolError, TargetStoreError) as exc:
                raise StoreCopyError("dataset policy runtime is not current") from exc
            policies[row["dataset_id"]] = policy
        if set(policies) != datasets:
            raise StoreCopyError("every dataset must have exactly one policy")

        missing_receipt = connection.execute(
            "SELECT 1 FROM releases r LEFT JOIN ingestion_receipts i "
            "USING(dataset_id,release_id) WHERE i.receipt_id IS NULL LIMIT 1"
        ).fetchone()
        if missing_receipt is not None:
            raise StoreCopyError("an ingested release has no receipt")
        receipts = list(
            connection.execute(
                "SELECT i.*,r.version,r.release_serial FROM ingestion_receipts i "
                "JOIN releases r USING(dataset_id,release_id) ORDER BY "
                "i.dataset_id,i.release_id"
            )
        )
        if not receipts:
            raise StoreCopyError("source store has no ingested releases")
        for receipt in receipts:
            _audit_receipt(
                connection,
                receipt,
                store_uuid=metadata["store_uuid"],
                policies=policies,
            )

        assets: list[dict[str, Any]] = []
        for row in connection.execute(
            "SELECT content_sha256,size,relative_path FROM target_assets "
            "ORDER BY content_sha256"
        ):
            digest = row["content_sha256"]
            expected = f"objects/sha256/{digest[:2]}/{digest}"
            if (
                not _is_digest(digest)
                or row["size"] < 0
                or row["relative_path"] != expected
            ):
                raise StoreCopyError("target asset registry is not canonical")
            assets.append(
                {
                    "sha256": digest,
                    "size": row["size"],
                    "relative_path": expected,
                }
            )
        orphan = connection.execute(
            "SELECT 1 FROM target_assets t WHERE NOT EXISTS ("
            "SELECT 1 FROM structure_asset_refs r WHERE r.content_sha256=t.content_sha256 "
            "UNION ALL SELECT 1 FROM scientific_asset_refs r "
            "WHERE r.content_sha256=t.content_sha256 UNION ALL SELECT 1 FROM "
            "release_control_asset_refs r WHERE r.content_sha256=t.content_sha256"
            ") LIMIT 1"
        ).fetchone()
        if orphan is not None:
            raise StoreCopyError("target asset registry contains an unreferenced object")

        counts = {
            "datasets": connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
            "releases": connection.execute("SELECT COUNT(*) FROM releases").fetchone()[0],
            "cases": connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0],
            "assets": len(assets),
            "asset_bytes": sum(item["size"] for item in assets),
        }
        return metadata, assets, counts
    except (sqlite3.DatabaseError, TargetStoreError) as exc:
        raise StoreCopyError("source SQLite store audit failed") from exc
    finally:
        connection.close()


def _copy_asset(source: Path, destination: Path, digest: str, size: int) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        source_descriptor = os.open(source, flags)
    except OSError as exc:
        raise StoreCopyError(f"source CAS asset is missing or unreadable: {digest}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StoreCopyError(f"source CAS asset is not a regular file: {digest}")
        checksum = hashlib.sha256()
        copied = 0
        with os.fdopen(source_descriptor, "rb", closefd=False) as input_stream:
            with destination.open("xb") as output_stream:
                for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                    output_stream.write(block)
                    checksum.update(block)
                    copied += len(block)
                output_stream.flush()
                os.fsync(output_stream.fileno())
        after = os.fstat(source_descriptor)
        try:
            current = os.stat(source, follow_symlinks=False)
        except OSError as exc:
            raise StoreCopyError(f"source CAS asset changed while copied: {digest}") from exc
        stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(current, field)
            for field in stable
        ):
            raise StoreCopyError(f"source CAS asset changed while copied: {digest}")
        if checksum.hexdigest() != digest or copied != size:
            raise StoreCopyError(f"source CAS asset failed integrity verification: {digest}")
    finally:
        os.close(source_descriptor)


def _audit_cas(root: Path, assets: Iterable[Mapping[str, Any]]) -> None:
    expected: set[str] = set()
    for asset in assets:
        relative = str(asset["relative_path"])
        expected.add(relative)
        path = root / relative
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise StoreCopyError(f"staged CAS asset is missing: {asset['sha256']}") from exc
        checksum = hashlib.sha256()
        size = 0
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode):
                raise StoreCopyError("staged CAS contains a non-regular object")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
                    size += len(block)
        finally:
            os.close(descriptor)
        if checksum.hexdigest() != asset["sha256"] or size != asset["size"]:
            raise StoreCopyError(f"staged CAS asset failed audit: {asset['sha256']}")

    actual: set[str] = set()
    for path in root.rglob("*"):
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise StoreCopyError("staged CAS contains a symbolic link")
        if stat.S_ISREG(status.st_mode):
            actual.add(path.relative_to(root).as_posix())
        elif not stat.S_ISDIR(status.st_mode):
            raise StoreCopyError("staged CAS contains a non-file object")
    if actual != expected:
        raise StoreCopyError("staged CAS does not exactly match the asset registry")


def _backup_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(str(destination))
    try:
        source_connection.execute("PRAGMA query_only=ON")
        source_connection.backup(destination_connection)
        destination_connection.commit()
    except sqlite3.DatabaseError as exc:
        raise StoreCopyError("SQLite online backup failed") from exc
    finally:
        destination_connection.close()
        source_connection.close()
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _package_root(value: str | Path) -> Path:
    package = Path(value).expanduser().resolve()
    if (
        not package.is_dir()
        or not (package / "__init__.py").is_file()
        or not (package / "dataset").is_dir()
    ):
        raise StoreCopyError(
            "target must be an existing Hotpot package directory containing "
            "__init__.py and dataset/"
        )
    running_package = Path(__file__).resolve().parents[1]
    if _runtime_contract(package) != _runtime_contract(running_package):
        raise StoreCopyError(
            "target Hotpot package does not use this exact current ds runtime"
        )
    return package


def copy_store(source_db: str | Path, target_package: str | Path) -> dict[str, Any]:
    """Copy an ingested current store into another Hotpot package atomically."""

    source = Path(source_db).expanduser().resolve()
    if not source.is_file():
        raise StoreCopyError("source database does not exist")
    source_assets = Path(str(source) + ".assets").resolve()
    package = _package_root(target_package)
    store_root = package / "dataset" / "data" / "store"
    active = store_root / "active"
    generations = store_root / "generations"
    destination = active / "datasets.sqlite3"

    store_root.mkdir(parents=True, exist_ok=True)
    lock_path = store_root / ".copy.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if os.path.lexists(active):
            raise StoreCopyError(
                "target active store already exists; copy never overwrites or merges"
            )

        generations.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".replica-", dir=generations))
        final = generations / f"replica-{uuid.uuid4().hex}"
        published = False
        try:
            staged_database = staging / "datasets.sqlite3"
            staged_assets = Path(str(staged_database) + ".assets")
            staged_assets.mkdir()
            _backup_database(source, staged_database)
            metadata, assets, counts = _audit_database(staged_database)
            if assets and not source_assets.is_dir():
                raise StoreCopyError("source CAS asset directory is missing")
            for asset in assets:
                relative = Path(asset["relative_path"])
                _copy_asset(
                    source_assets / relative,
                    staged_assets / relative,
                    asset["sha256"],
                    asset["size"],
                )
            _audit_cas(staged_assets, assets)
            _fsync_directory_tree(staged_assets)
            _fsync_directory(staging)

            staging.rename(final)
            _fsync_directory(generations)
            relative_generation = Path("generations") / final.name
            try:
                active.symlink_to(relative_generation, target_is_directory=True)
            except FileExistsError as exc:
                raise StoreCopyError("target active store appeared during publication") from exc
            _fsync_directory(store_root)
            published = True
            return {
                "source_database": str(source),
                "destination_database": str(destination),
                "generation_database": str(final / "datasets.sqlite3"),
                "target_package": str(package),
                "store_uuid": metadata["store_uuid"],
                "schema_id": metadata["schema_id"],
                "counts": counts,
                "audit": {"passed": True},
            }
        finally:
            if not published:
                if active.is_symlink() and active.readlink() == Path("generations") / final.name:
                    active.unlink()
                if staging.exists():
                    shutil.rmtree(staging)
                if final.exists():
                    shutil.rmtree(final)


__all__ = ["StoreCopyError", "copy_store"]
