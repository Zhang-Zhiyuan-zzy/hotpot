"""Private serialization and integrity checks for Bundle certificates.

Assessment belongs to :meth:`Bundle.synchronize`; this module only binds the
already synchronized facts, policy, generation, and scientific assets.  It is
deliberately not a public assessment API.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Union

from ._implementation import (
    PolicyProtocolError,
    require_current_runtime_policy,
    runtime_policy_is_current,
    validate_frozen_runtime_policy,
)
from .case import UnsynchronizedBundleError


CERTIFICATE_SCHEMA = "hotpot-ds-certificate/current"
ASSESSOR_VERSION = "hotpot-ds-assessor/current"

_ENVELOPE_FIELDS = {"schema", "certificate", "envelope_sha256"}
_CERTIFICATE_FIELDS = {
    "assessor_version",
    "dataset",
    "release",
    "target_context",
    "facts",
    "source",
    "policy",
    "generation",
    "runtime_policy",
    "checker_artifacts",
    "scientific_assets",
    "summary",
    "issuer",
    "issued_at",
}
_RELEASE_FIELDS = {"dataset_id", "release_id", "version", "release_serial"}
_FILE_FIELDS = {"path", "sha256", "size"}
_REVISION_FIELDS = {"revision", "digest"}
_GENERATION_FIELDS = {"id", "digest", "manifest"}
_CHECKER_FIELDS = {
    "path",
    "sha256",
    "size",
    "entrypoint",
    "checker_protocol_digest",
}
_ASSET_CONTAINER_FIELDS = {"items", "digest"}
_ASSET_FIELDS = {
    "case_id",
    "declaration_id",
    "path",
    "state",
    "sha256",
    "size",
}
_ASSET_STATES = {"present", "missing", "remote_not_materialized"}


class CertificateError(RuntimeError):
    """Base error for certificate creation and validation."""


class CertificateFormatError(CertificateError):
    """The certificate JSON does not use the one current closed schema."""


class CertificateIntegrityError(CertificateError):
    """Certificate-bound state or bytes no longer match."""


class CertificatePathError(CertificateIntegrityError):
    """A certificate-bound path is unsafe or is not the root certificate."""


@dataclass(frozen=True)
class _FileStamp:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _ObservedFile:
    path: Path
    stamp: _FileStamp


@dataclass(frozen=True)
class _BoundFile:
    path: str
    sha256: str
    size: int

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class _StateToken:
    manifest: bytes
    policy: bytes
    target_context: bytes
    control: tuple[tuple[str, Any], ...]
    governance_revision: int
    governance_digest: str


@dataclass(frozen=True)
class _CertificationSnapshot:
    root: Path
    state_token: _StateToken
    release: bytes
    target_context: bytes
    facts: _BoundFile
    source_revision: int
    source_digest: str
    policy_revision: int
    policy_digest: str
    generation_id: str
    generation_digest: str
    generation_manifest: _BoundFile
    runtime_policy: bytes
    checker_artifacts: bytes
    scientific_assets: bytes
    summary: bytes
    issuer: str
    issued_at: str
    generation_directory: Optional[Path]
    generation_tree: tuple[tuple[str, _FileStamp], ...]
    observed_files: tuple[_ObservedFile, ...]
    absent_paths: tuple[Path, ...]

    def certificate(self) -> Dict[str, Any]:
        assets = json.loads(self.scientific_assets)
        return {
            "assessor_version": ASSESSOR_VERSION,
            "dataset": json.loads(self.release)["dataset_id"],
            "release": json.loads(self.release),
            "target_context": json.loads(self.target_context),
            "facts": self.facts.to_dict(),
            "source": {
                "revision": self.source_revision,
                "digest": self.source_digest,
            },
            "policy": {
                "revision": self.policy_revision,
                "digest": self.policy_digest,
            },
            "generation": {
                "id": self.generation_id,
                "digest": self.generation_digest,
                "manifest": self.generation_manifest.to_dict(),
            },
            "runtime_policy": json.loads(self.runtime_policy),
            "checker_artifacts": json.loads(self.checker_artifacts),
            "scientific_assets": {
                "items": assets,
                "digest": _digest_value(assets),
            },
            "summary": json.loads(self.summary),
            "issuer": self.issuer,
            "issued_at": self.issued_at,
        }


def _file_stamp(metadata: os.stat_result) -> _FileStamp:
    return _FileStamp(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _current_file_stamp(path: Path) -> _FileStamp:
    try:
        metadata = path.stat()
    except OSError as error:
        raise CertificateIntegrityError(
            f"certificate-bound file became unavailable: {path}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise CertificateIntegrityError(
            f"certificate-bound path is not a regular file: {path}"
        )
    return _file_stamp(metadata)


def _stable_hash_file(path: Path) -> tuple[str, int, _FileStamp]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            before = _file_stamp(os.fstat(stream.fileno()))
            if not stat.S_ISREG(before.mode):
                raise CertificateIntegrityError(
                    f"certificate-bound path is not a regular file: {path}"
                )
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            after = _file_stamp(os.fstat(stream.fileno()))
        current = _current_file_stamp(path)
    except CertificateError:
        raise
    except OSError as error:
        raise CertificateIntegrityError(
            f"cannot read certificate-bound file: {path}"
        ) from error
    if before != after or after != current or size != after.size:
        raise CertificateIntegrityError(
            f"certificate-bound file changed while it was read: {path}"
        )
    return digest.hexdigest(), size, after


class _SnapshotHasher:
    """Hash every logical path once and retain cheap pre-publication stamps."""

    def __init__(self, root: Path):
        self.root = root
        self._files: Dict[Path, tuple[str, int, _FileStamp]] = {}
        self._absent: set[Path] = set()

    @staticmethod
    def _key(path: Path) -> Path:
        return Path(os.path.abspath(path))

    def digest(self, path: Path) -> tuple[str, int]:
        key = self._key(path)
        observed = self._files.get(key)
        if observed is None or not observed[0]:
            hashed = _stable_hash_file(key)
            if observed is not None and observed[2] != hashed[2]:
                raise CertificateIntegrityError(
                    f"certificate-bound file changed during validation: {path}"
                )
            observed = hashed
            self._files[key] = hashed
        return observed[0], observed[1]

    def descriptor(self, path: Path, name: str) -> _BoundFile:
        relative = _relative_path(self.root, path, name)
        digest, size = self.digest(path)
        return _BoundFile(relative, digest, size)

    def observe_validated(self, path: Path, stamp: _FileStamp) -> None:
        key = self._key(path)
        existing = self._files.get(key)
        if existing is not None and existing[2] != stamp:
            raise CertificateIntegrityError(
                f"certificate-bound file changed during validation: {path}"
            )
        if existing is None:
            self._files[key] = ("", stamp.size, stamp)

    def observe_absent(self, path: Path) -> None:
        key = self._key(path)
        if os.path.lexists(key):
            raise CertificateIntegrityError(
                f"certified-missing scientific asset appeared: {path}"
            )
        self._absent.add(key)

    @property
    def observed_files(self) -> tuple[_ObservedFile, ...]:
        return tuple(
            _ObservedFile(path, value[2])
            for path, value in sorted(
                self._files.items(), key=lambda item: item[0].as_posix()
            )
        )

    @property
    def absent_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self._absent, key=Path.as_posix))


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CertificateFormatError(
            "certificate content is not canonical JSON"
        ) from error


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value))


def _digest_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    digest, size, _ = _stable_hash_file(path)
    return digest, size


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _closed_mapping(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CertificateFormatError(
            f"{name} fields must be exactly {sorted(fields)}"
        )
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CertificateFormatError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    text = _nonempty_text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CertificateFormatError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _size(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CertificateFormatError(f"{name} must be a non-negative integer")
    return value


def _revision(value: Any, name: str) -> int:
    return _size(value, name)


def _safe_path(root: Path, value: Any, name: str) -> Path:
    text = _nonempty_text(value, name)
    if "\\" in text or "\x00" in text:
        raise CertificatePathError(f"{name} must be a safe POSIX relative path")
    relative = PurePosixPath(text)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise CertificatePathError(f"{name} must be a safe POSIX relative path")
    path = root.joinpath(*relative.parts)
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise CertificatePathError(f"{name} escapes the Bundle root") from error
    return path


def _relative_path(root: Path, path: Path, name: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError) as error:
        raise CertificatePathError(f"{name} is outside the Bundle root") from error
    _safe_path(root, relative, name)
    return relative


def _file_descriptor(root: Path, path: Path, name: str) -> Dict[str, Any]:
    relative = _relative_path(root, path, name)
    if not path.is_file():
        raise CertificateIntegrityError(f"{name} is not a regular file: {path}")
    digest, size = _hash_file(path)
    return {"path": relative, "sha256": digest, "size": size}


def _verify_file_descriptor(
    root: Path, value: Any, name: str, *, expected_path: Optional[str] = None
) -> Path:
    descriptor = _closed_mapping(value, _FILE_FIELDS, name)
    path_text = _nonempty_text(descriptor["path"], f"{name}.path")
    if expected_path is not None and path_text != expected_path:
        raise CertificateIntegrityError(
            f"{name}.path does not name {expected_path!r}"
        )
    path = _safe_path(root, path_text, f"{name}.path")
    expected_digest = _sha256(descriptor["sha256"], f"{name}.sha256")
    expected_size = _size(descriptor["size"], f"{name}.size")
    if not path.is_file():
        raise CertificateIntegrityError(f"certificate-bound file is missing: {path_text}")
    digest, size = _hash_file(path)
    if digest != expected_digest or size != expected_size:
        raise CertificateIntegrityError(
            f"certificate-bound file changed: {path_text}"
        )
    return path


def _checker_definitions(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("kind") == "custom":
            yield value
        for child in value.values():
            yield from _checker_definitions(child)
    elif isinstance(value, list):
        for child in value:
            yield from _checker_definitions(child)


def _checker_inventory(
    root: Path,
    policy: Mapping[str, Any],
    *,
    hasher: Optional[_SnapshotHasher] = None,
) -> list[Dict[str, Any]]:
    inventory = []
    seen = set()
    for definition in _checker_definitions(policy):
        source_digest = _sha256(
            definition.get("source_sha256"), "custom checker source_sha256"
        )
        entrypoint = _nonempty_text(
            definition.get("entrypoint"), "custom checker entrypoint"
        )
        protocol_digest = _sha256(
            definition.get("checker_protocol_digest"),
            "custom checker protocol digest",
        )
        source = definition.get("source")
        if (
            not isinstance(source, str)
            or hashlib.sha256(source.encode("utf-8")).hexdigest() != source_digest
        ):
            raise CertificateIntegrityError(
                "custom checker source does not match its policy digest"
            )
        key = (source_digest, entrypoint, protocol_digest)
        if key in seen:
            continue
        seen.add(key)
        path = root / "policy" / "checkers" / f"{source_digest}.py"
        descriptor = (
            _file_descriptor(root, path, "custom checker artifact")
            if hasher is None
            else hasher.descriptor(path, "custom checker artifact").to_dict()
        )
        if descriptor["sha256"] != source_digest:
            raise CertificateIntegrityError(
                "custom checker artifact does not match its policy digest"
            )
        inventory.append(
            {
                **descriptor,
                "entrypoint": entrypoint,
                "checker_protocol_digest": protocol_digest,
            }
        )
    return sorted(
        inventory,
        key=lambda item: (
            item["path"],
            item["entrypoint"],
            item["checker_protocol_digest"],
        ),
    )


def _normalize_assets(value: Any) -> list[Dict[str, Any]]:
    if not isinstance(value, list):
        raise CertificateFormatError("scientific asset inventory must be a list")
    inventory = []
    seen = set()
    for index, raw in enumerate(value):
        name = f"scientific_assets.items[{index}]"
        item = _closed_mapping(raw, _ASSET_FIELDS, name)
        state = item["state"]
        if state not in _ASSET_STATES:
            raise CertificateFormatError(f"{name}.state is not supported")
        case_id = _nonempty_text(item["case_id"], f"{name}.case_id")
        declaration_id = _nonempty_text(
            item["declaration_id"], f"{name}.declaration_id"
        )
        path = item["path"]
        digest = item["sha256"]
        size = item["size"]
        if state == "present":
            _nonempty_text(path, f"{name}.path")
            _sha256(digest, f"{name}.sha256")
            _size(size, f"{name}.size")
        elif state == "missing":
            _nonempty_text(path, f"{name}.path")
            if digest is not None or size is not None:
                raise CertificateFormatError(
                    f"{name} missing state cannot claim bytes"
                )
        else:
            if path is not None or size is not None:
                raise CertificateFormatError(
                    f"{name} remote_not_materialized state cannot claim a local file"
                )
            if digest is not None:
                _sha256(digest, f"{name}.sha256")
        normalized = {
            "case_id": case_id,
            "declaration_id": declaration_id,
            "path": path,
            "state": state,
            "sha256": digest,
            "size": size,
        }
        key = (case_id, declaration_id, path, state)
        if key in seen:
            raise CertificateFormatError(
                "scientific asset inventory contains a duplicate declaration state"
            )
        seen.add(key)
        inventory.append(normalized)
    return sorted(
        inventory,
        key=lambda item: (
            item["case_id"],
            item["declaration_id"],
            "" if item["path"] is None else item["path"],
            item["state"],
        ),
    )


def _verify_asset_files(root: Path, inventory: list[Dict[str, Any]]) -> None:
    for item in inventory:
        path_text = item["path"]
        if item["state"] == "remote_not_materialized":
            continue
        path = _safe_path(root, path_text, "scientific asset path")
        if item["state"] == "missing":
            if os.path.lexists(path):
                raise CertificateIntegrityError(
                    f"certified-missing scientific asset appeared: {path_text}"
                )
            continue
        if not path.is_file():
            raise CertificateIntegrityError(
                f"certificate-bound scientific asset is missing: {path_text}"
            )
        digest, size = _hash_file(path)
        if digest != item["sha256"] or size != item["size"]:
            raise CertificateIntegrityError(
                f"certificate-bound scientific asset changed: {path_text}"
            )


def _release_identity(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_id": manifest["dataset_id"],
        "release_id": manifest["release_id"],
        "version": manifest["version"],
        "release_serial": manifest.get("release_serial"),
    }


def _control_token(control: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(
        sorted(
            (key, value)
            for key, value in control.items()
            if key != "singleton"
        )
    )


def _asset_inventory_from_declarations(
    root: Path,
    declarations: list[tuple[str, str, Mapping[str, Any]]],
    hasher: _SnapshotHasher,
) -> list[Dict[str, Any]]:
    inventory = []
    for case_id, declaration_id, descriptor in declarations:
        path_text = descriptor.get("path")
        if not path_text:
            inventory.append(
                {
                    "case_id": case_id,
                    "declaration_id": declaration_id,
                    "path": None,
                    "state": "remote_not_materialized",
                    "sha256": descriptor.get("expected_sha256"),
                    "size": None,
                }
            )
            continue
        path = _safe_path(root, path_text, "scientific asset path")
        if not path.is_file():
            hasher.observe_absent(path)
            inventory.append(
                {
                    "case_id": case_id,
                    "declaration_id": declaration_id,
                    "path": path_text,
                    "state": "missing",
                    "sha256": None,
                    "size": None,
                }
            )
            continue
        digest, size = hasher.digest(path)
        inventory.append(
            {
                "case_id": case_id,
                "declaration_id": declaration_id,
                "path": path_text,
                "state": "present",
                "sha256": digest,
                "size": size,
            }
        )
    return _normalize_assets(inventory)


def _manifest_snapshot(
    root: Path,
    value: Any,
    ds_name: str,
) -> Dict[str, Any]:
    from ._bundle_state import _validate_manifest

    return _validate_manifest(root, value, ds_name)


def _capture_canonical_state(
    bundle: Any, hasher: _SnapshotHasher
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    str,
    list[Dict[str, Any]],
    int,
    str,
]:
    from ._bundle_state import (
        BundleStateError,
        _ASSET_FIELDS,
        _trusted_governance_binding,
    )
    from .bundle import _asset_descriptors, _record_id

    root = Path(bundle.root).resolve()
    state = bundle._state
    facts_path = root / "cases" / "facts.sqlite3"
    bundle_manifest_path = root / "bundle.json"
    facts_before = _current_file_stamp(facts_path)
    manifest_before = _current_file_stamp(bundle_manifest_path)
    declarations: list[tuple[str, str, Mapping[str, Any]]] = []

    def observe_case(case: Mapping[str, Any]) -> None:
        for collection in _ASSET_FIELDS:
            for record in case.get(collection, []):
                declaration_id = _record_id(record, collection)
                declarations.extend(
                    (str(case["id"]), declaration_id, descriptor)
                    for descriptor in _asset_descriptors(record)
                )

    try:
        connection = sqlite3.connect(
            f"file:{facts_path.as_posix()}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        stored_manifest = json.loads(
            connection.execute(
                "SELECT value_json FROM metadata WHERE key='manifest'"
            ).fetchone()[0]
        )
        current_manifest = json.loads(
            bundle_manifest_path.read_text(encoding="utf-8")
        )
        manifest = _manifest_snapshot(root, stored_manifest, bundle.name)
        if manifest != _manifest_snapshot(root, current_manifest, bundle.name):
            raise CertificateIntegrityError(
                "bundle.json differs from the immutable built manifest"
            )
        policy = json.loads(
            connection.execute(
                "SELECT policy_json FROM policy WHERE singleton=1"
            ).fetchone()[0]
        )
        target_context = json.loads(
            connection.execute(
                "SELECT value_json FROM metadata WHERE key='target_context'"
            ).fetchone()[0]
        )
        control = dict(
            connection.execute(
                "SELECT * FROM control WHERE singleton=1"
            ).fetchone()
        )
        source_digest = state._compute_source_proof(
            connection,
            manifest,
            file_digest=hasher.digest,
            case_observer=observe_case,
        )
    except CertificateError:
        raise
    except BundleStateError as error:
        raise CertificateIntegrityError(
            "canonical Bundle state failed certification"
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError, sqlite3.Error) as error:
        raise CertificateIntegrityError(
            "canonical Bundle state cannot be read for certification"
        ) from error
    finally:
        if "connection" in locals():
            connection.close()
    manifest_after = _current_file_stamp(bundle_manifest_path)
    if manifest_before != manifest_after:
        raise CertificateIntegrityError(
            "bundle manifest changed during certification"
        )
    hasher.observe_validated(bundle_manifest_path, manifest_after)
    assets = _asset_inventory_from_declarations(root, declarations, hasher)
    policy_digest = _digest_value(policy)
    governance_revision, governance_digest = _trusted_governance_binding()
    synchronized = bool(
        control["current_generation_id"]
        and control["source_digest_state"] == "current"
        and runtime_policy_is_current(policy.get("runtime_policy", {}))
        and control["source_revision"] == control["synchronized_source_revision"]
        and source_digest == control["source_digest"]
        and source_digest == control["synchronized_source_digest"]
        and control["policy_revision"] == control["synchronized_policy_revision"]
        and policy_digest == control["policy_digest"]
        and policy_digest == control["synchronized_policy_digest"]
        and governance_revision == control["governance_registry_revision"]
        and governance_digest == control["governance_registry_digest"]
        and governance_revision
        == control["synchronized_governance_registry_revision"]
        and governance_digest
        == control["synchronized_governance_registry_digest"]
    )
    if not synchronized:
        raise UnsynchronizedBundleError(
            "Bundle is not synchronized; run bundle.synchronize() first"
        )
    facts = hasher.descriptor(facts_path, "facts store")
    if facts.path != "cases/facts.sqlite3" or facts_before != _current_file_stamp(
        facts_path
    ):
        raise CertificateIntegrityError(
            "facts store changed during certification"
        )
    return (
        manifest,
        policy,
        target_context,
        control,
        source_digest,
        assets,
        governance_revision,
        governance_digest,
    )


def _generation_tree_stamps(directory: Path) -> tuple[tuple[str, _FileStamp], ...]:
    try:
        paths = sorted(
            (path for path in directory.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(directory).as_posix(),
        )
    except OSError as error:
        raise CertificateIntegrityError(
            "assessment generation changed during certification"
        ) from error
    return tuple(
        (path.relative_to(directory).as_posix(), _current_file_stamp(path))
        for path in paths
    )


@contextmanager
def _capture_current_generation(
    root: Path,
    control: Mapping[str, Any],
    hasher: _SnapshotHasher,
) -> Iterator[Any]:
    from ._generation import AssessmentGeneration, _GenerationLease

    generation_id = control["current_generation_id"]
    generation_digest = control["current_generation_digest"]
    directory = root / "assessments" / "generations" / generation_id
    with _GenerationLease(root, generation_id):
        before = _generation_tree_stamps(directory)
        generation = AssessmentGeneration._open_without_lease(
            root, generation_id, generation_digest
        )
        after = _generation_tree_stamps(directory)
        if before != after:
            raise CertificateIntegrityError(
                "assessment generation changed during certification"
            )
        for relative, stamp_value in after:
            hasher.observe_validated(directory / relative, stamp_value)
        yield generation, directory, after


@contextmanager
def _capture_supplied_generation(bundle: Any) -> Iterator[Any]:
    generation = bundle._current_generation
    if generation is None:
        raise UnsynchronizedBundleError(
            "Bundle has no current assessment generation; run bundle.synchronize() first"
        )
    with generation.lease():
        yield generation, None, ()


@contextmanager
def _certification_generation(
    bundle: Any,
    control: Mapping[str, Any],
    hasher: _SnapshotHasher,
) -> Iterator[Any]:
    if control.get("current_generation_id"):
        with _capture_current_generation(
            Path(bundle.root).resolve(), control, hasher
        ) as captured:
            yield captured
        return
    with _capture_supplied_generation(bundle) as captured:
        yield captured


def _capture_supplied_state(
    bundle: Any, hasher: _SnapshotHasher
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    str,
    list[Dict[str, Any]],
    int,
    str,
]:
    if not bundle.is_synchronized:
        raise UnsynchronizedBundleError(
            "Bundle is not synchronized; run bundle.synchronize() first"
        )
    state = bundle._state
    status = state.status
    declarations = _normalize_assets(bundle._scientific_asset_inventory())
    assets = []
    for item in declarations:
        current = dict(item)
        if item["state"] == "present":
            path = _safe_path(Path(bundle.root).resolve(), item["path"], "asset path")
            digest, size = hasher.digest(path)
            if digest != item["sha256"] or size != item["size"]:
                raise CertificateIntegrityError(
                    "scientific asset inventory does not match its bytes"
                )
            current.update({"sha256": digest, "size": size})
        elif item["state"] == "missing":
            hasher.observe_absent(
                _safe_path(Path(bundle.root).resolve(), item["path"], "asset path")
            )
        assets.append(current)
    control = {
        "source_revision": status["source_revision"],
        "source_digest": status["source_digest"],
        "policy_revision": status["policy_revision"],
        "policy_digest": status["policy_digest"],
        "current_generation_id": None,
        "current_generation_digest": None,
    }
    return (
        state.manifest,
        state.policy,
        state.target_context,
        control,
        status["source_digest"],
        assets,
        0,
        "",
    )


@contextmanager
def _capture_certification_snapshot(
    bundle: Any,
) -> Iterator[_CertificationSnapshot]:
    root = Path(bundle.root).resolve()
    state = bundle._state
    hasher = _SnapshotHasher(root)
    if hasattr(state, "connect") and hasattr(state, "_stored_manifest"):
        captured = _capture_canonical_state(bundle, hasher)
    else:
        captured = _capture_supplied_state(bundle, hasher)
    (
        manifest,
        policy,
        target_context,
        control,
        source_digest,
        assets,
        governance_revision,
        governance_digest,
    ) = captured
    require_current_runtime_policy(policy["runtime_policy"])
    with _certification_generation(bundle, control, hasher) as captured_generation:
        generation, generation_directory, generation_tree = captured_generation
        generation_manifest = generation.manifest
        expected_source = {
            "revision": control["source_revision"],
            "digest": source_digest,
        }
        expected_policy = {
            "revision": control["policy_revision"],
            "digest": _digest_value(policy),
        }
        generation_governance = generation_manifest.get("governance", {})
        if control.get("current_generation_id") and (
            generation_manifest["source"] != expected_source
            or generation_manifest["policy"] != expected_policy
            or generation_governance.get("revision") != governance_revision
            or generation_governance.get("digest") != governance_digest
        ):
            raise CertificateIntegrityError(
                "assessment generation does not match synchronized Bundle state"
            )
        runtime = validate_frozen_runtime_policy(
            generation_manifest["runtime_policy"]
        )
        if runtime != validate_frozen_runtime_policy(policy["runtime_policy"]):
            raise CertificateIntegrityError(
                "generation and policy runtime bindings differ"
            )
        manifest_binding = hasher.descriptor(
            Path(generation.manifest_path), "assessment generation manifest"
        )
        if control.get("current_generation_id"):
            summary = json.loads(
                (Path(generation.manifest_path).parent / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
        else:
            summary = generation.summary
        checkers = _checker_inventory(root, policy, hasher=hasher)
        facts = hasher.descriptor(
            root / "cases" / "facts.sqlite3", "facts store"
        )
        snapshot = _CertificationSnapshot(
            root=root,
            state_token=_StateToken(
                manifest=_canonical_bytes(manifest),
                policy=_canonical_bytes(policy),
                target_context=_canonical_bytes(target_context),
                control=_control_token(control),
                governance_revision=governance_revision,
                governance_digest=governance_digest,
            ),
            release=_canonical_bytes(_release_identity(manifest)),
            target_context=_canonical_bytes(target_context),
            facts=facts,
            source_revision=control["source_revision"],
            source_digest=source_digest,
            policy_revision=control["policy_revision"],
            policy_digest=_digest_value(policy),
            generation_id=generation.id,
            generation_digest=generation.digest,
            generation_manifest=manifest_binding,
            runtime_policy=_canonical_bytes(runtime),
            checker_artifacts=_canonical_bytes(checkers),
            scientific_assets=_canonical_bytes(assets),
            summary=_canonical_bytes(summary),
            issuer="Hotpot",
            issued_at=_utc_now(),
            generation_directory=generation_directory,
            generation_tree=generation_tree,
            observed_files=hasher.observed_files,
            absent_paths=hasher.absent_paths,
        )
        yield snapshot


def _validate_file_syntax(value: Any, name: str) -> None:
    descriptor = _closed_mapping(value, _FILE_FIELDS, name)
    _nonempty_text(descriptor["path"], f"{name}.path")
    _sha256(descriptor["sha256"], f"{name}.sha256")
    _size(descriptor["size"], f"{name}.size")


def _validate_certificate_syntax(certificate: Any, root: Path) -> Dict[str, Any]:
    value = _closed_mapping(certificate, _CERTIFICATE_FIELDS, "certificate")
    if value["assessor_version"] != ASSESSOR_VERSION:
        raise CertificateFormatError("unsupported certificate assessor version")
    _nonempty_text(value["dataset"], "certificate.dataset")
    release = _closed_mapping(value["release"], _RELEASE_FIELDS, "certificate.release")
    for field in ("dataset_id", "release_id", "version"):
        _nonempty_text(release[field], f"certificate.release.{field}")
    if release["dataset_id"] != value["dataset"]:
        raise CertificateFormatError(
            "certificate.release.dataset_id must equal certificate.dataset"
        )
    serial = release["release_serial"]
    if serial is not None and (
        not isinstance(serial, int)
        or isinstance(serial, bool)
        or not 1 <= serial <= 999
    ):
        raise CertificateFormatError(
            "certificate.release.release_serial must be null or an integer from 1 to 999"
        )
    if not isinstance(value["target_context"], Mapping):
        raise CertificateFormatError("certificate.target_context must be an object")
    _validate_file_syntax(value["facts"], "certificate.facts")
    if value["facts"]["path"] != "cases/facts.sqlite3":
        raise CertificatePathError("certificate facts path must be cases/facts.sqlite3")
    _safe_path(root, value["facts"]["path"], "certificate.facts.path")
    for field in ("source", "policy"):
        revision = _closed_mapping(
            value[field], _REVISION_FIELDS, f"certificate.{field}"
        )
        _revision(revision["revision"], f"certificate.{field}.revision")
        _sha256(revision["digest"], f"certificate.{field}.digest")
    generation = _closed_mapping(
        value["generation"], _GENERATION_FIELDS, "certificate.generation"
    )
    _nonempty_text(generation["id"], "certificate.generation.id")
    _sha256(generation["digest"], "certificate.generation.digest")
    _validate_file_syntax(
        generation["manifest"], "certificate.generation.manifest"
    )
    _safe_path(
        root,
        generation["manifest"]["path"],
        "certificate.generation.manifest.path",
    )
    try:
        runtime_policy = validate_frozen_runtime_policy(value["runtime_policy"])
    except PolicyProtocolError as error:
        raise CertificateIntegrityError(
            "certificate runtime policy is invalid"
        ) from error
    if not isinstance(value["checker_artifacts"], list):
        raise CertificateFormatError("certificate.checker_artifacts must be a list")
    checker_keys = set()
    for index, raw in enumerate(value["checker_artifacts"]):
        name = f"certificate.checker_artifacts[{index}]"
        checker = _closed_mapping(raw, _CHECKER_FIELDS, name)
        _safe_path(root, checker["path"], f"{name}.path")
        _sha256(checker["sha256"], f"{name}.sha256")
        _size(checker["size"], f"{name}.size")
        _nonempty_text(checker["entrypoint"], f"{name}.entrypoint")
        _sha256(
            checker["checker_protocol_digest"],
            f"{name}.checker_protocol_digest",
        )
        key = (
            checker["path"],
            checker["entrypoint"],
            checker["checker_protocol_digest"],
        )
        if key in checker_keys:
            raise CertificateFormatError("duplicate checker artifact binding")
        checker_keys.add(key)
    if value["checker_artifacts"] != sorted(
        value["checker_artifacts"],
        key=lambda item: (
            item["path"],
            item["entrypoint"],
            item["checker_protocol_digest"],
        ),
    ):
        raise CertificateFormatError(
            "certificate checker artifact inventory is not canonically ordered"
        )
    assets = _closed_mapping(
        value["scientific_assets"],
        _ASSET_CONTAINER_FIELDS,
        "certificate.scientific_assets",
    )
    items = _normalize_assets(assets["items"])
    if items != assets["items"]:
        raise CertificateFormatError(
            "certificate scientific asset inventory is not canonically ordered"
        )
    if (
        _sha256(
            assets["digest"], "certificate.scientific_assets.digest"
        )
        != _digest_value(items)
    ):
        raise CertificateIntegrityError(
            "scientific asset inventory digest does not match"
        )
    for item in items:
        if item["path"] is not None:
            _safe_path(root, item["path"], "scientific asset path")
    if not isinstance(value["summary"], Mapping):
        raise CertificateFormatError("certificate.summary must be an object")
    if value["issuer"] != "Hotpot":
        raise CertificateFormatError("unsupported certificate issuer")
    issued_at = _nonempty_text(value["issued_at"], "certificate.issued_at")
    if not issued_at.endswith("Z"):
        raise CertificateFormatError(
            "certificate.issued_at must be an ISO-8601 UTC timestamp"
        )
    try:
        parsed_time = datetime.fromisoformat(issued_at[:-1] + "+00:00")
    except ValueError as error:
        raise CertificateFormatError(
            "certificate.issued_at must be an ISO-8601 timestamp"
        ) from error
    if parsed_time.tzinfo is None:
        raise CertificateFormatError(
            "certificate.issued_at must be an ISO-8601 UTC timestamp"
        )
    return _json_copy(value)


def _load_json(path: Path) -> Any:
    def closed_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CertificateFormatError(f"duplicate JSON field: {key}")
            value[key] = item
        return value

    def invalid_constant(value: str) -> None:
        raise CertificateFormatError(f"invalid JSON number: {value}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=closed_object,
                parse_constant=invalid_constant,
            )
    except CertificateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CertificateFormatError(
            f"cannot read a valid Bundle certificate: {path}"
        ) from error


def _root_certificate(path: Union[str, Path]) -> tuple[Path, Path]:
    supplied = Path(path).expanduser().absolute()
    root = supplied.parent.resolve()
    expected = root / "certificate.json"
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise CertificateFormatError(
            f"Bundle certificate does not exist: {supplied}"
        ) from error
    if supplied.name != "certificate.json" or resolved != expected:
        raise CertificatePathError(
            "a Bundle certificate must be the regular root/certificate.json file"
        )
    if not expected.is_file() or expected.is_symlink():
        raise CertificatePathError(
            "a Bundle certificate must be the regular root/certificate.json file"
        )
    return root, expected


def load_certificate_envelope(path: Union[str, Path]) -> Dict[str, Any]:
    """Validate one root certificate envelope without constructing a Bundle."""

    root, certificate_path = _root_certificate(path)
    envelope = _load_json(certificate_path)
    value = _closed_mapping(envelope, _ENVELOPE_FIELDS, "certificate envelope")
    if value["schema"] != CERTIFICATE_SCHEMA:
        raise CertificateFormatError("unsupported certificate schema")
    certificate = _validate_certificate_syntax(value["certificate"], root)
    expected = _sha256(value["envelope_sha256"], "envelope_sha256")
    if _digest_value(certificate) != expected:
        raise CertificateIntegrityError("certificate envelope digest does not match")
    return {
        "schema": CERTIFICATE_SCHEMA,
        "certificate": certificate,
        "envelope_sha256": expected,
    }


def _verify_certificate_against_bundle(
    bundle: Any, certificate: Mapping[str, Any]
) -> None:
    if not bundle.is_synchronized:
        raise UnsynchronizedBundleError(
            "Bundle is not synchronized; run bundle.synchronize() first"
        )
    root = Path(bundle.root).resolve()
    state = bundle._state
    status = state.status
    generation = bundle._current_generation
    if generation is None:
        raise CertificateIntegrityError(
            "certificate names a Bundle without a current assessment generation"
        )
    if certificate["dataset"] != bundle.name:
        raise CertificateIntegrityError("certificate dataset does not match Bundle")
    if certificate["release"] != _release_identity(state.manifest):
        raise CertificateIntegrityError("certificate release does not match Bundle")
    if certificate["target_context"] != _json_copy(state.target_context):
        raise CertificateIntegrityError(
            "certificate target context does not match Bundle"
        )
    source = certificate["source"]
    if source != {
        "revision": status["source_revision"],
        "digest": status["source_digest"],
    }:
        raise CertificateIntegrityError("certificate source state is stale")
    policy = certificate["policy"]
    if policy != {
        "revision": status["policy_revision"],
        "digest": status["policy_digest"],
    }:
        raise CertificateIntegrityError("certificate policy state is stale")
    expected_generation = {
        "id": generation.id,
        "digest": generation.digest,
        "manifest": _file_descriptor(
            root, Path(generation.manifest_path), "assessment generation manifest"
        ),
    }
    if expected_generation["manifest"]["sha256"] != generation.manifest_digest:
        raise CertificateIntegrityError(
            "current generation manifest digest is inconsistent"
        )
    if certificate["generation"] != expected_generation:
        raise CertificateIntegrityError(
            "certificate assessment generation does not match Bundle"
        )
    try:
        frozen_runtime = validate_frozen_runtime_policy(
            state.policy["runtime_policy"]
        )
    except (KeyError, PolicyProtocolError) as error:
        raise CertificateIntegrityError(
            "Bundle policy runtime binding is invalid"
        ) from error
    if not (
        certificate["runtime_policy"]
        == generation.manifest["runtime_policy"]
        == frozen_runtime
    ):
        raise CertificateIntegrityError(
            "certificate, policy, and generation runtime bindings differ"
        )
    _verify_file_descriptor(
        root,
        certificate["facts"],
        "certificate.facts",
        expected_path="cases/facts.sqlite3",
    )
    expected_checkers = _checker_inventory(root, state.policy)
    if certificate["checker_artifacts"] != expected_checkers:
        raise CertificateIntegrityError(
            "certificate checker artifacts do not match Bundle policy"
        )
    for index, checker in enumerate(certificate["checker_artifacts"]):
        descriptor = {
            key: checker[key] for key in ("path", "sha256", "size")
        }
        _verify_file_descriptor(
            root, descriptor, f"certificate.checker_artifacts[{index}]"
        )
    certified_assets = certificate["scientific_assets"]["items"]
    current_assets = _normalize_assets(bundle._scientific_asset_inventory())
    if certified_assets != current_assets:
        raise CertificateIntegrityError(
            "certificate scientific asset inventory does not match Bundle"
        )
    _verify_asset_files(root, certified_assets)
    if certificate["summary"] != _json_copy(generation.summary):
        raise CertificateIntegrityError(
            "certificate assessment summary does not match Bundle"
        )


def _assert_observations_current(snapshot: _CertificationSnapshot) -> None:
    if (
        snapshot.generation_directory is not None
        and _generation_tree_stamps(snapshot.generation_directory)
        != snapshot.generation_tree
    ):
        raise CertificateIntegrityError(
            "assessment generation tree changed before certificate publication"
        )
    for observed in snapshot.observed_files:
        if _current_file_stamp(observed.path) != observed.stamp:
            raise CertificateIntegrityError(
                f"certificate-bound file changed before publication: {observed.path}"
            )
    for path in snapshot.absent_paths:
        if os.path.lexists(path):
            raise CertificateIntegrityError(
                f"certified-missing scientific asset appeared: {path}"
            )


def _current_state_token(bundle: Any, snapshot: _CertificationSnapshot) -> _StateToken:
    state = bundle._state
    if hasattr(state, "connect") and hasattr(state, "_stored_manifest"):
        from ._bundle_state import _trusted_governance_binding

        facts_path = Path(bundle.root).resolve() / "cases" / "facts.sqlite3"
        try:
            connection = sqlite3.connect(
                f"file:{facts_path.as_posix()}?mode=ro", uri=True
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            manifest = json.loads(
                connection.execute(
                    "SELECT value_json FROM metadata WHERE key='manifest'"
                ).fetchone()[0]
            )
            policy = json.loads(
                connection.execute(
                    "SELECT policy_json FROM policy WHERE singleton=1"
                ).fetchone()[0]
            )
            target_context = json.loads(
                connection.execute(
                    "SELECT value_json FROM metadata WHERE key='target_context'"
                ).fetchone()[0]
            )
            control = dict(
                connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
            )
        except (json.JSONDecodeError, sqlite3.Error) as error:
            raise CertificateIntegrityError(
                "canonical Bundle state changed before certificate publication"
            ) from error
        finally:
            if "connection" in locals():
                connection.close()
        governance_revision, governance_digest = _trusted_governance_binding()
        return _StateToken(
            manifest=_canonical_bytes(manifest),
            policy=_canonical_bytes(policy),
            target_context=_canonical_bytes(target_context),
            control=_control_token(control),
            governance_revision=governance_revision,
            governance_digest=governance_digest,
        )
    status = state.status
    expected_control = dict(snapshot.state_token.control)
    control = {key: status.get(key) for key in expected_control}
    return _StateToken(
        manifest=_canonical_bytes(state.manifest),
        policy=_canonical_bytes(state.policy),
        target_context=_canonical_bytes(state.target_context),
        control=_control_token(control),
        governance_revision=snapshot.state_token.governance_revision,
        governance_digest=snapshot.state_token.governance_digest,
    )


def _assert_snapshot_current(bundle: Any, snapshot: _CertificationSnapshot) -> None:
    _assert_observations_current(snapshot)
    if _current_state_token(bundle, snapshot) != snapshot.state_token:
        raise CertificateIntegrityError(
            "Bundle state changed before certificate publication"
        )
    _assert_observations_current(snapshot)


def _verify_certificate_against_snapshot(
    snapshot: _CertificationSnapshot, certificate: Mapping[str, Any]
) -> None:
    if certificate != snapshot.certificate():
        raise CertificateIntegrityError(
            "constructed certificate differs from its certification snapshot"
        )


@contextmanager
def _bundle_lock(root: Path) -> Iterator[None]:
    descriptor = os.open(root / ".bundle.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _atomic_write(
    path: Path,
    value: Mapping[str, Any],
    *,
    before_replace: Optional[Callable[[], None]] = None,
) -> None:
    payload = _canonical_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_bundle_certificate(bundle: Any) -> Path:
    """Atomically write the sole root certificate for a synchronized Bundle."""

    root = Path(bundle.root).resolve()
    with _bundle_lock(root):
        with _capture_certification_snapshot(bundle) as snapshot:
            certificate = _validate_certificate_syntax(
                snapshot.certificate(), root
            )
            _verify_certificate_against_snapshot(snapshot, certificate)
            envelope = {
                "schema": CERTIFICATE_SCHEMA,
                "certificate": certificate,
                "envelope_sha256": _digest_value(certificate),
            }
            path = root / "certificate.json"
            _atomic_write(
                path,
                envelope,
                before_replace=lambda: _assert_snapshot_current(bundle, snapshot),
            )
            return path


def verify_bundle_certificate(
    bundle: Any, path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Verify a root certificate against the current Bundle without mutation."""

    root = Path(bundle.root).resolve()
    certificate_path = root / "certificate.json" if path is None else Path(path)
    generation = bundle._current_generation
    with generation.lease():
        envelope = load_certificate_envelope(certificate_path)
        if certificate_path.expanduser().resolve() != root / "certificate.json":
            raise CertificatePathError(
                "certificate path does not name this Bundle's root certificate"
            )
        _verify_certificate_against_bundle(bundle, envelope["certificate"])
        return deepcopy(envelope)


def certificate_digest(path: Union[str, Path]) -> str:
    """Return the SHA-256 of the exact verified root-certificate bytes."""

    _, certificate_path = _root_certificate(path)
    load_certificate_envelope(certificate_path)
    return _hash_file(certificate_path)[0]


__all__ = [
    "CertificateError",
    "CertificateFormatError",
    "CertificateIntegrityError",
    "CertificatePathError",
    "certificate_digest",
    "load_certificate_envelope",
    "verify_bundle_certificate",
    "write_bundle_certificate",
]
