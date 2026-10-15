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
import uuid
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Optional, Union

from ._implementation import (
    PolicyProtocolError,
    require_current_runtime_policy,
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
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    except OSError as error:
        raise CertificateIntegrityError(
            f"cannot read certificate-bound file: {path}"
        ) from error
    return digest.hexdigest(), size


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


def _checker_inventory(root: Path, policy: Mapping[str, Any]) -> list[Dict[str, Any]]:
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
        descriptor = _file_descriptor(root, path, "custom checker artifact")
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


def _build_certificate(bundle: Any) -> Dict[str, Any]:
    root = Path(bundle.root).resolve()
    state = bundle._state
    status = state.status
    generation = bundle._current_generation
    if generation is None:
        raise UnsynchronizedBundleError(
            "Bundle has no current assessment generation; run bundle.synchronize() first"
        )
    manifest_path = Path(generation.manifest_path)
    manifest_descriptor = _file_descriptor(
        root, manifest_path, "assessment generation manifest"
    )
    if manifest_descriptor["sha256"] != generation.manifest_digest:
        raise CertificateIntegrityError(
            "assessment generation manifest digest does not match the generation"
        )
    assets = _normalize_assets(bundle._scientific_asset_inventory())
    _verify_asset_files(root, assets)
    policy = state.policy
    checkers = _checker_inventory(root, policy)
    facts = _file_descriptor(root, root / "cases" / "facts.sqlite3", "facts store")
    if facts["path"] != "cases/facts.sqlite3":
        raise CertificatePathError("facts store must be cases/facts.sqlite3")
    return {
        "assessor_version": ASSESSOR_VERSION,
        "dataset": bundle.name,
        "release": _release_identity(state.manifest),
        "target_context": _json_copy(state.target_context),
        "facts": facts,
        "source": {
            "revision": status["source_revision"],
            "digest": status["source_digest"],
        },
        "policy": {
            "revision": status["policy_revision"],
            "digest": status["policy_digest"],
        },
        "generation": {
            "id": generation.id,
            "digest": generation.digest,
            "manifest": manifest_descriptor,
        },
        "runtime_policy": validate_frozen_runtime_policy(
            generation.manifest["runtime_policy"]
        ),
        "checker_artifacts": checkers,
        "scientific_assets": {"items": assets, "digest": _digest_value(assets)},
        "summary": _json_copy(generation.summary),
        "issuer": "Hotpot",
        "issued_at": _utc_now(),
    }


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


@contextmanager
def _bundle_lock(root: Path) -> Iterator[None]:
    descriptor = os.open(root / ".bundle.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
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


def write_bundle_certificate(bundle: Any) -> Path:
    """Atomically write the sole root certificate for a synchronized Bundle."""

    root = Path(bundle.root).resolve()
    if not bundle.is_synchronized:
        raise UnsynchronizedBundleError(
            "Bundle is not synchronized; run bundle.synchronize() first"
        )
    require_current_runtime_policy(bundle._state.policy["runtime_policy"])
    with _bundle_lock(root):
        if not bundle.is_synchronized:
            raise UnsynchronizedBundleError(
                "Bundle changed before certification; run bundle.synchronize() first"
            )
        generation = bundle._current_generation
        with generation.lease():
            certificate = _validate_certificate_syntax(_build_certificate(bundle), root)
            _verify_certificate_against_bundle(bundle, certificate)
            envelope = {
                "schema": CERTIFICATE_SCHEMA,
                "certificate": certificate,
                "envelope_sha256": _digest_value(certificate),
            }
            path = root / "certificate.json"
            _atomic_write(path, envelope)
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
