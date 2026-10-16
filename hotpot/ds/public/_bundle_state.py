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
import stat
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Union

from ..gates._persistence import deserialize_gate_configs
from ._assessment import (
    validate_identity_schema,
    validate_observation,
    validate_property_policy,
)
from ._implementation import (
    PolicyProtocolError,
    require_current_runtime_policy,
    runtime_policy_is_current,
    validate_frozen_runtime_policy,
)


BUNDLE_SCHEMA = "hotpot-ds-bundle/current"
_DATABASE_SCHEMA_VERSION = 4
_SOURCE_PROOF_SCHEMA = "hotpot-ds-source-proof/current"
_SOURCE_DIGEST_STATES = frozenset({"current", "dirty"})
_CASE_INSERT_BATCH_SIZE = 512

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
_MANIFEST_ALLOWED = _MANIFEST_REQUIRED | {
    "dataset_id",
    "release_serial",
    "property_definitions",
    "property_comparators",
    "control_assets",
}
_CASE_ALLOWED = {
    "id",
    "local_id",
    "public_index",
    "identifiers",
    "metadata",
    "core_info",
    "entities",
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
    "entities": [],
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
    "entities",
    "structures",
    "observations",
    "claims",
    "evidence",
    "verifications",
    "attestations",
)
_ASSET_FIELDS = ("structures", "evidence")
_RECORD_ID_FIELDS = {
    "entities": "entity_id",
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


class DirtySourceStateError(BundleStateError):
    """Raised when a complete source proof is required but is not current."""


def _trusted_governance_binding() -> tuple[int, str]:
    """Return the current package-trusted registry revision and digest."""

    from ..governance import default_registry_snapshot, verify_registry

    verified = verify_registry(default_registry_snapshot())
    return verified.revision, verified.digest


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


def _validate_control_assets(root: Path, value: Any) -> list[Dict[str, Any]]:
    """Normalize declared controls without reading their payload bytes.

    Byte verification belongs exclusively to a complete source proof.  Keeping
    manifest validation structural prevents every SQLite connection from
    rehashing potentially large controls while still rejecting unsafe paths and
    ambiguous declarations.
    """

    if value is None:
        return []
    if not isinstance(value, list):
        raise BundleStateError("bundle.json.control_assets must be an array")
    result = []
    identities = set()
    paths = set()
    fields = {"category", "name", "path", "sha256", "size"}
    for index, raw in enumerate(value):
        name = f"bundle.json.control_assets[{index}]"
        if not isinstance(raw, dict) or set(raw) != fields:
            raise BundleStateError(f"{name} fields must be exactly {sorted(fields)}")
        category = _nonempty_string(raw["category"], f"{name}.category")
        asset_name = _nonempty_string(raw["name"], f"{name}.name")
        path = _safe_relative_path(root, raw["path"], f"{name}.path")
        digest = raw["sha256"]
        size = raw["size"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise BundleStateError(f"{name}.sha256 must be a lowercase SHA-256 digest")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BundleStateError(f"{name}.size must be a non-negative integer")
        identity = (category, asset_name)
        path_text = path.relative_to(root).as_posix()
        if identity in identities:
            raise BundleStateError("bundle.json.control_assets contains a duplicate category/name")
        if path_text in paths:
            raise BundleStateError("bundle.json.control_assets contains a duplicate path")
        identities.add(identity)
        paths.add(path_text)
        result.append(
            {
                "category": category,
                "name": asset_name,
                "path": path_text,
                "sha256": digest,
                "size": size,
            }
        )
    return sorted(result, key=lambda item: (item["category"], item["name"]))


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
    release_serial = value.get("release_serial")
    if release_serial is not None and (
        not isinstance(release_serial, int)
        or isinstance(release_serial, bool)
        or not 1 <= release_serial <= 999
    ):
        raise BundleStateError(
            "bundle.json.release_serial must be an integer from 1 to 999"
        )
    if not isinstance(value["provenance"], dict):
        raise BundleStateError("bundle.json.provenance must be an object")
    try:
        identity_schema = validate_identity_schema(value["identity_schema"])
    except ValueError as exc:
        raise BundleStateError(f"invalid bundle identity schema: {exc}") from exc
    for field in ("property_definitions", "property_comparators"):
        if field in value and not isinstance(value[field], dict):
            raise BundleStateError(f"bundle.json.{field} must be an object")
    property_definitions = value.get("property_definitions", {})
    property_comparators = value.get("property_comparators", {})
    try:
        validate_property_policy(property_definitions, property_comparators)
    except ValueError as exc:
        raise BundleStateError(f"invalid bundle property policy: {exc}") from exc
    _string_list(value["core_labels"], "bundle.json.core_labels")
    if property_definitions:
        missing_definitions = sorted(
            set(value["core_labels"]) - set(property_definitions)
        )
        if missing_definitions:
            raise BundleStateError(
                "bundle.json.core_labels lack property definitions: "
                + ", ".join(missing_definitions)
            )
    _string_list(value["core_info"], "bundle.json.core_info")
    _string_list(
        value["required_structure_roles"],
        "bundle.json.required_structure_roles",
    )
    control_assets = _validate_control_assets(root, value.get("control_assets"))
    intake = value["intake"]
    if not isinstance(intake, dict) or set(intake) != {"format", "path"}:
        raise BundleStateError("bundle.json.intake requires only format and path")
    if intake["format"] not in {"jsonl", "sqlite"}:
        raise BundleStateError("bundle.json.intake.format must be jsonl or sqlite")
    _safe_relative_path(root, intake["path"], "bundle.json.intake.path")
    normalized = _json_copy(value)
    normalized["dataset_id"] = dataset_id
    normalized["identity_schema"] = identity_schema
    if control_assets:
        normalized["control_assets"] = control_assets
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
    property_definitions: Optional[Mapping[str, Any]] = None,
    property_comparators: Optional[Mapping[str, Any]] = None,
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
    for record in case["observations"]:
        try:
            validate_observation(
                record,
                property_definitions or {},
                property_comparators or {},
            )
        except ValueError as exc:
            observation_id = record.get("observation_id", "<unknown>")
            raise BundleStateError(
                f"invalid observation {observation_id!r}: {exc}"
            ) from exc
    for attestation in case["attestations"]:
        if "signature_verified" in attestation:
            raise BundleStateError(
                "case.attestations.signature_verified is a derived trust result and is not input"
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


def _file_stamp(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _source_path_observation(
    path: Path,
) -> tuple[str, Optional[tuple[int, int, int, int, int, int]]]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return "missing", None
    except OSError as exc:
        raise BundleStateError(f"cannot observe source asset: {path}") from exc
    stamp = _file_stamp(metadata)
    if not stat.S_ISREG(metadata.st_mode):
        return "not-a-regular-file", stamp
    return "present", stamp


def _assert_source_observations_current(
    root: Path,
    observations: Mapping[
        str, tuple[str, Optional[tuple[int, int, int, int, int, int]]]
    ],
) -> None:
    for relative, observed in sorted(observations.items()):
        path = _safe_relative_path(root, relative, "asset path")
        if _source_path_observation(path) != observed:
            raise BundleStateError(
                f"source asset changed after aggregate proof: {path}"
            )


def _stable_file_hash(
    path: Path,
) -> tuple["hashlib._Hash", int, tuple[int, int, int, int, int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BundleStateError(f"cannot open stable source asset: {path}") from exc
    hasher = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BundleStateError(f"source asset is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
                size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_stamp = _file_stamp(before)
    after_stamp = _file_stamp(after)
    live_state, live_stamp = _source_path_observation(path)
    if (
        before_stamp != after_stamp
        or size != after.st_size
        or live_state != "present"
        or live_stamp != after_stamp
    ):
        raise BundleStateError(f"source asset changed while hashing: {path}")
    return hasher, size, after_stamp


def _digest_file(path: Path) -> tuple[str, int]:
    hasher, size, _ = _stable_file_hash(path)
    return hasher.hexdigest(), size


def _control_asset_source_binding(
    declaration: Mapping[str, Any],
    *,
    state: str,
    sha256: Optional[str],
    size: Optional[int],
) -> Dict[str, Any]:
    """Return the unambiguous control-asset value framed by source proofs."""

    return {
        "category": declaration["category"],
        "name": declaration["name"],
        "path": declaration["path"],
        "state": state,
        "sha256": sha256,
        "size": size,
    }


def _compute_source_proof_rows(
    root: Path,
    manifest: Mapping[str, Any],
    rows: Iterator[Mapping[str, Any]],
    *,
    file_digest: Callable[[Path], tuple[str, int]],
    case_observer: Optional[Callable[[Mapping[str, Any]], None]],
    scientific_paths: Optional[Iterator[str]] = None,
    observation_sink: Optional[
        Dict[str, tuple[str, Optional[tuple[int, int, int, int, int, int]]]]
    ] = None,
) -> str:
    """Compute the exact source proof from ordered canonical Case rows."""

    hasher = hashlib.sha256()
    _frame(hasher, "schema", _SOURCE_PROOF_SCHEMA.encode("utf-8"))
    _frame(hasher, "manifest", _canonical_bytes(manifest))
    discovered_paths: Optional[set[str]] = (
        set() if scientific_paths is None else None
    )
    for row in rows:
        payload = None
        if discovered_paths is not None or case_observer is not None:
            payload = json.loads(row["payload_json"])
        if case_observer is not None:
            case_observer(payload)
        _frame(
            hasher,
            f"case:{row['public_index']}:{row['case_id']}",
            row["payload_json"].encode("utf-8"),
        )
        if discovered_paths is not None:
            for field in _ASSET_FIELDS:
                for record in payload.get(field, []):
                    discovered_paths.update(_record_asset_paths(record))

    FileStamp = tuple[int, int, int, int, int, int]
    observations: Dict[
        str,
        tuple[str, Optional[str], Optional[int], Optional[FileStamp]],
    ] = {}

    def observe(relative: str) -> tuple[str, Optional[str], Optional[int]]:
        observed = observations.get(relative)
        if observed is not None:
            return observed[:3]
        path = _safe_relative_path(root, relative, "asset path")
        state, stamp = _source_path_observation(path)
        if state == "present":
            digest, size = file_digest(path)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef" for character in digest
                )
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
            ):
                raise BundleStateError(
                    "source proof file digest callback returned an invalid result"
                )
            current_state, current_stamp = _source_path_observation(path)
            if (
                current_state != "present"
                or current_stamp != stamp
                or size != stamp[3]
            ):
                raise BundleStateError(
                    f"source asset changed during aggregate proof: {path}"
                )
            observed = ("present", digest, size, stamp)
        else:
            observed = (state, None, None, stamp)
        observations[relative] = observed
        return observed[:3]

    for declaration in manifest.get("control_assets", []):
        state, digest, size = observe(declaration["path"])
        binding = _control_asset_source_binding(
            declaration,
            state=state,
            sha256=digest,
            size=size,
        )
        _frame(
            hasher,
            f"control:{declaration['category']}:{declaration['name']}",
            _canonical_bytes(binding),
        )
        if (
            state != "present"
            or digest != declaration["sha256"]
            or size != declaration["size"]
        ):
            raise BundleStateError(
                "declared control asset does not match its current bytes: "
                + declaration["path"]
            )

    paths = (
        sorted(discovered_paths)
        if discovered_paths is not None
        else scientific_paths or ()
    )
    for relative in paths:
        state, digest, _ = observe(relative)
        if state == "missing":
            _frame(hasher, f"asset:{relative}", b"missing")
        elif state == "not-a-regular-file":
            _frame(hasher, f"asset:{relative}", b"not-a-regular-file")
        else:
            _frame(
                hasher,
                f"asset:{relative}",
                b"present\0" + bytes.fromhex(digest or ""),
            )
    for relative, observed in sorted(observations.items()):
        path = _safe_relative_path(root, relative, "asset path")
        current_state, current_stamp = _source_path_observation(path)
        if (current_state, current_stamp) != (observed[0], observed[3]):
            raise BundleStateError(
                f"source asset changed during aggregate proof: {path}"
            )
    if observation_sink is not None:
        observation_sink.clear()
        observation_sink.update(
            (relative, (observed[0], observed[3]))
            for relative, observed in observations.items()
        )
    return hasher.hexdigest()


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


class _SynchronizationSnapshot(Mapping[str, Any]):
    """Immutable synchronization bindings plus repeatable ordered Case scans."""

    def __init__(
        self, values: Mapping[str, Any], connection: sqlite3.Connection
    ) -> None:
        self._values = _json_copy(dict(values))
        self._connection = connection

    def __getitem__(self, key: str) -> Any:
        return _json_copy(self._values[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def iter_cases(self) -> Iterator[Dict[str, Any]]:
        cursor = self._connection.execute(
            "SELECT payload_json FROM cases ORDER BY public_index,case_id"
        )
        for row in cursor:
            yield json.loads(row[0])


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
        for field in (
            "identity_schema",
            "required_structure_roles",
            "property_definitions",
            "property_comparators",
        ):
            manifest_value = manifest.get(field, {})
            if field in policy and policy[field] != manifest_value:
                raise BundleStateError(
                    f"persisted {field} must equal bundle.json.{field}"
                )
            policy[field] = _json_copy(manifest_value)
        if policy.get("schema") != "hotpot-ds-policy/current":
            raise BundleStateError("dataset policy schema is not current")
        try:
            deserialize_gate_configs(policy["gates"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BundleStateError("dataset Gate policy is not current") from exc
        try:
            require_current_runtime_policy(policy["runtime_policy"])
        except (KeyError, PolicyProtocolError) as exc:
            raise BundleStateError("dataset runtime policy is not current") from exc
        target = _json_copy(dict(target_context))
        policy_digest = _digest_json(policy)
        governance_revision, governance_digest = _trusted_governance_binding()
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
                connection.execute(
                    "CREATE TEMP TABLE build_source_assets("
                    "path TEXT PRIMARY KEY) WITHOUT ROWID"
                )
                batch = []
                asset_batch: set[str] = set()
                for case in state._iter_intake(manifest, intake):
                    batch.append(
                        (
                            case["id"],
                            case["public_index"],
                            case["local_id"],
                            _canonical_text(case),
                        )
                    )
                    for field in _ASSET_FIELDS:
                        for record in case.get(field, []):
                            asset_batch.update(_record_asset_paths(record))
                    if len(batch) == _CASE_INSERT_BATCH_SIZE:
                        connection.executemany(
                            """
                            INSERT INTO cases(
                                case_id, public_index, local_id, payload_json
                            ) VALUES (?, ?, ?, ?)
                            """,
                            batch,
                        )
                        connection.executemany(
                            "INSERT OR IGNORE INTO build_source_assets(path) VALUES (?)",
                            ((path,) for path in sorted(asset_batch)),
                        )
                        batch.clear()
                        asset_batch.clear()
                if batch:
                    connection.executemany(
                        """
                        INSERT INTO cases(
                            case_id, public_index, local_id, payload_json
                        ) VALUES (?, ?, ?, ?)
                        """,
                        batch,
                    )
                    connection.executemany(
                        "INSERT OR IGNORE INTO build_source_assets(path) VALUES (?)",
                        ((path,) for path in sorted(asset_batch)),
                    )
                source_digest = _compute_source_proof_rows(
                    state.root,
                    manifest,
                    connection.execute(
                        "SELECT case_id,public_index,payload_json FROM cases "
                        "ORDER BY public_index,case_id"
                    ),
                    file_digest=_digest_file,
                    case_observer=None,
                    scientific_paths=(
                        row[0]
                        for row in connection.execute(
                            "SELECT path FROM build_source_assets ORDER BY path"
                        )
                    ),
                )
                projection_row = {
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
                projection_path = state._write_audit_projection_rows(
                    [projection_row]
                )
                projection_digest, projection_size = _digest_file(projection_path)
                connection.execute(
                    """
                    INSERT INTO control(
                        singleton, source_revision, source_digest,
                        source_digest_state,
                        policy_revision, policy_digest,
                        governance_registry_revision,
                        governance_registry_digest,
                        audit_revision, audit_projection_revision,
                        audit_projection_size, audit_projection_digest
                    ) VALUES (1, 0, ?, 'current', 0, ?, ?, ?, 1, 1, ?, ?)
                    """,
                    (
                        source_digest,
                        policy_digest,
                        governance_revision,
                        governance_digest,
                        projection_size,
                        projection_digest,
                    ),
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
                        _canonical_text(projection_row["details"]),
                    ),
                )
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise BundleStateError("new facts.sqlite3 failed integrity_check")
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.close()
            connection = None
            state._write_policy_projection_value(policy)
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
        return state

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
                if (
                    connection.execute("PRAGMA user_version").fetchone()[0]
                    != _DATABASE_SCHEMA_VERSION
                ):
                    raise BundleStateError("facts.sqlite3 is not the current schema")
                stored_manifest = json.loads(
                    connection.execute(
                        "SELECT value_json FROM metadata WHERE key='manifest'"
                    ).fetchone()[0]
                )
                _validate_manifest(bundle_root, stored_manifest, ds_name)
                stored_policy = json.loads(
                    connection.execute(
                        "SELECT policy_json FROM policy WHERE singleton=1"
                    ).fetchone()[0]
                )
                if stored_policy.get("schema") != "hotpot-ds-policy/current":
                    raise BundleStateError("dataset policy schema is not current")
                try:
                    deserialize_gate_configs(stored_policy["gates"])
                    validate_identity_schema(stored_policy["identity_schema"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise BundleStateError("dataset assessment policy is invalid") from exc
                for field in (
                    "identity_schema",
                    "required_structure_roles",
                    "property_definitions",
                    "property_comparators",
                ):
                    if stored_policy.get(field) != stored_manifest.get(field, {}):
                        raise BundleStateError(
                            f"persisted {field} differs from the built manifest"
                        )
                try:
                    validate_frozen_runtime_policy(stored_policy["runtime_policy"])
                except (KeyError, PolicyProtocolError) as exc:
                    raise BundleStateError("dataset runtime policy is invalid") from exc
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
                        property_definitions=stored_manifest.get(
                            "property_definitions", {}
                        ),
                        property_comparators=stored_manifest.get(
                            "property_comparators", {}
                        ),
                    )
        except (sqlite3.Error, TypeError, KeyError, json.JSONDecodeError) as exc:
            if isinstance(exc, BundleStateError):
                raise
            raise BundleStateError("facts.sqlite3 is not a valid current Bundle store") from exc
        return state

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if not self.path.is_file():
            raise BundleNotBuiltError(
                "Bundle is not built; construct it with Bundle(...) first"
            )
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            if (
                connection.execute("PRAGMA user_version").fetchone()[0]
                != _DATABASE_SCHEMA_VERSION
            ):
                raise BundleStateError("facts.sqlite3 is not the current schema")
            self._assert_manifest_unchanged(connection)
            yield connection
        finally:
            connection.close()

    @property
    def manifest(self) -> Dict[str, Any]:
        with self.connect() as connection:
            return self._stored_manifest(connection)

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
        with self.connect() as connection:
            control = connection.execute(
                "SELECT source_digest,source_digest_state FROM control WHERE singleton=1"
            ).fetchone()
            if control["source_digest_state"] != "current":
                raise DirtySourceStateError(
                    "Bundle source digest is dirty; refresh source state first"
                )
            digest = self._compute_source_digest(
                connection, self._stored_manifest(connection)
            )
            return digest

    @property
    def policy_digest(self) -> str:
        return _digest_json(self.policy)

    @property
    def status(self) -> Dict[str, Any]:
        with self.connect() as connection:
            governance_revision, governance_digest = _trusted_governance_binding()
            policy_value = json.loads(
                connection.execute(
                    "SELECT policy_json FROM policy WHERE singleton=1"
                ).fetchone()[0]
            )
            policy_digest = _digest_json(policy_value)
            row = connection.execute("SELECT * FROM control WHERE singleton=1").fetchone()
            if row["source_digest_state"] == "current":
                source_digest = self._compute_source_digest(
                    connection, self._stored_manifest(connection)
                )
            else:
                source_digest = None
        status = dict(row)
        status["recorded_source_digest"] = status.pop("source_digest")
        status["recorded_policy_digest"] = status.pop("policy_digest")
        status["recorded_governance_registry_revision"] = status.pop(
            "governance_registry_revision"
        )
        status["recorded_governance_registry_digest"] = status.pop(
            "governance_registry_digest"
        )
        status["source_digest"] = source_digest
        status["source_integrity"] = (
            "dirty"
            if status["source_digest_state"] == "dirty"
            else (
                "current"
                if source_digest == status["recorded_source_digest"]
                else "changed_out_of_band"
            )
        )
        status["policy_digest"] = policy_digest
        status["governance_registry_revision"] = governance_revision
        status["governance_registry_digest"] = governance_digest
        status["runtime_policy_current"] = runtime_policy_is_current(
            policy_value.get("runtime_policy", {})
        )
        status["is_synchronized"] = bool(
            status["current_generation_id"]
            and status["runtime_policy_current"]
            and status["source_digest_state"] == "current"
            and status["source_revision"] == status["synchronized_source_revision"]
            and source_digest == status["recorded_source_digest"]
            and source_digest == status["synchronized_source_digest"]
            and status["policy_revision"] == status["synchronized_policy_revision"]
            and policy_digest == status["recorded_policy_digest"]
            and policy_digest == status["synchronized_policy_digest"]
            and governance_revision
            == status["recorded_governance_registry_revision"]
            and governance_digest
            == status["recorded_governance_registry_digest"]
            and governance_revision
            == status["synchronized_governance_registry_revision"]
            and governance_digest
            == status["synchronized_governance_registry_digest"]
        )
        projection = self.root / "audit" / "events.jsonl"
        try:
            projection_digest, projection_size = _digest_file(projection)
        except BundleStateError:
            projection_digest = None
            projection_size = None
        status["audit_projection_current"] = bool(
            status["audit_revision"] == status["audit_projection_revision"]
            and projection_size == status["audit_projection_size"]
            and projection_digest == status["audit_projection_digest"]
        )
        status.pop("singleton", None)
        return _json_copy(status)

    def refresh_source_state(
        self,
        *,
        _source_observations: Optional[
            Dict[str, tuple[str, Optional[tuple[int, int, int, int, int, int]]]]
        ] = None,
    ) -> Dict[str, Any]:
        """Create and persist a complete current source proof."""

        projection_event = None
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                digest = self._compute_source_digest(
                    connection,
                    self._stored_manifest(connection),
                    observation_sink=_source_observations,
                )
                if (
                    control["source_digest_state"] == "current"
                    and digest == control["source_digest"]
                ):
                    connection.rollback()
                    return {
                        "changed": False,
                        "source_revision": control["source_revision"],
                        "source_digest": digest,
                        "event_id": None,
                    }
                if control["source_digest_state"] == "dirty":
                    revision = control["source_revision"]
                    event_id = None
                    connection.execute(
                        "UPDATE control SET source_digest=?, "
                        "source_digest_state='current' WHERE singleton=1",
                        (digest,),
                    )
                else:
                    revision = control["source_revision"] + 1
                    audit_revision = control["audit_revision"] + 1
                    event_id = f"event-{uuid.uuid4().hex}"
                    created_at = _utc_now()
                    details = {
                        "previous_source_digest": control["source_digest"],
                        "current_source_digest": digest,
                    }
                    connection.execute(
                        "UPDATE control SET source_revision=?,source_digest=?, "
                        "source_digest_state='current',audit_revision=? "
                        "WHERE singleton=1",
                        (revision, digest, audit_revision),
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
                            created_at,
                            _canonical_text(details),
                            revision,
                            control["policy_revision"],
                        ),
                    )
                    projection_event = {
                        "event_id": event_id,
                        "created_at": created_at,
                        "action": "scientific_asset_state_changed",
                        "case_id": None,
                        "details": details,
                        "source_revision": revision,
                        "policy_revision": control["policy_revision"],
                    }
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        projection_updated = bool(
            projection_event
            and self._append_audit_projection_event(
                projection_event,
                expected_previous_revision=audit_revision - 1,
                revision=audit_revision,
            )
        )
        return {
            "changed": True,
            "source_revision": revision,
            "source_digest": digest,
            "event_id": event_id,
            "audit_projection_updated": projection_updated,
        }

    def refresh_governance_state(self) -> Dict[str, Any]:
        """Persist the latest trusted registry binding before reassessment."""

        projection_event = None
        with self.connect() as connection:
            revision, digest = _trusted_governance_binding()
            connection.execute("BEGIN IMMEDIATE")
            try:
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                previous_revision = control["governance_registry_revision"]
                previous_digest = control["governance_registry_digest"]
                if (revision, digest) == (previous_revision, previous_digest):
                    connection.rollback()
                    return {
                        "changed": False,
                        "governance_registry_revision": revision,
                        "governance_registry_digest": digest,
                        "event_id": None,
                    }
                if revision < previous_revision:
                    raise BundleStateError(
                        "trusted governance registry revision cannot regress"
                    )
                if revision == previous_revision:
                    raise BundleStateError(
                        "trusted governance registry digest changed without a revision"
                    )
                event_id = f"event-{uuid.uuid4().hex}"
                created_at = _utc_now()
                audit_revision = control["audit_revision"] + 1
                connection.execute(
                    "UPDATE control SET governance_registry_revision=?, "
                    "governance_registry_digest=?,audit_revision=? WHERE singleton=1",
                    (revision, digest, audit_revision),
                )
                details = {
                    "previous_revision": previous_revision,
                    "previous_digest": previous_digest,
                    "current_revision": revision,
                    "current_digest": digest,
                }
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, created_at, action, case_id, details_json,
                        source_revision, policy_revision
                    ) VALUES (?, ?, 'governance_registry_advanced', NULL, ?, ?, ?)
                    """,
                    (
                        event_id,
                        created_at,
                        _canonical_text(details),
                        control["source_revision"],
                        control["policy_revision"],
                    ),
                )
                projection_event = {
                    "event_id": event_id,
                    "created_at": created_at,
                    "action": "governance_registry_advanced",
                    "case_id": None,
                    "details": details,
                    "source_revision": control["source_revision"],
                    "policy_revision": control["policy_revision"],
                }
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        projection_updated = self._append_audit_projection_event(
            projection_event,
            expected_previous_revision=audit_revision - 1,
            revision=audit_revision,
        )
        return {
            "changed": True,
            "governance_registry_revision": revision,
            "governance_registry_digest": digest,
            "event_id": event_id,
            "audit_projection_updated": projection_updated,
        }

    def _prepare_synchronization(
        self,
        source_observations: Optional[
            Dict[str, tuple[str, Optional[tuple[int, int, int, int, int, int]]]]
        ],
    ) -> Dict[str, Any]:
        """Return one verified input snapshot without repeating the source proof."""

        prepared_source = self.refresh_source_state(
            _source_observations=source_observations
        )
        prepared_governance = self.refresh_governance_state()
        trusted_governance = _trusted_governance_binding()
        with self.connect() as connection:
            connection.execute("BEGIN")
            try:
                manifest = self._stored_manifest(connection)
                policy = json.loads(
                    connection.execute(
                        "SELECT policy_json FROM policy WHERE singleton=1"
                    ).fetchone()[0]
                )
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        if control["source_digest_state"] != "current":
            raise DirtySourceStateError(
                "Bundle source proof became dirty while synchronization was prepared"
            )
        if (
            control["source_revision"],
            control["source_digest"],
        ) != (
            prepared_source["source_revision"],
            prepared_source["source_digest"],
        ):
            raise ConcurrentStateError(
                "Bundle source inputs changed while synchronization was prepared"
            )
        policy_digest = _digest_json(policy)
        if policy_digest != control["policy_digest"]:
            raise BundleStateError(
                "persisted policy digest does not match the canonical policy"
            )
        require_current_runtime_policy(policy["runtime_policy"])
        governance = (
            control["governance_registry_revision"],
            control["governance_registry_digest"],
        )
        prepared_binding = (
            prepared_governance["governance_registry_revision"],
            prepared_governance["governance_registry_digest"],
        )
        if governance != prepared_binding or governance != trusted_governance:
            raise ConcurrentStateError(
                "trusted governance changed while synchronization was prepared"
            )
        return _json_copy(
            {
                "source_revision": control["source_revision"],
                "source_digest": control["source_digest"],
                "policy_revision": control["policy_revision"],
                "policy_digest": policy_digest,
                "governance_registry_revision": governance[0],
                "governance_registry_digest": governance[1],
                "manifest": manifest,
                "policy": policy,
            }
        )

    def prepare_synchronization(self) -> Dict[str, Any]:
        return self._prepare_synchronization(None)

    @contextmanager
    def synchronization_snapshot(self) -> Iterator[_SynchronizationSnapshot]:
        """Hold one repeatable read transaction across assessment input scans."""

        source_observations: Dict[
            str, tuple[str, Optional[tuple[int, int, int, int, int, int]]]
        ] = {}
        prepared = self._prepare_synchronization(source_observations)
        facts_observation = _source_path_observation(self.path)
        if facts_observation[0] != "present":
            raise BundleStateError("canonical facts store is not a regular file")
        with self.connect() as connection:
            connection.execute("BEGIN")
            try:
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                manifest = self._stored_manifest(connection)
                policy = json.loads(
                    connection.execute(
                        "SELECT policy_json FROM policy WHERE singleton=1"
                    ).fetchone()[0]
                )
                current = {
                    "source_revision": control["source_revision"],
                    "source_digest": control["source_digest"],
                    "policy_revision": control["policy_revision"],
                    "policy_digest": control["policy_digest"],
                    "governance_registry_revision": control[
                        "governance_registry_revision"
                    ],
                    "governance_registry_digest": control[
                        "governance_registry_digest"
                    ],
                    "manifest": manifest,
                    "policy": policy,
                }
                if (
                    control["source_digest_state"] != "current"
                    or current != prepared
                ):
                    raise ConcurrentStateError(
                        "Bundle inputs changed before the synchronization snapshot"
                    )
                if _source_path_observation(self.path) != facts_observation:
                    raise ConcurrentStateError(
                        "canonical facts changed before the synchronization snapshot"
                    )
                _assert_source_observations_current(
                    self.root, source_observations
                )
                try:
                    yield _SynchronizationSnapshot(prepared, connection)
                except BaseException:
                    raise
                else:
                    if _source_path_observation(self.path) != facts_observation:
                        raise ConcurrentStateError(
                            "canonical facts changed during the synchronization snapshot"
                        )
                    _assert_source_observations_current(
                        self.root, source_observations
                    )
            finally:
                connection.rollback()

    def mutate_case(
        self,
        case_id: str,
        action: str,
        mutation: Callable[[Dict[str, Any]], tuple[Mapping[str, Any], Any]],
    ) -> Dict[str, Any]:
        _nonempty_string(action, "action")
        projection_event = None
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM cases WHERE case_id=?", (case_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(case_id)
                old = json.loads(row["payload_json"])
                new_payload = _json_copy(old)
                details, result = mutation(new_payload)
                manifest = self._stored_manifest(connection)
                case = _validate_case(
                    self.root,
                    new_payload,
                    supplied_public_index=row["public_index"],
                    property_definitions=manifest.get(
                        "property_definitions", {}
                    ),
                    property_comparators=manifest.get(
                        "property_comparators", {}
                    ),
                )
                if case["id"] != case_id or case["local_id"] != row["local_id"]:
                    raise BundleStateError("case id and local_id are immutable")
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                revision = control["source_revision"] + 1
                audit_revision = control["audit_revision"] + 1
                connection.execute(
                    "UPDATE cases SET payload_json=? WHERE case_id=?",
                    (_canonical_text(case), case_id),
                )
                connection.execute(
                    "UPDATE control SET source_revision=?,source_digest=NULL,"
                    "source_digest_state='dirty',audit_revision=? WHERE singleton=1",
                    (revision, audit_revision),
                )
                event_details = _json_copy(dict(details))
                event_details.update(
                    {
                        "old_payload_digest": _digest_json(old),
                        "new_payload_digest": _digest_json(case),
                    }
                )
                event_id = f"event-{uuid.uuid4().hex}"
                created_at = _utc_now()
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, created_at, action, case_id, details_json,
                        source_revision, policy_revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        created_at,
                        action,
                        case_id,
                        _canonical_text(event_details),
                        revision,
                        control["policy_revision"],
                    ),
                )
                projection_event = {
                    "event_id": event_id,
                    "created_at": created_at,
                    "action": action,
                    "case_id": case_id,
                    "details": event_details,
                    "source_revision": revision,
                    "policy_revision": control["policy_revision"],
                }
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        projection_updated = self._append_audit_projection_event(
            projection_event,
            expected_previous_revision=audit_revision - 1,
            revision=audit_revision,
        )
        return {
            "case_id": case_id,
            "event_id": event_id,
            "source_revision": revision,
            "source_digest": None,
            "source_digest_state": "dirty",
            "audit_projection_updated": projection_updated,
            "result": result,
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
        if policy.get("schema") != "hotpot-ds-policy/current":
            raise BundleStateError("dataset policy schema is not current")
        try:
            require_current_runtime_policy(policy["runtime_policy"])
        except (KeyError, PolicyProtocolError) as exc:
            raise BundleStateError("dataset runtime policy is not current") from exc
        manifest = self.manifest
        try:
            validate_property_policy(
                policy.get("property_definitions", {}),
                policy.get("property_comparators", {}),
            )
        except ValueError as exc:
            raise BundleStateError(f"invalid persisted property policy: {exc}") from exc
        for field in ("property_definitions", "property_comparators"):
            if policy.get(field, {}) != manifest.get(field, {}):
                raise BundleStateError(
                    f"persisted {field} cannot differ from the immutable manifest"
                )
        digest = _digest_json(policy)
        projection_event = None
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                revision = control["policy_revision"] + 1
                audit_revision = control["audit_revision"] + 1
                connection.execute(
                    "UPDATE policy SET policy_json=? WHERE singleton=1",
                    (_canonical_text(policy),),
                )
                connection.execute(
                    "UPDATE control SET policy_revision=?,policy_digest=?,"
                    "audit_revision=? WHERE singleton=1",
                    (revision, digest, audit_revision),
                )
                event_id = f"event-{uuid.uuid4().hex}"
                created_at = _utc_now()
                event_details = _json_copy(dict(details or {}))
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, created_at, action, case_id, details_json,
                        source_revision, policy_revision
                    ) VALUES (?, ?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        event_id,
                        created_at,
                        action,
                        _canonical_text(event_details),
                        control["source_revision"],
                        revision,
                    ),
                )
                projection_event = {
                    "event_id": event_id,
                    "created_at": created_at,
                    "action": action,
                    "case_id": None,
                    "details": event_details,
                    "source_revision": control["source_revision"],
                    "policy_revision": revision,
                }
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self.write_policy_projection()
        projection_updated = self._append_audit_projection_event(
            projection_event,
            expected_previous_revision=audit_revision - 1,
            revision=audit_revision,
        )
        return {
            "event_id": event_id,
            "policy_revision": revision,
            "policy_digest": digest,
            "audit_projection_updated": projection_updated,
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
        expected_governance_registry_revision: int,
        expected_governance_registry_digest: str,
    ) -> Dict[str, Any]:
        _nonempty_string(generation_id, "generation_id")
        _nonempty_string(generation_digest, "generation_digest")
        with self.connect() as connection:
            governance_revision, governance_digest = _trusted_governance_binding()
            connection.execute("BEGIN IMMEDIATE")
            try:
                control = connection.execute(
                    "SELECT * FROM control WHERE singleton=1"
                ).fetchone()
                if control["source_digest_state"] != "current":
                    raise ConcurrentStateError(
                        "Bundle source proof is dirty before generation publication"
                    )
                current_source = self._compute_source_digest(
                    connection, self._stored_manifest(connection)
                )
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
                    control["governance_registry_revision"],
                    control["governance_registry_digest"],
                    governance_revision,
                    governance_digest,
                )
                expected = (
                    expected_source_revision,
                    expected_source_digest,
                    expected_policy_revision,
                    expected_policy_digest,
                    expected_governance_registry_revision,
                    expected_governance_registry_digest,
                    expected_governance_registry_revision,
                    expected_governance_registry_digest,
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
                        synchronized_governance_registry_revision=?,
                        synchronized_governance_registry_digest=?,
                        current_generation_id=?,
                        current_generation_digest=?
                    WHERE singleton=1
                      AND source_revision=? AND source_digest=?
                      AND source_digest_state='current'
                      AND policy_revision=? AND policy_digest=?
                      AND governance_registry_revision=?
                      AND governance_registry_digest=?
                    """,
                    (
                        expected_source_revision,
                        expected_source_digest,
                        expected_policy_revision,
                        expected_policy_digest,
                        expected_governance_registry_revision,
                        expected_governance_registry_digest,
                        generation_id,
                        generation_digest,
                        expected_source_revision,
                        expected_source_digest,
                        expected_policy_revision,
                        expected_policy_digest,
                        expected_governance_registry_revision,
                        expected_governance_registry_digest,
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
            "governance_registry_revision": expected_governance_registry_revision,
            "governance_registry_digest": expected_governance_registry_digest,
        }

    def refresh_audit_projection(self) -> Path:
        """Atomically rebuild the non-authoritative audit projection."""

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                control = connection.execute(
                    "SELECT audit_revision FROM control WHERE singleton=1"
                ).fetchone()
                revision = control["audit_revision"]
                rows = [
                    self._audit_projection_value(row)
                    for row in connection.execute(
                        "SELECT * FROM audit_events ORDER BY rowid"
                    )
                ]
                path = self._write_audit_projection_rows(rows)
                digest, size = _digest_file(path)
                connection.execute(
                    "UPDATE control SET audit_projection_revision=?,"
                    "audit_projection_size=?,audit_projection_digest=? "
                    "WHERE singleton=1",
                    (revision, size, digest),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return path

    def write_policy_projection(self) -> Path:
        return self._write_policy_projection_value(self.policy)

    def _write_audit_projection_rows(self, rows: list[dict[str, Any]]) -> Path:
        path = self.root / "audit" / "events.jsonl"
        content = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
        _atomic_write(path, content)
        return path

    @staticmethod
    def _audit_projection_value(row: Mapping[str, Any]) -> Dict[str, Any]:
        details = (
            row["details_json"]
            if isinstance(row, sqlite3.Row)
            else row["details"]
        )
        return {
            "event_id": row["event_id"],
            "created_at": row["created_at"],
            "action": row["action"],
            "case_id": row["case_id"],
            "details": details if isinstance(details, dict) else json.loads(details),
            "source_revision": row["source_revision"],
            "policy_revision": row["policy_revision"],
        }

    def _append_audit_projection_event(
        self,
        event: Mapping[str, Any],
        *,
        expected_previous_revision: int,
        revision: int,
    ) -> bool:
        """Append one event only when the projection is provably current."""

        content = _canonical_bytes(dict(event)) + b"\n"
        path = self.root / "audit" / "events.jsonl"
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                control = connection.execute(
                    "SELECT audit_revision,audit_projection_revision,"
                    "audit_projection_size,audit_projection_digest "
                    "FROM control WHERE singleton=1"
                ).fetchone()
                try:
                    current_hasher, current_size, current_stamp = _stable_file_hash(
                        path
                    )
                    current_digest = current_hasher.hexdigest()
                except BundleStateError:
                    current_digest = None
                    current_size = None
                    current_stamp = None
                if (
                    control["audit_revision"] != revision
                    or control["audit_projection_revision"]
                    != expected_previous_revision
                    or current_size != control["audit_projection_size"]
                    or current_digest != control["audit_projection_digest"]
                ):
                    connection.rollback()
                    return False
                flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
                try:
                    descriptor = os.open(path, flags)
                except OSError:
                    connection.rollback()
                    return False
                try:
                    if _file_stamp(os.fstat(descriptor)) != current_stamp:
                        connection.rollback()
                        return False
                    with os.fdopen(
                        descriptor, "ab", buffering=0, closefd=False
                    ) as stream:
                        written = stream.write(content)
                        os.fsync(stream.fileno())
                finally:
                    os.close(descriptor)
                expected_hasher = current_hasher.copy()
                expected_hasher.update(content)
                try:
                    new_digest, new_size = _digest_file(path)
                except BundleStateError:
                    connection.rollback()
                    return False
                if (
                    written != len(content)
                    or new_size != current_size + len(content)
                    or new_digest != expected_hasher.hexdigest()
                ):
                    connection.rollback()
                    return False
                changed = connection.execute(
                    "UPDATE control SET audit_projection_revision=?,"
                    "audit_projection_size=?,audit_projection_digest=? "
                    "WHERE singleton=1 AND audit_revision=? "
                    "AND audit_projection_revision=? "
                    "AND audit_projection_size=? "
                    "AND audit_projection_digest=?",
                    (
                        revision,
                        new_size,
                        new_digest,
                        revision,
                        expected_previous_revision,
                        current_size,
                        current_digest,
                    ),
                ).rowcount
                if changed != 1:
                    connection.rollback()
                    return False
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

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
            PRAGMA user_version=4;

            CREATE TABLE metadata(
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            ) STRICT;

            CREATE TABLE control(
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                source_revision INTEGER NOT NULL CHECK(source_revision >= 0),
                source_digest TEXT,
                source_digest_state TEXT NOT NULL
                    CHECK(source_digest_state IN ('current', 'dirty')),
                policy_revision INTEGER NOT NULL CHECK(policy_revision >= 0),
                policy_digest TEXT NOT NULL,
                governance_registry_revision INTEGER NOT NULL
                    CHECK(governance_registry_revision >= 0),
                governance_registry_digest TEXT NOT NULL,
                synchronized_source_revision INTEGER,
                synchronized_source_digest TEXT,
                synchronized_policy_revision INTEGER,
                synchronized_policy_digest TEXT,
                synchronized_governance_registry_revision INTEGER,
                synchronized_governance_registry_digest TEXT,
                current_generation_id TEXT,
                current_generation_digest TEXT,
                audit_revision INTEGER NOT NULL CHECK(audit_revision >= 0),
                audit_projection_revision INTEGER NOT NULL
                    CHECK(audit_projection_revision >= 0),
                audit_projection_size INTEGER NOT NULL
                    CHECK(audit_projection_size >= 0),
                audit_projection_digest TEXT NOT NULL
                    CHECK(
                        length(audit_projection_digest)=64
                        AND audit_projection_digest NOT GLOB '*[^0-9a-f]*'
                    ),
                CHECK(
                    (source_digest_state='current' AND source_digest IS NOT NULL)
                    OR (source_digest_state='dirty' AND source_digest IS NULL)
                ),
                CHECK(audit_projection_revision <= audit_revision)
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

    def _stored_manifest(self, connection: sqlite3.Connection) -> Dict[str, Any]:
        try:
            row = connection.execute(
                "SELECT value_json FROM metadata WHERE key='manifest'"
            ).fetchone()
            if row is None:
                raise BundleStateError(
                    "facts.sqlite3 has no canonical built manifest"
                )
            value = json.loads(row[0])
        except (sqlite3.Error, TypeError, json.JSONDecodeError) as exc:
            raise BundleStateError(
                "facts.sqlite3 has no valid canonical built manifest"
            ) from exc
        return _validate_manifest(self.root, value, self.ds_name)

    def _assert_manifest_unchanged(self, connection: sqlite3.Connection) -> None:
        stored = self._stored_manifest(connection)
        current = _validate_manifest(
            self.root,
            _read_json(self.root / "bundle.json", "bundle.json"),
            self.ds_name,
        )
        if current != stored:
            raise BundleStateError(
                "bundle.json differs from the immutable built manifest"
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
                            property_definitions=manifest.get(
                                "property_definitions", {}
                            ),
                            property_comparators=manifest.get(
                                "property_comparators", {}
                            ),
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
                    property_definitions=manifest.get(
                        "property_definitions", {}
                    ),
                    property_comparators=manifest.get(
                        "property_comparators", {}
                    ),
                )
        except sqlite3.Error as exc:
            raise BundleStateError("invalid SQLite intake") from exc
        finally:
            source.close()

    def _compute_source_digest(
        self,
        connection: sqlite3.Connection,
        manifest: Mapping[str, Any],
        *,
        observation_sink: Optional[
            Dict[str, tuple[str, Optional[tuple[int, int, int, int, int, int]]]]
        ] = None,
    ) -> str:
        return self._compute_source_proof(
            connection, manifest, observation_sink=observation_sink
        )

    def _compute_source_proof(
        self,
        connection: sqlite3.Connection,
        manifest: Mapping[str, Any],
        *,
        file_digest: Optional[Callable[[Path], tuple[str, int]]] = None,
        case_observer: Optional[Callable[[Mapping[str, Any]], None]] = None,
        observation_sink: Optional[
            Dict[str, tuple[str, Optional[tuple[int, int, int, int, int, int]]]]
        ] = None,
    ) -> str:
        """Compute the complete, runtime-bound source proof.

        Framing order is protocol, manifest, ordered canonical Case payloads,
        declared controls ordered by ``(category, name)``, then unique
        Case-declared scientific paths in lexical order.  A control frame binds
        category, name, path, observed state, observed SHA-256, and observed
        size; its observed bytes must equal the declaration.  A scientific
        frame binds its path and observed state/digest.  The optional digest
        callback lets a caller provide stable per-file hashing without changing
        proof semantics, and the Case observer is invoked during the sole Case
        scan.

        File observations are cached only for this proof invocation.  Case
        asset paths are always re-derived from canonical payloads and are never
        trusted from a persistent path cache.
        """

        digest_file = _digest_file if file_digest is None else file_digest
        return _compute_source_proof_rows(
            self.root,
            manifest,
            connection.execute(
                "SELECT case_id,public_index,payload_json FROM cases "
                "ORDER BY public_index,case_id"
            ),
            file_digest=digest_file,
            case_observer=case_observer,
            observation_sink=observation_sink,
        )


__all__ = [
    "BUNDLE_SCHEMA",
    "BundleAlreadyBuiltError",
    "BundleNotBuiltError",
    "BundleState",
    "BundleStateError",
    "ConcurrentStateError",
    "DirtySourceStateError",
]
