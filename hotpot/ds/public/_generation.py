"""Immutable full-Bundle assessment generations.

This module is private infrastructure.  It deliberately keeps source facts in
``BundleState`` and derived assessment results in a separately verifiable,
content-addressed generation directory.
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from ._assessment import (
    assess_case,
    canonical_digest,
    canonical_json,
    compare_observations,
    identity_key,
    observation_property,
    observation_with_resolved_entities,
    summarize_assessments,
    validate_identity_schema,
)
from ._bundle_state import BundleState
from ._implementation import (
    PolicyProtocolError,
    parser_binding_for_format,
    require_current_runtime_policy,
    validate_frozen_runtime_policy,
)


GENERATION_SCHEMA = "hotpot-ds-assessment-generation/current"
_DATABASE_SCHEMA_VERSION = 1
_ARCHIVE_CLASSES = ("incomplete", "ordinary", "benchmark")
_PARSEABLE_FORMATS = {"smi", "smiles", "mol", "mol2", "sdf", "xyz"}
_MANIFEST_FIELDS = {
    "schema",
    "generation_id",
    "generation_digest",
    "created_at",
    "source",
    "policy",
    "governance",
    "runtime_policy",
    "ordered_assessment_digest",
    "summary_digest",
    "files",
}


class AssessmentGenerationError(ValueError):
    """Base error for generation construction or validation failures."""


class GenerationIntegrityError(AssessmentGenerationError):
    """Raised when an immutable generation fails content verification."""


class AssetSourceChangedError(AssessmentGenerationError):
    """Raised when a scientific asset changes while it is being observed."""


class _AssessmentReader:
    """One-connection reader for an already verified immutable generation."""

    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def iter_cases(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        for row in self._connection.execute(
            "SELECT case_id,assessment_json FROM case_assessments "
            "ORDER BY public_index,case_id"
        ):
            yield str(row[0]), json.loads(row[1])


def _validate_generation_id(generation_id: Any) -> str:
    if (
        not isinstance(generation_id, str)
        or not generation_id.startswith("generation-")
        or "/" in generation_id
        or "\\" in generation_id
        or generation_id in {".", ".."}
    ):
        raise GenerationIntegrityError("invalid generation id")
    return generation_id


def _lease_directory(root: Path) -> Path:
    directory = root / "assessments" / "leases"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise GenerationIntegrityError("generation lease directory is invalid")
    return directory


def _open_lease_file(root: Path, generation_id: str) -> int:
    lock_path = _lease_directory(root) / f"{_validate_generation_id(generation_id)}.lock"
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise GenerationIntegrityError("generation lease is not a regular file")
    return descriptor


class _GenerationLease:
    """A process-scoped shared lease for one immutable generation.

    The lease uses an OS lock rather than a timestamp.  A process crash releases
    the lock automatically, while the harmless lock file remains available for
    future readers and collectors.
    """

    def __init__(self, root: Path, generation_id: str):
        self._root = root.resolve()
        self._generation_id = _validate_generation_id(generation_id)
        self._descriptor: Optional[int] = None

    def __enter__(self) -> "_GenerationLease":
        descriptor = _open_lease_file(self._root, self._generation_id)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            directory = (
                self._root
                / "assessments"
                / "generations"
                / self._generation_id
            )
            if directory.is_symlink() or not directory.is_dir():
                raise GenerationIntegrityError("assessment generation no longer exists")
        except Exception:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _certificate_generation_reference(
    root: Path,
) -> Tuple[Optional[Tuple[str, str]], bool]:
    """Return the root-certificate generation and whether its reference is unknown."""

    path = root / "certificate.json"
    if not path.exists():
        return None, False
    if path.is_symlink() or not path.is_file():
        return None, True
    try:
        from .certificate import load_certificate_envelope

        envelope = load_certificate_envelope(path)
        generation = envelope["certificate"]["generation"]
        generation_id = _validate_generation_id(generation["id"])
        generation_digest = generation["digest"]
        if (
            not isinstance(generation_digest, str)
            or len(generation_digest) != 64
            or any(character not in "0123456789abcdef" for character in generation_digest)
        ):
            return None, True
    except Exception:
        return None, True
    return (generation_id, generation_digest), False


def _generation_disk_reference(directory: Path) -> Optional[Tuple[str, str]]:
    try:
        manifest = json.loads(
            (directory / "generation.json").read_text(encoding="utf-8")
        )
        generation_id = _validate_generation_id(manifest["generation_id"])
        digest = manifest["generation_digest"]
        if generation_id != directory.name or not isinstance(digest, str):
            return None
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            return None
        return generation_id, digest
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        GenerationIntegrityError,
    ):
        return None


def _collect_obsolete_generations(
    root: Union[os.PathLike[str], str],
    *,
    current_reference: Optional[Tuple[str, str]],
    receipt_generation_digests: Sequence[str] = (),
    receipt_reference_unknown: bool = False,
) -> Dict[str, Any]:
    """Remove only generations proven free of durable or active references.

    The caller holds the Bundle's exclusive lock for the whole durable-reference
    check.  Each candidate must also yield a non-blocking exclusive lease lock;
    failure to establish that lock is a conservative retain decision.
    """

    bundle_root = Path(root).resolve()
    generations = bundle_root / "assessments" / "generations"
    certificate_reference, certificate_reference_unknown = (
        _certificate_generation_reference(bundle_root)
    )
    receipt_digests = frozenset(receipt_generation_digests)
    retain_all = bool(
        certificate_reference_unknown or receipt_reference_unknown
    )
    report: Dict[str, Any] = {
        "removed": [],
        "retained": {},
        "temporary_removed": [],
        "conservative": retain_all,
    }
    if not generations.exists():
        return report
    if generations.is_symlink() or not generations.is_dir():
        report["conservative"] = True
        return report

    for directory in sorted(generations.iterdir(), key=lambda path: path.name):
        if directory.name.startswith(".generation-"):
            if retain_all or directory.is_symlink() or not directory.is_dir():
                continue
            try:
                shutil.rmtree(directory)
                report["temporary_removed"].append(directory.name)
            except OSError:
                pass
            continue
        if not directory.name.startswith("generation-"):
            continue
        if directory.is_symlink() or not directory.is_dir():
            report["retained"][directory.name] = "invalid_candidate"
            continue

        disk_reference = _generation_disk_reference(directory)
        reasons = []
        if retain_all:
            reasons.append("unknown_durable_reference")
        if current_reference is not None and directory.name == current_reference[0]:
            reasons.append("current")
        if (
            certificate_reference is not None
            and directory.name == certificate_reference[0]
        ):
            reasons.append("certificate")
        if disk_reference is None and receipt_digests:
            reasons.append("unknown_receipt_binding")
        elif disk_reference is not None and disk_reference[1] in receipt_digests:
            reasons.append("receipt")
        if reasons:
            report["retained"][directory.name] = ",".join(reasons)
            continue

        try:
            descriptor = _open_lease_file(bundle_root, directory.name)
        except (OSError, GenerationIntegrityError):
            report["retained"][directory.name] = "lease_unverifiable"
            continue
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                report["retained"][directory.name] = "active_reader"
                continue
            try:
                shutil.rmtree(directory)
                report["removed"].append(directory.name)
            except OSError:
                report["retained"][directory.name] = "remove_failed"
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    if report["removed"] or report["temporary_removed"]:
        _fsync_directory(generations)
    return report


def _canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(value: os.stat_result) -> Tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _path_signature(path: Path) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return _stat_signature(path.lstat()), _stat_signature(path.stat())


def _stable_digest(path: Path) -> Tuple[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    before = _path_signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened_before = _stat_signature(os.fstat(stream.fileno()))
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        opened_after = _stat_signature(os.fstat(stream.fileno()))
    after = _path_signature(path)
    if opened_before != opened_after or before != after or opened_after != after[1]:
        raise AssetSourceChangedError(
            "scientific asset changed while its content was being hashed"
        )
    return digest.hexdigest(), after


def _snapshot_asset(
    source_path: Path,
    snapshot_directory: Path,
) -> Tuple[
    Path,
    str,
    Tuple[Tuple[int, ...], Tuple[int, ...]],
    Tuple[Tuple[int, ...], Tuple[int, ...]],
]:
    source_before = _path_signature(source_path)
    source_digest = hashlib.sha256()
    descriptor, snapshot_name = tempfile.mkstemp(
        prefix=".asset-snapshot-", dir=snapshot_directory
    )
    snapshot_path = Path(snapshot_name)
    try:
        with os.fdopen(descriptor, "wb") as target, source_path.open("rb") as source:
            opened_before = _stat_signature(os.fstat(source.fileno()))
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                source_digest.update(chunk)
                target.write(chunk)
            opened_after = _stat_signature(os.fstat(source.fileno()))
            target.flush()
            os.fsync(target.fileno())
        source_after = _path_signature(source_path)
        if (
            opened_before != opened_after
            or source_before != source_after
            or opened_after != source_after[1]
        ):
            raise AssetSourceChangedError(
                "scientific asset changed while its snapshot was being created"
            )
        snapshot_digest, snapshot_signature = _stable_digest(snapshot_path)
        if snapshot_digest != source_digest.hexdigest():
            raise AssessmentGenerationError(
                "scientific asset snapshot does not match the observed source bytes"
            )
        return snapshot_path, snapshot_digest, source_after, snapshot_signature
    except Exception:
        snapshot_path.unlink(missing_ok=True)
        raise


def _structure_is_parseable(path: Path, fmt: str) -> bool:
    from hotpot import Molecule

    molecule = Molecule.read_one(str(path), "smi" if fmt == "smiles" else fmt)
    return bool(molecule is not None and molecule.atoms)


def _write_fsynced(path: Path, content: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_generation_file(directory: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise GenerationIntegrityError("generation inventory contains an unsafe path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise GenerationIntegrityError("generation inventory contains an unsafe path")
    path = directory.joinpath(*relative.parts)
    try:
        path.resolve(strict=False).relative_to(directory.resolve())
    except ValueError as exc:
        raise GenerationIntegrityError("generation inventory path escapes its root") from exc
    return path


def _normalize_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = _json_copy(case)
    for observation in normalized.get("observations", []):
        if "property" not in observation and "property_type_id" not in observation:
            if "property_type" in observation:
                observation["property"] = observation["property_type"]
    return normalized


def _case_with_verified_attestations(
    case: Mapping[str, Any],
    manifest: Mapping[str, Any],
    registry: Any,
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    from ..governance import verify_hotpot_attestation

    normalized = _normalize_case(case)
    assessments = []
    for attestation in normalized.get("attestations", []):
        attestation.pop("signature_verified", None)
        attestation.pop("signature_verification_reason", None)
        if attestation.get("authority") != "Hotpot":
            continue
        verified, reason = verify_hotpot_attestation(
            attestation,
            manifest["ds_name"],
            manifest["release_id"],
            normalized["id"],
            registry,
        )
        attestation["signature_verified"] = verified
        assessments.append(
            {
                "attestation_id": attestation.get("attestation_id"),
                "signature_verified": verified,
                "reason": reason,
            }
        )
    return normalized, assessments


def _active(rows: Sequence[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    return (row for row in rows if row.get("status", "active") == "active")


def _gate_info(policy: Mapping[str, Any], gate_name: str) -> Dict[str, Any]:
    gates = policy.get("gates")
    if not isinstance(gates, list):
        raise AssessmentGenerationError("dataset Gate policy is not current")
    matches = [
        candidate
        for candidate in gates
        if isinstance(candidate, Mapping) and candidate.get("gate") == gate_name
    ]
    if len(matches) != 1:
        raise AssessmentGenerationError(
            f"dataset Gate policy requires exactly one {gate_name}"
        )
    return _json_copy(matches[0])


def _assessment_policy(
    policy: Mapping[str, Any],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    evidence = _gate_info(policy, "EvidenceGate")
    quality = _gate_info(policy, "QualityGate")
    benchmark = _gate_info(policy, "BenchmarkGate")
    tolerances = _json_copy(benchmark.get("numeric_tolerances", {}))
    definitions = _json_copy(policy.get("property_definitions", {}))
    comparators = _json_copy(policy.get("property_comparators", {}))
    return evidence, quality, benchmark, tolerances, definitions, comparators


def _expected_sha256(*values: Mapping[str, Any]) -> Optional[str]:
    for value in values:
        for key in ("expected_sha256", "sha256"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate.lower()
        checksum = value.get("checksum")
        if isinstance(checksum, Mapping) and checksum.get("algorithm") == "sha256":
            candidate = checksum.get("value")
            if isinstance(candidate, str) and candidate:
                return candidate.lower()
    return None


def _asset_declarations(
    case_id: str,
    collection: str,
    record: Mapping[str, Any],
) -> Iterator[Dict[str, Any]]:
    record_id_field = "structure_id" if collection == "structures" else "evidence_id"
    record_id = str(record[record_id_field])
    declarations = []

    if isinstance(record.get("path"), str) or isinstance(record.get("uri"), str):
        declarations.append(
            {
                "path": record.get("path"),
                "uri": record.get("uri"),
                "format": record.get("format"),
                "expected_sha256": _expected_sha256(record),
            }
        )

    assets = record.get("assets", [])
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, Mapping):
                declarations.append(
                    {
                        "path": asset.get("path"),
                        "uri": asset.get("uri"),
                        "format": asset.get("format") or record.get("format"),
                        "expected_sha256": _expected_sha256(asset, record),
                    }
                )

    representations = record.get("representations", {})
    if isinstance(representations, Mapping):
        iterable = representations.items()
    elif isinstance(representations, list):
        iterable = enumerate(representations)
    else:
        iterable = ()
    for name, representation in iterable:
        if isinstance(representation, str):
            declarations.append(
                {
                    "path": representation,
                    "uri": None,
                    "format": str(name),
                    "expected_sha256": _expected_sha256(record),
                }
            )
        elif isinstance(representation, Mapping):
            declarations.append(
                {
                    "path": representation.get("path"),
                    "uri": representation.get("uri"),
                    "format": representation.get("format") or str(name),
                    "expected_sha256": _expected_sha256(representation, record),
                }
            )

    if not declarations:
        declarations.append(
            {
                "path": None,
                "uri": record.get("uri"),
                "format": record.get("format"),
                "expected_sha256": _expected_sha256(record),
            }
        )

    for offset, declaration in enumerate(declarations):
        identity = {
            "case_id": case_id,
            "collection": collection,
            "record_id": record_id,
            "offset": offset,
            **declaration,
        }
        yield {
            "asset_id": "asset-" + canonical_digest(identity)[:32],
            "case_id": case_id,
            "collection": collection,
            "record_id": record_id,
            "structure_id": record_id if collection == "structures" else None,
            "role": record.get("role"),
            **declaration,
        }


def _assess_asset(
    root: Path,
    declaration: Mapping[str, Any],
    required_roles: Sequence[str],
    snapshot_directory: Path,
    runtime_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    relative_path = declaration.get("path")
    path = None
    if isinstance(relative_path, str):
        relative = PurePosixPath(relative_path)
        if (
            relative.is_absolute()
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise AssessmentGenerationError("asset path is not a safe relative POSIX path")
        path = root.joinpath(*relative.parts)
        try:
            path.resolve(strict=False).relative_to(root.resolve())
        except ValueError as exc:
            raise AssessmentGenerationError("asset path escapes the Bundle root") from exc

    present = bool(path is not None and path.is_file())
    declared_format = declaration.get("format")
    fmt = str(declared_format).lower().lstrip(".") if declared_format else ""
    if not fmt and path is not None:
        fmt = path.suffix.lower().lstrip(".")
    parse_result = "not_run"
    parser = None
    parser_binding = None
    parse_reason = None
    observed = None
    snapshot_path = None
    if present:
        try:
            (
                snapshot_path,
                observed,
                source_signature,
                snapshot_signature,
            ) = _snapshot_asset(path, snapshot_directory)
            if (
                declaration.get("collection") == "structures"
                and fmt in _PARSEABLE_FORMATS
            ):
                parser_binding = parser_binding_for_format(fmt, runtime_policy)
                if parser_binding is None:
                    raise AssessmentGenerationError(
                        f"no frozen parser binding exists for structure format {fmt!r}"
                    )
                parser = parser_binding["name"]
                try:
                    parse_result = (
                        "valid"
                        if _structure_is_parseable(snapshot_path, fmt)
                        else "invalid"
                    )
                except Exception:
                    parse_result = "invalid"
                if parse_result == "invalid":
                    parse_reason = "structure_parse_failed"

            snapshot_after, snapshot_signature_after = _stable_digest(snapshot_path)
            source_after, signature_after = _stable_digest(path)
            if (
                snapshot_after != observed
                or snapshot_signature_after != snapshot_signature
                or source_after != observed
                or signature_after != source_signature
            ):
                raise AssetSourceChangedError(
                    "scientific asset changed while it was being assessed"
                )
        except OSError as exc:
            raise AssetSourceChangedError(
                "scientific asset became unavailable while it was being assessed"
            ) from exc
        finally:
            if snapshot_path is not None:
                snapshot_path.unlink(missing_ok=True)

    expected = declaration.get("expected_sha256")
    if expected is None or not present:
        hash_result = "unchecked"
    else:
        hash_result = "match" if observed == expected else "mismatch"

    reasons = []
    if not present:
        reasons.append("asset_missing")
    if hash_result == "mismatch":
        reasons.append("asset_hash_mismatch")
    if parse_reason:
        reasons.append(parse_reason)
    if present and declaration.get("collection") == "structures" and fmt not in _PARSEABLE_FORMATS:
        reasons.append("structure_parser_not_configured")

    return {
        **_json_copy(declaration),
        "format": fmt or None,
        "presence": "present" if present else "missing",
        "observed_sha256": observed,
        "hash_result": hash_result,
        "parse_result": parse_result,
        "effective_required": bool(
            declaration.get("collection") == "structures"
            and (
                "*" in required_roles
                or declaration.get("role") in required_roles
            )
        ),
        "parser": parser,
        "parser_binding_digest": (
            parser_binding["digest"] if parser_binding is not None else None
        ),
        "parser_protocol": (
            _json_copy(parser_binding["protocol"])
            if parser_binding is not None
            else None
        ),
        "parser_implementation_digest": (
            parser_binding["source"]["digest"]
            if parser_binding is not None
            else None
        ),
        "parser_distribution_identity": (
            _json_copy(parser_binding["distributions"])
            if parser_binding is not None
            else None
        ),
        "parser_distribution_digest": (
            canonical_digest(parser_binding["distributions"])
            if parser_binding is not None
            else None
        ),
        "reasons": reasons,
    }


def _initialize_assessment_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        PRAGMA foreign_keys=ON;
        PRAGMA user_version=1;

        CREATE TABLE metadata(
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE case_assessments(
            case_id TEXT PRIMARY KEY,
            public_index INTEGER NOT NULL UNIQUE,
            archive_class TEXT NOT NULL,
            quality_level TEXT NOT NULL,
            evidence_level TEXT,
            benchmark INTEGER NOT NULL,
            assessment_json TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE evidence_assessments(
            case_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            evidence_level TEXT NOT NULL,
            assessment_json TEXT NOT NULL,
            PRIMARY KEY(case_id, target_id)
        ) WITHOUT ROWID;

        CREATE TABLE asset_assessments(
            asset_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            structure_id TEXT,
            role TEXT,
            presence TEXT NOT NULL,
            hash_result TEXT NOT NULL,
            parse_result TEXT NOT NULL,
            assessment_json TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE INDEX asset_assessments_case ON asset_assessments(case_id, asset_id);

        CREATE TABLE conflict_assessments(
            conflict_id TEXT PRIMARY KEY,
            sample_key TEXT NOT NULL,
            property_key TEXT NOT NULL,
            left_case_id TEXT NOT NULL,
            right_case_id TEXT NOT NULL,
            outcome TEXT NOT NULL,
            assessment_json TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE conflict_case_links(
            case_id TEXT NOT NULL,
            conflict_id TEXT NOT NULL,
            PRIMARY KEY(case_id, conflict_id),
            FOREIGN KEY(conflict_id) REFERENCES conflict_assessments(conflict_id)
        ) WITHOUT ROWID;

        CREATE TABLE conflict_groups(
            group_id TEXT PRIMARY KEY,
            sample_key TEXT NOT NULL,
            property_key TEXT NOT NULL,
            assessment_json TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TEMP TABLE case_order(
            public_index INTEGER PRIMARY KEY,
            case_id TEXT NOT NULL UNIQUE
        ) WITHOUT ROWID;
        """
    )


def _initialize_work_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=OFF;
        CREATE TABLE case_order(
            case_id TEXT PRIMARY KEY,
            public_index INTEGER NOT NULL UNIQUE
        ) WITHOUT ROWID;
        CREATE TABLE candidates(
            sample_key TEXT NOT NULL,
            property_key TEXT NOT NULL,
            case_id TEXT NOT NULL,
            observation_id TEXT NOT NULL,
            observation_json TEXT NOT NULL,
            PRIMARY KEY(sample_key, property_key, case_id, observation_id)
        ) WITHOUT ROWID;
        CREATE TABLE identity_reasons(
            case_id TEXT PRIMARY KEY,
            reasons_json TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE conflicting_cases(case_id TEXT PRIMARY KEY) WITHOUT ROWID;
        """
    )


def _stream_inputs(
    root: Path,
    case_source: Any,
    assessment: sqlite3.Connection,
    work: sqlite3.Connection,
    identity_schema: Mapping[str, Any],
    required_roles: Sequence[str],
    snapshot_directory: Path,
    runtime_policy: Mapping[str, Any],
) -> None:
    for source_case in case_source.iter_cases():
        case = _normalize_case(source_case)
        case_id = str(case["id"])
        public_index = int(case["public_index"])
        work.execute(
            "INSERT INTO case_order(case_id,public_index) VALUES (?,?)",
            (case_id, public_index),
        )
        assessment.execute(
            "INSERT INTO case_order(public_index,case_id) VALUES (?,?)",
            (public_index, case_id),
        )
        key, reasons = identity_key(case, identity_schema)
        work.execute(
            "INSERT INTO identity_reasons(case_id,reasons_json) VALUES (?,?)",
            (case_id, canonical_json(reasons)),
        )
        sample_key = key if key is not None else f"case:{case_id}"
        for offset, observation in enumerate(_active(case.get("observations", []))):
            value = observation_with_resolved_entities(case, observation)
            observation_id = str(
                value.get("observation_id") or f"{case_id}:observation:{offset}"
            )
            value["observation_id"] = observation_id
            property_key = str(observation_property(value) or "null")
            work.execute(
                """
                INSERT INTO candidates(
                    sample_key,property_key,case_id,observation_id,observation_json
                ) VALUES (?,?,?,?,?)
                """,
                (sample_key, property_key, case_id, observation_id, canonical_json(value)),
            )

        for collection in ("structures", "evidence"):
            for record in _active(case.get(collection, [])):
                for declaration in _asset_declarations(case_id, collection, record):
                    asset = _assess_asset(
                            root,
                        declaration,
                        required_roles,
                        snapshot_directory,
                        runtime_policy,
                    )
                    assessment.execute(
                        """
                        INSERT INTO asset_assessments(
                            asset_id,case_id,structure_id,role,presence,hash_result,
                            parse_result,assessment_json
                        ) VALUES (?,?,?,?,?,?,?,?)
                        """,
                        (
                            asset["asset_id"],
                            case_id,
                            asset.get("structure_id"),
                            asset.get("role"),
                            asset["presence"],
                            asset["hash_result"],
                            asset["parse_result"],
                            canonical_json(asset),
                        ),
                    )


def _assess_conflicts(
    assessment: sqlite3.Connection,
    work: sqlite3.Connection,
    tolerances: Mapping[str, Mapping[str, float]],
    identity_schema_digest: str,
    property_definitions: Optional[Mapping[str, Any]] = None,
    property_comparators: Optional[Mapping[str, Any]] = None,
) -> None:
    definitions = dict(property_definitions or {})
    comparators = dict(property_comparators or {})
    groups = work.execute(
        "SELECT DISTINCT sample_key,property_key FROM candidates "
        "ORDER BY sample_key,property_key"
    )
    for sample_key, property_key in groups:
        left_candidates = work.execute(
            """
            SELECT case_id,observation_id,observation_json
            FROM candidates
            WHERE sample_key=? AND property_key=?
            ORDER BY case_id,observation_id
            """,
            (sample_key, property_key),
        )
        candidate_count = 0
        consistent_count = 0
        exception_count = 0
        ordered = hashlib.sha256()
        for left_case_id, left_observation_id, left_json in left_candidates:
            left = json.loads(left_json)
            right_candidates = work.execute(
                """
                SELECT case_id,observation_id,observation_json
                FROM candidates
                WHERE sample_key=? AND property_key=?
                  AND (case_id,observation_id)>(?,?)
                ORDER BY case_id,observation_id
                """,
                (
                    sample_key,
                    property_key,
                    left_case_id,
                    left_observation_id,
                ),
            )
            for right_case_id, right_observation_id, right_json in right_candidates:
                result = compare_observations(
                    left,
                    json.loads(right_json),
                    tolerances,
                    property_definitions=definitions,
                    property_comparators=comparators,
                )
                if result["outcome"] == "noncandidate":
                    continue
                pair = {
                    "sample_key": sample_key,
                    "identity_schema_digest": identity_schema_digest,
                    "property_key": property_key,
                    "left_case_id": left_case_id,
                    "right_case_id": right_case_id,
                    "left_observation_id": left_observation_id,
                    "right_observation_id": right_observation_id,
                    **result,
                }
                encoded = _canonical_bytes(pair)
                ordered.update(len(encoded).to_bytes(8, "big"))
                ordered.update(encoded)
                candidate_count += 1
                if result["outcome"] == "consistent":
                    consistent_count += 1
                    continue
                exception_count += 1
                conflict_id = "conflict-" + canonical_digest(pair)[:32]
                assessment.execute(
                    """
                    INSERT INTO conflict_assessments(
                        conflict_id,sample_key,property_key,left_case_id,
                        right_case_id,outcome,assessment_json
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        conflict_id,
                        sample_key,
                        property_key,
                        left_case_id,
                        right_case_id,
                        result["outcome"],
                        canonical_json(pair),
                    ),
                )
                for case_id in dict.fromkeys((left_case_id, right_case_id)):
                    assessment.execute(
                        "INSERT INTO conflict_case_links(case_id,conflict_id) VALUES (?,?)",
                        (case_id, conflict_id),
                    )
                if result["outcome"] == "conflict":
                    work.execute(
                        "INSERT OR IGNORE INTO conflicting_cases(case_id) VALUES (?)",
                        (left_case_id,),
                    )
                    work.execute(
                        "INSERT OR IGNORE INTO conflicting_cases(case_id) VALUES (?)",
                        (right_case_id,),
                    )
        group = {
            "sample_key": sample_key,
            "identity_schema_digest": identity_schema_digest,
            "property_key": property_key,
            "candidate_count": candidate_count,
            "consistent_count": consistent_count,
            "exception_count": exception_count,
            "ordered_results_digest": ordered.hexdigest(),
        }
        group_id = "group-" + canonical_digest(
            [sample_key, property_key]
        )[:32]
        assessment.execute(
            """
            INSERT INTO conflict_groups(
                group_id,sample_key,property_key,assessment_json
            ) VALUES (?,?,?,?)
            """,
            (group_id, sample_key, property_key, canonical_json(group)),
        )


def _grouped_support_rows(
    rows: Iterator[Sequence[Any]],
) -> Iterator[Tuple[Tuple[int, str], list[Dict[str, Any]]]]:
    current: Optional[Tuple[int, str]] = None
    values: list[Dict[str, Any]] = []
    for public_index, case_id, value_json in rows:
        key = (int(public_index), str(case_id))
        if current is not None and key != current:
            yield current, values
            values = []
        current = key
        if value_json is not None:
            values.append(json.loads(value_json))
    if current is not None:
        yield current, values


def _assess_cases(
    case_source: Any,
    assessment: sqlite3.Connection,
    work: sqlite3.Connection,
    manifest: Mapping[str, Any],
    evidence_policy: Mapping[str, Any],
    quality_policy: Mapping[str, Any],
    benchmark_policy: Mapping[str, Any],
    property_definitions: Mapping[str, Any],
    property_comparators: Mapping[str, Any],
    governance_registry: Any,
    isolated_runner: Optional[Any],
) -> None:
    identity_rows = work.execute(
        """
        SELECT case_order.public_index,case_order.case_id,
               identity_reasons.reasons_json,
               CASE WHEN conflicting_cases.case_id IS NULL THEN 0 ELSE 1 END
        FROM case_order
        JOIN identity_reasons USING(case_id)
        LEFT JOIN conflicting_cases USING(case_id)
        ORDER BY case_order.public_index,case_order.case_id
        """
    )
    asset_groups = _grouped_support_rows(
        assessment.execute(
            """
            SELECT case_order.public_index,case_order.case_id,
                   asset_assessments.assessment_json
            FROM case_order
            LEFT JOIN asset_assessments USING(case_id)
            ORDER BY case_order.public_index,case_order.case_id,
                     asset_assessments.asset_id
            """
        )
    )
    conflict_groups = _grouped_support_rows(
        assessment.execute(
            """
            SELECT case_order.public_index,case_order.case_id,
                   conflict_assessments.assessment_json
            FROM case_order
            LEFT JOIN conflict_case_links USING(case_id)
            LEFT JOIN conflict_assessments USING(conflict_id)
            ORDER BY case_order.public_index,case_order.case_id,
                     conflict_case_links.conflict_id
            """
        )
    )
    for source_case, identity_row, asset_group, conflict_group in zip_longest(
        case_source.iter_cases(), identity_rows, asset_groups, conflict_groups
    ):
        if (
            source_case is None
            or identity_row is None
            or asset_group is None
            or conflict_group is None
        ):
            raise AssessmentGenerationError(
                "ordered Case support streams have different lengths"
            )
        public_index, row_case_id, reasons_json, conflicting = identity_row
        expected = (int(public_index), str(row_case_id))
        if asset_group[0] != expected or conflict_group[0] != expected:
            raise AssessmentGenerationError(
                "ordered Case support streams are not aligned"
            )
        source_key = (int(source_case["public_index"]), str(source_case["id"]))
        if source_key != expected:
            raise AssessmentGenerationError(
                "ordered Case input and support streams are not aligned"
            )
        case, governance_assessments = _case_with_verified_attestations(
            source_case, manifest, governance_registry
        )
        case_id = str(case["id"])
        if case_id != expected[1]:
            raise AssessmentGenerationError("Case input stream identity changed")
        assets = asset_group[1]
        conflict_rows = conflict_group[1]
        identity_reasons = json.loads(reasons_json)
        result = assess_case(
            case,
            core_labels=manifest["core_labels"],
            core_info=manifest["core_info"],
            required_structure_roles=manifest["required_structure_roles"],
            asset_assessments=assets,
            conflicting=bool(conflicting),
            benchmark_policy=benchmark_policy,
            evidence_policy=evidence_policy,
            quality_policy=quality_policy,
            conflict_assessments=conflict_rows,
            identity_reasons=identity_reasons,
            checker_context={
                "dataset": {
                    "ds_name": manifest["ds_name"],
                    "release_id": manifest["release_id"],
                    "version": manifest["version"],
                },
                "required": {
                    "core_labels": manifest["core_labels"],
                    "core_info": manifest["core_info"],
                    "structure_roles": manifest["required_structure_roles"],
                },
                "identity": {"reasons": identity_reasons},
                "property_policy": {
                    "definitions": _json_copy(property_definitions),
                    "comparators": _json_copy(property_comparators),
                },
                "conflicts": conflict_rows,
                "assets": assets,
            },
            isolated_runner=isolated_runner,
        )
        result["input_digest"] = canonical_digest(source_case)
        result["asset_assessments"] = assets
        result["conflict_assessments"] = conflict_rows
        result["identity_reasons"] = identity_reasons
        result["governance_assessments"] = governance_assessments
        assessment.execute(
            """
            INSERT INTO case_assessments(
                case_id,public_index,archive_class,quality_level,evidence_level,
                benchmark,assessment_json
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                case_id,
                case["public_index"],
                result["archive_class"],
                result["quality_level"],
                result["evidence_level"],
                int(result["benchmark"]),
                canonical_json(result),
            ),
        )
        for target_id, level in sorted(result["evidence_levels"].items()):
            evidence = {
                "case_id": case_id,
                "target_id": target_id,
                "evidence_level": level,
            }
            assessment.execute(
                """
                INSERT INTO evidence_assessments(
                    case_id,target_id,evidence_level,assessment_json
                ) VALUES (?,?,?,?)
                """,
                (case_id, target_id, level, canonical_json(evidence)),
            )


_DIGEST_QUERIES = (
    ("asset_assessments", "SELECT asset_id,assessment_json FROM asset_assessments ORDER BY asset_id"),
    ("conflict_assessments", "SELECT conflict_id,assessment_json FROM conflict_assessments ORDER BY conflict_id"),
    ("conflict_groups", "SELECT group_id,assessment_json FROM conflict_groups ORDER BY group_id"),
    ("case_assessments", "SELECT case_id,assessment_json FROM case_assessments ORDER BY public_index,case_id"),
    ("evidence_assessments", "SELECT case_id,target_id,assessment_json FROM evidence_assessments ORDER BY case_id,target_id"),
)


def _ordered_assessment_digest(connection: sqlite3.Connection) -> str:
    return _assessment_products(connection)[0]


def _assessment_products(
    connection: sqlite3.Connection,
    *,
    case_observer: Optional[
        Callable[[str, str, Mapping[str, Any]], None]
    ] = None,
) -> Tuple[str, Dict[str, Any]]:
    digest = hashlib.sha256()
    summary: Optional[Dict[str, Any]] = None
    for table, query in _DIGEST_QUERIES:
        table_bytes = table.encode("utf-8")
        digest.update(len(table_bytes).to_bytes(8, "big"))
        digest.update(table_bytes)
        if table != "case_assessments":
            for row in connection.execute(query):
                value = [table, *row]
                encoded = _canonical_bytes(value)
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
            continue

        def assessments() -> Iterator[Dict[str, Any]]:
            for case_id, assessment_json, archive_class in connection.execute(
                "SELECT case_id,assessment_json,archive_class "
                "FROM case_assessments ORDER BY public_index,case_id"
            ):
                value = [table, case_id, assessment_json]
                encoded = _canonical_bytes(value)
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                assessment = json.loads(assessment_json)
                if case_observer is not None:
                    case_observer(str(case_id), str(archive_class), assessment)
                yield assessment

        summary = summarize_assessments(assessments())
    if summary is None:
        raise AssessmentGenerationError("assessment digest omitted Case assessments")
    return digest.hexdigest(), summary


def _write_assessment_products(
    connection: sqlite3.Connection,
    directory: Path,
) -> Tuple[str, Dict[str, Any]]:
    streams = {
        archive_class: (directory / f"{archive_class}.ids").open("wb")
        for archive_class in _ARCHIVE_CLASSES
    }
    try:
        def write_case(
            case_id: str,
            archive_class: str,
            _assessment: Mapping[str, Any],
        ) -> None:
            streams[archive_class].write(f"{case_id}\n".encode("utf-8"))

        result = _assessment_products(connection, case_observer=write_case)
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
        return result
    finally:
        for stream in streams.values():
            stream.close()


def _validate_assessment_products(
    connection: sqlite3.Connection,
    directory: Path,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    streams = {
        archive_class: (directory / f"{archive_class}.ids").open(
            "r", encoding="utf-8"
        )
        for archive_class in _ARCHIVE_CLASSES
    }
    indexes_match = True
    try:
        def check_case(
            case_id: str,
            archive_class: str,
            _assessment: Mapping[str, Any],
        ) -> None:
            nonlocal indexes_match
            stream = streams.get(archive_class)
            if stream is None or stream.readline() != f"{case_id}\n":
                indexes_match = False

        ordered_digest, computed_summary = _assessment_products(
            connection, case_observer=check_case
        )
        if any(stream.readline() != "" for stream in streams.values()):
            indexes_match = False
    finally:
        for stream in streams.values():
            stream.close()

    if ordered_digest != manifest["ordered_assessment_digest"]:
        raise GenerationIntegrityError("ordered assessment digest does not match")
    if not indexes_match:
        raise GenerationIntegrityError("archive indexes do not match assessments")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        summary != computed_summary
        or canonical_digest(summary) != manifest["summary_digest"]
    ):
        raise GenerationIntegrityError("generation summary does not match assessments")
    return summary


def _inventory(directory: Path) -> list[Dict[str, Any]]:
    result = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "generation.json":
            continue
        relative = path.relative_to(directory).as_posix()
        result.append(
            {"path": relative, "size": path.stat().st_size, "sha256": _sha256_path(path)}
        )
    return result


def _logical_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: _json_copy(manifest[key])
        for key in (
            "schema",
            "source",
            "policy",
            "governance",
            "runtime_policy",
            "ordered_assessment_digest",
            "summary_digest",
            "files",
        )
    }


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    return canonical_digest(_logical_manifest(manifest))


class AssessmentGeneration:
    """A verified immutable assessment-generation snapshot."""

    def __init__(self, root: Path, directory: Path, manifest: Mapping[str, Any]):
        self._root = root
        self._directory = directory
        self._manifest = canonical_json(manifest)

    @classmethod
    def build(
        cls,
        root: Union[os.PathLike[str], str],
        state: BundleState,
        policy: Mapping[str, Any],
        governance_registry: Mapping[str, Any],
        *,
        input_snapshot: Mapping[str, Any],
        isolated_runner: Optional[Any] = None,
    ) -> "AssessmentGeneration":
        bundle_root = Path(root).resolve()
        if bundle_root != state.root:
            raise AssessmentGenerationError("generation root does not match BundleState")
        snapshot_fields = {
            "source_revision",
            "source_digest",
            "policy_revision",
            "policy_digest",
            "governance_registry_revision",
            "governance_registry_digest",
            "manifest",
            "policy",
        }
        if set(input_snapshot) != snapshot_fields or not callable(
            getattr(input_snapshot, "iter_cases", None)
        ):
            raise AssessmentGenerationError(
                "generation requires a current synchronization input snapshot"
            )
        status = {
            key: input_snapshot[key]
            for key in snapshot_fields - {"manifest", "policy"}
        }
        manifest = _json_copy(input_snapshot["manifest"])
        if manifest.get("ds_name") != state.ds_name:
            raise AssessmentGenerationError(
                "synchronization snapshot dataset does not match BundleState"
            )
        policy_value = _json_copy(policy)
        if policy_value != _json_copy(input_snapshot["policy"]):
            raise AssessmentGenerationError(
                "supplied policy is not the synchronization snapshot policy"
            )
        if canonical_digest(policy_value) != status["policy_digest"]:
            raise AssessmentGenerationError("supplied policy is not the persisted policy")
        try:
            runtime_policy = validate_frozen_runtime_policy(
                policy_value["runtime_policy"]
            )
            require_current_runtime_policy(runtime_policy)
        except (KeyError, PolicyProtocolError) as exc:
            raise AssessmentGenerationError(
                "persisted dataset runtime policy is not current"
            ) from exc

        from ..governance import verify_registry

        verified_registry = verify_registry(governance_registry)
        registry = verified_registry.to_dict()
        registry_digest = verified_registry.digest
        if (
            verified_registry.revision
            != status["governance_registry_revision"]
            or registry_digest != status["governance_registry_digest"]
        ):
            raise AssessmentGenerationError(
                "governance registry is not the synchronization snapshot registry"
            )
        try:
            identity_schema = validate_identity_schema(
                policy_value["identity_schema"]
            )
            required_structure_roles = policy_value["required_structure_roles"]
        except (KeyError, TypeError, ValueError) as exc:
            raise AssessmentGenerationError(
                "persisted dataset identity policy is not current"
            ) from exc
        (
            evidence_policy,
            quality_policy,
            benchmark_policy,
            tolerances,
            property_definitions,
            property_comparators,
        ) = _assessment_policy(policy_value)

        generations = bundle_root / "assessments" / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".generation-", dir=generations))
        assessment_path = temporary / "assessment.sqlite3"
        work_path = temporary / ".work.sqlite3"
        assessment: Optional[sqlite3.Connection] = None
        work: Optional[sqlite3.Connection] = None
        try:
            assessment = sqlite3.connect(assessment_path)
            work = sqlite3.connect(work_path)
            _initialize_assessment_database(assessment)
            _initialize_work_database(work)
            with assessment, work:
                _stream_inputs(
                    bundle_root,
                    input_snapshot,
                    assessment,
                    work,
                    identity_schema,
                    required_structure_roles,
                    temporary,
                    runtime_policy,
                )
                _assess_conflicts(
                    assessment,
                    work,
                    tolerances,
                    canonical_digest(identity_schema),
                    property_definitions,
                    property_comparators,
                )
                _assess_cases(
                    input_snapshot,
                    assessment,
                    work,
                    manifest,
                    evidence_policy,
                    quality_policy,
                    benchmark_policy,
                    property_definitions,
                    property_comparators,
                    verified_registry,
                    isolated_runner,
                )
                for key, value in (
                    ("source_revision", status["source_revision"]),
                    ("source_digest", status["source_digest"]),
                    ("policy_revision", status["policy_revision"]),
                    ("policy_digest", status["policy_digest"]),
                    ("governance_registry_digest", registry_digest),
                    ("runtime_policy_digest", runtime_policy["digest"]),
                ):
                    assessment.execute(
                        "INSERT INTO metadata(key,value_json) VALUES (?,?)",
                        (key, canonical_json(value)),
                    )
            ordered_digest, summary = _write_assessment_products(
                assessment, temporary
            )
            assessment.close()
            assessment = None
            work.close()
            work = None
            work_path.unlink()

            _write_fsynced(
                temporary / "summary.json", _canonical_bytes(summary) + b"\n"
            )
            _write_fsynced(
                temporary / "governance-registry.json",
                _canonical_bytes(registry) + b"\n",
            )
            files = _inventory(temporary)
            generation_manifest = {
                "schema": GENERATION_SCHEMA,
                "source": {
                    "revision": status["source_revision"],
                    "digest": status["source_digest"],
                },
                "policy": {
                    "revision": status["policy_revision"],
                    "digest": status["policy_digest"],
                },
                "governance": {
                    "revision": verified_registry.revision,
                    "digest": registry_digest,
                    "pinned_genesis_key_id": verified_registry.pinned_genesis_key_id,
                },
                "runtime_policy": runtime_policy,
                "ordered_assessment_digest": ordered_digest,
                "summary_digest": canonical_digest(summary),
                "files": files,
            }
            generation_digest = _manifest_digest(generation_manifest)
            generation_id = "generation-" + generation_digest[:24]
            generation_manifest.update(
                {
                    "generation_id": generation_id,
                    "generation_digest": generation_digest,
                    "created_at": _utc_now(),
                }
            )
            _write_fsynced(
                temporary / "generation.json",
                _canonical_bytes(generation_manifest) + b"\n",
            )
            _fsync_directory(temporary)

            destination = generations / generation_id
            if destination.exists():
                existing = cls.open(bundle_root, generation_id, generation_digest)
                shutil.rmtree(temporary)
                return existing
            try:
                os.rename(temporary, destination)
            except FileExistsError:
                existing = cls.open(bundle_root, generation_id, generation_digest)
                shutil.rmtree(temporary)
                return existing
            _fsync_directory(generations)
            return cls.open(bundle_root, generation_id, generation_digest)
        finally:
            if assessment is not None:
                assessment.close()
            if work is not None:
                work.close()
            if temporary.exists():
                shutil.rmtree(temporary)

    @classmethod
    def open(
        cls,
        root: Union[os.PathLike[str], str],
        generation_id: str,
        expected_digest: Optional[str] = None,
    ) -> "AssessmentGeneration":
        bundle_root = Path(root).resolve()
        _validate_generation_id(generation_id)
        with _GenerationLease(bundle_root, generation_id):
            return cls._open_without_lease(
                bundle_root,
                generation_id,
                expected_digest,
            )

    @classmethod
    def _open_without_lease(
        cls,
        bundle_root: Path,
        generation_id: str,
        expected_digest: Optional[str],
    ) -> "AssessmentGeneration":
        directory = bundle_root / "assessments" / "generations" / generation_id
        manifest_path = directory / "generation.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GenerationIntegrityError("generation.json is missing or invalid") from exc
        if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
            raise GenerationIntegrityError("generation.json has an invalid schema")
        if manifest.get("schema") != GENERATION_SCHEMA:
            raise GenerationIntegrityError("generation schema is not current")
        if not isinstance(manifest.get("created_at"), str):
            raise GenerationIntegrityError("generation timestamp is invalid")
        for field, expected_fields in (
            ("source", {"revision", "digest"}),
            ("policy", {"revision", "digest"}),
            (
                "governance",
                {"revision", "digest", "pinned_genesis_key_id"},
            ),
        ):
            value = manifest.get(field)
            if not isinstance(value, dict) or set(value) != expected_fields:
                raise GenerationIntegrityError(
                    f"generation {field} binding is invalid"
                )
        try:
            runtime_policy = validate_frozen_runtime_policy(
                manifest["runtime_policy"]
            )
        except (KeyError, PolicyProtocolError) as exc:
            raise GenerationIntegrityError(
                "generation runtime policy binding is invalid"
            ) from exc
        if manifest.get("generation_id") != generation_id:
            raise GenerationIntegrityError("generation id does not match its directory")
        digest = _manifest_digest(manifest)
        if manifest.get("generation_digest") != digest:
            raise GenerationIntegrityError("generation logical digest does not match")
        if expected_digest is not None and digest != expected_digest:
            raise GenerationIntegrityError("generation digest does not match expected digest")
        if generation_id != "generation-" + digest[:24]:
            raise GenerationIntegrityError("generation id is not content addressed")

        inventory = manifest.get("files")
        if not isinstance(inventory, list):
            raise GenerationIntegrityError("generation inventory must be an array")
        declared = set()
        for item in inventory:
            if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                raise GenerationIntegrityError("generation inventory item is invalid")
            path = _safe_generation_file(directory, item["path"])
            relative = path.relative_to(directory).as_posix()
            if relative in declared:
                raise GenerationIntegrityError("generation inventory has a duplicate path")
            declared.add(relative)
            if path.is_symlink() or not path.is_file():
                raise GenerationIntegrityError(f"generation file is missing: {relative}")
            if path.stat().st_size != item["size"] or _sha256_path(path) != item["sha256"]:
                raise GenerationIntegrityError(f"generation file content changed: {relative}")
        actual = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.name != "generation.json"
        }
        if actual != declared:
            raise GenerationIntegrityError("generation inventory is incomplete")
        if any(path.name.endswith(("-wal", "-shm")) for path in directory.rglob("*")):
            raise GenerationIntegrityError("generation contains a SQLite sidecar")

        assessment_path = directory / "assessment.sqlite3"
        try:
            connection = sqlite3.connect(
                f"file:{assessment_path.as_posix()}?mode=ro", uri=True
            )
            connection.row_factory = sqlite3.Row
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise GenerationIntegrityError("assessment.sqlite3 failed integrity_check")
            if connection.execute("PRAGMA user_version").fetchone()[0] != _DATABASE_SCHEMA_VERSION:
                raise GenerationIntegrityError("assessment.sqlite3 schema is not current")
            metadata = {
                row[0]: json.loads(row[1])
                for row in connection.execute(
                    "SELECT key,value_json FROM metadata ORDER BY key"
                )
            }
            expected_metadata = {
                "source_revision": manifest["source"]["revision"],
                "source_digest": manifest["source"]["digest"],
                "policy_revision": manifest["policy"]["revision"],
                "policy_digest": manifest["policy"]["digest"],
                "governance_registry_digest": manifest["governance"]["digest"],
                "runtime_policy_digest": runtime_policy["digest"],
            }
            if metadata != expected_metadata:
                raise GenerationIntegrityError(
                    "assessment metadata does not match generation bindings"
                )
            _validate_assessment_products(connection, directory, manifest)
            for row in connection.execute(
                "SELECT assessment_json FROM asset_assessments ORDER BY asset_id"
            ):
                asset = json.loads(row[0])
                parser_name = asset.get("parser")
                parser_fields = {
                    "parser_binding_digest": asset.get("parser_binding_digest"),
                    "parser_protocol": asset.get("parser_protocol"),
                    "parser_implementation_digest": asset.get(
                        "parser_implementation_digest"
                    ),
                    "parser_distribution_identity": asset.get(
                        "parser_distribution_identity"
                    ),
                    "parser_distribution_digest": asset.get(
                        "parser_distribution_digest"
                    ),
                }
                if parser_name is None:
                    if any(value is not None for value in parser_fields.values()):
                        raise GenerationIntegrityError(
                            "unparsed asset contains a parser binding"
                        )
                    continue
                binding = parser_binding_for_format(
                    str(asset.get("format") or ""), runtime_policy
                )
                if binding is None or (
                    parser_name != binding["name"]
                    or parser_fields["parser_binding_digest"] != binding["digest"]
                    or parser_fields["parser_protocol"] != binding["protocol"]
                    or parser_fields["parser_implementation_digest"]
                    != binding["source"]["digest"]
                    or parser_fields["parser_distribution_identity"]
                    != binding["distributions"]
                    or parser_fields["parser_distribution_digest"]
                    != canonical_digest(binding["distributions"])
                ):
                    raise GenerationIntegrityError(
                        "asset parser binding does not match the frozen runtime policy"
                    )
        except (sqlite3.Error, OSError, UnicodeError, json.JSONDecodeError) as exc:
            if isinstance(exc, GenerationIntegrityError):
                raise
            raise GenerationIntegrityError("assessment generation cannot be verified") from exc
        finally:
            if "connection" in locals():
                connection.close()

        try:
            registry = json.loads(
                (directory / "governance-registry.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GenerationIntegrityError("governance registry is invalid") from exc
        try:
            from ..governance import verify_registry

            verified_registry = verify_registry(registry)
        except Exception as exc:
            raise GenerationIntegrityError("governance registry is not trusted") from exc
        if (
            verified_registry.digest != manifest["governance"]["digest"]
            or verified_registry.revision != manifest["governance"]["revision"]
            or verified_registry.pinned_genesis_key_id
            != manifest["governance"]["pinned_genesis_key_id"]
        ):
            raise GenerationIntegrityError("governance registry binding does not match")
        return cls(bundle_root, directory, manifest)

    @property
    def id(self) -> str:
        return json.loads(self._manifest)["generation_id"]

    @property
    def digest(self) -> str:
        return json.loads(self._manifest)["generation_digest"]

    @property
    def manifest(self) -> Dict[str, Any]:
        return json.loads(self._manifest)

    @property
    def manifest_path(self) -> Path:
        return self._directory / "generation.json"

    @property
    def manifest_digest(self) -> str:
        with self.lease():
            return _sha256_path(self.manifest_path)

    @property
    def summary(self) -> Dict[str, Any]:
        with self.lease():
            return json.loads(
                (self._directory / "summary.json").read_text(encoding="utf-8")
            )

    @contextmanager
    def lease(self) -> Iterator["AssessmentGeneration"]:
        """Hold this private generation stable for one complete read operation."""

        with _GenerationLease(self._root, self.id):
            yield self

    def get_case(self, case_id: str) -> Dict[str, Any]:
        with self.lease():
            with sqlite3.connect(
                f"file:{(self._directory / 'assessment.sqlite3').as_posix()}?mode=ro",
                uri=True,
            ) as connection:
                row = connection.execute(
                    "SELECT assessment_json FROM case_assessments WHERE case_id=?",
                    (case_id,),
                ).fetchone()
        if row is None:
            raise KeyError(case_id)
        return json.loads(row[0])

    @contextmanager
    def _reader(self) -> Iterator[_AssessmentReader]:
        """Hold one lease and one read-only connection for a batch read."""

        assessment_path = self._directory / "assessment.sqlite3"
        with self.lease():
            connection = sqlite3.connect(
                f"file:{assessment_path.as_posix()}?mode=ro", uri=True
            )
            try:
                yield _AssessmentReader(connection)
            finally:
                connection.close()

    def iter_cases(self) -> Iterator[Dict[str, Any]]:
        with self.lease():
            connection = sqlite3.connect(
                f"file:{(self._directory / 'assessment.sqlite3').as_posix()}?mode=ro",
                uri=True,
            )
            try:
                for row in connection.execute(
                    "SELECT assessment_json FROM case_assessments ORDER BY public_index,case_id"
                ):
                    yield json.loads(row[0])
            finally:
                connection.close()

    def iter_archive_ids(self, archive_class: str) -> Iterator[str]:
        if archive_class not in _ARCHIVE_CLASSES:
            raise ValueError(f"unknown archive class: {archive_class}")
        with self.lease():
            with (self._directory / f"{archive_class}.ids").open(
                "r", encoding="utf-8"
            ) as stream:
                for line in stream:
                    yield line.rstrip("\n")


__all__ = [
    "AssessmentGeneration",
    "AssessmentGenerationError",
    "AssetSourceChangedError",
    "GenerationIntegrityError",
]
