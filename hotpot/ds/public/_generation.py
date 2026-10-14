"""Immutable full-Bundle assessment generations.

This module is private infrastructure.  It deliberately keeps source facts in
``BundleState`` and derived assessment results in a separately verifiable,
content-addressed generation directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from ._assessment import (
    assess_case,
    canonical_digest,
    canonical_json,
    compare_observations,
    identity_key,
    summarize_assessments,
)
from ._bundle_state import BundleState


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
    "ordered_assessment_digest",
    "summary_digest",
    "files",
}


class AssessmentGenerationError(ValueError):
    """Base error for generation construction or validation failures."""


class GenerationIntegrityError(AssessmentGenerationError):
    """Raised when an immutable generation fails content verification."""


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


def _identity_schema(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    schema = _json_copy(manifest.get("identity_schema", {}))
    if "components" not in schema and isinstance(schema.get("fields"), list):
        schema["components"] = [
            {
                "key": field,
                "path": field,
                "required": True,
                "normalizer": "text",
            }
            for field in schema["fields"]
        ]
    return schema


def _gate_info(policy: Mapping[str, Any], gate_name: str) -> Dict[str, Any]:
    gates = policy.get("gates", {})
    candidates: Sequence[Any]
    if isinstance(gates, Mapping):
        candidates = (
            gates.get(gate_name),
            gates.get(gate_name.removesuffix("Gate").lower()),
        )
    elif isinstance(gates, list):
        candidates = tuple(gates)
    else:
        candidates = ()
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("gate", gate_name) == gate_name:
            return _json_copy(candidate)
    direct = policy.get(gate_name.removesuffix("Gate").lower(), {})
    return _json_copy(direct) if isinstance(direct, Mapping) else {}


def _enabled_checks(value: Any) -> Dict[str, bool]:
    if not isinstance(value, Mapping):
        return {}
    enabled = {}
    for name, setting in value.items():
        if isinstance(setting, Mapping):
            enabled[str(name)] = bool(setting.get("enabled", True))
        else:
            enabled[str(name)] = setting is not False
    return enabled


def _assessment_policy(policy: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    benchmark = _gate_info(policy, "BenchmarkGate")
    quality = _gate_info(policy, "QualityGate")
    benchmark_policy = {
        "checklist": _enabled_checks(benchmark.get("checklist", {})),
        "direct_entry": _enabled_checks(benchmark.get("direct_entry", {})),
        "vetoes": list(benchmark.get("vetoes", ())),
        "numeric_tolerances": _json_copy(benchmark.get("numeric_tolerances", {})),
        "structure_checks": _enabled_checks(quality.get("structure_checks", {})),
    }
    return benchmark_policy, benchmark_policy["numeric_tolerances"]


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
    observed = _sha256_path(path) if present else None
    expected = declaration.get("expected_sha256")
    if expected is None or not present:
        hash_result = "unchecked"
    else:
        hash_result = "match" if observed == expected else "mismatch"

    declared_format = declaration.get("format")
    fmt = str(declared_format).lower().lstrip(".") if declared_format else ""
    if not fmt and path is not None:
        fmt = path.suffix.lower().lstrip(".")
    parse_result = "not_run"
    parser = None
    parse_reason = None
    if declaration.get("collection") == "structures" and present and fmt in _PARSEABLE_FORMATS:
        parser = "hotpot.Molecule.read_one"
        try:
            from hotpot import Molecule

            molecule = Molecule.read_one(
                str(path), "smi" if fmt == "smiles" else fmt
            )
            parse_result = "valid" if molecule is not None and molecule.atoms else "invalid"
        except Exception:
            parse_result = "invalid"
        if parse_result == "invalid":
            parse_reason = "structure_parse_failed"

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
            and declaration.get("role") in required_roles
        ),
        "parser": parser,
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

        CREATE TABLE conflict_groups(
            group_id TEXT PRIMARY KEY,
            sample_key TEXT NOT NULL,
            property_key TEXT NOT NULL,
            assessment_json TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )


def _initialize_work_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=OFF;
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
    state: BundleState,
    assessment: sqlite3.Connection,
    work: sqlite3.Connection,
    identity_schema: Mapping[str, Any],
    required_roles: Sequence[str],
) -> None:
    for source_case in state.iter_cases():
        case = _normalize_case(source_case)
        case_id = str(case["id"])
        key, reasons = identity_key(case, identity_schema)
        work.execute(
            "INSERT INTO identity_reasons(case_id,reasons_json) VALUES (?,?)",
            (case_id, canonical_json(reasons)),
        )
        if key is not None:
            for offset, observation in enumerate(_active(case.get("observations", []))):
                value = _json_copy(observation)
                observation_id = str(
                    value.get("observation_id") or f"{case_id}:observation:{offset}"
                )
                value["observation_id"] = observation_id
                property_key = str(
                    value.get("property")
                    or value.get("property_type_id")
                    or value.get("property_type")
                    or "null"
                )
                work.execute(
                    """
                    INSERT INTO candidates(
                        sample_key,property_key,case_id,observation_id,observation_json
                    ) VALUES (?,?,?,?,?)
                    """,
                    (key, property_key, case_id, observation_id, canonical_json(value)),
                )

        for collection in ("structures", "evidence"):
            for record in _active(case.get(collection, [])):
                for declaration in _asset_declarations(case_id, collection, record):
                    asset = _assess_asset(state.root, declaration, required_roles)
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
) -> None:
    groups = work.execute(
        "SELECT DISTINCT sample_key,property_key FROM candidates "
        "ORDER BY sample_key,property_key"
    )
    for sample_key, property_key in groups:
        rows = list(
            work.execute(
                """
                SELECT case_id,observation_id,observation_json FROM candidates
                WHERE sample_key=? AND property_key=?
                ORDER BY case_id,observation_id
                """,
                (sample_key, property_key),
            )
        )
        candidate_count = 0
        consistent_count = 0
        exception_count = 0
        ordered = hashlib.sha256()
        for left_index, left_row in enumerate(rows):
            for right_row in rows[left_index + 1 :]:
                left = json.loads(left_row[2])
                right = json.loads(right_row[2])
                result = compare_observations(left, right, tolerances)
                if result["outcome"] == "noncandidate":
                    continue
                pair = {
                    "sample_key": sample_key,
                    "property_key": property_key,
                    "left_case_id": left_row[0],
                    "right_case_id": right_row[0],
                    "left_observation_id": left_row[1],
                    "right_observation_id": right_row[1],
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
                        left_row[0],
                        right_row[0],
                        result["outcome"],
                        canonical_json(pair),
                    ),
                )
                if result["outcome"] == "conflict":
                    work.execute(
                        "INSERT OR IGNORE INTO conflicting_cases(case_id) VALUES (?)",
                        (left_row[0],),
                    )
                    work.execute(
                        "INSERT OR IGNORE INTO conflicting_cases(case_id) VALUES (?)",
                        (right_row[0],),
                    )
        group = {
            "sample_key": sample_key,
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


def _assess_cases(
    state: BundleState,
    assessment: sqlite3.Connection,
    work: sqlite3.Connection,
    manifest: Mapping[str, Any],
    benchmark_policy: Mapping[str, Any],
    governance_registry: Any,
) -> None:
    for source_case in state.iter_cases():
        case, governance_assessments = _case_with_verified_attestations(
            source_case, manifest, governance_registry
        )
        case_id = str(case["id"])
        assets = [
            json.loads(row[0])
            for row in assessment.execute(
                "SELECT assessment_json FROM asset_assessments "
                "WHERE case_id=? ORDER BY asset_id",
                (case_id,),
            )
        ]
        conflict_rows = [
            json.loads(row[0])
            for row in assessment.execute(
                """
                SELECT assessment_json FROM conflict_assessments
                WHERE left_case_id=? OR right_case_id=?
                ORDER BY conflict_id
                """,
                (case_id, case_id),
            )
        ]
        identity_reasons = json.loads(
            work.execute(
                "SELECT reasons_json FROM identity_reasons WHERE case_id=?",
                (case_id,),
            ).fetchone()[0]
        )
        conflicting = (
            work.execute(
                "SELECT 1 FROM conflicting_cases WHERE case_id=?", (case_id,)
            ).fetchone()
            is not None
        )
        result = assess_case(
            case,
            core_labels=manifest["core_labels"],
            core_info=manifest["core_info"],
            required_structure_roles=manifest["required_structure_roles"],
            asset_assessments=assets,
            conflicting=conflicting,
            benchmark_policy=benchmark_policy,
        )
        result["input_digest"] = canonical_digest(source_case)
        result["asset_assessments"] = assets
        result["conflict_assessments"] = conflict_rows
        result["identity_reasons"] = identity_reasons
        result["governance_assessments"] = governance_assessments
        result["reasons"].extend(identity_reasons)
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
    digest = hashlib.sha256()
    for table, query in _DIGEST_QUERIES:
        table_bytes = table.encode("utf-8")
        digest.update(len(table_bytes).to_bytes(8, "big"))
        digest.update(table_bytes)
        for row in connection.execute(query):
            value = [table, *row]
            encoded = _canonical_bytes(value)
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


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
    ) -> "AssessmentGeneration":
        bundle_root = Path(root).resolve()
        if bundle_root != state.root:
            raise AssessmentGenerationError("generation root does not match BundleState")
        status = state.status
        if status["source_digest"] != status["recorded_source_digest"]:
            raise AssessmentGenerationError(
                "source assets changed; refresh BundleState before synchronization"
            )
        if status["policy_digest"] != status["recorded_policy_digest"]:
            raise AssessmentGenerationError(
                "policy changed outside BundleState; persist it before synchronization"
            )
        policy_value = _json_copy(policy)
        if canonical_digest(policy_value) != status["policy_digest"]:
            raise AssessmentGenerationError("supplied policy is not the persisted policy")

        from ..governance import verify_registry

        verified_registry = verify_registry(governance_registry)
        registry = verified_registry.to_dict()
        registry_digest = verified_registry.digest
        manifest = state.manifest
        identity_schema = _identity_schema(manifest)
        benchmark_policy, tolerances = _assessment_policy(policy_value)

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
                    state,
                    assessment,
                    work,
                    identity_schema,
                    manifest["required_structure_roles"],
                )
                _assess_conflicts(assessment, work, tolerances)
                _assess_cases(
                    state,
                    assessment,
                    work,
                    manifest,
                    benchmark_policy,
                    verified_registry,
                )
                for key, value in (
                    ("source_revision", status["source_revision"]),
                    ("source_digest", status["source_digest"]),
                    ("policy_revision", status["policy_revision"]),
                    ("policy_digest", status["policy_digest"]),
                    ("governance_registry_digest", registry_digest),
                ):
                    assessment.execute(
                        "INSERT INTO metadata(key,value_json) VALUES (?,?)",
                        (key, canonical_json(value)),
                    )
            ordered_digest = _ordered_assessment_digest(assessment)
            summary = summarize_assessments(
                json.loads(row[0])
                for row in assessment.execute(
                    "SELECT assessment_json FROM case_assessments ORDER BY public_index,case_id"
                )
            )
            assessment.execute("VACUUM")
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
            for archive_class in _ARCHIVE_CLASSES:
                with sqlite3.connect(assessment_path) as connection:
                    identifiers = [
                        row[0]
                        for row in connection.execute(
                            "SELECT case_id FROM case_assessments "
                            "WHERE archive_class=? ORDER BY public_index,case_id",
                            (archive_class,),
                        )
                    ]
                encoded = "".join(f"{case_id}\n" for case_id in identifiers).encode("utf-8")
                _write_fsynced(temporary / f"{archive_class}.ids", encoded)

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
        if (
            not isinstance(generation_id, str)
            or not generation_id.startswith("generation-")
            or "/" in generation_id
            or "\\" in generation_id
            or generation_id in {".", ".."}
        ):
            raise GenerationIntegrityError("invalid generation id")
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
            }
            if metadata != expected_metadata:
                raise GenerationIntegrityError(
                    "assessment metadata does not match generation bindings"
                )
            if _ordered_assessment_digest(connection) != manifest["ordered_assessment_digest"]:
                raise GenerationIntegrityError("ordered assessment digest does not match")
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            computed_summary = summarize_assessments(
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT assessment_json FROM case_assessments ORDER BY public_index,case_id"
                )
            )
            if summary != computed_summary or canonical_digest(summary) != manifest["summary_digest"]:
                raise GenerationIntegrityError("generation summary does not match assessments")
            for archive_class in _ARCHIVE_CLASSES:
                expected_ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT case_id FROM case_assessments "
                        "WHERE archive_class=? ORDER BY public_index,case_id",
                        (archive_class,),
                    )
                ]
                actual_ids = (directory / f"{archive_class}.ids").read_text(
                    encoding="utf-8"
                ).splitlines()
                if actual_ids != expected_ids:
                    raise GenerationIntegrityError(
                        f"{archive_class} index does not match assessments"
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
        return _sha256_path(self.manifest_path)

    @property
    def summary(self) -> Dict[str, Any]:
        return json.loads((self._directory / "summary.json").read_text(encoding="utf-8"))

    def get_case(self, case_id: str) -> Dict[str, Any]:
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

    def iter_cases(self) -> Iterator[Dict[str, Any]]:
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
        with (self._directory / f"{archive_class}.ids").open(
            "r", encoding="utf-8"
        ) as stream:
            for line in stream:
                yield line.rstrip("\n")


__all__ = [
    "AssessmentGeneration",
    "AssessmentGenerationError",
    "GenerationIntegrityError",
]
