"""Frozen identity for the one current dataset-assessment runtime.

The runtime policy is scientific input.  It binds the Python/SQLite runtime,
the assessment implementation, the isolated-checker protocol, and every
configured structure parser in one canonically digestible JSON document.
Structural validation of a frozen document is deliberately separate from an
ambient-runtime comparison so a certificate can prove what was used without
silently loading a compatibility implementation.
"""

from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import re
import sqlite3
import sys
import sysconfig
from typing import Any, Mapping

from ..gates._checker import CHECKER_PROTOCOL_DIGEST, _CHECKER_PROTOCOL


RUNTIME_POLICY_SCHEMA = "hotpot-ds-runtime-policy/current"
ASSESSMENT_RUNTIME_SCHEMA = "hotpot-ds-assessment-runtime/current"
ASSESSMENT_STORE_PROTOCOL_SCHEMA = "hotpot-ds-assessment-store/current"
CHECKER_PROTOCOL_SCHEMA = "hotpot-ds-checker/current"
PARSER_REGISTRY_SCHEMA = "hotpot-ds-parser-registry/current"
PARSER_BINDING_SCHEMA = "hotpot-ds-parser-binding/current"
PARSER_PROTOCOL_SCHEMA = "hotpot-ds-molecule-parser/current"
PARSER_SELECTION_SCHEMA = "hotpot-ds-parser-selection/current"

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_ASSESSMENT_IMPLEMENTATION_FILES = (
    "ds/gates/_base.py",
    "ds/gates/_checker.py",
    "ds/gates/_gates.py",
    "ds/gates/_persistence.py",
    "ds/governance.py",
    "ds/public/_assessment.py",
    "ds/public/_implementation.py",
)
_PARSER_IMPLEMENTATION_FILES = (
    "__init__.py",
    "cheminfo/__init__.py",
    "cheminfo/_io/__init__.py",
    "cheminfo/_io/_io.py",
    "cheminfo/core.py",
    "cheminfo/crystal.py",
    "cheminfo/elements.py",
    "cheminfo/obconvert.py",
)
_SEMANTIC_SYMBOLS = {
    "ds/public/_bundle_state.py": (
        "BUNDLE_SCHEMA",
        "_DATABASE_SCHEMA_VERSION",
        "_SOURCE_PROOF_SCHEMA",
        "_SOURCE_DIGEST_STATES",
        "_canonical_bytes",
        "_safe_relative_path",
        "_CASE_ALLOWED",
        "_CASE_DEFAULTS",
        "_ARRAY_FIELDS",
        "_ASSET_FIELDS",
        "_RECORD_ID_FIELDS",
        "_validate_control_assets",
        "_validate_manifest",
        "_validate_record_list",
        "_validate_case",
        "_record_asset_paths",
        "_frame",
        "_file_stamp",
        "_source_path_observation",
        "_assert_source_observations_current",
        "_stable_file_hash",
        "_digest_file",
        "_control_asset_source_binding",
        "_compute_source_proof_rows",
        "_SynchronizationSnapshot",
        "BundleState.build",
        "BundleState.connect",
        "BundleState.source_digest",
        "BundleState.status",
        "BundleState.refresh_source_state",
        "BundleState.refresh_governance_state",
        "BundleState._prepare_synchronization",
        "BundleState.prepare_synchronization",
        "BundleState.synchronization_snapshot",
        "BundleState.mutate_case",
        "BundleState.mutate_policy",
        "BundleState.publish_generation",
        "BundleState.refresh_audit_projection",
        "BundleState._append_audit_projection_event",
        "BundleState._initialize",
        "BundleState._stored_manifest",
        "BundleState._assert_manifest_unchanged",
        "BundleState._iter_intake",
        "BundleState._compute_source_proof",
        "BundleState._compute_source_digest",
    ),
    "ds/public/_generation.py": (
        "_DATABASE_SCHEMA_VERSION",
        "_ARCHIVE_CLASSES",
        "_PARSEABLE_FORMATS",
        "_canonical_bytes",
        "_json_copy",
        "_stable_digest",
        "_snapshot_asset",
        "_structure_is_parseable",
        "_normalize_case",
        "_case_with_verified_attestations",
        "_active",
        "_gate_info",
        "_assessment_policy",
        "_expected_sha256",
        "_asset_declarations",
        "_assess_asset",
        "_initialize_assessment_database",
        "_initialize_work_database",
        "_stream_inputs",
        "_assess_conflicts",
        "_assess_cases",
        "_grouped_support_rows",
        "_assessment_products",
        "_write_assessment_products",
        "_validate_assessment_products",
        "_ordered_assessment_digest",
        "_DIGEST_QUERIES",
        "AssessmentGeneration.build",
    ),
    "ds/public/bundle.py": (
        "_gate_set",
        "_policy_value",
        "_policy_has_custom_checkers",
        "Bundle.use_gate",
        "Bundle.synchronize",
        "Bundle._mutate_case",
        "_assert_manifest_policy",
    ),
    "ds/public/case.py": (
        "_EVIDENCE_DETAIL_FIELDS",
        "_VERIFICATION_DETAIL_FIELDS",
        "_VERIFICATION_PROCEDURE",
        "_evidence_details",
        "_verification_details",
        "Case",
    ),
}

# Every distribution imported by the assessment boundary or the molecular
# parser is recorded, whether present or absent.  Absence is therefore part of
# the runtime identity rather than an unrecorded ambient fact.
_RUNTIME_DISTRIBUTIONS = (
    "cryptography",
    "Cython",
    "hotpot-zzy",
    "networkx",
    "numpy",
    "openbabel",
    "openbabel-wheel",
    "pandas",
    "periodictable",
    "PubChemPy",
    "rdkit",
    "scipy",
    "thermo",
)
_PARSER_DISTRIBUTIONS = tuple(
    name for name in _RUNTIME_DISTRIBUTIONS if name != "cryptography"
)
_PARSER_FORMATS = ("mol", "mol2", "sdf", "smi", "smiles", "xyz")

_ASSESSMENT_STORE_PROTOCOL = {
    "schema": ASSESSMENT_STORE_PROTOCOL_SCHEMA,
    "facts_sqlite_user_version": 4,
    "source_proof_schema": "hotpot-ds-source-proof/current",
    "sqlite_user_version": 1,
    "ordered_digest_tables": [
        "asset_assessments",
        "conflict_assessments",
        "conflict_groups",
        "case_assessments",
        "evidence_assessments",
    ],
}
_PARSER_PROTOCOL = {
    "schema": PARSER_PROTOCOL_SCHEMA,
    "entrypoint": "hotpot.Molecule.read_one",
    "input": "content-addressed-local-snapshot-path",
    "format_argument": "normalized-declared-format",
    "result": "first-hotpot-molecule-without-parser-exception",
    "backend": "openbabel",
}
_PARSER_SELECTION_PROTOCOL = {
    "schema": PARSER_SELECTION_SCHEMA,
    "normalization": "strip-leading-dot-and-ascii-lowercase",
    "asset_collection": "structures",
    "unknown_format": "unconfigured",
}


class PolicyProtocolError(RuntimeError):
    """A runtime-policy document is invalid or not the current runtime."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest_document(value: Mapping[str, Any]) -> str:
    """Digest a self-identifying document without its top-level digest."""

    return _digest_value({key: item for key, item in value.items() if key != "digest"})


def _with_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document["digest"] = _digest_document(document)
    return document


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assignment_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return None
    return target.id if isinstance(target, ast.Name) else None


def _find_symbol(tree: ast.Module, qualified_name: str) -> ast.AST:
    names = qualified_name.split(".")
    body = tree.body
    node = None
    for name in names:
        matches = []
        for candidate in body:
            declared = (
                candidate.name
                if isinstance(
                    candidate, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
                else _assignment_name(candidate)
            )
            if declared == name:
                matches.append(candidate)
        if len(matches) != 1:
            raise RuntimeError(
                f"semantic implementation symbol is not unique: {qualified_name}"
            )
        node = matches[0]
        body = getattr(node, "body", ())
    return node


def _semantic_source_digest(path: Path, symbols: tuple[str, ...]) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = []
    for name in symbols:
        node = _find_symbol(tree, name)
        segment = ast.get_source_segment(source, node)
        if segment is None:
            raise RuntimeError(f"cannot recover semantic source: {name}")
        selected.append({"symbol": name, "source": segment})
    return _digest_value(selected)


def _source_binding(
    files: tuple[str, ...],
    semantic_symbols: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    value = {
        "files": [
            {"path": relative, "sha256": _sha256(_PACKAGE_ROOT / relative)}
            for relative in sorted(files)
        ],
        "semantic_symbols": [
            {
                "path": relative,
                "symbols": list(symbols),
                "sha256": _semantic_source_digest(_PACKAGE_ROOT / relative, symbols),
            }
            for relative, symbols in sorted((semantic_symbols or {}).items())
        ],
    }
    return _with_digest(value)


def _distribution_identity(name: str) -> dict[str, Any]:
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    try:
        installed_version: str | None = version(name)
    except PackageNotFoundError:
        installed_version = None
    return {
        "name": normalized,
        "installed": installed_version is not None,
        "version": installed_version,
    }


def _distribution_identities(names: tuple[str, ...]) -> list[dict[str, Any]]:
    return sorted(
        (_distribution_identity(name) for name in names),
        key=lambda item: item["name"],
    )


def _parser_binding() -> dict[str, Any]:
    protocol = _with_digest(_PARSER_PROTOCOL)
    source = _source_binding(_PARSER_IMPLEMENTATION_FILES)
    value = {
        "schema": PARSER_BINDING_SCHEMA,
        "name": "hotpot.Molecule.read_one/openbabel",
        "formats": list(_PARSER_FORMATS),
        "protocol": protocol,
        "source": source,
        "distributions": _distribution_identities(_PARSER_DISTRIBUTIONS),
    }
    return _with_digest(value)


def _build_runtime_policy() -> dict[str, Any]:
    assessment_source = _source_binding(
        _ASSESSMENT_IMPLEMENTATION_FILES,
        _SEMANTIC_SYMBOLS,
    )
    store_protocol = _with_digest(_ASSESSMENT_STORE_PROTOCOL)
    checker = {
        "schema": CHECKER_PROTOCOL_SCHEMA,
        "protocol": _json_copy(_CHECKER_PROTOCOL),
        "digest": CHECKER_PROTOCOL_DIGEST,
    }
    parser = _parser_binding()
    selection = _with_digest(_PARSER_SELECTION_PROTOCOL)
    parsers = _with_digest(
        {
            "schema": PARSER_REGISTRY_SCHEMA,
            "selection": selection,
            "entries": [parser],
        }
    )
    runtime = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "hexversion": sys.hexversion,
            "cache_tag": str(sys.implementation.cache_tag),
            "soabi": str(sysconfig.get_config_var("SOABI")),
        },
        "sqlite": {"version": sqlite3.sqlite_version},
        "distributions": _distribution_identities(_RUNTIME_DISTRIBUTIONS),
    }
    return _with_digest(
        {
            "schema": RUNTIME_POLICY_SCHEMA,
            "assessment": {
                "schema": ASSESSMENT_RUNTIME_SCHEMA,
                "source": assessment_source,
                "store_protocol": store_protocol,
            },
            "checker": checker,
            "parsers": parsers,
            "runtime": runtime,
        }
    )


@lru_cache(maxsize=1)
def _runtime_policy_snapshot() -> str:
    return _canonical_bytes(_build_runtime_policy()).decode("utf-8")


def runtime_policy() -> dict[str, Any]:
    """Return a defensive copy of the one current runtime-policy document."""

    return json.loads(_runtime_policy_snapshot())


def runtime_policy_digest() -> str:
    """Return the canonical digest embedded in the current runtime policy."""

    return runtime_policy()["digest"]


def _clear_runtime_policy_cache() -> None:
    _runtime_policy_snapshot.cache_clear()


def _require_object(value: Any, fields: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PolicyProtocolError(f"{context} fields are invalid")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyProtocolError(f"{context} must be a non-empty string")
    return value


def _require_digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise PolicyProtocolError(f"{context} is not a SHA-256 digest")
    return value


def _require_string_list(
    value: Any,
    context: str,
    *,
    canonical_order: bool = False,
) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise PolicyProtocolError(f"{context} must contain unique non-empty strings")
    if canonical_order and value != sorted(value):
        raise PolicyProtocolError(f"{context} is not canonically ordered")
    return value


def _validate_self_digest(value: Mapping[str, Any], context: str) -> None:
    digest = _require_digest(value.get("digest"), f"{context} digest")
    if digest != _digest_document(value):
        raise PolicyProtocolError(f"{context} digest does not match its content")


def _validate_source(value: Any, context: str) -> None:
    source = _require_object(value, {"files", "semantic_symbols", "digest"}, context)
    files = source["files"]
    symbols = source["semantic_symbols"]
    if not isinstance(files, list) or not isinstance(symbols, list):
        raise PolicyProtocolError(f"{context} inventories must be arrays")
    file_paths = []
    for offset, item in enumerate(files):
        entry = _require_object(item, {"path", "sha256"}, f"{context} file {offset}")
        file_paths.append(_require_string(entry["path"], f"{context} file path"))
        _require_digest(entry["sha256"], f"{context} file digest")
    if file_paths != sorted(set(file_paths)):
        raise PolicyProtocolError(f"{context} files are not canonically ordered")
    symbol_paths = []
    for offset, item in enumerate(symbols):
        entry = _require_object(
            item,
            {"path", "symbols", "sha256"},
            f"{context} semantic source {offset}",
        )
        symbol_paths.append(_require_string(entry["path"], f"{context} semantic path"))
        names = entry["symbols"]
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise PolicyProtocolError(f"{context} semantic symbols are invalid")
        _require_digest(entry["sha256"], f"{context} semantic digest")
    if symbol_paths != sorted(set(symbol_paths)):
        raise PolicyProtocolError(f"{context} semantic sources are not canonically ordered")
    _validate_self_digest(source, context)


def _validate_distribution_inventory(value: Any, context: str) -> None:
    if not isinstance(value, list):
        raise PolicyProtocolError(f"{context} must be an array")
    names = []
    for offset, item in enumerate(value):
        entry = _require_object(
            item,
            {"name", "installed", "version"},
            f"{context} item {offset}",
        )
        name = _require_string(entry["name"], f"{context} name")
        if name != re.sub(r"[-_.]+", "-", name).lower():
            raise PolicyProtocolError(f"{context} name is not normalized")
        names.append(name)
        installed = entry["installed"]
        installed_version = entry["version"]
        if type(installed) is not bool:
            raise PolicyProtocolError(f"{context} installed marker must be boolean")
        if installed:
            _require_string(installed_version, f"{context} version")
        elif installed_version is not None:
            raise PolicyProtocolError(f"{context} absent distribution has a version")
    if names != sorted(set(names)):
        raise PolicyProtocolError(f"{context} is not canonically ordered")


def _validate_parser_binding(value: Any) -> None:
    parser = _require_object(
        value,
        {"schema", "name", "formats", "protocol", "source", "distributions", "digest"},
        "parser binding",
    )
    if parser["schema"] != PARSER_BINDING_SCHEMA:
        raise PolicyProtocolError("parser binding schema is not current")
    _require_string(parser["name"], "parser name")
    formats = _require_string_list(
        parser["formats"], "parser formats", canonical_order=True
    )
    if not formats:
        raise PolicyProtocolError("parser formats must not be empty")
    protocol = _require_object(
        parser["protocol"],
        {
            "schema",
            "entrypoint",
            "input",
            "format_argument",
            "result",
            "backend",
            "digest",
        },
        "parser protocol",
    )
    if protocol["schema"] != PARSER_PROTOCOL_SCHEMA:
        raise PolicyProtocolError("parser protocol schema is not current")
    for field in ("entrypoint", "input", "format_argument", "result", "backend"):
        _require_string(protocol[field], f"parser protocol {field}")
    _validate_self_digest(protocol, "parser protocol")
    _validate_source(parser["source"], "parser source")
    _validate_distribution_inventory(parser["distributions"], "parser distributions")
    _validate_self_digest(parser, "parser binding")


def _validate_checker_protocol(value: Any) -> None:
    protocol = _require_object(
        value,
        {
            "arguments",
            "entrypoint",
            "imports",
            "isolation",
            "request_fields",
            "response_variants",
            "result",
            "worker_source_sha256",
        },
        "checker protocol content",
    )
    arguments = protocol["arguments"]
    if not isinstance(arguments, list) or not arguments:
        raise PolicyProtocolError("checker protocol arguments must be an array")
    argument_names = []
    for offset, item in enumerate(arguments):
        argument = _require_object(
            item, {"name", "type"}, f"checker protocol argument {offset}"
        )
        argument_names.append(
            _require_string(argument["name"], "checker protocol argument name")
        )
        _require_string(argument["type"], "checker protocol argument type")
    if len(argument_names) != len(set(argument_names)):
        raise PolicyProtocolError("checker protocol argument names are duplicated")
    _require_string(protocol["entrypoint"], "checker protocol entrypoint")
    _require_string_list(
        protocol["imports"], "checker protocol imports", canonical_order=True
    )

    isolation = _require_object(
        protocol["isolation"],
        {"filesystem", "network", "process", "environment", "limits"},
        "checker isolation",
    )
    for field in ("filesystem", "network", "process"):
        _require_string(isolation[field], f"checker isolation {field}")
    environment = isolation["environment"]
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        for key, item in environment.items()
    ):
        raise PolicyProtocolError("checker isolation environment is invalid")
    limits = isolation["limits"]
    if not isinstance(limits, Mapping) or any(
        not isinstance(key, str)
        or not key
        or type(item) is not int
        or item < 1
        for key, item in limits.items()
    ):
        raise PolicyProtocolError("checker isolation limits are invalid")

    _require_string_list(protocol["request_fields"], "checker request fields")
    response_variants = _require_object(
        protocol["response_variants"],
        {"success_fields", "error_fields"},
        "checker response variants",
    )
    _require_string_list(
        response_variants["success_fields"], "checker success fields"
    )
    _require_string_list(response_variants["error_fields"], "checker error fields")
    _require_string_list(protocol["result"], "checker result fields")
    _require_digest(
        protocol["worker_source_sha256"], "checker worker source digest"
    )


def validate_frozen_runtime_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a frozen policy without comparing it to this process."""

    policy = _require_object(
        value,
        {"schema", "assessment", "checker", "parsers", "runtime", "digest"},
        "runtime policy",
    )
    if policy["schema"] != RUNTIME_POLICY_SCHEMA:
        raise PolicyProtocolError("runtime policy schema is not current")

    assessment = _require_object(
        policy["assessment"],
        {"schema", "source", "store_protocol"},
        "assessment runtime",
    )
    if assessment["schema"] != ASSESSMENT_RUNTIME_SCHEMA:
        raise PolicyProtocolError("assessment runtime schema is not current")
    _validate_source(assessment["source"], "assessment source")
    store = _require_object(
        assessment["store_protocol"],
        {
            "schema",
            "facts_sqlite_user_version",
            "source_proof_schema",
            "sqlite_user_version",
            "ordered_digest_tables",
            "digest",
        },
        "assessment store protocol",
    )
    if store["schema"] != ASSESSMENT_STORE_PROTOCOL_SCHEMA:
        raise PolicyProtocolError("assessment store protocol schema is not current")
    if type(store["sqlite_user_version"]) is not int or store["sqlite_user_version"] < 1:
        raise PolicyProtocolError("assessment store version is invalid")
    if store["facts_sqlite_user_version"] != 4:
        raise PolicyProtocolError("facts store version is invalid")
    if store["source_proof_schema"] != "hotpot-ds-source-proof/current":
        raise PolicyProtocolError("source proof schema is not current")
    tables = store["ordered_digest_tables"]
    if (
        not isinstance(tables, list)
        or not tables
        or any(not isinstance(name, str) or not name for name in tables)
        or len(tables) != len(set(tables))
    ):
        raise PolicyProtocolError("assessment digest-table order is invalid")
    _validate_self_digest(store, "assessment store protocol")

    checker = _require_object(
        policy["checker"], {"schema", "protocol", "digest"}, "checker protocol"
    )
    if checker["schema"] != CHECKER_PROTOCOL_SCHEMA:
        raise PolicyProtocolError("checker protocol schema is not current")
    _validate_checker_protocol(checker["protocol"])
    if _require_digest(checker["digest"], "checker protocol digest") != _digest_value(
        checker["protocol"]
    ):
        raise PolicyProtocolError("checker protocol digest does not match its content")

    parsers = _require_object(
        policy["parsers"],
        {"schema", "selection", "entries", "digest"},
        "parser registry",
    )
    if parsers["schema"] != PARSER_REGISTRY_SCHEMA:
        raise PolicyProtocolError("parser registry schema is not current")
    selection = _require_object(
        parsers["selection"],
        {"schema", "normalization", "asset_collection", "unknown_format", "digest"},
        "parser selection protocol",
    )
    if selection["schema"] != PARSER_SELECTION_SCHEMA:
        raise PolicyProtocolError("parser selection schema is not current")
    for field in ("normalization", "asset_collection", "unknown_format"):
        _require_string(selection[field], f"parser selection {field}")
    _validate_self_digest(selection, "parser selection protocol")
    entries = parsers["entries"]
    if not isinstance(entries, list):
        raise PolicyProtocolError("parser registry entries must be an array")
    for entry in entries:
        _validate_parser_binding(entry)
    names = [entry["name"] for entry in entries]
    if names != sorted(set(names)):
        raise PolicyProtocolError("parser registry entries are not canonically ordered")
    claimed_formats = [fmt for entry in entries for fmt in entry["formats"]]
    if len(claimed_formats) != len(set(claimed_formats)):
        raise PolicyProtocolError("a parser format has multiple bindings")
    _validate_self_digest(parsers, "parser registry")

    runtime = _require_object(
        policy["runtime"], {"python", "sqlite", "distributions"}, "runtime identity"
    )
    python = _require_object(
        runtime["python"],
        {"implementation", "version", "hexversion", "cache_tag", "soabi"},
        "Python identity",
    )
    for field in ("implementation", "version", "cache_tag", "soabi"):
        _require_string(python[field], f"Python {field}")
    if type(python["hexversion"]) is not int:
        raise PolicyProtocolError("Python hexversion is invalid")
    sqlite = _require_object(runtime["sqlite"], {"version"}, "SQLite identity")
    _require_string(sqlite["version"], "SQLite version")
    _validate_distribution_inventory(runtime["distributions"], "runtime distributions")

    _validate_self_digest(policy, "runtime policy")
    return _json_copy(policy)


def runtime_policy_is_current(value: Mapping[str, Any]) -> bool:
    """Return whether a valid frozen policy is exactly this process's policy."""

    try:
        frozen = validate_frozen_runtime_policy(value)
    except PolicyProtocolError:
        return False
    return frozen["digest"] == runtime_policy_digest() and frozen == runtime_policy()


def require_current_runtime_policy(value: Mapping[str, Any]) -> None:
    """Reject invalid or non-current runtime input without adapting it."""

    validate_frozen_runtime_policy(value)
    if not runtime_policy_is_current(value):
        raise PolicyProtocolError(
            "dataset runtime policy is not current; an explicit owner policy "
            "migration, reassessment, and recertification are required"
        )


def parser_binding_for_format(
    format_name: str,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the frozen parser binding selected for ``format_name``."""

    selected_policy = (
        runtime_policy()
        if policy is None
        else validate_frozen_runtime_policy(policy)
    )
    normalized = format_name.lower().lstrip(".")
    for parser in selected_policy["parsers"]["entries"]:
        if normalized in parser["formats"]:
            return _json_copy(parser)
    return None


__all__ = [
    "PolicyProtocolError",
    "parser_binding_for_format",
    "require_current_runtime_policy",
    "runtime_policy",
    "runtime_policy_digest",
    "runtime_policy_is_current",
    "validate_frozen_runtime_policy",
]
