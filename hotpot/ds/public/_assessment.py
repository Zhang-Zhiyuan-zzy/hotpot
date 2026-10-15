"""Deterministic Stage 13 assessment primitives.

The orchestration layer streams facts into immutable generations.  This module
contains the side-effect-free scientific/governance rules used for each Case
and conflict candidate, which keeps synchronization reproducible and testable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..gates._checker import (
    CheckerIsolationUnavailableError,
    _CheckerDefinition,
    evaluate_configured_check,
)


B_LEVELS = tuple(f"B{index}" for index in range(5))
E_LEVELS = tuple(f"E{index}" for index in range(5))

_PROPERTY_TYPES = {"integer", "real", "category", "text", "boolean", "json"}
_PROPERTY_DEFINITION_FIELDS = {
    "type",
    "schema_version",
    "unit",
    "allowed_values",
    "normalizer",
    "context_fields",
}
_PROPERTY_COMPARATOR_FIELDS = {
    "kind",
    "absolute_tolerance",
    "relative_tolerance",
    "use_uncertainty",
    "unit_conversions",
}
_CONTEXT_ROOTS = {
    "subject",
    "site",
    "phase",
    "conditions",
    "participants",
    "method",
    "branch",
}
_DEFAULT_CONTEXT_FIELDS = (
    "subject",
    "site",
    "phase",
    "conditions",
    "participants",
    "method",
    "branch",
)

_DEFAULT_EVIDENCE_CHECKS = {
    name: {"kind": "builtin", "enabled": True}
    for name in (
        "extracted_material",
        "explicit_report",
        "usable_locator",
        "primary_source_verified",
        "independent_corroboration",
    )
}
_DEFAULT_EVIDENCE_LEVELS = {
    "E0": (),
    "E1": ("extracted_material",),
    "E2": ("extracted_material", "explicit_report", "usable_locator"),
    "E3": (
        "extracted_material",
        "explicit_report",
        "usable_locator",
        "primary_source_verified",
    ),
    "E4": (
        "extracted_material",
        "explicit_report",
        "usable_locator",
        "primary_source_verified",
        "independent_corroboration",
    ),
}
_DEFAULT_QUALITY_CHECKS = {
    name: {"kind": "builtin", "enabled": True}
    for name in (
        "governed_collection",
        "provenance_complete",
        "subject_mapping_audited",
        "case_complete",
        "reviewed",
        "conflict_adjudicated",
        "primary_source_supported",
        "validation_controls",
        "independent_corroboration",
        "reproducible_policy",
    )
}
_DEFAULT_QUALITY_LEVELS = {
    "B0": (),
    "B1": ("provenance_complete", "subject_mapping_audited"),
    "B2": (
        "provenance_complete",
        "subject_mapping_audited",
        "case_complete",
        "reviewed",
        "conflict_adjudicated",
    ),
    "B3": (
        "provenance_complete",
        "subject_mapping_audited",
        "case_complete",
        "reviewed",
        "conflict_adjudicated",
        "primary_source_supported",
        "validation_controls",
    ),
    "B4": (
        "provenance_complete",
        "subject_mapping_audited",
        "case_complete",
        "reviewed",
        "conflict_adjudicated",
        "primary_source_supported",
        "validation_controls",
        "independent_corroboration",
        "reproducible_policy",
    ),
}
_DEFAULT_STRUCTURE_CHECKS = {
    name: {"kind": "builtin", "enabled": True}
    for name in (
        "asset_available",
        "asset_hash_valid",
        "asset_parseable",
        "declared_role_valid",
    )
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


_IDENTITY_COMPONENT_FIELDS = {
    "key",
    "path",
    "required",
    "value_kind",
    "normalizer",
    "comparator",
    "structure_role",
    "mismatch_is_decisive",
    "implementation_digest",
    "protocol_digest",
}
_IDENTITY_VALUE_KINDS = {"text", "number", "boolean", "set", "json"}
_IDENTITY_NORMALIZERS = {"text", "casefold", "number", "boolean", "set", "json"}
_IDENTITY_PATH_ROOTS = {"core_info", "identifiers"}
_STRUCTURE_IDENTITY_ROOTS = {
    "formal_charge",
    "atom_count",
    "molecular_graph",
    "stereochemistry",
    "isotopes",
    "protonation_state",
    "tautomer_state",
    "crystal_context",
    "metadata",
}
_IDENTITY_PROTOCOL = "hotpot-ds-identity/current"
_IDENTITY_IMPLEMENTATION_DIGEST = hashlib.sha256(
    b"hotpot-ds-identity/current:exact-components:v1"
).hexdigest()


def validate_identity_schema(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and materialize the one current identity-policy schema."""

    if not isinstance(value, Mapping) or set(value) != {"components"}:
        raise ValueError("identity_schema requires only components")
    components = value["components"]
    if not isinstance(components, list):
        raise ValueError("identity_schema.components must be an array")
    normalized = []
    keys = set()
    for index, raw in enumerate(components):
        prefix = f"identity_schema.components[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{prefix} must be an object")
        extra = set(raw) - _IDENTITY_COMPONENT_FIELDS
        if extra:
            raise ValueError(
                f"{prefix} contains unsupported fields: {', '.join(sorted(extra))}"
            )
        key = raw.get("key")
        path = raw.get("path")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{prefix}.key must be a non-empty string")
        if key in keys:
            raise ValueError(f"identity_schema contains duplicate key {key!r}")
        keys.add(key)
        if not isinstance(path, str) or not path.strip() or "__" in path:
            raise ValueError(f"{prefix}.path must be a dotted current-schema path")
        parts = path.split(".")
        if any(not part or not part.replace("_", "a").isalnum() for part in parts):
            raise ValueError(f"{prefix}.path contains an invalid path segment")
        required = raw.get("required", False)
        decisive = raw.get("mismatch_is_decisive", True)
        if type(required) is not bool or type(decisive) is not bool:
            raise ValueError(
                f"{prefix}.required and mismatch_is_decisive must be Boolean"
            )
        normalizer = raw.get("normalizer", "text")
        if normalizer not in _IDENTITY_NORMALIZERS:
            raise ValueError(f"{prefix}.normalizer is not current")
        default_kind = {
            "text": "text",
            "casefold": "text",
            "number": "number",
            "boolean": "boolean",
            "set": "set",
            "json": "json",
        }[normalizer]
        value_kind = raw.get("value_kind", default_kind)
        if value_kind not in _IDENTITY_VALUE_KINDS or (
            normalizer in {"casefold", "number", "boolean", "set"}
            and value_kind != default_kind
        ):
            raise ValueError(f"{prefix}.value_kind does not match its normalizer")
        comparator = raw.get("comparator", "exact")
        if comparator != "exact":
            raise ValueError(f"{prefix}.comparator must be 'exact'")
        structure_role = raw.get("structure_role")
        if structure_role is None:
            if len(parts) < 2 or parts[0] not in _IDENTITY_PATH_ROOTS:
                raise ValueError(
                    f"{prefix}.path must start with core_info or identifiers"
                )
        else:
            if not isinstance(structure_role, str) or not structure_role.strip():
                raise ValueError(f"{prefix}.structure_role must be non-empty")
            if parts[0] not in _STRUCTURE_IDENTITY_ROOTS:
                raise ValueError(
                    f"{prefix}.path is not a semantic structure identity field"
                )
        protocol_digest = canonical_digest(
            {
                "protocol": _IDENTITY_PROTOCOL,
                "value_kind": value_kind,
                "normalizer": normalizer,
                "comparator": comparator,
            }
        )
        for field, expected in (
            ("implementation_digest", _IDENTITY_IMPLEMENTATION_DIGEST),
            ("protocol_digest", protocol_digest),
        ):
            supplied = raw.get(field)
            if supplied is not None and supplied != expected:
                raise ValueError(f"{prefix}.{field} is not current")
        component = {
            "key": key,
            "path": path,
            "required": required,
            "value_kind": value_kind,
            "normalizer": normalizer,
            "comparator": comparator,
            "mismatch_is_decisive": decisive,
            "implementation_digest": _IDENTITY_IMPLEMENTATION_DIGEST,
            "protocol_digest": protocol_digest,
        }
        if structure_role is not None:
            component["structure_role"] = structure_role
        normalized.append(component)
    return {"components": normalized}


def observation_property(observation: Mapping[str, Any]) -> Optional[str]:
    """Resolve the current observation property namespace without guessing."""

    supplied = [
        observation.get(field)
        for field in ("property", "property_type_id", "property_type")
        if observation.get(field) is not None
    ]
    if not supplied:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in supplied):
        return None
    normalized = {value.strip() for value in supplied}
    if len(normalized) != 1:
        return None
    return normalized.pop()


def validate_property_policy(
    definitions: Mapping[str, Any], comparators: Mapping[str, Any]
) -> None:
    """Validate the compact, persisted property-policy vocabulary."""

    if not isinstance(definitions, Mapping):
        raise ValueError("property_definitions must be an object")
    if not isinstance(comparators, Mapping):
        raise ValueError("property_comparators must be an object")
    for name, raw in definitions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("property definition names must be non-empty strings")
        if not isinstance(raw, Mapping):
            raise ValueError(f"property_definitions.{name} must be an object")
        extra = set(raw) - _PROPERTY_DEFINITION_FIELDS
        if extra:
            raise ValueError(
                f"property_definitions.{name} has unsupported fields: "
                + ", ".join(sorted(extra))
            )
        value_type = raw.get("type")
        if value_type not in _PROPERTY_TYPES:
            raise ValueError(
                f"property_definitions.{name}.type must be one of "
                + ", ".join(sorted(_PROPERTY_TYPES))
            )
        schema_version = raw.get("schema_version", "current")
        if not isinstance(schema_version, str) or not schema_version:
            raise ValueError(
                f"property_definitions.{name}.schema_version must be a non-empty string"
            )
        if "unit" in raw and raw["unit"] is not None and (
            not isinstance(raw["unit"], str) or not raw["unit"]
        ):
            raise ValueError(
                f"property_definitions.{name}.unit must be null or a non-empty string"
            )
        normalizer = raw.get("normalizer", "identity")
        if normalizer not in {"identity", "casefold"}:
            raise ValueError(
                f"property_definitions.{name}.normalizer must be identity or casefold"
            )
        if normalizer == "casefold" and value_type not in {"category", "text"}:
            raise ValueError(
                f"property_definitions.{name}.normalizer casefold requires a string type"
            )
        allowed = raw.get("allowed_values")
        if allowed is not None:
            if value_type != "category" or not isinstance(allowed, list) or not allowed:
                raise ValueError(
                    f"property_definitions.{name}.allowed_values requires a non-empty category list"
                )
            if any(not isinstance(value, str) for value in allowed):
                raise ValueError(
                    f"property_definitions.{name}.allowed_values must contain strings"
                )
            normalized_allowed = [
                value.casefold() if normalizer == "casefold" else value
                for value in allowed
            ]
            if len(normalized_allowed) != len(set(normalized_allowed)):
                raise ValueError(
                    f"property_definitions.{name}.allowed_values contains duplicates"
                )
        context_fields = raw.get("context_fields", [])
        if not isinstance(context_fields, list) or any(
            not _valid_context_path(value) for value in context_fields
        ):
            raise ValueError(
                f"property_definitions.{name}.context_fields contains an invalid path"
            )
        if len(context_fields) != len(set(context_fields)):
            raise ValueError(
                f"property_definitions.{name}.context_fields contains duplicates"
            )

    for name, raw in comparators.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("property comparator names must be non-empty strings")
        if definitions and name not in definitions:
            raise ValueError(
                f"property_comparators.{name} has no matching property definition"
            )
        if not isinstance(raw, Mapping):
            raise ValueError(f"property_comparators.{name} must be an object")
        extra = set(raw) - _PROPERTY_COMPARATOR_FIELDS
        if extra:
            raise ValueError(
                f"property_comparators.{name} has unsupported fields: "
                + ", ".join(sorted(extra))
            )
        definition = definitions.get(name, {})
        kind = _comparator_kind(raw, definition)
        if kind not in {"exact", "numeric", "categorical"}:
            raise ValueError(
                f"property_comparators.{name}.kind must be exact, numeric, or categorical"
            )
        if kind == "numeric" and definition.get("type") not in {
            None,
            "integer",
            "real",
        }:
            raise ValueError(
                f"property_comparators.{name} numeric comparison requires a numeric property"
            )
        if kind == "categorical" and definition.get("type") not in {
            None,
            "category",
            "boolean",
        }:
            raise ValueError(
                f"property_comparators.{name} categorical comparison requires a categorical property"
            )
        for field in ("absolute_tolerance", "relative_tolerance"):
            value = raw.get(field, 0.0)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"property_comparators.{name}.{field} must be a finite non-negative number"
                )
            if kind != "numeric" and field in raw:
                raise ValueError(
                    f"property_comparators.{name}.{field} requires kind numeric"
                )
        if "use_uncertainty" in raw and not isinstance(raw["use_uncertainty"], bool):
            raise ValueError(
                f"property_comparators.{name}.use_uncertainty must be Boolean"
            )
        if "use_uncertainty" in raw and kind != "numeric":
            raise ValueError(
                f"property_comparators.{name}.use_uncertainty requires kind numeric"
            )
        conversions = raw.get("unit_conversions", {})
        if not isinstance(conversions, Mapping):
            raise ValueError(
                f"property_comparators.{name}.unit_conversions must be an object"
            )
        if conversions and (
            "unit" not in definition or not isinstance(definition.get("unit"), str)
        ):
            raise ValueError(
                f"property_comparators.{name}.unit_conversions requires a canonical property unit"
            )
        for unit, conversion in conversions.items():
            if not isinstance(unit, str) or not unit:
                raise ValueError(
                    f"property_comparators.{name}.unit_conversions has an invalid unit"
                )
            _conversion_parameters(conversion, f"property_comparators.{name}")


def validate_observation(
    observation: Mapping[str, Any],
    definitions: Mapping[str, Any],
    comparators: Mapping[str, Any],
) -> None:
    """Validate one source observation against declared dataset policy."""

    name = observation_property(observation)
    if name is None:
        raise ValueError("observation property must be one unambiguous non-empty string")
    if "value" not in observation:
        raise ValueError(f"observation {name!r} requires value")
    if definitions and name not in definitions:
        raise ValueError(f"observation property {name!r} is not declared")
    for field in ("conditions", "method"):
        if field in observation and not isinstance(observation[field], Mapping):
            raise ValueError(f"observation.{field} must be an object")
    if "participants" in observation and not isinstance(observation["participants"], list):
        raise ValueError("observation.participants must be an array")
    if "unit" in observation and observation["unit"] is not None and (
        not isinstance(observation["unit"], str) or not observation["unit"]
    ):
        raise ValueError("observation.unit must be null or a non-empty string")

    definition = definitions.get(name)
    if definition is None:
        return
    expected_schema = definition.get("schema_version", "current")
    actual_schema = observation.get("schema_version", "current")
    if actual_schema != expected_schema:
        raise ValueError(
            f"observation {name!r} schema_version must be {expected_schema!r}"
        )
    if not _value_matches_type(observation["value"], definition["type"]):
        raise ValueError(
            f"observation {name!r} value must have type {definition['type']}"
        )
    normalized_value = _normalize_property_value(observation["value"], definition)
    if "allowed_values" in definition:
        allowed = {
            _normalize_property_value(value, definition)
            for value in definition["allowed_values"]
        }
        if normalized_value not in allowed:
            raise ValueError(
                f"observation {name!r} value is outside allowed_values"
            )
    if "unit" in definition:
        canonical_unit = definition["unit"]
        actual_unit = observation.get("unit")
        conversions = comparators.get(name, {}).get("unit_conversions", {})
        if canonical_unit is None and actual_unit is not None:
            raise ValueError(f"observation {name!r} must be unitless")
        if canonical_unit is not None and actual_unit not in {
            canonical_unit,
            *conversions.keys(),
        }:
            raise ValueError(
                f"observation {name!r} unit is not canonical or convertible"
            )


def _valid_context_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or value.startswith(".") or value.endswith("."):
        return False
    parts = value.split(".")
    if parts[0] not in _CONTEXT_ROOTS or any(not part for part in parts):
        return False
    return len(parts) == 1 or parts[0] in {"conditions", "method"}


def _value_matches_type(value: Any, value_type: str) -> bool:
    if value_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == "real":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if value_type in {"category", "text"}:
        return isinstance(value, str)
    if value_type == "boolean":
        return isinstance(value, bool)
    return True


def _normalize_property_value(value: Any, definition: Mapping[str, Any]) -> Any:
    if definition.get("normalizer") == "casefold" and isinstance(value, str):
        return value.strip().casefold()
    return value


def _comparator_kind(
    comparator: Mapping[str, Any], definition: Mapping[str, Any]
) -> str:
    declared = comparator.get("kind")
    if declared is not None:
        return str(declared)
    if "absolute_tolerance" in comparator or "relative_tolerance" in comparator:
        return "numeric"
    if definition.get("type") in {"integer", "real"}:
        return "numeric"
    if definition.get("type") in {"category", "boolean"}:
        return "categorical"
    return "exact"


def _conversion_parameters(value: Any, prefix: str) -> Tuple[float, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        scale, offset = float(value), 0.0
    elif isinstance(value, Mapping) and set(value) <= {"scale", "offset"}:
        scale = value.get("scale", 1.0)
        offset = value.get("offset", 0.0)
        if any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in (scale, offset)
        ):
            raise ValueError(f"{prefix}.unit_conversions values must be numeric")
        scale, offset = float(scale), float(offset)
    else:
        raise ValueError(
            f"{prefix}.unit_conversions values require scale and optional offset"
        )
    if not math.isfinite(scale) or not math.isfinite(offset) or scale == 0:
        raise ValueError(
            f"{prefix}.unit_conversions values require finite non-zero scale"
        )
    return scale, offset


def _active(rows: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [
        deepcopy(dict(row))
        for row in rows
        if row.get("status", "active") == "active"
    ]


def _reason(code: str, message: str, **details: Any) -> Dict[str, Any]:
    result = {"code": code, "message": message}
    if details:
        result["details"] = details
    return result


def _target_id(row: Mapping[str, Any], kind: str, offset: int) -> str:
    return str(row.get(f"{kind}_id") or f"{kind}:{offset}")


@lru_cache(maxsize=128)
def _definition_from_json(encoded: str) -> _CheckerDefinition:
    return _CheckerDefinition.from_info(json.loads(encoded))


def _definition(value: Any) -> _CheckerDefinition:
    return _definition_from_json(canonical_json(value))


def _configured_checks(
    configured: Mapping[str, Any],
    builtin_results: Mapping[str, Tuple[bool, str]],
    case: Mapping[str, Any],
    checker_context: Mapping[str, Any],
    isolated_runner: Optional[Any],
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for name, value in sorted(configured.items()):
        checker = _definition(value)
        if checker.kind == "builtin" and checker.enabled is False:
            results[name] = {
                "kind": "builtin",
                "status": "disabled",
                "passed": False,
                "reason": "disabled_by_policy",
            }
            continue
        try:
            passed, explanation = evaluate_configured_check(
                checker,
                case,
                checker_context,
                builtin_result=builtin_results.get(
                    name, (False, "builtin_check_not_available")
                ),
                isolated_runner=isolated_runner,
            )
        except CheckerIsolationUnavailableError:
            raise
        except Exception as exc:
            results[name] = {
                "kind": checker.kind,
                "status": "checker_error",
                "passed": False,
                "reason": "checker_error",
                "error_type": type(exc).__name__,
            }
            continue
        results[name] = {
            "kind": checker.kind,
            "status": "passed" if passed else "failed",
            "passed": passed,
            "reason": explanation,
        }
    return results


def _highest_level(
    levels: Sequence[str],
    requirements: Mapping[str, Sequence[str]],
    checks: Mapping[str, Mapping[str, Any]],
) -> str:
    achieved = levels[0]
    for level in levels:
        required = requirements.get(level, ())
        if all(checks.get(name, {}).get("passed") is True for name in required):
            achieved = level
        else:
            break
    return achieved


def _verification_is_valid(
    verification: Mapping[str, Any], evidence: Mapping[str, Any]
) -> bool:
    if verification.get("method") == "Human":
        return bool(verification.get("verifier"))
    return verification.get("method") == "AI" and all(
        verification.get(field)
        for field in ("verifier", "procedure_version", "verified_at")
    ) and bool(
        evidence.get("locator")
        or evidence.get("quote")
        or evidence.get("matched_text")
    )


def _identified_primary_source(evidence: Mapping[str, Any]) -> bool:
    return bool(
        evidence.get("doi")
        or evidence.get("scientific_work_id")
        or str(evidence.get("url", "")).startswith(("http://", "https://"))
    )


def _is_supporting_evidence(evidence: Mapping[str, Any]) -> bool:
    """Return whether an evidence link is explicit affirmative support."""

    return evidence.get("relation") == "supports"


def _evidence_targets(
    case: Mapping[str, Any],
) -> tuple[list[Tuple[str, str, Dict[str, Any]]], Dict[str, list[Tuple[str, str]]]]:
    """Return canonical targets and their unambiguous public references."""

    targets: list[Tuple[str, str, Dict[str, Any]]] = []
    aliases: Dict[str, list[Tuple[str, str]]] = defaultdict(list)
    for kind, name in (
        ("observation", "observations"),
        ("claim", "claims"),
    ):
        for offset, row in enumerate(_active(case.get(name, []))):
            target_id = _target_id(row, kind, offset)
            targets.append((target_id, kind, row))
            aliases[target_id].append((target_id, kind))
            property_name = observation_property(row)
            label = row.get("label")
            for alias in dict.fromkeys(
                str(value).strip()
                for value in (property_name, label)
                if isinstance(value, str) and value.strip()
            ):
                aliases[alias].append((target_id, kind))
    return targets, aliases


def _resolve_evidence_target(
    reference: Any,
    aliases: Mapping[str, Sequence[Tuple[str, str]]],
) -> list[Tuple[str, str]]:
    target = str(reference or "")
    candidates = list(dict.fromkeys(aliases.get(target, ())))
    exact = [candidate for candidate in candidates if candidate[0] == target]
    return exact if exact else candidates


def _canonical_work_id(evidence: Mapping[str, Any]) -> Optional[str]:
    """Normalize a scientific-work identity for independence comparison."""

    doi = evidence.get("doi")
    if doi:
        value = str(doi).strip().casefold()
    else:
        value = str(evidence.get("scientific_work_id") or "").strip().casefold()
    for prefix in (
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "https://doi.org/",
        "http://doi.org/",
        "doi:",
    ):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
            return f"doi:{value}" if value else None
    if value.startswith("10.") and "/" in value:
        return f"doi:{value}"
    return value or None


def _lineage_graph(
    evidence: Sequence[Mapping[str, Any]],
) -> Dict[str, set[str]]:
    graph: Dict[str, set[str]] = defaultdict(set)
    for item in evidence:
        lineage = item.get("source_lineage_id")
        if not isinstance(lineage, str) or not lineage.strip():
            continue
        lineage_id = lineage.strip().casefold()
        graph.setdefault(lineage_id, set())
        parents = item.get("derived_from", [])
        if isinstance(parents, Sequence) and not isinstance(parents, str):
            graph[lineage_id].update(
                str(parent).strip().casefold()
                for parent in parents
                if isinstance(parent, str) and parent.strip()
            )
    return graph


def _lineage_roots(
    lineage: Any,
    graph: Mapping[str, set[str]],
) -> set[str]:
    if not isinstance(lineage, str) or not lineage.strip():
        return set()
    start = lineage.strip().casefold()

    def visit(node: str, trail: frozenset[str]) -> set[str]:
        if node in trail or node not in graph:
            return set()
        parents = graph[node]
        if not parents:
            return {node}
        roots: set[str] = set()
        for parent in parents:
            roots.update(visit(parent, trail | {node}))
        return roots

    return visit(start, frozenset())


def _relationship_integrity_reasons(
    case: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Resolve operative Case-local relationships without discarding bad facts.

    Invalid relations remain archivable source facts, but their Case becomes
    incomplete and therefore cannot bypass integrity through direct benchmark
    admission.
    """

    reasons: list[Dict[str, Any]] = []
    collections = {
        name: list(case.get(name, []))
        for name in _ARRAY_FIELDS_FOR_RELATIONSHIPS
    }
    ids_by_collection = {}
    for name, rows in collections.items():
        id_field = _RELATIONSHIP_ID_FIELDS[name]
        ids_by_collection[name] = {
            str(item.get(id_field)): item
            for item in rows
            if item.get(id_field) not in (None, "")
        }
    active_ids = {
        name: {
            record_id
            for record_id, item in records.items()
            if item.get("status", "active") == "active"
        }
        for name, records in ids_by_collection.items()
    }

    _targets, target_aliases = _evidence_targets(case)
    lineage_graph = _lineage_graph(collections["evidence"])

    for item in _active(collections["evidence"]):
        evidence_id = str(item.get("evidence_id"))
        target = str(item.get("target") or "")
        resolved = _resolve_evidence_target(target, target_aliases)
        if not target or not resolved:
            reasons.append(
                _reason(
                    "evidence_target_unresolved",
                    "Active evidence does not resolve to an active observation or claim.",
                    evidence_id=evidence_id,
                    target=target or None,
                )
            )
        elif len(resolved) != 1:
            reasons.append(
                _reason(
                    "evidence_target_ambiguous",
                    "Active evidence resolves to more than one Case target.",
                    evidence_id=evidence_id,
                    target=target,
                )
            )
        target_kind = item.get("target_kind")
        if target_kind is not None and (
            target_kind not in {"observation", "claim"}
            or len(resolved) != 1
            or resolved[0][1] != target_kind
        ):
            reasons.append(
                _reason(
                    "evidence_target_kind_mismatch",
                    "Evidence target_kind does not match the resolved target.",
                    evidence_id=evidence_id,
                    target=target or None,
                    target_kind=target_kind,
                )
            )
        relation = item.get("relation")
        if relation not in {"supports", "contradicts", "context"}:
            reasons.append(
                _reason(
                    "evidence_relation_invalid",
                    "Evidence relation is outside the closed relation vocabulary.",
                    evidence_id=evidence_id,
                    relation=relation,
                )
            )

        lineage = item.get("source_lineage_id")
        parents = item.get("derived_from", [])
        if isinstance(parents, str) or not isinstance(parents, Sequence):
            reasons.append(
                _reason(
                    "evidence_lineage_invalid",
                    "Evidence derived_from must be an array of result-lineage identifiers.",
                    evidence_id=evidence_id,
                )
            )
        else:
            for parent in parents:
                if not isinstance(parent, str) or not parent.strip():
                    reasons.append(
                        _reason(
                            "evidence_lineage_invalid",
                            "Each derived_from value must be a non-empty result-lineage identifier.",
                            evidence_id=evidence_id,
                        )
                    )
                    continue
                parent_id = str(parent).strip().casefold()
                if (
                    not parent_id
                    or (
                        isinstance(lineage, str)
                        and parent_id == lineage.strip().casefold()
                    )
                    or parent_id not in lineage_graph
                ):
                    reasons.append(
                        _reason(
                            "evidence_lineage_unresolved",
                            "Evidence lineage does not resolve to another declared result lineage.",
                            evidence_id=evidence_id,
                            derived_from=parent_id or None,
                        )
                    )
        if isinstance(lineage, str) and lineage.strip() and not _lineage_roots(
            lineage, lineage_graph
        ):
            reasons.append(
                _reason(
                    "evidence_lineage_cycle",
                    "Evidence result-lineage ancestry is cyclic or unresolved.",
                    evidence_id=evidence_id,
                    source_lineage_id=lineage,
                )
            )

    for item in _active(collections["verifications"]):
        verification_id = str(item.get("verification_id"))
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id not in active_ids["evidence"]:
            reasons.append(
                _reason(
                    "verification_evidence_unresolved",
                    "Active verification does not resolve to active evidence.",
                    verification_id=verification_id,
                    evidence_id=evidence_id or None,
                )
            )

    for name, records in ids_by_collection.items():
        id_field = _RELATIONSHIP_ID_FIELDS[name]
        for record_id, item in records.items():
            for relation, reciprocal in (
                ("supersedes", "superseded_by"),
                ("superseded_by", "supersedes"),
            ):
                raw_targets = item.get(relation)
                if raw_targets is None:
                    continue
                targets = (
                    list(raw_targets)
                    if isinstance(raw_targets, list)
                    else [raw_targets]
                )
                if not targets:
                    reasons.append(
                        _reason(
                            "supersession_reference_unresolved",
                            "A supersession link cannot be an empty array.",
                            collection=name,
                            record_id=record_id,
                            relation=relation,
                        )
                    )
                for target in targets:
                    target_id = str(target)
                    referenced = records.get(target_id)
                    if target_id == record_id or referenced is None:
                        reasons.append(
                            _reason(
                                "supersession_reference_unresolved",
                                "A supersession link does not resolve within its fact collection.",
                                collection=name,
                                record_id=record_id,
                                relation=relation,
                                target=target_id,
                            )
                        )
                        continue
                    reciprocal_value = referenced.get(reciprocal)
                    reciprocal_targets = (
                        list(reciprocal_value)
                        if isinstance(reciprocal_value, list)
                        else [reciprocal_value]
                    )
                    if record_id not in {
                        str(value)
                        for value in reciprocal_targets
                        if value is not None
                    }:
                        reasons.append(
                            _reason(
                                "supersession_relation_inconsistent",
                                "A supersession link lacks its reciprocal relation.",
                                collection=name,
                                record_id=record_id,
                                relation=relation,
                                target=target_id,
                                id_field=id_field,
                            )
                        )

    entity_ids = active_ids["entities"]

    def entity_reference(
        value: Any, *, collection: str, record_id: str, field: str
    ) -> None:
        if value not in (None, "") and str(value) not in entity_ids:
            reasons.append(
                _reason(
                    "entity_reference_unresolved",
                    "An operative entity link does not resolve within the Case.",
                    collection=collection,
                    record_id=record_id,
                    field=field,
                    entity_id=str(value),
                )
            )

    for item in _active(collections["entities"]):
        record_id = str(item.get("entity_id"))
        entity_reference(
            item.get("parent_entity_id"),
            collection="entities",
            record_id=record_id,
            field="parent_entity_id",
        )
        relations = item.get("relations", [])
        if not isinstance(relations, list):
            reasons.append(
                _reason(
                    "entity_relation_invalid",
                    "Entity relations must be an array.",
                    entity_id=record_id,
                )
            )
            continue
        for relation in relations:
            if not isinstance(relation, Mapping):
                reasons.append(
                    _reason(
                        "entity_relation_invalid",
                        "Each entity relation must be an object.",
                        entity_id=record_id,
                    )
                )
                continue
            if not isinstance(relation.get("predicate"), str) or not relation[
                "predicate"
            ].strip():
                reasons.append(
                    _reason(
                        "entity_relation_invalid",
                        "Each entity relation requires a non-empty predicate.",
                        entity_id=record_id,
                    )
                )
            if relation.get("subject_entity_id") not in (None, record_id):
                reasons.append(
                    _reason(
                        "entity_relation_invalid",
                        "An entity-owned relation must name that entity as its subject.",
                        entity_id=record_id,
                        subject_entity_id=relation.get("subject_entity_id"),
                    )
                )
            for field in ("subject_entity_id", "object_entity_id"):
                if relation.get(field) in (None, ""):
                    reasons.append(
                        _reason(
                            "entity_relation_invalid",
                            "Each entity relation requires both entity endpoints.",
                            entity_id=record_id,
                            field=field,
                        )
                    )
                    continue
                entity_reference(
                    relation.get(field),
                    collection="entities",
                    record_id=record_id,
                    field=f"relations.{field}",
                )

    for name in ("structures", "observations", "claims"):
        id_field = _RELATIONSHIP_ID_FIELDS[name]
        for item in _active(collections[name]):
            record_id = str(item.get(id_field))
            entity_reference(
                item.get("entity_id") or item.get("subject_entity_id"),
                collection=name,
                record_id=record_id,
                field=(
                    "entity_id" if item.get("entity_id") is not None else "subject_entity_id"
                ),
            )
            participants = item.get("participants", [])
            if isinstance(participants, list):
                for participant in participants:
                    if isinstance(participant, Mapping):
                        entity_reference(
                            participant.get("entity_id"),
                            collection=name,
                            record_id=record_id,
                            field="participants.entity_id",
                        )
    return reasons


_ARRAY_FIELDS_FOR_RELATIONSHIPS = (
    "entities",
    "structures",
    "observations",
    "claims",
    "evidence",
    "verifications",
    "attestations",
)
_RELATIONSHIP_ID_FIELDS = {
    "entities": "entity_id",
    "structures": "structure_id",
    "observations": "observation_id",
    "claims": "claim_id",
    "evidence": "evidence_id",
    "verifications": "verification_id",
    "attestations": "attestation_id",
}


def assess_evidence(
    case: Mapping[str, Any],
    required_targets: Optional[Iterable[str]] = None,
    *,
    policy: Optional[Mapping[str, Any]] = None,
    checker_context: Optional[Mapping[str, Any]] = None,
    isolated_runner: Optional[Any] = None,
) -> Dict[str, Any]:
    """Assess active observation/claim support and its conservative Case level."""

    configured_policy = dict(policy or {})
    configured_checks = configured_policy.get("checks", _DEFAULT_EVIDENCE_CHECKS)
    level_checks = configured_policy.get("level_checks", _DEFAULT_EVIDENCE_LEVELS)

    targets, target_aliases = _evidence_targets(case)
    evidence = _active(case.get("evidence", []))
    verifications = _active(case.get("verifications", []))
    verification_by_evidence: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for verification in verifications:
        verification_by_evidence[str(verification.get("evidence_id"))].append(
            verification
        )

    levels: Dict[str, str] = {}
    target_checks: Dict[str, Dict[str, Any]] = {}
    reasons: list[Dict[str, Any]] = []
    lineage_graph = _lineage_graph(evidence)
    for target_id, _target_kind, _target in targets:
        linked = [
            item
            for item in evidence
            if _resolve_evidence_target(item.get("target"), target_aliases)
            == [(target_id, _target_kind)]
            and _is_supporting_evidence(item)
        ]
        explicitly_located = [
            item
            for item in linked
            if item.get("locator") or item.get("quote") or item.get("matched_text")
        ]
        verified = []
        for item in explicitly_located:
            checks = verification_by_evidence.get(str(item.get("evidence_id")), [])
            if _identified_primary_source(item) and any(
                _verification_is_valid(check, item) for check in checks
            ):
                verified.append(item)

        independent_chains: list[Tuple[str, set[str]]] = []
        for item in verified:
            work = _canonical_work_id(item)
            roots = _lineage_roots(item.get("source_lineage_id"), lineage_graph)
            if work and roots:
                independent_chains.append((work, roots))
        independently_corroborated = any(
            left_work != right_work and left_roots.isdisjoint(right_roots)
            for index, (left_work, left_roots) in enumerate(independent_chains)
            for right_work, right_roots in independent_chains[index + 1 :]
        )
        if len(verified) >= 2 and not independently_corroborated:
            reasons.append(
                _reason(
                    "evidence_independence_unproven",
                    "Several verified sources exist, but independent result lineage is not established.",
                    target=target_id,
                )
            )
        builtin = {
            "extracted_material": (bool(linked), "linked_material_present"),
            "explicit_report": (
                bool(explicitly_located),
                "explicit_report_present",
            ),
            "usable_locator": (bool(explicitly_located), "usable_locator_present"),
            "primary_source_verified": (
                bool(verified),
                "identified_primary_source_verified",
            ),
            "independent_corroboration": (
                independently_corroborated,
                "independent_primary_results_corroborate",
            ),
        }
        context = deepcopy(dict(checker_context or {}))
        context["evidence_target"] = {
            "target_id": target_id,
            "target": deepcopy(_target),
        }
        evaluated = _configured_checks(
            configured_checks,
            builtin,
            case,
            context,
            isolated_runner,
        )
        target_checks[target_id] = evaluated
        level = _highest_level(E_LEVELS, level_checks, evaluated)
        levels[target_id] = level

    required = list(
        dict.fromkeys(
            str(target_id)
            for target_id in (
                required_targets
                if required_targets is not None
                else (target_id for target_id, _, _ in targets)
            )
        )
    )
    if not required:
        aggregate = None
    else:
        aggregate = min(
            (levels.get(target_id, "E0") for target_id in required),
            key=lambda item: int(item[1]),
        )
        for target_id in required:
            if target_id not in levels:
                reasons.append(
                    _reason(
                        "expected_evidence_target_missing",
                        "A required evidence-bearing target is absent.",
                        target=target_id,
                    )
                )
    return {
        "levels": levels,
        "aggregate": aggregate,
        "checks": target_checks,
        "reasons": reasons,
    }


def _identity_path_value(value: Mapping[str, Any], path: str) -> Tuple[bool, Any]:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    if current is None or current == "" or current == [] or current == {}:
        return False, None
    return True, current


def _identity_component_value(
    case: Mapping[str, Any], component: Mapping[str, Any]
) -> Tuple[str, Any]:
    structure_role = component.get("structure_role")
    source: Mapping[str, Any] = case
    if structure_role is not None:
        matches = []
        for structure in _active(case.get("structures", [])):
            roles = {structure.get("role"), *structure.get("roles", [])}
            if structure_role in roles:
                matches.append(structure)
        if not matches:
            return "missing", None
        if len(matches) != 1:
            return "ambiguous", None
        source = matches[0]
    present, value = _identity_path_value(source, str(component["path"]))
    if not present:
        return "missing", None
    normalizer = component["normalizer"]
    try:
        if normalizer == "text":
            if not isinstance(value, str) or not value.strip():
                return "invalid", None
            return "present", value.strip()
        if normalizer == "casefold":
            if not isinstance(value, str) or not value.strip():
                return "invalid", None
            return "present", value.strip().casefold()
        if normalizer == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return "invalid", None
            number = Decimal(str(value))
            if not number.is_finite():
                return "invalid", None
            normalized = number.normalize()
            return "present", format(normalized, "f")
        if normalizer == "boolean":
            return ("present", value) if type(value) is bool else ("invalid", None)
        if normalizer == "set":
            if not isinstance(value, list):
                return "invalid", None
            unique = {canonical_json(item): deepcopy(item) for item in value}
            return "present", [unique[key] for key in sorted(unique)]
        canonical_json(value)
        return "present", deepcopy(value)
    except (InvalidOperation, TypeError, ValueError):
        return "invalid", None


def _identity_components(
    case: Mapping[str, Any], identity_schema: Mapping[str, Any]
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    values: Dict[str, Any] = {}
    reasons = []
    for component in identity_schema["components"]:
        state, value = _identity_component_value(case, component)
        key = component["key"]
        if state == "present":
            values[key] = value
            continue
        if state != "missing" or component["required"]:
            reasons.append(
                _reason(
                    "identity_indeterminate",
                    "An identity component is missing, ambiguous, or invalid.",
                    component=key,
                    state=state,
                )
            )
    return values, reasons


def identity_key(
    case: Mapping[str, Any], identity_schema: Mapping[str, Any]
) -> Tuple[Optional[str], list[Dict[str, Any]]]:
    """Return a deterministic cross-Case equivalence key when proven."""

    schema = validate_identity_schema(identity_schema)
    if not schema["components"]:
        return f"case:{case['id']}", []
    normalized, reasons = _identity_components(case, schema)
    if reasons:
        return None, reasons
    if not normalized:
        return None, [
            _reason(
                "identity_indeterminate",
                "No persisted identity component produced a comparable value.",
            )
        ]
    ordered = [[item["key"], normalized[item["key"]]] for item in schema["components"] if item["key"] in normalized]
    return canonical_digest(ordered), []


def sample_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    identity_schema: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return ``same``, ``different``, or ``indeterminate`` under policy."""

    if str(left["id"]) == str(right["id"]):
        return {"outcome": "same", "reasons": []}
    schema = validate_identity_schema(identity_schema)
    if not schema["components"]:
        return {
            "outcome": "indeterminate",
            "reasons": [
                _reason(
                    "identity_indeterminate",
                    "No cross-Case identity components are configured.",
                )
            ],
        }
    left_values, left_reasons = _identity_components(left, schema)
    right_values, right_reasons = _identity_components(right, schema)
    reasons = left_reasons + right_reasons
    if reasons:
        return {"outcome": "indeterminate", "reasons": reasons}
    compared = False
    for component in schema["components"]:
        key = component["key"]
        if key not in left_values or key not in right_values:
            continue
        compared = True
        if canonical_json(left_values[key]) == canonical_json(right_values[key]):
            continue
        if component["mismatch_is_decisive"]:
            return {
                "outcome": "different",
                "reasons": [
                    _reason(
                        "identity_component_mismatch",
                        "A decisive identity component differs.",
                        component=key,
                    )
                ],
            }
        return {
            "outcome": "indeterminate",
            "reasons": [
                _reason(
                    "identity_indeterminate",
                    "A non-decisive identity component differs.",
                    component=key,
                )
            ],
        }
    return {
        "outcome": "same" if compared else "indeterminate",
        "reasons": []
        if compared
        else [
            _reason(
                "identity_indeterminate",
                "No comparable cross-Case identity component is present.",
            )
        ],
    }


def observation_context(
    observation: Mapping[str, Any],
    definition: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the policy-relevant context without the observed value."""

    fields = (
        tuple(definition["context_fields"])
        if definition is not None and "context_fields" in definition
        else _DEFAULT_CONTEXT_FIELDS
    )
    context = {
        "property": observation_property(observation),
        "schema_version": observation.get("schema_version", "current"),
        "unit": observation.get("unit"),
    }
    for field in fields:
        present, value = _context_value(observation, field)
        context[field] = _normalize_context_value(value) if present else None
    return context


def observation_with_resolved_entities(
    case: Mapping[str, Any], observation: Mapping[str, Any]
) -> Dict[str, Any]:
    """Resolve Case-local entity IDs into stable comparison descriptors."""

    resolved = deepcopy(dict(observation))
    candidates: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for structure in _active(case.get("structures", [])):
        entity_id = structure.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            continue
        roles = sorted(
            {
                role
                for role in (structure.get("role"), *structure.get("roles", []))
                if isinstance(role, str) and role
            }
        )
        if roles:
            candidates[entity_id].append({"kind": "structure", "roles": roles})
    for entity in _active(case.get("entities", [])):
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            continue
        identifiers = sorted(
            (
                {
                    "namespace": str(item["namespace"]),
                    "value": str(item["value"]),
                }
                for item in entity.get("external_identifiers", [])
                if isinstance(item, Mapping)
                and item.get("namespace") not in (None, "")
                and item.get("value") not in (None, "")
            ),
            key=canonical_json,
        )
        if identifiers:
            candidates[entity_id] = [
                {"kind": "external_identifiers", "values": identifiers}
            ]
            continue
        roles = sorted(
            {
                role
                for role in (entity.get("role"), *entity.get("roles", []))
                if isinstance(role, str) and role
            }
        )
        if roles and not candidates.get(entity_id):
            candidates[entity_id].append(
                {
                    "kind": "case_role",
                    "entity_kind": entity.get("entity_kind"),
                    "roles": roles,
                }
            )

    descriptor_counts = Counter(
        canonical_json(descriptor)
        for values in candidates.values()
        for descriptor in values
    )

    def resolve(reference: Any) -> Any:
        if not isinstance(reference, str) or not reference:
            return {
                "_hotpot_entity_identity": "indeterminate",
                "reason": "invalid_entity_reference",
            }
        descriptors = candidates.get(reference, [])
        if len(descriptors) != 1:
            return {
                "_hotpot_entity_identity": "indeterminate",
                "reason": "entity_reference_unresolved_or_ambiguous",
            }
        descriptor = descriptors[0]
        if descriptor_counts[canonical_json(descriptor)] != 1:
            return {
                "_hotpot_entity_identity": "indeterminate",
                "reason": "entity_role_ambiguous",
            }
        return descriptor

    if "subject" not in resolved and "subject_entity_id" in resolved:
        resolved["subject"] = resolve(resolved["subject_entity_id"])
    if isinstance(resolved.get("site"), str):
        resolved["site"] = resolve(resolved["site"])
    participants = resolved.get("participants")
    if isinstance(participants, list):
        normalized_participants = []
        for participant in participants:
            item = deepcopy(participant)
            if isinstance(item, dict) and "entity_id" in item:
                item["entity_id"] = resolve(item["entity_id"])
            normalized_participants.append(item)
        resolved["participants"] = normalized_participants
    return resolved


def compare_observations(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    tolerances: Mapping[str, Mapping[str, float]],
    *,
    property_definitions: Optional[Mapping[str, Any]] = None,
    property_comparators: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare one same-sample pair under persisted property policy."""

    definitions = dict(property_definitions or {})
    comparators = dict(property_comparators or {})
    left_property = observation_property(left)
    right_property = observation_property(right)
    if left_property is None or right_property is None:
        return {
            "outcome": "indeterminate",
            "reason": "observation_property_indeterminate",
        }
    if left_property != right_property:
        return {"outcome": "noncandidate", "reason": "different_property"}
    definition = definitions.get(left_property, {})
    comparator, comparator_source = _effective_comparator(
        left_property,
        definition,
        comparators,
        tolerances,
    )
    comparator_digest = canonical_digest(
        {
            "property": left_property,
            "definition": definition,
            "comparator": comparator,
            "source": comparator_source,
        }
    )
    left_context = observation_context(left, definition)
    right_context = observation_context(right, definition)
    expected_schema = definition.get("schema_version", "current")
    if left_context["schema_version"] != right_context["schema_version"]:
        return {
            "outcome": "noncomparable",
            "reason": "property_schema_noncomparable",
            "comparison_context": {"left": left_context, "right": right_context},
            "comparator_digest": comparator_digest,
        }
    if definition and left_context["schema_version"] != expected_schema:
        return {
            "outcome": "indeterminate",
            "reason": "property_schema_indeterminate",
            "comparison_context": {"left": left_context, "right": right_context},
            "comparator_digest": comparator_digest,
        }

    context_result = _compare_context(left, right, definition)
    if context_result is not None:
        context_result["comparator_digest"] = comparator_digest
        return context_result

    unit_error, values = _comparison_values(
        left, right, definition, comparator or {}
    )
    if unit_error is not None:
        outcome, reason = unit_error
        return {
            "outcome": outcome,
            "reason": reason,
            "comparison_context": {"left": left_context, "right": right_context},
            "comparator_digest": comparator_digest,
        }
    left_value, right_value, left_uncertainty, right_uncertainty = values
    comparable_context = {"left": left_context, "right": right_context}

    if comparator is None:
        if canonical_json(left_value) == canonical_json(right_value):
            return {
                "outcome": "consistent",
                "reason": "value_equal",
                "comparison_context": comparable_context,
                "comparator_digest": comparator_digest,
            }
        return {
            "outcome": "indeterminate",
            "reason": "property_comparator_not_configured",
            "comparison_context": comparable_context,
            "comparator_digest": comparator_digest,
        }

    kind = _comparator_kind(comparator, definition)
    if kind in {"exact", "categorical"}:
        left_value = _normalize_property_value(left_value, definition)
        right_value = _normalize_property_value(right_value, definition)
        equal = canonical_json(left_value) == canonical_json(right_value)
        return {
            "outcome": "consistent" if equal else "conflict",
            "reason": "exact_value_equal" if equal else "scientific_value_conflict",
            "comparison_context": comparable_context,
            "comparator_digest": comparator_digest,
        }

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in (left_value, right_value)
    ):
        return {
            "outcome": "indeterminate",
            "reason": "observation_value_invalid",
            "comparison_context": comparable_context,
            "comparator_digest": comparator_digest,
        }
    absolute = float(comparator.get("absolute_tolerance", 0.0))
    relative = float(comparator.get("relative_tolerance", 0.0))
    allowed = max(
        absolute,
        relative * max(abs(float(left_value)), abs(float(right_value))),
    )
    if comparator.get("use_uncertainty", True):
        allowed = max(allowed, left_uncertainty, right_uncertainty)
    difference = abs(float(left_value) - float(right_value))
    equal = difference <= allowed
    return {
        "outcome": "consistent" if equal else "conflict",
        "reason": "within_numeric_tolerance"
        if equal
        else "scientific_value_conflict",
        "difference": difference,
        "allowed": allowed,
        "comparison_context": comparable_context,
        "comparator_digest": comparator_digest,
    }


def _effective_comparator(
    property_name: str,
    definition: Mapping[str, Any],
    comparators: Mapping[str, Any],
    tolerances: Mapping[str, Mapping[str, float]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    specific = comparators.get(property_name)
    if specific is not None:
        return dict(specific), "property_policy"
    gate_rule = tolerances.get(property_name)
    if gate_rule is not None:
        return (
            {
                "kind": "numeric",
                "absolute_tolerance": gate_rule.get("absolute", 0.0),
                "relative_tolerance": gate_rule.get("relative", 0.0),
                "use_uncertainty": True,
            },
            "benchmark_gate_fallback",
        )
    if definition.get("type") in {"category", "boolean"}:
        return {"kind": "categorical"}, "property_definition_default"
    return None, "unconfigured"


def _context_value(
    observation: Mapping[str, Any], path: str
) -> Tuple[bool, Any]:
    value: Any = observation
    for offset, part in enumerate(path.split(".")):
        if (
            offset == 0
            and part == "subject"
            and isinstance(value, Mapping)
            and "subject" not in value
            and "subject_entity_id" in value
        ):
            value = value["subject_entity_id"]
            continue
        if not isinstance(value, Mapping) or part not in value:
            return False, None
        value = value[part]
    if value in (None, "", [], {}):
        return False, None
    return True, value


def _normalize_context_value(value: Any) -> Any:
    if isinstance(value, list):
        normalized = [_normalize_context_value(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_context_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def _compare_context(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    declared = "context_fields" in definition
    fields = (
        tuple(definition["context_fields"])
        if declared
        else _DEFAULT_CONTEXT_FIELDS
    )
    left_context = observation_context(left, definition)
    right_context = observation_context(right, definition)
    for field in fields:
        left_present, left_value = _context_value(left, field)
        right_present, right_value = _context_value(right, field)
        if declared and (not left_present or not right_present):
            return {
                "outcome": "indeterminate",
                "reason": "observation_context_indeterminate",
                "context_field": field,
                "comparison_context": {
                    "left": left_context,
                    "right": right_context,
                },
            }
        if left_present != right_present:
            return {
                "outcome": "indeterminate",
                "reason": "observation_context_indeterminate",
                "context_field": field,
                "comparison_context": {
                    "left": left_context,
                    "right": right_context,
                },
            }
        if left_present and (
            _contains_indeterminate_entity(left_value)
            or _contains_indeterminate_entity(right_value)
        ):
            return {
                "outcome": "indeterminate",
                "reason": "observation_context_indeterminate",
                "context_field": field,
                "comparison_context": {
                    "left": left_context,
                    "right": right_context,
                },
            }
        if left_present and canonical_json(
            _normalize_context_value(left_value)
        ) != canonical_json(_normalize_context_value(right_value)):
            return {
                "outcome": "noncomparable",
                "reason": "observation_context_noncomparable",
                "context_field": field,
                "comparison_context": {
                    "left": left_context,
                    "right": right_context,
                },
            }
    return None


def _contains_indeterminate_entity(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("_hotpot_entity_identity") == "indeterminate":
            return True
        return any(_contains_indeterminate_entity(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_indeterminate_entity(item) for item in value)
    return False


def _comparison_values(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    definition: Mapping[str, Any],
    comparator: Mapping[str, Any],
) -> Tuple[
    Optional[Tuple[str, str]],
    Optional[Tuple[Any, Any, float, float]],
]:
    left_value = left.get("value")
    right_value = right.get("value")
    left_unit = left.get("unit")
    right_unit = right.get("unit")
    if "unit" not in definition:
        if left_unit != right_unit:
            outcome = (
                "indeterminate"
                if left_unit is None or right_unit is None
                else "noncomparable"
            )
            reason = (
                "observation_unit_indeterminate"
                if outcome == "indeterminate"
                else "observation_unit_noncomparable"
            )
            return (outcome, reason), None
        return None, (
            left_value,
            right_value,
            _uncertainty(left.get("uncertainty")),
            _uncertainty(right.get("uncertainty")),
        )

    canonical_unit = definition["unit"]
    if canonical_unit is None:
        if left_unit is not None or right_unit is not None:
            return ("noncomparable", "observation_unit_noncomparable"), None
        return None, (
            left_value,
            right_value,
            _uncertainty(left.get("uncertainty")),
            _uncertainty(right.get("uncertainty")),
        )
    if left_unit is None or right_unit is None:
        return ("indeterminate", "observation_unit_indeterminate"), None
    conversions = comparator.get("unit_conversions", {})
    try:
        left_scale, left_offset = (
            (1.0, 0.0)
            if left_unit == canonical_unit
            else _conversion_parameters(
                conversions[left_unit], "property comparator"
            )
        )
        right_scale, right_offset = (
            (1.0, 0.0)
            if right_unit == canonical_unit
            else _conversion_parameters(
                conversions[right_unit], "property comparator"
            )
        )
    except (KeyError, ValueError):
        return ("noncomparable", "observation_unit_noncomparable"), None
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (left_value, right_value)
    ):
        if left_unit != canonical_unit or right_unit != canonical_unit:
            return ("indeterminate", "unit_conversion_requires_numeric_value"), None
        return None, (
            left_value,
            right_value,
            _uncertainty(left.get("uncertainty")),
            _uncertainty(right.get("uncertainty")),
        )
    return None, (
        float(left_value) * left_scale + left_offset,
        float(right_value) * right_scale + right_offset,
        _uncertainty(left.get("uncertainty")) * abs(left_scale),
        _uncertainty(right.get("uncertainty")) * abs(right_scale),
    )


def _uncertainty(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return abs(float(value))
    if isinstance(value, Mapping):
        for key in ("absolute", "standard", "value"):
            item = value.get(key)
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                return abs(float(item))
    return 0.0


def find_conflicts(
    cases: Iterable[Mapping[str, Any]],
    identity_schema: Mapping[str, Any],
    tolerances: Mapping[str, Mapping[str, float]],
    *,
    property_definitions: Optional[Mapping[str, Any]] = None,
    property_comparators: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministically compare unique candidates within identity groups."""

    groups: Dict[str, list[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    identity_reasons: Dict[str, list[Dict[str, Any]]] = {}
    for case in cases:
        case_id = str(case["id"])
        key, reasons = identity_key(case, identity_schema)
        identity_reasons[case_id] = reasons
        sample_key = key if key is not None else f"case:{case_id}"
        for offset, observation in enumerate(_active(case.get("observations", []))):
            candidate = observation_with_resolved_entities(case, observation)
            candidate.setdefault(
                "observation_id", f"{case_id}:observation:{offset}"
            )
            groups[sample_key].append((case_id, candidate))

    rows = []
    conflicting_cases = set()
    summaries = []
    for sample_key in sorted(groups):
        candidates = sorted(
            groups[sample_key],
            key=lambda item: (item[0], str(item[1]["observation_id"])),
        )
        pair_results = []
        consistent = 0
        exceptions = 0
        for left_index, (left_case, left) in enumerate(candidates):
            for right_case, right in candidates[left_index + 1 :]:
                result = compare_observations(
                    left,
                    right,
                    tolerances,
                    property_definitions=property_definitions,
                    property_comparators=property_comparators,
                )
                if result["outcome"] == "noncandidate":
                    continue
                pair = {
                    "sample_key": sample_key,
                    "left_case_id": left_case,
                    "right_case_id": right_case,
                    "left_observation_id": left["observation_id"],
                    "right_observation_id": right["observation_id"],
                    **result,
                }
                pair_results.append(pair)
                if result["outcome"] == "consistent":
                    consistent += 1
                else:
                    exceptions += 1
                    rows.append(pair)
                if result["outcome"] == "conflict":
                    conflicting_cases.update((left_case, right_case))
        summaries.append(
            {
                "sample_key": sample_key,
                "candidate_count": len(pair_results),
                "consistent_count": consistent,
                "exception_count": exceptions,
                "ordered_results_digest": canonical_digest(pair_results),
            }
        )
    return {
        "exceptions": rows,
        "groups": summaries,
        "conflicting_cases": sorted(conflicting_cases),
        "identity_reasons": identity_reasons,
    }


def assess_case(
    case: Mapping[str, Any],
    *,
    core_labels: Sequence[str],
    core_info: Sequence[str],
    required_structure_roles: Sequence[str],
    asset_assessments: Sequence[Mapping[str, Any]],
    conflicting: bool,
    benchmark_policy: Mapping[str, Any],
    evidence_policy: Optional[Mapping[str, Any]] = None,
    quality_policy: Optional[Mapping[str, Any]] = None,
    conflict_assessments: Sequence[Mapping[str, Any]] = (),
    identity_reasons: Sequence[Mapping[str, Any]] = (),
    checker_context: Optional[Mapping[str, Any]] = None,
    isolated_runner: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute the sole authoritative CaseAssessment for one generation."""

    reasons: list[Dict[str, Any]] = [
        deepcopy(dict(reason)) for reason in identity_reasons
    ]
    reasons.extend(_relationship_integrity_reasons(case))
    observations = _active(case.get("observations", []))
    active_properties = {
        item.get("property") or item.get("property_type_id") for item in observations
    }
    missing_labels = sorted(set(core_labels) - active_properties)
    missing_info = sorted(
        name
        for name in core_info
        if case.get("core_info", {}).get(name) in (None, "", [], {})
    )
    if missing_labels:
        reasons.append(
            _reason(
                "missing_core_labels",
                "Required core labels are missing.",
                fields=missing_labels,
            )
        )
    if missing_info:
        reasons.append(
            _reason(
                "missing_core_info",
                "Required core information is missing.",
                fields=missing_info,
            )
        )

    structures = _active(case.get("structures", []))
    asset_by_structure: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for assessment in asset_assessments:
        asset_by_structure[str(assessment.get("structure_id"))].append(assessment)

    def has_valid_representation(structure: Mapping[str, Any]) -> bool:
        return any(
            assessment.get("presence") == "present"
            and assessment.get("hash_result") in {"match", "unchecked"}
            and assessment.get("parse_result") == "valid"
            for assessment in asset_by_structure.get(
                str(structure.get("structure_id")), []
            )
        )

    any_structure_required = "*" in required_structure_roles
    if any_structure_required:
        if not structures:
            reasons.append(
                _reason(
                    "missing_required_structure",
                    "At least one structure is required, but none was supplied.",
                    role="*",
                )
            )
        elif not any(has_valid_representation(item) for item in structures):
            reasons.append(
                _reason(
                    "invalid_required_structure",
                    "No supplied structure has a valid representation.",
                    role="*",
                )
            )
    for role in (item for item in required_structure_roles if item != "*"):
        matching = [item for item in structures if item.get("role") == role]
        if not matching:
            reasons.append(
                _reason(
                    "missing_required_structure",
                    "A required structure role is absent.",
                    role=role,
                )
            )
            continue
        valid = any(has_valid_representation(item) for item in matching)
        if not valid:
            reasons.append(
                _reason(
                    "invalid_required_structure",
                    "No valid asset exists for a required structure role.",
                    role=role,
                )
            )
    if not required_structure_roles and not structures:
        reasons.append(
            _reason(
                "structure_not_provided",
                "No structure was supplied; this dataset does not require one.",
            )
        )
    chemistry_checks = dict(quality_policy or {}).get("structure_checks", {})
    if structures and not any(
        isinstance(definition, Mapping) and definition.get("kind") == "custom"
        for definition in chemistry_checks.values()
    ):
        reasons.append(
            _reason(
                "structure_chemistry_not_checked",
                "No chemistry-specific structure checker was configured.",
            )
        )
    if conflicting:
        reasons.append(
            _reason(
                "scientific_value_conflict",
                "Comparable observations for the same sample significantly disagree.",
            )
        )

    required_target_ids = [
        _target_id(item, "observation", offset)
        for offset, item in enumerate(observations)
        if (item.get("property") or item.get("property_type_id")) in core_labels
    ]
    required_target_ids.extend(
        f"core-label:{name}" for name in missing_labels
    )
    evidence = assess_evidence(
        case,
        required_target_ids if core_labels else None,
        policy=evidence_policy,
        checker_context=checker_context,
        isolated_runner=isolated_runner,
    )
    reasons.extend(evidence["reasons"])
    evidence_level = evidence["aggregate"]

    incomplete_codes = {
        "missing_core_labels",
        "missing_core_info",
        "missing_required_structure",
        "invalid_required_structure",
        "scientific_value_conflict",
        "evidence_target_unresolved",
        "evidence_target_ambiguous",
        "evidence_target_kind_mismatch",
        "evidence_relation_invalid",
        "evidence_lineage_invalid",
        "evidence_lineage_unresolved",
        "evidence_lineage_cycle",
        "verification_evidence_unresolved",
        "supersession_reference_unresolved",
        "supersession_relation_inconsistent",
        "entity_reference_unresolved",
        "entity_relation_invalid",
    }
    incomplete = any(item["code"] in incomplete_codes for item in reasons)
    e_index = -1 if evidence_level is None else int(evidence_level[1])
    identity_determinate = not any(
        item["code"] == "identity_indeterminate" for item in reasons
    )
    human_verified = any(
        item.get("method") == "Human"
        for item in _active(case.get("verifications", []))
    )
    configured_quality = dict(quality_policy or {})
    quality_checks = configured_quality.get("checks", _DEFAULT_QUALITY_CHECKS)
    structure_check_config = configured_quality.get(
        "structure_checks",
        _DEFAULT_STRUCTURE_CHECKS,
    )
    declared_structure_ids = {
        str(item.get("structure_id")) for item in structures
    }
    structure_assets = [
        item
        for item in asset_assessments
        if str(item.get("structure_id")) in declared_structure_ids
    ]
    has_structures = bool(structures)
    structure_builtins = {
        "asset_available": (
            not has_structures
            or bool(structure_assets)
            and all(item.get("presence") == "present" for item in structure_assets),
            "declared_structure_assets_are_available",
        ),
        "asset_hash_valid": (
            not has_structures
            or bool(structure_assets)
            and all(
                item.get("hash_result") in {"match", "unchecked"}
                for item in structure_assets
            ),
            "declared_structure_asset_hashes_have_no_mismatch",
        ),
        "asset_parseable": (
            not has_structures
            or all(
                any(
                    item.get("parse_result") == "valid"
                    for item in structure_assets
                    if str(item.get("structure_id"))
                    == str(structure.get("structure_id"))
                )
                for structure in structures
            ),
            "each_declared_structure_has_a_parseable_representation",
        ),
        "declared_role_valid": (
            all(item.get("role") for item in structures)
            and (
                not any_structure_required
                or any(has_valid_representation(item) for item in structures)
            )
            and all(
                any(item.get("role") == role for item in structures)
                for role in required_structure_roles
                if role != "*"
            ),
            "structure_roles_are_declared_and_structure_requirements_are_satisfied",
        ),
    }
    structure_results = _configured_checks(
        structure_check_config,
        structure_builtins,
        case,
        dict(checker_context or {}),
        isolated_runner,
    )
    enabled_structure_results = [
        item for item in structure_results.values() if item["status"] != "disabled"
    ]
    structure_controls_passed = all(
        item["passed"] is True for item in enabled_structure_results
    )
    quality_builtins = {
        "governed_collection": (True, "canonical_bundle_integrity_checked"),
        "provenance_complete": (
            bool(case.get("provenance") or case.get("metadata")),
            "case_provenance_present",
        ),
        "subject_mapping_audited": (
            bool(case.get("identifiers") or structures),
            "case_subject_is_mapped",
        ),
        "case_complete": (not incomplete, "required_case_facts_are_complete"),
        "reviewed": (e_index >= 3, "required_targets_are_verified"),
        "conflict_adjudicated": (
            not conflicting,
            "no_unresolved_scientific_value_conflict",
        ),
        "primary_source_supported": (
            e_index >= 3,
            "required_targets_have_verified_primary_support",
        ),
        "validation_controls": (
            human_verified and structure_controls_passed,
            "human_review_and_enabled_structure_controls_pass",
        ),
        "independent_corroboration": (
            e_index >= 4,
            "required_targets_are_independently_corroborated",
        ),
        "reproducible_policy": (True, "persisted_policy_is_digest_bound"),
    }
    quality_results = _configured_checks(
        quality_checks,
        quality_builtins,
        case,
        dict(checker_context or {}),
        isolated_runner,
    )
    quality = _highest_level(
        B_LEVELS,
        configured_quality.get("level_checks", _DEFAULT_QUALITY_LEVELS),
        quality_results,
    )
    if incomplete:
        quality = "B0"

    attestations = _active(case.get("attestations", []))
    effective = [
        item
        for item in attestations
        if item.get("authority") in {"Hotpot", "User"}
        and (item.get("authority") != "Hotpot" or item.get("signature_verified") is True)
    ]
    approvals = [item for item in effective if item.get("approved") is True]
    rejections = [item for item in effective if item.get("approved") is False]
    actors = defaultdict(set)
    for item in effective:
        actors[bool(item.get("approved"))].add(
            (str(item.get("authority")), str(item.get("actor")))
        )
    disagreement = bool(actors[True] and actors[False])
    if disagreement:
        reasons.append(
            _reason(
                "benchmark_attestation_disagreement",
                "Active benchmark approvals and rejections disagree.",
            )
        )

    primary_verified = _verified_primary_source(case, required_target_ids)
    comparison_indeterminate = any(
        item.get("outcome") == "indeterminate" for item in conflict_assessments
    )
    conflict_free = not conflicting and not comparison_indeterminate
    benchmark_builtins = {
        "verified_primary_source": (
            primary_verified,
            "every_core_target_has_verified_primary_support",
        ),
        "conflict_free": (
            conflict_free,
            "no_conflict_or_indeterminate_comparison",
        ),
        "hotpot_or_user_attestation": (
            bool(approvals),
            "active_verified_hotpot_or_user_approval_present",
        ),
        "attestation_not_rejected": (
            not rejections and not disagreement,
            "no_active_hotpot_or_user_rejection_or_disagreement",
        ),
    }
    checklist = benchmark_policy["checklist"]
    direct_config = benchmark_policy["direct_entry"]
    veto_names = tuple(benchmark_policy["vetoes"])
    checklist_results = _configured_checks(
        checklist,
        benchmark_builtins,
        case,
        dict(checker_context or {}),
        isolated_runner,
    )
    direct_results = _configured_checks(
        direct_config,
        benchmark_builtins,
        case,
        dict(checker_context or {}),
        isolated_runner,
    )
    veto_config = {}
    for name in veto_names:
        if name in checklist:
            definition = checklist[name]
        elif name in direct_config:
            definition = direct_config[name]
        else:
            definition = {"kind": "builtin", "enabled": True}
        if (
            isinstance(definition, Mapping)
            and definition.get("kind") == "builtin"
            and definition.get("enabled") is False
        ):
            definition = {"kind": "builtin", "enabled": True}
        veto_config[name] = definition
    veto_results = _configured_checks(
        veto_config,
        benchmark_builtins,
        case,
        dict(checker_context or {}),
        isolated_runner,
    )
    enabled_checklist = [
        item for item in checklist_results.values() if item["status"] != "disabled"
    ]
    enabled_direct = [
        item for item in direct_results.values() if item["status"] != "disabled"
    ]
    checklist_passed = bool(enabled_checklist) and all(
        item["passed"] is True for item in enabled_checklist
    )
    direct = any(item["passed"] is True for item in enabled_direct)
    veto_passed = all(item["passed"] is True for item in veto_results.values())
    benchmark = bool(
        not incomplete
        and identity_determinate
        and veto_passed
        and (direct or checklist_passed)
    )
    if incomplete:
        benchmark_status = "ineligible"
    elif benchmark:
        benchmark_status = "admitted"
    elif disagreement or rejections:
        benchmark_status = "rejected"
    elif not identity_determinate or not veto_passed:
        benchmark_status = "vetoed"
    else:
        benchmark_status = "not_admitted"
    archive_class = "incomplete" if incomplete else "benchmark" if benchmark else "ordinary"

    return {
        "case_id": str(case["id"]),
        "quality_level": quality,
        "evidence_level": evidence_level,
        "evidence_levels": evidence["levels"],
        "archive_class": archive_class,
        "benchmark": benchmark,
        "benchmark_status": benchmark_status,
        "completeness": "incomplete" if incomplete else "complete",
        "checklist": {
            name: item["passed"] for name, item in checklist_results.items()
        },
        "direct_entry": direct,
        "veto_passed": veto_passed,
        "identity_determinate": identity_determinate,
        "gate_results": {
            "evidence": evidence["checks"],
            "quality": quality_results,
            "structure": structure_results,
            "benchmark_checklist": checklist_results,
            "benchmark_direct_entry": direct_results,
            "benchmark_vetoes": veto_results,
        },
        "reasons": reasons,
        "input_digest": canonical_digest(case),
    }


def _verified_primary_source(
    case: Mapping[str, Any], required_targets: Sequence[str]
) -> bool:
    evidence = _active(case.get("evidence", []))
    targets, target_aliases = _evidence_targets(case)
    target_kinds = {target_id: kind for target_id, kind, _ in targets}
    verifications = _active(case.get("verifications", []))
    verification_by_evidence: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for verification in verifications:
        verification_by_evidence[str(verification.get("evidence_id"))].append(
            verification
        )
    targets = list(required_targets)
    if not targets:
        return False
    for target in targets:
        target_kind = target_kinds.get(target)
        if target_kind is None:
            return False
        candidates = [
            item
            for item in evidence
            if _resolve_evidence_target(item.get("target"), target_aliases)
            == [(target, target_kind)]
            and _is_supporting_evidence(item)
        ]
        if not any(
            _identified_primary_source(item)
            and (item.get("locator") or item.get("quote") or item.get("matched_text"))
            and any(
                _verification_is_valid(verification, item)
                for verification in verification_by_evidence.get(
                    str(item.get("evidence_id")), []
                )
            )
            for item in candidates
        ):
            return False
    return True


def summarize_assessments(assessments: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    archive = Counter()
    quality = Counter()
    evidence = Counter()
    benchmark = Counter()
    reasons = Counter()
    count = 0
    for item in assessments:
        count += 1
        archive[str(item["archive_class"])] += 1
        quality[str(item["quality_level"])] += 1
        evidence[str(item["evidence_level"] or "not_applicable")] += 1
        benchmark[str(item["benchmark_status"])] += 1
        reasons.update(reason["code"] for reason in item.get("reasons", []))
    return {
        "cases": count,
        "archive_class": dict(sorted(archive.items())),
        "quality": dict(sorted(quality.items())),
        "evidence": dict(sorted(evidence.items())),
        "benchmark": dict(sorted(benchmark.items())),
        "reasons": dict(sorted(reasons.items())),
    }


__all__ = [
    "B_LEVELS",
    "E_LEVELS",
    "assess_case",
    "assess_evidence",
    "canonical_digest",
    "canonical_json",
    "compare_observations",
    "find_conflicts",
    "identity_key",
    "observation_context",
    "observation_property",
    "summarize_assessments",
    "validate_observation",
    "validate_property_policy",
]
