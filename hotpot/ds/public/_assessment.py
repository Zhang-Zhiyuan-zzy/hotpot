"""Deterministic Stage 13 assessment primitives.

The orchestration layer streams facts into immutable generations.  This module
contains the side-effect-free scientific/governance rules used for each Case
and conflict candidate, which keeps synchronization reproducible and testable.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..gates._checker import (
    CheckerIsolationUnavailableError,
    _CheckerDefinition,
    evaluate_configured_check,
)


B_LEVELS = tuple(f"B{index}" for index in range(5))
E_LEVELS = tuple(f"E{index}" for index in range(5))

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
        return True
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

    observations = _active(case.get("observations", []))
    claims = _active(case.get("claims", []))
    targets = [
        (_target_id(row, "observation", offset), row)
        for offset, row in enumerate(observations)
    ] + [
        (_target_id(row, "claim", offset), row)
        for offset, row in enumerate(claims)
    ]
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
    for target_id, _target in targets:
        linked = [item for item in evidence if str(item.get("target")) == target_id]
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

        independent_roots = set()
        independent_works = set()
        for item in verified:
            lineage = item.get("source_lineage_id")
            work = item.get("scientific_work_id") or item.get("doi")
            if lineage and work and not item.get("derived_from"):
                independent_roots.add(str(lineage).strip().lower())
                independent_works.add(str(work).strip().lower())
        independently_corroborated = (
            len(independent_roots) >= 2 and len(independent_works) >= 2
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

    required = list(required_targets) if required_targets is not None else [
        target_id for target_id, _ in targets
    ]
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


def identity_key(
    case: Mapping[str, Any], identity_schema: Mapping[str, Any]
) -> Tuple[Optional[str], list[Dict[str, Any]]]:
    """Return a deterministic equivalence key or an indeterminate result."""

    components = identity_schema.get("components", [])
    if not components:
        return f"case:{case['id']}", []
    normalized = []
    reasons = []
    for component in components:
        path = str(component["path"])
        value: Any = case
        for part in path.replace("__", ".").split("."):
            if isinstance(value, Mapping) and part in value:
                value = value[part]
            else:
                value = None
                break
        if value in (None, "", [], {}):
            if component.get("required", False):
                reasons.append(
                    _reason(
                        "identity_indeterminate",
                        "A required identity component is missing.",
                        component=component.get("key", path),
                    )
                )
                return None, reasons
            continue
        normalizer = component.get("normalizer", "text")
        if normalizer == "casefold":
            value = str(value).strip().casefold()
        elif normalizer == "number":
            value = float(value)
        elif normalizer == "set":
            value = sorted(set(value))
        else:
            value = str(value).strip()
        normalized.append([component.get("key", path), value])
    if not normalized:
        return None, [
            _reason(
                "identity_indeterminate",
                "No persisted identity component produced a comparable value.",
            )
        ]
    return canonical_digest(normalized), reasons


def observation_context(observation: Mapping[str, Any]) -> Dict[str, Any]:
    """Canonical comparison context, deliberately excluding the target value."""

    return {
        "property": observation.get("property")
        or observation.get("property_type_id"),
        "schema_version": observation.get("schema_version", "current"),
        "subject": observation.get("subject"),
        "site": observation.get("site"),
        "phase": observation.get("phase"),
        "conditions": observation.get("conditions", {}),
        "participants": observation.get("participants", []),
        "method": observation.get("method", {}),
        "branch": observation.get("branch"),
        "unit": observation.get("unit"),
    }


def compare_observations(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    tolerances: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    """Compare one already same-sample candidate pair."""

    left_context = observation_context(left)
    right_context = observation_context(right)
    if left_context["property"] != right_context["property"]:
        return {"outcome": "noncandidate", "reason": "different_property"}
    if canonical_json(left_context) != canonical_json(right_context):
        return {
            "outcome": "indeterminate",
            "reason": "observation_context_indeterminate",
        }
    left_value = left.get("value")
    right_value = right.get("value")
    if isinstance(left_value, (int, float)) and not isinstance(left_value, bool) and isinstance(
        right_value, (int, float)
    ) and not isinstance(right_value, bool):
        property_name = str(left_context["property"])
        rule = tolerances.get(property_name)
        if left_value == right_value:
            return {"outcome": "consistent", "reason": "numeric_equal"}
        if rule is None:
            return {
                "outcome": "indeterminate",
                "reason": "numeric_tolerance_not_configured",
            }
        absolute = float(rule.get("absolute", 0.0))
        relative = float(rule.get("relative", 0.0))
        allowed = max(absolute, relative * max(abs(left_value), abs(right_value)))
        uncertainty = max(
            _uncertainty(left.get("uncertainty")),
            _uncertainty(right.get("uncertainty")),
        )
        allowed = max(allowed, uncertainty)
        difference = abs(left_value - right_value)
        return {
            "outcome": "consistent" if difference <= allowed else "conflict",
            "reason": "within_numeric_tolerance"
            if difference <= allowed
            else "scientific_value_conflict",
            "difference": difference,
            "allowed": allowed,
        }
    if canonical_json(left_value) == canonical_json(right_value):
        return {"outcome": "consistent", "reason": "categorical_equal"}
    return {"outcome": "conflict", "reason": "scientific_value_conflict"}


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
) -> Dict[str, Any]:
    """Deterministically compare unique candidates within identity groups."""

    groups: Dict[str, list[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    identity_reasons: Dict[str, list[Dict[str, Any]]] = {}
    for case in cases:
        case_id = str(case["id"])
        key, reasons = identity_key(case, identity_schema)
        identity_reasons[case_id] = reasons
        if key is None:
            continue
        for offset, observation in enumerate(_active(case.get("observations", []))):
            candidate = deepcopy(observation)
            candidate.setdefault(
                "observation_id", f"{case_id}:observation:{offset}"
            )
            groups[key].append((case_id, candidate))

    rows = []
    conflicting_cases = set()
    summaries = []
    for sample_key in sorted(groups):
        candidates = sorted(
            groups[sample_key], key=lambda item: str(item[1]["observation_id"])
        )
        pair_results = []
        consistent = 0
        exceptions = 0
        for left_index, (left_case, left) in enumerate(candidates):
            for right_case, right in candidates[left_index + 1 :]:
                result = compare_observations(left, right, tolerances)
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
    checker_context: Optional[Mapping[str, Any]] = None,
    isolated_runner: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute the sole authoritative CaseAssessment for one generation."""

    reasons: list[Dict[str, Any]] = []
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
    for role in required_structure_roles:
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
        valid = any(
            any(
                assessment.get("presence") == "present"
                and assessment.get("hash_result") in {"match", "unchecked"}
                and assessment.get("parse_result") == "valid"
                for assessment in asset_by_structure.get(str(item.get("structure_id")), [])
            )
            for item in matching
        )
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
    }
    incomplete = any(item["code"] in incomplete_codes for item in reasons)
    e_index = -1 if evidence_level is None else int(evidence_level[1])
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
            and all(
                any(item.get("role") == role for item in structures)
                for role in required_structure_roles
            ),
            "structure_roles_are_declared_and_required_roles_are_present",
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
        not incomplete and veto_passed and (direct or checklist_passed)
    )
    if incomplete:
        benchmark_status = "ineligible"
    elif benchmark:
        benchmark_status = "admitted"
    elif disagreement or rejections:
        benchmark_status = "rejected"
    elif not veto_passed:
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
        candidates = [item for item in evidence if str(item.get("target")) == target]
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
    "summarize_assessments",
]
