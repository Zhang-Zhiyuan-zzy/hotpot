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
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


B_LEVELS = tuple(f"B{index}" for index in range(5))
E_LEVELS = tuple(f"E{index}" for index in range(5))


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


def assess_evidence(
    case: Mapping[str, Any], required_targets: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """Assess active observation/claim support and its conservative Case level."""

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
    reasons: list[Dict[str, Any]] = []
    for target_id, _target in targets:
        linked = [item for item in evidence if str(item.get("target")) == target_id]
        level = "E0"
        if linked:
            level = "E1"
        explicitly_located = [
            item
            for item in linked
            if item.get("locator") or item.get("quote") or item.get("matched_text")
        ]
        if explicitly_located:
            level = "E2"
        verified = []
        for item in explicitly_located:
            checks = verification_by_evidence.get(str(item.get("evidence_id")), [])
            for check in checks:
                if check.get("method") == "Human":
                    verified.append(item)
                    break
                if check.get("method") == "AI" and all(
                    check.get(field)
                    for field in ("verifier", "procedure_version", "verified_at")
                ):
                    verified.append(item)
                    break
        if verified:
            level = "E3"

        independent_roots = set()
        independent_works = set()
        for item in verified:
            lineage = item.get("source_lineage_id")
            work = item.get("scientific_work_id") or item.get("doi")
            if lineage and work and not item.get("derived_from"):
                independent_roots.add(str(lineage).strip().lower())
                independent_works.add(str(work).strip().lower())
        if len(independent_roots) >= 2 and len(independent_works) >= 2:
            level = "E4"
        elif len(verified) >= 2:
            reasons.append(
                _reason(
                    "evidence_independence_unproven",
                    "Several verified sources exist, but independent result lineage is not established.",
                    target=target_id,
                )
            )
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
    if structures and not benchmark_policy.get("structure_checks"):
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
    evidence = assess_evidence(case, required_target_ids if core_labels else None)
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
    has_provenance = bool(case.get("provenance") or case.get("metadata"))
    subject_mapped = bool(case.get("identifiers") or structures)
    e_index = -1 if evidence_level is None else int(evidence_level[1])
    human_verified = any(
        item.get("method") == "Human"
        for item in _active(case.get("verifications", []))
    )

    if incomplete:
        quality = "B0"
    elif has_provenance and subject_mapped:
        quality = "B1"
        if e_index >= 2:
            quality = "B2"
        if e_index >= 3 and human_verified:
            quality = "B3"
        if e_index >= 4:
            quality = "B4"
    else:
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
    checklist = benchmark_policy.get("checklist", {})
    checklist_results = {
        "verified_primary_source": primary_verified,
        "conflict_free": not conflicting,
    }
    enabled_checklist = [
        name for name, setting in checklist.items() if setting is not False
    ]
    checklist_passed = bool(enabled_checklist) and all(
        checklist_results.get(name, False) for name in enabled_checklist
    )
    direct = bool(approvals)
    veto_passed = not conflicting and not rejections and not disagreement
    benchmark = bool(
        not incomplete and veto_passed and (direct or checklist_passed)
    )
    if incomplete:
        benchmark_status = "ineligible"
    elif benchmark:
        benchmark_status = "admitted"
    elif disagreement or rejections:
        benchmark_status = "rejected"
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
        "checklist": checklist_results,
        "direct_entry": direct,
        "veto_passed": veto_passed,
        "reasons": reasons,
        "input_digest": canonical_digest(case),
    }


def _verified_primary_source(
    case: Mapping[str, Any], required_targets: Sequence[str]
) -> bool:
    evidence = _active(case.get("evidence", []))
    verifications = _active(case.get("verifications", []))
    verified_ids = {
        str(item.get("evidence_id"))
        for item in verifications
        if item.get("method") in {"AI", "Human"}
    }
    targets = list(required_targets)
    if not targets:
        return False
    for target in targets:
        candidates = [item for item in evidence if str(item.get("target")) == target]
        if not any(
            str(item.get("evidence_id")) in verified_ids
            and (item.get("doi") or str(item.get("url", "")).startswith(("http://", "https://")))
            and (item.get("locator") or item.get("quote") or item.get("matched_text"))
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
