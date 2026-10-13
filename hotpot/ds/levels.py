"""Dataset quality and observation evidence levels."""

from __future__ import annotations

from enum import Enum
import re
from typing import Dict, Tuple


class QualityLevel(str, Enum):
    B0 = "B0_collection"
    B1 = "B1_candidate"
    B2 = "B2_curated"
    B3 = "B3_validated"
    B4 = "B4_reference"


class EvidenceLevel(str, Enum):
    E0 = "E0_insufficient"
    E1 = "E1_extracted"
    E2 = "E2_explicit_reported"
    E3 = "E3_primary_source_supported"
    E4 = "E4_independently_corroborated"


QUALITY_LEVEL_CRITERIA: Dict[QualityLevel, Tuple[str, ...]] = {
    QualityLevel.B0: (),
    QualityLevel.B1: (
        "provenance_complete",
        "subject_mapping_rule_audited",
    ),
    QualityLevel.B2: (
        "provenance_complete",
        "subject_mapping_rule_audited",
        "domain_reviewed",
        "all_records_reviewed",
        "conflicts_resolved",
        "family_groups_available",
        "accuracy_estimate_available",
    ),
    QualityLevel.B3: (
        "provenance_complete",
        "subject_mapping_rule_audited",
        "domain_reviewed",
        "all_records_reviewed",
        "conflicts_resolved",
        "family_groups_available",
        "accuracy_estimate_available",
        "primary_source_supported_adjudication",
        "dual_review",
        "split_frozen",
        "metric_frozen",
        "leakage_audited",
        "conditions_complete",
        "uncertainty_policy_defined",
    ),
    QualityLevel.B4: (
        "provenance_complete",
        "subject_mapping_rule_audited",
        "domain_reviewed",
        "all_records_reviewed",
        "conflicts_resolved",
        "family_groups_available",
        "accuracy_estimate_available",
        "primary_source_supported_adjudication",
        "dual_review",
        "split_frozen",
        "metric_frozen",
        "leakage_audited",
        "conditions_complete",
        "uncertainty_policy_defined",
        "external_audit",
        "multi_source_corroboration",
        "reproducible_fixed_release",
    ),
}

QUALITY_LEVEL_ARTIFACTS: Dict[QualityLevel, Tuple[str, ...]] = {
    QualityLevel.B0: (),
    QualityLevel.B1: ("provenance_record", "subject_mapping_report", "task_definition"),
    QualityLevel.B2: (
        "review_report",
        "review_coverage",
        "accuracy_estimate",
        "conflict_policy",
        "family_group_key",
    ),
    QualityLevel.B3: (
        "primary_source_review_report",
        "independent_review_report",
        "split_manifest",
        "metric_definition",
        "leakage_report",
        "conditions_policy",
        "uncertainty_policy",
    ),
    QualityLevel.B4: (
        "external_audit_report",
        "corroboration_report",
        "release_manifest_sha256",
    ),
}

def _artifact_present(name: str, value: object) -> bool:
    if name == "accuracy_estimate":
        return (
            isinstance(value, dict)
            and isinstance(value.get("sample_size"), int)
            and value["sample_size"] > 0
            and isinstance(value.get("estimated_accuracy"), (int, float))
            and 0 <= value["estimated_accuracy"] <= 1
            and isinstance(value.get("reference_standard"), str)
            and bool(value["reference_standard"])
            and isinstance(value.get("quantity_assessed"), str)
            and bool(value["quantity_assessed"])
        )
    if name == "release_manifest_sha256":
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if name == "review_coverage":
        return (
            isinstance(value, dict)
            and isinstance(value.get("reviewed_count"), int)
            and isinstance(value.get("total_count"), int)
            and value["reviewed_count"] > 0
            and value["reviewed_count"] == value["total_count"]
        )
    return isinstance(value, (str, dict)) and bool(value)


def level_is_supported(quality: Dict[str, object], level: QualityLevel) -> bool:
    rank = tuple(QualityLevel).index(level)
    required_artifacts = tuple(
        artifact
        for candidate in tuple(QualityLevel)[: rank + 1]
        for artifact in QUALITY_LEVEL_ARTIFACTS[candidate]
    )
    supported = all(
        quality.get(requirement) is True
        for requirement in QUALITY_LEVEL_CRITERIA[level]
    ) and all(
        _artifact_present(artifact, quality.get(artifact))
        for artifact in required_artifacts
    )
    if supported and level_rank(level) >= level_rank(QualityLevel.B2):
        review_coverage = quality["review_coverage"]
        accuracy_estimate = quality["accuracy_estimate"]
        supported = (
            review_coverage["reviewed_count"]
            == accuracy_estimate["sample_size"]
        )
    return supported


def highest_satisfied_level(quality: Dict[str, object]) -> QualityLevel:
    for level in reversed(tuple(QualityLevel)):
        if level_is_supported(quality, level):
            return level
    return QualityLevel.B0


def level_rank(level: QualityLevel) -> int:
    return tuple(QualityLevel).index(level)
