"""Loss-aware adapter for the CSD reported electronic-state text database.

The source database contains text-derived claims and site mappings, not an
independently validated electronic-state measurement.  Owner adjudications can
reject extraction errors, quarantine contested values, link independently
curated structures, and add explicitly derived descriptors without rewriting
the preserved supplier evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from .._paths import AUDITS_ROOT
from ..levels import EvidenceLevel, QualityLevel
from ..models import (
    STANDARD_VERSION,
    Claim,
    ClaimEvidence,
    ClaimMembership,
    ComputedDescriptor,
    Dataset,
    DatasetBundle,
    DatasetRecord,
    DatasetRelease,
    DatasetSubset,
    Entity,
    EntityRelation,
    EvidenceItem,
    ExternalIdentifier,
    Observation,
    ObservationEvidence,
    ObservationMembership,
    RecordEntity,
    Source,
    SubsetMembership,
)
from ..properties import builtin_property_types


_SCOPES = frozenset({"all", "oxidation", "spin"})


def _key(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _family(refcode: str) -> str:
    match = re.match(r"^([A-Za-z]{6})(?:\d+)?$", refcode)
    return match.group(1).upper() if match else refcode.upper()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _metadata(connection: sqlite3.Connection) -> Dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM metadata ORDER BY key"))


def _b1_quality(property_name: str, view: str, source_metadata: Dict[str, str]) -> Dict[str, object]:
    return {
        "provenance_complete": True,
        "subject_mapping_rule_audited": True,
        "provenance_scope": (
            "complete from portable SQLite fields to extracted candidate; "
            "not a per-record primary-paper evidence chain"
        ),
        "provenance_record": {
            "source_entry_jsonl_sha256": source_metadata.get("source_entry_jsonl_sha256"),
            "source_site_jsonl_sha256": source_metadata.get("source_site_jsonl_sha256"),
            "schema_version": source_metadata.get("schema_version"),
        },
        "subject_mapping_report": {
            "selection_view": view,
            "status": "single retained site candidate; not independent scientific validation",
        },
        "task_definition": {
            "task": f"predict reported {property_name} from a grouped CSD entry",
            "target_subject": "mapped metal site",
            "group_key": "record.metadata.refcode_family",
            "leakage_policy": "all variants of one refcode family must remain in one split",
        },
    }


def convert_csd_electronic_state(
    database_path: Union[str, Path],
    *,
    license_name: str,
    access_policy: str = "restricted",
    source_citation: str = "Cambridge Structural Database (CSD) reported-text fields",
    version: str = "reported-text-v1",
    scope: str = "all",
    limit: Optional[int] = None,
    quality_assessment_path: Optional[Union[str, Path]] = None,
    evidence_audit_path: Optional[Union[str, Path]] = None,
) -> DatasetBundle:
    """Convert the portable CSD text database to the generic dataset model.

    ``license_name`` is deliberately mandatory: the adapter cannot infer the
    caller's CSD licence or redistribution rights.  ``limit`` selects the first
    entries in refcode order and exists for inspection/tests; omitted means the
    complete selected scope.
    """
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {sorted(_SCOPES)}")

    path = Path(database_path)
    source_database_sha256 = _sha256(path)
    assessment_path = (
        Path(quality_assessment_path)
        if quality_assessment_path is not None
        else AUDITS_ROOT / "csd-electronic-state-quality.json"
    )
    quality_assessment = None
    assessment_sha256 = None
    known_conflicts: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    if quality_assessment_path is not None and not assessment_path.is_file():
        raise FileNotFoundError(assessment_path)
    if assessment_path.is_file():
        quality_assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
        assessed_database_sha256 = quality_assessment.get("database", {}).get("sha256")
        if assessed_database_sha256 != source_database_sha256:
            raise ValueError("quality assessment does not match the CSD source database")
        assessment_sha256 = _sha256(assessment_path)
        disagreements = (
            quality_assessment.get("oxidation", {})
            .get("crosscheck_against_current_hotpot", {})
            .get("disagreements", [])
        )
        for disagreement in disagreements:
            for state in disagreement.get("csd_text_states", []):
                known_conflicts[
                    (
                        str(disagreement["refcode"]),
                        str(disagreement["element"]),
                        int(state),
                    )
                ] = {
                    "comparison_value": disagreement.get("hotpot_curated_state"),
                    "assessment_sha256": assessment_sha256,
                    "status": "unresolved_cross_dataset_disagreement",
                }

    audit_candidate = (
        Path(evidence_audit_path)
        if evidence_audit_path is not None
        else AUDITS_ROOT / "evidence_regrade_audit.json"
    )
    supplier_audit = None
    supplier_audit_sha256 = None
    if evidence_audit_path is not None and not audit_candidate.is_file():
        raise FileNotFoundError(audit_candidate)
    if audit_candidate.is_file():
        candidate = json.loads(audit_candidate.read_text(encoding="utf-8"))
        audited_sha256 = candidate.get("scope", {}).get("database_sha256")
        if audited_sha256 == source_database_sha256:
            supplier_audit = candidate
            supplier_audit_sha256 = _sha256(audit_candidate)
        elif evidence_audit_path is not None:
            raise ValueError("evidence audit does not match the CSD source database")

    false_positive_oxidation: Set[Tuple[str, str, int]] = set()
    context_corrections: Dict[str, Set[str]] = {}
    curated_spin_links: Dict[str, Dict[str, object]] = {}
    nominal_spin_rows: Dict[str, List[Dict[str, object]]] = {}
    nominal_spin_assessment: Dict[str, object] = {}
    if supplier_audit is not None:
        for item in supplier_audit.get("information_extraction_errors", []):
            if item.get("error_type") == "oxidation_false_positive_nomenclature_parenthesis":
                false_positive_oxidation.add(
                    (
                        str(item["identifier"]),
                        str(item["element"]),
                        int(item["incorrect_extracted_value"]),
                    )
                )
            elif item.get("error_type") == "spin_context_underclassification":
                context_corrections.setdefault(str(item["identifier"]), set()).update(
                    str(flag) for flag in item.get("context_flags", [])
                )
        for item in supplier_audit.get("recommended_evidence_downgrades", []):
            if "current_csd_reported_states" in item:
                for state in item["current_csd_reported_states"]:
                    known_conflicts[
                        (str(item["identifier"]), str(item["element"]), int(state))
                    ] = {
                        "comparison_value": item.get("sco95_curated_state"),
                        "audit_sha256": supplier_audit_sha256,
                        "status": "owner_quarantined_cross_dataset_disagreement",
                    }
        curated_spin_links = {
            str(item["identifier"]): dict(item)
            for item in supplier_audit.get("recommended_evidence_upgrades", [])
        }
        nominal_spin_assessment = supplier_audit.get(
            "literature_supported_nominal_spin_assessment", {}
        )
        for item in nominal_spin_assessment.get("per_record_assessments", []):
            nominal_spin_rows.setdefault(str(item["identifier"]), []).append(dict(item))
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    try:
        source_metadata = _metadata(connection)
        connection.execute("CREATE TEMP TABLE selected_refcodes (refcode TEXT PRIMARY KEY)")
        predicates = {
            "all": "has_oxidation_text = 1 OR has_spin_text = 1",
            "oxidation": "has_oxidation_text = 1",
            "spin": "has_spin_text = 1",
        }
        selection_sql = (
            "INSERT INTO selected_refcodes SELECT refcode FROM entries "
            f"WHERE {predicates[scope]} ORDER BY refcode"
        )
        parameters: Tuple[int, ...] = ()
        if limit is not None:
            selection_sql += " LIMIT ?"
            parameters = (limit,)
        connection.execute(selection_sql, parameters)

        entry_rows = connection.execute(
            "SELECT e.* FROM entries e JOIN selected_refcodes r USING (refcode) "
            "ORDER BY e.refcode"
        ).fetchall()
        site_rows = connection.execute(
            "SELECT s.* FROM sites s JOIN selected_refcodes r USING (refcode) "
            "ORDER BY s.refcode, s.site_id"
        ).fetchall()
        context_by_refcode: Dict[str, List[Dict[str, str]]] = {}
        for row in connection.execute(
            "SELECT f.* FROM entry_spin_context_flags f "
            "JOIN selected_refcodes r USING (refcode) "
            "ORDER BY f.refcode, f.flag, f.source_text, f.matched_text"
        ):
            context_by_refcode.setdefault(row["refcode"], []).append(
                {
                    "flag": row["flag"],
                    "source_text": row["source_text"],
                    "matched_text": row["matched_text"],
                }
            )
        selected_refcodes = {row["refcode"] for row in entry_rows}
        for refcode, flags in context_corrections.items():
            if refcode not in selected_refcodes:
                continue
            existing = context_by_refcode.setdefault(refcode, [])
            existing_flags = {item["flag"] for item in existing}
            existing.extend(
                {
                    "flag": flag,
                    "source_text": "owner evidence audit",
                    "matched_text": flag,
                }
                for flag in sorted(flags - existing_flags)
            )

        oxidation_entry_grades = {
            (row["refcode"], row["element"]): row["grade"]
            for row in connection.execute(
                "SELECT e.* FROM entry_oxidation_evidence e "
                "JOIN selected_refcodes r USING (refcode)"
            )
        }
        oxidation_site_grades = {
            row["site_id"]: (row["grade"], row["evidence_count"])
            for row in connection.execute(
                "SELECT e.* FROM site_oxidation_evidence e JOIN sites s USING (site_id) "
                "JOIN selected_refcodes r USING (refcode)"
            )
        }
        spin_entry_grades = {
            row["refcode"]: row["grade"]
            for row in connection.execute(
                "SELECT e.* FROM entry_spin_evidence e "
                "JOIN selected_refcodes r USING (refcode)"
            )
        }
        spin_site_grades = {
            row["site_id"]: (row["grade"], row["evidence_count"])
            for row in connection.execute(
                "SELECT e.* FROM site_spin_evidence e JOIN sites s USING (site_id) "
                "JOIN selected_refcodes r USING (refcode)"
            )
        }

        dataset_id = "csd-electronic-state"
        release_id = f"{dataset_id}.v{_safe(version)}"
        datasets = (
            Dataset(
                dataset_id,
                "CSD reported electronic-state text",
                "Text claims and rule-derived site candidates for oxidation and spin categories.",
            ),
        )
        releases = (
            DatasetRelease(
                release_id,
                dataset_id,
                _safe(version),
                QualityLevel.B0.value,
                license_name,
                access_policy,
                {
                    "adapter": "csd-electronic-state",
                    "source_database": path.name,
                    "source_database_sha256": source_database_sha256,
                    "source_metadata": source_metadata,
                    "quality_assessment": (
                        {
                            "file": assessment_path.name,
                            "sha256": assessment_sha256,
                            "version": quality_assessment.get("assessment_version"),
                        }
                        if quality_assessment is not None
                        else None
                    ),
                    "owner_evidence_audit": (
                        {
                            "file": audit_candidate.name,
                            "sha256": supplier_audit_sha256,
                            "disposition": "owner-adjudicated; supplier value judgments are non-binding",
                        }
                        if supplier_audit is not None
                        else None
                    ),
                    "scope": scope,
                    "limit": limit,
                },
                {},
            ),
        )

        subset_specs = []
        if scope in {"all", "oxidation"}:
            subset_specs.extend(
                [
                    (
                        "ox-t1-active",
                        "Oxidation T1 active site candidates",
                        QualityLevel.B1,
                        {"database_view": "v_ox_t1_active_sites"},
                        _b1_quality(
                            "oxidation state", "v_ox_t1_active_sites", source_metadata
                        ),
                    ),
                    (
                        "oxidation-claims",
                        "Oxidation text claims not admitted as T1 observations",
                        QualityLevel.B0,
                        {"selection": "oxidation mentions not linked to an admitted observation"},
                        {},
                    ),
                ]
            )
        if scope in {"all", "spin"}:
            subset_specs.extend(
                [
                    (
                        "spin-t1-active-unqualified-claims",
                        "Regex-unflagged spin T1 extracted claims",
                        QualityLevel.B0,
                        {
                            "database_view": "v_sp_t1_active_unqualified_sites",
                            "meaning": "no configured context regex matched; not a scientific clearance",
                        },
                        {},
                    ),
                    (
                        "spin-other-claims",
                        "Conditional or unresolved spin text claims",
                        QualityLevel.B0,
                        {"selection": "spin mentions outside the unqualified T1 candidate view"},
                        {},
                    ),
                ]
            )
        subset_ids = {
            local_id: f"{release_id}:subset:{local_id}"
            for local_id, *_ in subset_specs
        }
        subsets = tuple(
            DatasetSubset(
                subset_ids[local_id], release_id, None, name, level.value, selection, quality
            )
            for local_id, name, level, selection, quality in subset_specs
        )

        record_id_by_refcode = {
            row["refcode"]: _key("csd-record", f"{release_id}:{row['refcode']}")
            for row in entry_rows
        }
        entry_entity_by_refcode = {
            row["refcode"]: _key("csd-entry", f"{release_id}:{row['refcode']}")
            for row in entry_rows
        }
        site_entity_by_id = {
            row["site_id"]: _key("csd-site", f"{release_id}:{row['site_id']}")
            for row in site_rows
        }
        records = tuple(
            DatasetRecord(
                record_id_by_refcode[row["refcode"]],
                release_id,
                row["refcode"],
                {
                    "refcode": row["refcode"],
                    "refcode_family": _family(row["refcode"]),
                    "crystal_grade": row["crystal_grade"],
                    "formula_text": row["formula_text"],
                    "has_oxidation_text": bool(row["has_oxidation_text"]),
                    "has_spin_text": bool(row["has_spin_text"]),
                    "spin_context_flags": context_by_refcode.get(row["refcode"], []),
                },
            )
            for row in entry_rows
        )

        entities: List[Entity] = []
        external_identifiers: List[ExternalIdentifier] = []
        record_entities: List[RecordEntity] = []
        for row in entry_rows:
            refcode = row["refcode"]
            entity_id = entry_entity_by_refcode[refcode]
            entities.append(Entity(entity_id, "csd_entry", None, {"refcode_family": _family(refcode)}))
            external_identifiers.append(
                ExternalIdentifier(entity_id, "CSD_identifier", refcode, None)
            )
            record_entities.append(
                RecordEntity(record_id_by_refcode[refcode], entity_id, "reported_entry")
            )
        for row in site_rows:
            entity_id = site_entity_by_id[row["site_id"]]
            oxidation_grade = oxidation_site_grades.get(row["site_id"])
            spin_grade = spin_site_grades.get(row["site_id"])
            entities.append(
                Entity(
                    entity_id,
                    "metal_site",
                    entry_entity_by_refcode[row["refcode"]],
                    {
                        "core_id": row["core_id"],
                        "representative_label": row["representative_label"],
                        "element": row["element"],
                        "coordination_number": row["coordination_number"],
                        "primary_core_class": row["primary_core_class"],
                        "core_classification_status": row["core_classification_status"],
                        "oxidation_evidence_grade": oxidation_grade[0] if oxidation_grade else None,
                        "oxidation_evidence_count": oxidation_grade[1] if oxidation_grade else None,
                        "spin_evidence_grade": spin_grade[0] if spin_grade else None,
                        "spin_evidence_count": spin_grade[1] if spin_grade else None,
                    },
                )
            )
            external_identifiers.append(
                ExternalIdentifier(entity_id, "CSD_site_id", row["site_id"], None)
            )
            record_entities.append(
                RecordEntity(record_id_by_refcode[row["refcode"]], entity_id, "metal_site")
            )

        entity_relations: List[EntityRelation] = []
        linked_structure_entities: Set[str] = set()
        for refcode, link in sorted(curated_spin_links.items()):
            if refcode not in selected_refcodes:
                continue
            structure_id = str(link["sco95_structure_id"])
            structure_entity_id = f"entity:structure:{structure_id}"
            if structure_entity_id not in linked_structure_entities:
                entities.append(
                    Entity(structure_entity_id, "molecular_structure", None, {})
                )
                linked_structure_entities.add(structure_entity_id)
            entity_relations.append(
                EntityRelation(
                    entry_entity_by_refcode[refcode],
                    "has_independently_curated_structure",
                    structure_entity_id,
                    {
                        "curated_dataset": "spin",
                        "curated_subset": "SCO-95",
                        "spin_category": link["spin_category"],
                        "spin_multiplicity": link["sco95_spin_multiplicity"],
                        "owner_disposition": (
                            "relation_added_without_upgrading_the_CSD_text_claim"
                        ),
                    },
                )
            )

        descriptors: List[ComputedDescriptor] = []
        assumption_set = nominal_spin_assessment.get("applicability_assumption_set", {})
        for refcode, rows in sorted(nominal_spin_rows.items()):
            if refcode not in selected_refcodes:
                continue
            for row in rows:
                site_entity_id = site_entity_by_id.get(str(row["site_id"]))
                if site_entity_id is None:
                    continue
                descriptors.append(
                    ComputedDescriptor(
                        _key(
                            "csd-nominal-spin-descriptor",
                            f"{release_id}:{row['site_id']}:{row['derivation_rule_id']}",
                        ),
                        site_entity_id,
                        "nominal_local_spin_multiplicity",
                        int(row["derived_nominal_multiplicity_2S_plus_1"]),
                        None,
                        {
                            "derivation_rule_id": row["derivation_rule_id"],
                            "formal_oxidation_state": row["oxidation_state"],
                            "spin_category": row["spin_category"],
                            "assumption_set_id": assumption_set.get("assumption_set_id"),
                            "supporting_reference_ids": row.get(
                                "supporting_reference_ids", []
                            ),
                        },
                        {
                            "annotation_kind": "derived_nominal_local_spin",
                            "not_an_experimental_observation": True,
                            "evidence_level_not_applicable": True,
                            "supplier_audit_sha256": supplier_audit_sha256,
                        },
                    )
                )

        selected_property_types = (
            ({"oxidation_state"} if scope in {"all", "oxidation"} else set())
            | (
                {"spin_category", "spin_transition_behavior", "spin_state_text"}
                if scope in {"all", "spin"}
                else set()
            )
        )
        property_types = tuple(
            row
            for row in builtin_property_types()
            if row.property_type_id in selected_property_types
        )

        observations: List[Observation] = []
        observation_by_ox: Dict[Tuple[str, str, int], str] = {}
        memberships: Set[Tuple[str, str]] = set()
        observation_memberships: Set[Tuple[str, str]] = set()
        claim_memberships: Set[Tuple[str, str]] = set()
        if scope in {"all", "oxidation"}:
            for row in connection.execute(
                "SELECT v.* FROM v_ox_t1_active_sites v "
                "JOIN selected_refcodes r USING (refcode) "
                "ORDER BY v.refcode, v.site_id, v.oxidation_state"
            ):
                observation_key = (
                    row["refcode"],
                    row["element"],
                    row["oxidation_state"],
                )
                if observation_key in known_conflicts or observation_key in false_positive_oxidation:
                    continue
                observation_id = _key(
                    "csd-ox-observation",
                    f"{release_id}:{row['site_id']}:{row['oxidation_state']}",
                )
                observation_by_ox[observation_key] = observation_id
                observations.append(
                    Observation(
                        observation_id,
                        record_id_by_refcode[row["refcode"]],
                        site_entity_by_id[row["site_id"]],
                        "oxidation_state",
                        row["oxidation_state"],
                        None,
                        {},
                        {},
                        "active",
                        EvidenceLevel.E2.value,
                        {
                            "mapping_rule": source_metadata.get("oxidation_t1_semantics"),
                            "selection_view": "v_ox_t1_active_sites",
                        },
                        {"source_grade": "OX-T1_EXPLICIT_SINGLE_SITE_CANDIDATE"},
                    )
                )
                memberships.add(
                    (subset_ids["ox-t1-active"], record_id_by_refcode[row["refcode"]])
                )
                observation_memberships.add(
                    (subset_ids["ox-t1-active"], observation_id)
                )

        source_id = _key(
            "source",
            f"{source_citation}|{license_name}|{access_policy}",
        )
        source_rows = [
            Source(
                source_id,
                "reported_database_text",
                source_citation,
                None,
                license_name,
                access_policy,
                {},
            ),
        ]
        audit_source_id = None
        if assessment_sha256 is not None:
            audit_source_id = _key("source", f"quality-assessment:{assessment_sha256}")
            source_rows.append(
                Source(
                    audit_source_id,
                    "cross_dataset_quality_assessment",
                    "Hotpot CSD electronic-state quality assessment",
                    None,
                    license_name,
                    access_policy,
                    {"assessment_sha256": assessment_sha256},
                )
            )
        owner_audit_source_id = None
        if supplier_audit_sha256 is not None:
            owner_audit_source_id = _key(
                "source", f"owner-evidence-adjudication:{supplier_audit_sha256}"
            )
            source_rows.append(
                Source(
                    owner_audit_source_id,
                    "owner_evidence_adjudication",
                    "Hotpot owner adjudication of the supplier evidence regrade audit",
                    None,
                    license_name,
                    access_policy,
                    {
                        "supplier_audit_sha256": supplier_audit_sha256,
                        "authority": "Hotpot dataset owner",
                    },
                )
            )
        sources = tuple(source_rows)
        evidence: List[EvidenceItem] = []
        observation_evidence: List[ObservationEvidence] = []
        claims: List[Claim] = []
        claim_evidence: List[ClaimEvidence] = []
        claim_by_ox: Dict[Tuple[str, str, int], str] = {}

        if scope in {"all", "oxidation"}:
            oxidation_t1_sites = {
                (row["refcode"], row["element"], row["oxidation_state"]):
                    site_entity_by_id[row["site_id"]]
                for row in connection.execute(
                    "SELECT v.* FROM v_ox_t1_sites v "
                    "JOIN selected_refcodes r USING (refcode)"
                )
            }
            query = """
                SELECT m.*, GROUP_CONCAT(ms.oxidation_state) AS parsed_states
                FROM oxidation_mentions m
                JOIN selected_refcodes r USING (refcode)
                LEFT JOIN oxidation_mention_states ms USING (mention_id)
                GROUP BY m.mention_id
                ORDER BY m.mention_id
            """
            for row in connection.execute(query):
                states = tuple(
                    sorted(int(value) for value in row["parsed_states"].split(","))
                ) if row["parsed_states"] else ()
                evidence_id = _key(
                    "csd-ox-evidence", f"{release_id}:{row['mention_id']}"
                )
                grade = oxidation_entry_grades.get((row["refcode"], row["element"]))
                single_key = (
                    (row["refcode"], row["element"], states[0])
                    if len(states) == 1
                    else None
                )
                rejected_false_positive = single_key in false_positive_oxidation
                contested = single_key in known_conflicts
                evidence.append(
                    EvidenceItem(
                        evidence_id,
                        source_id,
                        "verbatim_database_field",
                        f"CSD:{row['refcode']}/{row['source_tag']}/oxidation-mention:{row['mention_id']}",
                        row["source_text"],
                        row["matched_text"],
                        {
                            "mention_id": row["mention_id"],
                            "source_tag": row["source_tag"],
                            "element": row["element"],
                            "parsed_oxidation_states": list(states),
                            "entry_evidence_grade": grade,
                            "owner_disposition": (
                                "rejected_extraction_false_positive"
                                if rejected_false_positive
                                else "quarantined_cross_dataset_disagreement"
                                if contested
                                else "unchanged"
                            ),
                        },
                    )
                )
                linked = {
                    observation_by_ox[(row["refcode"], row["element"], state)]
                    for state in states
                    if (row["refcode"], row["element"], state) in observation_by_ox
                }
                if linked and len(states) == 1:
                    observation_evidence.extend(
                        ObservationEvidence(observation_id, evidence_id, "reported_text")
                        for observation_id in sorted(linked)
                    )
                else:
                    claim_id = _key(
                        "csd-ox-claim", f"{release_id}:{row['mention_id']}"
                    )
                    state_text = ", ".join(
                        f"{state:+d}" for state in states
                    ) or "unparsed"
                    if single_key is not None:
                        claim_by_ox[single_key] = claim_id
                    claims.append(
                        Claim(
                            claim_id,
                            record_id_by_refcode[row["refcode"]],
                            (
                                oxidation_t1_sites.get(
                                    (row["refcode"], row["element"], states[0])
                                )
                                if len(states) == 1
                                else None
                            ) or entry_entity_by_refcode[row["refcode"]],
                            "oxidation_state",
                            (
                                f"rejected oxidation-state extraction for {row['element']}: "
                                f"{state_text}"
                                if rejected_false_positive
                                else f"reported oxidation-state candidate(s) for "
                                f"{row['element']}: {state_text}"
                            ),
                            "rejected" if rejected_false_positive else "unresolved",
                            (
                                EvidenceLevel.E0.value
                                if rejected_false_positive
                                else EvidenceLevel.E2.value
                                if contested
                                else EvidenceLevel.E1.value
                            ),
                            {
                                "source_tag": row["source_tag"],
                                "element": row["element"],
                                "parsed_oxidation_states": list(states),
                                "entry_evidence_grade": grade,
                                "benchmark_target_eligible": False
                                if rejected_false_positive or contested
                                else None,
                                "owner_disposition": (
                                    "rejected_extraction_false_positive"
                                    if rejected_false_positive
                                    else "quarantined_cross_dataset_disagreement"
                                    if contested
                                    else "unresolved_source_claim"
                                ),
                                **(
                                    {"cross_dataset_disagreement": known_conflicts[single_key]}
                                    if contested
                                    else {}
                                ),
                            },
                        )
                    )
                    claim_evidence.append(ClaimEvidence(claim_id, evidence_id, "verbatim_source"))
                    memberships.add(
                        (
                            subset_ids["oxidation-claims"],
                            record_id_by_refcode[row["refcode"]],
                        )
                    )
                    claim_memberships.add(
                        (subset_ids["oxidation-claims"], claim_id)
                    )

            disagreement_source_id = owner_audit_source_id or audit_source_id
            if disagreement_source_id is not None:
                for key, conflict in sorted(known_conflicts.items()):
                    observation_id = observation_by_ox.get(key)
                    claim_id = claim_by_ox.get(key)
                    if observation_id is None and claim_id is None:
                        continue
                    refcode, element, reported_state = key
                    evidence_id = _key(
                        "csd-crosscheck-evidence",
                        f"{release_id}:{refcode}:{element}:{reported_state}",
                    )
                    evidence.append(
                        EvidenceItem(
                            evidence_id,
                            disagreement_source_id,
                            "cross_dataset_disagreement",
                            f"quality-assessment/oxidation/{refcode}/{element}",
                            None,
                            (
                                f"CSD text {reported_state:+d}; comparison dataset "
                                f"{int(conflict['comparison_value']):+d}"
                            ),
                            dict(conflict),
                        )
                    )
                    if observation_id is not None:
                        observation_evidence.append(
                            ObservationEvidence(
                                observation_id,
                                evidence_id,
                                "known_conflict",
                            )
                        )
                    else:
                        claim_evidence.append(
                            ClaimEvidence(claim_id, evidence_id, "owner_adjudication")
                        )
            if owner_audit_source_id is not None:
                for refcode, element, rejected_state in sorted(false_positive_oxidation):
                    claim_id = claim_by_ox.get((refcode, element, rejected_state))
                    if claim_id is None:
                        continue
                    evidence_id = _key(
                        "csd-owner-rejection",
                        f"{release_id}:{refcode}:{element}:{rejected_state}",
                    )
                    evidence.append(
                        EvidenceItem(
                            evidence_id,
                            owner_audit_source_id,
                            "owner_extraction_adjudication",
                            f"evidence-regrade-audit/oxidation/{refcode}/{element}",
                            None,
                            None,
                            {
                                "status": "rejected_false_positive",
                                "extracted_value": rejected_state,
                                "audit_sha256": supplier_audit_sha256,
                            },
                        )
                    )
                    claim_evidence.append(
                        ClaimEvidence(claim_id, evidence_id, "owner_adjudication")
                    )

        if scope in {"all", "spin"}:
            spin_unqualified_candidates = {
                (row["refcode"], row["spin_label"])
                for row in connection.execute(
                    "SELECT v.* FROM v_sp_t1_active_unqualified_sites v "
                    "JOIN selected_refcodes r USING (refcode)"
                )
                if not context_by_refcode.get(row["refcode"])
            }
            for row in connection.execute(
                "SELECT m.* FROM spin_mentions m JOIN selected_refcodes r USING (refcode) "
                "ORDER BY m.mention_id"
            ):
                flags = context_by_refcode.get(row["refcode"], [])
                grade = spin_entry_grades.get(row["refcode"])
                evidence_id = _key(
                    "csd-spin-evidence", f"{release_id}:{row['mention_id']}"
                )
                evidence.append(
                    EvidenceItem(
                        evidence_id,
                        source_id,
                        "verbatim_database_field",
                        f"CSD:{row['refcode']}/{row['source_tag']}/spin-mention:{row['mention_id']}",
                        row["source_text"],
                        row["matched_text"],
                        {
                            "mention_id": row["mention_id"],
                            "source_tag": row["source_tag"],
                            "spin_label": row["spin_label"],
                            "entry_evidence_grade": grade,
                            "context_flags": flags,
                            "evidence_level": EvidenceLevel.E1.value,
                        },
                    )
                )
                spin_label = row["spin_label"]
                if spin_label in {"low_spin", "intermediate_spin", "high_spin"}:
                    property_type_id = "spin_category"
                    claim_text = f"reported spin-category candidate: {spin_label}"
                elif spin_label in {
                    "spin_crossover", "spin_transition", "spin_switching"
                }:
                    property_type_id = "spin_transition_behavior"
                    claim_text = f"reported spin-transition candidate: {spin_label}"
                else:
                    property_type_id = "spin_state_text"
                    claim_text = f"reported unresolved spin-state candidate: {spin_label}"
                claim_id = _key(
                    "csd-spin-claim", f"{release_id}:{row['mention_id']}"
                )
                curated_link = curated_spin_links.get(row["refcode"])
                claims.append(
                    Claim(
                        claim_id,
                        record_id_by_refcode[row["refcode"]],
                        entry_entity_by_refcode[row["refcode"]],
                        property_type_id,
                        claim_text,
                        "unresolved",
                        EvidenceLevel.E1.value,
                        {
                            "source_tag": row["source_tag"],
                            "spin_label": spin_label,
                            "entry_evidence_grade": grade,
                            "context_flags": flags,
                            "subject_mapping_status": "not_audited",
                            "not_spin_multiplicity": True,
                            "related_curated_structure": (
                                {
                                    "structure_id": curated_link["sco95_structure_id"],
                                    "dataset": "spin",
                                    "subset": "SCO-95",
                                    "owner_evidence_disposition": (
                                        "keep_CSD_claim_at_E1_and_link_curated_observation"
                                    ),
                                }
                                if curated_link is not None
                                else None
                            ),
                        },
                    )
                )
                claim_evidence.append(ClaimEvidence(claim_id, evidence_id, "verbatim_source"))
                subset_key = (
                    "spin-t1-active-unqualified-claims"
                    if (row["refcode"], spin_label) in spin_unqualified_candidates
                    else "spin-other-claims"
                )
                subset_id = subset_ids[subset_key]
                memberships.add((subset_id, record_id_by_refcode[row["refcode"]]))
                claim_memberships.add((subset_id, claim_id))

        populated_subsets = {subset_id for subset_id, _ in memberships}
        subsets = tuple(
            subset
            for subset in subsets
            if subset.quality_level == QualityLevel.B0.value
            or subset.subset_id in populated_subsets
        )

        return DatasetBundle(
            standard_version=STANDARD_VERSION,
            datasets=datasets,
            releases=releases,
            subsets=subsets,
            records=records,
            memberships=tuple(SubsetMembership(*row) for row in sorted(memberships)),
            observation_memberships=tuple(
                ObservationMembership(*row) for row in sorted(observation_memberships)
            ),
            claim_memberships=tuple(
                ClaimMembership(*row) for row in sorted(claim_memberships)
            ),
            entities=tuple(entities),
            external_identifiers=tuple(external_identifiers),
            record_entities=tuple(record_entities),
            entity_relations=tuple(entity_relations),
            assets=(),
            structures=(),
            structure_assets=(),
            descriptors=tuple(descriptors),
            property_types=property_types,
            observations=tuple(observations),
            observation_participants=(),
            sources=sources,
            evidence=tuple(evidence),
            observation_evidence=tuple(observation_evidence),
            claims=tuple(claims),
            claim_evidence=tuple(claim_evidence),
            claim_resolutions=(),
        )
    finally:
        connection.close()

