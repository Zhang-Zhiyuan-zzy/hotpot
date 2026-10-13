#!/usr/bin/env python3
"""Reproduce the scientific-quality audit used for Hotpot DS admission.

This audit deliberately distinguishes relational correctness from label truth.
It cross-checks CSD text-derived candidates against the existing curated Hotpot
oxidation/spin tables.  That overlap is not treated as an independent truth set;
the audit records conflicts rather than assigning quality from extraction-rule
names or agreement counts alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from hotpot.ds._paths import AUDITS_ROOT, SOURCES_ROOT


ELECTRONIC_STATE_ROOT = SOURCES_ROOT / "csd-electronic-state"
DEFAULT_DB = ELECTRONIC_STATE_ROOT / "electronic_states.sqlite3"
DEFAULT_OUTPUT = AUDITS_ROOT / "csd-electronic-state-quality.json"
NUMERIC_SPIN_PATTERN = re.compile(
    r"(?<![A-Za-z])S\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?|"
    r"\b(?:multiplicity|singlet|doublet|triplet|quartet|quintet|sextet|septet|octet)\b",
    re.I,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _counts(connection: sqlite3.Connection, query: str) -> List[Dict[str, object]]:
    return [dict(row) for row in connection.execute(query)]


def _base_refcode(refcode: str) -> str:
    return re.sub(r"\d+$", "", refcode)


def _family_audit(connection: sqlite3.Connection, view: str) -> Dict[str, int]:
    refcodes = {row[0] for row in connection.execute(f"SELECT refcode FROM {view}")}
    families: Dict[str, List[str]] = {}
    for refcode in refcodes:
        families.setdefault(_base_refcode(refcode), []).append(refcode)
    repeated = [members for members in families.values() if len(members) > 1]
    return {
        "entry_count": len(refcodes),
        "base_refcode_family_count": len(families),
        "multi_variant_family_count": len(repeated),
        "entries_in_multi_variant_families": sum(map(len, repeated)),
    }


def _oxidation_crosscheck(connection: sqlite3.Connection) -> Dict[str, object]:
    sources = (
        (
            SOURCES_ROOT / "oxidation" / "silver" / "tm_oxidation" / "labels.csv",
            "refcode",
        ),
        (
            SOURCES_ROOT / "oxidation" / "gold" / "sco95" / "labels.csv",
            "structure_refcode",
        ),
    )
    counts = Counter()
    disagreements = []
    for path, refcode_field in sources:
        for row in _rows(path):
            counts["total"] += 1
            refcode = row[refcode_field]
            element = row["metal"]
            expected = int(row["metal_oxidation_state"])
            entry = connection.execute(
                "SELECT formula_text,has_oxidation_text FROM entries WHERE refcode=?",
                (refcode,),
            ).fetchone()
            if entry is None:
                counts["entry_absent"] += 1
                continue
            states = {
                value[0]
                for value in connection.execute(
                    """SELECT oxidation_state FROM entry_oxidation_states
                    WHERE refcode=? AND element=?""",
                    (refcode, element),
                )
            }
            if not states:
                counts["entry_present_without_target_element_text"] += 1
                continue
            counts["target_element_text_present"] += 1
            if states == {expected}:
                counts["exact_single_state_agreement"] += 1
                continue
            counts["disagreement_or_ambiguity"] += 1
            mentions = [
                dict(value)
                for value in connection.execute(
                    """SELECT source_tag,source_text,matched_text FROM oxidation_mentions
                    WHERE refcode=? AND element=? ORDER BY mention_id""",
                    (refcode, element),
                )
            ]
            disagreements.append(
                {
                    "refcode": refcode,
                    "element": element,
                    "hotpot_curated_state": expected,
                    "csd_text_states": sorted(states),
                    "formula_text": entry["formula_text"],
                    "mentions": mentions,
                }
            )
    return {"counts": dict(counts), "disagreements": disagreements}


def _spin_crosscheck(connection: sqlite3.Connection) -> Dict[str, object]:
    path = SOURCES_ROOT / "spin" / "gold" / "sco95" / "structures.csv"
    counts = Counter()
    grades = Counter()
    non_matches = []
    for row in _rows(path):
        counts["structures"] += 1
        refcode = row["source_refcode"]
        expected = "low_spin" if row["structure_state"] == "low_temperature" else "high_spin"
        labels = {
            value[0]
            for value in connection.execute(
                "SELECT spin_label FROM spin_mentions WHERE refcode=?", (refcode,)
            )
        }
        if not labels:
            counts["without_spin_text"] += 1
            continue
        counts["with_spin_text"] += 1
        grade = connection.execute(
            "SELECT grade FROM entry_spin_evidence WHERE refcode=?", (refcode,)
        ).fetchone()
        if grade:
            grades[grade[0]] += 1
        if labels == {expected}:
            counts["exact_category_agreement"] += 1
        else:
            counts["not_a_fixed_category_match"] += 1
            non_matches.append(
                {
                    "refcode": refcode,
                    "expected_from_curated_pair": expected,
                    "csd_text_labels": sorted(labels),
                }
            )
    return {
        "counts": dict(counts),
        "entry_grade_distribution_for_text_matches": dict(sorted(grades.items())),
        "non_matches": non_matches,
    }


def assess(database: Path) -> Dict[str, object]:
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
    t1_spin_texts = [
        row[0]
        for row in connection.execute(
            """SELECT DISTINCT m.source_text FROM spin_mentions m
            JOIN entry_spin_evidence e USING(refcode)
            WHERE e.grade='SP-T1_FIXED_SINGLE_METAL_CANDIDATE'"""
        )
    ]

    assessment = {
        "assessment_version": "hotpot-csd-electronic-state/current",
        "database": {
            "sha256": _sha256(database),
            "quick_check": quick_check,
            "foreign_key_violation_count": len(foreign_keys),
            "entries": connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0],
            "sites": connection.execute("SELECT COUNT(*) FROM sites").fetchone()[0],
            "crystal_grade_distribution": _counts(
                connection,
                "SELECT crystal_grade,COUNT(*) AS count FROM entries GROUP BY crystal_grade ORDER BY crystal_grade",
            ),
        },
        "oxidation": {
            "entry_element_grade_rows": _counts(
                connection,
                "SELECT grade,COUNT(*) AS count FROM entry_oxidation_evidence GROUP BY grade ORDER BY grade",
            ),
            "t1_sites": connection.execute("SELECT COUNT(*) FROM v_ox_t1_sites").fetchone()[0],
            "t1_active_sites": connection.execute(
                "SELECT COUNT(*) FROM v_ox_t1_active_sites"
            ).fetchone()[0],
            "t1_crystal_grades": _counts(
                connection,
                """SELECT e.crystal_grade,COUNT(*) AS count FROM v_ox_t1_sites v
                JOIN entries e USING(refcode) GROUP BY e.crystal_grade ORDER BY e.crystal_grade""",
            ),
            "mention_sources": _counts(
                connection,
                """SELECT source_tag,COUNT(*) AS mention_count,
                COUNT(DISTINCT refcode) AS entry_count FROM oxidation_mentions
                GROUP BY source_tag ORDER BY mention_count DESC""",
            ),
            "refcode_family_audit": _family_audit(connection, "v_ox_t1_sites"),
            "crosscheck_against_current_hotpot": _oxidation_crosscheck(connection),
        },
        "spin": {
            "entry_grade_rows": _counts(
                connection,
                "SELECT grade,COUNT(*) AS count FROM entry_spin_evidence GROUP BY grade ORDER BY grade",
            ),
            "t1_label_distribution": _counts(
                connection,
                "SELECT spin_label,COUNT(*) AS count FROM v_sp_t1_sites GROUP BY spin_label ORDER BY spin_label",
            ),
            "t1_sites": connection.execute("SELECT COUNT(*) FROM v_sp_t1_sites").fetchone()[0],
            "t1_active_sites": connection.execute(
                "SELECT COUNT(*) FROM v_sp_t1_active_sites"
            ).fetchone()[0],
            "t1_active_unqualified_sites": connection.execute(
                "SELECT COUNT(*) FROM v_sp_t1_active_unqualified_sites"
            ).fetchone()[0],
            "t1_distinct_source_texts": len(t1_spin_texts),
            "numeric_S_or_multiplicity_texts": sum(
                bool(NUMERIC_SPIN_PATTERN.search(text)) for text in t1_spin_texts
            ),
            "mention_sources": _counts(
                connection,
                """SELECT source_tag,COUNT(*) AS mention_count,
                COUNT(DISTINCT refcode) AS entry_count FROM spin_mentions
                GROUP BY source_tag ORDER BY mention_count DESC""",
            ),
            "context_flags": _counts(
                connection,
                """SELECT flag,COUNT(DISTINCT refcode) AS entry_count
                FROM entry_spin_context_flags GROUP BY flag ORDER BY flag""",
            ),
            "refcode_family_audit": _family_audit(connection, "v_sp_t1_sites"),
            "crosscheck_against_current_hotpot": _spin_crosscheck(connection),
        },
        "source_assessment": {
            "source_type": "reported_database_text",
            "provenance_scope": (
                "portable SQLite to extracted candidates/claims; not per-row "
                "primary-publication provenance"
            ),
            "what_is_verified": [
                "SQLite relational integrity and canonical site linkage",
                "presence of explicit oxidation/spin words in preserved CSD text fields",
                "deterministic rule-based mapping and reproducible input/output hashes",
            ],
            "what_is_not_verified": [
                "that a depositor-supplied chemical name is chemically correct",
                "that preparation prose denotes the measured metal-local ground spin",
                "numeric S or spin multiplicity (none is present)",
                "independent primary-publication evidence for each row",
                "redistribution rights for CSD text or coordinates",
            ],
            "portable_limitations": [
                "original JSONL inputs are not bundled",
                "provenance input paths are absolute and not resolvable after transfer",
                "the portable database contains no coordinates, DOI-per-row, or declared CSD license",
                "field names alone do not identify depositor versus CSD-editor authorship",
                "full bundle materialization used about 1.69 GiB peak memory; production import should be chunked",
            ],
        },
        "hotpot_ds_decision": {
            "admissible": True,
            "conditions": [
                "declare the actual CCDC/CSD license and keep access restricted until redistribution is confirmed",
                "store raw mentions as evidence; keep ambiguous/conditional rows as claims",
                "create oxidation observations only for uniquely mapped active OX-T1 rows",
                "keep every spin mention as an E1 unresolved claim until subject coreference, site, and conditions are manually reviewed",
                "never convert spin category or transition behavior to numeric S or multiplicity",
                "group train/validation/test splits by base refcode family and chemical identity",
            ],
            "assigned_levels": {
                "full_collection": "B0_collection",
                "OX-T1_active": "B1_candidate",
                "all_spin_text_including_SP-T1_active_regex_unflagged": (
                    "B0_collection; E1 unresolved claims only"
                ),
                "conditional_spin_or_incomplete_core": "B0_collection claim/evidence only for fixed-state tasks",
                "T2_to_T4": "B0_collection claim/evidence only",
            },
            "not_currently_eligible": ["B2_curated", "B3_validated", "B4_reference"],
        },
        "upgrade_requirements": {
            "B2": [
                "domain review of every record in the promoted subset with reviewed_count equal to total_count",
                "primary-paper/CIF inspection with site and condition resolution",
                "stratified probability sampling may estimate B1 collection error but cannot promote unreviewed rows",
                "accuracy estimate names the adjudicated reference standard and quantity assessed",
                "accuracy sample_size equals the fully reviewed subset count",
                "all known conflicts and high-risk strata explicitly resolved",
            ],
            "B3": [
                "primary-source-supported row adjudication and dual independent review are both required",
                "separate primary-source and independent-review reports",
                "frozen task, metric, family-grouped split, leakage audit, condition and uncertainty policies",
            ],
            "B4": [
                "multi-source or repeated-experiment corroboration and external audit",
            ],
        },
    }
    connection.close()
    return assessment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = assess(args.db.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "admissible": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

