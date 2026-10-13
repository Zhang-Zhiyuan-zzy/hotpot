#!/usr/bin/env python3
"""Build the portable normalized SQLite electronic-state evidence database."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

from hotpot.ds._paths import AUDITS_ROOT, DATASET_ROOT, HOTPOT_PACKAGE_ROOT, SOURCES_ROOT


OX_T1 = "OX-T1_EXPLICIT_SINGLE_SITE_CANDIDATE"
SP_T1 = "SP-T1_FIXED_SINGLE_METAL_CANDIDATE"
ELECTRONIC_STATE_ROOT = SOURCES_ROOT / "csd-electronic-state"
DEFAULT_SCHEMA = DATASET_ROOT / "schemas" / "csd-electronic-state.sql"
DEFAULT_OUTPUT = ELECTRONIC_STATE_ROOT / "electronic_states.sqlite3"
DEFAULT_AUDIT_OUTPUT = AUDITS_ROOT / "csd-electronic-state-build-audit.json"
DEFAULT_PROVENANCE_OUTPUT = AUDITS_ROOT / "csd-electronic-state-build-provenance.json"

SPIN_CONTEXT_PATTERNS = {
    "population_or_isomer": re.compile(
        r"\b(?:majority|minority|mixture|fraction|isomer|metastable|ground state|partial|trapped)\b|\d+(?:\.\d+)?\s*%\b",
        re.I,
    ),
    "stimulus_or_condition": re.compile(
        r"\b(?:irradiat\w*|photo\w*|light|dark|temperature|pressure|heated|cooled|cooling|warming|room[- ]temperature)\b|\d+(?:\.\d+)?\s*K\b",
        re.I,
    ),
    "transition_language": re.compile(
        r"\b(?:transition|cross[ -]?over|conversion|switching)\b", re.I
    ),
}
NUMERIC_SPIN_PATTERN = re.compile(
    r"(?<![A-Za-z])S\s*=\s*[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?|"
    r"\b(?:multiplicity|singlet|doublet|triplet|quartet|quintet|sextet|septet|octet)\b",
    re.I,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(connection: sqlite3.Connection, sql: str, parameters=()):
    return connection.execute(sql, parameters).fetchone()[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-jsonl", type=Path, required=True)
    parser.add_argument("--site-jsonl", type=Path, required=True)
    parser.add_argument("--canonical-db", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--provenance-output", type=Path, default=DEFAULT_PROVENANCE_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        if not args.replace:
            raise FileExistsError(f"output exists: {args.output}")
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    connection = sqlite3.connect(args.output)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(args.schema.read_text(encoding="utf-8"))

    entry_count = 0
    oxidation_mention_id = 0
    spin_mention_id = 0
    spin_candidate_pairs: list[tuple[str, str]] = []
    with args.entry_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if args.limit is not None and entry_count >= args.limit:
                break
            record = json.loads(line)
            refcode = record["refcode"]
            connection.execute(
                "INSERT INTO entries VALUES(?,?,?,?,?)",
                (
                    refcode,
                    record["crystal_grade"],
                    record.get("formula_text"),
                    int(bool(record["oxidation_mentions"])),
                    int(bool(record["spin_mentions"])),
                ),
            )
            connection.executemany(
                "INSERT INTO entry_cores VALUES(?,?)",
                ((refcode, core_id) for core_id in record["core_ids"]),
            )
            by_element: dict[str, set[int]] = {}
            for mention in record["oxidation_mentions"]:
                oxidation_mention_id += 1
                element = mention["element"]
                states = set(mention["oxidation_state_candidates"])
                by_element.setdefault(element, set()).update(states)
                connection.execute(
                    "INSERT INTO oxidation_mentions VALUES(?,?,?,?,?,?)",
                    (
                        oxidation_mention_id,
                        refcode,
                        element,
                        mention["source_tag"],
                        mention["source_text"],
                        mention["matched_text"],
                    ),
                )
                connection.executemany(
                    "INSERT INTO oxidation_mention_states VALUES(?,?)",
                    ((oxidation_mention_id, state) for state in sorted(states)),
                )
            for element, grade in record["oxidation_grades_by_element"].items():
                connection.execute(
                    "INSERT INTO entry_oxidation_evidence VALUES(?,?,?)", (refcode, element, grade)
                )
                connection.executemany(
                    "INSERT INTO entry_oxidation_states VALUES(?,?,?)",
                    ((refcode, element, state) for state in sorted(by_element[element])),
                )
            if record["spin_grade"]:
                connection.execute(
                    "INSERT INTO entry_spin_evidence VALUES(?,?)", (refcode, record["spin_grade"])
                )
                spin_candidate_pairs.extend(
                    (refcode, site_id) for site_id in record["spin_candidate_site_ids"]
                )
            for mention in record["spin_mentions"]:
                spin_mention_id += 1
                connection.execute(
                    "INSERT INTO spin_mentions VALUES(?,?,?,?,?,?)",
                    (
                        spin_mention_id,
                        refcode,
                        mention["source_tag"],
                        mention["source_text"],
                        mention["matched_text"],
                        mention["spin_label"],
                    ),
                )
                for flag, pattern in SPIN_CONTEXT_PATTERNS.items():
                    for match in pattern.finditer(mention["source_text"]):
                        connection.execute(
                            "INSERT OR IGNORE INTO entry_spin_context_flags VALUES(?,?,?,?)",
                            (refcode, flag, mention["source_text"], match.group(0)),
                        )
            entry_count += 1
            if entry_count % 10_000 == 0:
                connection.commit()

    selected_refcodes = {row[0] for row in connection.execute("SELECT refcode FROM entries")}
    site_count = 0
    site_ids: set[str] = set()
    with args.site_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["refcode"] not in selected_refcodes:
                continue
            site_id = record["site_id"]
            site_ids.add(site_id)
            connection.execute(
                "INSERT INTO sites VALUES(?,?,?,?,?,?,?,?)",
                (
                    site_id,
                    record["refcode"],
                    record["core_id"],
                    record["representative_label"],
                    record["element"],
                    record["coordination_number"],
                    record["primary_core_class"],
                    record["core_classification_status"],
                ),
            )
            if record["oxidation_evidence_grade"]:
                connection.execute(
                    "INSERT INTO site_oxidation_evidence VALUES(?,?,?)",
                    (site_id, record["oxidation_evidence_grade"], record["oxidation_evidence_count"]),
                )
                connection.executemany(
                    "INSERT INTO site_oxidation_states VALUES(?,?)",
                    ((site_id, state) for state in record["oxidation_state_candidates"]),
                )
            if record["spin_evidence_grade"]:
                connection.execute(
                    "INSERT INTO site_spin_evidence VALUES(?,?,?)",
                    (site_id, record["spin_evidence_grade"], record["spin_evidence_count"]),
                )
                connection.executemany(
                    "INSERT INTO site_spin_labels VALUES(?,?)",
                    ((site_id, label) for label in record["spin_labels"]),
                )
            site_count += 1
            if site_count % 10_000 == 0:
                connection.commit()

    connection.executemany(
        "INSERT INTO entry_spin_candidate_sites(refcode,site_id) VALUES(?,?)",
        spin_candidate_pairs,
    )

    metadata = {
        "schema_version": "current",
        "dataset_name": "CSD electronic-state reported-text evidence",
        "created_at": started_at,
        "oxidation_t1_semantics": (
            "Explicit unique integer state in chemical-name text plus exactly one retained site orbit "
            "for that element; unique localization candidate, not independent validation."
        ),
        "spin_t1_semantics": (
            "One fixed high/low/intermediate-spin text category plus exactly one plausible spin-capable "
            "metal-site candidate; categorical, not numerical S, and not proof of metal-local spin."
        ),
        "source_entry_jsonl_sha256": sha256_file(args.entry_jsonl),
        "source_site_jsonl_sha256": sha256_file(args.site_jsonl),
    }
    connection.executemany("INSERT INTO metadata VALUES(?,?)", sorted(metadata.items()))
    connection.commit()

    canonical = sqlite3.connect(f"file:{args.canonical_db.resolve().as_posix()}?mode=ro", uri=True)
    canonical_sites = {
        row[0]: row[1:]
        for row in canonical.execute(
            """
            SELECT s.site_id,s.refcode,s.core_id,s.element,s.coordination_number,
                   x.primary_class,x.classification_status,e.crystal_grade,e.status
            FROM metal_sites s
            JOIN core_classifications x USING(core_id)
            JOIN entries e USING(refcode)
            """
        )
    }
    canonical_entries = {
        row[0]: (row[1], row[2])
        for row in canonical.execute("SELECT refcode,crystal_grade,status FROM entries")
    }
    missing_sites = site_ids - canonical_sites.keys()
    mismatched_sites = 0
    nonaccepted_entries = {
        refcode for refcode in selected_refcodes
        if canonical_entries.get(refcode, (None, None))[1] != "accepted_metal_extracted"
    }
    crystal_grade_mismatches = 0
    for refcode, grade in connection.execute("SELECT refcode,crystal_grade FROM entries"):
        if canonical_entries.get(refcode, (None, None))[0] != grade:
            crystal_grade_mismatches += 1
    for row in connection.execute(
        "SELECT site_id,refcode,core_id,element,coordination_number,primary_core_class,core_classification_status FROM sites"
    ):
        canonical_row = canonical_sites.get(row[0])
        if canonical_row is None:
            continue
        if tuple(row[1:]) != tuple(canonical_row[:6]):
            mismatched_sites += 1

    sp_t1_source_texts = [
        row[0]
        for row in connection.execute(
            """
            SELECT DISTINCT m.source_text FROM spin_mentions m
            JOIN entry_spin_evidence e USING(refcode) WHERE e.grade=?
            """,
            (SP_T1,),
        )
    ]
    numeric_spin_text_count = sum(bool(NUMERIC_SPIN_PATTERN.search(text)) for text in sp_t1_source_texts)
    checks = {
        "quick_check": scalar(connection, "PRAGMA quick_check"),
        "foreign_key_violation_count": len(connection.execute("PRAGMA foreign_key_check").fetchall()),
        "entry_count": scalar(connection, "SELECT COUNT(*) FROM entries"),
        "site_count": scalar(connection, "SELECT COUNT(*) FROM sites"),
        "missing_canonical_site_count": len(missing_sites),
        "canonical_site_field_mismatch_count": mismatched_sites,
        "nonaccepted_entry_count": len(nonaccepted_entries),
        "crystal_grade_mismatch_count": crystal_grade_mismatches,
        "ox_t1_sites": scalar(connection, "SELECT COUNT(*) FROM v_ox_t1_sites"),
        "ox_t1_entries": scalar(connection, "SELECT COUNT(DISTINCT refcode) FROM v_ox_t1_sites"),
        "ox_t1_cores": scalar(connection, "SELECT COUNT(DISTINCT core_id) FROM v_ox_t1_sites"),
        "sp_t1_sites": scalar(connection, "SELECT COUNT(*) FROM v_sp_t1_sites"),
        "sp_t1_entries": scalar(connection, "SELECT COUNT(DISTINCT refcode) FROM v_sp_t1_sites"),
        "sp_t1_cores": scalar(connection, "SELECT COUNT(DISTINCT core_id) FROM v_sp_t1_sites"),
        "ox_t1_active_sites": scalar(connection, "SELECT COUNT(*) FROM v_ox_t1_active_sites"),
        "ox_t1_active_entries": scalar(connection, "SELECT COUNT(DISTINCT refcode) FROM v_ox_t1_active_sites"),
        "ox_t1_active_cores": scalar(connection, "SELECT COUNT(DISTINCT core_id) FROM v_ox_t1_active_sites"),
        "sp_t1_active_sites": scalar(connection, "SELECT COUNT(*) FROM v_sp_t1_active_sites"),
        "sp_t1_active_entries": scalar(connection, "SELECT COUNT(DISTINCT refcode) FROM v_sp_t1_active_sites"),
        "sp_t1_context_qualified_entries": scalar(
            connection,
            "SELECT COUNT(DISTINCT f.refcode) FROM entry_spin_context_flags f JOIN v_sp_t1_sites p USING(refcode)",
        ),
        "sp_t1_active_unqualified_sites": scalar(
            connection, "SELECT COUNT(*) FROM v_sp_t1_active_unqualified_sites"
        ),
        "sp_t1_active_unqualified_entries": scalar(
            connection, "SELECT COUNT(DISTINCT refcode) FROM v_sp_t1_active_unqualified_sites"
        ),
        "joint_t1_sites": scalar(connection, "SELECT COUNT(*) FROM v_joint_t1_sites"),
        "ox_t1_invalid_state_cardinality": scalar(
            connection,
            "SELECT COUNT(*) FROM (SELECT site_id,COUNT(*) n FROM v_ox_t1_sites GROUP BY site_id HAVING n<>1)",
        ),
        "sp_t1_invalid_label_cardinality": scalar(
            connection,
            "SELECT COUNT(*) FROM (SELECT site_id,COUNT(*) n FROM v_sp_t1_sites GROUP BY site_id HAVING n<>1)",
        ),
        "sp_t1_source_text_count": len(sp_t1_source_texts),
        "sp_t1_numeric_s_or_multiplicity_text_count": numeric_spin_text_count,
    }
    checks["passed"] = (
        checks["quick_check"] == "ok"
        and checks["foreign_key_violation_count"] == 0
        and checks["missing_canonical_site_count"] == 0
        and checks["canonical_site_field_mismatch_count"] == 0
        and checks["nonaccepted_entry_count"] == 0
        and checks["crystal_grade_mismatch_count"] == 0
        and checks["ox_t1_invalid_state_cardinality"] == 0
        and checks["sp_t1_invalid_label_cardinality"] == 0
    )
    canonical.close()

    oxidation_distribution = Counter(
        (row[0], row[1]) for row in connection.execute("SELECT element,oxidation_state FROM v_ox_t1_sites")
    )
    spin_distribution = Counter(
        (row[0], row[1]) for row in connection.execute("SELECT element,spin_label FROM v_sp_t1_sites")
    )
    audit = {
        "schema_version": "current",
        "conclusion": {
            "oxidation_t1": (
                "Counts are confirmed for explicit unique text values and unique retained site-orbit candidates; "
                "they are not independently verified oxidation states."
            ),
            "spin_t1": (
                "Counts are confirmed for fixed categorical text and a unique plausible metal candidate; "
                "they are not numerical spin values and do not prove metal-local spin."
            ),
        },
        "checks": checks,
        "oxidation_t1_distribution": [
            {"element": key[0], "oxidation_state": key[1], "site_count": count}
            for key, count in sorted(oxidation_distribution.items())
        ],
        "spin_t1_distribution": [
            {"element": key[0], "spin_label": key[1], "site_count": count}
            for key, count in sorted(spin_distribution.items())
        ],
    }
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    connection.execute("PRAGMA optimize")
    connection.execute("VACUUM")
    connection.close()
    completed_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=HOTPOT_PACKAGE_ROOT.parent, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    provenance = {
        "schema_version": "current",
        "run_id": "electronic_state_database_build",
        "started_at": started_at,
        "completed_at": completed_at,
        "code_revision": revision,
        "command": [str(Path(__file__).resolve()), *(__import__("sys").argv[1:])],
        "inputs": {
            "entry_jsonl": str(args.entry_jsonl.resolve()),
            "site_jsonl": str(args.site_jsonl.resolve()),
            "canonical_db": str(args.canonical_db.resolve()),
            "schema": str(args.schema.resolve()),
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "audit": str(args.audit_output.resolve()),
        "checks": checks,
    }
    args.provenance_output.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, sort_keys=True))
    return 0 if checks["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
