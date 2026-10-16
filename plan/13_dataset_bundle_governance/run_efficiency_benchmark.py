#!/usr/bin/env python3
"""Run one reproducible Hotpot dataset lifecycle performance trial.

Run this script in separate detached worktrees to compare implementations.  It
prints one JSON object and keeps input generation and reliability checks outside
the measured lifecycle phases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import hotpot
from hotpot.ds.public._assessment import canonical_json
from hotpot.ds.public._bundle_state import BUNDLE_SCHEMA
from hotpot.ds.public.bundle import Bundle
from hotpot.ds.public.certificate import verify_bundle_certificate
from hotpot.ds.storage import TargetStore


def _case(index: int) -> dict[str, Any]:
    case_id = f"case-{index:08d}"
    return {
        "id": case_id,
        "local_id": f"local-{index:08d}",
        "public_index": index,
        "identifiers": {"benchmark_id": f"BENCH-{index:08d}"},
        "core_info": {"sample": case_id},
        "observations": [
            {
                "observation_id": f"observation-{index:08d}",
                "property": "spin",
                "value": index % 7,
            }
        ],
        "structures": [
            {
                "structure_id": f"structure-{index:08d}",
                "role": "primary",
                "format": "smi",
                "path": "structures/shared.smi",
                "primary": True,
            }
        ],
    }


def _prepare(root: Path, count: int) -> None:
    intake = root / "evidence" / "intake" / "cases.jsonl"
    intake.parent.mkdir(parents=True)
    with intake.open("w", encoding="utf-8") as stream:
        for index in range(count):
            stream.write(canonical_json(_case(index)) + "\n")
    structures = root / "structures"
    structures.mkdir()
    (structures / "shared.smi").write_text("O\n", encoding="utf-8")
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ds_name": "efficiency-benchmark",
        "release_id": "efficiency-benchmark.v001",
        "release_serial": 1,
        "version": "001",
        "title": "Dataset efficiency benchmark",
        "description": "Synthetic lifecycle performance fixture",
        "license": "CC0-1.0",
        "access_policy": "open",
        "provenance": {"created_by": "run_efficiency_benchmark.py"},
        "core_labels": ["spin"],
        "core_info": ["sample"],
        "intake": {"format": "jsonl", "path": "evidence/intake/cases.jsonl"},
        "identity_schema": {
            "components": [
                {
                    "key": "sample",
                    "path": "core_info.sample",
                    "required": True,
                    "normalizer": "text",
                }
            ]
        },
        "required_structure_roles": ["primary"],
    }
    (root / "bundle.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )


def _measure(action: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    result = action()
    return result, {
        "wall_seconds": (time.perf_counter_ns() - wall_start) / 1_000_000_000,
        "cpu_seconds": (time.process_time_ns() - cpu_start) / 1_000_000_000,
    }


def _semantic_result(database: Path, edits: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    store = TargetStore(database)
    case_count = 0
    first_note = None
    for case in store.iter_cases(
        "efficiency-benchmark",
        "efficiency-benchmark.v001",
        include_incomplete=True,
    ):
        facts = case["facts"]
        assessment = case["assessment"]
        observations = facts.get("observations", [])
        value = {
            "case_id": case["id"],
            "core_info": facts.get("core_info", {}),
            "observations": [
                [item.get("property"), item.get("value")]
                for item in observations
                if item.get("status", "active") == "active"
            ],
            "quality_level": assessment.get("quality_level"),
            "evidence_level": assessment.get("evidence_level"),
            "archive_class": assessment.get("archive_class"),
            "benchmark": bool(assessment.get("benchmark")),
        }
        encoded = canonical_json(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        if case_count == 0:
            first_note = facts.get("core_info", {}).get("perf_note")
        case_count += 1
    with sqlite3.connect(database) as connection:
        assessment_count = connection.execute(
            "SELECT COUNT(*) FROM case_assessments "
            "WHERE dataset_id='efficiency-benchmark' "
            "AND release_id='efficiency-benchmark.v001'"
        ).fetchone()[0]
        structure_assets = connection.execute(
            "SELECT COUNT(DISTINCT content_sha256) FROM structure_asset_refs "
            "WHERE dataset_id='efficiency-benchmark' "
            "AND release_id='efficiency-benchmark.v001'"
        ).fetchone()[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    if first_note != edits - 1:
        raise AssertionError("the measured Case mutation was not persisted")
    if integrity != "ok" or foreign_keys:
        raise AssertionError("target-store integrity validation failed")
    return {
        "case_count": case_count,
        "assessment_count": assessment_count,
        "unique_structure_assets": structure_assets,
        "semantic_sha256": digest.hexdigest(),
        "integrity_check": integrity,
        "foreign_key_violations": len(foreign_keys),
    }


def run(count: int, edits: int, label: str, keep_root: bool) -> dict[str, Any]:
    trial_root = Path(tempfile.mkdtemp(prefix="hotpot-ds-efficiency-"))
    bundle_root = trial_root / "bundle"
    database = trial_root / "datasets.sqlite3"
    _prepare(bundle_root, count)
    try:
        bundle, build = _measure(
            lambda: Bundle(bundle_root, "efficiency-benchmark", db=database)
        )
        target = bundle.cases[0]

        def mutate() -> None:
            for index in range(edits):
                target.set_core_info("perf_note", index)

        _, mutation = _measure(mutate)
        sync_result, synchronize = _measure(bundle.synchronize)
        _, certify = _measure(bundle.write_certificate)
        ingest_result, ingest = _measure(
            lambda: bundle.ingest(
                include_incomplete=True,
                reason="Synthetic performance validation",
            )
        )
        verify_bundle_certificate(bundle)
        semantic = _semantic_result(database, edits)
        if ingest_result["included_count"] != count:
            raise AssertionError("ingestion did not include every synthetic Case")
        if ingest_result["excluded_count"] != 0:
            raise AssertionError("ingestion unexpectedly excluded synthetic Cases")
        return {
            "schema": "hotpot-ds-efficiency-benchmark/current",
            "label": label,
            "hotpot_module": str(Path(hotpot.__file__).resolve()),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cases": count,
            "edits": edits,
            "timings": {
                "build": build,
                "mutation": mutation,
                "synchronize": synchronize,
                "certify": certify,
                "ingest": ingest,
            },
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "generation_id": sync_result["generation_id"],
            "receipt_digest": ingest_result["receipt_digest"],
            "semantic": semantic,
            "trial_root": str(trial_root) if keep_root else None,
        }
    finally:
        if not keep_root:
            shutil.rmtree(trial_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="number of Cases")
    parser.add_argument("--edits", type=int, default=8)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-root", action="store_true")
    args = parser.parse_args()
    result = run(args.n, args.edits, args.label, args.keep_root)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
