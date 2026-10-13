#!/usr/bin/env python3
"""Rebuild the one current local dataset store through the public quality gate."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Sequence

from hotpot import ds
from hotpot.ds._paths import AUDITS_ROOT, DATA_ROOT, DEFAULT_STORE_PATH, SOURCES_ROOT
from hotpot.ds.adapters import build_curated_dataset, convert_csd_electronic_state
from hotpot.ds.public.certificate import CONFIGURE_VERSION


DATASET_NAMES = ("oxidation", "spin", "csd-electronic-state")
CORE_LABELS = {
    "oxidation": ["oxidation_state"],
    "spin": ["spin_multiplicity"],
    "csd-electronic-state": [],
}
DATASET_RULES = {
    "csd-electronic-state": {"require_structure_or_label": False},
}


def _bundle(name: str):
    if name in {"oxidation", "spin"}:
        return build_curated_dataset(name)
    return convert_csd_electronic_state(
        SOURCES_ROOT / "csd-electronic-state" / "electronic_states.sqlite3",
        license_name="CSD-restricted",
        quality_assessment_path=AUDITS_ROOT / "csd-electronic-state-quality.json",
        evidence_audit_path=AUDITS_ROOT / "evidence_regrade_audit.json",
    )


def rebuild_store(
    destination: Path = DEFAULT_STORE_PATH,
    datasets: Iterable[str] = DATASET_NAMES,
) -> dict:
    """Build, certify, validate, and atomically install a current store."""

    selected = tuple(datasets)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    building = destination.with_name(f".{destination.name}.building")
    if building.exists():
        building.unlink()

    certificate_root = destination.parent / "certificates"
    certificate_root.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for name in selected:
            internal = _bundle(name)
            bundle = ds.load_bundle(internal, asset_root=DATA_ROOT)
            certificate = certificate_root / f"{name}.certificate.json"
            ds.write_certificate(
                bundle,
                certificate,
                core_labels=CORE_LABELS[name],
                configure={
                    "schema_version": CONFIGURE_VERSION,
                    "generic": {},
                    "datasets": {name: DATASET_RULES.get(name, {})},
                },
            )
            results.append(
                ds.store.ingest(bundle, certificate=certificate, db=building)
            )

        connection = sqlite3.connect(str(building))
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("datasets", "dataset_releases", "records", "structures", "observations", "claims")
            }
        finally:
            connection.close()
        if integrity != "ok":
            raise RuntimeError(f"rebuilt store failed SQLite integrity_check: {integrity}")
        os.replace(building, destination)
    finally:
        if building.exists():
            building.unlink()

    return {
        "database": str(destination),
        "datasets": list(selected),
        "integrity_check": integrity,
        "counts": counts,
        "ingestions": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=None,
        metavar="DATASET",
        help="datasets to include; defaults to all current local sources",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_STORE_PATH,
        help=f"output SQLite store (default: {DEFAULT_STORE_PATH})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected = DATASET_NAMES if not args.datasets else args.datasets
    unknown = sorted(set(selected) - set(DATASET_NAMES))
    if unknown:
        parser.error(f"unknown datasets: {', '.join(unknown)}")
    print(json.dumps(rebuild_store(args.database, selected), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
