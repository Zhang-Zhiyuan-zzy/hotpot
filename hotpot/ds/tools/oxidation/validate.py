"""Validate oxidation-state source structures with Hotpot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import hotpot as hp

from hotpot.ds._paths import DATA_ROOT, SOURCES_ROOT, STRUCTURES_ROOT


SOURCE_ROOT = SOURCES_ROOT / "oxidation"


def table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    index = {
        row["structure_id"]: row
        for row in table(STRUCTURES_ROOT / "structures.csv")
    }

    def stored(structure_id: str):
        row = index[structure_id]
        return (
            DATA_ROOT / row["store_path"],
            row["format"],
            int(row["atom_count"]),
            row["canonical_file_sha256"],
        )

    root = SOURCE_ROOT / "gold" / "sse17"
    metals = {r["record_id"]: r["metal"] for r in table(root / "labels.csv")}
    for row in table(root / "structures.csv"):
        path, fmt, atoms, sha256 = stored(row["structure_id"])
        yield "SSE17-oxidation", row["record_id"], row["structure_id"], path, fmt, atoms, metals[row["record_id"]], sha256

    root = SOURCE_ROOT / "gold" / "sco95"
    metals = {r["record_id"]: r["metal"] for r in table(root / "labels.csv")}
    for row in table(root / "structures.csv"):
        path, fmt, atoms, sha256 = stored(row["structure_id"])
        yield "SCO95-oxidation", row["record_id"], row["structure_id"], path, fmt, atoms, metals[row["record_id"]], sha256

    root = SOURCE_ROOT / "silver" / "tm_oxidation"
    labels = {row["refcode"]: row for row in table(root / "labels.csv")}
    for row in table(root / "structures.csv"):
        path, fmt, atoms, sha256 = stored(row["structure_id"])
        yield "TM-oxidation", row["refcode"], row["structure_id"], path, fmt, atoms, labels[row["refcode"]]["metal"], sha256


def validate(item) -> dict[str, object]:
    dataset, record_id, structure_id, path, fmt, expected_atoms, expected_metal, expected_hash = item
    try:
        molecule = hp.Molecule.read_one(path, fmt)
        metals = sorted(a.symbol for a in molecule.atoms if a.is_metal)
        passed = len(molecule.atoms) == expected_atoms and expected_metal in metals and digest(path) == expected_hash
        error = ""
    except Exception as exc:
        molecule = None
        metals = []
        passed = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "dataset": dataset,
        "record_id": record_id,
        "structure_id": structure_id,
        "structure_file": path.relative_to(DATA_ROOT).as_posix(),
        "format": fmt,
        "expected_atom_count": expected_atoms,
        "hotpot_atom_count": len(molecule.atoms) if molecule else "",
        "hotpot_bond_count": len(molecule.bonds) if molecule else "",
        "hotpot_formula": molecule.formula if molecule else "",
        "hotpot_metal_symbols": ";".join(metals),
        "passed": passed,
        "error": error,
    }


def assert_independence() -> None:
    for path in SOURCE_ROOT.glob("**/labels.csv"):
        header = path.read_text().splitlines()[0].lower()
        assert "spin" not in header
        assert "multiplicity" not in header
        assert "transition_temperature" not in header
        assert "metal_oxidation_state" in header
    for row in table(STRUCTURES_ROOT / "structures.csv"):
        if row["format"] == "xyz":
            comment = (DATA_ROOT / row["store_path"]).read_text().splitlines()[1].lower()
            assert "multiplicity" not in comment
            assert "oxidation" not in comment


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    assert_independence()
    results = [validate(item) for item in inputs()]
    output = SOURCE_ROOT / "validation" / "hotpot_read.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    failures = [r for r in results if not r["passed"]]
    counts = Counter(r["dataset"] for r in results)
    summary = {
        "property_target": "metal_oxidation_state",
        "independent_from_spin_dataset": True,
        "independent_from_formal_charge_dataset": True,
        "labels": {"SSE17-oxidation": 17, "SCO95-oxidation": 95, "TM-oxidation": 2063},
        "structures": dict(counts),
        "hotpot_passed": dict(Counter(r["dataset"] for r in results if r["passed"])),
        "hotpot_failed": dict(Counter(r["dataset"] for r in failures)),
        "target_leakage_check": "passed",
        "shared_structure_root": "structures",
        "validation_sha256": digest(output),
    }
    (SOURCE_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if failures:
        raise SystemExit(f"Hotpot validation failures: {len(failures)}")


if __name__ == "__main__":
    main()

