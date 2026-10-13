"""Audit metal and ligand composition of oxidation and spin datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import hotpot as hp

from hotpot.ds._paths import DATA_ROOT, SOURCES_ROOT, STRUCTURES_ROOT


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_metals() -> dict[tuple[str, str, str], str]:
    result: dict[tuple[str, str, str], str] = {}
    tables = [
        ("oxidation", "SSE17-oxidation", "oxidation/gold/sse17/labels.csv", "record_id"),
        ("oxidation", "SCO95-oxidation", "oxidation/gold/sco95/labels.csv", "record_id"),
        ("oxidation", "TM-oxidation", "oxidation/silver/tm_oxidation/labels.csv", "refcode"),
        ("spin", "SSE17", "spin/gold/sse17/labels.csv", "record_id"),
        ("spin", "SCO-95", "spin/gold/sco95/labels.csv", "pair_id"),
        ("spin", "TM-GSspin", "spin/silver/tm_gsspin/labels.csv", "refcode"),
    ]
    for dataset, subset, relative, id_column in tables:
        for row in read_csv(SOURCES_ROOT / relative):
            result[(dataset, subset, row[id_column])] = row["metal"]
    return result


def connected_components(adjacency: list[set[int]], removed: int) -> list[set[int]]:
    unseen = set(range(len(adjacency))) - {removed}
    components = []
    while unseen:
        stack = [unseen.pop()]
        component = set(stack)
        while stack:
            for neighbor in adjacency[stack.pop()] - {removed}:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def analyze(path: Path, fmt: str, target_symbol: str) -> dict[str, object]:
    molecule = hp.Molecule.read_one(path, fmt)
    atoms = list(molecule.atoms)
    target_indices = [i for i, atom in enumerate(atoms) if atom.symbol == target_symbol]
    target_index = target_indices[0]

    adjacency = [set() for _ in atoms]
    explicit_donors = set()
    bond_pairs = []
    for bond in molecule.bonds:
        left, right = bond.atoms_indices
        adjacency[left].add(right)
        adjacency[right].add(left)
        bond_pairs.append((left, right))
        if left == target_index:
            explicit_donors.add(right)
        elif right == target_index:
            explicit_donors.add(left)

    target = atoms[target_index]
    inferred_donors = set(explicit_donors)
    for index, atom in enumerate(atoms):
        if index == target_index:
            continue
        distance = math.dist(target.coordinates, atom.coordinates)
        cutoff = 1.35 * (target.covalent_radius + atom.covalent_radius)
        if distance <= cutoff:
            inferred_donors.add(index)

    components = connected_components(adjacency, target_index)
    ligands = [component for component in components if component & inferred_donors]
    organic_ligands = 0
    for component in ligands:
        organic = any(
            {atoms[left].symbol, atoms[right].symbol} in ({"C"}, {"C", "H"})
            for left, right in bond_pairs
            if left in component and right in component
        )
        organic_ligands += organic

    metal_symbols = [atom.symbol for atom in atoms if atom.is_metal]
    return {
        "target_metal": target_symbol,
        "target_metal_count": len(target_indices),
        "all_metal_atom_count": len(metal_symbols),
        "metal_symbols": ";".join(metal_symbols),
        "inferred_donor_count": len(inferred_donors),
        "ligand_count": len(ligands),
        "organic_ligand_count": organic_ligands,
    }


def distributions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    metrics = ("target_metal_count", "all_metal_atom_count", "ligand_count", "organic_ligand_count")
    scopes = [("all", rows)] + [
        (subset, [row for row in rows if row["subset"] == subset])
        for subset in sorted({str(row["subset"]) for row in rows})
    ]
    for scope, scoped_rows in scopes:
        for metric in metrics:
            for value, count in sorted(Counter(int(row[metric]) for row in scoped_rows).items()):
                output.append({
                    "scope": scope, "metric": metric, "value": value,
                    "count": count, "percentage": f"{100 * count / len(scoped_rows):.4f}",
                })
    return output


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    index = {
        row["structure_id"]: row
        for row in read_csv(STRUCTURES_ROOT / "structures.csv")
    }
    metals = target_metals()
    cache: dict[tuple[str, str], dict[str, object]] = {}
    by_dataset: dict[str, list[dict[str, object]]] = {"oxidation": [], "spin": []}

    for link in read_csv(STRUCTURES_ROOT / "record_structure_links.csv"):
        key = (link["dataset"], link["subset"], link["record_id"])
        target_symbol = metals[key]
        structure = index[link["structure_id"]]
        cache_key = (link["structure_id"], target_symbol)
        if cache_key not in cache:
            cache[cache_key] = analyze(
                DATA_ROOT / structure["store_path"], structure["format"], target_symbol
            )
        by_dataset[link["dataset"]].append({
            "subset": link["subset"], "record_id": link["record_id"],
            "structure_role": link["structure_role"], "structure_id": link["structure_id"],
            **cache[cache_key],
        })

    summary = {}
    for dataset, rows in by_dataset.items():
        output = SOURCES_ROOT / dataset / "validation"
        write_csv(output / "structure_composition.csv", rows)
        write_csv(output / "composition_distributions.csv", distributions(rows))
        write_csv(
            output / "multiple_metal_atoms.csv",
            [row for row in rows if int(row["all_metal_atom_count"]) > 1],
        )
        summary[dataset] = {
            "structure_references": len(rows),
            "all_have_one_target_metal": all(int(row["target_metal_count"]) == 1 for row in rows),
            "references_with_multiple_metal_atoms": sum(int(row["all_metal_atom_count"]) > 1 for row in rows),
            "single_ligand_references": sum(int(row["ligand_count"]) == 1 for row in rows),
        }
    (STRUCTURES_ROOT / "composition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
