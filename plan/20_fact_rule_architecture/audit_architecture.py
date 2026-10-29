#!/usr/bin/env python3
"""Ratcheted source checks for the incremental Plan20 migration.

The frozen inventory is discovery evidence. The shrinking allowlist records
remaining debt, while new ownership packages have no inherited exceptions.
Runtime ownership, buffer isolation and native ABI tests remain separate gates.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess


PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[1]
ALLOWLIST = PLAN / "architecture_allowlist.json"
spec = importlib.util.spec_from_file_location(
    "_plan20_inventory", PLAN / "capture_baseline_inventory.py"
)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)

LAYERS = {"structure", "views", "rules", "operations", "adapters", "reference"}
EXTERNAL = {"rdkit", "openbabel", "thermo", "pubchempy"}
CANONICAL = inventory.PRIVATE_FACTS | inventory.FACT_PROPERTIES | {
    "_atom_table", "_uid", "_next_atom_uid", "_next_bond_uid", "_constraints",
    "_results", "_dependencies", "_source_frames", "_crystal", "_pair_wiberg_bond_orders",
    # Scalar projections are writes to the same facts as their array aliases.
    "n", "s", "p", "d", "f", "g", "valence", "electron_configuration",
    "x", "y", "z", "id", "x_constraint", "y_constraint", "z_constraint",
}
MUTATORS = {
    "append", "extend", "insert", "remove", "pop", "clear", "update",
    "setdefault", "__setitem__", "fill", "put", "resize", "sort",
}
CARRIER_METHODS = {
    "create_atom", "add_atom", "add_bond", "add_bonds", "add_component",
    "remove_atom", "remove_atoms", "remove_bond", "remove_bonds", "replace_atom",
    "updata_atoms", "conformer_load", "conformer_add", "conformer_clear",
    "hide_metal_ligand_bonds", "recover_metal_ligand_bonds",
}
CARRIERS = {"Molecule", "Atom", "Bond", "BondKind", "Crystal"}
# These declared numerical values initialize their own coordinate fields; they
# do not receive a live carrier to edit. Keep path, class, method, field and
# operation exact so foreign receivers and later buffer mutations stay checked.
VALUE_INITIALIZERS = {
    ("hotpot/cheminfo/rules/mathematics/geometry/object.py", "Point", "__init__", "coordinates"):
        "object.__setattr__",
    ("hotpot/cheminfo/adapters/structure/geometry.py", "BondRingFrameWorkspace", "__post_init__", "coordinates"):
        "object.__setattr__",
    ("hotpot/cheminfo/rules/mathematics/internal_coordinates.py", "InternalCoordinates", "__init__", "coordinates"):
        "attribute",
}


def debt_key(record):
    """Line shifts do not change a file/symbol/operation identity."""
    return tuple((key, record[key]) for key in (
        "rule", "path", "scope", "target", "attribute", "local", "kind"
    ) if key in record)


def ratchet_findings(actual, allowed, previous):
    observed = Counter(map(debt_key, actual))
    current = Counter(map(debt_key, allowed))
    ceiling = Counter(map(debt_key, previous))
    return {
        "new_debt": list((observed - current).elements()),
        "unretired_allowlist_entries": list((current - observed).elements()),
        "allowlist_growth": list((current - ceiling).elements()),
    }


def layer_findings(path, source):
    parts = Path(path).parts
    if parts[:2] != ("hotpot", "cheminfo") or len(parts) < 4 or parts[2] not in LAYERS:
        return []
    layer = parts[2]
    tree = ast.parse(source, filename=path)
    parents = {child: parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
    findings = []
    seen = set()

    def add(node, rule):
        if (node, rule) in seen:
            return
        seen.add((node, rule))
        findings.append({"path": path, "line": node.lineno,
                         "rule": rule, "target": ast.unparse(node)})

    def local_import(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return True
        return False

    def owns_fact(node):
        return any(isinstance(item, ast.Attribute) and item.attr in CANONICAL
                   for item in ast.walk(node))

    def initializes_value(node, receiver, attribute, operation):
        if not isinstance(receiver, ast.Name) or receiver.id != "self":
            return False
        scope = node
        while scope in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = parents[scope]
        cls = parents.get(scope)
        if not isinstance(cls, ast.ClassDef) or not isinstance(parents.get(cls), ast.Module):
            return False
        return VALUE_INITIALIZERS.get((path, cls.name, scope.name, attribute)) == operation

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                module = inventory.imported_module(path, node)
                targets = [module] + [f"{module}.{alias.name}" for alias in node.names
                                      if alias.name != "*"]
            for target in targets:
                target_parts = target.split(".")
                target_layer = (target_parts[2] if target_parts[:2] == ["hotpot", "cheminfo"]
                                and len(target_parts) > 2 else None)
                if target in {"hotpot", "hotpot.cheminfo"} or (
                    target_layer in LAYERS and len(target_parts) == 3
                    and isinstance(node, ast.ImportFrom)
                    and any(alias.name in CARRIERS for alias in node.names)
                ):
                    add(node, "internal_upward_facade_import")
                if layer in {"structure", "views"}:
                    forbidden = target_layer in {"rules", "adapters"}
                    eager = (target_parts[0] in EXTERNAL | {"scipy"}
                             or target_layer == "operations"
                             or any(part.endswith("_native") for part in target_parts)
                             or any(alias.name.endswith("_native") for alias in node.names))
                    if forbidden or (eager and not local_import(node)):
                        add(node, "carrier_implementation_import")
                if layer == "rules" and (
                    target_parts[0] in EXTERNAL
                    or target_layer in {"core", "operations", "adapters"}
                    or (target_layer == "structure" and len(target_parts) > 3
                        and target_parts[3] in {"molecule", "atom", "bond", "_commit", "_mutation", "_builder"})
                ):
                    add(node, "rule_execution_or_carrier_dependency")
                if layer == "reference" and (
                    target_parts[0] in EXTERNAL
                    or (target_parts[0] == "hotpot" and target_layer != "reference")
                ):
                    add(node, "reference_execution_dependency")
                if layer == "adapters" and target_layer == "operations":
                    add(node, "adapter_upward_operation_import")
        if layer not in {"rules", "adapters"}:
            continue
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            initialization = (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                              and initializes_value(node, node.value, node.attr, "attribute"))
            if owns_fact(node) and not initialization:
                add(node, "canonical_fact_write")
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in {"setattr", "delattr", "object.__setattr__", "object.__delattr__"}:
                if len(node.args) >= 2 and (
                    not isinstance(node.args[1], ast.Constant)
                    or node.args[1].value in CANONICAL
                ):
                    attribute = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
                    if not initializes_value(node, node.args[0], attribute, name):
                        add(node, "canonical_or_dynamic_fact_write")
            if isinstance(node.func, ast.Attribute) and (
                (node.func.attr in MUTATORS and owns_fact(node.func.value))
                or node.func.attr in CARRIER_METHODS
            ):
                add(node, "canonical_mutation_call")
    return findings


def source_paths(root):
    return sorted(path for path in subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--",
         "hotpot", "tests"], cwd=root, text=True,
    ).splitlines() if path.endswith(".py") and (root / path).is_file())


def audit(root=ROOT, *, base_ref="HEAD"):
    paths = source_paths(root)
    records = inventory.static_capture(paths, root=root)
    relative = ALLOWLIST.relative_to(ROOT)
    allowed = json.loads((root / relative).read_text())["entries"]
    original = [json.loads(line) for line in
                (root / PLAN.relative_to(ROOT) / "baseline_inventory/architecture_debt.jsonl").read_text().splitlines()]
    previous_file = subprocess.run(
        ["git", "show", f"{base_ref}:{relative.as_posix()}"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    # The first introduction has no previous allowlist; the Stage1 inventory
    # remains its ceiling. Missing later refs are errors, not a new baseline.
    if previous_file.returncode:
        subprocess.run(["git", "rev-parse", "--verify", base_ref], cwd=root,
                       check=True, capture_output=True)
        previous = original
    else:
        previous = json.loads(previous_file.stdout)["entries"]
    ratchet = ratchet_findings(records["architecture_debt"], allowed, previous)
    ratchet["exceeds_frozen_inventory"] = list((Counter(map(debt_key, allowed))
                                              - Counter(map(debt_key, original))).elements())
    layers = [finding for path in paths for finding in layer_findings(path, (root / path).read_text())]
    definitions = {name: [row["symbol"] for row in records["classes"]
                         if row["path"].startswith("hotpot/cheminfo/") and row["name"] == name]
                   for name in sorted(CARRIERS)}
    duplicate_carriers = {name: values for name, values in definitions.items() if len(values) != 1}
    return {
        "schema_version": 1,
        "status": "passed" if not any(ratchet.values()) and not layers and not duplicate_carriers else "failed",
        "base_ref": base_ref,
        "scanned_python_files": len(paths),
        "remaining_debt": len(records["architecture_debt"]),
        "ratchet": ratchet,
        "layer_findings": layers,
        "carrier_definition_findings": duplicate_carriers,
        "scope": "Source/import/write ratchet and migrated-layer checks. Runtime alias/permission/lifecycle, semantic class identity and native ABI gates remain separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--base-ref", default="HEAD", help="Pre-change commit; CI should select its actual merge base.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.repository.resolve(), base_ref=args.base_ref)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
