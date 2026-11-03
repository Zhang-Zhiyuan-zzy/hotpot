#!/usr/bin/env python3
"""Ratcheted source checks for the incremental Plan20 migration.

The frozen inventory is discovery evidence. The shrinking allowlist records
remaining debt. Stage9 retires every legacy-core import slot; canonical
carriers and views have one declared defining path. Runtime ownership,
buffer isolation and native ABI tests remain separate gates.
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
TYPE_OWNERS = {
    "Molecule": "hotpot.cheminfo.structure.molecule.Molecule",
    "MolBundle": "hotpot.cheminfo.structure.bundle.MolBundle",
    "Atom": "hotpot.cheminfo.structure.atom.Atom",
    "Bond": "hotpot.cheminfo.structure.bond.Bond",
    "BondKind": "hotpot.cheminfo.structure.bond_kind.BondKind",
    "Crystal": "hotpot.cheminfo.structure.crystal.Crystal",
    "MolBlock": "hotpot.cheminfo.structure._fields.MolBlock",
    "Conformers": "hotpot.cheminfo.structure.conformer.Conformers",
    "NotInSameMolecule": "hotpot.cheminfo.structure.errors.NotInSameMolecule",
    "ObjNotInMolecule": "hotpot.cheminfo.structure.errors.ObjNotInMolecule",
    "AtomSeq": "hotpot.cheminfo.views.sequence.AtomSeq",
    "AtomPair": "hotpot.cheminfo.views.pairs.AtomPair",
    "AtomPairs": "hotpot.cheminfo.views.pairs.AtomPairs",
    "Angle": "hotpot.cheminfo.views.geometry.Angle",
    "Torsion": "hotpot.cheminfo.views.geometry.Torsion",
    "JointRing": "hotpot.cheminfo.views.topology.JointRing",
    "Ring": "hotpot.cheminfo.views.topology.Ring",
}
CARRIERS = set(TYPE_OWNERS)
CRYSTAL_PATH = "hotpot/cheminfo/structure/crystal.py"
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


def frozen_ratchet_findings(actual, allowed, previous, original):
    """Retirement grants no import relocation or frozen-inventory growth credit."""
    findings = ratchet_findings(actual, allowed, previous)
    findings["exceeds_frozen_inventory"] = list((
        Counter(map(debt_key, allowed)) - Counter(map(debt_key, original))
    ).elements())
    return findings


def active_core_findings(path, source):
    """Reject retired carrier modules/entry calls independently of the allowlist."""
    findings = []
    retired = {
        "hotpot.cheminfo.core": "core",
        "hotpot.cheminfo.bundle": "bundle",
        "hotpot.cheminfo.geometry": "geometry",
        "hotpot.cheminfo.graph": "graph",
        "hotpot.cheminfo._native": "native_transport",
    }
    for module, label in retired.items():
        if path in {module.replace(".", "/") + ".py", module.replace(".", "/") + "/__init__.py"}:
            findings.append({"path": path, "line": 1, "rule": f"legacy_{label}_module"})
    for node in ast.walk(ast.parse(source, filename=path)):
        targets = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = inventory.imported_module(path, node)
            targets = [module] + [f"{module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if name in {"import_module", "__import__"}:
                targets = [arg.value for arg in node.args
                           if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
        for module, label in retired.items():
            if any(target == module or target.startswith(module + ".") for target in targets):
                findings.append({"path": path, "line": node.lineno,
                                 "rule": f"active_legacy_{label}_reference", "target": ast.unparse(node)})
    return findings


def carrier_definition_findings(classes):
    definitions = {name: [row["symbol"] for row in classes
                         if row["path"].startswith("hotpot/cheminfo/") and row["name"] == name]
                   for name in TYPE_OWNERS}
    return {name: actual for name, actual in definitions.items()
            if actual != [TYPE_OWNERS[name]]}


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

    def lexical_scope(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                return node

    def exact_import(node, module, name):
        return (isinstance(node, ast.ImportFrom)
                and inventory.imported_module(path, node) == module
                and [(alias.name, alias.asname) for alias in node.names] == [(name, None)])

    def crystal_math_import(node):
        scope = lexical_scope(node)
        cls = parents.get(scope)
        return (path == CRYSTAL_PATH
                and exact_import(node, "hotpot.cheminfo.rules.mathematics.cell", "_matrix_to_params")
                and isinstance(scope, ast.FunctionDef) and scope.name == "_matrix_to_params"
                and ast.unparse(scope.args) == "matrix"
                and [ast.unparse(item) for item in scope.decorator_list] == ["staticmethod"]
                and isinstance(cls, ast.ClassDef) and cls.name == "Crystal"
                and parents.get(cls) is tree)

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
                    forbidden = target_layer in {"rules", "adapters"} and not crystal_math_import(node)
                    eager = (target_parts[0] in EXTERNAL | {"scipy"}
                             or target_layer == "operations"
                             or any(part.endswith("_native") for part in target_parts)
                             or any(alias.name.endswith("_native") for alias in node.names))
                    if forbidden or (eager and (path == CRYSTAL_PATH or not local_import(node))):
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
    ratchet = frozen_ratchet_findings(records["architecture_debt"], allowed, previous, original)
    layers = [finding for path in paths for finding in layer_findings(path, (root / path).read_text())]
    # Runnable plan/tool consumers must also obey the hard cut. Their old
    # scientific manifests are history; inert strings are not active imports.
    legacy_paths = sorted(set(paths) | {
        path for path in subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "plan", "tools"],
            cwd=root, text=True,
        ).splitlines() if path.endswith(".py") and (root / path).is_file()
    })
    legacy = [finding for path in legacy_paths for finding in active_core_findings(path, (root / path).read_text())]
    duplicate_carriers = carrier_definition_findings(records["classes"])
    return {
        "schema_version": 1,
        "status": "passed" if not any(ratchet.values()) and not layers and not legacy and not duplicate_carriers else "failed",
        "base_ref": base_ref,
        "scanned_python_files": len(paths),
        "legacy_consumer_python_files": len(legacy_paths),
        "remaining_debt": len(records["architecture_debt"]),
        "ratchet": ratchet,
        "layer_findings": layers,
        "legacy_core_findings": legacy,
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
