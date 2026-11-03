#!/usr/bin/env python3
"""Ratcheted source checks for the incremental Plan20 migration.

The frozen inventory is discovery evidence. The shrinking allowlist records
remaining debt. Fixed Stage7A–8D relocations carry their original
import slots; new ownership packages otherwise have no inherited exceptions. Runtime ownership,
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
CARRIERS = {"Molecule", "Atom", "Bond", "BondKind", "Crystal"}
CRYSTAL_PATH = "hotpot/cheminfo/structure/crystal.py"
# Main Section9.2 / Stage7A P01 relocates one existing import, not a new
# exception. Only convert.py supplies its slot; Stage9 must retire the lookup.
# The evidence record under stage07/external-adapters/architecture_guard is
# descriptive, never an extensible migration configuration.
CORE_IMPORT_SOURCE = {
    "rule": "legacy_core_import", "path": "hotpot/cheminfo/convert.py",
    "scope": "", "target": "hotpot.cheminfo.core.Molecule", "local": "Molecule",
}
CORE_IMPORT_DESTINATION = {
    **CORE_IMPORT_SOURCE, "path": "hotpot/cheminfo/operations/conversion.py",
    "scope": "_molecule_type",
}
# Stage8A A1/A7 splits read-only capture and working-copy preparation. These three
# module-scope bindings use only their own former resolver slots until Stage9;
# this fixed tuple is not a configurable relocation map or growth allowance.
# Stage8C relocates the three eager carrier bindings of the one public search
# implementation. Their annotations/isinstance contracts still use current core
# identities; only these exact source slots can transfer until Stage9.
SEARCH_CORE_IMPORT_TRANSFERS = tuple(
    (
        {"rule": "legacy_core_import", "path": "hotpot/cheminfo/search/search.py",
         "scope": "", "target": f"hotpot.cheminfo.core.{name}", "local": name},
        {"rule": "legacy_core_import", "path": "hotpot/cheminfo/operations/search/search.py",
         "scope": "", "target": f"hotpot.cheminfo.core.{name}", "local": name},
    )
    for name in ("Molecule", "Atom", "Bond")
)
SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS = (
    (
        {"rule": "legacy_core_import",
         "path": "hotpot/cheminfo/calculator/state_resolver.py", "scope": "",
         "target": "hotpot.cheminfo.core.Bond", "local": "Bond"},
        {"rule": "legacy_core_import",
         "path": "hotpot/cheminfo/adapters/structure/scientific_state.py", "scope": "",
         "target": "hotpot.cheminfo.core.Bond", "local": "Bond"},
    ),
    (
        {"rule": "legacy_core_import",
         "path": "hotpot/cheminfo/calculator/state_resolver.py", "scope": "",
         "target": "hotpot.cheminfo.core.BondKind", "local": "BondKind"},
        {"rule": "legacy_core_import",
         "path": "hotpot/cheminfo/adapters/structure/scientific_state.py", "scope": "",
         "target": "hotpot.cheminfo.core.BondKind", "local": "BondKind"},
    ),
    (
        {"rule": "legacy_core_import",
         "path": "hotpot/cheminfo/calculator/state_resolver.py", "scope": "",
         "target": "hotpot.cheminfo.core.Molecule", "local": "Molecule"},
        {"rule": "legacy_core_import",
         "path": "hotpot/cheminfo/operations/calculation/scientific_state/facts.py", "scope": "",
         "target": "hotpot.cheminfo.core.Molecule", "local": "Molecule"},
    ),
)
# Stage8B B3c relocates the existing TYPE_CHECKING-only native transport
# annotation. It consumes only its prior native.py Molecule slot until Stage9;
# the deleted native_packing.py Bond/Molecule slots retire without transfer.
NATIVE_TRANSPORT_CORE_IMPORT_SOURCE = {
    "rule": "legacy_core_import", "path": "hotpot/cheminfo/forcefields/native.py",
    "scope": "", "target": "hotpot.cheminfo.core.Molecule", "local": "Molecule",
}
NATIVE_TRANSPORT_CORE_IMPORT_DESTINATION = {
    **NATIVE_TRANSPORT_CORE_IMPORT_SOURCE,
    "path": "hotpot/cheminfo/adapters/native/forcefields/native.py",
}
# Accepted Stage8B B3b moved this existing TYPE_CHECKING-only command
# annotation verbatim. Its CLI slot is independent of the native transport slot;
# neither can supply growth credit or substitute for the other before Stage9.
COMMAND_CORE_IMPORT_SOURCE = {
    "rule": "legacy_core_import", "path": "hotpot/cheminfo/forcefields/cli.py",
    "scope": "", "target": "hotpot.cheminfo.core.Molecule", "local": "Molecule",
}
COMMAND_CORE_IMPORT_DESTINATION = {
    **COMMAND_CORE_IMPORT_SOURCE, "path": "hotpot/commands/ff.py",
}
# Stage8D carries only these seven existing bindings. The old feature Atom,
# two site-recognition bindings and combined conformer-adapter annotation retire;
# their slots cannot pay for imports in a new provider or lower adapter.
INFERENCE_CORE_IMPORT_TRANSFERS = tuple(
    (
        {"rule": "legacy_core_import", "path": f"hotpot/cheminfo/{old}",
         "scope": "", "target": f"hotpot.cheminfo.core.{name}", "local": local},
        {"rule": "legacy_core_import", "path": f"hotpot/cheminfo/{new}",
         "scope": "", "target": f"hotpot.cheminfo.core.{name}", "local": local},
    )
    for old, new, bindings in (
        ("AImodels/cbond/apply.py", "operations/inference/cbond/apply.py",
         (("Atom", "Atom"), ("Molecule", "Molecule"))),
        ("AImodels/data_extract.py", "operations/inference/features.py",
         (("Molecule", "Molecule"), ("AtomPair", "AtomPair"))),
        ("AImodels/mca/serialization.py", "operations/inference/mca/serialization.py",
         (("Molecule", "HotpotMolecule"),)),
        ("calculator/mca_inference.py", "operations/inference/mca/attachment.py",
         (("Molecule", "Molecule"),)),
        ("conformers/sampler.py", "operations/conformers/sampler.py",
         (("Molecule", "Molecule"),)),
    )
    for name, local in bindings
)

CORE_IMPORT_TRANSFERS = (
    (CORE_IMPORT_SOURCE, CORE_IMPORT_DESTINATION),
    *SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS,
    (NATIVE_TRANSPORT_CORE_IMPORT_SOURCE, NATIVE_TRANSPORT_CORE_IMPORT_DESTINATION),
    (COMMAND_CORE_IMPORT_SOURCE, COMMAND_CORE_IMPORT_DESTINATION),
    *SEARCH_CORE_IMPORT_TRANSFERS,
    *INFERENCE_CORE_IMPORT_TRANSFERS,
)
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


def stage7a_ratchet_findings(actual, allowed, previous, original):
    """Keep real debt identities; transfer only the enumerated existing slots."""
    findings = ratchet_findings(actual, allowed, previous)
    current = Counter(map(debt_key, allowed))
    for name, reference in (("allowlist_growth", previous),
                            ("exceeds_frozen_inventory", original)):
        ceiling = Counter(map(debt_key, reference))
        for source, destination in CORE_IMPORT_TRANSFERS:
            old = debt_key(source)
            new = debt_key(destination)
            if current[new] and not current[old] and ceiling[old] and not ceiling[new]:
                ceiling[old] -= 1
                ceiling[new] += 1
        findings[name] = list((current - ceiling).elements())
    return findings


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

    def core_lookup_import(node):
        # A same-named class has the same inventory scope string as a function.
        scope = lexical_scope(node)
        return (exact_import(node, "hotpot.cheminfo.core", "Molecule")
                and isinstance(scope, ast.FunctionDef) and parents.get(scope) is tree
                and scope.name == "_molecule_type" and not scope.decorator_list
                and ast.unparse(scope.args) == "")

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
                if (path == CORE_IMPORT_DESTINATION["path"]
                        and (target == "hotpot.cheminfo.core" or target.startswith("hotpot.cheminfo.core."))
                        and not core_lookup_import(node)):
                    add(node, "invalid_legacy_core_import_relocation")
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
    ratchet = stage7a_ratchet_findings(records["architecture_debt"], allowed, previous, original)
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
