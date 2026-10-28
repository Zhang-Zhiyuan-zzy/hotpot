#!/usr/bin/env python3
"""Capture Plan20 source/API inventories without changing production files.

Static references are evidence candidates, not proof of consumer completeness.
Runtime capture imports the selected checkout in this process and only probes
new temporary objects. All generated records go to the caller's output folder.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plan/20_fact_rule_architecture/fact_rule_architecture_implementation_plan.md"
CONTRACT_MARKERS = {"dataclass", "TypedDict", "Protocol", "NamedTuple"}
CACHES = {
    "_angles": "Stage4", "_torsions": "Stage4", "_atom_pairs": "Stage4",
    "_rings": "Stage5", "_ring_indices_cache": "Stage5", "_ligand_rings": "Stage5",
    "_ligand_rings_signature": "Stage5", "_obmol": "Stage7A", "_row2idx": "Stage7A", "_obcell": "Stage7A",
    "_hided_metal_bonds": "Stage7B", "_hided_covalent_bonds": "Stage7B",
    "_mca": "Stage7C", "_mca_sites": "Stage7C", "_model": "Stage8D", "_environ": "Stage8D",
}
PRIVATE_FACTS = set(CACHES) | {"_atoms", "_bonds", "_stereo_payload", "attrs", "_topology_index"}
FACT_PROPERTIES = {"atomic_number", "symbol", "formal_charge", "partial_charge", "coordinates", "is_aromatic", "implicit_hydrogens", "bond_order", "bond_kind", "bond_direction", "constraint", "charge", "spin", "crystal", "properties"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expression(node):
    return ast.unparse(node) if node is not None else None


def leaf(node):
    if isinstance(node, ast.Call):
        return leaf(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def module_name(path):
    parts = list(Path(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_module(path, node):
    if not node.level:
        return node.module or ""
    parts = module_name(path).split(".")
    if Path(path).name != "__init__.py":
        parts.pop()
    parts = parts[:len(parts) - node.level + 1]
    if node.module:
        parts.extend(node.module.split("."))
    return ".".join(parts)


def role(path, name, markers=()):
    if "Protocol" in markers:
        return "read_or_behavior_protocol"
    if path.endswith("core.py"):
        return {
            "Molecule": "fact_carrier", "Atom": "fact_handle", "Bond": "fact_carrier",
            "Conformers": "source_and_result_collection_to_split", "AtomPairs": "derived_collection",
            "AtomSeq": "derived_view", "AtomPair": "derived_view", "Angle": "derived_view",
            "Torsion": "derived_view", "Ring": "derived_view", "JointRing": "derived_view",
            "InternalCoordinates": "mathematical_calculation", "MolBlock": "dynamic_field_implementation",
        }.get(name, "support_type")
    if name == "Crystal":
        return "independent_cell_value"
    if name == "MolBundle":
        return "carrier_collection"
    if "/_native/" in path or "/rdWrappers/" in path or "/obWrappers/" in path:
        return "native_or_standalone_contract"
    if "Snapshot" in name or "Facts" in name or "Topology" in name or "Identity" in name:
        return "snapshot_identity_or_topology_value"
    if "Result" in name or "Prediction" in name or "Report" in name or "Finding" in name:
        return "scientific_result_or_observation"
    if "Task" in name or "Request" in name or "Message" in name:
        return "operation_input_or_message"
    if "Options" in name or "Settings" in name or "Policy" in name:
        return "operation_or_rule_configuration"
    if "/geometry/" in path:
        return "mathematical_value_or_rule"
    return "domain_contract_or_implementation_to_review"


def owner_stage(path, name=""):
    if path.endswith("core.py"):
        if name in {"Angle", "Torsion", "AtomSeq", "AtomPair", "AtomPairs", "InternalCoordinates"}:
            return "Stage4"
        if name in {"Ring", "JointRing"}:
            return "Stage5"
        if name == "Conformers":
            return "Stage7D"
        return "Stage9"
    if any(x in path for x in ("/parse_smarts.py", "/ob2chem.py", "/org_compounds.py", "/solvents.py", "/calculator/base.py", "/_clib/")):
        return "Stage2"
    if "/stereo.py" in path or "rdconvert.py" in path:
        return "Stage6"
    if "/kekulize/" in path or path.endswith("elements.py"):
        return "Stage5"
    if "/geometry/" in path or "/graph/" in path:
        return "Stage4"
    if "/calculator/" in path:
        return "Stage7C" if "mca_inference" in path else "Stage8A"
    if "/forcefields/" in path:
        return "Stage8B"
    if "/search/" in path:
        return "Stage8C"
    if "/AImodels/" in path or "/conformers/" in path:
        return "Stage8D"
    if "/mol_assemble/" in path or path.endswith("mol_statistics.py"):
        return "Stage8E"
    if "/rdWrappers/" in path or "/obWrappers/" in path or "/_native/" in path:
        return "Stage10"
    if "/main/" in path or "/works/" in path:
        return "Stage11"
    if path in {"hotpot/__init__.py", "hotpot/cheminfo/__init__.py"}:
        return "Stage12"
    if path.startswith("hotpot/cheminfo/"):
        return "Stage7A"
    return "Stage9"  # outer consumers migrate chemistry references only


def function_record(node):
    decorators = [expression(d) for d in node.decorator_list]
    kind = "method"
    if "property" in decorators:
        kind = "property_read"
    elif any(d.endswith(".setter") for d in decorators):
        kind = "property_write"
    elif any(d.endswith(".deleter") for d in decorators):
        kind = "property_delete"
    elif "classmethod" in decorators:
        kind = "classmethod"
    elif "staticmethod" in decorators:
        kind = "staticmethod"
    return {"name": node.name, "line": node.lineno, "kind": kind,
            "signature": f"({ast.unparse(node.args)})", "returns": expression(node.returns),
            "decorators": decorators,
            "read_attribute_names": sorted({n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)}),
            "written_attribute_names": sorted({n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del))})}


def static_capture(paths, *, root=ROOT):
    classes, functions, imports, writes, debt, aliases = [], [], [], [], [], []
    exports, cli_arguments, property_writes = [], [], []
    bindings = defaultdict(dict)
    calls = []
    for path in paths:
        tree = ast.parse((root / path).read_text(), filename=path)
        module = module_name(path)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                bindings[module][node.name] = f"{module}.{node.name}"
            if isinstance(node, ast.AnnAssign) and leaf(node.annotation) == "TypeAlias":
                aliases.append({"path": path, "line": node.lineno, "name": expression(node.target), "definition": expression(node.value)})
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                exports.append({"path": path, "line": node.lineno, "declaration": expression(node.value),
                                "literal_names": [item.value for item in ast.walk(node.value) if isinstance(item, ast.Constant) and isinstance(item.value, str)]})
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        def enclosing(node):
            names = []
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.append(node.name)
            return ".".join(reversed(names))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and path.startswith("hotpot/"):
                scope = enclosing(node)
                markers = {leaf(n) for n in node.decorator_list + node.bases} & CONTRACT_MARKERS
                fields = [{"name": expression(n.target), "annotation": expression(n.annotation), "default": expression(n.value), "line": n.lineno}
                          for n in node.body if isinstance(n, ast.AnnAssign)]
                instance_fields = sorted({n.attr for member in node.body if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                                          for n in ast.walk(member) if isinstance(n, ast.Attribute)
                                          and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) and n.value.id == "self"})
                methods = [function_record(n) for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append({"symbol": ".".join(filter(None, [module, scope, node.name])), "name": node.name,
                                "path": path, "line": node.lineno, "markers": sorted(markers),
                                "bases": [expression(n) for n in node.bases], "decorators": [expression(n) for n in node.decorator_list],
                                "role": role(path, node.name, markers), "stage": owner_stage(path, node.name),
                                "current_owner": module, "migration_scope": "full chemistry responsibility review" if path.startswith("hotpot/cheminfo/") else "outer consumer: chemistry imports only; retain domain behavior",
                                "fields": fields, "instance_field_names": instance_fields, "methods": methods,
                                "public_name_candidate": not node.name.startswith("_"), "nested_scope": scope})
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and path.startswith("hotpot/") and parents.get(node) is tree:
                functions.append({"path": path, "scope": enclosing(node), **function_record(node)})
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                base = imported_module(path, node) if isinstance(node, ast.ImportFrom) else ""
                for alias in node.names:
                    target = ".".join(filter(None, [base, alias.name]))
                    local = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                    record = {"path": path, "line": node.lineno, "scope": enclosing(node), "target": target, "local": local}
                    imports.append(record)
                    if not record["scope"] and alias.name != "*":
                        bindings[module][local] = target if isinstance(node, ast.ImportFrom) or alias.asname else alias.name.split(".")[0]
                    if target == "hotpot.cheminfo.core" or target.startswith("hotpot.cheminfo.core."):
                        debt.append({**record, "rule": "legacy_core_import", "stage": "Stage9"})
                    if alias.name == "*" and Path(path).name == "__init__.py" and path.startswith("hotpot/"):
                        debt.append({**record, "rule": "wildcard_package_export", "stage": owner_stage(path) if "/mol_assemble/" in path else "Stage12"})
                    if path in {"hotpot/cheminfo/core.py", "hotpot/cheminfo/crystal.py", "hotpot/cheminfo/bundle.py", "hotpot/__init__.py"} and not record["scope"] and target.split(".")[0] in {"rdkit", "openbabel", "scipy", "thermo", "pubchempy"}:
                        debt.append({**record, "rule": "carrier_or_root_eager_toolkit_import", "stage": "Stage12" if path == "hotpot/__init__.py" else owner_stage(path)})
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                record = {"path": path, "line": node.lineno, "scope": enclosing(node), "target": expression(node), "attribute": node.attr}
                if node.attr in FACT_PROPERTIES:
                    property_writes.append({**record, "classification": "syntactic write candidate; verify receiver/ownership before treating as a rule violation"})
                if node.attr in PRIVATE_FACTS:
                    writes.append(record)
                if node.attr in CACHES and path.startswith("hotpot/"):
                    debt.append({**record, "rule": "forbidden_retained_state_write", "stage": CACHES[node.attr]})
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
                for attribute in ast.walk(node.value):
                    if isinstance(attribute, ast.Attribute) and attribute.attr in PRIVATE_FACTS:
                        writes.append({"path": path, "line": node.lineno, "scope": enclosing(node), "target": expression(node), "attribute": attribute.attr, "kind": "buffer_or_mapping_item_write"})
            if isinstance(node, ast.Call):
                calls.append({"path": path, "line": node.lineno, "scope": enclosing(node), "expression": expression(node.func)})
                name = expression(node.func)
                if isinstance(node.func, ast.Attribute) and node.func.attr in {"add_argument", "add_parser"} and path.startswith("hotpot/"):
                    cli_arguments.append({"path": path, "line": node.lineno, "scope": enclosing(node),
                                          "call": name, "arguments": [expression(arg) for arg in node.args],
                                          "keywords": {arg.arg or "**": expression(arg.value) for arg in node.keywords}})
                if name in {"setattr", "object.__setattr__"} and len(node.args) >= 2:
                    attribute = expression(node.args[1])
                    if isinstance(node.args[1], ast.Constant) and node.args[1].value in PRIVATE_FACTS:
                        writes.append({"path": path, "line": node.lineno, "scope": enclosing(node), "target": expression(node), "attribute": node.args[1].value})
                if path.startswith("hotpot/") and name in {"sys.path.append", "sys.path.insert"}:
                    debt.append({"path": path, "line": node.lineno, "scope": enclosing(node), "rule": "sys_path_mutation", "stage": owner_stage(path), "target": expression(node)})
            if isinstance(node, ast.Attribute) and expression(node) == "sys.modules" and path in {"hotpot/cheminfo/core.py", "hotpot/cheminfo/_io/_io.py"}:
                debt.append({"path": path, "line": node.lineno, "scope": enclosing(node), "rule": "import_cycle_workaround", "stage": "Stage7A", "target": "sys.modules"})
            if isinstance(node, ast.Name) and node.id == "WeakKeyDictionary" and isinstance(node.ctx, ast.Load) and path.startswith("hotpot/cheminfo/search/"):
                debt.append({"path": path, "line": node.lineno, "scope": enclosing(node), "rule": "molecule_keyed_search_cache", "stage": "Stage8C", "target": node.id})

    def resolve(symbol):
        seen = set()
        while symbol not in seen:
            seen.add(symbol)
            module, _, name = symbol.rpartition(".")
            candidate = bindings.get(module, {}).get(name, symbol)
            if candidate == symbol:
                break
            symbol = candidate
        return symbol

    import_consumers = defaultdict(list)
    for record in imports:
        import_consumers[resolve(record["target"])].append(record)
    constructions = defaultdict(list)
    for call in calls:
        base, dot, suffix = call["expression"].partition(".")
        bound = bindings[module_name(call["path"])].get(base)
        if bound:
            constructions[resolve(bound + (dot + suffix if dot else ""))].append(call)
    for record in classes:
        record["import_consumers"] = import_consumers.get(record["symbol"], [])
        record["constructor_call_candidates"] = constructions.get(record["symbol"], [])
        record["consumer_evidence_scope"] = "resolved explicit imports and module-bound calls; dynamic/re-export/pickle consumers require review"
    return {"classes": classes, "functions": functions, "imports": imports, "private_writes": writes,
            "architecture_debt": debt, "explicit_type_aliases": aliases,
            "declared_exports": exports, "cli_arguments": cli_arguments,
            "canonical_property_write_candidates": property_writes}


def interface_records(classes, runtime):
    public_dunders = {"__init__", "__iter__", "__next__", "__len__", "__getitem__", "__setitem__", "__delitem__", "__contains__", "__getattr__", "__setattr__", "__call__"}
    records = []
    for cls in classes:
        if not cls["path"].startswith("hotpot/cheminfo/") or not cls["public_name_candidate"]:
            continue
        for method in cls["methods"]:
            if method["name"].startswith("_") and method["name"] not in public_dunders:
                continue
            records.append({"type": cls["symbol"], "role": cls["role"], "current_owner": cls["current_owner"],
                            "path": cls["path"], "entry": method["name"], "stage": cls["stage"],
                            "evidence": "declared source interface; read/write attribute names are syntactic observations", **method})
        for field in cls["fields"]:
            if not field["name"].startswith("_"):
                records.append({"type": cls["symbol"], "role": cls["role"], "current_owner": cls["symbol"], "path": cls["path"],
                                "entry": field["name"], "kind": "annotated_field", "stage": cls["stage"],
                                "evidence": "declared annotation/default; public and buffer/lifecycle review still applies", **field})
    if runtime:
        for cls, fields in runtime["dynamic_fields"].items():
            for field in fields:
                name = field["name"]
                if name in {"partial_charge"}:
                    target = "selected PartialChargeSet"; stage = "Stage7C"; unit = "elementary charge"
                elif name.endswith("constraint"):
                    target = "Molecule ConstraintSet"; stage = "Stage7B"; unit = "boolean"
                elif name in {"n", "s", "p", "d", "f", "g", "valence"}:
                    target = "pure chemistry calculation"; stage = "Stage5"; unit = "dimensionless integer"
                elif cls == "Atom":
                    target = "Molecule _AtomFactTable"; stage = "Stage3"; unit = "angstrom" if name in {"x", "y", "z"} else "domain scalar"
                else:
                    target = "Bond coupled representation"; stage = "Stage5"; unit = "domain scalar"
                records.append({"type": f"hotpot.cheminfo.core.{cls}", "role": "fact_handle" if cls == "Atom" else "fact_carrier",
                                "current_owner": f"{cls}.attrs[{field['offset']}]", "entry": name, "kind": "dynamic_field",
                                "source_storage": "float64 attrs row cast by MolBlock", "shape": "scalar", "unit": unit,
                                "target_owner": target, "stage": stage, "evidence": "runtime _attrs_enumerator/_attrs_dict; target details governed by A1-A13", **field})
    return records


def runtime_capture():
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import hotpot
    import hotpot.cheminfo as cheminfo
    from hotpot.cheminfo import core
    from hotpot.cheminfo.crystal import Crystal
    from hotpot.__main__ import _COMMANDS

    exports = {}
    for module in (hotpot, cheminfo):
        entries = []
        for name in sorted(n for n in vars(module) if not n.startswith("_")):
            value = vars(module)[name]
            definition = f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
            signature = None
            if callable(value) and getattr(value, "__module__", "").startswith("hotpot"):
                try:
                    signature = str(inspect.signature(value))
                except (ValueError, TypeError):
                    signature = "not introspectable"
            entries.append({"name": name, "definition": definition, "kind": "class" if inspect.isclass(value) else "function" if inspect.isfunction(value) else "module" if inspect.ismodule(value) else "value", "signature": signature})
        exports[module.__name__] = {"declared_all": getattr(module, "__all__", None), "visible_names": entries}
    dynamic = {}
    for cls in (core.Atom, core.Bond):
        dynamic[cls.__name__] = [{"name": name, "offset": index, "cast": cls._attrs_dict[name].__name__, "read": True, "write": True,
                                  "custom_setter": getattr(cls._attrs_setter.get(name), "__name__", None)}
                                 for index, name in enumerate(cls._attrs_enumerator)]
    array = np.zeros(len(core.Atom._attrs_enumerator), dtype=float)
    atom = core.Atom(attrs_array=array, update_electron_config=False)
    probes = {"Atom.attrs_aliases_constructor_array": bool(np.shares_memory(array, atom.attrs)),
              "Atom.coordinates_aliases_attrs": bool(np.shares_memory(np.asarray(atom.coordinates), atom.attrs)),
              "Molecule.coordinates_aliases_attrs": bool(np.shares_memory(atom.mol.coordinates, atom.attrs)),
              "Atom.partial_charge_without_result": atom.partial_charge,
              "Crystal_constructor": str(inspect.signature(Crystal))}
    for label, obj, attr in [("Atom.mca_without_result", atom, "mca"), ("Molecule.mca_sites_without_result", atom.mol, "mca_sites")]:
        try:
            result = getattr(obj, attr)
            probes[label] = {"value": repr(result)}
        except Exception as error:
            probes[label] = {"exception": f"{type(error).__module__}.{type(error).__qualname__}", "message": str(error)}
    native_modules = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("hotpot.") and filename and filename.endswith(".so"):
            native_modules[name] = {"path": filename, "sha256": digest(Path(filename))}
    return {"exports": exports, "dynamic_fields": dynamic, "observed_behavior": probes, "native_modules": native_modules,
            "identity_checks": {f"hotpot.{name} is core.{name}": getattr(hotpot, name) is getattr(core, name) for name in ("Molecule", "Atom", "Bond", "BondKind")},
            "cli_commands": {name: {"module": spec.module, "help": spec.help} for name, spec in _COMMANDS.items()},
            "loaded_external_modules": sorted(name for name in sys.modules if name.split(".")[0] in {"rdkit", "openbabel", "scipy", "thermo", "pubchempy"})}


def migration_stage(section, current):
    text = current.lower()
    if current == "`geometry/convert.py`":
        return "Stage4"
    if current.startswith("`kekulize/"):
        return "Stage5"
    if "_hided_" in text:
        return "Stage7B"
    if "direct `cheminfo.geometry`" in text:
        return "Stage10"
    if current.startswith("generic `Molecule.setattr`"):
        return "Stage3"
    if current == "`MolReader`, `MolWriter`, `read_mol`":
        return "Stage7A"
    if current == "`build3d`, `optimize`":
        return "Stage8B"
    if "_topology_index" in text or current == "`bundle.py`":
        return "Stage9"
    if "coordination.py" in text and ("carrier extraction" in text or "sequencing" in text):
        return "Stage8B"
    if "metalcluster" in text:
        return "Plan21"
    if section == "9.3":
        if any(s in text for s in ("main", "works")):
            return "Stage11"
        if any(s in text for s in ("__init__", "__version__", "package_root", "readme/docs")):
            return "Stage12"
        return "Stage10" if "setup.py" in text else "Stage9"
    if any(s in text for s in ("core_func", "_clib", "parse_smarts", "ob2chem", "org_compounds", "solvents.py", "calculator.base", "calculator/base", "atompairkey", "placeholder", "assign_aromatic")):
        return "Stage2"
    if any(s in text for s in ("cli", "main/works", "chemdata/coformers", "chemdata/solvent", "molprops")):
        return "Stage11"
    if any(s in text for s in ("__init__", "hotpot.version", "hotpot.package_root", "wildcard surface")):
        return "Stage12" if "assembly" not in text else "Stage8E"
    if "rdwrappers" in text or "obwrappers" in text or "rdwrapper" in text or "obwrapper" in text:
        return "Stage10"
    if any(s in text for s in ("stereo",)):
        return "Stage6"
    if "mca result" in text or "attached mca" in text or "partial-charge storage" in text or "partial_charge" in text or "molecule.properties" in text or "annotations" in text:
        return "Stage7C"
    if "_model" in text or "_environ" in text or "aimodels" in text or text.startswith("mca ") or "calculator/mca" in text:
        return "Stage8D"
    if any(s in text for s in ("_obmol", "_row2idx", "call_thermo", "get_thermo", "to_obmol", "read_one", "convert.py", "core_utils", "crystal", "draw.py", "pubchem", "rdworks", "_io/", "_io`/`core")):
        return "Stage7A"
    if "hidden" in text or "hide/recover" in text:
        return "Stage7B"
    if "constraint" in text:
        return "Stage4" if "angle" in text or "torsion" in text else "Stage7B"
    if "conformers" in text or "conformer.py" in text or "conformers/" in text:
        return "Stage8D" if "/" in text else "Stage7D"
    if any(s in text for s in ("forcefield", "force-field", "placement/radii", "session/workflow", "native_packing", "native_adapters", "native pure", "native structure")):
        return "Stage8B"
    if "search" in text or "smarts" in text or "recursive" in text:
        return "Stage8C"
    if "assembly" in text or "mol_assemble" in text or "statistics" in text or "mol_statistics" in text:
        return "Stage8E"
    if any(s in text for s in ("scientific", "calculator", "backendidentity", "state-composition", "staterequest", "molecularfacts", "provider", "state_rules")):
        return "Stage8A"
    if "elementproperties" in text or "ideal_bond_length" in text or "native packing structs" in text or "molblock" in text or "builder" in text or "parser/converter" in text or "ad hoc writes" in text or "identity" in text or "uid" in text:
        return "Stage3"
    if any(s in text for s in ("kekul", "aromatic", "valence", "charge, spin", "elements.py", "_metal_valence", "bond kind")):
        return "Stage5"
    if any(s in text for s in ("geometry", "graph", "ring/cycle", "distances", "internalcoordinates", "atomseq", "atompairs", "update_angles", "_angles")):
        return "Stage4"
    if "chemdata/" in text or "fragtemplete" in text:
        return "Stage11"
    if "_native/" in text:
        return "Stage10"
    if any(s in text for s in ("molecule", "atom", "bond", "core.py", "molreader", "molwriter", "read_mol", "molbundle", "notinsamemolecule", "objnotinmolecule", "public python")):
        return "Stage9"
    raise ValueError(f"Unclassified migration row {section}: {current}")


def migration_rows():
    active = False
    section = ""
    rows = []
    for number, line in enumerate(PLAN.read_text().splitlines(), 1):
        if line.startswith("## 9."):
            active = True
        elif line.startswith("## 10."):
            break
        if not active:
            continue
        if line.startswith("### 9."):
            section = line.split()[1]
        if not line.startswith("| ") or section not in {"9.1", "9.2", "9.3", "9.6"}:
            continue
        cells = [x.strip() for x in line.strip("|").split("|")]
        if cells[0].startswith("Current "):
            continue
        rows.append({"id": f"{section}-{sum(r['section'] == section for r in rows) + 1:03}", "section": section,
                     "plan_line": number, "current": cells[0], "disposition": cells[1:], "stage": migration_stage(section, cells[0])})
    # Mixed rows have several semantic owners. These explicit subdivisions
    # preserve the stage order instead of burying all work under a file move.
    splits = {
        "9.1-001": [("Stage3", "fact declarations and controlled storage"), ("Stage9", "sole Molecule class relocation after semantic extraction")],
        "9.1-002": [("Stage3", "sole Atom fact table and projections"), ("Stage5", "lazy chemistry"), ("Stage9", "sole Atom class relocation")],
        "9.1-003": [("Stage5", "coupled bond representation"), ("Stage7B", "constraint projections"), ("Stage9", "sole Bond and BondKind definitions")],
        "9.1-005": [("Stage3", "fact declarations and one table access path"), ("Stage5", "remove eager chemistry from dynamic writers"), ("Stage9", "remove superseded MolBlock implementation")],
        "9.1-006": [("Stage4", "angles/torsions/paths/pairs and constraint preservation"), ("Stage5", "Ring/JointRing read-only transition with writer migration")],
        "9.1-010": [("Stage4", "angle/torsion constraints before cache deletion"), ("Stage7B", "remaining coordinate/bond constraints and full audit")],
        "9.1-016": [("Stage5", "elemental/formal-charge/H calculation"), ("Stage7C", "partial-charge attachment"), ("Stage8A", "net-charge/spin scientific-state workflow")],
        "9.1-026": [("Stage4", "angle/torsion/pair caches"), ("Stage5", "ring caches"), ("Stage7A", "external-object caches")],
        "9.2-018": [("Stage8E", "single assembly consumer/schema"), ("Stage11", "Decision13 governed asset relocation or explicit deferral")],
        "9.2-025": [("Stage3", "single transport/schema reuse boundary"), ("Stage10", "physical Python/C++ transport relocation")],
        "9.2-026": [("Stage7A", "PyG translation boundary"), ("Stage9", "MolBundle collection owner")],
        "9.2-038": [("Stage7C", "remove duplicated attached MCA values"), ("Stage8D", "relocate the remaining inference operation")],
        "9.2-047": [("Stage4", "characterized cell matrix mathematics"), ("Stage7A", "six-parameter carrier and fresh toolkit adapter")],
        "9.2-070": [("Stage5", "named bond-perception assignment"), ("Stage7A", "lossless Open Babel translation"), ("Stage7B", "task-local topology selections")],
        "9.2-076": [("Stage6", "stereo transport translation"), ("Stage7A", "remaining RDKit conversion boundary")],
        "9.6-006": [("Stage7A", "conversion/PyG adapters"), ("Stage9", "MolBundle class relocation")],
        "9.6-010": [("Stage7A", "thermo adapter"), ("Stage8B", "build/optimize workflow"), ("Stage8C", "search operation")],
        "9.6-011": [("Stage4", "mathematical queries"), ("Stage5", "chemical queries"), ("Stage9", "thin carrier facades")],
        "9.6-017": [("Stage4", "remove angle/torsion cache update APIs after preserving constraints"), ("Stage5", "remove mutating Ring APIs atomically with assignment migration")],
        "9.6-028": [("Stage3", "Atom fact table"), ("Stage9", "remaining carrier membership/class relocation")],
        "9.6-030": [("Stage4", "angle/torsion/pair caches"), ("Stage5", "ring caches")],
        "9.6-037": [("Stage3", "one-shot factual builder contract"), ("Stage7A", "route I/O and toolkit conversion construction through builder")],
        "9.6-038": [("Stage3", "closed commits and ratcheted writer gate"), ("Stage5", "chemical writers"), ("Stage6", "stereo writers"), ("Stage7C", "result attachment writers"), ("Stage8B", "force-field writers"), ("Stage8D", "inference writers"), ("Stage12", "final empty writer allowlist")],
        "9.6-040": [("Stage3", "shared field declarations and existing transport reuse"), ("Stage7C", "selected-result packing"), ("Stage8B", "remove subsystem packing duplication"), ("Stage10", "physical shared transport/header migration")],
        "9.6-043": [("Stage4", "shared graph/cycle ownership"), ("Stage8C", "RI consumer ownership"), ("Stage10", "extension build relocation")],
        "9.6-044": [("Stage4", "pure geometry ownership"), ("Stage10", "extension build relocation")],
    }
    for row in rows:
        row["responsibilities"] = [{"stage": stage, "work": work} for stage, work in splits.get(row["id"], [(row["stage"], "complete the documented row disposition")])]
        row["completion_stage"] = row["responsibilities"][-1]["stage"]
        row["verification_map"] = "../stage_verification_map.json"
        row["verification_stage_keys"] = sorted({part["stage"] for part in row["responsibilities"] if part["stage"] != "Plan21"})
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records))


def main():
    global ROOT, PLAN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT, help="frozen source checkout to inspect/import")
    parser.add_argument("--runtime", action="store_true", help="also capture live facades, class identity and temporary-object behavior")
    args = parser.parse_args()
    ROOT = args.source_root.resolve()
    PLAN = ROOT / "plan/20_fact_rule_architecture/fact_rule_architecture_implementation_plan.md"
    args.output.mkdir(parents=True, exist_ok=True)
    tracked = git("ls-files", "-z").split("\0")
    python_paths = sorted(p for p in tracked if p.endswith(".py") and p.startswith(("hotpot/", "tests/")))
    records = static_capture(python_paths)
    docs = []
    native = []
    for path in tracked:
        full = ROOT / path
        if path.endswith(".md"):
            for number, line in enumerate(full.read_text().splitlines(), 1):
                for match in re.finditer(r"\b(?:from|import)\s+(hotpot(?:\.\w+)+)", line):
                    docs.append({"path": path, "line": number, "module": match.group(1), "planning_or_historical": path.startswith("plan/")})
        if path.startswith("hotpot/") and Path(path).suffix in {".h", ".hpp", ".cpp", ".c"}:
            for number, line in enumerate(full.read_text().splitlines(), 1):
                match = re.match(r"\s*(?:typedef\s+)?struct\s+(\w+)\s*(?::[^{}]+)?\{", line)
                if match:
                    native.append({"path": path, "line": number, "name": match.group(1), "stage": owner_stage(path)})
    rows = migration_rows()
    for name, values in records.items():
        write_jsonl(args.output / f"{name}.jsonl", values)
    write_jsonl(args.output / "documentation_imports.jsonl", docs)
    write_jsonl(args.output / "native_struct_candidates.jsonl", native)
    write_json(args.output / "migration_inventory.json", rows)
    runtime = runtime_capture() if args.runtime else None
    if runtime:
        write_json(args.output / "runtime_surface.json", runtime)
    write_jsonl(args.output / "interface_inventory.jsonl", interface_records(records["classes"], runtime))
    source = [{"path": p, "sha256": digest(ROOT / p)} for p in python_paths if p.startswith("hotpot/")]
    markers = Counter(marker for record in records["classes"] for marker in record["markers"])
    write_json(args.output / "manifest.json", {
        "schema_version": 1, "source_root": str(ROOT), "source_commit": git("rev-parse", "HEAD"), "production_tree": git("rev-parse", "HEAD:hotpot"),
        "capture_tool_sha256": digest(Path(__file__)),
        "plan_sha256": digest(PLAN), "source_files": source, "consumer_files": [{"path": p, "sha256": digest(ROOT / p)} for p in python_paths if p.startswith("tests/")],
        "python_executable": sys.executable, "python_version": sys.version,
        "counts": {"production_python_files": len(source), "class_definitions": len(records["classes"]), "contract_markers": dict(markers),
                   "contract_definitions": sum(bool(r["markers"]) for r in records["classes"]), "explicit_type_aliases": len(records["explicit_type_aliases"]),
                   "native_struct_candidates": len(native), "migration_rows": len(rows), "architecture_debt_entries": len(records["architecture_debt"]),
                   "direct_core_import_files": len({r["path"] for r in records["architecture_debt"] if r["rule"] == "legacy_core_import"})},
        "runtime_capture": args.runtime, "net_consolidation": {"measured_removed_or_merged": 0, "measured_added": 0, "status": "baseline only; proposed changes are not implemented reductions"},
        "limitations": ["Role labels are initial source-based classifications; ambiguous classes remain explicitly marked for review.",
                        "Static imports/calls/private writes are exact syntax observations, not exhaustive proof of dynamic/pickle consumers or source ownership.",
                        "Native struct scan records ordinary named definitions; macro-generated/anonymous schemas and ABI field layouts require component review.",
                        "New contract-specific tests and final API decisions are recorded separately; inventory capture is not their acceptance."],
        "outputs": {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "manifest.json"},
    })
    print(json.dumps({"output": str(args.output), "production_files": len(source), "contract_markers": dict(markers), "migration_rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
