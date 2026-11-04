"""Reject regressions while the legacy areas are migrated incrementally."""

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "_plan20_architecture", ROOT / "plan/20_fact_rule_architecture/audit_architecture.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_repository_architecture_ratchet():
    report = audit.audit(base_ref=os.environ.get("HOTPOT_ARCHITECTURE_BASE_REF", "HEAD"))
    assert report["status"] == "passed", report


def test_source_scan_includes_package_and_test_roots(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    sources = {"hotpot/__init__.py", "hotpot/_cli.py", "hotpot/cheminfo/leaf.py",
               "tests/test_root.py", "tests/cheminfo/test_leaf.py"}
    for name in sources | {"hotpot/reference.json", "README.md"}:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    assert set(audit.source_paths(tmp_path)) == sources


def test_retirement_cannot_be_hidden_by_editing_line_numbers_or_allowlist():
    old = {"path": "hotpot/cheminfo/core.py", "scope": "Molecule.__init__",
           "rule": "forbidden_retained_state_write", "target": "self._rings", "line": 10}
    shifted = {**old, "line": 80}
    assert not any(audit.ratchet_findings([shifted], [old], [old]).values())
    assert audit.ratchet_findings([], [old], [old])["unretired_allowlist_entries"]
    assert not any(audit.ratchet_findings([], [], [old]).values())
    assert audit.ratchet_findings([old], [old], [])["allowlist_growth"]
    assert audit.ratchet_findings([old, old], [old], [old])["new_debt"]


@pytest.mark.parametrize("source", [
    "import rdkit.Chem",
    "from openbabel import openbabel",
    "from ..graph import _graph_native",
    "from ..operations.chemistry import calculate",
    "def convert():\n    from ..adapters.openbabel import conversion",
    "def calculate():\n    from ..rules.chemistry import valence",
])
def test_structure_cannot_acquire_execution_dependencies(source):
    assert audit.layer_findings("hotpot/cheminfo/structure/atom.py", source)


def test_thin_structure_facade_can_enter_named_operation_locally():
    source = "def calculate(self):\n    from ..operations.chemistry import calculate\n    return calculate(self)"
    assert audit.layer_findings("hotpot/cheminfo/structure/atom.py", source) == []


@pytest.mark.parametrize("layer", ["rules", "adapters"])
@pytest.mark.parametrize("source", [
    "def run(subject):\n    subject.atomic_number = 6",
    "def run(subject):\n    subject.attrs[0] = 6",
    "def run(subject):\n    del subject._stereo_payload",
    "def run(subject):\n    object.__setattr__(subject, '_atom_table', None)",
    "def run(subject, field):\n    setattr(subject, field, 6)",
    "def run(subject):\n    subject._atoms.append(6)",
    "def run(subject):\n    subject.coordinates.fill(0)",
    "def run(subject):\n    subject.create_atom(atomic_number=6)",
    "def run(subject):\n    subject.updata_atoms(atomic_number=6)",
    "def run(subject):\n    subject.x = 1",
    "def run(subject):\n    subject.id = 10",
    "def run(subject):\n    subject.n = 2",
    "def run(subject):\n    subject.x_constraint = True",
    "def run(subject):\n    subject._pair_wiberg_bond_orders[(0, 1)] = 1.25",
    "def run(subject):\n    subject._pair_wiberg_bond_orders.clear()",
    "def run(subject):\n    object.__setattr__(subject, '_pair_wiberg_bond_orders', {})",
])
def test_rules_and_adapters_cannot_write_canonical_facts(layer, source):
    assert audit.layer_findings(f"hotpot/cheminfo/{layer}/example.py", source)


_VALUE_INITIALIZERS = (
    ("hotpot/cheminfo/rules/mathematics/geometry/object.py", "Point", "__init__",
     'object.__setattr__(self, "coordinates", (1., 2., 3.))'),
    ("hotpot/cheminfo/adapters/structure/geometry.py", "BondRingFrameWorkspace", "__post_init__",
     'object.__setattr__(self, "coordinates", coordinate_snapshot)'),
    ("hotpot/cheminfo/rules/mathematics/internal_coordinates.py", "InternalCoordinates", "__init__",
     'self.coordinates = coordinates'),
)


@pytest.mark.parametrize('path,cls,method,statement', _VALUE_INITIALIZERS)
def test_reviewed_numerical_values_can_initialize_their_own_fields(path, cls, method, statement):
    source = f'class {cls}:\n    def {method}(self):\n        {statement}\n'
    assert audit.layer_findings(path, source) == []


@pytest.mark.parametrize('path,cls,method,initialization', _VALUE_INITIALIZERS)
@pytest.mark.parametrize('statement', [
    'subject.coordinates = coordinates',
    'self.mol.coordinates = coordinates',
    'object.__setattr__(foreign, "coordinates", coordinates)',
    'object.__setattr__(self.mol, "coordinates", coordinates)',
    'object.__setattr__(self, field, coordinates)',
    'object.__setattr__(self, "charge", 1)',
    'self.coordinates[:] = 1',
    'self.coordinates.fill(0)',
    'del self.coordinates',
])
def test_value_initialization_does_not_exempt_foreign_dynamic_or_buffer_writes(
        path, cls, method, initialization, statement):
    source = f'class {cls}:\n    def {method}(self):\n        {initialization}\n        {statement}\n'
    assert audit.layer_findings(path, source)


@pytest.mark.parametrize('path,cls,method,statement', _VALUE_INITIALIZERS)
@pytest.mark.parametrize('changed_scope', ['path', 'class', 'method', 'nested'])
def test_value_initialization_permission_has_exact_path_and_direct_method_scope(
        path, cls, method, statement, changed_scope):
    if changed_scope == 'path':
        path = path.replace('.py', '_other.py')
    elif changed_scope == 'class':
        cls += 'Other'
    elif changed_scope == 'method':
        method = 'update'
    source = f'class {cls}:\n    def {method}(self):\n        {statement}\n'
    if changed_scope == 'nested':
        source = f'class {cls}:\n    def {method}(self):\n        def nested():\n            {statement}\n'
    assert audit.layer_findings(path, source)


@pytest.mark.parametrize("source", [
    "from hotpot import Molecule",
    "from hotpot.cheminfo.structure import Atom",
    "from hotpot.cheminfo.structure.atom import Atom",
    "from hotpot.cheminfo.operations import chemistry",
    "from hotpot.cheminfo.adapters import openbabel",
    "from ..structure import _commit",
    "from ..structure import atom as atom_module",
    "from hotpot.cheminfo.structure import _builder as builder",
])
def test_rules_cannot_import_carriers_or_execution(source):
    assert audit.layer_findings("hotpot/cheminfo/rules/example.py", source)


@pytest.mark.parametrize("snapshot_import", [
    "from hotpot.cheminfo.structure.snapshots import MolecularSnapshot",
    "from ..structure import snapshots",
    "from ..structure.snapshots import MolecularSnapshot as Snapshot",
])
def test_rules_can_read_values_and_modify_task_local_arrays(snapshot_import):
    source = snapshot_import + """
def calculate(snapshot):
    output = snapshot.coordinates.copy()
    output[:] += 1
    return output
"""
    assert audit.layer_findings("hotpot/cheminfo/rules/example.py", source) == []


_CRYSTAL_DELEGATE = """class Crystal:
    @staticmethod
    def _matrix_to_params(matrix):
        from ..rules.mathematics.cell import _matrix_to_params
        return _matrix_to_params(matrix)
"""


def test_crystal_value_can_delegate_locally_to_the_one_cell_math_function():
    assert audit.layer_findings("hotpot/cheminfo/structure/crystal.py", _CRYSTAL_DELEGATE) == []


@pytest.mark.parametrize("changed_scope", [
    "path", "class", "method", "module", "nested", "nested_class", "class_method",
    "instance_method", "signature", "async", "nested_owner",
])
def test_crystal_math_delegate_permission_has_exact_path_and_direct_static_method(changed_scope):
    path = "hotpot/cheminfo/structure/crystal.py"
    source = _CRYSTAL_DELEGATE
    if changed_scope == "path":
        path = path.replace("crystal.py", "other.py")
    elif changed_scope == "class":
        source = source.replace("class Crystal:", "class Other:")
    elif changed_scope == "method":
        source = source.replace("def _matrix_to_params", "def other")
    elif changed_scope == "module":
        source = "from ..rules.mathematics.cell import _matrix_to_params"
    elif changed_scope == "nested":
        source = source.replace("        from", "        def nested():\n            from")
    elif changed_scope == "nested_class":
        source = source.replace("        from", "        class Nested:\n            from")
    elif changed_scope == "class_method":
        source = source.replace("@staticmethod", "@classmethod")
    elif changed_scope == "instance_method":
        source = source.replace("    @staticmethod\n", "")
    elif changed_scope == "signature":
        source = source.replace("(matrix)", "(self, matrix)")
    elif changed_scope == "async":
        source = source.replace("    def ", "    async def ")
    elif changed_scope == "nested_owner":
        source = "def outer():\n" + "\n".join("    " + line for line in source.splitlines())
    assert audit.layer_findings(path, source)


@pytest.mark.parametrize("statement", [
    "from ..rules.mathematics.cell import _matrix_to_params as other",
    "from ..rules.mathematics.cell import _matrix_to_params as _matrix_to_params",
    "from ..rules.mathematics.cell import _matrix_to_params, other",
    "from ..rules.mathematics.cell import other",
    "from ..rules.mathematics.cell import *",
    "from ..rules.chemistry import valence",
    "from ..adapters.openbabel import crystal",
    "from ..operations.chemistry import calculate",
    "from ..graph import _graph_native",
    "from . import _fact_native",
    "from openbabel import openbabel",
    "import rdkit.Chem",
])
def test_crystal_math_delegate_does_not_exempt_other_rules_adapters_or_execution(statement):
    source = _CRYSTAL_DELEGATE.replace(
        "from ..rules.mathematics.cell import _matrix_to_params", statement)
    assert audit.layer_findings("hotpot/cheminfo/structure/crystal.py", source)


def test_crystal_delegate_does_not_change_the_existing_local_native_commit_boundary():
    source = "def commit():\n    from . import _fact_native\n"
    assert audit.layer_findings("hotpot/cheminfo/structure/_commit.py", source) == []


@pytest.mark.parametrize("path", [
    "hotpot/cheminfo/convert.py",
    "hotpot/cheminfo/operations/conversion.py",
    "hotpot/cheminfo/operations/search/search.py",
    "hotpot/cheminfo/adapters/structure/scientific_state.py",
    "hotpot/commands/ff.py",
    "hotpot/plugins/xtb/stage.py",
    "tests/test_carrier.py",
])
@pytest.mark.parametrize("source", [
    "from hotpot.cheminfo.core import Molecule",
    "import hotpot.cheminfo.core as carrier",
    "def query():\n    from hotpot.cheminfo.core import Atom",
    "if TYPE_CHECKING:\n    from hotpot.cheminfo.core import Bond",
    "importlib.import_module('hotpot.cheminfo.core')",
    "__import__('hotpot.cheminfo.core.Molecule')",
])
def test_old_core_bindings_and_entry_calls_are_unconditionally_rejected(path, source):
    findings = audit.active_core_findings(path, source)
    assert findings and {row["rule"] for row in findings} == {"active_legacy_core_reference"}


def test_core_module_cannot_return_as_an_empty_or_reexport_shim():
    for source in ("", "from .structure.molecule import Molecule"):
        assert audit.active_core_findings("hotpot/cheminfo/core.py", source)


def test_relative_old_core_import_is_rejected_but_fixture_text_is_not_execution():
    assert audit.active_core_findings("hotpot/cheminfo/operations/leaf.py", "from ..core import Molecule")
    fixture = 'source = "from hotpot.cheminfo.core import Molecule"'
    assert audit.active_core_findings("tests/test_guard.py", fixture) == []


def test_forged_old_allowlist_entry_cannot_hide_active_core_import():
    source = "from hotpot.cheminfo.core import Molecule"
    record = {"rule": "legacy_core_import", "path": "hotpot/cheminfo/leaf.py",
              "scope": "", "target": "hotpot.cheminfo.core.Molecule", "local": "Molecule"}
    assert not any(audit.frozen_ratchet_findings([record], [record], [record], [record]).values())
    assert audit.active_core_findings(record["path"], source)


def test_retired_import_slot_cannot_supply_relocation_or_growth_credit():
    old = {"rule": "legacy_core_import", "path": "hotpot/cheminfo/convert.py",
           "scope": "", "target": "hotpot.cheminfo.core.Molecule", "local": "Molecule"}
    moved = {**old, "path": "hotpot/cheminfo/operations/conversion.py", "scope": "_molecule_type"}
    findings = audit.frozen_ratchet_findings([moved], [moved], [old], [old])
    assert findings["allowlist_growth"] and findings["exceeds_frozen_inventory"]


@pytest.mark.parametrize("name", list(audit.TYPE_OWNERS))
def test_type_has_one_exact_declaring_owner(name):
    owner = audit.TYPE_OWNERS[name]
    record = {"name": name, "path": owner.rsplit(".", 1)[0].replace(".", "/") + ".py",
              "symbol": owner}
    assert name not in audit.carrier_definition_findings([record])
    assert audit.carrier_definition_findings([])[name] == []
    wrong = {**record, "path": "hotpot/cheminfo/structure/wrong.py",
             "symbol": "hotpot.cheminfo.structure.wrong." + name}
    assert name in audit.carrier_definition_findings([wrong])
    assert name in audit.carrier_definition_findings([record, wrong])


def test_legacy_relocation_allowances_are_absent():
    assert not any(name.endswith("CORE_IMPORT_TRANSFERS") for name in vars(audit))
    assert "CORE_IMPORT_DESTINATION" not in vars(audit)


@pytest.mark.parametrize("source", [
    "import hotpot.cheminfo.bundle",
    "from hotpot.cheminfo.bundle import MolBundle",
    "import importlib\nimportlib.import_module('hotpot.cheminfo.bundle')",
    "__import__('hotpot.cheminfo.bundle')",
])
def test_retired_bundle_imports_and_entries_cannot_return(source):
    findings = audit.active_core_findings("tests/test_bundle_consumer.py", source)
    assert any(item["rule"] == "active_legacy_bundle_reference" for item in findings)


def test_retired_bundle_module_is_rejected_even_without_imports():
    assert audit.active_core_findings("hotpot/cheminfo/bundle.py", "") == [
        {"path": "hotpot/cheminfo/bundle.py", "line": 1, "rule": "legacy_bundle_module"}
    ]


@pytest.mark.parametrize("module,label", [
    ("hotpot.cheminfo.geometry", "geometry"),
    ("hotpot.cheminfo.graph", "graph"),
    ("hotpot.cheminfo._native", "native_transport"),
])
@pytest.mark.parametrize("entry", [
    "import {module}",
    "from {module} import native",
    "importlib.import_module('{module}.native')",
])
def test_retired_native_owners_cannot_return(module, label, entry):
    findings = audit.active_core_findings(
        "hotpot/cheminfo/operations/consumer.py", entry.format(module=module))
    assert {row["rule"] for row in findings} == {f"active_legacy_{label}_reference"}
    assert audit.active_core_findings(module.replace(".", "/") + "/__init__.py", "")


@pytest.mark.parametrize("path,source", [
    ("hotpot/cheminfo/structure/symmetry.py", "from ..rules.mathematics import symmetry as _math"),
    ("hotpot/cheminfo/structure/_crystal.py", "from ..rules.mathematics import symmetry as _affine"),
    ("hotpot/cheminfo/structure/_crystal.py", "from ..rules.mathematics.cell import _cell_matrix, _validate_matrix, _check_metric"),
])
def test_exact_crystal_value_imports_are_not_scientific_execution(path, source):
    assert audit.layer_findings(path, source) == []
    assert audit.layer_findings(path.replace('.py', '_other.py'), source)
    assert audit.layer_findings(path, 'def nested():\n    ' + source)
    assert audit.layer_findings(path, source.replace('mathematics', 'chemistry'))
    assert audit.layer_findings(path, source.replace('import ', 'import other, '))


def test_current_crystal_thin_operation_facade_retains_external_and_eager_boundaries():
    source = "class Crystal:\n    def expand(self):\n        from ..operations.crystal_generation import _expand_crystal\n        return _expand_crystal(self, (2, 1, 1))\n"
    path = 'hotpot/cheminfo/structure/crystal.py'
    assert audit.layer_findings(path, source) == []
    assert audit.layer_findings(path, 'from ..operations.crystal_generation import _expand_crystal')
    assert audit.layer_findings(path, source.replace('from ..operations.crystal_generation import _expand_crystal', 'from openbabel import openbabel'))
    assert audit.layer_findings(path, source.replace('from ..operations.crystal_generation import _expand_crystal', 'from . import _fact_native'))
    assert audit.layer_findings(path, source.replace('from ..operations.crystal_generation import _expand_crystal', 'from ..operations.forcefields._native import calculate'))
    assert audit.layer_findings(path, source.replace('from ..operations.crystal_generation import _expand_crystal', 'from ..operations.forcefields import _forcefields_native'))
    assert audit.layer_findings(path, source.replace('from ..operations.crystal_generation import _expand_crystal', 'from ..adapters.crystal import expand'))
    assert audit.layer_findings(path, _CRYSTAL_DELEGATE.replace('..rules.mathematics.cell', '..operations.crystal')) == []


@pytest.mark.parametrize("method,arguments,operation", [
    ("read", "crystal", "_matrix_to_params"),
    ("write", "crystal, value", "_set_parameter"),
])
def test_crystal_scalar_property_factory_has_exact_local_operation_scope(method, arguments, operation):
    path = "hotpot/cheminfo/structure/crystal.py"
    statement = f"from ..operations.crystal import {operation}"
    source = f"def _parameter(index):\n    def {method}({arguments}):\n        {statement}\n"
    assert audit.layer_findings(path, source) == []
    # Ordinary carriers already allow local operation imports; Crystal's
    # stricter property-factory scope does not change that independent rule.
    assert audit.layer_findings(path.replace("crystal.py", "other.py"), source) == []
    invalid = [
        (path, source.replace("def _parameter(index)", "def other(index)")),
        (path, source.replace("def _parameter(index)", "def _parameter(other)")),
        (path, source.replace(f"def {method}(", "def other(")),
        (path, source.replace(f"({arguments})", "(other)")),
        (path, source.replace("        from", "        def nested():\n            from")),
        (path, source.replace(statement, f"from ..operations.crystal import {operation}, other")),
        (path, source.replace("..operations.crystal", "..operations.chemistry")),
        (path, source.replace(statement, "from openbabel import openbabel")),
        (path, source.replace(statement, "from ..adapters.crystal import value")),
        (path, source.replace(statement, "from . import _fact_native")),
        (path, statement),
    ]
    for changed_path, changed_source in invalid:
        assert audit.layer_findings(changed_path, changed_source), changed_source


@pytest.mark.parametrize('method,receiver,field', [
    ('encode', 'data', 'bond_order'), ('encode', 'data', 'bond_kind'),
    ('encode', 'data', 'bond_direction'), ('_foreign', 'normalized', 'bond_order'),
])
def test_only_fresh_reviewed_pyg_transport_receivers_can_initialize_fields(method, receiver, field):
    path = 'hotpot/cheminfo/adapters/_pyg_molecule.py'
    source = f'def {method}(subject):\n    from torch_geometric.data import Data\n    {receiver} = Data()\n    {receiver}.{field} = 1\n'
    assert audit.layer_findings(path, source) == []
    invalid = [
        (path.replace('.py', '_other.py'), source),
        (path, source.replace(f'def {method}(', 'def other(')),
        (path, source.replace(f'{receiver} = Data()', f'{receiver} = subject')),
        (path, source.replace(f'{receiver} = Data()', f'{receiver} = Data()\n    {receiver} = subject')),
        (path, source.replace('from torch_geometric.data import Data', 'from other import Data')),
        (path, source.replace(f'{receiver} = Data()', f'Data = fake\n    {receiver} = Data()')),
        (path, source.replace(f'{receiver}.{field} =', f'subject.{field} =')),
        (path, source.replace(f'{receiver}.{field} = 1', f'{receiver}.{field}[:] = 1')),
        (path, source.replace(f'{receiver}.{field} = 1', f'del {receiver}.{field}')),
        (path, source.replace(f'{receiver}.{field} = 1', f'def nested():\n        {receiver}.{field} = 1')),
    ]
    for changed_path, changed_source in invalid:
        assert audit.layer_findings(changed_path, changed_source), changed_source
