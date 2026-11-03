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


_CORE_IMPORT_SOURCE = {
    "rule": "legacy_core_import", "path": "hotpot/cheminfo/convert.py",
    "scope": "", "target": "hotpot.cheminfo.core.Molecule", "local": "Molecule",
    "stage": "Stage9", "line": 10,
}
_CORE_IMPORT_DESTINATION = {
    **_CORE_IMPORT_SOURCE, "path": "hotpot/cheminfo/operations/conversion.py",
    "scope": "_molecule_type", "line": 20,
}


def test_approved_core_lookup_relocation_uses_one_existing_ceiling_slot():
    old, new = _CORE_IMPORT_SOURCE, _CORE_IMPORT_DESTINATION
    assert audit.ratchet_findings([new], [new], [old])["allowlist_growth"]
    assert not any(audit.stage7a_ratchet_findings([new], [new], [old], [old]).values())
    shifted = {**new, "line": 200}
    assert not any(audit.stage7a_ratchet_findings([shifted], [new], [old], [old]).values())
    assert not any(audit.stage7a_ratchet_findings([new], [new], [new], [old]).values())
    assert not any(audit.stage7a_ratchet_findings([], [], [new], [old]).values())
    assert audit.stage7a_ratchet_findings([new], [new], [], [old])["allowlist_growth"]
    assert audit.stage7a_ratchet_findings([new], [new], [old], [])["exceeds_frozen_inventory"]


@pytest.mark.parametrize("field,value", [
    ("path", "hotpot/cheminfo/operations/other.py"),
    ("scope", "other"),
    ("scope", "_molecule_type.nested"),
    ("target", "hotpot.cheminfo.core.Atom"),
    ("local", "OtherMolecule"),
    ("rule", "other_import"),
    ("kind", "other"),
])
def test_core_import_relocation_cannot_authorize_another_identity(field, value):
    old = _CORE_IMPORT_SOURCE
    changed = {**_CORE_IMPORT_DESTINATION, field: value}
    report = audit.stage7a_ratchet_findings([changed], [changed], [old], [old])
    assert report["allowlist_growth"]
    assert report["exceeds_frozen_inventory"]


@pytest.mark.parametrize("source_path", [
    "hotpot/cheminfo/core_utils.py", "hotpot/cheminfo/_io/_io.py", "hotpot/cheminfo/draw.py",
])
def test_other_retired_core_imports_supply_no_relocation_credit(source_path):
    old = {**_CORE_IMPORT_SOURCE, "path": source_path}
    new = _CORE_IMPORT_DESTINATION
    report = audit.stage7a_ratchet_findings([new], [new], [old], [old])
    assert report["allowlist_growth"]
    assert report["exceeds_frozen_inventory"]


def test_core_import_relocation_preserves_multiplicity_and_actual_allowlist_matching():
    old, new = _CORE_IMPORT_SOURCE, _CORE_IMPORT_DESTINATION
    report = audit.stage7a_ratchet_findings([old, new], [old, new], [old], [old])
    assert report["allowlist_growth"] and report["exceeds_frozen_inventory"]
    report = audit.stage7a_ratchet_findings([new, new], [new, new], [old], [old])
    assert len(report["allowlist_growth"]) == len(report["exceeds_frozen_inventory"]) == 1
    assert audit.stage7a_ratchet_findings([new, new], [new], [old], [old])["new_debt"]
    assert audit.stage7a_ratchet_findings([new], [old], [old], [old])["new_debt"]
    assert audit.stage7a_ratchet_findings([new], [old], [old], [old])["unretired_allowlist_entries"]
    assert audit.stage7a_ratchet_findings([], [new], [old], [old])["unretired_allowlist_entries"]
    assert audit.stage7a_ratchet_findings([old], [old], [new], [old])["allowlist_growth"]


def test_core_import_relocation_does_not_hide_other_frozen_inventory_growth():
    old, new = _CORE_IMPORT_SOURCE, _CORE_IMPORT_DESTINATION
    other = {**old, "path": "hotpot/cheminfo/operations/other.py"}
    report = audit.stage7a_ratchet_findings([new, other], [new, other], [old], [old])
    assert report["allowlist_growth"] == [audit.debt_key(other)]
    assert report["exceeds_frozen_inventory"] == [audit.debt_key(other)]


def test_relocated_core_import_is_captured_at_its_real_source_scope(tmp_path):
    path = _CORE_IMPORT_DESTINATION["path"]
    source = "def _molecule_type():\n    from ..core import Molecule\n    return Molecule\n"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(source)
    rows = audit.inventory.static_capture([path], root=tmp_path)["architecture_debt"]
    assert len(rows) == 1
    assert audit.debt_key(rows[0]) == audit.debt_key(_CORE_IMPORT_DESTINATION)
    assert rows[0]["stage"] == "Stage9"
    assert audit.layer_findings(path, source) == []


@pytest.mark.parametrize("source", [
    "from ..core import Molecule",
    "class _molecule_type:\n    from ..core import Molecule",
    "async def _molecule_type():\n    from ..core import Molecule",
    "def _molecule_type(argument):\n    from ..core import Molecule",
    "@decorator\ndef _molecule_type():\n    from ..core import Molecule",
    "def _molecule_type():\n    class Nested:\n        from ..core import Molecule",
    "def _molecule_type():\n    def nested():\n        from ..core import Molecule",
    "def _molecule_type():\n    from ..core import Molecule as Molecule",
    "def _molecule_type():\n    from ..core import Molecule as Other",
    "def _molecule_type():\n    from ..core import Molecule, Atom",
    "def _molecule_type():\n    from ..core import Atom",
    "def _molecule_type():\n    from .. import core",
])
def test_core_lookup_relocation_has_exact_function_scope_and_binding(source):
    assert audit.layer_findings(_CORE_IMPORT_DESTINATION["path"], source)


_SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS = tuple(
    (
        {"rule": "legacy_core_import",
         "path": "hotpot/cheminfo/calculator/state_resolver.py", "scope": "",
         "target": f"hotpot.cheminfo.core.{symbol}", "local": symbol,
         "stage": "Stage9", "line": 10},
        {"rule": "legacy_core_import",
         "path": ("hotpot/cheminfo/operations/calculation/scientific_state/facts.py"
                  if symbol == "Molecule"
                  else "hotpot/cheminfo/adapters/structure/scientific_state.py"), "scope": "",
         "target": f"hotpot.cheminfo.core.{symbol}", "local": symbol,
         "stage": "Stage9", "line": 9},
    )
    for symbol in ("Bond", "BondKind", "Molecule")
)


@pytest.mark.parametrize("old,new", _SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS)
def test_scientific_state_core_relocation_uses_only_its_existing_slot(old, new):
    assert audit.ratchet_findings([new], [new], [old])["allowlist_growth"]
    assert not any(audit.stage7a_ratchet_findings([new], [new], [old], [old]).values())
    shifted = {**new, "line": 200}
    assert not any(audit.stage7a_ratchet_findings([shifted], [new], [old], [old]).values())
    assert not any(audit.stage7a_ratchet_findings([new], [new], [new], [old]).values())
    assert not any(audit.stage7a_ratchet_findings([], [], [new], [old]).values())


_NATIVE_TRANSPORT_CORE_IMPORT_SOURCE = {
    "rule": "legacy_core_import", "path": "hotpot/cheminfo/forcefields/native.py",
    "scope": "", "target": "hotpot.cheminfo.core.Molecule", "local": "Molecule",
    "stage": "Stage9", "line": 39,
}
_NATIVE_TRANSPORT_CORE_IMPORT_DESTINATION = {
    **_NATIVE_TRANSPORT_CORE_IMPORT_SOURCE,
    "path": "hotpot/cheminfo/adapters/native/forcefields/native.py", "line": 41,
}


_COMMAND_CORE_IMPORT_SOURCE = {
    "rule": "legacy_core_import", "path": "hotpot/cheminfo/forcefields/cli.py",
    "scope": "", "target": "hotpot.cheminfo.core.Molecule", "local": "Molecule",
    "stage": "Stage9", "line": 46,
}
_COMMAND_CORE_IMPORT_DESTINATION = {
    **_COMMAND_CORE_IMPORT_SOURCE, "path": "hotpot/commands/ff.py", "line": 46,
}
_TRANSPORT_CORE_IMPORT_TRANSFERS = (
    (_NATIVE_TRANSPORT_CORE_IMPORT_SOURCE, _NATIVE_TRANSPORT_CORE_IMPORT_DESTINATION),
    (_COMMAND_CORE_IMPORT_SOURCE, _COMMAND_CORE_IMPORT_DESTINATION),
)


def test_all_fixed_core_relocations_preserve_both_historical_ceilings():
    pairs = ((_CORE_IMPORT_SOURCE, _CORE_IMPORT_DESTINATION),
             *_SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS,
             *_TRANSPORT_CORE_IMPORT_TRANSFERS)
    old = [source for source, _ in pairs]
    new = [destination for _, destination in pairs]
    for previous in (old, new, [new[0], *old[1:]], [old[0], *new[1:]]):
        assert not any(audit.stage7a_ratchet_findings(new, new, previous, old).values())
    assert not any(audit.stage7a_ratchet_findings([], [], new, old).values())


@pytest.mark.parametrize("old,new", _SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS)
@pytest.mark.parametrize("field,value", [
    ("path", "hotpot/cheminfo/adapters/structure/other.py"),
    ("scope", "extract_molecular_facts"),
    ("scope", "extract_molecular_facts.nested"),
    ("target", "hotpot.cheminfo.core.Atom"),
    ("local", "OtherCarrier"),
    ("rule", "other_import"),
    ("kind", "other"),
])
def test_scientific_state_relocation_rejects_another_identity(old, new, field, value):
    changed = {**new, field: value}
    report = audit.stage7a_ratchet_findings([changed], [changed], [old], [old])
    assert report["allowlist_growth"]
    assert report["exceeds_frozen_inventory"]


@pytest.mark.parametrize("old,new", _SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS)
@pytest.mark.parametrize("field,value", [
    ("path", "hotpot/cheminfo/calculator/other.py"),
    ("scope", "extract_molecular_facts"),
    ("target", "hotpot.cheminfo.core.Atom"),
    ("local", "OtherCarrier"),
])
def test_scientific_state_relocation_requires_its_exact_predecessor(old, new, field, value):
    other = {**old, field: value}
    report = audit.stage7a_ratchet_findings([new], [new], [other], [other])
    assert report["allowlist_growth"]
    assert report["exceeds_frozen_inventory"]


@pytest.mark.parametrize("old,new", _SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS)
def test_scientific_state_relocation_requires_a_slot_in_each_ceiling(old, new):
    assert audit.stage7a_ratchet_findings([new], [new], [], [old])["allowlist_growth"]
    assert audit.stage7a_ratchet_findings([new], [new], [old], [])["exceeds_frozen_inventory"]


@pytest.mark.parametrize("index", range(3))
def test_scientific_state_relocation_cannot_borrow_another_symbol_slot(index):
    pairs = _SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS
    new = pairs[index][1]
    other_sources = [source for i, (source, _) in enumerate(pairs) if i != index]
    report = audit.stage7a_ratchet_findings([new], [new], other_sources, other_sources)
    assert report["allowlist_growth"]
    assert report["exceeds_frozen_inventory"]


@pytest.mark.parametrize("old,new", _SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS)
def test_scientific_state_relocation_preserves_multiplicity_and_actual_matching(old, new):
    report = audit.stage7a_ratchet_findings([old, new], [old, new], [old], [old])
    assert report["allowlist_growth"] and report["exceeds_frozen_inventory"]
    report = audit.stage7a_ratchet_findings([new, new], [new, new], [old], [old])
    assert len(report["allowlist_growth"]) == len(report["exceeds_frozen_inventory"]) == 1
    assert audit.stage7a_ratchet_findings([new, new], [new], [old], [old])["new_debt"]
    assert audit.stage7a_ratchet_findings([new], [old], [old], [old])["new_debt"]
    assert audit.stage7a_ratchet_findings([new], [old], [old], [old])["unretired_allowlist_entries"]
    assert audit.stage7a_ratchet_findings([], [new], [old], [old])["unretired_allowlist_entries"]
    assert audit.stage7a_ratchet_findings([old], [old], [new], [old])["allowlist_growth"]


def test_scientific_state_relocation_carries_no_unrelated_growth_credit():
    old = [source for source, _ in _SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS]
    new = [destination for _, destination in _SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS]
    other = {**new[0], "path": "hotpot/cheminfo/adapters/structure/other.py"}
    report = audit.stage7a_ratchet_findings([*new, other], [*new, other], old, old)
    assert report["allowlist_growth"] == [audit.debt_key(other)]
    assert report["exceeds_frozen_inventory"] == [audit.debt_key(other)]


def test_scientific_state_relocated_imports_are_captured_at_the_real_source_scope(tmp_path):
    sources = {
        "hotpot/cheminfo/adapters/structure/scientific_state.py":
            "from hotpot.cheminfo.core import Bond, BondKind\n",
        "hotpot/cheminfo/operations/calculation/scientific_state/facts.py":
            "from hotpot.cheminfo.core import Molecule\n",
    }
    for path, source in sources.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True)
        target.write_text(source)
        assert audit.layer_findings(path, source) == []
    rows = audit.inventory.static_capture(sources, root=tmp_path)["architecture_debt"]
    expected = [new for _, new in _SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS]
    assert len(rows) == 3
    assert {audit.debt_key(row) for row in rows} == {audit.debt_key(row) for row in expected}
    assert {row["stage"] for row in rows} == {"Stage9"}


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


@pytest.mark.parametrize("old,new", _TRANSPORT_CORE_IMPORT_TRANSFERS)
def test_transport_and_command_core_relocations_consume_only_their_own_single_slots(old, new):
    assert audit.ratchet_findings([new], [new], [old])["allowlist_growth"]
    assert not any(audit.stage7a_ratchet_findings([new], [new], [old], [old]).values())
    assert not any(audit.stage7a_ratchet_findings([new], [new], [new], [old]).values())
    assert audit.stage7a_ratchet_findings([new], [new], [], [old])["allowlist_growth"]
    assert audit.stage7a_ratchet_findings([new], [new], [old], [])["exceeds_frozen_inventory"]
    duplicate = audit.stage7a_ratchet_findings([new, new], [new, new], [old], [old])
    assert len(duplicate["allowlist_growth"]) == len(duplicate["exceeds_frozen_inventory"]) == 1
    both = audit.stage7a_ratchet_findings([old, new], [old, new], [old], [old])
    assert both["allowlist_growth"] and both["exceeds_frozen_inventory"]
    other_sources = (
        _CORE_IMPORT_SOURCE,
        *(source for source, _ in _SCIENTIFIC_STATE_CORE_IMPORT_TRANSFERS),
        *(source for source, _ in _TRANSPORT_CORE_IMPORT_TRANSFERS if source != old),
        {**old, "path": "hotpot/cheminfo/forcefields/native_packing.py"},
        {**old, "scope": "nested"},
        {**old, "target": "hotpot.cheminfo.core.Bond", "local": "Bond"},
        {**old, "local": "OtherMolecule"},
    )
    for other in other_sources:
        report = audit.stage7a_ratchet_findings([new], [new], [other], [other])
        assert report["allowlist_growth"] and report["exceeds_frozen_inventory"]
    for field, value in (
        ("path", "hotpot/cheminfo/adapters/native/other.py"),
        ("scope", "nested"), ("target", "hotpot.cheminfo.core.Bond"),
        ("local", "OtherMolecule"),
    ):
        changed = {**new, field: value}
        report = audit.stage7a_ratchet_findings([changed], [changed], [old], [old])
        assert report["allowlist_growth"] and report["exceeds_frozen_inventory"]


@pytest.mark.parametrize("old,new", _TRANSPORT_CORE_IMPORT_TRANSFERS)
def test_transport_and_command_annotations_are_captured_at_their_real_scopes(tmp_path, old, new):
    path = new["path"]
    source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from hotpot.cheminfo.core import Molecule\n"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(source)
    rows = audit.inventory.static_capture([path], root=tmp_path)["architecture_debt"]
    assert len(rows) == 1
    assert audit.debt_key(rows[0]) == audit.debt_key(new)
    assert rows[0]["stage"] == "Stage9"
    assert audit.layer_findings(path, source) == []


@pytest.mark.parametrize("old,new", audit.SEARCH_CORE_IMPORT_TRANSFERS)
def test_search_core_bindings_transfer_only_the_exact_existing_slot(old, new):
    assert audit.ratchet_findings([new], [new], [old])["allowlist_growth"]
    assert not any(audit.stage7a_ratchet_findings([new], [new], [old], [old]).values())
    assert not any(audit.stage7a_ratchet_findings([new], [new], [new], [old]).values())
    assert audit.stage7a_ratchet_findings([new], [new], [], [old])["allowlist_growth"]
    assert audit.stage7a_ratchet_findings([new], [new], [old], [])["exceeds_frozen_inventory"]
    for actual in ([new, new], [old, new]):
        result = audit.stage7a_ratchet_findings(actual, actual, [old], [old])
        assert result["allowlist_growth"] and result["exceeds_frozen_inventory"]
    for field, value in (("path", "hotpot/cheminfo/operations/search/other.py"),
                         ("scope", "nested"), ("local", "OtherCarrier")):
        changed = {**new, field: value}
        result = audit.stage7a_ratchet_findings([changed], [changed], [old], [old])
        assert result["allowlist_growth"] and result["exceeds_frozen_inventory"]
    other_sources = [source for source, _ in audit.CORE_IMPORT_TRANSFERS if source != old]
    result = audit.stage7a_ratchet_findings([new], [new], other_sources, other_sources)
    assert result["allowlist_growth"] and result["exceeds_frozen_inventory"]


def test_search_core_bindings_are_captured_at_their_actual_eager_module_scope(tmp_path):
    path = "hotpot/cheminfo/operations/search/search.py"
    source = "from hotpot.cheminfo.core import Molecule, Atom, Bond\n"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text(source)
    rows = audit.inventory.static_capture([path], root=tmp_path)["architecture_debt"]
    expected = [new for _, new in audit.SEARCH_CORE_IMPORT_TRANSFERS]
    assert {audit.debt_key(row) for row in rows} == {audit.debt_key(row) for row in expected}
    assert len(rows) == 3
    assert {row["stage"] for row in rows} == {"Stage9"}
    assert audit.layer_findings(path, source) == []
