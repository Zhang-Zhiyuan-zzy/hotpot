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
