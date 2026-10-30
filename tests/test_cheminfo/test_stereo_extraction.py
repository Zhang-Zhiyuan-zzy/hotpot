"""Installed-aware contracts for the Stage6 stereo owner/capture extraction."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, replace
import gc
import importlib
import inspect
from pathlib import Path
from queue import Queue
from struct import pack
from threading import Barrier, Event, Thread
import weakref

import numpy as np
import pytest

from hotpot.cheminfo import stereo
from hotpot.cheminfo.adapters.structure import stereochemistry as stereo_adapter
from hotpot.cheminfo import Bond, Molecule
from hotpot.cheminfo.operations import stereochemistry as stereo_operations
from hotpot.cheminfo.rules.chemistry.stereochemistry import (
    algebra, alignment, assignment, coordinates, graph, validation,
)
from hotpot.cheminfo.search.native import _NativeTarget
from hotpot.cheminfo.structure import snapshots, stereo as contracts
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._dependencies import GROUP_INDEX, _FactStatus
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES


_CONTENTION = (
    "Stereochemistry alignment cannot acquire a distinct owner while a factual transaction is active"
)
_DRIFT = "Captured stereochemistry facts changed; retry the operation"


def _molecule(numbers, edges=(), *, hydrogens=None):
    mol = Molecule()
    counts = (0,) * len(numbers) if hydrogens is None else hydrogens
    for number, count in zip(numbers, counts):
        mol._create_atom(atomic_number=number, implicit_hydrogens=count)
    for first, second, order in edges:
        mol._add_bond(first, second, bond_order=order)
    mol._update_topology(clear_conformers=False)
    return mol


def _record(family="TH", value="even"):
    text = f"hotpot.stereo/2/{family}/{value}"
    return stereo.StereoRecord(
        stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,)),
        tuple(stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row) for row in range(1, 5)),
        text, None, "hotpot.stereo/2", text,
    )


def _stereo_molecule(*, coordination=False):
    numbers = (78, 9, 17, 35, 53) if coordination else (14, 6, 7, 8, 9)
    mol = _molecule(numbers, tuple((0, row, 1.0) for row in range(1, 5)))
    record = _record("SP", "1") if coordination else _record()
    stereo.set_stereo_payload(mol, stereo.StereoPayload(1, (record,), (), ()))
    return mol


def _start_thread(function):
    result = Queue()

    def run():
        try:
            result.put((True, function()))
        except BaseException as error:
            result.put((False, error))

    thread = Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def _finish_thread(worker, *, timeout=10):
    thread, result = worker
    thread.join(timeout)
    assert not thread.is_alive(), "bounded stereo concurrency check timed out"
    succeeded, value = result.get_nowait()
    if not succeeded:
        raise value
    return value


@contextmanager
def _held_lock(lock):
    entered, release = Event(), Event()

    def hold():
        with lock:
            entered.set()
            assert release.wait(10), "lock-holder release timed out"

    worker = _start_thread(hold)
    try:
        assert entered.wait(10), "lock-holder entry timed out"
        yield
    finally:
        release.set()
        _finish_thread(worker)


def _assert_owners_unlocked(*owners):
    def probe():
        observed = []
        for owner in owners:
            acquired = owner._mutation_lock.acquire(blocking=False)
            observed.append(acquired)
            if acquired:
                owner._mutation_lock.release()
        return observed

    assert _finish_thread(_start_thread(probe)) == [True] * len(owners)


def _need_h_refresh(mol):
    with mol._mutation_lock:
        mol._atom_table._group_status[:len(mol._atoms), GROUP_INDEX["IMPLICIT_H"]] = _FactStatus.NEEDS_RECALC


def _h_status(mol):
    return tuple(int(value) for value in mol._atom_table._group_status[:len(mol._atoms), GROUP_INDEX["IMPLICIT_H"]])


class _ItemsCallback(dict):
    def __init__(self, values, callback):
        super().__init__(values)
        self.callback = callback

    def items(self):
        self.callback()
        return super().items()


class _ArrayCallback:
    def __init__(self, values, callback):
        self.values = values
        self.callback = callback

    def __array__(self, dtype=None, copy=None):
        self.callback()
        return np.asarray(self.values, dtype=dtype)


def _module_tree(module):
    # inspect the active source origin, including an installed wheel's .py files.
    origin = Path(module.__spec__.origin)
    assert origin.is_file()
    return ast.parse(origin.read_text()), origin


def _runtime_imports(module):
    tree, _ = _module_tree(module)
    imports = []

    def visit(node):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            imports.append(importlib.util.resolve_name(name, module.__package__) if node.level else name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return imports


def _assert_exact_native_value(expected, actual):
    assert type(actual) is type(expected)
    if type(expected) is dict:
        assert tuple(actual) == tuple(expected)
        for key in expected:
            _assert_exact_native_value(expected[key], actual[key])
    elif type(expected) in (tuple, list):
        assert len(actual) == len(expected)
        for left, right in zip(expected, actual):
            _assert_exact_native_value(left, right)
    elif type(expected) is float:
        assert pack("!d", actual) == pack("!d", expected)
    else:
        assert actual == expected


# Exact source/installed predecessor fixture: virtual-H/target-raw.pickle,
# SHA256 290c85d766d77ff0a8f66a102193588cebb8e2e2c9fb84cda29074f891dc87fe.
# Embedded ordinary values keep this contract runnable without checkout artifacts.
_VIRTUAL_H_BASELINE = {'keys': (0, 1, 2),
 'edges': ((0, 1), (0, 2)),
 'atoms': ({'fields': {'atomic_number': 6,
                       'isotope': 0,
                       'formal_charge': 0,
                       'radical_electrons': 0,
                       'is_aromatic': False,
                       '_hotpot_stereo_row': 0},
            'full': {'degree': 2,
                     'connectivity': 2,
                     'valence': 2.0,
                     'ring_count': 0,
                     'ring_sizes': ()},
            'ligand': {'degree': 2,
                       'connectivity': 2,
                       'valence': 2.0,
                       'ring_count': 0,
                       'ring_sizes': ()},
            'object': None},
           {'fields': {'atomic_number': 1,
                       'isotope': 0,
                       'formal_charge': 0,
                       'radical_electrons': 0,
                       'is_aromatic': False,
                       '_hotpot_stereo_row': 1,
                       '_hotpot_stereo_h_slot': 0},
            'full': {'degree': 1,
                     'connectivity': 1,
                     'valence': 1.0,
                     'ring_count': 0,
                     'ring_sizes': ()},
            'ligand': {'degree': 1,
                       'connectivity': 1,
                       'valence': 1.0,
                       'ring_count': 0,
                       'ring_sizes': ()},
            'object': None},
           {'fields': {'atomic_number': 1,
                       'isotope': 0,
                       'formal_charge': 0,
                       'radical_electrons': 0,
                       'is_aromatic': False,
                       '_hotpot_stereo_row': 2,
                       '_hotpot_stereo_h_slot': 1},
            'full': {'degree': 1,
                     'connectivity': 1,
                     'valence': 1.0,
                     'ring_count': 0,
                     'ring_sizes': ()},
            'ligand': {'degree': 1,
                       'connectivity': 1,
                       'valence': 1.0,
                       'ring_count': 0,
                       'ring_sizes': ()},
            'object': None}),
 'bonds': ({'fields': {'bond_order': 1.0, 'is_aromatic': False},
            'bond_kind': 'single',
            'is_aromatic': False,
            'object': None},
           {'fields': {'bond_order': 1.0, 'is_aromatic': False},
            'bond_kind': 'single',
            'is_aromatic': False,
            'object': None})}


def test_active_module_owners_are_acyclic_and_rules_have_no_live_dependencies():
    modules = (contracts, snapshots, algebra, graph, coordinates, validation,
               assignment, alignment, stereo_adapter, stereo_operations, stereo)
    observed = {module.__name__: _runtime_imports(module) for module in modules}
    pure = (algebra, graph, coordinates, validation, assignment, alignment)
    forbidden = (
        "hotpot.cheminfo.core", "hotpot.cheminfo.operations", "hotpot.cheminfo.adapters",
        "hotpot.cheminfo.search", "hotpot.cheminfo.stereo", "hotpot.cheminfo._native",
        "rdkit", "openbabel",
    )
    for module in pure:
        assert not any(name == prefix or name.startswith(prefix + ".")
                       for name in observed[module.__name__] for prefix in forbidden)
        tree, _ = _module_tree(module)
        assert not any(isinstance(node, ast.Name) and node.id in {"Molecule", "Atom", "Bond"}
                       for node in ast.walk(tree))
    assert set(observed[contracts.__name__]) == {"__future__", "dataclasses", "enum"}
    # Runtime owner edges are resolved from the active modules, not repository paths.
    visiting, done = set(), set()

    def visit(name):
        assert name not in visiting, f"stereo owner import cycle at {name}"
        if name in done:
            return
        visiting.add(name)
        for dependency in observed[name]:
            if dependency in observed:
                visit(dependency)
        visiting.remove(name)
        done.add(name)

    for name in observed:
        visit(name)


def test_the_facade_has_only_five_forwarders_and_no_private_scientific_aliases():
    tree, origin = _module_tree(stereo)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert tuple(node.name for node in functions) == (
        "get_stereo_payload", "set_stereo_payload", "validate_stereochemistry",
        "assign_stereochemistry", "align_stereochemistry",
    )
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
    for function in functions:
        body = function.body[1:] if isinstance(function.body[0], ast.Expr) else function.body
        assert len(body) == 1
        statement = body[0]
        call = statement.value if isinstance(statement, (ast.Return, ast.Expr)) else None
        assert isinstance(call, ast.Call), (origin, function.name)
    for name in ("_remap_stereo_payload", "_gate6_projection", "_parse_level2_descriptor",
                 "_transform_level2_descriptor", "_finite_coordinate_array"):
        assert not hasattr(stereo, name)
    assert inspect.getmodule(algebra._remap_stereo_payload) is algebra
    assert inspect.getmodule(graph._gate6_carrier_pair_equivalence) is graph
    assert inspect.getmodule(coordinates._gate4_coordinate_assignments) is coordinates


def test_exact_snapshot_and_relocated_class_rosters_have_no_extra_owner_types():
    expected = {
        contracts: set(stereo.__all__[:18]),
        snapshots: {"_BondingSnapshot", "_AromaticAssignment", "_StereoSnapshot", "_Gate6Graph"},
        algebra: {"_Level2DescriptorFamily", "_Level2Descriptor", "_Level2AlgebraResult",
                  "_Gate7CoordinationSite", "_AlignmentMatch"},
        graph: {"_Gate6Equivalence", "_Gate6SemanticContext"},
        coordinates: {"_Gate7CoordinateCandidate"},
        validation: set(), assignment: set(), alignment: set(),
        stereo_adapter: set(), stereo_operations: set(), stereo: set(),
    }
    for module, names in expected.items():
        tree, _ = _module_tree(module)
        assert {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)} == names
    assert tuple(field.name for field in fields(snapshots._StereoSnapshot)) == (
        "payload", "atom_keys", "bonds",
    )
    assert tuple(field.name for field in fields(snapshots._Gate6Graph)) == (
        "target_spec", "directed_dative_edges", "real_atom_count", "virtual_hydrogens",
    )


def test_owner_only_capture_and_guard_do_not_wait_for_conformer_lock():
    mol = _stereo_molecule()

    def capture_and_check():
        capture = stereo_operations._capture_stereo_owner(mol)
        with stereo_operations._stereo_owner_scope(mol):
            stereo_adapter._check_stereo_capture(mol, capture)
        return capture[1].dependencies

    with _held_lock(mol._conformers._mutation_lock):
        assert _finish_thread(_start_thread(capture_and_check)) == ()


def test_exact_predecessor_target_packing_and_real_native_mapping():
    mol = _molecule((6, 1), ((0, 1, 1.0),), hydrogens=(1, 0))
    fact = stereo.AtomTransportRecord(1, 1, None, None, None, "test")
    stereo.set_stereo_payload(mol, stereo.StereoPayload(1, (), (fact,), ()))
    projected, issues = stereo_operations._gate6_projection(mol)
    assert issues == ()
    packed = stereo_adapter._pack_gate6_target(projected)
    _assert_exact_native_value(_VIRTUAL_H_BASELINE, packed)
    mappings = tuple(_NativeTarget(packed).iter_mappings(graph._gate6_query_spec(projected, {0: 0, 1: 2})))
    assert {tuple(sorted(mapping.items())) for mapping in mappings} == {
        ((0, 0), (1, 2), (2, 1)),
    }
    assert projected.real_atom_count == 2
    assert projected.virtual_hydrogens == ((2,), ())
    assert not any(key in projected.target_spec["keys"] for key in (id(atom) for atom in mol.atoms))


def test_projection_and_pack_preserve_signed_zero_bond_order():
    mol = _molecule((6, 6), ((0, 1, 1.0),))
    mol._bonds[0]._attrs[0] = -0.0
    projected, issues = stereo_operations._gate6_projection(mol)
    assert issues == ()
    packed = stereo_adapter._pack_gate6_target(projected)
    assert pack("!d", packed["bonds"][0]["fields"]["bond_order"]) == pack("!d", -0.0)


@pytest.mark.parametrize("coordination", (False, True))
def test_same_owner_alignment_reuses_capture_preparation_validation_and_target(monkeypatch, coordination):
    mol = _stereo_molecule(coordination=coordination)
    calls = {"capture": 0, "projection": 0, "H": 0, "validation": 0, "native": 0, "pack": 0}

    def count(owner, name, key):
        original = getattr(owner, name)

        def counted(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)

    for owner, name, key in (
        (stereo_operations, "_capture_stereo_snapshot", "capture"),
        (stereo_operations, "_gate6_projection", "projection"),
        (stereo_operations, "_require_implicit_hydrogens", "H"),
        (stereo_operations, "_validate_graph_stereochemistry", "validation"),
        (_NativeTarget, "__init__", "native"),
        (stereo_operations, "_pack_gate6_target", "pack"),
    ):
        count(owner, name, key)
    result = stereo.align_stereochemistry(mol, mol, dict(enumerate(range(5))))
    assert result.is_equivalent and result.issues == ()
    assert calls == {"capture": 1, "projection": 1, "H": 1, "validation": 1,
                     "native": 0 if coordination else 1, "pack": 0 if coordination else 1}


def test_same_owner_alignment_keeps_both_issue_prefix_positions():
    mol = _molecule((6,))
    record = replace(_record(), descriptor="hotpot.stereo/2/TH/invalid")
    stereo.set_stereo_payload(mol, stereo.StereoPayload(1, (record,), (), ()))
    result = stereo.align_stereochemistry(mol, mol, dict(enumerate(range(5))))
    messages = [issue.message for issue in result.issues]
    reference = [message.removeprefix("reference: ") for message in messages if message.startswith("reference: ")]
    target = [message.removeprefix("target: ") for message in messages if message.startswith("target: ")]
    assert reference and reference == target
    assert messages.index("reference: " + reference[0]) < messages.index("target: " + target[0])


def test_one_pack_and_one_real_target_serve_all_pair_queries_without_retention(monkeypatch):
    mol = _stereo_molecule()
    references = {"snapshots": [], "graphs": [], "targets": [], "callables": []}
    counts = {"pack": 0, "native": 0, "queries": 0}
    original_capture = stereo_operations._capture_stereo_snapshot
    original_pack = stereo_operations._pack_gate6_target
    original_init = _NativeTarget.__init__
    original_iter = _NativeTarget.iter_mappings
    original_rule = stereo_operations._gate6_stereogenicity_issues

    def captured(owner):
        result = original_capture(owner)
        references["snapshots"].append(weakref.ref(result[0]))
        return result

    def packed(projected):
        counts["pack"] += 1
        references["graphs"].append(weakref.ref(projected))
        return original_pack(projected)

    def initialized(target, spec):
        counts["native"] += 1
        references["targets"].append(weakref.ref(target))
        original_init(target, spec)

    def mappings(target, query):
        counts["queries"] += 1
        return original_iter(target, query)

    def rule(*args):
        references["callables"].append(weakref.WeakMethod(args[-1]))
        return original_rule(*args)

    monkeypatch.setattr(stereo_operations, "_capture_stereo_snapshot", captured)
    monkeypatch.setattr(stereo_operations, "_pack_gate6_target", packed)
    monkeypatch.setattr(_NativeTarget, "__init__", initialized)
    monkeypatch.setattr(_NativeTarget, "iter_mappings", mappings)
    monkeypatch.setattr(stereo_operations, "_gate6_stereogenicity_issues", rule)
    carrier_references = [weakref.ref(value) for value in (mol, *mol._atoms, *mol._bonds)]
    report = stereo.validate_stereochemistry(mol)
    assert report.is_valid
    assert counts == {"pack": 1, "native": 1, "queries": 6}
    gc.collect()
    assert all(items and all(item() is None for item in items) for items in references.values())
    del mol
    gc.collect()
    assert all(reference() is None for reference in carrier_references)


def test_capture_keeps_former_outside_endpoint_alive_only_for_the_operation():
    mol = _molecule((6, 7), ((0, 1, 1.0),))
    outside = _molecule((8,))
    endpoint = outside._atoms[0]
    endpoint_reference = weakref.ref(endpoint)
    mol._bonds[0]._atoms = (mol._atoms[0], endpoint)
    capture = stereo_operations._capture_stereo_owner(mol)
    snapshot = capture[0]
    mol._bonds[0]._atoms = (mol._atoms[0], mol._atoms[1])
    del endpoint, outside
    gc.collect()
    assert endpoint_reference() is not None
    assert snapshot.bonds[0][1] == id(endpoint_reference())
    del capture
    gc.collect()
    assert endpoint_reference() is None
    assert isinstance(snapshot.bonds[0][1], int)


@pytest.mark.parametrize("mutation", (
    "payload_schema", "record_order", "atom_order", "bond_identity", "partition",
    "bond_order", "bond_kind", "bond_direction", "endpoint_order",
))
def test_deferred_raw_guard_rejects_consumed_drift_before_H_or_native(monkeypatch, mutation):
    mol = _stereo_molecule()
    payload = stereo.get_stereo_payload(mol)
    if mutation == "record_order":
        payload = replace(payload, records=(*payload.records, replace(payload.records[0], descriptor="opaque")))
        stereo.set_stereo_payload(mol, payload)
    capture = stereo_operations._capture_stereo_owner(mol)
    bond = mol._bonds[0]
    with mol._mutation_lock:
        if mutation == "payload_schema":
            stereo.set_stereo_payload(mol, replace(payload, schema_version=47))
        elif mutation == "record_order":
            stereo.set_stereo_payload(mol, replace(payload, records=payload.records[::-1]))
        elif mutation == "atom_order":
            mol._atoms.reverse()
        elif mutation == "bond_identity":
            mol._bonds[0] = Bond(bond.atom1, bond.atom2, bond_order=bond.bond_order)
        elif mutation == "partition":
            # Moving the last active row preserves the flattened bond sequence.
            mol._hided_covalent_bonds.append(mol._bonds.pop())
        elif mutation == "bond_order":
            bond._attrs[0] = 2.0
        elif mutation == "bond_kind":
            bond._bond_kind = type(bond.bond_kind).DOUBLE
        elif mutation == "bond_direction":
            bond._bond_direction = "wedge"
        else:
            bond._atoms = (bond.atom2, bond.atom1)

    def forbidden(*args, **kwargs):
        pytest.fail("raw drift reached deferred H/scalar/native work")

    monkeypatch.setattr(stereo_operations, "_require_implicit_hydrogens", forbidden)
    monkeypatch.setattr(stereo_operations, "_stereo_prepared_fields", forbidden)
    monkeypatch.setattr(_NativeTarget, "__init__", forbidden)
    # The check must not manufacture a second raw snapshot.
    monkeypatch.setattr(stereo_adapter, "_StereoSnapshot", forbidden)
    with pytest.raises(RuntimeError) as caught:
        stereo_operations._gate6_projection(mol, capture=capture)
    assert str(caught.value) == _DRIFT


@pytest.mark.parametrize("field,before,after", (
    ("bond_order", 0.0, -0.0),
    ("bond_direction", False, 0),
))
def test_raw_guard_compares_consumed_scalar_types_and_float_bits(field, before, after):
    mol = _molecule((6, 6), ((0, 1, 1.0),))
    bond = mol._bonds[0]
    if field == "bond_order":
        bond._attrs[0] = before
    else:
        bond._bond_direction = before
    capture = stereo_operations._capture_stereo_owner(mol)
    with stereo_operations._stereo_owner_scope(mol):
        if field == "bond_order":
            bond._attrs[0] = after
        else:
            bond._bond_direction = after
        with pytest.raises(RuntimeError) as caught:
            stereo_adapter._check_stereo_capture(mol, capture)
    assert str(caught.value) == _DRIFT


def test_owner_token_mismatch_keeps_existing_error_before_preparation(monkeypatch):
    mol = _stereo_molecule()
    capture = stereo_operations._capture_stereo_owner(mol)
    mol._owner_token = object()

    def forbidden(*args, **kwargs):
        pytest.fail("foreign owner reached H preparation")

    monkeypatch.setattr(stereo_operations, "_require_implicit_hydrogens", forbidden)
    with pytest.raises(RuntimeError) as caught:
        stereo_operations._gate6_projection(mol, capture=capture)
    assert str(caught.value) == "Molecular source owner changed; recalculate before committing"


def test_no_candidate_does_not_prepare_check_or_construct_native(monkeypatch):
    mol = _molecule((6,))
    opaque = replace(_record(), descriptor="foreign-namespace/opaque")
    stereo.set_stereo_payload(mol, stereo.StereoPayload(47, (opaque,), (), ()))

    def forbidden(*args, **kwargs):
        pytest.fail("ineligible payload reached deferred projection work")

    for name in ("_require_implicit_hydrogens", "_check_stereo_capture", "_stereo_prepared_fields", "_pack_gate6_target"):
        monkeypatch.setattr(stereo_operations, name, forbidden)
    monkeypatch.setattr(_NativeTarget, "__init__", forbidden)
    report = stereo.assign_stereochemistry(mol)
    assert report.payload is stereo.get_stereo_payload(mol)
    assert report.derived == ()
    assert report.unresolved


def test_invalid_coordination_head_still_projects_before_gate2_without_native(monkeypatch):
    mol = _stereo_molecule(coordination=True)
    payload = stereo.get_stereo_payload(mol)
    stereo.set_stereo_payload(mol, replace(payload, records=(replace(payload.records[0], descriptor="hotpot.stereo/2/SP/bad"),)))
    calls = []
    original_h = stereo_operations._require_implicit_hydrogens
    original_gate2 = stereo_operations._validate_gate2_stereochemistry

    def prepared(owner):
        calls.append("H")
        return original_h(owner)

    def validated(*args, **kwargs):
        assert calls == ["H"]
        calls.append("Gate2")
        return original_gate2(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid coordination-only payload constructed a native target")

    monkeypatch.setattr(stereo_operations, "_require_implicit_hydrogens", prepared)
    monkeypatch.setattr(stereo_operations, "_validate_gate2_stereochemistry", validated)
    monkeypatch.setattr(_NativeTarget, "__init__", forbidden)
    report = stereo.validate_stereochemistry(mol)
    assert not report.is_valid
    assert calls == ["H", "Gate2"]
    assert report.issues[0].code is stereo.StereoIssueCode.INVALID_DESCRIPTOR


def test_preflight_returns_all_raw_issues_before_deferred_fields(monkeypatch):
    mol = _molecule((6, 7, 8), ((0, 1, 1.0),))
    fact = stereo.AtomTransportRecord(0, None, None, None, None, "test")
    stereo.set_stereo_payload(mol, stereo.StereoPayload(1, (), (fact, fact, replace(fact, atom=3)), ()))
    mol._add_bond(1, 2, bond_kind="dative", bond_direction="sideways")
    mol._bonds[0]._attrs[0] = float("nan")
    mol._hided_covalent_bonds.append(Bond(mol._atoms[0], mol._atoms[1], bond_order=1.0))
    loop = Bond(mol._atoms[1], mol._atoms[2], bond_order=1.0)
    loop._atoms = (mol._atoms[1], mol._atoms[1])
    mol._hided_covalent_bonds.append(loop)

    def forbidden(*args, **kwargs):
        pytest.fail("preflight issues reached H/aromatic/native work")

    monkeypatch.setattr(stereo_operations, "_require_implicit_hydrogens", forbidden)
    monkeypatch.setattr(stereo_operations, "_stereo_prepared_fields", forbidden)
    monkeypatch.setattr(_NativeTarget, "__init__", forbidden)
    projected, issues = stereo_operations._gate6_projection(mol)
    assert projected is None
    assert [issue.message for issue in issues] == [
        "duplicate atom transport facts for row 0",
        "atom transport fact references out-of-range row 3",
        "represented bond (0, 1) has non-finite order",
        "dative bond (1, 2) has unknown direction 'sideways'",
        "parallel represented bonds share edge (0, 1)",
        "represented graph contains self-loop at row 1",
    ]


def test_mapping_array_and_native_callbacks_run_after_stereo_scopes_release(monkeypatch):
    reference = _stereo_molecule()
    target = reference.copy()
    captures, callbacks = [], []
    original_capture = stereo_operations._capture_stereo_snapshot
    original_native = _NativeTarget.iter_mappings

    def captured(owner):
        captures.append(owner)
        return original_capture(owner)

    def mapped():
        assert captures == [reference, target]
        _assert_owners_unlocked(reference, target)
        assert stereo.get_stereo_payload(reference).records
        callbacks.append("mapping")

    def native(owner, query):
        _assert_owners_unlocked(reference, target)
        assert stereo.get_stereo_payload(target).records
        callbacks.append("native")
        return original_native(owner, query)

    monkeypatch.setattr(stereo_operations, "_capture_stereo_snapshot", captured)
    monkeypatch.setattr(_NativeTarget, "iter_mappings", native)
    result = stereo.align_stereochemistry(reference, target, _ItemsCallback(dict(enumerate(range(5))), mapped))
    assert result.is_equivalent
    assert callbacks[0] == "mapping" and "native" in callbacks

    completed = []
    original_graph = stereo_operations._validate_graph_stereochemistry

    def validated(*args, **kwargs):
        result = original_graph(*args, **kwargs)
        completed.append("graph complete")
        return result

    def array():
        assert completed == ["graph complete"]
        _assert_owners_unlocked(reference, target)
        callbacks.append("array")

    monkeypatch.setattr(stereo_operations, "_validate_graph_stereochemistry", validated)
    values = np.asarray(((0, 0, 0), (1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)), dtype=np.float64)
    before = values.tobytes()
    stereo.validate_stereochemistry(reference, coordinates=_ArrayCallback(values, array))
    assert callbacks[-1] == "array"
    assert values.tobytes() == before


def test_global_graph_issue_suppresses_array_conversion_and_native(monkeypatch):
    mol = _molecule((14, 6, 7, 8, 9, 7, 29), tuple((0, row, 1.0) for row in range(1, 5)))
    mol._add_bond(5, 6, bond_kind="dative", bond_direction="sideways")
    stereo.set_stereo_payload(mol, stereo.StereoPayload(1, (_record(),), (), ()))

    def forbidden(*args, **kwargs):
        pytest.fail("a global preflight issue reached array/H/native work")

    monkeypatch.setattr(stereo_operations, "_require_implicit_hydrogens", forbidden)
    monkeypatch.setattr(_NativeTarget, "__init__", forbidden)
    report = stereo.validate_stereochemistry(mol, coordinates=_ArrayCallback(None, forbidden))
    assert any(issue.record_index is None and "dative" in issue.message for issue in report.issues)


def test_unconsumed_atomic_fields_are_selected_late_and_priority_reuses_prepared_Z():
    reference = _stereo_molecule()
    target = reference.copy()

    def change_target_element():
        target._atoms[1].atomic_number = 16

    alignment_report = stereo.align_stereochemistry(
        reference, target, _ItemsCallback(dict(enumerate(range(5))), change_target_element),
    )
    assert alignment_report.is_equivalent
    assert target._atoms[1].atomic_number == 16
    values = np.asarray(((0, 0, 0), (1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)), dtype=np.float64)
    expected = stereo.assign_stereochemistry(reference, coordinates=values)

    def change_after_preparation():
        _assert_owners_unlocked(reference)
        reference._atoms[1].atomic_number = 53

    assigned = stereo.assign_stereochemistry(reference, coordinates=_ArrayCallback(values, change_after_preparation))
    assert assigned.derived and assigned.derived == expected.derived
    assert reference._atoms[1].atomic_number == 53


@pytest.mark.parametrize("outer_owner", ("reference", "target", "third"))
def test_distinct_owner_alignment_uses_nonblocking_scopes_for_any_tracked_transaction(monkeypatch, outer_owner):
    reference, target, third = _stereo_molecule(), _stereo_molecule(), _molecule((6,))
    outer = {"reference": reference, "target": target, "third": third}[outer_owner]
    scopes = []
    original_scope = stereo_operations._stereo_owner_scope

    @contextmanager
    def recorded(owner, *, nonblocking=False):
        scopes.append((owner, nonblocking))
        with original_scope(owner, nonblocking=nonblocking):
            yield

    monkeypatch.setattr(stereo_operations, "_stereo_owner_scope", recorded)

    def run():
        with _FactTransaction(outer, producer=ATOM_SOURCE_CODES["explicit"]):
            result = stereo.align_stereochemistry(reference, target, dict(enumerate(range(5))))
            assert outer._mutation_lock._is_owned()
            return result

    assert _finish_thread(_start_thread(run)).is_equivalent
    assert scopes and all(nonblocking for _, nonblocking in scopes)
    assert [owner for owner, _ in scopes[:2]] == [reference, target]
    _assert_owners_unlocked(reference, target, third)


def test_same_owner_alignment_can_reenter_under_an_unrelated_outer_transaction(monkeypatch):
    mol, third = _stereo_molecule(), _molecule((6,))
    scopes = []
    original_scope = stereo_operations._stereo_owner_scope

    @contextmanager
    def recorded(owner, *, nonblocking=False):
        scopes.append(nonblocking)
        with original_scope(owner, nonblocking=nonblocking):
            yield

    monkeypatch.setattr(stereo_operations, "_stereo_owner_scope", recorded)
    with _FactTransaction(third, producer=ATOM_SOURCE_CODES["explicit"]):
        assert stereo.align_stereochemistry(mol, mol, dict(enumerate(range(5)))).is_equivalent
        assert third._mutation_lock._is_owned()
    assert scopes and not any(scopes)


@pytest.mark.parametrize("outer_owner", ("reference", "target", "third"))
def test_initial_contention_fails_before_mapping_H_or_native_and_retains_outer_lock(monkeypatch, outer_owner):
    reference, target, third = _stereo_molecule(), _stereo_molecule(), _molecule((6,))
    outer = {"reference": reference, "target": target, "third": third}[outer_owner]
    contended = reference if outer_owner == "target" else target
    callbacks = []
    mapping = _ItemsCallback(dict(enumerate(range(5))), lambda: callbacks.append("mapping"))

    def forbidden(*args, **kwargs):
        pytest.fail("initial capture contention reached H/native work")

    monkeypatch.setattr(stereo_operations, "_require_implicit_hydrogens", forbidden)
    monkeypatch.setattr(_NativeTarget, "__init__", forbidden)

    def run():
        with _FactTransaction(outer, producer=ATOM_SOURCE_CODES["explicit"]):
            with pytest.raises(RuntimeError) as caught:
                stereo.align_stereochemistry(reference, target, mapping)
            assert outer._mutation_lock._is_owned()
            return str(caught.value)

    with _held_lock(contended._mutation_lock):
        assert _finish_thread(_start_thread(run)) == _CONTENTION
        _assert_owners_unlocked(outer)
    assert callbacks == []
    _assert_owners_unlocked(reference, target, third)


@pytest.mark.parametrize("outer_owner", ("reference", "target", "third"))
def test_late_contention_uses_the_same_nonblocking_route_at_deferred_preparation(monkeypatch, outer_owner):
    reference, target, third = _stereo_molecule(), _stereo_molecule(), _molecule((6,))
    outer = {"reference": reference, "target": target, "third": third}[outer_owner]
    contended = reference if outer_owner == "target" else target
    requested, entered, release = Event(), Event(), Event()
    events = []
    _need_h_refresh(reference)
    original_h = stereo_operations._require_implicit_hydrogens
    original_init = _NativeTarget.__init__

    def prepared(owner):
        events.append("H:reference" if owner is reference else "H:target")
        return original_h(owner)

    def initialized(owner, spec):
        events.append("native")
        original_init(owner, spec)

    def hold_late():
        assert requested.wait(10), "mapping callback did not request late contention"
        with contended._mutation_lock:
            entered.set()
            assert release.wait(10), "late lock-holder release timed out"

    def mapping_callback():
        events.append("mapping")
        requested.set()
        assert entered.wait(10), "late lock-holder did not acquire the deferred owner"

    monkeypatch.setattr(stereo_operations, "_require_implicit_hydrogens", prepared)
    monkeypatch.setattr(_NativeTarget, "__init__", initialized)
    holder = _start_thread(hold_late)

    def run():
        with _FactTransaction(outer, producer=ATOM_SOURCE_CODES["explicit"]):
            with pytest.raises(RuntimeError) as caught:
                stereo.align_stereochemistry(reference, target,
                    _ItemsCallback(dict(enumerate(range(5))), mapping_callback))
            assert outer._mutation_lock._is_owned()
            return str(caught.value)

    try:
        assert _finish_thread(_start_thread(run)) == _CONTENTION
    finally:
        requested.set()
        release.set()
        _finish_thread(holder)
    assert events == (["mapping"] if contended is reference else ["mapping", "H:reference", "native"])
    if contended is target:
        assert _h_status(reference) == (int(_FactStatus.CURRENT),) * len(reference._atoms)
    _assert_owners_unlocked(reference, target, third)


@pytest.mark.parametrize("escape_outer_transaction", (False, True))
def test_late_drift_keeps_prior_side_effects_and_respects_existing_outer_rollback(monkeypatch, escape_outer_transaction):
    reference, target = _stereo_molecule(), _stereo_molecule()
    _need_h_refresh(reference)
    before = _h_status(reference)
    payload = stereo.get_stereo_payload(target)
    events = []
    original_h = stereo_operations._require_implicit_hydrogens
    original_init = _NativeTarget.__init__

    def prepared(owner):
        events.append("H:reference" if owner is reference else "H:target")
        return original_h(owner)

    def initialized(owner, spec):
        events.append("native")
        original_init(owner, spec)

    def changed():
        events.append("mapping")
        stereo.set_stereo_payload(target, replace(payload, schema_version=47))

    monkeypatch.setattr(stereo_operations, "_require_implicit_hydrogens", prepared)
    monkeypatch.setattr(_NativeTarget, "__init__", initialized)

    def call():
        return stereo.align_stereochemistry(reference, target,
            _ItemsCallback(dict(enumerate(range(5))), changed))

    with pytest.raises(RuntimeError) as caught:
        if escape_outer_transaction:
            with _FactTransaction(reference, producer=ATOM_SOURCE_CODES["explicit"]):
                call()
        else:
            call()
    assert str(caught.value) == _DRIFT
    assert events == ["mapping", "H:reference", "native"]
    assert stereo.get_stereo_payload(target).schema_version == 47
    expected = before if escape_outer_transaction else (int(_FactStatus.CURRENT),) * len(reference._atoms)
    assert _h_status(reference) == expected


def test_invalid_mapping_keeps_coordination_projection_then_validation_and_issue_order(monkeypatch):
    reference, target = _stereo_molecule(coordination=True), _stereo_molecule(coordination=True)
    for owner in (reference, target):
        payload = stereo.get_stereo_payload(owner)
        stereo.set_stereo_payload(owner, replace(payload, records=(replace(payload.records[0], descriptor="hotpot.stereo/2/SP/bad"),)))
    events = []
    original_h = stereo_operations._require_implicit_hydrogens
    original_validation = stereo_operations._validate_graph_stereochemistry

    def prepared(owner):
        events.append("H:reference" if owner is reference else "H:target")
        return original_h(owner)

    def validated(owner, **context):
        events.append("validate:reference" if owner is reference else "validate:target")
        return original_validation(owner, **context)

    monkeypatch.setattr(stereo_operations, "_require_implicit_hydrogens", prepared)
    monkeypatch.setattr(stereo_operations, "_validate_graph_stereochemistry", validated)
    result = stereo.align_stereochemistry(reference, target, _ItemsCallback({}, lambda: events.append("mapping")))
    assert events == ["mapping", "H:reference", "H:target", "validate:reference", "validate:target"]
    sides = ["reference" if issue.message.startswith("reference: ") else
             "target" if issue.message.startswith("target: ") else "mapping" for issue in result.issues]
    assert sides == ["mapping", "mapping", "reference", "target"]


def test_reversed_identity_pairs_capture_in_reference_target_order_outside_callbacks(monkeypatch):
    lower, higher = sorted((_stereo_molecule(), _stereo_molecule()), key=id)
    captures = []
    original_capture = stereo_operations._capture_stereo_snapshot

    def captured(owner):
        captures.append(owner)
        return original_capture(owner)

    monkeypatch.setattr(stereo_operations, "_capture_stereo_snapshot", captured)
    for reference, target in ((higher, lower), (lower, higher)):
        captures.clear()

        def callback():
            assert captures == [reference, target]
            _assert_owners_unlocked(reference, target)

        result = _finish_thread(_start_thread(lambda: stereo.align_stereochemistry(
            reference, target, _ItemsCallback(dict(enumerate(range(5))), callback))))
        assert result.is_equivalent


def test_opposite_tracked_outer_transactions_do_not_add_a_cross_owner_wait(monkeypatch):
    from threading import get_ident

    lower, higher = sorted((_stereo_molecule(), _stereo_molecule()), key=id)
    captured_together = Barrier(2)
    seen = set()
    original_capture = stereo_operations._capture_stereo_snapshot

    def captured(owner):
        result = original_capture(owner)
        identity = get_ident()
        if identity not in seen:
            seen.add(identity)
            captured_together.wait(timeout=10)
        return result

    monkeypatch.setattr(stereo_operations, "_capture_stereo_snapshot", captured)

    def run(reference, target):
        with _FactTransaction(reference, producer=ATOM_SOURCE_CODES["explicit"]):
            try:
                report = stereo.align_stereochemistry(reference, target, dict(enumerate(range(5))))
            except RuntimeError as error:
                assert str(error) == _CONTENTION
                return "contention"
            assert report.is_equivalent
            return "success"

    workers = [_start_thread(lambda: run(lower, higher)), _start_thread(lambda: run(higher, lower))]
    try:
        outcomes = [_finish_thread(worker) for worker in workers]
    finally:
        captured_together.abort()
    assert "contention" in outcomes
    _assert_owners_unlocked(lower, higher)
