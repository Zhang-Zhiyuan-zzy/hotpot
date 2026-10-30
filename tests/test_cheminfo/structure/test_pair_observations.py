"""Preserve supplied Wiberg observations independently of generated pair views."""

from copy import copy, deepcopy
from dataclasses import replace
import hashlib
import json
import os
import pickle
import shlex
import subprocess
import sys
import sysconfig

import pytest

from hotpot import Molecule
from hotpot.cheminfo.forcefields.working_copy import _capture_working_source, _check_working_source
from hotpot.cheminfo.structure._commit import _FactTransaction
from tests.test_cheminfo.structure.test_fact_native import _ALLOCATOR_HELPER_SOURCE


def molecule():
    mol = Molecule()
    removed = mol.create_atom(atomic_number=6)
    mol.remove_atom(removed)
    for row in range(3):
        mol.create_atom(atomic_number=6, coordinates=(row, 0., 0.))
    mol.add_bond(0, 1)
    mol.add_bond(1, 2)
    return mol


def test_default_zero_is_not_a_supplied_observation_and_record_uses_rows():
    mol = molecule()
    pair = mol.atom_pairs[(mol.atoms[0], mol.atoms[2])]
    assert pair.wiberg_bond_order == 0
    assert not mol._pair_wiberg_bond_orders
    pair.wiberg_bond_order = 0.
    record, _ = mol._capture_factual_record()
    assert record.pair_wiberg_bond_orders == ((0, 2, 0.),)
    assert mol._pair_wiberg_bond_orders == {(mol.atoms[0]._uid, mol.atoms[2]._uid): 0.}
    assert mol.atom_pairs[(mol.atoms[2], mol.atoms[0])].wiberg_bond_order == 0.


@pytest.mark.parametrize('entries,message', [
    (((0, 1, 1.25, 99),), 'triples'),
    (((0, 1),), 'triples'),
    (((0, 0, 1.25),), 'distinct'),
    (((0, 1, 1.25), (1, 0, 1.5)), 'unique'),
    (((0, 3, 1.25),), 'range'),
])
def test_record_requires_exact_unique_in_range_pair_triples(entries, message):
    record, _ = molecule()._capture_factual_record()
    with pytest.raises(ValueError, match=message):
        replace(record, pair_wiberg_bond_orders=entries)


@pytest.mark.parametrize('clone', [copy, deepcopy, lambda mol: pickle.loads(pickle.dumps(mol))])
def test_source_values_survive_copy_and_current_pickle_with_uid_remap(clone):
    mol = molecule()
    pairs = mol.atom_pairs
    pairs[(mol.atoms[0], mol.atoms[1])].wiberg_bond_order = 0.
    pairs[(mol.atoms[0], mol.atoms[2])].wiberg_bond_order = 1.25
    other = clone(mol)
    assert len(other._pair_wiberg_bond_orders) == 2
    assert other.atom_pairs[(other.atoms[0], other.atoms[1])].wiberg_bond_order == 0.
    assert other.atom_pairs[(other.atoms[0], other.atoms[2])].attrs == [1.25, 3]
    other.atom_pairs[(other.atoms[0], other.atoms[2])].wiberg_bond_order = 2.
    assert pairs[(mol.atoms[0], mol.atoms[2])].wiberg_bond_order == 1.25


def test_merge_and_component_subset_preserve_supplied_values():
    source = molecule()
    source.atom_pairs[(source.atoms[0], source.atoms[2])].wiberg_bond_order = 1.25
    target = Molecule()
    target.create_atom(atomic_number=8)
    target.add_component(source)
    assert target.atom_pairs[(target.atoms[1], target.atoms[3])].wiberg_bond_order == 1.25
    parts = target.components
    assert sorted(len(part._pair_wiberg_bond_orders) for part in parts) == [0, 1]
    assert target._pair_wiberg_bond_orders is not source._pair_wiberg_bond_orders


def test_repeated_observation_and_topology_failure_restore_values_and_presence():
    mol = molecule()
    pair = mol.atom_pairs[(mol.atoms[0], mol.atoms[1])]
    other = mol.atom_pairs[(mol.atoms[0], mol.atoms[2])]
    pair.wiberg_bond_order = 0.
    initial = dict(mol._pair_wiberg_bond_orders)
    owner = mol._pair_wiberg_bond_orders
    original = KeyboardInterrupt('pair edit interrupted')
    with pytest.raises(KeyboardInterrupt) as caught:
        with _FactTransaction(mol, producer=1):
            pair.wiberg_bond_order = 1.
            other.wiberg_bond_order = 2.
            mol.remove_bond(mol.bonds[-1])
            pair.wiberg_bond_order = 3.
            raise original
    assert caught.value is original
    assert mol._pair_wiberg_bond_orders is owner
    assert mol._pair_wiberg_bond_orders == initial
    assert pair.wiberg_bond_order == 0.


def test_removed_member_observations_do_not_follow_compacted_or_reused_rows():
    mol = molecule()
    first, second, third = mol.atoms
    pairs = mol.atom_pairs
    pairs[(first, second)].wiberg_bond_order = 0.
    pairs[(second, third)].wiberg_bond_order = 1.25
    mol.remove_atom(first)
    assert mol._pair_wiberg_bond_orders == {(second._uid, third._uid): 1.25}
    added = mol.create_atom(atomic_number=8)
    assert mol.atom_pairs[(second, added)].wiberg_bond_order == 0
    record, _ = mol._capture_factual_record()
    assert record.pair_wiberg_bond_orders == ((0, 1, 1.25),)


def _dictionary_allocator_source():
    source = _ALLOCATOR_HELPER_SOURCE.replace(
        'static PyMemAllocatorEx previous;',
        'static PyMemAllocatorEx previous_mem, previous_obj;')
    source = source.replace('previous.', '((PyMemAllocatorEx *)ctx)->')
    source = source.replace('static size_t history[64];',
                            'static size_t history[64];\nstatic int history_domain[64];')
    source = source.replace('deny(size_t size)', 'deny(void *ctx, size_t size)')
    source = source.replace('deny(size)', 'deny(ctx, size)')
    source = source.replace('deny(count * size)', 'deny(ctx, count * size)')
    source = source.replace('if (history_count < 64) history[history_count++] = size;', '''
    if (history_count < 64) {
        history_domain[history_count] = ctx == &previous_mem ? 1 : 2;
        history[history_count++] = size;
    }''')
    source = source.replace('"allocation_sizes", sizes);',
                            '"allocation_sizes", sizes, "allocation_domains", domains);')
    source = source.replace('"{s:n,s:n,s:n,s:N}"', '"{s:n,s:n,s:n,s:N,s:N}"')
    source = source.replace('    return Py_BuildValue("{s:n,s:n,s:n,s:N,s:N}",', '''
    PyObject *domains = PyList_New(history_count);
    if (!domains) { Py_DECREF(sizes); return NULL; }
    for (size_t i = 0; i < history_count; ++i)
        PyList_SET_ITEM(domains, i, PyUnicode_FromString(history_domain[i] == 1 ? "MEM" : "OBJ"));
    return Py_BuildValue("{s:n,s:n,s:n,s:N,s:N}",''')
    source = source.replace('''    PyMem_GetAllocator(PYMEM_DOMAIN_MEM, &previous);
    PyMemAllocatorEx wrapper = {
        NULL, probe_malloc, probe_calloc, probe_realloc, probe_free
    };
    PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &wrapper);''', '''
    PyMem_GetAllocator(PYMEM_DOMAIN_MEM, &previous_mem);
    PyMem_GetAllocator(PYMEM_DOMAIN_OBJ, &previous_obj);
    PyMemAllocatorEx wrapper = {
        &previous_mem, probe_malloc, probe_calloc, probe_realloc, probe_free
    };
    PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &wrapper);
    wrapper.ctx = &previous_obj;
    PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &wrapper);''')
    return source


@pytest.fixture(scope='module')
def dictionary_allocator_helper(tmp_path_factory):
    directory = tmp_path_factory.mktemp('dictionary-allocator')
    source = directory / 'seed_alloc_probe.c'
    binary = directory / ('seed_alloc_probe' + sysconfig.get_config_var('EXT_SUFFIX'))
    source.write_text(_dictionary_allocator_source())
    argv = shlex.split(sysconfig.get_config_var('CC')) + [
        '-shared', '-fPIC', '-O2', '-I' + sysconfig.get_config_var('INCLUDEPY'),
        str(source), '-o', str(binary),
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    record = {'argv': argv, 'python': sys.executable, 'python_version': sys.version,
              'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    (directory / 'build.json').write_text(json.dumps(record, indent=2))
    assert result.returncode == 0, result.stderr
    record['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (directory / 'build.json').write_text(json.dumps(record, indent=2))
    return directory


@pytest.mark.parametrize('owner', ['pair', 'index'])
@pytest.mark.parametrize('operation', ['topology', 'forcefield'])
@pytest.mark.parametrize('exception_name', ['KeyboardInterrupt', 'MemoryError'])
def test_real_dictionary_bucket_allocation_rejection_preserves_owner_undo(
        dictionary_allocator_helper, owner, operation, exception_name):
    command = r'''
import json, sys
from itertools import combinations
sys.path.insert(0, sys.argv[1])
import seed_alloc_probe as alloc
from hotpot import Molecule
from hotpot.cheminfo.forcefields import working_copy
from hotpot.cheminfo.structure import _commit
from hotpot.cheminfo.structure._dependencies import _dependency_state
owner, operation, exception_name = sys.argv[2:]
mol = Molecule()
atoms = tuple(mol.create_atom(atomic_number=6, coordinates=(row, row % 2, row % 3))
              for row in range(20))
for row in range(19):
    mol.add_bond(row, row + 1)
pairs = mol.atom_pairs
members = tuple(combinations(atoms, 2))
for row, pair in enumerate(members[:128]):
    pairs[pair].wiberg_bond_order = float(row)
observations, index, table = mol._pair_wiberg_bond_orders, mol._topology_index, mol._atom_table
expected_observations, expected_index = dict(observations), dict(index)
angle = mol.angles[0]
degrees = angle.degrees
def state():
    # Save raw facts before a public read could repair faulty rollback state.
    status, seeds = _dependency_state(mol._atom_table)
    raw = (mol._atom_table.matrix().tobytes(), mol._atom_table.presence().tobytes(),
           mol._atom_table.sources().tobytes(), status.tobytes(), seeds,
           tuple(atom._uid for atom in mol.atoms),
           tuple((bond._uid, bond.atom1._uid, bond.atom2._uid) for bond in mol.bonds),
           mol._constraints)
    return raw, mol.atom_matrix.tobytes()
# Settle the lazy chemistry once, before the tested transaction starts.
mol.atom_matrix.tobytes()
before = state()

# Measure the actual CPython bucket request for this owner's dictionary shape,
# then show that denying it breaks the former clear/update restoration path.
shadow = dict(observations if owner == 'pair' else index)
saved = dict(shadow)
shadow.clear()
alloc.arm(0)
shadow.update(saved)
calibration = alloc.disarm()
bucket_size = max(calibration['allocation_sizes'])
shadow.clear()
control_failed = False
alloc.arm(bucket_size)
try:
    shadow.update(saved)
except MemoryError:
    control_failed = True
finally:
    control = alloc.disarm()
assert control_failed and control['rejected'] > 0

sentinel = {'KeyboardInterrupt': KeyboardInterrupt, 'MemoryError': MemoryError}[exception_name]('original')
caught = None
try:
    if operation == 'topology':
        with _commit._FactTransaction(mol, producer=1) as transaction:
            pairs[members[0]].wiberg_bond_order = 300.
            pairs[members[-1]].wiberg_bond_order = 400.
            mol._add_bond(0, 2)
            transaction.touch_topology()
            mol.remove_atom(atoms[-1])
            added = mol.create_atom(atomic_number=8)
            mol._add_bond(atoms[0], added)
            mol.atom_pairs[members[0]].wiberg_bond_order = 500.
            alloc.arm(bucket_size)
            raise sentinel
    else:
        source = working_copy._capture_working_source(mol)
        working = working_copy._hydrogenated_working_copy(mol, add_hydrogens=False, source=source)
        added = working.create_atom(atomic_number=1)
        working.add_bond(working.atoms[0], added)
        commit_stereo = _commit._commit_stereo
        def interrupt(current, source, payload):
            commit_stereo(current, source, payload)
            current.atom_pairs[tuple(current.atoms[:2])].wiberg_bond_order = 500.
            current._add_bond(0, 2)
            alloc.arm(bucket_size)
            raise sentinel
        _commit._commit_stereo = interrupt
        working_copy._commit_working_copy(mol, working, source=source)
except BaseException as error:
    caught = error
finally:
    allocation = alloc.disarm()
result = {
    'same_exception': caught is sentinel,
    'same_pair_owner': mol._pair_wiberg_bond_orders is observations,
    'pair_values_restored': observations == expected_observations,
    'same_index_owner': mol._topology_index is index,
    'index_entries_restored': index.keys() == expected_index.keys()
        and all(index.get(key) is value for key, value in expected_index.items()),
    'same_table': mol._atom_table is table,
    'facts_restored': state() == before,
    'original_view_current': bool(angle.degrees == degrees),
    'calibration': calibration, 'allocator_control': control, 'rollback_allocation': allocation,
}
print(json.dumps(result))
assert all(result[key] for key in ('same_exception', 'same_pair_owner', 'pair_values_restored',
    'same_index_owner', 'index_entries_restored', 'same_table', 'facts_restored', 'original_view_current'))
assert allocation['rejected'] == 0
'''
    argv = [sys.executable, '-c', command, str(dictionary_allocator_helper), owner, operation, exception_name]
    result = subprocess.run(argv, capture_output=True, text=True, env=os.environ)
    record = {'argv': argv, 'returncode': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr}
    name = f'{owner}-{operation}-{exception_name}.json'
    (dictionary_allocator_helper / name).write_text(json.dumps(record, indent=2))
    assert result.returncode == 0, result.stdout + result.stderr


def test_working_copy_source_guards_observations_carried_by_complete_commit():
    mol = molecule()
    source = _capture_working_source(mol)
    mol.atom_pairs[(mol.atoms[0], mol.atoms[2])].wiberg_bond_order = 0.
    with pytest.raises(RuntimeError, match='forcefield_inputs'):
        _check_working_source(mol, source)
