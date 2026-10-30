"""Native ingress evidence, including real post-store rollback boundaries.

The earlier ndarray-subclass __setitem__ hooks cannot intercept pointer writes.
The operation-local failpoints below interrupt the actual native stores instead.
"""

from dataclasses import replace
import gc
import hashlib
import json
import os
import shlex
import subprocess
import sys
import sysconfig
import weakref

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.structure._atom_table import _AtomFactTable
from hotpot.cheminfo.structure._commit import (
    _FactTransaction, _commit_atom_facts, _fact_engine, _native_fail_after,
)
from hotpot.cheminfo.structure._dependencies import (
    _dependency_state, _ensure_group, _seed,
)
from hotpot.cheminfo.structure._fields import FIELD_INDEX
from hotpot.cheminfo.structure._identity import _capture_source
from hotpot.cheminfo.structure._mutation import _normalize_atom_facts


def molecule():
    mol = Molecule()
    for _ in range(3):
        mol.create_atom(atomic_number=6)
    return mol


def state(mol):
    table = mol._atom_table
    statuses, seeds = _dependency_state(table)
    return table.matrix().tobytes(), table.presence().tobytes(), table.sources().tobytes(), statuses.tobytes(), seeds


@pytest.mark.parametrize('rows', [0, (0, 2)])
@pytest.mark.parametrize('boundary', ['value', 'source', 'presence'])
@pytest.mark.parametrize('exception', [KeyboardInterrupt, MemoryError])
def test_standalone_uses_same_native_post_store_restorer(rows, boundary, exception):
    table = _AtomFactTable(np.arange(57., dtype=np.float64).reshape(3, 19), presence=0)
    before = table.matrix().tobytes(), table.presence().tobytes(), table.sources().tobytes()
    with _native_fail_after(boundary, exception=exception), pytest.raises(exception):
        table.update(rows, {'x': -0.0, 'formal_charge': -2})
    assert (table.matrix().tobytes(), table.presence().tobytes(), table.sources().tobytes()) == before


@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('boundary', ['value', 'source', 'presence', 'status', 'seed_add', 'seed_remove'])
@pytest.mark.parametrize('exception', [KeyboardInterrupt, MemoryError])
def test_public_native_post_store_failure_restores_every_owner(batch, boundary, exception):
    mol = molecule()
    table = mol._atom_table
    field = 'implicit_hydrogens' if boundary in ('status', 'seed_remove') else 'formal_charge'
    value = 2 if field == 'implicit_hydrogens' else -2
    table._group_status[:, 1] = 1
    for seeds in table._dirty_seeds:
        seeds.clear()
    table._dirty_seeds[1].update(atom._uid for atom in mol.atoms)
    before = state(mol)
    with _native_fail_after(boundary, exception=exception), pytest.raises(exception):
        if batch:
            mol.updata_atoms(**{field: value})
        else:
            setattr(mol.atoms[0], field, value)
    assert state(mol) == before


@pytest.mark.parametrize('exception', [KeyboardInterrupt, MemoryError])
@pytest.mark.parametrize('boundary', ['enrollment', 'value', 'source', 'presence', 'seed_add'])
def test_caught_native_failure_poisons_same_owner_outer_transaction(exception, boundary):
    mol = molecule()
    for seeds in mol._atom_table._dirty_seeds:
        seeds.clear()
    before = state(mol)
    with pytest.raises(RuntimeError, match='nested factual'):
        with _FactTransaction(mol, producer=1):
            mol.atoms[1].x = 8
            try:
                with _native_fail_after(boundary, exception=exception):
                    mol.atoms[0].formal_charge = 4
            except exception:
                pass
    assert state(mol) == before


def test_native_and_python_dependency_mutations_share_reverse_journal_order():
    mol = molecule()
    table = mol._atom_table
    table._group_status[:, 3] = 4
    for seeds in table._dirty_seeds:
        seeds.clear()
    before = state(mol)
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1):
            mol.updata_atoms(formal_charge=2)
            _seed(mol, 3, tuple(atom._uid for atom in mol.atoms))
            _ensure_group(mol, 'FORMAL_CHARGE_COVERAGE')
            mol.atoms[0].formal_charge = 1
            raise KeyboardInterrupt('after native/Python/native status and seed changes')
    assert state(mol) == before


def test_conversion_callback_detachment_rejects_scalar_before_store():
    mol = molecule()
    atom = mol.atoms[0]

    class RemoveDuringFloat(float):
        def __float__(self):
            mol.remove_atom(atom)
            return 7.0

    with pytest.raises(ValueError, match='member|UID'):
        atom.x = RemoveDuringFloat(7)
    assert atom not in mol.atoms
    assert atom.x == 0


def test_conversion_callback_growth_rebinds_current_arrays():
    mol = molecule()
    atom = mol.atoms[0]
    original = mol._atom_table._blocks[0]

    class GrowDuringArray:
        def __array__(self, dtype=None, copy=None):
            for _ in range(8):
                mol.create_atom(atomic_number=6)
            return np.asarray(4.25, dtype=dtype)

    atom.formal_charge = GrowDuringArray()
    assert mol._atom_table._blocks[0] is not original
    assert mol._atom_table.raw(0, 'formal_charge') == 4.25


def test_batch_callback_changes_selection_and_rolls_back_whole_operation():
    mol = molecule()
    first = mol.atoms[0]
    before = state(mol), tuple(mol.atoms)

    class RemoveDuringArray:
        def __array__(self, dtype=None, copy=None):
            mol.remove_atom(first)
            return np.array([1., 2., 3.], dtype=dtype)

    with pytest.raises(ValueError, match='member|UID'):
        mol.updata_atoms(formal_charge=RemoveDuringArray())
    assert state(mol) == before[0]
    assert tuple(mol.atoms) == before[1]


def test_prepared_update_rechecks_captured_content_and_uid_owner():
    mol = molecule()
    atom = mol.atoms[0]
    update = _normalize_atom_facts(_capture_source(mol, ('geometry',)), (0,), (atom._uid,), {'x': 2})
    atom.y = 8
    with pytest.raises(RuntimeError, match='geometry'):
        _commit_atom_facts(mol, update)
    before = state(mol)
    object.__setattr__(atom, '_uid_owner', object())
    with pytest.raises(ValueError, match='UID'):
        _commit_atom_facts(mol, replace(update, source=_capture_source(mol)))
    assert state(mol) == before


def test_raw_float64_bits_survive_native_scalar_and_bulk():
    mol = molecule()
    bits = np.array([0x8000000000000000, 0x7ff8000000000042, 0xfff8000000000021], dtype=np.uint64)
    values = bits.view(np.float64)
    mol.updata_atoms(formal_charge=values)
    np.testing.assert_array_equal(mol._atom_table.column('formal_charge').view(np.uint64), bits)
    for atom, value in zip(mol.atoms, values):
        atom.formal_charge = value
    np.testing.assert_array_equal(mol._atom_table.column('formal_charge').view(np.uint64), bits)


def test_compiled_assignment_descriptor_binds_self():
    mol = molecule()
    atom = mol.atoms[0]
    type(atom).__setattr__(atom, 'formal_charge', 1.25)
    assert mol._atom_table.raw(0, 'formal_charge') == 1.25


def test_extension_import_occurs_only_at_first_fact_operation():
    command = '''import json, sys
import hotpot
import hotpot.cheminfo.structure
name = 'hotpot.cheminfo.structure._fact_native'
before = name in sys.modules
mol = hotpot.Molecule()
mol.create_atom(atomic_number=6)
print(json.dumps([before, name in sys.modules]))
'''
    result = subprocess.run([sys.executable, '-c', command], check=True, capture_output=True, text=True, env=os.environ)
    assert json.loads(result.stdout.strip()) == [False, True]


def test_missing_extension_fails_before_the_first_factual_transaction():
    command = '''import json, sys
import hotpot
class BlockFactNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'hotpot.cheminfo.structure._fact_native':
            raise ImportError('deliberately unavailable native artifact')
sys.meta_path.insert(0, BlockFactNative())
mol = hotpot.Molecule()
try:
    mol.create_atom(atomic_number=6)
except ImportError as error:
    print(json.dumps([len(mol.atoms), '_fact_native' in str(error), 'build/install' in str(error)]))
else:
    raise AssertionError('missing native writer was bypassed')
'''
    result = subprocess.run([sys.executable, '-c', command], check=True, capture_output=True, text=True, env=os.environ)
    assert json.loads(result.stdout.strip()) == [0, True, True]


# This helper wraps the real CPython MEM allocator; Python exception injection
# alone cannot establish that physical seed restoration allocates no buckets.
_ALLOCATOR_HELPER_SOURCE = r'''
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyMemAllocatorEx previous;
static int enabled;
static size_t rejected_size;
static size_t attempts;
static size_t rejected;
static size_t history[64];
static size_t history_count;

static int deny(size_t size) {
    if (!enabled) return 0;
    ++attempts;
    if (history_count < 64) history[history_count++] = size;
    if (size != rejected_size) return 0;
    ++rejected;
    return 1;
}

static void *probe_malloc(void *ctx, size_t size) {
    if (deny(size)) return NULL;
    return previous.malloc(previous.ctx, size);
}

static void *probe_calloc(void *ctx, size_t count, size_t size) {
    if (deny(count * size)) return NULL;
    return previous.calloc(previous.ctx, count, size);
}

static void *probe_realloc(void *ctx, void *ptr, size_t size) {
    if (deny(size)) return NULL;
    return previous.realloc(previous.ctx, ptr, size);
}

static void probe_free(void *ctx, void *ptr) {
    previous.free(previous.ctx, ptr);
}

static PyObject *arm(PyObject *self, PyObject *arg) {
    rejected_size = PyLong_AsSize_t(arg);
    if (PyErr_Occurred()) return NULL;
    attempts = rejected = history_count = 0;
    enabled = 1;
    Py_RETURN_NONE;
}

static PyObject *disarm(PyObject *self, PyObject *unused) {
    enabled = 0;
    PyObject *sizes = PyList_New(history_count);
    if (!sizes) return NULL;
    for (size_t i = 0; i < history_count; ++i)
        PyList_SET_ITEM(sizes, i, PyLong_FromSize_t(history[i]));
    return Py_BuildValue("{s:n,s:n,s:n,s:N}",
                         "rejected_size", (Py_ssize_t)rejected_size,
                         "attempts", (Py_ssize_t)attempts,
                         "rejected", (Py_ssize_t)rejected,
                         "allocation_sizes", sizes);
}

static PyObject *set_state(PyObject *self, PyObject *arg) {
    PySetObject *set = (PySetObject *)arg;
    return Py_BuildValue("{s:n,s:n,s:n,s:K,s:n,s:n,s:N}",
                         "fill", set->fill,
                         "used", set->used,
                         "mask", set->mask,
                         "table_address", (unsigned long long)(uintptr_t)set->table,
                         "hash", set->hash, "finger", set->finger,
                         "buckets", PyBytes_FromStringAndSize((const char *)set->table,
                                                  (set->mask + 1) * sizeof(setentry)));
}

static PyObject *probe_allocation(PyObject *self, PyObject *arg) {
    size_t size = PyLong_AsSize_t(arg);
    if (PyErr_Occurred()) return NULL;
    void *value = PyMem_Malloc(size);
    if (value) { PyMem_Free(value); Py_RETURN_FALSE; }
    Py_RETURN_TRUE;
}

static PyMethodDef methods[] = {
    {"probe_allocation", probe_allocation, METH_O, NULL},
    {"arm", arm, METH_O, NULL},
    {"disarm", disarm, METH_NOARGS, NULL},
    {"set_state", set_state, METH_O, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "seed_alloc_probe", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_seed_alloc_probe(void) {
    PyObject *result = PyModule_Create(&module);
    if (!result) return NULL;
    PyMem_GetAllocator(PYMEM_DOMAIN_MEM, &previous);
    PyMemAllocatorEx wrapper = {
        NULL, probe_malloc, probe_calloc, probe_realloc, probe_free
    };
    PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &wrapper);
    return result;
}
'''


@pytest.fixture(scope='module')
def seed_allocator_helper(tmp_path_factory):
    directory = tmp_path_factory.mktemp('seed-allocator')
    source = directory / 'seed_alloc_probe.c'
    binary = directory / ('seed_alloc_probe' + sysconfig.get_config_var('EXT_SUFFIX'))
    source.write_text(_ALLOCATOR_HELPER_SOURCE)
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


@pytest.mark.parametrize('scenario,rejected_size', [('clear', 512), ('tombstone', 1024)])
@pytest.mark.parametrize('exception_name', ['KeyboardInterrupt', 'MemoryError'])
@pytest.mark.parametrize('reject', [False, True])
def test_real_allocator_failure_cannot_replace_original_exception_or_break_seed_undo(
        seed_allocator_helper, scenario, rejected_size, exception_name, reject):
    command = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import seed_alloc_probe as alloc
from hotpot import Molecule
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._dependencies import _dependency_state, _ensure_group, _seed
scenario, exception_name, size = sys.argv[2:]
sentinel = {'KeyboardInterrupt': KeyboardInterrupt, 'MemoryError': MemoryError}[exception_name]('original')
mol = Molecule()
atoms = tuple(mol.create_atom(atomic_number=6) for _ in range(128))
mol.updata_atoms(formal_charge=0)
uids = tuple(atom._uid for atom in atoms)
_seed(mol, 3, uids[:64 if scenario == 'clear' else 76])
seeds = mol._atom_table._dirty_seeds[3]
def state():
    table = mol._atom_table
    status, seeds = _dependency_state(table)
    return (table.matrix().tobytes(), table.presence().tobytes(),
            table.sources().tobytes(), status.tobytes(), seeds)
before, physical = state(), alloc.set_state(seeds)
caught = None
try:
    with _FactTransaction(mol, producer=1):
        if scenario == 'clear':
            _ensure_group(mol, 'FORMAL_CHARGE_COVERAGE')
        else:
            for atom in atoms[:75]:
                atom.formal_charge = 2
            _seed(mol, 3, (uids[76],))
        alloc.arm(int(size))
        raise sentinel
except BaseException as error:
    caught = error
finally:
    allocation = alloc.disarm()
result = {'same_exception': caught is sentinel, 'state_restored': state() == before,
          'physical_restored': alloc.set_state(seeds) == physical,
          'same_set': mol._atom_table._dirty_seeds[3] is seeds, 'allocation': allocation}
control_size = int(size) or 512
alloc.arm(control_size)
control_rejected = alloc.probe_allocation(control_size)
control = alloc.disarm()
result['allocator_control'] = control
print(json.dumps(result))
assert all(result[key] for key in ('same_exception', 'state_restored', 'physical_restored', 'same_set'))
assert control_rejected and control['rejected'] == 1
assert allocation['attempts'] == 0
'''
    size = rejected_size if reject else 0
    argv = [sys.executable, '-c', command, str(seed_allocator_helper), scenario, exception_name, str(size)]
    result = subprocess.run(argv, capture_output=True, text=True, env=os.environ)
    name = f'{scenario}-{exception_name}-{size}'
    record = {'argv': argv, 'returncode': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr}
    (seed_allocator_helper / (name + '.json')).write_text(json.dumps(record, indent=2))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('boundary', ['enrollment', 'status', 'seed_clear'])
@pytest.mark.parametrize('exception', [KeyboardInterrupt, MemoryError])
def test_clear_failure_cancels_only_installed_body_swaps(boundary, exception):
    mol = molecule()
    mol.updata_atoms(formal_charge=0)
    mol._atom_table._group_status[:, 3] = 0
    engine = _fact_engine()
    before = state(mol)
    owners = mol._atom_table._dirty_seeds
    with pytest.raises(RuntimeError, match='nested factual'):
        with _FactTransaction(mol, producer=1):
            _seed(mol, 3, tuple(range(200, 280)))
            try:
                with _native_fail_after(boundary, exception=exception):
                    _ensure_group(mol, 'FORMAL_CHARGE_COVERAGE')
            except exception:
                pass
            # A locally cancelled capsule remains earlier in the journal.
            engine._update_dependency_seeds(mol, (3,), (900,), True)
    assert state(mol) == before
    assert all(a is b for a, b in zip(owners, mol._atom_table._dirty_seeds))


@pytest.mark.parametrize('collisions', [False, True])
def test_seed_collision_batches_and_multiple_storage_generations_restore(collisions):
    mol = molecule()
    engine = _fact_engine()
    owners = mol._atom_table._dirty_seeds
    # Equal hashes exercise the closed integer comparison and overlay placement.
    stride = sys.hash_info.modulus if collisions else 1
    uids = tuple(int(10**10 + index * stride) for index in range(180))
    engine._update_dependency_seeds(mol, (3,), uids[:4], True)
    before = state(mol)
    references = tuple(sys.getrefcount(uid) for uid in uids)
    sentinel = KeyboardInterrupt('multi-generation seed undo')
    with pytest.raises(KeyboardInterrupt) as caught:
        with _FactTransaction(mol, producer=1):
            for offset in range(3):
                engine._update_dependency_seeds(mol, (3,), uids, True)
                assert set(uids) <= owners[3]
                engine._update_dependency_seeds(mol, (3,), uids[1::2], False)
                engine._update_dependency_seeds(mol, (3,), uids[1::2], True)
                engine._clear_dependency_seeds(mol, 3, ())
                engine._update_dependency_seeds(mol, (3,), uids[offset:offset + 3], True)
                gc.collect()
            raise sentinel
    assert caught.value is sentinel
    assert state(mol) == before
    assert tuple(sys.getrefcount(uid) for uid in uids) == references
    assert all(a is b for a, b in zip(owners, mol._atom_table._dirty_seeds))
    probes = tuple(weakref.ref(owner) for owner in owners)
    del owners, mol
    gc.collect()
    assert all(probe() is None for probe in probes)


def test_seed_undo_preserves_old_owners_across_growth_removal_and_remap():
    mol = molecule()
    engine = _fact_engine()
    atoms = tuple(mol.atoms)
    table = mol._atom_table
    owners = table._dirty_seeds
    before = state(mol)
    with pytest.raises(MemoryError):
        with _FactTransaction(mol, producer=1):
            _ensure_group(mol, 'AROMATIC_REPRESENTATION')
            for _ in range(30):
                mol.create_atom(atomic_number=6)
            engine._update_dependency_seeds(mol, (3,), tuple(atom._uid for atom in mol.atoms), True)
            mol.remove_atom(atoms[1])
            _ensure_group(mol, 'FORMAL_CHARGE_COVERAGE')
            mol.updata_atoms(formal_charge=2)
            raise MemoryError('after table growth, seed rehash and row removal')
    assert mol._atom_table is table
    assert tuple(mol.atoms) == atoms
    assert state(mol) == before
    assert all(a is b for a, b in zip(owners, mol._atom_table._dirty_seeds))


@pytest.mark.parametrize('outer', [False, True])
def test_real_rehash_preparation_allocation_failure_precedes_all_stores(seed_allocator_helper, outer):
    command = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import seed_alloc_probe as alloc
from hotpot import Molecule
from hotpot.cheminfo.structure._commit import _FactTransaction, _fact_engine
from hotpot.cheminfo.structure._dependencies import _dependency_state
mol = Molecule()
for _ in range(8): mol.create_atom(atomic_number=6)
mol.updata_atoms(formal_charge=0)
engine = _fact_engine()
uids = tuple(atom._uid for atom in mol.atoms[:5])
seeds = mol._atom_table._dirty_seeds[3]
projection = mol.atom_matrix.tobytes()
engine._clear_dependency_seeds(mol, 3, ())
assert not seeds
before = alloc.set_state(seeds), _dependency_state(mol._atom_table), projection
def failing_update():
    caught = None
    alloc.arm(512)
    try:
        engine._update_dependency_seeds(mol, (3,), uids, True)
    except BaseException as error:
        caught = error
    finally:
        record = alloc.disarm()
    assert type(caught) is MemoryError
    assert record['rejected'] >= 1
    return record
poisoned = False
if sys.argv[2] == '1':
    try:
        with _FactTransaction(mol, producer=1):
            record = failing_update()
    except RuntimeError as error:
        poisoned = 'nested factual' in str(error)
    assert poisoned
else:
    record = failing_update()
status, pending = _dependency_state(mol._atom_table)
assert alloc.set_state(seeds) == before[0]
assert status.tobytes() == before[1][0].tobytes() and pending == before[1][1]
assert mol.atom_matrix.tobytes() == before[2]
print(json.dumps({'allocation': record, 'outer_poisoned': poisoned, 'unchanged': True}))
'''
    argv = [sys.executable, '-c', command, str(seed_allocator_helper), str(int(outer))]
    result = subprocess.run(argv, capture_output=True, text=True, env=os.environ)
    record = {'argv': argv, 'returncode': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr}
    (seed_allocator_helper / f'prepare-outer-{outer}.json').write_text(json.dumps(record, indent=2))
    assert result.returncode == 0, result.stdout + result.stderr


def test_partial_rehash_installation_restores_only_applied_generations(seed_allocator_helper):
    command = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import seed_alloc_probe as alloc
from hotpot import Molecule
from hotpot.cheminfo.structure._commit import _FactTransaction, _fact_engine, _native_fail_after
from hotpot.cheminfo.structure._dependencies import _dependency_state
engine = _fact_engine()
results = []
for outer in (False, True):
    boundaries = [('value', 1), ('seed_add', 1), ('seed_add', 5), ('seed_add', 6),
                  ('seed_add', 13), ('seed_add', 16), ('seed_add', 21)]
    if outer: boundaries.append(('enrollment', 1))
    for exception in (KeyboardInterrupt, MemoryError):
        for boundary, occurrence in boundaries:
            mol = Molecule()
            for _ in range(8): mol.create_atom(atomic_number=6)
            mol.updata_atoms(formal_charge=0)
            for group in range(4): engine._clear_dependency_seeds(mol, group, ())
            owners = mol._atom_table._dirty_seeds
            physical = tuple(alloc.set_state(seeds) for seeds in owners)
            def state():
                table = mol._atom_table
                status, seeds = _dependency_state(table)
                return (table.matrix().tobytes(), table.presence().tobytes(),
                        table.sources().tobytes(), status.tobytes(), seeds)
            before = state()
            def failing_update():
                caught = None
                try:
                    with _native_fail_after(boundary, occurrence, exception):
                        mol.updata_atoms(formal_charge=2)
                except BaseException as error:
                    caught = error
                assert type(caught) is exception, (outer, exception, boundary, occurrence, caught)
                assert state() == before
                assert tuple(alloc.set_state(seeds) for seeds in owners) == physical
            if outer:
                poisoned = False
                try:
                    with _FactTransaction(mol, producer=1):
                        failing_update()
                        # The cancelled entry must remain harmless after a later
                        # successful rehash creates yet another storage generation.
                        engine._update_dependency_seeds(mol, (0, 1, 2), tuple(range(1000, 1080)), True)
                except RuntimeError as error:
                    poisoned = 'nested factual' in str(error)
                assert poisoned
            else:
                failing_update()
            assert state() == before
            assert tuple(alloc.set_state(seeds) for seeds in owners) == physical
            assert all(a is b for a, b in zip(owners, mol._atom_table._dirty_seeds))
            results.append([outer, exception.__name__, boundary, occurrence])
print(json.dumps({'passed': results}))
'''
    argv = [sys.executable, '-c', command, str(seed_allocator_helper)]
    result = subprocess.run(argv, capture_output=True, text=True, env=os.environ)
    record = {'argv': argv, 'returncode': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr}
    (seed_allocator_helper / 'partial-rehash.json').write_text(json.dumps(record, indent=2))
    assert result.returncode == 0, result.stdout + result.stderr
