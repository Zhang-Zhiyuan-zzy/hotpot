"""Finite assembly, configuration and owned-worker lifecycle contracts."""

from collections import Counter, defaultdict
import hashlib
from importlib.resources import files
import json
import multiprocessing as mp
import os
from queue import Empty
import time

import pytest

from hotpot.cheminfo.operations.assembly import (
    AlkylGraft, AssembleFactory, AtomLink, AtomReplace, BondAdding,
    EdgeShoulder, RingWedge, alkyl_generator,
)
from hotpot.cheminfo.operations.assembly import assembler as factory_module
from hotpot.cheminfo.operations.assembly.assembler import MolBatch
from hotpot.cheminfo.operations.io import read_mol


class _CarbonGrowth:
    def graft(self, frame):
        length = len(frame.atoms)
        return {'C' * (length + 1): None} if length < 4 else {}


class _UnknownFailure:
    def graft(self, frame):
        try:
            try:
                raise LookupError('deep original cause')
            except LookupError as original:
                raise KeyError('original site failure') from original
        except KeyError as cause:
            error = ValueError('unknown action failure')
            error.__notes__ = ['original action note']
            raise error from cause


class _UnpicklableFailure:
    def graft(self, frame):
        error = ValueError('unpicklable action failure')
        error.unpicklable = lambda: None
        raise error


class _SelectivePause:
    def graft(self, frame):
        if frame.atoms[0].atomic_number == 7:
            time.sleep(10)
        return {'CC': None}


class _AbruptExit:
    def graft(self, frame):
        os._exit(7)


class _SilentCompletion(AssembleFactory):
    @staticmethod
    def _make_batch(list_smi, list_assembler, q):
        return


class _InterruptProgress:
    def __init__(self, **kwargs):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def set_description(self, value):
        pass

    def update(self):
        raise KeyboardInterrupt('parent interrupted after a completed batch')


def _wait_for_release(event):
    event.wait(30)


@pytest.fixture
def owned_workers(monkeypatch):
    """An unrelated child must survive every factory lifecycle being checked."""
    release = mp.Event()
    unrelated = mp.get_context().Process(target=_wait_for_release, args=(release,))
    unrelated.start()
    processes, queues = [], []
    make_process, make_queue = factory_module.mp.Process, factory_module.mp.Queue

    def track_process(*args, **kwargs):
        process = make_process(*args, **kwargs)
        processes.append(process)
        return process

    def track_queue(*args, **kwargs):
        queue = make_queue(*args, **kwargs)
        queues.append(queue)
        return queue

    monkeypatch.setattr(factory_module.mp, 'Process', track_process)
    monkeypatch.setattr(factory_module.mp, 'Queue', track_queue)
    try:
        yield processes, queues
        assert unrelated.is_alive(), 'factory stopped an unrelated child'
        alive = mp.active_children()
        assert all(process not in alive for process in processes)
        assert all(queue._closed for queue in queues)
        assert all(queue._thread is None or not queue._thread.is_alive() for queue in queues)
    finally:
        release.set()
        unrelated.join(timeout=5)
        if unrelated.is_alive():
            unrelated.terminate()
            unrelated.join(timeout=5)
        unrelated.close()


@pytest.mark.parametrize(('items', 'size', 'expected'), [
    ([], 2, []),
    (['C', 'N'], 2, [['C', 'N']]),
    (['C', 'N', 'O', 'F'], 2, [['C', 'N'], ['O', 'F']]),
    (['C', 'N', 'O'], 2, [['C', 'N'], ['O']]),
])
def test_batches_partition_each_input_once_and_refresh(items, size, expected):
    batch = MolBatch(items, size)
    assert len(batch) == len(expected)
    assert list(batch) == expected
    assert list(batch) == []
    batch.refresh()
    if expected:
        assert next(batch) == expected[0]
        assert list(batch) == expected[1:]
    else:
        with pytest.raises(StopIteration):
            next(batch)


@pytest.mark.parametrize(('definition', 'expected_type', 'count'), [
    ({'method': 'EdgeShoulder', 'smiles': 'CC', 'points': [[0, 1]]}, EdgeShoulder, 1),
    ({'method': 'AtomLink', 'smiles': 'O', 'points': [[0]]}, AtomLink, 1),
    ({'method': 'AtomReplace', 'elements': ['N', 'O']}, AtomReplace, 2),
    ({'method': 'BondAdding'}, BondAdding, 1),
    ({'method': 'AlkylGraft', 'link_length': [1, 2]}, AlkylGraft, 2),
    ({'method': 'RingWedge', 'smiles': 'C1CCC1', 'points': [[0]]}, RingWedge, 1),
])
@pytest.mark.parametrize('excluded', [False, True])
def test_all_six_loader_methods_return_lists_with_exclusion(definition, expected_type, count, excluded, tmp_path):
    definition = dict(definition)
    if excluded:
        definition['ExStrct'] = 'C'
    path = tmp_path / 'assemblers.json'
    path.write_text(json.dumps([definition]))
    assemblers = AssembleFactory.load_assembler_file(path)
    assert isinstance(assemblers, list)
    assert len(assemblers) == count
    assert all(type(assembler) is expected_type for assembler in assemblers)
    assert all((assembler.exclude_searcher is not None) == excluded for assembler in assemblers)
    if excluded:
        assert all(assembler.graft(read_mol('C')) == {} for assembler in assemblers)


def test_default_resource_preserves_exact_26_row_asset_and_loads_it():
    resource = files('hotpot.cheminfo.operations.assembly').joinpath('data/fragment_template.json')
    content = resource.read_bytes()
    assert hashlib.sha256(content).hexdigest() == '0ac6a05cd76c06d3a918bccf7c59acd053315c1b8629754153ad4b7185f08d52'
    definitions = json.loads(content)
    assert len(definitions) == 26
    assert Counter(row['method'] for row in definitions) == {
        'EdgeShoulder': 8, 'AtomLink': 15, 'AtomReplace': 1,
        'BondAdding': 1, 'AlkylGraft': 1,
    }
    assemblers = AssembleFactory.load_default_assembler()
    assert {type(assembler) for assembler in assemblers} == {
        EdgeShoulder, AtomLink, AtomReplace, BondAdding, AlkylGraft,
    }


@pytest.mark.parametrize('parallel', [False, True])
def test_finite_candidate_expansion_and_periodic_save_cadence(parallel, tmp_path, monkeypatch, owned_workers):
    monkeypatch.chdir(tmp_path)
    factory = AssembleFactory(
        [_CarbonGrowth()], iter_step=3, max_running=None,
        save_per_step=2, catch_path='results.smi',
    )
    snapshots = []
    save = factory._save_results

    def observe_save(results):
        snapshots.append(set(results))
        save(results)

    monkeypatch.setattr(factory, '_save_results', observe_save)
    if parallel:
        results = factory.mp_make(['C'], nproc=2, batch_size=1)
    else:
        results = factory.make([read_mol('C')])
    assert results == {'C', 'CC', 'CCC', 'CCCC'}
    assert [len(snapshot) for snapshot in snapshots] == [2, 3, 3, 4]
    assert set((tmp_path / 'results.smi').read_text().splitlines()) == results


@pytest.mark.parametrize('parallel', [False, True])
def test_real_one_step_methanol_candidates(parallel, owned_workers):
    factory = AssembleFactory([AtomLink(read_mol('O'), (0,))], iter_step=1)
    frame = read_mol('C')
    results = factory.mp_make([frame], nproc=1) if parallel else factory.make([frame])
    assert results == {'C', 'CO'}
    assert frame.smiles == 'C'
    assert frame.atoms[0].implicit_hydrogens == 4


def test_empty_input_and_disabled_timeout(owned_workers):
    factory = AssembleFactory([_CarbonGrowth()], iter_step=1)
    assert factory.mp_make([], nproc=1, timeout=0) == set()
    assert factory.mp_make(['C'], nproc=1, timeout=0) == {'C', 'CC'}


def test_fast_one_item_completion_is_drained_after_confirmed_exit(monkeypatch, owned_workers):
    processes, _ = owned_workers
    make_queue = factory_module.mp.Queue

    def queue_with_exit_race(*args, **kwargs):
        queue = make_queue(*args, **kwargs)
        receive = queue.get_nowait
        first_poll = True

        def get_nowait():
            nonlocal first_poll
            if first_poll:
                first_poll = False
                # The initial poll was empty, then the worker flushed and exited
                # before the parent inspected its exit status.
                process = processes[-1]
                process.join(timeout=5)
                assert process.exitcode == 0
                raise Empty
            return receive()

        queue.get_nowait = get_nowait
        return queue

    monkeypatch.setattr(factory_module.mp, 'Queue', queue_with_exit_race)
    factory = AssembleFactory([_CarbonGrowth()], iter_step=1)
    assert factory.mp_make(['C'], nproc=1, batch_size=1) == {'C', 'CC'}


def test_unknown_worker_error_keeps_type_direct_cause_and_full_remote_traceback(tmp_path, owned_workers):
    output = tmp_path / 'partial.smi'
    factory = AssembleFactory([_UnknownFailure()], iter_step=1, catch_path=output)
    with pytest.raises(ValueError, match='unknown action failure') as caught:
        factory.mp_make(['C'], nproc=1, timeout=2)
    assert isinstance(caught.value.__cause__, KeyError)
    assert caught.value.__notes__[0] == 'original action note'
    remote = '\n'.join(caught.value.__notes__)
    assert 'LookupError: deep original cause' in remote
    assert 'original site failure' in remote
    assert 'direct cause' in remote
    assert 'in graft' in remote
    assert 'epoch=0, batch=1' in remote and "inputs=['C']" in remote
    assert output.read_text().splitlines() == ['C']



@pytest.mark.parametrize('parallel', [False, True])
def test_failed_result_save_keeps_original_error_and_existing_notes(monkeypatch, owned_workers, parallel):
    factory = AssembleFactory([_UnknownFailure()], iter_step=1)

    def fail_save(results):
        raise OSError('result store unavailable')

    monkeypatch.setattr(factory, '_save_results', fail_save)
    with pytest.raises(ValueError, match='unknown action failure') as caught:
        if parallel:
            factory.mp_make(['C'], nproc=1, timeout=2)
        else:
            factory.make([read_mol('C')])
    assert isinstance(caught.value.__cause__, KeyError)
    notes = caught.value.__notes__
    assert notes[0] == 'original action note'
    assert notes[-1] == 'Could not persist completed assembly results: result store unavailable'
    if parallel:
        assert 'original site failure' in notes[1]
        assert 'Assembly worker epoch=0, batch=1' in notes[1]


def test_unpicklable_error_fails_explicitly_with_original_traceback(owned_workers):
    factory = AssembleFactory([_UnpicklableFailure()], iter_step=1)
    with pytest.raises(RuntimeError, match='assembly.error_transport') as caught:
        factory.mp_make(['C'], nproc=1, timeout=2)
    assert 'unpicklable action failure' in str(caught.value)
    assert 'ValueError' in str(caught.value)
    assert 'graft' in str(caught.value)


def test_timeout_identifies_batch_and_persists_completed_results(tmp_path, owned_workers):
    output = tmp_path / 'partial.smi'
    factory = AssembleFactory([_SelectivePause()], iter_step=1, catch_path=output)
    with pytest.raises(TimeoutError, match='assembly.batch_timeout') as caught:
        factory.mp_make(['C', 'N'], nproc=2, batch_size=1, timeout=1)
    diagnostic = str(caught.value)
    assert 'epoch=0, batch=' in diagnostic and "inputs=['N']" in diagnostic
    assert 'timeout=1s' in diagnostic
    assert 'completed_batches=1' in diagnostic
    assert 'completed_input_assembler_pairs=1' in diagnostic
    assert set(output.read_text().splitlines()) == {'C', 'N', 'CC'}


@pytest.mark.parametrize(('factory_type', 'assembler', 'exitcode'), [
    (AssembleFactory, _AbruptExit(), 7),
    (_SilentCompletion, _CarbonGrowth(), 0),
])
def test_abrupt_exit_or_missing_completion_fails_without_waiting_for_timeout(factory_type, assembler, exitcode, owned_workers):
    factory = factory_type([assembler], iter_step=1)
    with pytest.raises(RuntimeError, match='assembly.worker_exit') as caught:
        factory.mp_make(['C'], nproc=1, timeout=0)
    assert f'exitcode={exitcode}' in str(caught.value)
    assert 'epoch=0, batch=1' in str(caught.value)


def test_strict_cap_stops_and_cleans_only_owned_workers(owned_workers):
    factory = AssembleFactory([_SelectivePause()], iter_step=3, max_running=2)
    assert factory.mp_make(['C', 'N'], nproc=2, batch_size=1) == {'C', 'N', 'CC'}


def test_parent_interrupt_cleans_owned_sibling_and_progress(monkeypatch, owned_workers):
    progress = []

    def make_progress(**kwargs):
        bar = _InterruptProgress(**kwargs)
        progress.append(bar)
        return bar

    monkeypatch.setattr(factory_module, 'tqdm', make_progress)
    factory = AssembleFactory([_SelectivePause()], iter_step=1)
    with pytest.raises(KeyboardInterrupt, match='parent interrupted'):
        factory.mp_make(['C', 'N'], nproc=2, batch_size=1)
    assert all(bar.closed for bar in progress)


def test_start_failure_closes_owned_channel_and_preserves_original_error(monkeypatch, owned_workers):
    make_process = factory_module.mp.Process

    def fail_start(*args, **kwargs):
        process = make_process(*args, **kwargs)

        def fail():
            raise OSError('original process start failure')

        process.start = fail
        return process

    monkeypatch.setattr(factory_module.mp, 'Process', fail_start)
    factory = AssembleFactory([_CarbonGrowth()], iter_step=1)
    with pytest.raises(OSError, match='original process start failure'):
        factory.mp_make(['C'], nproc=1)


def test_rooted_alkyl_counts_capped_formulas_and_attachment_isomers():
    collection = alkyl_generator([1, 2, 3, 4])
    assert isinstance(collection, defaultdict)
    assert {length: len(fragments) for length, fragments in collection.items()} == {1: 1, 2: 1, 3: 2, 4: 4}
    for length, fragments in collection.items():
        for fragment in fragments:
            mol = fragment.frag
            assert fragment.action_points == [0]
            assert len(mol.atoms) == length and len(mol.bonds) == length - 1
            assert all(atom.atomic_number == 6 for atom in mol.atoms)
            assert all(len(atom.neighbours) <= 4 for atom in mol.atoms)
            assert all(atom.formal_charge == 0 for atom in mol.atoms)
            assert all(bond.bond_order == 1 for bond in mol.bonds)
            assert sum(atom.implicit_hydrogens for atom in mol.atoms) == 2 * length + 2
    # Root degree distinguishes primary/secondary propyl; branch distances also
    # distinguish the two primary butyl attachment isomers independently of keys.
    assert sorted(len(fragment.frag.atoms[0].neighbours) for fragment in collection[3]) == [1, 2]
    signatures = set()
    for fragment in collection[4]:
        mol = fragment.frag
        root = mol.atoms[0]
        signatures.add((
            len(root.neighbours),
            tuple(sorted(len(atom.neighbours) for atom in root.neighbours)),
        ))
    assert signatures == {(1, (2,)), (1, (3,)), (2, (1, 2)), (3, (1, 1, 1))}
