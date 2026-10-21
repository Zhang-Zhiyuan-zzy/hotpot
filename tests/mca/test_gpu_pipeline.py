import queue
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from mca.gpu_backend import MCAGpuPredictor
from mca.gpu_consumer import (
    GpuBatchRequest,
    GpuBatchResult,
    GpuConsumerConfig,
    GpuConsumerFailure,
    GpuConsumerReady,
    gpu_consumer_main,
)
from mca.gpu_memory import MCAGpuOutOfMemoryError
from tests.mca._helpers import plan_logical_batches


def _coordinator():
    predictor = object.__new__(MCAGpuPredictor)
    predictor._next_request_id = 0
    predictor._next_gpu_position = 0
    predictor._request_queues = [queue.Queue()]
    predictor._result_queue = queue.Queue()
    predictor._gpu_processes = []
    predictor.gpu_indices = (0,)
    predictor.gpu_memory_totals = (80 * 1024**3,)
    predictor.gpu_batch_size = 4
    predictor.variant = "fp16"
    return predictor


def test_split_molecule_encoding_is_computed_once_and_scattered_to_both_chunks():
    predictor = _coordinator()
    batches = plan_logical_batches([8], 4)
    state = SimpleNamespace(
        values=np.empty(8, dtype=np.float64),
        filled=np.zeros(8, dtype=bool),
    )
    states = {0: state}
    features = {0: {"feature": np.asarray([1])}}
    cache = {}
    pending_keys = set()
    chunks = {}
    waiting = {}
    requests = {}

    for batch in batches:
        predictor._submit_batch(
            batch,
            states,
            features,
            cache,
            pending_keys,
            chunks,
            waiting,
            requests,
        )

    request = predictor._request_queues[0].get_nowait()
    assert request.molecule_indices == (0,)
    assert predictor._request_queues[0].empty()
    predictor._result_queue.put(
        GpuBatchResult(
            request_id=request.request_id,
            batch_ordinal=request.batch_ordinal,
            gpu_index=0,
            molecule_indices=(0,),
            pad_width=8,
            predictions=(np.arange(8, dtype=np.float64),),
        )
    )
    predictor._drain_gpu_result(
        states,
        cache,
        pending_keys,
        chunks,
        waiting,
        requests,
    )

    assert state.filled.tolist() == [True] * 8
    assert state.values.tolist() == list(range(8))
    assert not chunks
    assert not requests


def test_conveyor_submits_partial_features_without_waiting_for_sites(monkeypatch):
    from mca import gpu_backend as module

    molecules = {
        "first": SimpleNamespace(
            atom_count=2,
            atom_symbols=("C", "N"),
            smiles="CN",
            formal_charge=0,
        ),
        "second": SimpleNamespace(
            atom_count=4,
            atom_symbols=("C", "C", "C", "N"),
            smiles="CCCN",
            formal_charge=0,
        ),
    }
    monkeypatch.setattr(module, "normalize_molecule", molecules.__getitem__)
    monkeypatch.setattr(module, "validate_dto", lambda *args: None)

    class Process:
        pid = 123
        exitcode = None

    class DeferredSitePool:
        _pool = [Process()]

        def __init__(self):
            self.callbacks = []
            self.completed = False

        def apply_async(self, function, args, callback, error_callback):
            self.callbacks.append((args[0], callback))

        def release(self):
            assert not self.completed
            self.completed = True
            for task, callback in self.callbacks:
                callback(SimpleNamespace(molecule_index=task.molecule_index, sites=()))

    class DeferredFeaturePool:
        _pool = [Process()]

        def __init__(self):
            self.second = None

        def apply_async(self, function, args, callback, error_callback):
            task = args[0]
            result = SimpleNamespace(
                molecule_index=task.molecule_index,
                feature={"index": np.asarray([task.molecule_index])},
            )
            if task.molecule_index == 0:
                callback(result)
            else:
                self.second = (callback, result)

        def release_second(self):
            callback, result = self.second
            callback(result)
            self.second = None

    site_pool = DeferredSitePool()
    feature_pool = DeferredFeaturePool()
    submitted = []
    result_queue = queue.Queue()

    class ImmediateGpuQueue:
        def put(self, request):
            assert not site_pool.completed
            submitted.append(request)
            predictions = tuple(
                np.arange(molecules[("first", "second")[index]].atom_count)
                + 10 * index
                for index in request.molecule_indices
            )
            result_queue.put(
                GpuBatchResult(
                    request_id=request.request_id,
                    batch_ordinal=request.batch_ordinal,
                    gpu_index=0,
                    molecule_indices=request.molecule_indices,
                    pad_width=request.pad_width,
                    predictions=predictions,
                )
            )
            if len(submitted) == 1:
                feature_pool.release_second()
            else:
                site_pool.release()

    predictor = object.__new__(MCAGpuPredictor)
    predictor.gpu_batch_size = 2
    predictor.max_atoms = 512
    predictor.conformer_seed = 42
    predictor.allow_charged = False
    predictor.variant = "fp16"
    predictor.gpu_indices = (0,)
    predictor.resource_plan = SimpleNamespace(
        site_worker_assignments=(object(),),
        infer_worker_assignments=(object(),),
    )
    predictor._site_pool = site_pool
    predictor._feature_pool = feature_pool
    predictor._next_request_id = 0
    predictor._next_gpu_position = 0
    predictor._request_queues = [ImmediateGpuQueue()]
    predictor._result_queue = result_queue
    predictor._gpu_processes = []

    results = list(predictor._iter_predict(["first", "second"]))

    assert [request.batch_ordinal for request in submitted] == [0, 1]
    assert [request.molecule_indices for request in submitted] == [(0,), (1,)]
    assert [result.smiles for result in results] == ["CN", "CCCN"]
    assert [
        atom.mca_kj_mol for atom in results[1].atom_predictions
    ] == [10.0, 11.0, 12.0, 13.0]
    assert site_pool.completed


def test_local_cpu_worker_exception_type_is_preserved(monkeypatch):
    from mca import gpu_backend as module

    molecule = SimpleNamespace(
        atom_count=1,
        atom_symbols=("C",),
        smiles="C",
        formal_charge=0,
    )
    monkeypatch.setattr(module, "normalize_molecule", lambda value: molecule)
    monkeypatch.setattr(module, "validate_dto", lambda *args: None)

    class Process:
        pid = 123
        exitcode = None

    class SitePool:
        _pool = [Process()]

        @staticmethod
        def apply_async(function, args, callback, error_callback):
            callback(SimpleNamespace(molecule_index=0, sites=()))

    class FeaturePool:
        _pool = [Process()]

        @staticmethod
        def apply_async(function, args, callback, error_callback):
            error_callback(ValueError("feature failed"))

    predictor = object.__new__(MCAGpuPredictor)
    predictor.gpu_batch_size = 2
    predictor.max_atoms = 512
    predictor.conformer_seed = 42
    predictor.allow_charged = False
    predictor.variant = "fp16"
    predictor.gpu_indices = (0,)
    predictor.resource_plan = SimpleNamespace(
        site_worker_assignments=(object(),),
        infer_worker_assignments=(object(),),
    )
    predictor._site_pool = SitePool()
    predictor._feature_pool = FeaturePool()
    predictor._gpu_processes = []

    with pytest.raises(ValueError, match="feature failed"):
        list(predictor._iter_predict(["C"]))


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"gpu_index": 1}, "different GPU"),
        ({"batch_ordinal": 4}, "batch ordinal"),
        ({"predictions": ()}, "prediction count"),
    ],
)
def test_gpu_result_metadata_is_validated(changed, message):
    predictor = _coordinator()
    batch = plan_logical_batches([2], 2)[0]
    states = {
        0: SimpleNamespace(
            values=np.empty(2, dtype=np.float64),
            filled=np.zeros(2, dtype=bool),
        )
    }
    features = {0: {"feature": np.asarray([1])}}
    cache = {}
    pending_keys = set()
    chunks = {}
    waiting = {}
    requests = {}
    predictor._submit_batch(
        batch,
        states,
        features,
        cache,
        pending_keys,
        chunks,
        waiting,
        requests,
    )
    request = predictor._request_queues[0].get_nowait()
    values = {
        "request_id": request.request_id,
        "batch_ordinal": request.batch_ordinal,
        "gpu_index": 0,
        "molecule_indices": request.molecule_indices,
        "pad_width": request.pad_width,
        "predictions": (np.asarray([1.0, 2.0]),),
    }
    values.update(changed)
    predictor._result_queue.put(GpuBatchResult(**values))

    with pytest.raises(RuntimeError, match=message):
        predictor._drain_gpu_result(
            states,
            cache,
            pending_keys,
            chunks,
            waiting,
            requests,
        )


def test_remote_cuda_oom_reports_current_and_next_batch_with_cause():
    predictor = _coordinator()
    predictor.gpu_batch_size = 512
    failure = GpuConsumerFailure(
        gpu_index=0,
        request_id=3,
        batch_ordinal=2,
        exception_type="RuntimeException",
        message="CUDA out of memory while allocating tensor",
        traceback_text="remote traceback",
    )

    error = predictor._remote_failure(failure, "all-atom inference")

    assert isinstance(error, MCAGpuOutOfMemoryError)
    assert error.batch_size == 512
    assert error.next_batch_size == 256
    assert "gpu_batch_size=512" in str(error)
    assert "gpu_batch_size=256" in str(error)
    assert error.__cause__ is not None


def test_gpu_consumer_owns_one_runtime_and_returns_all_atom_rows(monkeypatch):
    from mca import gpu_consumer as module

    observed = []

    class Runtime:
        variant = "fp16"

        def predict(self, arrays, batch_size):
            observed.append((arrays, batch_size))
            return np.asarray([[10.0, 20.0]])

        def close(self):
            observed.append("closed")

    monkeypatch.setattr(module, "pin_current_process", lambda cpu_ids: None)
    monkeypatch.setattr(
        module,
        "create_cuda_runtime_from_artifact",
        lambda **kwargs: observed.append(kwargs) or Runtime(),
    )
    monkeypatch.setattr(
        module,
        "collate_molecule_rows",
        lambda features, indices, max_atoms: {
            "features": features,
            "indices": tuple(indices),
            "max_atoms": max_atoms,
        },
    )
    requests = queue.Queue()
    results = queue.Queue()
    requests.put(
        GpuBatchRequest(
            request_id=4,
            batch_ordinal=2,
            pad_width=2,
            molecule_indices=(7,),
            features=({"feature": np.asarray([1])},),
        )
    )
    requests.put(None)

    gpu_consumer_main(
        3,
        (8, 9),
        GpuConsumerConfig("/resolved/all-atom.onnx", "fp16", 512, 2048),
        requests,
        results,
    )

    ready = results.get_nowait()
    result = results.get_nowait()
    assert ready == GpuConsumerReady(3, "fp16", 512)
    assert result.gpu_index == 3
    assert result.molecule_indices == (7,)
    assert result.predictions[0].tolist() == [10.0, 20.0]
    assert observed[0]["device_id"] == 3
    assert observed[1][1] == 2048
    assert observed[-1] == "closed"


def test_two_gpu_consumers_receive_the_same_resolved_artifact():
    configs = []

    class Process:
        exitcode = None

        def __init__(self, *, target, args, name):
            self.args = args
            self.name = name

        def start(self):
            gpu_index, _cpu_ids, config, _request_queue, result_queue = self.args
            configs.append(config)
            result_queue.put(
                GpuConsumerReady(gpu_index, config.variant, config.max_atoms)
            )

    class Context:
        @staticmethod
        def Queue(maxsize=0):
            return queue.Queue(maxsize=maxsize)

    Context.Process = Process

    predictor = object.__new__(MCAGpuPredictor)
    predictor._context = Context()
    predictor._request_queues = []
    predictor._gpu_processes = []
    predictor.gpu_indices = (1, 3)
    predictor.variant = "fp16"
    predictor.max_atoms = 512
    predictor.gpu_batch_size = 2048
    predictor.resource_plan = SimpleNamespace(
        gpu_cpu_assignments=(
            SimpleNamespace(gpu_index=1, cpu_ids=(2, 3)),
            SimpleNamespace(gpu_index=3, cpu_ids=(4, 5)),
        )
    )

    predictor._start_gpu_consumers("/resolved/all-atom.onnx")

    assert len(configs) == 2
    assert configs[0] is configs[1]
    assert configs[0] == GpuConsumerConfig(
        "/resolved/all-atom.onnx",
        "fp16",
        512,
        2048,
    )


def test_gpu_predictor_rejects_use_after_close():
    predictor = object.__new__(MCAGpuPredictor)
    predictor._closed = True

    with pytest.raises(RuntimeError, match="closed"):
        next(predictor.iter_predict(["C"]))


def test_gpu_predictor_fails_before_resource_queries_without_cuda(monkeypatch):
    from mca import gpu_backend as module

    monkeypatch.setattr(module, "cuda_provider_available", lambda: False)
    monkeypatch.setattr(
        module,
        "gpu_total_memories_bytes",
        lambda indices: pytest.fail("GPU memory must not be queried"),
    )

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        MCAGpuPredictor(site_cpu_per_gpu=1, infer_cpu_per_gpu=1)


def test_close_restores_the_calling_thread_affinity(monkeypatch):
    from mca import gpu_backend as module

    predictor = object.__new__(MCAGpuPredictor)
    predictor._closed = False
    predictor._site_pool = None
    predictor._feature_pool = None
    predictor._request_queues = []
    predictor._gpu_processes = []
    predictor._result_queue = None
    predictor._site_cpu_queue = None
    predictor._feature_cpu_queue = None
    predictor._owner_affinity_restricted = True
    predictor._owner_native_id = 123
    predictor._owner_cpu_affinity = (4, 5, 6)
    calls = []
    monkeypatch.setattr(
        module.os,
        "sched_setaffinity",
        lambda process, cpus: calls.append((process, cpus)),
    )

    predictor.close()

    assert calls == [(123, {4, 5, 6})]
    assert not predictor._owner_affinity_restricted


def test_startup_failure_restores_affinity_before_propagating(monkeypatch):
    from mca import gpu_backend as module

    plan = SimpleNamespace(
        gpu_indices=(0,),
        assigned_cpu_ids=(2, 3),
        available_cpu_ids=(2, 3, 4, 5),
    )
    calls = []
    monkeypatch.setattr(module, "cuda_provider_available", lambda: True)
    monkeypatch.setattr(module, "build_gpu_resource_plan", lambda **kwargs: plan)
    monkeypatch.setattr(
        module,
        "gpu_total_memories_bytes",
        lambda indices: (_ for _ in ()).throw(RuntimeError("query failed")),
    )
    monkeypatch.setattr(
        module,
        "pin_current_process",
        lambda cpus: calls.append((0, set(cpus))),
    )
    monkeypatch.setattr(
        module.os,
        "sched_setaffinity",
        lambda process, cpus: calls.append((process, cpus)),
    )

    with pytest.raises(RuntimeError, match="query failed"):
        MCAGpuPredictor()

    assert calls[0] == (0, {2, 3})
    assert calls[1][1] == {2, 3, 4, 5}


def test_clean_consumer_exit_fails_fast(monkeypatch):
    from mca import gpu_backend as module

    predictor = object.__new__(MCAGpuPredictor)
    predictor._result_queue = queue.Queue()
    predictor._gpu_processes = [
        SimpleNamespace(exitcode=0, name="clean-exit-consumer")
    ]
    monkeypatch.setattr(module, "_IPC_POLL_SECONDS", 0.001)

    with pytest.raises(RuntimeError, match="exitcode=0"):
        predictor._receive_gpu_message(deadline=time.monotonic() + 0.1)


def test_gpu_shutdown_timeout_budget_is_constant_across_processes(monkeypatch):
    from mca import gpu_backend as module

    clock = [100.0]
    join_timeouts = []

    class Process:
        def join(self, timeout):
            join_timeouts.append(timeout)
            clock[0] += timeout

        @staticmethod
        def is_alive():
            return True

        @staticmethod
        def terminate():
            pass

        @staticmethod
        def kill():
            pass

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    MCAGpuPredictor._stop_processes([Process() for _ in range(8)])

    assert clock[0] == pytest.approx(100.0 + 3 * module._SHUTDOWN_TIMEOUT_SECONDS)
    assert sum(join_timeouts) == pytest.approx(
        3 * module._SHUTDOWN_TIMEOUT_SECONDS
    )


def test_cpu_worker_exit_fails_fast():
    pool = SimpleNamespace(
        _pool=[SimpleNamespace(pid=123, exitcode=0)],
    )

    with pytest.raises(RuntimeError, match="worker exited unexpectedly"):
        MCAGpuPredictor._check_pool_health(
            object.__new__(MCAGpuPredictor),
            pool,
            (123,),
            "feature construction",
        )


def test_close_does_not_wait_for_unconsumed_queue_feeder():
    script = """
import multiprocessing as mp
from hotpot.cheminfo.AImodels.mca.gpu_backend import MCAGpuPredictor

request_queue = mp.get_context("spawn").Queue(maxsize=2)
request_queue.put(b"x" * (16 * 1024 * 1024))
predictor = object.__new__(MCAGpuPredictor)
predictor._closed = False
predictor._site_pool = None
predictor._feature_pool = None
predictor._request_queues = [request_queue]
predictor._gpu_processes = []
predictor._result_queue = None
predictor._site_cpu_queue = None
predictor._feature_cpu_queue = None
predictor._owner_affinity_restricted = False
predictor.close()
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
