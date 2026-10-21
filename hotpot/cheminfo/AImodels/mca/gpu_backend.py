"""Bounded spawn-based CPU/GPU pipeline for MCA inference."""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import os
import pickle
import queue
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np

from .batching import LogicalBatch, LogicalBatchPlanner, scatter_batch_predictions
from .cpu_backend import MCACpuPredictor
from .gpu_consumer import (
    GpuBatchRequest,
    GpuBatchResult,
    GpuConsumerConfig,
    GpuConsumerFailure,
    GpuConsumerReady,
    gpu_consumer_main,
)
from .gpu_memory import (
    MCAGpuOutOfMemoryError,
    gpu_memory_bytes,
    gpu_total_memories_bytes,
    is_cuda_out_of_memory,
    resolve_gpu_batch_size,
    validate_gpu_batch_size,
)
from .pipeline import (
    DEFAULT_CONFORMER_SEED,
    PreparedMolecule,
    build_prediction,
    is_single_molecule_input,
    prepare_molecule,
)
from .resources import (
    DEFAULT_INFER_CPU_PER_GPU,
    DEFAULT_SITE_CPU_PER_GPU,
    build_gpu_resource_plan,
    gpu_window_atom_budget,
    pin_current_process,
    pin_worker_from_queue,
    validate_gpu_resource_request,
)
from .result_types import MoleculePrediction
from .runtime import cuda_provider_available, resolved_model_artifact
from .serialization import MoleculeDTO, normalize_molecule, validate_dto
from .workers import (
    FeatureTask,
    SiteDetectionTask,
    build_features_worker,
    detect_sites_worker,
)


_IPC_POLL_SECONDS = 0.25
_STARTUP_TIMEOUT_SECONDS = 300.0
_TASK_TIMEOUT_SECONDS = 900.0
_SHUTDOWN_TIMEOUT_SECONDS = 5.0
_PREPARED_SPOOL_MEMORY_BYTES = 8 * 1024**2
MIN_GPU_SITE_COUNT = 256


logger = logging.getLogger(__name__)


@dataclass
class _MoleculeState:
    molecule: MoleculeDTO
    values: np.ndarray
    filled: np.ndarray
    sites: tuple[tuple[int, str], ...] | None = None
    feature_consumed: bool = False
    admission_credit_released: bool = False


@dataclass
class _PendingChunk:
    batch: LogicalBatch
    missing_keys: set[tuple[int, int]]


@dataclass(frozen=True)
class _PendingRequest:
    keys: tuple[tuple[int, int], ...]
    gpu_index: int
    batch_ordinal: int


class _PreparedMoleculeSpool:
    def __init__(self):
        self._file = tempfile.SpooledTemporaryFile(
            max_size=_PREPARED_SPOOL_MEMORY_BYTES,
            mode="w+b",
        )
        self._count = 0

    def append(self, prepared: PreparedMolecule) -> None:
        pickle.dump(prepared, self._file, protocol=pickle.HIGHEST_PROTOCOL)
        self._count += 1

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        self._file.seek(0)
        for _ in range(self._count):
            yield pickle.load(self._file)

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _enqueue_pool_result(completion_queue, stage, result) -> None:
    completion_queue.put((stage, True, result))


def _enqueue_pool_error(completion_queue, stage, error) -> None:
    completion_queue.put((stage, False, error))


class MCAGpuPredictor:
    """GPU-preferred MCA predictor with lazy, persistent CUDA workers."""

    def __init__(
        self,
        model_dir=None,
        *,
        variant: str | None = None,
        gpu_count: int | None = None,
        gpu_indices: Sequence[int] | None = None,
        site_cpu_per_gpu: int = DEFAULT_SITE_CPU_PER_GPU,
        infer_cpu_per_gpu: int = DEFAULT_INFER_CPU_PER_GPU,
        gpu_batch_size: int | None = None,
        conformer_seed: int = DEFAULT_CONFORMER_SEED,
        verify_model: bool = True,
        allow_charged: bool = False,
        model_source=None,
    ):
        indices, site_count, infer_count = validate_gpu_resource_request(
            gpu_count=gpu_count,
            gpu_indices=gpu_indices,
            site_cpu_per_gpu=site_cpu_per_gpu,
            infer_cpu_per_gpu=infer_cpu_per_gpu,
        )
        if gpu_batch_size is not None:
            validate_gpu_batch_size(gpu_batch_size)
        self.gpu_indices = indices
        self.resource_plan = None
        self.gpu_memory_totals = ()
        self.gpu_batch_size = gpu_batch_size
        self.conformer_seed = conformer_seed
        self.allow_charged = allow_charged
        self._model_options = {
            "model_dir": model_dir,
            "variant": variant,
            "verify_model": verify_model,
            "model_source": model_source,
        }
        self._resource_options = {
            "gpu_indices": indices,
            "site_cpu_per_gpu": site_count,
            "infer_cpu_per_gpu": infer_count,
        }
        self._requested_gpu_batch_size = gpu_batch_size
        self._closed = False
        self._gpu_started = False
        self._predict_lock = threading.Lock()
        self._next_request_id = 0
        self._next_gpu_position = 0
        self._context = mp.get_context("spawn")
        self._site_pool = None
        self._feature_pool = None
        self._site_cpu_queue = None
        self._feature_cpu_queue = None
        self._result_queue = None
        self._request_queues = []
        self._gpu_processes = []
        self._owner_native_id = None
        self._owner_cpu_affinity = ()
        self._owner_affinity_restricted = False

    def _ensure_gpu_started(self) -> None:
        if self._gpu_started:
            return
        if not cuda_provider_available():
            raise RuntimeError("CUDAExecutionProvider is not available")
        self.resource_plan = build_gpu_resource_plan(**self._resource_options)
        self._owner_native_id = threading.get_native_id()
        self._owner_cpu_affinity = self.resource_plan.available_cpu_ids
        try:
            pin_current_process(self.resource_plan.assigned_cpu_ids)
            self._owner_affinity_restricted = True
            self.gpu_memory_totals = gpu_total_memories_bytes(self.gpu_indices)
            self.gpu_batch_size = resolve_gpu_batch_size(
                self.gpu_memory_totals,
                self._requested_gpu_batch_size,
            )
            self._start_cpu_pools()
            with resolved_model_artifact(
                **self._model_options,
            ) as artifact:
                self.variant = artifact.variant
                self.max_atoms = int(artifact.manifest["max_atoms"])
                self._start_gpu_consumers(str(artifact.session_path))
            self._gpu_started = True
        except BaseException:
            self.close()
            raise

    def _cpu_id_queue(self, cpu_ids: Sequence[int]):
        cpu_queue = self._context.Queue()
        for cpu_id in cpu_ids:
            cpu_queue.put(cpu_id)
        return cpu_queue

    def _start_cpu_pools(self) -> None:
        site_cpu_ids = tuple(
            assignment.cpu_id
            for assignment in self.resource_plan.site_worker_assignments
        )
        infer_cpu_ids = tuple(
            assignment.cpu_id
            for assignment in self.resource_plan.infer_worker_assignments
        )
        self._site_cpu_queue = self._cpu_id_queue(site_cpu_ids)
        self._feature_cpu_queue = self._cpu_id_queue(infer_cpu_ids)
        self._site_pool = self._context.Pool(
            len(site_cpu_ids),
            initializer=pin_worker_from_queue,
            initargs=(self._site_cpu_queue,),
        )
        self._feature_pool = self._context.Pool(
            len(infer_cpu_ids),
            initializer=pin_worker_from_queue,
            initargs=(self._feature_cpu_queue,),
        )

    def _start_gpu_consumers(self, session_path: str) -> None:
        self._result_queue = self._context.Queue()
        config = GpuConsumerConfig(
            session_path=session_path,
            variant=self.variant,
            max_atoms=self.max_atoms,
            batch_size=self.gpu_batch_size,
        )
        assignments = {
            assignment.gpu_index: assignment.cpu_ids
            for assignment in self.resource_plan.gpu_cpu_assignments
        }
        for gpu_index in self.gpu_indices:
            request_queue = self._context.Queue(maxsize=2)
            process = self._context.Process(
                target=gpu_consumer_main,
                args=(
                    gpu_index,
                    assignments[gpu_index],
                    config,
                    request_queue,
                    self._result_queue,
                ),
                name=f"hotpot-mca-gpu-{gpu_index}",
            )
            process.start()
            self._request_queues.append(request_queue)
            self._gpu_processes.append(process)

        ready = set()
        startup_deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        while len(ready) < len(self.gpu_indices):
            message = self._receive_gpu_message(deadline=startup_deadline)
            if isinstance(message, GpuConsumerFailure):
                raise self._remote_failure(message, "CUDA session initialization")
            if not isinstance(message, GpuConsumerReady):
                raise RuntimeError(
                    f"Unexpected MCA GPU initialization message: {message!r}"
                )
            if message.variant != self.variant or message.max_atoms != self.max_atoms:
                raise RuntimeError(
                    "MCA GPU consumers loaded inconsistent model artifacts"
                )
            ready.add(message.gpu_index)

    def predict(self, molecules):
        single = is_single_molecule_input(molecules)
        values = [molecules] if single else molecules
        results = list(self.iter_predict(values))
        return results[0] if single else results

    def iter_predict(self, molecules) -> Iterator[MoleculePrediction]:
        if self._closed:
            raise RuntimeError("The MCA GPU predictor is closed")
        if not self._predict_lock.acquire(blocking=False):
            raise RuntimeError(
                "The MCA GPU predictor does not support concurrent predictions"
            )
        completed = False
        try:
            yield from self._iter_predict_routed(molecules)
            completed = True
        finally:
            self._predict_lock.release()
            if not completed:
                self.close()

    def _iter_predict_routed(self, molecules) -> Iterator[MoleculePrediction]:
        molecule_iterator = iter(molecules)
        site_count = 0
        with _PreparedMoleculeSpool() as prepared_molecules:
            while site_count < MIN_GPU_SITE_COUNT:
                try:
                    value = next(molecule_iterator)
                except StopIteration:
                    break
                prepared = prepare_molecule(
                    value,
                    len(prepared_molecules),
                    self.allow_charged,
                )
                prepared_molecules.append(prepared)
                site_count += len(prepared.sites)

            if not prepared_molecules:
                return
            if site_count < MIN_GPU_SITE_COUNT:
                logger.debug(
                    "MCA backend routing: requested=gpu effective=cpu "
                    "site_count=%d threshold=%d",
                    site_count,
                    MIN_GPU_SITE_COUNT,
                )
                with MCACpuPredictor(
                    conformer_seed=self.conformer_seed,
                    allow_charged=self.allow_charged,
                    **self._model_options,
                ) as predictor:
                    yield from predictor._iter_predict_prepared(prepared_molecules)
                return

            self._ensure_gpu_started()
            yield from self._iter_predict(
                itertools.chain(prepared_molecules, molecule_iterator)
            )

    def _iter_predict(self, molecules) -> Iterator[MoleculePrediction]:
        planner = LogicalBatchPlanner(self.gpu_batch_size)
        states: dict[int, _MoleculeState] = {}
        features: dict[int, dict[str, np.ndarray]] = {}
        encoding_cache: dict[tuple[int, int], np.ndarray] = {}
        pending_encoding_keys: set[tuple[int, int]] = set()
        pending_chunks: dict[int, _PendingChunk] = {}
        waiting_chunks: dict[tuple[int, int], set[int]] = {}
        pending_requests: dict[int, _PendingRequest] = {}
        pending_batches = deque()
        completion_queue = queue.Queue()
        pending_cpu = {"site": 0, "feature": 0}
        pending_cpu_started = {"site": {}, "feature": {}}
        next_input_index = 0
        next_output_index = 0
        admission_atom_count = 0
        resident_atom_count = 0
        input_exhausted = False
        planner_finished = False
        max_inflight = max(1, 2 * len(self.gpu_indices))
        worker_count = (
            len(self.resource_plan.site_worker_assignments)
            + len(self.resource_plan.infer_worker_assignments)
        )
        window_atom_budget = gpu_window_atom_budget(
            self.gpu_batch_size,
            worker_count,
        )
        resident_atom_capacity = window_atom_budget + max_inflight * (
            self.gpu_batch_size + 2 * self.max_atoms
        )
        molecule_iterator = iter(molecules)
        pool_state = {
            "site": (
                self._site_pool,
                self._pool_process_ids(self._site_pool),
                "site detection",
            ),
            "feature": (
                self._feature_pool,
                self._pool_process_ids(self._feature_pool),
                "feature construction",
            ),
        }

        while next_output_index < next_input_index or not input_exhausted:
            while (
                not input_exhausted
                and admission_atom_count < window_atom_budget
                and resident_atom_count < resident_atom_capacity
            ):
                try:
                    value = next(molecule_iterator)
                except StopIteration:
                    input_exhausted = True
                    break
                molecule_index = next_input_index
                next_input_index += 1
                if isinstance(value, PreparedMolecule):
                    molecule = value.molecule
                    sites = value.sites
                else:
                    molecule = normalize_molecule(value)
                    validate_dto(molecule, molecule_index, self.allow_charged)
                    sites = None
                states[molecule_index] = _MoleculeState(
                    molecule=molecule,
                    values=np.empty(molecule.atom_count, dtype=np.float64),
                    filled=np.zeros(molecule.atom_count, dtype=bool),
                    sites=sites,
                )
                admission_atom_count += molecule.atom_count
                resident_atom_count += molecule.atom_count
                pending_batches.extend(
                    planner.push(((molecule_index, molecule.atom_count),))
                )
                submitted_at = time.monotonic()
                pending_cpu["feature"] += 1
                pending_cpu_started["feature"][molecule_index] = submitted_at
                if sites is None:
                    pending_cpu["site"] += 1
                    pending_cpu_started["site"][molecule_index] = submitted_at
                    self._site_pool.apply_async(
                        detect_sites_worker,
                        (SiteDetectionTask(molecule_index, molecule),),
                        callback=partial(
                            _enqueue_pool_result,
                            completion_queue,
                            "site",
                        ),
                        error_callback=partial(
                            _enqueue_pool_error,
                            completion_queue,
                            "site",
                        ),
                    )
                self._feature_pool.apply_async(
                    build_features_worker,
                    (
                        FeatureTask(
                            molecule_index,
                            molecule,
                            self.max_atoms,
                            self.conformer_seed,
                        ),
                    ),
                    callback=partial(
                        _enqueue_pool_result,
                        completion_queue,
                        "feature",
                    ),
                    error_callback=partial(
                        _enqueue_pool_error,
                        completion_queue,
                        "feature",
                    ),
                )

            if input_exhausted and not planner_finished:
                pending_batches.extend(planner.finish())
                planner_finished = True

            made_progress = True
            while made_progress:
                made_progress = False
                while pending_batches:
                    batch = pending_batches[0]
                    if any(
                        molecule_index not in features
                        for molecule_index in batch.unique_molecule_indices
                    ):
                        break
                    if len(pending_requests) >= max_inflight:
                        if self._try_drain_gpu_result(
                            states,
                            encoding_cache,
                            pending_encoding_keys,
                            pending_chunks,
                            waiting_chunks,
                            pending_requests,
                        ):
                            made_progress = True
                            continue
                        break
                    pending_batches.popleft()
                    self._submit_batch(
                        batch,
                        states,
                        features,
                        encoding_cache,
                        pending_encoding_keys,
                        pending_chunks,
                        waiting_chunks,
                        pending_requests,
                    )
                    admission_atom_count -= self._release_submitted_features(
                        batch,
                        states,
                        features,
                    )
                    made_progress = True

                while pending_requests and self._try_drain_gpu_result(
                    states,
                    encoding_cache,
                    pending_encoding_keys,
                    pending_chunks,
                    waiting_chunks,
                    pending_requests,
                ):
                    made_progress = True

                ready, new_output_index = self._pop_ready_results(
                    states,
                    encoding_cache,
                    next_output_index,
                )
                if ready:
                    resident_atom_count -= sum(
                        len(result.atom_predictions) for result in ready
                    )
                    next_output_index = new_output_index
                    made_progress = True
                    yield from ready

            if next_output_index == next_input_index and input_exhausted:
                break

            if (
                not input_exhausted
                and admission_atom_count < window_atom_budget
                and resident_atom_count < resident_atom_capacity
            ):
                continue

            try:
                stage, succeeded, result = completion_queue.get_nowait()
            except queue.Empty:
                stage = None
            if stage is not None:
                admission_atom_count -= self._handle_cpu_completion(
                    stage,
                    succeeded,
                    result,
                    states,
                    features,
                    pending_cpu,
                    pending_cpu_started,
                    pool_state,
                )
                continue

            if any(pending_cpu.values()):
                remaining = self._remaining_cpu_timeout(pending_cpu_started)
                if remaining <= 0:
                    operations = ", ".join(
                        pool_state[stage][2]
                        for stage, count in pending_cpu.items()
                        if count
                    )
                    raise TimeoutError(
                        f"MCA {operations} exceeded {_TASK_TIMEOUT_SECONDS:g} seconds"
                    )
                try:
                    stage, succeeded, result = completion_queue.get(
                        timeout=min(_IPC_POLL_SECONDS, remaining)
                    )
                except queue.Empty:
                    for stage, count in pending_cpu.items():
                        if count:
                            pool, expected_pids, operation = pool_state[stage]
                            self._check_pool_health(pool, expected_pids, operation)
                    continue
                admission_atom_count -= self._handle_cpu_completion(
                    stage,
                    succeeded,
                    result,
                    states,
                    features,
                    pending_cpu,
                    pending_cpu_started,
                    pool_state,
                )
                continue

            if pending_requests:
                self._drain_gpu_result(
                    states,
                    encoding_cache,
                    pending_encoding_keys,
                    pending_chunks,
                    waiting_chunks,
                    pending_requests,
                )
                continue

            raise RuntimeError("MCA GPU pipeline stopped making progress")

        if next_output_index != next_input_index:
            raise RuntimeError(
                "MCA GPU pipeline finished without producing every input molecule"
            )

    @staticmethod
    def _release_submitted_features(batch, states, features) -> int:
        released_atoms = 0
        for row in batch.rows:
            state = states[row.molecule_index]
            if row.atom_index == state.molecule.atom_count - 1:
                features.pop(row.molecule_index, None)
                state.feature_consumed = True
                released_atoms += MCAGpuPredictor._release_admission_credit(
                    state
                )
        return released_atoms

    @staticmethod
    def _release_admission_credit(state: _MoleculeState) -> int:
        if (
            state.admission_credit_released
            or state.sites is None
            or not state.feature_consumed
        ):
            return 0
        state.admission_credit_released = True
        return state.molecule.atom_count

    @staticmethod
    def _remaining_cpu_timeout(pending_cpu_started) -> float:
        oldest = min(
            started_at
            for stage_tasks in pending_cpu_started.values()
            for started_at in stage_tasks.values()
        )
        return oldest + _TASK_TIMEOUT_SECONDS - time.monotonic()

    def _handle_cpu_completion(
        self,
        stage,
        succeeded,
        result,
        states,
        features,
        pending_cpu,
        pending_cpu_started,
        pool_state,
    ) -> int:
        for pool, expected_pids, operation in pool_state.values():
            self._check_pool_health(pool, expected_pids, operation)
        pending_cpu[stage] -= 1
        if not succeeded:
            raise result
        pending_cpu_started[stage].pop(result.molecule_index)
        if stage == "feature":
            features[result.molecule_index] = result.feature
            return 0
        state = states[result.molecule_index]
        state.sites = result.sites
        return self._release_admission_credit(state)

    def _submit_batch(
        self,
        batch,
        states,
        features,
        encoding_cache,
        pending_encoding_keys,
        pending_chunks,
        waiting_chunks,
        pending_requests,
    ) -> None:
        missing_keys = {
            key for key in batch.encoding_keys if key not in encoding_cache
        }
        pending_chunks[batch.ordinal] = _PendingChunk(batch, set(missing_keys))
        for key in missing_keys:
            waiting_chunks.setdefault(key, set()).add(batch.ordinal)

        request_keys = tuple(
            key for key in batch.encoding_keys if key in missing_keys
            and key not in pending_encoding_keys
        )
        if request_keys:
            request_id = self._next_request_id
            self._next_request_id += 1
            request = GpuBatchRequest(
                request_id=request_id,
                batch_ordinal=batch.ordinal,
                pad_width=batch.pad_width,
                molecule_indices=tuple(key[0] for key in request_keys),
                features=tuple(features[key[0]] for key in request_keys),
            )
            position = self._next_gpu_position
            self._next_gpu_position = (position + 1) % len(self._request_queues)
            self._request_queues[position].put(request)
            pending_requests[request_id] = _PendingRequest(
                keys=request_keys,
                gpu_index=self.gpu_indices[position],
                batch_ordinal=batch.ordinal,
            )
            pending_encoding_keys.update(request_keys)

        self._resolve_cached_chunks(
            states,
            encoding_cache,
            pending_chunks,
        )

    def _drain_gpu_result(
        self,
        states,
        encoding_cache,
        pending_encoding_keys,
        pending_chunks,
        waiting_chunks,
        pending_requests,
        *,
        message=None,
    ) -> None:
        if message is None:
            message = self._receive_gpu_message()
        if isinstance(message, GpuConsumerFailure):
            raise self._remote_failure(message, "all-atom inference")
        if not isinstance(message, GpuBatchResult):
            raise RuntimeError(f"Unexpected MCA GPU result message: {message!r}")
        expected = pending_requests.pop(message.request_id)
        if message.gpu_index != expected.gpu_index:
            raise RuntimeError(
                "MCA GPU result came from a different GPU than the submitted request"
            )
        if message.batch_ordinal != expected.batch_ordinal:
            raise RuntimeError(
                "MCA GPU result batch ordinal does not match the submitted request"
            )
        returned_keys = tuple(
            (molecule_index, message.pad_width)
            for molecule_index in message.molecule_indices
        )
        if returned_keys != expected.keys:
            raise RuntimeError("MCA GPU result keys do not match the submitted request")
        if len(message.predictions) != len(expected.keys):
            raise RuntimeError(
                "MCA GPU result prediction count does not match the submitted request"
            )
        for key, prediction in zip(returned_keys, message.predictions):
            encoding_cache[key] = prediction
            pending_encoding_keys.remove(key)
            for chunk_ordinal in waiting_chunks.pop(key, ()):
                pending_chunks[chunk_ordinal].missing_keys.remove(key)
        self._resolve_cached_chunks(
            states,
            encoding_cache,
            pending_chunks,
        )

    def _try_drain_gpu_result(
        self,
        states,
        encoding_cache,
        pending_encoding_keys,
        pending_chunks,
        waiting_chunks,
        pending_requests,
    ) -> bool:
        try:
            message = self._result_queue.get_nowait()
        except queue.Empty:
            self._check_gpu_health()
            return False
        self._drain_gpu_result(
            states,
            encoding_cache,
            pending_encoding_keys,
            pending_chunks,
            waiting_chunks,
            pending_requests,
            message=message,
        )
        return True

    @staticmethod
    def _resolve_cached_chunks(
        states,
        encoding_cache,
        pending_chunks,
    ) -> None:
        ready_ordinals = [
            ordinal
            for ordinal, chunk in pending_chunks.items()
            if not chunk.missing_keys
        ]
        for ordinal in ready_ordinals:
            batch = pending_chunks.pop(ordinal).batch
            values = scatter_batch_predictions(batch, encoding_cache)
            for row, value in zip(batch.rows, values):
                state = states[row.molecule_index]
                state.values[row.atom_index] = value
                state.filled[row.atom_index] = True

    def _pop_ready_results(self, states, encoding_cache, next_output_index):
        results = []
        while next_output_index in states:
            state = states[next_output_index]
            if state.sites is None or not bool(np.all(state.filled)):
                break
            results.append(
                build_prediction(
                    state.molecule,
                    state.sites,
                    state.values,
                    self.variant,
                )
            )
            del states[next_output_index]
            for key in [key for key in encoding_cache if key[0] == next_output_index]:
                del encoding_cache[key]
            next_output_index += 1
        return results, next_output_index

    @staticmethod
    def _pool_processes(pool):
        return tuple(getattr(pool, "_pool", ()))

    @classmethod
    def _pool_process_ids(cls, pool):
        return tuple(process.pid for process in cls._pool_processes(pool))

    def _check_pool_health(self, pool, expected_pids, operation):
        processes = self._pool_processes(pool)
        current_pids = tuple(process.pid for process in processes)
        dead = [
            process
            for process in processes
            if process.exitcode is not None
        ]
        if dead or current_pids != expected_pids:
            details = ", ".join(
                f"pid={process.pid} exitcode={process.exitcode}"
                for process in dead
            ) or (
                f"worker set changed from {expected_pids!r} "
                f"to {current_pids!r}"
            )
            raise RuntimeError(
                f"MCA {operation} worker exited unexpectedly: {details}"
            )

    def _receive_gpu_message(self, *, deadline=None):
        if deadline is None:
            deadline = time.monotonic() + _TASK_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for an MCA GPU consumer")
            try:
                return self._result_queue.get(
                    timeout=min(_IPC_POLL_SECONDS, remaining)
                )
            except queue.Empty:
                self._check_gpu_health()

    def _check_gpu_health(self):
        dead = [
            process
            for process in self._gpu_processes
            if process.exitcode is not None
        ]
        if dead:
            details = ", ".join(
                f"{process.name} exitcode={process.exitcode}"
                for process in dead
            )
            raise RuntimeError(f"MCA GPU consumer exited unexpectedly: {details}")

    def _remote_failure(self, failure, operation):
        remote_error = RuntimeError(
            f"{failure.exception_type}: {failure.message}\n{failure.traceback_text}"
        )
        if is_cuda_out_of_memory(remote_error):
            total = dict(zip(self.gpu_indices, self.gpu_memory_totals)).get(
                failure.gpu_index
            )
            try:
                free = gpu_memory_bytes(failure.gpu_index).free_bytes
            except Exception:
                free = None
            error = MCAGpuOutOfMemoryError(
                gpu_index=failure.gpu_index,
                batch_size=self.gpu_batch_size,
                operation=operation,
                total_memory_bytes=total,
                free_memory_bytes=free,
                variant=self.variant,
            )
            error.__cause__ = remote_error
            return error
        error = RuntimeError(
            f"MCA GPU {failure.gpu_index} failed during {operation}: "
            f"{failure.exception_type}: {failure.message}\n"
            f"Remote traceback:\n{failure.traceback_text}"
        )
        error.__cause__ = remote_error
        return error

    @staticmethod
    def _stop_pool(pool) -> None:
        if pool is None:
            return
        processes = MCAGpuPredictor._pool_processes(pool)

        def terminate_pool():
            try:
                pool.terminate()
            except Exception:
                pass

        terminate_thread = threading.Thread(
            target=terminate_pool,
            name="hotpot-mca-pool-terminate",
            daemon=True,
        )
        terminate_thread.start()
        terminate_thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.join(timeout=remaining)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
            except (OSError, ValueError):
                pass

    @staticmethod
    def _stop_processes(processes) -> None:
        processes = tuple(processes)

        def wait_for_stage():
            deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
            for process in processes:
                try:
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
                except (OSError, ValueError):
                    pass

        wait_for_stage()
        alive = []
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
                    alive.append(process)
            except (OSError, ValueError):
                pass
        processes = tuple(alive)
        wait_for_stage()

        alive = []
        for process in processes:
            try:
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    alive.append(process)
            except (OSError, ValueError):
                pass
        processes = tuple(alive)
        wait_for_stage()

    @staticmethod
    def _close_queue(ipc_queue) -> None:
        if ipc_queue is None:
            return
        try:
            ipc_queue.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass
        try:
            ipc_queue.close()
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            for pool in (self._site_pool, self._feature_pool):
                self._stop_pool(pool)

            for request_queue in self._request_queues:
                try:
                    request_queue.put_nowait(None)
                except (queue.Full, ValueError):
                    pass
            self._stop_processes(self._gpu_processes)

            for ipc_queue in (
                *self._request_queues,
                self._result_queue,
                self._site_cpu_queue,
                self._feature_cpu_queue,
            ):
                if ipc_queue is not None:
                    self._close_queue(ipc_queue)
        finally:
            if self._owner_affinity_restricted:
                try:
                    os.sched_setaffinity(
                        self._owner_native_id,
                        set(self._owner_cpu_affinity),
                    )
                except OSError:
                    pass
                finally:
                    self._owner_affinity_restricted = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
