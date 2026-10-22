"""Bounded CPU MCA backend with result-compatible atom-row batching."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
from collections.abc import Iterable, Iterator
from itertools import islice

import numpy as np

from .batching import LogicalBatchPlanner, scatter_batch_predictions
from .featurizer import collate_molecule_rows
from .pipeline import (
    DEFAULT_CONFORMER_SEED,
    PreparedMolecule,
    build_prediction,
    is_single_molecule_input,
)
from .resources import (
    current_cpu_affinity,
    pin_current_process,
    pin_worker_from_queue,
    resolve_cpu_ids,
)
from .result_types import MoleculePrediction
from .runtime import create_cpu_runtime
from .serialization import MoleculeDTO, normalize_molecule, validate_dto
from .workers import (
    CPUPreprocessResult,
    CPUPreprocessTask,
    cpu_preprocess_worker,
)


CPU_PREPROCESS_WINDOW_MOLECULES = 64


class MCACpuPredictor:
    def __init__(
        self,
        model_dir=None,
        *,
        variant: str | None = None,
        batch_size: int = 64,
        cpu_count: int | None = None,
        conformer_seed: int = DEFAULT_CONFORMER_SEED,
        verify_model: bool = True,
        allow_charged: bool = False,
        model_source=None,
    ):
        owner_affinity = current_cpu_affinity()
        self.cpu_ids = resolve_cpu_ids(cpu_count, cpu_affinity=owner_affinity)
        self.cpu_count = len(self.cpu_ids)
        pin_current_process(self.cpu_ids)
        try:
            self.runtime = create_cpu_runtime(
                model_dir=model_dir,
                variant=variant,
                verify_model=verify_model,
                model_source=model_source,
                cpu_threads=self.cpu_count,
            )
        finally:
            os.sched_setaffinity(0, set(owner_affinity))

        self.batch_size = batch_size
        self.conformer_seed = conformer_seed
        self.max_atoms = int(self.runtime.manifest["max_atoms"])
        self.allow_charged = allow_charged
        self._context = mp.get_context("spawn")
        self._pool = None
        self._cpu_queue = None
        self._closed = False
        self._predict_lock = threading.Lock()

    def predict(self, molecules):
        single = is_single_molecule_input(molecules)
        values = [molecules] if single else molecules
        results = list(self.iter_predict(values))
        return results[0] if single else results

    def iter_predict(self, molecules) -> Iterator[MoleculePrediction]:
        if self._closed:
            raise RuntimeError("The MCA CPU predictor is closed")
        if not self._predict_lock.acquire(blocking=False):
            raise RuntimeError(
                "The MCA CPU predictor does not support concurrent predictions"
            )
        owner_affinity = current_cpu_affinity()
        try:
            pin_current_process(self.cpu_ids)
            yield from self._iter_predict(molecules)
        finally:
            os.sched_setaffinity(0, set(owner_affinity))
            self._predict_lock.release()

    def _ensure_pool(self):
        if self._pool is None:
            self._cpu_queue = self._context.SimpleQueue()
            for cpu_id in self.cpu_ids:
                self._cpu_queue.put(cpu_id)
            self._pool = self._context.Pool(
                processes=self.cpu_count,
                initializer=pin_worker_from_queue,
                initargs=(self._cpu_queue,),
            )
        return self._pool

    def _preprocess_task(self, value, molecule_index: int) -> CPUPreprocessTask:
        if isinstance(value, PreparedMolecule):
            molecule = value.molecule
            sites = value.sites
        else:
            molecule = normalize_molecule(value)
            validate_dto(molecule, molecule_index, self.allow_charged)
            sites = None
        return CPUPreprocessTask(
            molecule_index=molecule_index,
            molecule=molecule,
            sites=sites,
            max_atoms=self.max_atoms,
            conformer_seed=self.conformer_seed,
        )

    def _preprocess_window(
        self,
        values,
        first_molecule_index: int,
    ) -> tuple[CPUPreprocessResult, ...]:
        tasks = tuple(
            self._preprocess_task(value, first_molecule_index + offset)
            for offset, value in enumerate(values)
        )
        if len(tasks) == 1:
            return (cpu_preprocess_worker(tasks[0]),)
        return tuple(self._ensure_pool().map(cpu_preprocess_worker, tasks, chunksize=1))

    def _iter_predict(self, molecules) -> Iterator[MoleculePrediction]:
        yield from self._iter_predict_prepared(molecules)

    def _iter_predict_prepared(
        self,
        prepared_molecules: Iterable[PreparedMolecule],
    ) -> Iterator[MoleculePrediction]:
        planner = LogicalBatchPlanner(self.batch_size)
        states: dict[int, dict[str, object]] = {}
        features: dict[int, dict[str, np.ndarray]] = {}
        encoding_cache: dict[tuple[int, int], np.ndarray] = {}
        next_input_index = 0
        next_output_index = 0
        input_iterator = iter(prepared_molecules)
        window_size = min(CPU_PREPROCESS_WINDOW_MOLECULES, self.batch_size)

        while values := tuple(islice(input_iterator, window_size)):
            results = self._preprocess_window(values, next_input_index)
            next_input_index += len(results)
            for result in results:
                molecule_index = result.molecule_index
                molecule = result.molecule
                features[molecule_index] = result.feature
                states[molecule_index] = {
                    "molecule": molecule,
                    "sites": result.sites,
                    "values": np.empty(molecule.atom_count, dtype=np.float64),
                    "filled": np.zeros(molecule.atom_count, dtype=bool),
                }

                for batch in planner.push(((molecule_index, molecule.atom_count),)):
                    self._execute_batch(batch, states, features, encoding_cache)
                    self._release_planned_features(batch, states, features)
                ready, next_output_index = self._pop_ready(
                    states,
                    encoding_cache,
                    next_output_index,
                )
                yield from ready

        for batch in planner.finish():
            self._execute_batch(batch, states, features, encoding_cache)
            self._release_planned_features(batch, states, features)
        ready, next_output_index = self._pop_ready(
            states,
            encoding_cache,
            next_output_index,
        )
        yield from ready
        if next_output_index != next_input_index:
            raise RuntimeError(
                "MCA CPU pipeline finished without producing every input molecule"
            )

    def _execute_batch(self, batch, states, features, encoding_cache) -> None:
        missing_indices = [
            molecule_index
            for molecule_index in batch.unique_molecule_indices
            if (molecule_index, batch.pad_width) not in encoding_cache
        ]
        if missing_indices:
            arrays = collate_molecule_rows(
                features,
                missing_indices,
                max_atoms=batch.pad_width,
            )
            predictions = self.runtime.predict(arrays, self.batch_size)
            encoding_cache.update(
                {
                    (molecule_index, batch.pad_width): predictions[row]
                    for row, molecule_index in enumerate(missing_indices)
                }
            )
        values = scatter_batch_predictions(batch, encoding_cache)
        for row, value in zip(batch.rows, values):
            state = states[row.molecule_index]
            state["values"][row.atom_index] = value
            state["filled"][row.atom_index] = True

    @staticmethod
    def _release_planned_features(batch, states, features) -> None:
        for row in batch.rows:
            molecule = states[row.molecule_index]["molecule"]
            if row.atom_index == molecule.atom_count - 1:
                features.pop(row.molecule_index, None)

    def _pop_ready(self, states, encoding_cache, next_output_index):
        results = []
        while next_output_index in states:
            state = states[next_output_index]
            if not bool(np.all(state["filled"])):
                break
            molecule: MoleculeDTO = state["molecule"]
            values = state["values"]
            results.append(
                build_prediction(
                    molecule,
                    state["sites"],
                    values,
                    self.runtime.variant,
                )
            )
            del states[next_output_index]
            for key in [key for key in encoding_cache if key[0] == next_output_index]:
                del encoding_cache[key]
            next_output_index += 1
        return results, next_output_index

    def close(self):
        if self._closed:
            return
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.close()
            pool.join()
        cpu_queue = getattr(self, "_cpu_queue", None)
        if cpu_queue is not None:
            cpu_queue.close()
        self.runtime.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
