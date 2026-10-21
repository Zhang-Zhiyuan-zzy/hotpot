"""Bounded CPU MCA backend with result-compatible atom-row batching."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator

import numpy as np

from .batching import LogicalBatchPlanner, scatter_batch_predictions
from .featurizer import collate_molecule_rows
from .pipeline import (
    DEFAULT_CONFORMER_SEED,
    PreparedMolecule,
    build_features,
    build_prediction,
    is_single_molecule_input,
    prepare_molecule,
)
from .result_types import MoleculePrediction
from .runtime import create_cpu_runtime
from .serialization import MoleculeDTO


class MCACpuPredictor:
    def __init__(
        self,
        model_dir=None,
        *,
        variant: str | None = None,
        batch_size: int = 64,
        conformer_seed: int = DEFAULT_CONFORMER_SEED,
        verify_model: bool = True,
        allow_charged: bool = False,
        model_source=None,
    ):
        self.runtime = create_cpu_runtime(
            model_dir=model_dir,
            variant=variant,
            verify_model=verify_model,
            model_source=model_source,
        )
        self.batch_size = batch_size
        self.conformer_seed = conformer_seed
        self.max_atoms = int(self.runtime.manifest["max_atoms"])
        self.allow_charged = allow_charged
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
            raise RuntimeError("The MCA CPU predictor does not support concurrent predictions")
        try:
            yield from self._iter_predict(molecules)
        finally:
            self._predict_lock.release()

    def _iter_predict(self, molecules) -> Iterator[MoleculePrediction]:
        prepared = (
            value
            if isinstance(value, PreparedMolecule)
            else prepare_molecule(value, position, self.allow_charged)
            for position, value in enumerate(molecules)
        )
        yield from self._iter_predict_prepared(prepared)

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

        for prepared in prepared_molecules:
            molecule_index = next_input_index
            next_input_index += 1
            molecule = prepared.molecule
            features[molecule_index] = build_features(
                molecule,
                self.max_atoms,
                self.conformer_seed,
            )
            states[molecule_index] = {
                "molecule": molecule,
                "sites": prepared.sites,
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
        self.runtime.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
