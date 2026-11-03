"""Provider-independent logical atom-row batching for MCA inference."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar


Prediction = TypeVar("Prediction")
EncodingKey = tuple[int, int]


@dataclass(frozen=True)
class AtomRow:
    molecule_index: int
    atom_index: int


@dataclass(frozen=True)
class LogicalBatch:
    """One frozen result-compatible flattened atom-row chunk."""

    ordinal: int
    rows: tuple[AtomRow, ...]
    pad_width: int
    unique_molecule_indices: tuple[int, ...]

    @property
    def encoding_keys(self) -> tuple[EncodingKey, ...]:
        return tuple(
            (molecule_index, self.pad_width)
            for molecule_index in self.unique_molecule_indices
        )


class LogicalBatchPlanner:
    """Freeze global chunks independently of worker and input-window timing."""

    def __init__(self, batch_size: int, *, first_molecule_index: int = 0):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self._next_molecule_index = first_molecule_index
        self._next_ordinal = 0
        self._reorder_buffer: dict[int, int] = {}
        self._pending_rows: list[tuple[AtomRow, int]] = []
        self._finished = False

    def push(
        self,
        molecules: Iterable[tuple[int, int]],
    ) -> tuple[LogicalBatch, ...]:
        """Accept an arbitrary completion window and emit newly full chunks."""

        if self._finished:
            raise RuntimeError("LogicalBatchPlanner is already finished")
        for molecule_index, atom_count in molecules:
            if molecule_index < self._next_molecule_index:
                raise ValueError(
                    f"Molecule index {molecule_index} was already planned"
                )
            if molecule_index in self._reorder_buffer:
                raise ValueError(f"Duplicate molecule index {molecule_index}")
            if atom_count < 0:
                raise ValueError("atom_count must not be negative")
            self._reorder_buffer[molecule_index] = atom_count

        completed = []
        while self._next_molecule_index in self._reorder_buffer:
            molecule_index = self._next_molecule_index
            atom_count = self._reorder_buffer.pop(molecule_index)
            self._next_molecule_index += 1
            for atom_index in range(atom_count):
                self._pending_rows.append(
                    (AtomRow(molecule_index, atom_index), atom_count)
                )
                if len(self._pending_rows) == self.batch_size:
                    completed.append(self._freeze_pending())
        return tuple(completed)

    def finish(self) -> tuple[LogicalBatch, ...]:
        """Flush the final partial chunk after all indexed inputs arrived."""

        if self._finished:
            return ()
        if self._reorder_buffer:
            first_buffered = min(self._reorder_buffer)
            raise ValueError(
                f"Cannot finish logical batches: molecule index "
                f"{self._next_molecule_index} is missing before {first_buffered}"
            )
        self._finished = True
        return (self._freeze_pending(),) if self._pending_rows else ()

    def _freeze_pending(self) -> LogicalBatch:
        rows_with_counts = tuple(self._pending_rows)
        self._pending_rows.clear()
        rows = tuple(row for row, _ in rows_with_counts)
        pad_width = max(atom_count for _, atom_count in rows_with_counts)
        unique_molecule_indices = tuple(
            dict.fromkeys(row.molecule_index for row in rows)
        )
        batch = LogicalBatch(
            ordinal=self._next_ordinal,
            rows=rows,
            pad_width=pad_width,
            unique_molecule_indices=unique_molecule_indices,
        )
        self._next_ordinal += 1
        return batch
def scatter_batch_predictions(
    batch: LogicalBatch,
    molecule_predictions: Mapping[EncodingKey, Sequence[Prediction]],
) -> tuple[Prediction, ...]:
    """Restore the flattened atom-row order from per-molecule all-atom outputs."""

    return tuple(
        molecule_predictions[(row.molecule_index, batch.pad_width)][row.atom_index]
        for row in batch.rows
    )
