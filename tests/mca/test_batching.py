import pytest

from mca.batching import (
    AtomRow,
    LogicalBatchPlanner,
    scatter_batch_predictions,
)
from tests.mca._helpers import plan_logical_batches


def _legacy_chunks(atom_counts, batch_size):
    rows = [
        (molecule_index, atom_index)
        for molecule_index, atom_count in enumerate(atom_counts)
        for atom_index in range(atom_count)
    ]
    chunks = []
    for ordinal, start in enumerate(range(0, len(rows), batch_size)):
        chunk_rows = rows[start : start + batch_size]
        chunks.append(
            (
                ordinal,
                chunk_rows,
                max(atom_counts[index] for index, _ in chunk_rows),
                tuple(dict.fromkeys(index for index, _ in chunk_rows)),
            )
        )
    return chunks


def test_materialized_planner_exactly_matches_legacy_flat_atom_row_chunks():
    atom_counts = [2, 7, 1, 5]
    actual = plan_logical_batches(atom_counts, batch_size=4)

    assert [
        (
            batch.ordinal,
            [(row.molecule_index, row.atom_index) for row in batch.rows],
            batch.pad_width,
            batch.unique_molecule_indices,
        )
        for batch in actual
    ] == _legacy_chunks(atom_counts, 4)


def test_window_boundaries_do_not_change_global_chunks_or_ordinals():
    expected = plan_logical_batches([3, 4, 5, 2], batch_size=5)
    planner = LogicalBatchPlanner(batch_size=5)
    actual = []
    actual.extend(planner.push(((0, 3),)))
    actual.extend(planner.push(((1, 4), (2, 5))))
    actual.extend(planner.push(((3, 2),)))
    actual.extend(planner.finish())

    assert tuple(actual) == expected


def test_out_of_order_worker_completion_is_reordered_before_batch_freeze():
    expected = plan_logical_batches([2, 6, 3, 4], batch_size=5)
    planner = LogicalBatchPlanner(batch_size=5)

    assert planner.push(((2, 3),)) == ()
    first = planner.push(((0, 2),))
    assert first == ()
    middle = planner.push(((3, 4), (1, 6)))
    tail = planner.finish()

    assert middle + tail == expected


def test_one_molecule_can_span_batches_and_reuse_the_same_padding_key():
    batches = plan_logical_batches([2, 8, 3], batch_size=4)

    assert batches[0].rows == (
        AtomRow(0, 0),
        AtomRow(0, 1),
        AtomRow(1, 0),
        AtomRow(1, 1),
    )
    assert batches[0].pad_width == 8
    assert batches[0].encoding_keys == ((0, 8), (1, 8))
    assert batches[1].encoding_keys == ((1, 8),)
    assert batches[2].pad_width == 8
    assert batches[2].encoding_keys == ((1, 8), (2, 8))
    assert batches[3].pad_width == 3
    assert batches[3].encoding_keys == ((2, 3),)


def test_same_molecule_is_keyed_separately_when_chunk_pad_width_changes():
    batches = plan_logical_batches([9, 5], batch_size=6)

    assert batches[0].encoding_keys == ((0, 9),)
    assert batches[1].encoding_keys == ((0, 9), (1, 9))
    assert batches[2].encoding_keys == ((1, 5),)


def test_scatter_uses_molecule_and_padding_width_cache_key():
    batch = plan_logical_batches([2, 3], batch_size=4)[0]
    predictions = {
        (0, 3): (10.0, 11.0),
        (1, 3): (20.0, 21.0, 22.0),
    }
    assert scatter_batch_predictions(batch, predictions) == (
        10.0,
        11.0,
        20.0,
        21.0,
    )


def test_finish_rejects_a_gap_and_is_idempotent_after_success():
    planner = LogicalBatchPlanner(batch_size=4)
    planner.push(((1, 2),))
    with pytest.raises(ValueError, match="molecule index 0 is missing"):
        planner.finish()

    complete = LogicalBatchPlanner(batch_size=4)
    complete.push(((0, 1),))
    assert len(complete.finish()) == 1
    assert complete.finish() == ()
