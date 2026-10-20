from mca.batching import LogicalBatchPlanner


def plan_logical_batches(atom_counts, batch_size):
    planner = LogicalBatchPlanner(batch_size)
    batches = list(
        planner.push(
            (molecule_index, atom_count)
            for molecule_index, atom_count in enumerate(atom_counts)
        )
    )
    batches.extend(planner.finish())
    return tuple(batches)
