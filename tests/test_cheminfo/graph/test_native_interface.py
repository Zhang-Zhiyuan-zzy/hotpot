"""Direct Python/C++ Relevant Cycle interface checks."""

from hotpot.cheminfo.graph import _relevant_cycles


def test_native_interface_exchanges_edges_limits_and_cycle_edge_ids():
    assert _relevant_cycles.relevant_cycles(
        [(0, 1), (0, 2), (1, 2)],
        8,
        10_000,
    ) == [[0, 1, 2]]
    assert _relevant_cycles.relevant_cycles([], None, None) == []
