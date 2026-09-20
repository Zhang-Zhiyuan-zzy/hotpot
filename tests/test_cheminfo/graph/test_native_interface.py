"""Python/C++ interface checks before the cycle algorithm is introduced."""

import pytest

from hotpot.cheminfo.graph import relevant_cycles
from hotpot.cheminfo.graph import _relevant_cycles


def test_native_interface_round_trips_edges_and_optional_limits():
    assert _relevant_cycles._interface_probe(
        [(0, 1), (1, 2), (0, 2)],
        8,
        10_000,
    ) == (
        [[0, 1], [1, 2], [0, 2]],
        8,
        10_000,
    )
    assert _relevant_cycles._interface_probe([], None, None) == ([], None, None)


def test_public_interface_reaches_the_native_core_stub():
    with pytest.raises(
        NotImplementedError,
        match=(
            "native interface is connected.*"
            "edges=3, max_size=8, max_cycles=23"
        ),
    ):
        relevant_cycles(
            [(7, 9), (9, 4), (4, 7)],
            max_size=8,
            max_cycles=23,
        )
