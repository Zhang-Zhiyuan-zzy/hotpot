from hotpot.cheminfo.forcefields.native import (
    MetalPlacementOptions,
    RingScreeningOptions,
)
from hotpot.cheminfo.forcefields.settings import (
    _BOND_RING_MAX_SIZE,
    _MAXIMUM_RELEVANT_CYCLE_COUNT,
)
from hotpot.cheminfo.geometry.settings import DEFAULT_GEOMETRY_SETTINGS


def test_forcefield_ring_policy_defaults_are_shared() -> None:
    placement = MetalPlacementOptions()
    screening = RingScreeningOptions()

    assert placement.maximum_actionable_ring_size == _BOND_RING_MAX_SIZE == 16
    assert screening.maximum_actionable_ring_size == _BOND_RING_MAX_SIZE
    assert (
        screening.maximum_relevant_cycle_count
        == _MAXIMUM_RELEVANT_CYCLE_COUNT
        == 10000
    )
    assert DEFAULT_GEOMETRY_SETTINGS.surface.maximum_cycle_vertices == 8
