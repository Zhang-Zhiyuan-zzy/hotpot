"""Characterization fence for the native geometry migration."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from hotpot.cheminfo import geometry

from tests.geometry_characterization import (
    PUBLIC_COMPUTATIONAL_PARAMETERS,
    build_geometry_characterization,
)


EXPECTED_PUBLIC_API = (
    "NumericToleranceSettings",
    "SurfaceEnumerationSettings",
    "GeometrySettings",
    "DEFAULT_GEOMETRY_SETTINGS",
    "Point",
    "Line",
    "Segment",
    "Plane",
    "Triangle",
    "Cycle",
    "PlanarityKind",
    "LineRelationKind",
    "PointCycleLocation",
    "SurfaceEmbeddingState",
    "SurfaceSegmentState",
    "PiercingState",
    "SegmentCycleFeature",
    "SegmentCycleIndeterminacy",
    "CycleSurfaceModel",
    "PlanarityMeasurement",
    "LineRelation",
    "PointPairDistance",
    "ClosestCycleEdge",
    "SurfaceFamilyEvidence",
    "SegmentCycleRelation",
    "SegmentCycleScreening",
    "measure_planarity",
    "determine_line_relation",
    "line_distance",
    "point_segment_distance",
    "segment_segment_distance",
    "point_pair_distances",
    "find_point_pairs_below_distance",
    "locate_point_in_planar_cycle",
    "iter_segment_cycle_relations",
    "iter_segment_cycle_screenings",
    "determine_segment_cycle_relation",
    "closest_cycle_edge",
    "PairScope",
    "RingScope",
    "AtomGeometry",
    "AtomPairTarget",
    "BondGeometry",
    "RingGeometry",
    "BondRingTarget",
    "AtomPairDistance",
    "BondRingFinding",
    "RingEdgeDistance",
    "BondRingScanReport",
    "BondRingScreeningReport",
    "BondRingScreeningPlan",
    "BondRingFrameWorkspace",
    "point_from_atom",
    "segment_from_bond",
    "cycle_from_ring",
    "iter_atom_geometries",
    "iter_atom_pair_targets",
    "iter_ring_geometries",
    "iter_bond_ring_targets",
    "measure_atom_pair_distances",
    "determine_bond_ring_relation",
    "iter_bond_ring_findings",
    "scan_bond_ring_relations",
    "prepare_bond_ring_screening_plan",
    "prepare_bond_ring_frame",
    "screen_bond_ring_workspace",
    "screen_segments_against_ring_workspace",
    "screen_bonds_against_rings",
    "screen_bond_ring_relations",
    "determine_bond_ring_piercing_state",
)


def test_public_api_inventory_is_frozen_for_native_migration():
    assert tuple(geometry.__all__) == EXPECTED_PUBLIC_API
    assert len(geometry.__all__) == len(set(geometry.__all__))
    assert all(hasattr(geometry, name) for name in EXPECTED_PUBLIC_API)


def test_public_computational_parameter_contract_is_frozen():
    actual = {
        name: tuple(inspect.signature(getattr(geometry, name)).parameters)
        for name in PUBLIC_COMPUTATIONAL_PARAMETERS
    }

    assert actual == PUBLIC_COMPUTATIONAL_PARAMETERS


def test_geometry_results_match_pre_native_golden_corpus():
    golden_path = (
        Path(__file__).parents[2]
        / "fixtures"
        / "geometry"
        / "native_migration_golden.json"
    )
    expected = json.loads(golden_path.read_text(encoding="utf-8"))

    assert build_geometry_characterization() == expected

