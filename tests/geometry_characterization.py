"""Deterministic geometry corpus shared by migration tests and profiles."""

from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from typing import Dict, Iterable, List, Optional, Tuple, Union

from hotpot.cheminfo import geometry as geo


JSONScalar = Union[None, bool, int, float, str]
JSONValue = Union[JSONScalar, List["JSONValue"], Dict[str, "JSONValue"]]
SerializedNumber = Union[float, str]


PUBLIC_COMPUTATIONAL_PARAMETERS: Dict[str, Tuple[str, ...]] = {
    "measure_planarity": ("cycle", "settings"),
    "determine_line_relation": ("first", "second", "settings"),
    "line_distance": ("first", "second", "settings"),
    "point_segment_distance": ("point", "segment", "settings"),
    "segment_segment_distance": ("first", "second", "settings"),
    "point_pair_distances": ("points",),
    "find_point_pairs_below_distance": ("points", "threshold"),
    "locate_point_in_planar_cycle": ("point", "cycle", "plane", "settings"),
    "iter_segment_cycle_relations": ("segments", "cycle", "settings"),
    "iter_segment_cycle_screenings": ("segments", "cycle", "settings"),
    "determine_segment_cycle_relation": ("segment", "cycle", "settings"),
    "closest_cycle_edge": ("cycle", "segment", "settings"),
    "point_from_atom": ("atom",),
    "segment_from_bond": ("bond",),
    "cycle_from_ring": ("ring",),
    "iter_atom_geometries": ("structure",),
    "iter_atom_pair_targets": ("structure", "pair_scope"),
    "iter_ring_geometries": ("mol", "ring_scope", "max_ring_size"),
    "iter_bond_ring_targets": ("mol", "ring_scope", "max_ring_size"),
    "measure_atom_pair_distances": ("structure", "pair_scope"),
    "determine_bond_ring_relation": ("ring", "bond", "settings"),
    "iter_bond_ring_findings": (
        "mol",
        "ring_scope",
        "max_ring_size",
        "settings",
    ),
    "scan_bond_ring_relations": (
        "mol",
        "ring_scope",
        "max_ring_size",
        "settings",
    ),
    "prepare_bond_ring_screening_plan": (
        "mol",
        "ring_scope",
        "max_ring_size",
        "bonds",
        "settings",
    ),
    "prepare_bond_ring_frame": ("plan",),
    "screen_bond_ring_workspace": (
        "workspace",
        "bond_keys",
        "ring_keys",
        "stop_after_confirmed",
    ),
    "screen_segments_against_ring_workspace": (
        "segments",
        "workspace",
        "bond_keys",
        "ring_keys",
        "stop_after_confirmed",
    ),
    "screen_bonds_against_rings": (
        "mol",
        "bonds",
        "ring_scope",
        "max_ring_size",
        "settings",
    ),
    "screen_bond_ring_relations": (
        "mol",
        "ring_scope",
        "max_ring_size",
        "settings",
    ),
    "determine_bond_ring_piercing_state": (
        "mol",
        "ring_scope",
        "max_ring_size",
        "settings",
    ),
}


def _number(value: float) -> SerializedNumber:
    numeric = float(value)
    if isfinite(numeric):
        return round(numeric, 12)
    if numeric != numeric:
        return "nan"
    return "inf" if numeric > 0.0 else "-inf"


def _coordinates(point: geo.Point) -> List[SerializedNumber]:
    return [_number(value) for value in point.coordinates]


def _planarity(measurement: geo.PlanarityMeasurement) -> Dict[str, JSONValue]:
    return {
        "kind": measurement.kind.value,
        "centroid": _coordinates(measurement.centroid),
        "normal": (
            None
            if measurement.normal is None
            else [_number(value) for value in measurement.normal]
        ),
        "singular_values": [
            _number(value) for value in measurement.singular_values
        ],
        "maximum_deviation": _number(measurement.maximum_deviation),
        "rms_deviation": _number(measurement.rms_deviation),
        "length_scale": _number(measurement.length_scale),
        "length_tolerance": _number(measurement.length_tolerance),
    }


def _line_relation(relation: geo.LineRelation) -> Dict[str, JSONValue]:
    return {
        "kind": relation.kind.value,
        "distance": (
            None if relation.distance is None else _number(relation.distance)
        ),
        "parallel_measure": _number(relation.parallel_measure),
    }


def _closest_edge(
    closest: Optional[geo.ClosestCycleEdge],
) -> Optional[Dict[str, JSONValue]]:
    if closest is None:
        return None
    return {
        "edge_index": closest.edge_index,
        "edge": [
            _coordinates(closest.edge.start),
            _coordinates(closest.edge.end),
        ],
        "distance": _number(closest.distance),
    }


def _relation(relation: geo.SegmentCycleRelation) -> Dict[str, JSONValue]:
    evidence = asdict(relation.surface_evidence)
    return {
        "state": relation.state.value,
        "features": sorted(feature.value for feature in relation.features),
        "indeterminacy_causes": sorted(
            cause.value for cause in relation.indeterminacy_causes
        ),
        "surface_model": (
            None if relation.surface_model is None else relation.surface_model.value
        ),
        "intersection_points": [
            _coordinates(point) for point in relation.intersection_points
        ],
        "closest_boundary_edge": _closest_edge(
            relation.closest_boundary_edge
        ),
        "surface_evidence": evidence,
    }


def _screenings(
    screenings: Iterable[geo.SegmentCycleScreening],
) -> List[Dict[str, JSONValue]]:
    return [
        {
            "state": screening.state.value,
            "aabb_separated": screening.aabb_separated,
            "surface_complete": screening.surface_complete,
            "relation_materialized": screening.relation is not None,
        }
        for screening in screenings
    ]


def build_geometry_characterization() -> Dict[str, JSONValue]:
    """Evaluate public geometry operations on a fixed migration corpus."""

    square = geo.Cycle(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
         (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    warped_square = geo.Cycle(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
         (2.0, 2.0, 0.4), (0.0, 2.0, 0.0))
    )
    bow_tie = geo.Cycle(
        ((0.0, 0.0, 0.0), (2.0, 2.0, 0.0),
         (0.0, 2.0, 0.0), (2.0, 0.0, 0.0))
    )
    collinear = geo.Cycle(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    )
    piercing = geo.Segment((1.0, 1.0, -1.0), (1.0, 1.0, 1.0))
    extension = geo.Segment((1.0, 1.0, 1.0), (1.0, 1.0, 2.0))
    edge_contact = geo.Segment((0.0, 1.0, -1.0), (0.0, 1.0, 1.0))
    nonplanar_piercing = geo.Segment(
        (0.6, 0.8, -1.0), (0.6, 0.8, 1.0)
    )
    nonplanar_uncertain = geo.Segment(
        (0.8, 0.8, -1.0), (0.8, 0.8, 1.0)
    )
    far_segment = geo.Segment((10.0, 10.0, 4.0), (11.0, 10.0, 4.0))
    points = (
        geo.Point((0.0, 0.0, 0.0)),
        geo.Point((1.0, 0.0, 0.0)),
        geo.Point((3.0, 0.0, 0.0)),
    )

    point_pair_results = geo.point_pair_distances(points)
    below_results = geo.find_point_pairs_below_distance(points, 3.0)

    return {
        "schema_version": 1,
        "coordinate_unit": "angstrom",
        "planarity": {
            "planar_square": _planarity(geo.measure_planarity(square)),
            "warped_square": _planarity(geo.measure_planarity(warped_square)),
            "collinear_cycle": _planarity(geo.measure_planarity(collinear)),
        },
        "line_relations": {
            "intersecting": _line_relation(geo.determine_line_relation(
                geo.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                geo.Line((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
            )),
            "parallel": _line_relation(geo.determine_line_relation(
                geo.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                geo.Line((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
            )),
            "coincident": _line_relation(geo.determine_line_relation(
                geo.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                geo.Line((3.0, 0.0, 0.0), (-4.0, 0.0, 0.0)),
            )),
            "skew": _line_relation(geo.determine_line_relation(
                geo.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                geo.Line((0.0, 1.0, 1.0), (0.0, 1.0, 0.0)),
            )),
        },
        "distances": {
            "line": _number(geo.line_distance(
                geo.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                geo.Line((0.0, 1.0, 1.0), (0.0, 1.0, 0.0)),
            )),
            "point_segment": _number(geo.point_segment_distance(
                geo.Point((1.0, 1.0, 0.0)),
                geo.Segment((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            )),
            "segment_segment": _number(geo.segment_segment_distance(
                geo.Segment((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                geo.Segment((0.5, -1.0, 0.0), (0.5, 1.0, 0.0)),
            )),
            "point_pairs": [
                [item.first_index, item.second_index, _number(item.distance)]
                for item in point_pair_results
            ],
            "below_strict_threshold": [
                [item.first_index, item.second_index, _number(item.distance)]
                for item in below_results
            ],
        },
        "point_cycle_locations": {
            label: geo.locate_point_in_planar_cycle(
                geo.Point(point),
                square,
                geo.Plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ).value
            for label, point in (
                ("interior", (1.0, 1.0, 0.0)),
                ("boundary", (2.0, 1.0, 0.0)),
                ("exterior", (3.0, 1.0, 0.0)),
            )
        },
        "segment_cycle_relations": {
            "planar_piercing": _relation(
                geo.determine_segment_cycle_relation(piercing, square)
            ),
            "line_extension": _relation(
                geo.determine_segment_cycle_relation(extension, square)
            ),
            "edge_contact": _relation(
                geo.determine_segment_cycle_relation(edge_contact, square)
            ),
            "nonplanar_piercing": _relation(
                geo.determine_segment_cycle_relation(
                    nonplanar_piercing, warped_square
                )
            ),
            "nonplanar_uncertain": _relation(
                geo.determine_segment_cycle_relation(
                    nonplanar_uncertain, warped_square
                )
            ),
            "self_intersecting_cycle": _relation(
                geo.determine_segment_cycle_relation(
                    geo.Segment((0.5, 1.0, -1.0), (0.5, 1.0, 1.0)),
                    bow_tie,
                )
            ),
        },
        "batch": {
            "relation_states": [
                relation.state.value
                for relation in geo.iter_segment_cycle_relations(
                    (piercing, extension, far_segment), square
                )
            ],
            "screenings": _screenings(geo.iter_segment_cycle_screenings(
                (piercing, extension, far_segment), square
            )),
        },
        "closest_edge_tie": _closest_edge(
            geo.closest_cycle_edge(square, piercing)
        ),
    }
