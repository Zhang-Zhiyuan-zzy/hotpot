#pragma once

#include "cycle_surface.hpp"
#include "primitives.hpp"
#include "types.hpp"

#include <cstdint>
#include <optional>
#include <vector>


namespace hotpot::geometry {


enum class PiercingState : std::uint8_t {
    PIERCES,
    DOES_NOT_PIERCE,
    UNDETERMINED,
};


enum class SegmentCycleFeature : std::uint8_t {
    TRANSVERSE_INTERIOR,
    LINE_EXTENSION_INTERIOR,
    CYCLE_EDGE_CONTACT,
    CYCLE_VERTEX_CONTACT,
    SEGMENT_ENDPOINT_CONTACT,
    COPLANAR_CONTACT,
};


enum class SegmentCycleIndeterminacy : std::uint8_t {
    NONFINITE_INPUT,
    NUMERIC_BAND,
    TOLERANCE_DOMAIN,
    DEGENERATE_CYCLE,
    DEGENERATE_SEGMENT,
    DEGENERATE_TRIANGLE,
    SELF_INTERSECTION,
    SURFACE_DISAGREEMENT,
    INCOMPLETE_SURFACE_FAMILY,
    SURFACE_CONSTRUCTION,
};


enum class CycleSurfaceModel : std::uint8_t {
    PLANAR_POLYGON,
    VERTEX_TRIANGULATION_FAMILY,
};


struct ClosestCycleEdge {
    std::size_t edge_index;
    double distance;
};


struct SurfaceFamilyEvidence {
    bool enumeration_complete;
    std::size_t enumerated_surface_count;
    std::size_t embedded_surface_count;
    std::size_t proven_non_embedded_surface_count;
    std::size_t construction_undetermined_count;
    std::size_t intersecting_surface_count;
    std::size_t non_piercing_surface_count;
    std::size_t evaluation_undetermined_count;
    std::size_t segment_triangle_tests_used;
    std::size_t triangle_pair_tests_used;
};


struct SegmentCycleRelation {
    PiercingState state;
    std::vector<SegmentCycleFeature> features;
    std::vector<SegmentCycleIndeterminacy> indeterminacy_causes;
    std::optional<CycleSurfaceModel> surface_model;
    std::vector<Point3> intersection_points;
    std::optional<ClosestCycleEdge> closest_boundary_edge;
    SurfaceFamilyEvidence surface_evidence;
};


struct SegmentCycleScreening {
    PiercingState state;
    std::optional<SegmentCycleRelation> relation;
    bool aabb_separated;
    bool surface_complete;
};


namespace detail {


struct SegmentCycleQuery {
    std::optional<PredicateTolerances> predicate_tolerances;
    std::optional<SegmentCycleIndeterminacy> cause;
};


SegmentCycleQuery prepare_segment_cycle_query(
    const Segment3& segment,
    ArrayView<Point3> cycle,
    const NumericTolerances& tolerances
);


std::optional<ClosestCycleEdge> closest_cycle_edge(
    ArrayView<Point3> cycle,
    const Segment3& segment,
    const NumericTolerances& tolerances
);


}  // namespace detail


std::optional<ClosestCycleEdge> closest_cycle_edge(
    const PreparedPlanarCycle& cycle,
    const Segment3& segment
);


SegmentCycleRelation determine_planar_segment_cycle_relation(
    const Segment3& segment,
    const PreparedPlanarCycle& cycle
);


SegmentCycleScreening screen_planar_segment_cycle(
    const Segment3& segment,
    const PreparedPlanarCycle& cycle,
    bool materialize_relation
);


std::vector<SegmentCycleRelation> planar_segment_cycle_relations(
    ArrayView<Segment3> segments,
    const PreparedPlanarCycle& cycle
);


std::vector<SegmentCycleScreening> planar_segment_cycle_screenings(
    ArrayView<Segment3> segments,
    const PreparedPlanarCycle& cycle
);


}  // namespace hotpot::geometry
