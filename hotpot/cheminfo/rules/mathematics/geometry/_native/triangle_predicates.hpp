#pragma once

#include "nonplanar_surface.hpp"

#include <cstdint>
#include <optional>


namespace hotpot::geometry::detail {


enum class TriangleHitKind : std::uint8_t {
    STRICT_INTERIOR,
    TRIANGLE_BOUNDARY,
    SEGMENT_ENDPOINT,
    LINE_EXTENSION_INTERIOR,
    COPLANAR,
    SEPARATED,
    DEGENERATE,
    UNDETERMINED,
};


struct TriangleHit {
    TriangleHitKind kind;
    std::optional<Point3> point;
};


TriangleHit segment_triangle_relation(
    const Segment3& segment,
    const PreparedTriangleGeometry& triangle,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances
);


}  // namespace hotpot::geometry::detail
