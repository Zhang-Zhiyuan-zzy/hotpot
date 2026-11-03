#pragma once

#include "cycle_surface.hpp"

#include <array>
#include <utility>


namespace hotpot::geometry::detail {


double orient2d(
    const Point2& first,
    const Point2& second,
    const Point2& point
) noexcept;


double point_segment_distance_2d(
    const Point2& point,
    const Point2& start,
    const Point2& end
) noexcept;


double segment_segment_distance_2d(
    const Point2& first_start,
    const Point2& first_end,
    const Point2& second_start,
    const Point2& second_end,
    double squared_length_tolerance
) noexcept;


bool aabb_stably_separated_2d(
    const Point2& first_start,
    const Point2& first_end,
    const Point2& second_start,
    const Point2& second_end,
    double padding
) noexcept;


PolygonSimplicity projected_polygon_simplicity(
    ArrayView<Point2> polygon,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances
) noexcept;


std::pair<bool, bool> projected_segment_polygon_contact(
    const std::array<Point2, 2>& segment,
    ArrayView<Point2> polygon,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances
);


}  // namespace hotpot::geometry::detail
