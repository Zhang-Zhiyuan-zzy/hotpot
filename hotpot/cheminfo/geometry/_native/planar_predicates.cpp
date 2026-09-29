#include "planar_predicates.hpp"

#include <algorithm>
#include <cmath>
#include <limits>


namespace hotpot::geometry {
namespace {


constexpr double pi = 3.141592653589793238462643383279502884;


}  // namespace


namespace detail {


double orient2d(
    const Point2& first,
    const Point2& second,
    const Point2& point
) noexcept {
    return (
        (second[0] - first[0]) * (point[1] - first[1])
        - (second[1] - first[1]) * (point[0] - first[0])
    );
}


double point_segment_distance_2d(
    const Point2& point,
    const Point2& start,
    const Point2& end
) noexcept {
    const Point2 direction = {end[0] - start[0], end[1] - start[1]};
    const double squared_length = (
        direction[0] * direction[0] + direction[1] * direction[1]
    );
    if (squared_length == 0.0) {
        return std::hypot(point[0] - start[0], point[1] - start[1]);
    }
    const double parameter = std::clamp(
        (
            (point[0] - start[0]) * direction[0]
            + (point[1] - start[1]) * direction[1]
        ) / squared_length,
        0.0,
        1.0
    );
    return std::hypot(
        point[0] - (start[0] + parameter * direction[0]),
        point[1] - (start[1] + parameter * direction[1])
    );
}


double segment_segment_distance_2d(
    const Point2& first_start,
    const Point2& first_end,
    const Point2& second_start,
    const Point2& second_end,
    double squared_length_tolerance
) noexcept {
    const Point2 first_direction = {
        first_end[0] - first_start[0],
        first_end[1] - first_start[1],
    };
    const Point2 second_direction = {
        second_end[0] - second_start[0],
        second_end[1] - second_start[1],
    };
    const Point2 offset = {
        first_start[0] - second_start[0],
        first_start[1] - second_start[1],
    };
    const double first_squared = (
        first_direction[0] * first_direction[0]
        + first_direction[1] * first_direction[1]
    );
    const double second_squared = (
        second_direction[0] * second_direction[0]
        + second_direction[1] * second_direction[1]
    );
    if (first_squared <= squared_length_tolerance) {
        return point_segment_distance_2d(
            first_start,
            second_start,
            second_end
        );
    }
    if (second_squared <= squared_length_tolerance) {
        return point_segment_distance_2d(
            second_start,
            first_start,
            first_end
        );
    }
    const double direction_dot = (
        first_direction[0] * second_direction[0]
        + first_direction[1] * second_direction[1]
    );
    const double first_offset = (
        first_direction[0] * offset[0] + first_direction[1] * offset[1]
    );
    const double second_offset = (
        second_direction[0] * offset[0] + second_direction[1] * offset[1]
    );
    const double denominator = (
        first_squared * second_squared - direction_dot * direction_dot
    );
    double first_parameter = 0.0;
    if (denominator != 0.0) {
        first_parameter = std::clamp(
            (
                direction_dot * second_offset
                - first_offset * second_squared
            ) / denominator,
            0.0,
            1.0
        );
    }
    double second_parameter = (
        direction_dot * first_parameter + second_offset
    ) / second_squared;
    if (second_parameter < 0.0) {
        second_parameter = 0.0;
        first_parameter = std::clamp(
            -first_offset / first_squared,
            0.0,
            1.0
        );
    } else if (second_parameter > 1.0) {
        second_parameter = 1.0;
        first_parameter = std::clamp(
            (direction_dot - first_offset) / first_squared,
            0.0,
            1.0
        );
    }
    const Point2 first_closest = {
        first_start[0] + first_parameter * first_direction[0],
        first_start[1] + first_parameter * first_direction[1],
    };
    const Point2 second_closest = {
        second_start[0] + second_parameter * second_direction[0],
        second_start[1] + second_parameter * second_direction[1],
    };
    return std::hypot(
        first_closest[0] - second_closest[0],
        first_closest[1] - second_closest[1]
    );
}


bool aabb_stably_separated_2d(
    const Point2& first_start,
    const Point2& first_end,
    const Point2& second_start,
    const Point2& second_end,
    double padding
) noexcept {
    for (std::size_t axis = 0; axis < 2; ++axis) {
        const double first_minimum = std::min(first_start[axis], first_end[axis]);
        const double first_maximum = std::max(first_start[axis], first_end[axis]);
        const double second_minimum = std::min(second_start[axis], second_end[axis]);
        const double second_maximum = std::max(second_start[axis], second_end[axis]);
        if (
            first_maximum + padding < second_minimum
            || second_maximum + padding < first_minimum
        ) {
            return true;
        }
    }
    return false;
}


PolygonSimplicity projected_polygon_simplicity(
    ArrayView<Point2> polygon,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances
) noexcept {
    const std::size_t count = polygon.size();
    const double guard = tolerances.predicate_guard_factor;
    for (std::size_t first_index = 0; first_index < count; ++first_index) {
        const std::size_t first_next = (first_index + 1) % count;
        for (
            std::size_t second_index = first_index + 1;
            second_index < count;
            ++second_index
        ) {
            const std::size_t second_next = (second_index + 1) % count;
            if (
                first_index == second_index
                || first_next == second_index
                || second_next == first_index
            ) {
                continue;
            }
            if (aabb_stably_separated_2d(
                    polygon[first_index],
                    polygon[first_next],
                    polygon[second_index],
                    polygon[second_next],
                    predicate_tolerances.aabb
                )) {
                continue;
            }
            const std::array<double, 4> orientations = {
                orient2d(
                    polygon[first_index],
                    polygon[first_next],
                    polygon[second_index]
                ),
                orient2d(
                    polygon[first_index],
                    polygon[first_next],
                    polygon[second_next]
                ),
                orient2d(
                    polygon[second_index],
                    polygon[second_next],
                    polygon[first_index]
                ),
                orient2d(
                    polygon[second_index],
                    polygon[second_next],
                    polygon[first_next]
                ),
            };
            if (std::any_of(
                    orientations.begin(),
                    orientations.end(),
                    [guard, &predicate_tolerances](double value) {
                        return std::abs(value) <= guard * predicate_tolerances.area;
                    }
                )) {
                return PolygonSimplicity::UNDETERMINED;
            }
            if (
                std::signbit(orientations[0]) != std::signbit(orientations[1])
                && std::signbit(orientations[2]) != std::signbit(orientations[3])
            ) {
                return PolygonSimplicity::SELF_INTERSECTING;
            }
        }
    }
    return PolygonSimplicity::SIMPLE;
}


std::pair<bool, bool> projected_segment_polygon_contact(
    const std::array<Point2, 2>& segment,
    ArrayView<Point2> polygon,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances
) {
    const PointCycleLocation start_location = locate_projected_point(
        segment[0],
        polygon,
        predicate_tolerances,
        tolerances
    );
    const PointCycleLocation end_location = locate_projected_point(
        segment[1],
        polygon,
        predicate_tolerances,
        tolerances
    );
    if (
        start_location == PointCycleLocation::INTERIOR
        || start_location == PointCycleLocation::BOUNDARY
        || end_location == PointCycleLocation::INTERIOR
        || end_location == PointCycleLocation::BOUNDARY
    ) {
        return {true, false};
    }
    if (
        start_location == PointCycleLocation::UNDETERMINED
        || end_location == PointCycleLocation::UNDETERMINED
    ) {
        return {false, true};
    }

    double minimum_distance = std::numeric_limits<double>::infinity();
    const double squared_tolerance = (
        predicate_tolerances.length * predicate_tolerances.length
    );
    for (std::size_t index = 0; index < polygon.size(); ++index) {
        minimum_distance = std::min(
            minimum_distance,
            segment_segment_distance_2d(
                segment[0],
                segment[1],
                polygon[index],
                polygon[(index + 1) % polygon.size()],
                squared_tolerance
            )
        );
    }
    if (minimum_distance <= predicate_tolerances.length) {
        return {true, false};
    }
    if (
        minimum_distance
        <= tolerances.predicate_guard_factor * predicate_tolerances.length
    ) {
        return {false, true};
    }
    return {false, false};
}


}  // namespace detail


PointCycleLocation locate_projected_point(
    const Point2& point,
    ArrayView<Point2> polygon,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances
) {
    double boundary_distance = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < polygon.size(); ++index) {
        boundary_distance = std::min(
            boundary_distance,
            detail::point_segment_distance_2d(
                point,
                polygon[index],
                polygon[(index + 1) % polygon.size()]
            )
        );
    }
    const double guard = tolerances.predicate_guard_factor;
    if (boundary_distance <= predicate_tolerances.length) {
        return PointCycleLocation::BOUNDARY;
    }
    if (boundary_distance <= guard * predicate_tolerances.length) {
        return PointCycleLocation::UNDETERMINED;
    }

    double angle_sum = 0.0;
    for (std::size_t index = 0; index < polygon.size(); ++index) {
        const Point2 first = {
            polygon[index][0] - point[0],
            polygon[index][1] - point[1],
        };
        const Point2 second = {
            polygon[(index + 1) % polygon.size()][0] - point[0],
            polygon[(index + 1) % polygon.size()][1] - point[1],
        };
        angle_sum += std::atan2(
            first[0] * second[1] - first[1] * second[0],
            first[0] * second[0] + first[1] * second[1]
        );
    }
    const double winding = angle_sum / (2.0 * pi);
    if (std::abs(std::abs(winding) - 1.0) <= tolerances.winding_residual) {
        return PointCycleLocation::INTERIOR;
    }
    if (std::abs(winding) <= tolerances.winding_residual) {
        return PointCycleLocation::EXTERIOR;
    }
    return PointCycleLocation::UNDETERMINED;
}


}  // namespace hotpot::geometry
