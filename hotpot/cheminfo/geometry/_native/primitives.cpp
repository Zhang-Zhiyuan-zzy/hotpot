#include "primitives.hpp"
#include "vector_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>


namespace hotpot::geometry {
namespace {

using detail::add_scaled;
using detail::cross;
using detail::diameter;
using detail::dot;
using detail::finite;
using detail::nan_value;
using detail::normalized;
using detail::point_distance;
using detail::scale_safe_norm;
using detail::subtract;


double clamp_parameter(double value) noexcept {
    return std::isnan(value) ? 0.0 : std::clamp(value, 0.0, 1.0);
}


PointSegmentMeasurement point_segment_measurement_unchecked(
    const Point3& point,
    const Segment3& segment,
    double length_tolerance
) noexcept {
    const Point3 direction = subtract(segment.end, segment.start);
    const double segment_length = std::sqrt(dot(direction, direction));
    if (segment_length <= length_tolerance) {
        return {
            point_distance(point, segment.start),
            segment.start,
            0.0,
            true,
        };
    }

    const double squared_length = dot(direction, direction);
    const double parameter = clamp_parameter(
        dot(subtract(point, segment.start), direction) / squared_length
    );
    const Point3 closest = add_scaled(segment.start, direction, parameter);
    return {
        point_distance(point, closest),
        closest,
        parameter,
        false,
    };
}


}  // namespace


LineRelation determine_line_relation(
    const Line3& first,
    const Line3& second,
    const NumericTolerances& tolerances
) {
    if (
        !finite(first.origin)
        || !finite(first.direction)
        || !finite(second.origin)
        || !finite(second.direction)
    ) {
        return {
            LineRelationKind::UNDETERMINED,
            std::nullopt,
            nan_value(),
        };
    }

    const std::optional<Point3> first_direction = normalized(first.direction);
    const std::optional<Point3> second_direction = normalized(second.direction);
    if (!first_direction.has_value() || !second_direction.has_value()) {
        return {
            LineRelationKind::DEGENERATE,
            std::nullopt,
            nan_value(),
        };
    }

    const Point3 direction_cross = cross(*first_direction, *second_direction);
    const double parallel_measure = scale_safe_norm(direction_cross);
    if (!std::isfinite(parallel_measure)) {
        return {
            LineRelationKind::UNDETERMINED,
            std::nullopt,
            nan_value(),
        };
    }

    const double angular_tolerance = std::max(
        tolerances.parameter,
        tolerances.machine_epsilon_factor
            * std::numeric_limits<double>::epsilon()
    );
    const double guard = tolerances.predicate_guard_factor;
    const Point3 separation = subtract(second.origin, first.origin);
    if (parallel_measure == 0.0) {
        const Point3 perpendicular = cross(separation, *first_direction);
        const double distance = std::sqrt(dot(perpendicular, perpendicular));
        const double length_tolerance = (
            tolerances.absolute_length
            + tolerances.effective_relative_length() * distance
        );
        if (distance <= length_tolerance) {
            return {
                LineRelationKind::COINCIDENT,
                distance,
                parallel_measure,
            };
        }
        if (distance > guard * length_tolerance) {
            return {
                LineRelationKind::PARALLEL,
                distance,
                parallel_measure,
            };
        }
        return {
            LineRelationKind::UNDETERMINED,
            std::nullopt,
            parallel_measure,
        };
    }
    if (parallel_measure <= guard * angular_tolerance) {
        return {
            LineRelationKind::UNDETERMINED,
            std::nullopt,
            parallel_measure,
        };
    }

    const double distance = (
        std::abs(dot(separation, direction_cross)) / parallel_measure
    );
    const double length_tolerance = (
        tolerances.absolute_length
        + tolerances.effective_relative_length() * distance
    );
    if (distance <= length_tolerance) {
        return {
            LineRelationKind::INTERSECTING,
            distance,
            parallel_measure,
        };
    }
    if (distance > guard * length_tolerance) {
        return {
            LineRelationKind::SKEW,
            distance,
            parallel_measure,
        };
    }
    return {
        LineRelationKind::UNDETERMINED,
        std::nullopt,
        parallel_measure,
    };
}


double line_distance(
    const Line3& first,
    const Line3& second,
    const NumericTolerances& tolerances
) {
    const LineRelation relation = determine_line_relation(
        first,
        second,
        tolerances
    );
    return relation.distance.value_or(nan_value());
}


PointSegmentMeasurement point_segment_measurement(
    const Point3& point,
    const Segment3& segment,
    const NumericTolerances& tolerances
) {
    if (!finite(point) || !finite(segment.start) || !finite(segment.end)) {
        const Point3 nan_point = {nan_value(), nan_value(), nan_value()};
        return {nan_value(), nan_point, nan_value(), false};
    }

    const std::array<Point3, 3> points = {point, segment.start, segment.end};
    const double length_scale = diameter(
        ArrayView<Point3>(points.data(), points.size())
    );
    const double length_tolerance = (
        length_scale > 0.0
        ? tolerances.effective_length(length_scale)
        : tolerances.absolute_length
    );
    return point_segment_measurement_unchecked(
        point,
        segment,
        length_tolerance
    );
}


double point_segment_distance(
    const Point3& point,
    const Segment3& segment,
    const NumericTolerances& tolerances
) {
    return point_segment_measurement(point, segment, tolerances).distance;
}


SegmentSegmentMeasurement segment_segment_measurement(
    const Segment3& first,
    const Segment3& second,
    const NumericTolerances& tolerances
) {
    if (
        !finite(first.start)
        || !finite(first.end)
        || !finite(second.start)
        || !finite(second.end)
    ) {
        const Point3 nan_point = {nan_value(), nan_value(), nan_value()};
        return {
            nan_value(),
            nan_point,
            nan_point,
            nan_value(),
            nan_value(),
            false,
            false,
        };
    }

    const std::array<Point3, 4> points = {
        first.start,
        first.end,
        second.start,
        second.end,
    };
    const double length_scale = diameter(
        ArrayView<Point3>(points.data(), points.size())
    );
    const double length_tolerance = (
        length_scale > 0.0
        ? tolerances.effective_length(length_scale)
        : tolerances.absolute_length
    );
    const double squared_length_tolerance = length_tolerance * length_tolerance;

    const Point3 first_direction = subtract(first.end, first.start);
    const Point3 second_direction = subtract(second.end, second.start);
    const Point3 offset = subtract(first.start, second.start);
    const double first_squared = dot(first_direction, first_direction);
    const double second_squared = dot(second_direction, second_direction);
    const bool first_degenerate = first_squared <= squared_length_tolerance;
    const bool second_degenerate = second_squared <= squared_length_tolerance;

    if (first_degenerate) {
        const PointSegmentMeasurement measurement =
            point_segment_measurement_unchecked(
                first.start,
                second,
                0.0
            );
        return {
            measurement.distance,
            first.start,
            measurement.closest_point,
            0.0,
            measurement.parameter,
            true,
            second_degenerate,
        };
    }
    if (second_degenerate) {
        const PointSegmentMeasurement measurement =
            point_segment_measurement_unchecked(
                second.start,
                first,
                0.0
            );
        return {
            measurement.distance,
            measurement.closest_point,
            second.start,
            measurement.parameter,
            0.0,
            false,
            true,
        };
    }

    const double direction_dot = dot(first_direction, second_direction);
    const double first_offset = dot(first_direction, offset);
    const double second_offset = dot(second_direction, offset);
    const double denominator = (
        first_squared * second_squared - direction_dot * direction_dot
    );
    double first_parameter = 0.0;
    if (denominator != 0.0) {
        first_parameter = clamp_parameter(
            (
                direction_dot * second_offset
                - first_offset * second_squared
            ) / denominator
        );
    }
    double second_parameter = (
        direction_dot * first_parameter + second_offset
    ) / second_squared;
    if (second_parameter < 0.0) {
        second_parameter = 0.0;
        first_parameter = clamp_parameter(-first_offset / first_squared);
    } else if (second_parameter > 1.0) {
        second_parameter = 1.0;
        first_parameter = clamp_parameter(
            (direction_dot - first_offset) / first_squared
        );
    }

    const Point3 first_closest = add_scaled(
        first.start,
        first_direction,
        first_parameter
    );
    const Point3 second_closest = add_scaled(
        second.start,
        second_direction,
        second_parameter
    );
    return {
        point_distance(first_closest, second_closest),
        first_closest,
        second_closest,
        first_parameter,
        second_parameter,
        false,
        false,
    };
}


double segment_segment_distance(
    const Segment3& first,
    const Segment3& second,
    const NumericTolerances& tolerances
) {
    return segment_segment_measurement(first, second, tolerances).distance;
}


std::vector<PointPairDistance> point_pair_distances(
    ArrayView<Point3> points
) {
    std::vector<PointPairDistance> results;
    results.reserve(points.size() * (points.size() - (points.size() > 0)) / 2);
    for (std::size_t first = 0; first < points.size(); ++first) {
        for (std::size_t second = first + 1; second < points.size(); ++second) {
            results.push_back({
                first,
                second,
                point_distance(points[first], points[second]),
            });
        }
    }
    return results;
}


std::vector<PointPairDistance> point_pair_distances(
    ArrayView<Point3> points,
    ArrayView<IndexPair> pairs
) {
    std::vector<PointPairDistance> results;
    results.reserve(pairs.size());
    for (const IndexPair& pair : pairs) {
        results.push_back({
            pair[0],
            pair[1],
            point_distance(points[pair[0]], points[pair[1]]),
        });
    }
    return results;
}


std::vector<PointPairDistance> find_point_pairs_below_distance(
    ArrayView<Point3> points,
    double threshold
) {
    std::vector<PointPairDistance> results;
    for (std::size_t first = 0; first < points.size(); ++first) {
        for (std::size_t second = first + 1; second < points.size(); ++second) {
            const double distance = point_distance(points[first], points[second]);
            if (distance < threshold) {
                results.push_back({first, second, distance});
            }
        }
    }
    return results;
}


}  // namespace hotpot::geometry
