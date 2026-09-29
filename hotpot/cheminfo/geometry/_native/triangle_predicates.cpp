#include "triangle_predicates.hpp"

#include "vector_math.hpp"

#include <algorithm>
#include <array>
#include <cmath>


namespace hotpot::geometry::detail {
namespace {


std::array<double, 3> barycentric_coordinates(
    const Point3& point,
    const PreparedTriangleGeometry& triangle
) noexcept {
    const Point3& first = triangle.coordinates[0];
    const Point3& second = triangle.coordinates[1];
    const Point3& third = triangle.coordinates[2];
    const double squared_normal = dot(triangle.normal, triangle.normal);
    const double first_weight = dot(
        cross(subtract(second, point), subtract(third, point)),
        triangle.normal
    ) / squared_normal;
    const double second_weight = dot(
        cross(subtract(third, point), subtract(first, point)),
        triangle.normal
    ) / squared_normal;
    return {
        first_weight,
        second_weight,
        1.0 - first_weight - second_weight,
    };
}


TriangleHitKind classify_barycentric(
    const std::array<double, 3>& barycentric,
    double parameter_tolerance,
    double guard
) noexcept {
    if (std::all_of(
            barycentric.begin(),
            barycentric.end(),
            [guard, parameter_tolerance](double value) {
                return value > guard * parameter_tolerance;
            }
        )) {
        return TriangleHitKind::STRICT_INTERIOR;
    }
    if (
        std::all_of(
            barycentric.begin(),
            barycentric.end(),
            [parameter_tolerance](double value) {
                return value >= -parameter_tolerance;
            }
        )
        && std::any_of(
            barycentric.begin(),
            barycentric.end(),
            [parameter_tolerance](double value) {
                return std::abs(value) <= parameter_tolerance;
            }
        )
    ) {
        return TriangleHitKind::TRIANGLE_BOUNDARY;
    }
    if (std::any_of(
            barycentric.begin(),
            barycentric.end(),
            [guard, parameter_tolerance](double value) {
                return value < -guard * parameter_tolerance;
            }
        )) {
        return TriangleHitKind::SEPARATED;
    }
    return TriangleHitKind::UNDETERMINED;
}


}  // namespace


TriangleHit segment_triangle_relation(
    const Segment3& segment,
    const PreparedTriangleGeometry& triangle,
    const PredicateTolerances& predicate_tolerances,
    const NumericTolerances& tolerances
) {
    if (
        !finite(segment.start)
        || !finite(segment.end)
        || !all_finite(ArrayView<Point3>(
            triangle.coordinates.data(), triangle.coordinates.size()
        ))
    ) {
        return {TriangleHitKind::UNDETERMINED, std::nullopt};
    }

    const double guard = tolerances.predicate_guard_factor;
    const double segment_length = norm(subtract(segment.end, segment.start));
    if (segment_length <= predicate_tolerances.length) {
        return {TriangleHitKind::DEGENERATE, std::nullopt};
    }
    if (segment_length <= guard * predicate_tolerances.length) {
        return {TriangleHitKind::UNDETERMINED, std::nullopt};
    }
    if (triangle.normal_length <= predicate_tolerances.area) {
        return {TriangleHitKind::DEGENERATE, std::nullopt};
    }
    if (triangle.normal_length <= guard * predicate_tolerances.area) {
        return {TriangleHitKind::UNDETERMINED, std::nullopt};
    }

    Point3 unit_normal = triangle.normal;
    for (double& value : unit_normal) {
        value /= triangle.normal_length;
    }
    const Point3& first = triangle.coordinates[0];
    const double start_height = dot(
        unit_normal, subtract(segment.start, first)
    );
    const double end_height = dot(unit_normal, subtract(segment.end, first));
    const double start_absolute = std::abs(start_height);
    const double end_absolute = std::abs(end_height);
    if (
        start_absolute <= predicate_tolerances.length
        && end_absolute <= predicate_tolerances.length
    ) {
        return {TriangleHitKind::COPLANAR, std::nullopt};
    }
    if (
        (
            predicate_tolerances.length < start_absolute
            && start_absolute <= guard * predicate_tolerances.length
        )
        || (
            predicate_tolerances.length < end_absolute
            && end_absolute <= guard * predicate_tolerances.length
        )
    ) {
        return {TriangleHitKind::UNDETERMINED, std::nullopt};
    }

    const bool one_endpoint = (
        (
            start_absolute <= predicate_tolerances.length
            && end_absolute > guard * predicate_tolerances.length
        )
        || (
            end_absolute <= predicate_tolerances.length
            && start_absolute > guard * predicate_tolerances.length
        )
    );
    const double height_difference = start_height - end_height;
    if (one_endpoint) {
        const Point3 point = start_absolute <= predicate_tolerances.length
            ? segment.start
            : segment.end;
        const TriangleHitKind location = classify_barycentric(
            barycentric_coordinates(point, triangle),
            predicate_tolerances.parameter,
            guard
        );
        if (
            location == TriangleHitKind::STRICT_INTERIOR
            || location == TriangleHitKind::TRIANGLE_BOUNDARY
        ) {
            return {TriangleHitKind::SEGMENT_ENDPOINT, point};
        }
        return {location, point};
    }

    if (std::abs(height_difference) <= predicate_tolerances.length) {
        return {TriangleHitKind::SEPARATED, std::nullopt};
    }
    if (
        std::abs(height_difference)
        <= guard * predicate_tolerances.length
    ) {
        return {TriangleHitKind::UNDETERMINED, std::nullopt};
    }
    if (guard * predicate_tolerances.parameter >= 0.5) {
        return {TriangleHitKind::UNDETERMINED, std::nullopt};
    }

    const double parameter = start_height / height_difference;
    const Point3 direction = subtract(segment.end, segment.start);
    const Point3 point = {
        segment.start[0] + parameter * direction[0],
        segment.start[1] + parameter * direction[1],
        segment.start[2] + parameter * direction[2],
    };
    const TriangleHitKind location = classify_barycentric(
        barycentric_coordinates(point, triangle),
        predicate_tolerances.parameter,
        guard
    );
    if (
        location == TriangleHitKind::TRIANGLE_BOUNDARY
        || location == TriangleHitKind::UNDETERMINED
        || location == TriangleHitKind::SEPARATED
    ) {
        return {location, point};
    }

    const double parameter_tolerance = predicate_tolerances.parameter;
    if (
        guard * parameter_tolerance < parameter
        && parameter < 1.0 - guard * parameter_tolerance
    ) {
        return {TriangleHitKind::STRICT_INTERIOR, point};
    }
    if (
        std::abs(parameter) <= parameter_tolerance
        || std::abs(1.0 - parameter) <= parameter_tolerance
    ) {
        return {TriangleHitKind::SEGMENT_ENDPOINT, point};
    }
    if (
        parameter < -guard * parameter_tolerance
        || parameter > 1.0 + guard * parameter_tolerance
    ) {
        return {TriangleHitKind::LINE_EXTENSION_INTERIOR, point};
    }
    return {TriangleHitKind::UNDETERMINED, point};
}


}  // namespace hotpot::geometry::detail
