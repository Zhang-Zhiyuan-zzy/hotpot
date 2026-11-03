#pragma once

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>


namespace hotpot::geometry::detail {


inline double nan_value() noexcept {
    return std::numeric_limits<double>::quiet_NaN();
}


inline Point3 subtract(const Point3& first, const Point3& second) noexcept {
    return {
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    };
}


inline Point3 add_scaled(
    const Point3& point,
    const Point3& direction,
    double scale
) noexcept {
    return {
        point[0] + scale * direction[0],
        point[1] + scale * direction[1],
        point[2] + scale * direction[2],
    };
}


inline double dot(const Point3& first, const Point3& second) noexcept {
    return (
        first[0] * second[0]
        + first[1] * second[1]
        + first[2] * second[2]
    );
}


inline Point3 cross(const Point3& first, const Point3& second) noexcept {
    return {
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    };
}


inline bool finite(const Point3& point) noexcept {
    return (
        std::isfinite(point[0])
        && std::isfinite(point[1])
        && std::isfinite(point[2])
    );
}


inline bool all_finite(ArrayView<Point3> points) noexcept {
    return std::all_of(points.begin(), points.end(), finite);
}


inline double scale_safe_norm(const Point3& vector) noexcept {
    const double maximum = std::max({
        std::abs(vector[0]),
        std::abs(vector[1]),
        std::abs(vector[2]),
    });
    if (maximum == 0.0) {
        return 0.0;
    }
    if (!std::isfinite(maximum)) {
        return nan_value();
    }
    const double first = vector[0] / maximum;
    const double second = vector[1] / maximum;
    const double third = vector[2] / maximum;
    return maximum * std::sqrt(
        first * first + second * second + third * third
    );
}


inline double norm(const Point3& vector) noexcept {
    return std::sqrt(dot(vector, vector));
}


inline std::optional<Point3> normalized(const Point3& direction) noexcept {
    if (!finite(direction)) {
        return std::nullopt;
    }
    const double length = scale_safe_norm(direction);
    if (length == 0.0) {
        return std::nullopt;
    }
    return Point3{
        direction[0] / length,
        direction[1] / length,
        direction[2] / length,
    };
}


inline double point_distance(
    const Point3& first,
    const Point3& second
) noexcept {
    if (!finite(first) || !finite(second)) {
        return nan_value();
    }
    return norm(subtract(first, second));
}


inline double diameter(ArrayView<Point3> points) noexcept {
    double result = 0.0;
    for (std::size_t first = 0; first < points.size(); ++first) {
        for (std::size_t second = first + 1; second < points.size(); ++second) {
            result = std::max(
                result,
                point_distance(points[first], points[second])
            );
        }
    }
    return result;
}


}  // namespace hotpot::geometry::detail
