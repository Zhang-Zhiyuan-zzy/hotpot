#include "spatial.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>


namespace hotpot::geometry {


Aabb aabb_bounds(ArrayView<Point3> points) {
    if (points.empty()) {
        throw std::invalid_argument(
            "an AABB requires at least one three-dimensional point"
        );
    }
    Aabb bounds{points[0], points[0]};
    for (std::size_t point_index = 1; point_index < points.size(); ++point_index) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (std::isnan(points[point_index][axis])) {
                bounds.minimum[axis] = std::numeric_limits<double>::quiet_NaN();
                bounds.maximum[axis] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            if (std::isnan(bounds.minimum[axis])) {
                continue;
            }
            bounds.minimum[axis] = std::min(
                bounds.minimum[axis],
                points[point_index][axis]
            );
            bounds.maximum[axis] = std::max(
                bounds.maximum[axis],
                points[point_index][axis]
            );
        }
    }
    return bounds;
}


bool aabb_stably_separated(
    const Aabb& first,
    const Aabb& second,
    double padding
) noexcept {
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (
            first.maximum[axis] + padding < second.minimum[axis]
            || second.maximum[axis] + padding < first.minimum[axis]
        ) {
            return true;
        }
    }
    return false;
}


std::vector<std::uint8_t> aabb_separation_mask(
    ArrayView<Aabb> first,
    ArrayView<Aabb> second,
    ArrayView<double> paddings
) {
    if (first.size() != second.size() || first.size() != paddings.size()) {
        throw std::invalid_argument(
            "paired AABB batches and paddings must have equal lengths"
        );
    }
    std::vector<std::uint8_t> separated(first.size());
    for (std::size_t index = 0; index < first.size(); ++index) {
        separated[index] = static_cast<std::uint8_t>(
            aabb_stably_separated(first[index], second[index], paddings[index])
        );
    }
    return separated;
}


std::vector<std::uint8_t> segment_aabb_separation_mask(
    ArrayView<Segment3> segments,
    const Aabb& target,
    ArrayView<double> paddings
) {
    if (segments.size() != paddings.size()) {
        throw std::invalid_argument(
            "the segment batch and paddings must have equal lengths"
        );
    }
    std::vector<std::uint8_t> separated(segments.size());
    for (std::size_t index = 0; index < segments.size(); ++index) {
        const std::array<Point3, 2> endpoints = {
            segments[index].start,
            segments[index].end,
        };
        separated[index] = static_cast<std::uint8_t>(
            aabb_stably_separated(
                aabb_bounds(ArrayView<Point3>(endpoints.data(), endpoints.size())),
                target,
                paddings[index]
            )
        );
    }
    return separated;
}


std::vector<IndexPair> aabb_candidate_pairs(
    ArrayView<Aabb> first,
    ArrayView<Aabb> second,
    double padding
) {
    std::vector<IndexPair> candidates;
    for (std::size_t first_index = 0; first_index < first.size(); ++first_index) {
        for (
            std::size_t second_index = 0;
            second_index < second.size();
            ++second_index
        ) {
            if (!aabb_stably_separated(
                    first[first_index],
                    second[second_index],
                    padding
                )) {
                candidates.push_back({first_index, second_index});
            }
        }
    }
    return candidates;
}


}  // namespace hotpot::geometry
