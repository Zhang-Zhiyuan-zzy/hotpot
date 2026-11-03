#pragma once

#include "types.hpp"

#include <cstdint>
#include <vector>


namespace hotpot::geometry {


Aabb aabb_bounds(ArrayView<Point3> points);


bool aabb_stably_separated(
    const Aabb& first,
    const Aabb& second,
    double padding
) noexcept;


std::vector<std::uint8_t> aabb_separation_mask(
    ArrayView<Aabb> first,
    ArrayView<Aabb> second,
    ArrayView<double> paddings
);


std::vector<std::uint8_t> segment_aabb_separation_mask(
    ArrayView<Segment3> segments,
    const Aabb& target,
    ArrayView<double> paddings
);


std::vector<IndexPair> aabb_candidate_pairs(
    ArrayView<Aabb> first,
    ArrayView<Aabb> second,
    double padding
);


}  // namespace hotpot::geometry
