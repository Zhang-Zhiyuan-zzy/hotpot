#pragma once

#include "contracts.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>


namespace hotpot::forcefields {
namespace detail {


struct PlacementDonorTarget {
    std::int32_t atom_index;
    std::int32_t group_index;
    double target_distance_angstrom;
    std::vector<std::int32_t> neighbour_indices;
};


struct MetalPlacementTarget {
    std::int32_t metal_index;
    std::vector<PlacementDonorTarget> donors;
};


std::vector<MetalPlacementTarget> select_metal_placement_targets(
    const ComplexSessionInput& input,
    const std::vector<std::int32_t>& component_ids,
    double coordination_distance_scale
);


MetalPlacementTarget select_metal_placement_target(
    const ComplexSessionInput& input,
    const std::vector<std::int32_t>& component_ids,
    std::int32_t metal_index,
    double coordination_distance_scale
);


}  // namespace detail
}  // namespace hotpot::forcefields
