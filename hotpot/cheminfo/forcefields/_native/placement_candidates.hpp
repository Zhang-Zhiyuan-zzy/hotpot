#pragma once

#include "placement_evidence.hpp"
#include "placement_policy.hpp"

#include <vector>


namespace hotpot::forcefields {
namespace detail {


std::vector<PlacementProposal> generate_metal_placement_candidates(
    const PlacementEvaluationWorkspace& workspace,
    const MetalPlacementOptions& options
);


}  // namespace detail
}  // namespace hotpot::forcefields
