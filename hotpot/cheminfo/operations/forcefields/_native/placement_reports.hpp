#pragma once

#include "../../../rules/physics/forcefields/_native/placement_evidence.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


namespace hotpot::forcefields {


struct MetalPlacementResult {
    std::int32_t metal_index;
    PlacementStatus status;
    Coordinate original_coordinates;
    Coordinate selected_coordinates;
    bool moved;
    std::size_t candidates_evaluated;
    PlacementCandidateEvidence selected_evidence;
    std::vector<PlacementCandidateEvidence> retained_candidates;
    std::size_t excluded_large_cycle_count;
    std::vector<std::string> warning_codes;
};


struct MetalPlacementReport {
    std::vector<MetalPlacementResult> metals;
    std::vector<Coordinate> selected_coordinates;
    std::vector<std::string> warning_codes;
};


}  // namespace hotpot::forcefields
