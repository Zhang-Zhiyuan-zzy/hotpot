#pragma once

#include "placement_reports.hpp"
#include "../../../rules/physics/forcefields/_native/placement_policy.hpp"
#include "structure_session.hpp"

#include <cstdint>


namespace hotpot::forcefields {


PlacementCandidateEvidence assess_metal_position(
    const StructureSession& session,
    std::int32_t metal_index,
    const Coordinate& candidate,
    const MetalPlacementOptions& options = {}
);


MetalPlacementResult place_metal(
    StructureSession& session,
    std::int32_t metal_index,
    const MetalPlacementOptions& options = {}
);


MetalPlacementReport place_metals(
    StructureSession& session,
    const MetalPlacementOptions& options = {}
);


}  // namespace hotpot::forcefields
