#pragma once

#include "contracts.hpp"
#include "stage_contracts.hpp"
#include "structure_session.hpp"


namespace hotpot::forcefields {


void validate_coordination_request(
    const StructureSession& session,
    const CoordinationStageOptions& options,
    const PerturbationOffsetBatch& perturbation_offsets
);


CoordinationStageResult restore_coordination(
    StructureSession& session,
    const CoordinationStageOptions& options,
    const PerturbationOffsetBatch& perturbation_offsets
);


}  // namespace hotpot::forcefields
