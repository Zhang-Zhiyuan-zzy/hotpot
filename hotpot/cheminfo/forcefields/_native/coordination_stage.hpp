#pragma once

#include "contracts.hpp"
#include "stage_contracts.hpp"
#include "structure_session.hpp"


namespace hotpot::forcefields {


CoordinationStageResult restore_coordination(
    StructureSession& session,
    const CoordinationStageOptions& options,
    const PerturbationOffsetBatch& perturbation_offsets
);


}  // namespace hotpot::forcefields
