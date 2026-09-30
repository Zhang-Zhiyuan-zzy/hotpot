#pragma once

#include "coordination_stage.hpp"
#include "optimization_stage.hpp"


namespace hotpot::forcefields {


ComplexWorkflowResult run_complex_workflow(
    StructureSession& session,
    const CoordinationStageOptions& coordination_options,
    const ComplexOptimizationOptions& optimization_options,
    const PerturbationOffsetBatch& coordination_offsets,
    const PerturbationOffsetBatch& untangling_offsets,
    const PerturbationOffsetBatch& optimization_offsets
);


}  // namespace hotpot::forcefields
