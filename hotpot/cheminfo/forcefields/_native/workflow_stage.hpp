#pragma once

#include "coordination_stage.hpp"
#include "optimization_stage.hpp"
#include "../../obWrappers/_native/native_engine.hpp"

#include <utility>


namespace hotpot::forcefields {


class ComplexWorkflowSetupFailure final :
    public hotpot::obwrappers::ForceFieldSetupFailure {
public:
    ComplexWorkflowSetupFailure(
        const hotpot::obwrappers::ForceFieldSetupFailure& error,
        CoordinationStageResult completed_coordination
    ) :
        hotpot::obwrappers::ForceFieldSetupFailure(
            error.forcefield(), error.stage(), error.what()
        ),
        completed_coordination_(std::move(completed_coordination)) {}

    const CoordinationStageResult& completed_coordination() const noexcept {
        return completed_coordination_;
    }

private:
    CoordinationStageResult completed_coordination_;
};


ComplexWorkflowResult run_complex_workflow(
    StructureSession& session,
    const CoordinationStageOptions& coordination_options,
    const ComplexOptimizationOptions& optimization_options,
    const PerturbationOffsetBatch& coordination_offsets,
    const PerturbationOffsetBatch& untangling_offsets,
    const PerturbationOffsetBatch& optimization_offsets
);


ComplexWorkflowResult run_complex_workflow(
    const ComplexSessionInput& input,
    const CoordinationStageOptions& coordination_options,
    const ComplexOptimizationOptions& optimization_options,
    const PerturbationOffsetBatch& coordination_offsets,
    const PerturbationOffsetBatch& untangling_offsets,
    const PerturbationOffsetBatch& optimization_offsets
);


}  // namespace hotpot::forcefields
