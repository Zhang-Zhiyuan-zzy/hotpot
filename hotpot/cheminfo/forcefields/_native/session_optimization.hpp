#pragma once

#include "../../obWrappers/_native/native_engine.hpp"
#include "structure_session.hpp"

#include <cstddef>
#include <string>


namespace hotpot::forcefields {


hotpot::obwrappers::SingleOptimizationResult single_optimize_session(
    StructureSession& session,
    const std::string& forcefield,
    std::size_t steps,
    double singularity_threshold,
    double repair_angle_radians
);


hotpot::obwrappers::OptimizationResult optimize_session(
    StructureSession& session,
    const hotpot::obwrappers::OptimizationOptions& options,
    const PerturbationOffsetBatch& perturbation_offsets,
    double singularity_threshold,
    double repair_angle_radians
);


}  // namespace hotpot::forcefields
