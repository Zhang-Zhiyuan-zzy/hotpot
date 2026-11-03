#pragma once

#include "stage_contracts.hpp"
#include "structure_session.hpp"

namespace hotpot::forcefields {


void validate_complex_optimization_request(
    const StructureSession& session,
    const ComplexOptimizationOptions& options,
    const PerturbationOffsetBatch& untangling_offsets,
    const PerturbationOffsetBatch& optimization_offsets
);


ComplexOptimizationResult optimize_complex(
    StructureSession& session,
    const ComplexOptimizationOptions& options,
    const PerturbationOffsetBatch& untangling_offsets,
    const PerturbationOffsetBatch& optimization_offsets
);
}  // namespace hotpot::forcefields
