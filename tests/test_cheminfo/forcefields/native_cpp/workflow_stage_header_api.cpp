#include "workflow_stage.hpp"

#include <type_traits>


namespace ff = hotpot::forcefields;


using SessionWorkflowEntry = ff::ComplexWorkflowResult (*)(
    ff::StructureSession&,
    const ff::CoordinationStageOptions&,
    const ff::ComplexOptimizationOptions&,
    const ff::PerturbationOffsetBatch&,
    const ff::PerturbationOffsetBatch&,
    const ff::PerturbationOffsetBatch&
);

using InputWorkflowEntry = ff::ComplexWorkflowResult (*)(
    const ff::ComplexSessionInput&,
    const ff::CoordinationStageOptions&,
    const ff::ComplexOptimizationOptions&,
    const ff::PerturbationOffsetBatch&,
    const ff::PerturbationOffsetBatch&,
    const ff::PerturbationOffsetBatch&
);


static_assert(std::is_same_v<
    decltype(static_cast<SessionWorkflowEntry>(&ff::run_complex_workflow)),
    SessionWorkflowEntry
>);
static_assert(std::is_same_v<
    decltype(static_cast<InputWorkflowEntry>(&ff::run_complex_workflow)),
    InputWorkflowEntry
>);
