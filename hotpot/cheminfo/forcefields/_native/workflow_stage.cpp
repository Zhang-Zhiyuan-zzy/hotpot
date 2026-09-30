#include "workflow_stage.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>


namespace hotpot::forcefields {
namespace {


void append_unique(
    std::vector<std::string>& values,
    const std::string& value
) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}


NativeTrajectoryStart earliest_start(
    NativeTrajectoryStart first,
    NativeTrajectoryStart second
) noexcept {
    return static_cast<std::int32_t>(first)
            <= static_cast<std::int32_t>(second)
        ? first
        : second;
}


struct AppendedTrajectoryIndices {
    std::int64_t selected = -1;
    std::int64_t terminal = -1;
};


AppendedTrajectoryIndices append_trajectory(
    NativeTrajectoryBatch& destination,
    const NativeTrajectoryBatch& source
) {
    std::vector<std::size_t> revision_mapping;
    revision_mapping.reserve(source.topology_revisions.size());
    for (const auto& revision : source.topology_revisions) {
        revision_mapping.push_back(
            destination.append_topology_revision(revision)
        );
    }

    const auto frame_offset = destination.frame_count();
    for (auto frame : source.frames) {
        frame.topology_revision = static_cast<std::int32_t>(
            revision_mapping[static_cast<std::size_t>(
                frame.topology_revision
            )]
        );
        destination.append(std::move(frame));
    }
    return {
        source.selected_frame_index < 0
            ? -1
            : static_cast<std::int64_t>(frame_offset)
                + source.selected_frame_index,
        source.terminal_frame_index < 0
            ? -1
            : static_cast<std::int64_t>(frame_offset)
                + source.terminal_frame_index,
    };
}


NativeTrajectoryBatch combine_trajectories(
    const CoordinationStageResult& coordination,
    const ComplexOptimizationResult& optimization
) {
    NativeTrajectoryBatch combined{
        optimization.atom_count(),
        optimization.trajectory.ligand_bond_count,
        optimization.trajectory.intended_coordination_bond_count,
        earliest_start(
            coordination.trajectory.start,
            optimization.trajectory.start
        ),
        {},
        {},
        -1,
        -1,
    };
    append_trajectory(combined, coordination.trajectory);
    const auto optimization_indices = append_trajectory(
        combined, optimization.trajectory
    );
    combined.selected_frame_index = optimization_indices.selected;
    combined.terminal_frame_index = optimization_indices.terminal;
    combined.validate();
    return combined;
}


}  // namespace


ComplexWorkflowResult run_complex_workflow(
    StructureSession& session,
    const CoordinationStageOptions& coordination_options,
    const ComplexOptimizationOptions& optimization_options,
    const PerturbationOffsetBatch& coordination_offsets,
    const PerturbationOffsetBatch& untangling_offsets,
    const PerturbationOffsetBatch& optimization_offsets
) {
    validate_coordination_request(
        session, coordination_options, coordination_offsets
    );
    validate_complex_optimization_request(
        session,
        optimization_options,
        untangling_offsets,
        optimization_offsets
    );

    auto coordination = restore_coordination(
        session, coordination_options, coordination_offsets
    );
    auto optimization = optimize_complex(
        session,
        optimization_options,
        untangling_offsets,
        optimization_offsets
    );
    auto trajectory = combine_trajectories(coordination, optimization);

    std::vector<std::string> warning_codes;
    for (const auto& warning : coordination.warning_codes) {
        append_unique(warning_codes, warning);
    }
    for (const auto& warning : optimization.warning_codes) {
        append_unique(warning_codes, warning);
    }

    ComplexWorkflowResult result{
        std::move(coordination),
        std::move(optimization),
        {},
        {},
        {},
        std::move(warning_codes),
        std::move(trajectory),
    };
    result.selected_coordinates = result.optimization.selected_coordinates;
    result.terminal_coordinates = result.optimization.terminal_coordinates;
    result.final_active_coordination_mask =
        result.optimization.final_active_coordination_mask;
    result.validate();
    return result;
}


}  // namespace hotpot::forcefields
