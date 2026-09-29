#include "session_optimization.hpp"

#include "../../obWrappers/_native/openbabel_adapter.hpp"
#include "structure_session_internal.hpp"

#include <mutex>
#include <stdexcept>


namespace hotpot::forcefields {


namespace {


void record_session_coordinate_change(
    StructureSession& session,
    const std::vector<Coordinate>& before,
    const std::vector<Coordinate>& after
) {
    if (before != after) {
        StructureSessionAccess::record_coordinate_change(session);
    }
}


}  // namespace


hotpot::obwrappers::SingleOptimizationResult single_optimize_session(
    StructureSession& session,
    const std::string& forcefield,
    std::size_t steps,
    double singularity_threshold,
    double repair_angle_radians
) {
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    auto& molecule = StructureSessionAccess::obmol(session);
    const auto before = hotpot::obwrappers::extract_coordinates(molecule);
    auto result = hotpot::obwrappers::single_optimize_in_place(
        molecule,
        forcefield,
        steps,
        singularity_threshold,
        repair_angle_radians
    );
    record_session_coordinate_change(session, before, result.coordinates);
    return result;
}


hotpot::obwrappers::OptimizationResult optimize_session(
    StructureSession& session,
    const hotpot::obwrappers::OptimizationOptions& options,
    const PerturbationOffsetBatch& perturbation_offsets,
    double singularity_threshold,
    double repair_angle_radians
) {
    perturbation_offsets.validate();
    if (perturbation_offsets.atom_count != session.atom_count()) {
        throw std::invalid_argument(
            "perturbation atom_count must match the structure session"
        );
    }
    std::lock_guard<std::recursive_mutex> lock(
        hotpot::obwrappers::openbabel_runtime_mutex()
    );
    auto& molecule = StructureSessionAccess::obmol(session);
    const auto before = hotpot::obwrappers::extract_coordinates(molecule);
    auto result = hotpot::obwrappers::optimize_in_place(
        molecule,
        options,
        perturbation_offsets.frames,
        singularity_threshold,
        repair_angle_radians
    );
    record_session_coordinate_change(session, before, result.coordinates);
    return result;
}


}  // namespace hotpot::forcefields
