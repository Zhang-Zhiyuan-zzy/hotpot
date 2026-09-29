#include "trajectory.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>


namespace hotpot::forcefields {


namespace {


void validate_binary_mask(
    const std::vector<std::uint8_t>& mask,
    std::size_t expected_size,
    const char* name
) {
    if (mask.size() != expected_size) {
        throw std::invalid_argument(
            std::string(name) + " has an unexpected size"
        );
    }
    if (std::any_of(mask.begin(), mask.end(), [](std::uint8_t value) {
            return value > 1;
        })) {
        throw std::invalid_argument(
            std::string(name) + " must contain only zero or one"
        );
    }
}


void validate_frame_index(
    std::int64_t frame_index,
    std::size_t frame_count,
    const char* name
) {
    if (frame_index < -1
        || (frame_index >= 0
            && static_cast<std::size_t>(frame_index) >= frame_count)) {
        throw std::invalid_argument(
            std::string(name) + " is outside the trajectory"
        );
    }
}


}  // namespace


std::size_t NativeTrajectoryBatch::frame_count() const noexcept {
    return frames.size();
}


std::size_t NativeTrajectoryBatch::append_topology_revision(
    NativeTopologyRevision revision
) {
    validate_binary_mask(
        revision.active_ligand_bond_mask,
        ligand_bond_count,
        "active_ligand_bond_mask"
    );
    validate_binary_mask(
        revision.active_coordination_bond_mask,
        intended_coordination_bond_count,
        "active_coordination_bond_mask"
    );
    const auto matching = std::find_if(
        topology_revisions.begin(),
        topology_revisions.end(),
        [&revision](const NativeTopologyRevision& existing) {
            return existing.active_ligand_bond_mask
                    == revision.active_ligand_bond_mask
                && existing.active_coordination_bond_mask
                    == revision.active_coordination_bond_mask;
        }
    );
    if (matching != topology_revisions.end()) {
        return static_cast<std::size_t>(
            std::distance(topology_revisions.begin(), matching)
        );
    }
    topology_revisions.push_back(std::move(revision));
    return topology_revisions.size() - 1;
}


void NativeTrajectoryBatch::append(NativeTrajectoryFrame frame) {
    if (frame.coordinates.size() != atom_count) {
        throw std::invalid_argument(
            "trajectory frame coordinate count must equal atom_count"
        );
    }
    if (frame.topology_revision < 0
        || static_cast<std::size_t>(frame.topology_revision)
            >= topology_revisions.size()) {
        throw std::invalid_argument(
            "trajectory frame topology revision is unavailable"
        );
    }
    frames.push_back(std::move(frame));
}


void NativeTrajectoryBatch::select(std::size_t frame_index) {
    if (frame_index >= frames.size()) {
        throw std::out_of_range("selected frame index is outside the trajectory");
    }
    selected_frame_index = static_cast<std::int64_t>(frame_index);
}


void NativeTrajectoryBatch::set_terminal(std::size_t frame_index) {
    if (frame_index >= frames.size()) {
        throw std::out_of_range("terminal frame index is outside the trajectory");
    }
    terminal_frame_index = static_cast<std::int64_t>(frame_index);
}


void NativeTrajectoryBatch::validate() const {
    NativeTrajectoryBatch validated{
        atom_count,
        ligand_bond_count,
        intended_coordination_bond_count,
        start,
        {},
        {},
        -1,
        -1,
    };
    for (const auto& revision : topology_revisions) {
        validated.append_topology_revision(revision);
    }
    if (validated.topology_revisions.size() != topology_revisions.size()) {
        throw std::invalid_argument(
            "trajectory topology revisions must be unique"
        );
    }
    for (const auto& frame : frames) {
        validated.append(frame);
    }
    validate_frame_index(
        selected_frame_index, frames.size(), "selected_frame_index"
    );
    validate_frame_index(
        terminal_frame_index, frames.size(), "terminal_frame_index"
    );
}


}  // namespace hotpot::forcefields
