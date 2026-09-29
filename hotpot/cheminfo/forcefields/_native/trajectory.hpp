#pragma once

#include "contracts.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>


namespace hotpot::forcefields {


// This controls optional diagnostics only. Required factual frames remain.
enum class FrameDetail : std::uint8_t {
    NONE = 0,
    OPTIMIZATION = 1,
    ALL_ATTEMPTS = 2,
};


enum class NativeTrajectoryStart : std::int32_t {
    LIGAND_BUILD = 0,
    COORDINATION_RESTORATION = 1,
    COMPLEX_UNTANGLING = 2,
    FINAL_OPTIMIZATION = 3,
};


enum class NativeTrajectoryStage : std::int32_t {
    LIGAND_BUILD = 0,
    COORDINATION_RESTORATION = 1,
    COMPLEX_UNTANGLING = 2,
    FINAL_OPTIMIZATION = 3,
};


enum class NativeTrajectoryEvent : std::int32_t {
    INITIAL = 0,
    BUILD_COMPLETE = 1,
    WARMUP_COMPLETE = 2,
    COORDINATION_READY = 3,
    BOND_TRIAL = 4,
    BOND_ACCEPTED = 5,
    BOND_REJECTED = 6,
    BOND_ROLLBACK = 7,
    BOND_FORCED = 8,
    METAL_RELOCATION_TRIAL = 9,
    METAL_RELOCATED = 10,
    METAL_RELOCATION_FAILED = 11,
    TOPOLOGY_CHECKPOINT = 12,
    RING_OPENED = 13,
    PERTURBED = 14,
    OPTIMIZED = 15,
    RING_CLOSED = 16,
    SETTLED = 17,
    ROLLED_BACK = 18,
    EPOCH_COMPLETE = 19,
    TERMINAL = 20,
};


struct NativeRingFrameEvidence {
    std::size_t confirmed_piercing_count = 0;
    std::optional<std::size_t> uncertain_relation_count;
    std::optional<std::string> ring_scope;
    std::optional<std::size_t> max_ring_size;
    std::optional<std::size_t> selected_ring_count;
    std::optional<std::size_t> excluded_ring_count;
    std::optional<std::size_t> candidate_pair_count;
    std::optional<std::size_t> aabb_separated_pair_count;
    std::optional<std::size_t> exact_pair_count;
    std::optional<std::size_t> does_not_pierce_pair_count;
    std::optional<bool> scan_complete;
};


struct NativeCoordinationFrameEvidence {
    std::optional<BondIndex> bond_atom_indices;
    std::optional<bool> accepted;
    std::size_t pending_bond_count = 0;
    bool forced = false;
    std::size_t piercing_relation_count = 0;
    std::size_t undetermined_relation_count = 0;
    std::size_t excluded_ring_count = 0;
    std::optional<std::int32_t> metal_atom_index;
    std::optional<std::string> relocation_status;
    std::size_t relocation_candidates_evaluated = 0;
    std::vector<std::int32_t> safe_donor_atom_indices;
    std::optional<double> minimum_normalized_clearance;
    std::optional<double> coordination_distance_deviation;
};


struct NativeOptimizationFrameEvidence {
    bool converged = false;
    bool exploded = false;
    bool finite_coordinates = true;
    bool finite_energy = true;
    bool finite_gradients = true;
    std::optional<double> rms_gradient_kj_mol_angstrom;
    std::optional<double> max_gradient_kj_mol_angstrom;
    std::optional<double> energy_change_kj_mol;
    std::optional<double> max_displacement_angstrom;
};


using NativeFrameEvidence = std::variant<
    std::monostate,
    NativeRingFrameEvidence,
    NativeCoordinationFrameEvidence,
    NativeOptimizationFrameEvidence
>;


struct NativeTopologyRevision {
    std::vector<std::uint8_t> active_ligand_bond_mask;
    std::vector<std::uint8_t> active_coordination_bond_mask;
};


struct NativeTrajectoryFrame {
    std::vector<Coordinate> coordinates;
    NativeTrajectoryStage stage;
    NativeTrajectoryEvent event;
    std::optional<std::int32_t> component_index;
    std::optional<std::int32_t> attempt;
    std::optional<std::int64_t> step;
    std::optional<double> energy_kj_mol;
    NativeFrameEvidence evidence;
    std::int32_t topology_revision;
};


struct NativeTrajectoryBatch {
    std::size_t atom_count = 0;
    std::size_t ligand_bond_count = 0;
    std::size_t intended_coordination_bond_count = 0;
    NativeTrajectoryStart start = NativeTrajectoryStart::COORDINATION_RESTORATION;
    std::vector<NativeTopologyRevision> topology_revisions;
    std::vector<NativeTrajectoryFrame> frames;
    std::int64_t selected_frame_index = -1;
    std::int64_t terminal_frame_index = -1;

    std::size_t frame_count() const noexcept;
    std::size_t append_topology_revision(NativeTopologyRevision revision);
    void append(NativeTrajectoryFrame frame);
    void select(std::size_t frame_index);
    void set_terminal(std::size_t frame_index);
    void validate() const;
};


}  // namespace hotpot::forcefields
