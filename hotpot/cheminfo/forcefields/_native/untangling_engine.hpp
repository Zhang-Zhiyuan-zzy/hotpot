#pragma once

#include "session_optimization.hpp"
#include "topology_workspace.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>


namespace hotpot::forcefields {


struct RingUntanglingOptions {
    std::string forcefield = "UFF";
    std::size_t short_optimization_steps = 100;
    double torsion_singularity_threshold = 1.0e-6;
    double torsion_repair_angle_radians = 1.0e-3;
    detail::RingWorkspaceOptions ring_workspace;

    void validate() const;
};


struct RingUntanglingAttemptCursor {
    std::size_t attempt_limit = 30;
    std::size_t attempts_completed = 0;

    std::size_t remaining() const noexcept;
    bool exhausted() const noexcept;
    void validate() const;
};


enum class RingUntanglingEvent : std::uint8_t {
    TOPOLOGY_CHECKPOINT = 0,
    RING_OPENED = 1,
    PERTURBED = 2,
    OPEN_TOPOLOGY_OPTIMIZED = 3,
    RING_CLOSED = 4,
    ROLLED_BACK = 5,
};


struct RingUntanglingStep {
    RingUntanglingEvent event = RingUntanglingEvent::TOPOLOGY_CHECKPOINT;
    std::size_t attempt = 0;
    StructureSnapshot snapshot;
    std::optional<BondIndex> opening_bond_key;
    std::optional<double> energy_kj_mol;
    std::optional<hotpot::geometry::PiercingState> observed_state;
    std::optional<std::size_t> confirmed_piercing_count;
};


struct RingUntanglingResult {
    bool resolved = true;
    std::size_t attempts_used = 0;
    std::size_t initial_piercing_count = 0;
    std::size_t final_piercing_count = 0;
    std::size_t minimum_piercing_count = 0;
    std::size_t full_checkpoint_count = 0;
    std::size_t watch_cycle_preparation_count = 0;
    detail::BondRingCheckpoint final_checkpoint;
    std::vector<RingUntanglingStep> steps;
    std::vector<std::string> warning_codes;
};


RingUntanglingResult untangle_ring_piercings(
    StructureSession& session,
    const RingUntanglingOptions& options,
    const PerturbationOffsetBatch& perturbation_offsets,
    RingUntanglingAttemptCursor& attempt_cursor,
    const detail::BondRingCheckpoint& entry_checkpoint
);


}  // namespace hotpot::forcefields
