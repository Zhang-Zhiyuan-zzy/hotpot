#pragma once

#include "contracts.hpp"

#include "../../geometry/_native/batch.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>


namespace hotpot::forcefields {
namespace detail {


inline constexpr std::size_t default_maximum_actionable_ring_size = 16;
inline constexpr std::size_t default_maximum_relevant_cycle_count = 10000;


enum class RingGraphScope : std::uint8_t {
    LIGAND_SKELETON = 0,
    FULL_GRAPH = 1,
};


struct RingWorkspaceOptions {
    RingGraphScope scope = RingGraphScope::FULL_GRAPH;
    std::size_t maximum_actionable_ring_size =
        default_maximum_actionable_ring_size;
    std::size_t maximum_relevant_cycle_count =
        default_maximum_relevant_cycle_count;
    hotpot::geometry::NumericTolerances geometry_tolerances =
        hotpot::geometry::default_numeric_tolerances();
    hotpot::geometry::SurfaceEnumerationLimits surface_limits =
        hotpot::geometry::default_surface_enumeration_limits();

    void validate() const;
};


struct RingTopologyWorkspace {
    std::vector<std::vector<std::size_t>> atom_indices;
    std::vector<std::vector<BondIndex>> edge_keys;
    std::size_t relevant_cycle_count = 0;
    std::size_t excluded_large_cycle_count = 0;
};


struct PreparedRingWorkspace {
    RingTopologyWorkspace topology;
    hotpot::geometry::PreparedCycleBatch prepared_cycles;
};


struct SegmentRingScreeningReport {
    hotpot::geometry::PiercingState state =
        hotpot::geometry::PiercingState::DOES_NOT_PIERCE;
    std::size_t selected_ring_count = 0;
    std::size_t excluded_ring_count = 0;
    std::size_t candidate_pair_count = 0;
    std::size_t aabb_separated_pair_count = 0;
    std::size_t exact_pair_count = 0;
    std::size_t piercing_pair_count = 0;
    std::size_t does_not_pierce_pair_count = 0;
    std::size_t undetermined_pair_count = 0;
    bool scan_complete = true;
};


RingTopologyWorkspace prepare_ring_topology(
    const ComplexSessionInput& input,
    const std::vector<std::uint8_t>& active_ligand_bond_mask,
    const std::vector<std::uint8_t>& active_coordination_mask,
    const RingWorkspaceOptions& options
);


PreparedRingWorkspace prepare_ring_workspace(
    RingTopologyWorkspace topology,
    const std::vector<Coordinate>& coordinates,
    const RingWorkspaceOptions& options
);


PreparedRingWorkspace prepare_ring_workspace(
    const ComplexSessionInput& input,
    const std::vector<Coordinate>& coordinates,
    const std::vector<std::uint8_t>& active_ligand_bond_mask,
    const std::vector<std::uint8_t>& active_coordination_mask,
    const RingWorkspaceOptions& options
);


SegmentRingScreeningReport screen_segment_against_rings(
    const hotpot::geometry::Segment3& segment,
    const PreparedRingWorkspace& workspace,
    std::optional<BondIndex> segment_bond_key = std::nullopt,
    bool stop_after_confirmed = false
);


class RingWorkspaceCache final {
public:
    const PreparedRingWorkspace& prepare(
        const ComplexSessionInput& input,
        const StructureSnapshot& snapshot,
        const RingWorkspaceOptions& options
    );

    std::size_t topology_preparation_count() const noexcept {
        return topology_preparation_count_;
    }

    std::size_t geometry_preparation_count() const noexcept {
        return geometry_preparation_count_;
    }

private:
    std::optional<RingTopologyWorkspace> topology_;
    std::optional<PreparedRingWorkspace> prepared_;
    std::uint64_t topology_revision_ = 0;
    std::uint64_t coordinate_revision_ = 0;
    bool has_topology_revision_ = false;
    bool has_coordinate_revision_ = false;
    RingWorkspaceOptions options_;
    bool has_options_ = false;
    std::size_t topology_preparation_count_ = 0;
    std::size_t geometry_preparation_count_ = 0;
};


}  // namespace detail
}  // namespace hotpot::forcefields
