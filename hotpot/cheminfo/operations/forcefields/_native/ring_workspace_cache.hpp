#pragma once

#include "session_contracts.hpp"
#include "../../../rules/physics/forcefields/_native/topology_workspace.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>


namespace hotpot::forcefields {
namespace detail {


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
