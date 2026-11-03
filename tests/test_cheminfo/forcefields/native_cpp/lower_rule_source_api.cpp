#include "hotpot/cheminfo/structure/_native/forcefield_input.hpp"
#include "hotpot/cheminfo/rules/physics/forcefields/_native/placement_candidates.hpp"
#include "hotpot/cheminfo/rules/physics/forcefields/_native/placement_policy.hpp"
#include "hotpot/cheminfo/rules/physics/forcefields/_native/radii.hpp"
#include "hotpot/cheminfo/rules/physics/forcefields/_native/target_selection.hpp"
#include "hotpot/cheminfo/rules/physics/forcefields/_native/topology_workspace.hpp"

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace ff = hotpot::forcefields;
namespace native = hotpot::native;
namespace detail = hotpot::forcefields::detail;
namespace geo = hotpot::geometry;

static_assert(std::is_same_v<ff::BondIndex, native::BondIndex>);
static_assert(std::is_same_v<ff::BondKind, native::BondKind>);
static_assert(std::is_same_v<ff::Coordinate, native::Coordinate>);

int main() {
    const ff::ComplexSessionInput input{
        1,
        {6, 6, 6, 6, 30},
        std::vector<std::int32_t>(5, 0),
        std::vector<double>(5, 0.0),
        {{-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0},
         {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0}, {3.0, 0.0, 0.0}},
        std::vector<std::uint8_t>(5, 0),
        {{0, 1}, {1, 2}, {2, 3}, {3, 0}},
        std::vector<double>(4, 1.0),
        std::vector<ff::BondKind>(4, ff::BondKind::SINGLE),
        std::vector<std::uint8_t>(4, 0),
        {4}, {}, {}, {}, std::nullopt,
    };
    input.validate();
    const auto view = ff::ligand_molecule_view(input);
    assert(view.coordinates.data() == input.coordinates.data());
    assert(view.bond_indices.data() == input.ligand_bond_indices.data());
    native::validate_molecule_view(view);

    const detail::RingWorkspaceOptions options;
    const auto workspace = detail::prepare_ring_workspace(
        input, input.coordinates, {1, 1, 1, 1}, {}, options
    );
    assert(workspace.topology.relevant_cycle_count == 1);
    const auto crossing = detail::screen_segment_against_rings(
        geo::Segment3{{0.0, 0.0, -1.0}, {0.0, 0.0, 1.0}}, workspace
    );
    assert(crossing.state == geo::PiercingState::PIERCES);
    assert(crossing.scan_complete);
    const auto clear = detail::screen_segment_against_rings(
        geo::Segment3{{3.0, 0.0, -1.0}, {3.0, 0.0, 1.0}}, workspace
    );
    assert(clear.state == geo::PiercingState::DOES_NOT_PIERCE);
    assert(clear.scan_complete);
    assert(input.coordinates == std::vector<ff::Coordinate>({
        {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0},
        {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0}, {3.0, 0.0, 0.0},
    }));
}
