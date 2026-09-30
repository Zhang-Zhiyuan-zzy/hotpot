#include "topology_workspace.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>


namespace ff = hotpot::forcefields;
namespace detail = hotpot::forcefields::detail;
namespace geo = hotpot::geometry;


namespace {


ff::ComplexSessionInput square_input() {
    return ff::ComplexSessionInput{
        1,
        {6, 6, 6, 6, 1, 1, 63},
        std::vector<std::int32_t>(7, 0),
        std::vector<double>(7, 0.0),
        {
            {-1.0, -1.0, 0.0},
            {1.0, -1.0, 0.0},
            {1.0, 1.0, 0.0},
            {-1.0, 1.0, 0.0},
            {0.0, 0.0, -1.0},
            {0.0, 0.0, 1.0},
            {0.0, -3.0, 0.0},
        },
        std::vector<std::uint8_t>(7, 0),
        {{0, 1}, {1, 2}, {2, 3}, {3, 0}},
        std::vector<double>(4, 1.0),
        std::vector<ff::BondKind>(4, ff::BondKind::SINGLE),
        std::vector<std::uint8_t>(4, 0),
        {6},
        {{6, 0}, {6, 1}},
        {1.0, 1.0},
        {ff::BondKind::DATIVE, ff::BondKind::DATIVE},
        std::nullopt,
    };
}


ff::ComplexSessionInput seventeen_membered_ring_input() {
    constexpr std::size_t atom_count = 17;
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::vector<ff::Coordinate> coordinates;
    std::vector<ff::BondIndex> bonds;
    coordinates.reserve(atom_count);
    bonds.reserve(atom_count);
    for (std::size_t index = 0; index < atom_count; ++index) {
        const double angle = 2.0 * pi * static_cast<double>(index)
            / static_cast<double>(atom_count);
        coordinates.push_back({std::cos(angle), std::sin(angle), 0.0});
        bonds.push_back({
            static_cast<std::int32_t>(index),
            static_cast<std::int32_t>((index + 1) % atom_count),
        });
    }
    return ff::ComplexSessionInput{
        1,
        std::vector<std::int32_t>(atom_count, 6),
        std::vector<std::int32_t>(atom_count, 0),
        std::vector<double>(atom_count, 0.0),
        std::move(coordinates),
        std::vector<std::uint8_t>(atom_count, 0),
        bonds,
        std::vector<double>(atom_count, 1.0),
        std::vector<ff::BondKind>(atom_count, ff::BondKind::SINGLE),
        std::vector<std::uint8_t>(atom_count, 0),
        {},
        {},
        {},
        {},
        std::nullopt,
    };
}


void test_relevant_cycle_filter_and_batch_screening() {
    const auto input = square_input();
    detail::RingWorkspaceOptions options;
    const auto workspace = detail::prepare_ring_workspace(
        input,
        input.coordinates,
        {1, 1, 1, 1},
        {0, 0},
        options
    );
    assert(workspace.topology.relevant_cycle_count == 1);
    assert(workspace.topology.excluded_large_cycle_count == 0);
    assert(workspace.prepared_cycles.cycle_count() == 1);
    assert(workspace.topology.atom_indices[0]
        == std::vector<std::size_t>({0, 1, 2, 3}));

    const auto piercing = detail::screen_segment_against_rings(
        geo::Segment3{input.coordinates[4], input.coordinates[5]},
        workspace,
        std::nullopt,
        false
    );
    assert(piercing.state == geo::PiercingState::PIERCES);
    assert(piercing.candidate_pair_count == 1);
    assert(piercing.piercing_pair_count == 1);
    assert(piercing.scan_complete);

    const auto own_edge = detail::screen_segment_against_rings(
        geo::Segment3{input.coordinates[0], input.coordinates[1]},
        workspace,
        ff::BondIndex{0, 1},
        false
    );
    assert(own_edge.state == geo::PiercingState::DOES_NOT_PIERCE);
    assert(own_edge.candidate_pair_count == 0);

    options.maximum_actionable_ring_size = 3;
    const auto filtered = detail::prepare_ring_workspace(
        input,
        input.coordinates,
        {1, 1, 1, 1},
        {0, 0},
        options
    );
    assert(filtered.topology.relevant_cycle_count == 1);
    assert(filtered.topology.excluded_large_cycle_count == 1);
    assert(filtered.prepared_cycles.cycle_count() == 0);
}


void test_full_graph_and_ligand_skeleton_scopes_are_distinct() {
    const auto input = square_input();
    detail::RingWorkspaceOptions options;
    options.scope = detail::RingGraphScope::FULL_GRAPH;
    const auto full_graph = detail::prepare_ring_topology(
        input,
        {1, 1, 1, 1},
        {1, 1},
        options
    );
    assert(full_graph.relevant_cycle_count == 2);

    options.scope = detail::RingGraphScope::LIGAND_SKELETON;
    const auto ligand = detail::prepare_ring_topology(
        input,
        {1, 1, 1, 1},
        {1, 1},
        options
    );
    assert(ligand.relevant_cycle_count == 1);
    assert(ligand.atom_indices[0]
        == std::vector<std::size_t>({0, 1, 2, 3}));
}


void test_default_ring_limit_excludes_seventeen_membered_cycle() {
    static_assert(detail::default_maximum_actionable_ring_size == 16);
    static_assert(detail::default_maximum_relevant_cycle_count == 10000);
    const auto input = seventeen_membered_ring_input();
    const auto workspace = detail::prepare_ring_workspace(
        input,
        input.coordinates,
        std::vector<std::uint8_t>(input.ligand_bond_count(), 1),
        {},
        detail::RingWorkspaceOptions{}
    );
    assert(workspace.topology.relevant_cycle_count == 1);
    assert(workspace.topology.excluded_large_cycle_count == 1);
    assert(workspace.prepared_cycles.cycle_count() == 0);
}


void test_workspace_cache_tracks_topology_and_coordinate_revisions() {
    const auto input = square_input();
    ff::StructureSnapshot snapshot{
        input.coordinates,
        {1, 1, 1, 1},
        {0, 0},
        {0, 0, 0, 0, 1, 2, 3},
        4,
        4,
        0,
        0,
    };
    detail::RingWorkspaceCache cache;
    detail::RingWorkspaceOptions options;
    static_cast<void>(cache.prepare(input, snapshot, options));
    static_cast<void>(cache.prepare(input, snapshot, options));
    assert(cache.topology_preparation_count() == 1);
    assert(cache.geometry_preparation_count() == 1);

    snapshot.coordinates[0][2] = 0.1;
    ++snapshot.coordinate_revision;
    static_cast<void>(cache.prepare(input, snapshot, options));
    assert(cache.topology_preparation_count() == 1);
    assert(cache.geometry_preparation_count() == 2);

    snapshot.active_coordination_mask = {1, 1};
    ++snapshot.topology_revision;
    static_cast<void>(cache.prepare(input, snapshot, options));
    assert(cache.topology_preparation_count() == 2);
    assert(cache.geometry_preparation_count() == 3);

    options.maximum_actionable_ring_size = 3;
    static_cast<void>(cache.prepare(input, snapshot, options));
    assert(cache.topology_preparation_count() == 3);
    assert(cache.geometry_preparation_count() == 4);
}


}  // namespace


int main() {
    test_relevant_cycle_filter_and_batch_screening();
    test_full_graph_and_ligand_skeleton_scopes_are_distinct();
    test_default_ring_limit_excludes_seventeen_membered_cycle();
    test_workspace_cache_tracks_topology_and_coordinate_revisions();
}
