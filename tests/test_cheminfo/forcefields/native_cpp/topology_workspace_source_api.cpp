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


ff::ComplexSessionInput pierced_square_input() {
    auto input = square_input();
    input.ligand_bond_indices.push_back({5, 4});
    input.ligand_bond_orders.push_back(1.0);
    input.ligand_bond_kinds.push_back(ff::BondKind::SINGLE);
    input.ligand_bond_aromatic.push_back(0);
    return input;
}


ff::ComplexSessionInput permuted_pierced_square_input() {
    auto input = square_input();
    input.ligand_bond_indices = {
        {5, 4}, {3, 0}, {2, 3}, {1, 2}, {0, 1},
    };
    input.ligand_bond_orders = std::vector<double>(5, 1.0);
    input.ligand_bond_kinds =
        std::vector<ff::BondKind>(5, ff::BondKind::SINGLE);
    input.ligand_bond_aromatic = std::vector<std::uint8_t>(5, 0);
    return input;
}


ff::ComplexSessionInput twice_pierced_square_input() {
    auto input = pierced_square_input();
    input.atomic_numbers.insert(input.atomic_numbers.end(), {1, 1});
    input.formal_charges.insert(input.formal_charges.end(), {0, 0});
    input.partial_charges.insert(input.partial_charges.end(), {0.0, 0.0});
    input.coordinates.push_back({0.5, 0.0, -1.0});
    input.coordinates.push_back({0.5, 0.0, 1.0});
    input.atom_aromatic.insert(input.atom_aromatic.end(), {0, 0});
    input.ligand_bond_indices.push_back({8, 7});
    input.ligand_bond_orders.push_back(1.0);
    input.ligand_bond_kinds.push_back(ff::BondKind::SINGLE);
    input.ligand_bond_aromatic.push_back(0);
    return input;
}


ff::ComplexSessionInput mixed_ring_size_input() {
    constexpr std::size_t atom_count = 18;
    std::vector<ff::BondIndex> bonds = {{0, 1}, {1, 2}, {2, 0}, {1, 3}};
    for (std::int32_t atom = 3; atom < 17; ++atom) {
        bonds.push_back({atom, atom + 1});
    }
    bonds.push_back({17, 0});
    return ff::ComplexSessionInput{
        1,
        std::vector<std::int32_t>(atom_count, 6),
        std::vector<std::int32_t>(atom_count, 0),
        std::vector<double>(atom_count, 0.0),
        std::vector<ff::Coordinate>(atom_count, {0.0, 0.0, 0.0}),
        std::vector<std::uint8_t>(atom_count, 0),
        bonds,
        std::vector<double>(bonds.size(), 1.0),
        std::vector<ff::BondKind>(bonds.size(), ff::BondKind::SINGLE),
        std::vector<std::uint8_t>(bonds.size(), 0),
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
    assert(full_graph.atom_indices == std::vector<std::vector<std::size_t>>({
        {0, 1, 2, 3},
        {0, 1, 6},
    }));
    assert(full_graph.edge_memberships.at(ff::BondIndex{0, 1}) == 2);
    assert(full_graph.edge_memberships.at(ff::BondIndex{0, 6}) == 1);
    assert(full_graph.edge_memberships.at(ff::BondIndex{1, 6}) == 1);

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
    assert(workspace.topology.edge_memberships.size() == 17);
    for (const auto& item : workspace.topology.edge_memberships) {
        assert(item.second == 1);
    }
}


void test_checkpoint_retains_actionable_evidence_and_counts() {
    const auto input = twice_pierced_square_input();
    detail::RingWorkspaceOptions options;
    const auto workspace = detail::prepare_ring_workspace(
        input,
        input.coordinates,
        {1, 1, 1, 1, 1, 1},
        {0, 0},
        options
    );
    const auto checkpoint = detail::scan_bond_ring_checkpoint(
        workspace,
        options
    );
    assert(checkpoint.scope == detail::RingGraphScope::FULL_GRAPH);
    assert(checkpoint.maximum_actionable_ring_size == 16);
    assert(checkpoint.maximum_relevant_cycle_count == 10000);
    assert(checkpoint.relevant_cycle_count == 1);
    assert(checkpoint.selected_ring_count == 1);
    assert(checkpoint.excluded_ring_count == 0);
    assert(checkpoint.active_bond_count == 6);
    assert(checkpoint.candidate_pair_count == 2);
    assert(checkpoint.aabb_separated_pair_count == 0);
    assert(checkpoint.exact_pair_count == 2);
    assert(checkpoint.piercing_pair_count == 2);
    assert(checkpoint.does_not_pierce_pair_count == 0);
    assert(checkpoint.undetermined_pair_count == 0);
    assert(checkpoint.scan_complete);
    assert(checkpoint.state == geo::PiercingState::PIERCES);
    assert(checkpoint.actionable_findings.size() == 2);
    const auto& finding = checkpoint.actionable_findings.front();
    assert(finding.ring_index == 0);
    assert(finding.ring_atom_indices
        == std::vector<std::size_t>({0, 1, 2, 3}));
    assert(finding.bond_key == ff::BondIndex({4, 5}));
    assert(finding.relation.state == geo::PiercingState::PIERCES);
    assert(!finding.aabb_separated);
    assert(finding.surface_complete);
    assert(checkpoint.actionable_findings[1].bond_key
        == ff::BondIndex({7, 8}));
    assert(checkpoint.actionable_findings[1].relation.state
        == geo::PiercingState::PIERCES);
}


void test_checkpoint_is_stable_under_input_edge_permutation() {
    const auto input = pierced_square_input();
    const auto permuted_input = permuted_pierced_square_input();
    detail::RingWorkspaceOptions options;
    const auto first_workspace = detail::prepare_ring_workspace(
        input,
        input.coordinates,
        {1, 1, 1, 1, 1},
        {0, 0},
        options
    );
    const auto second_workspace = detail::prepare_ring_workspace(
        permuted_input,
        permuted_input.coordinates,
        {1, 1, 1, 1, 1},
        {0, 0},
        options
    );
    const auto first = detail::scan_bond_ring_checkpoint(
        first_workspace,
        options
    );
    const auto second = detail::scan_bond_ring_checkpoint(
        second_workspace,
        options
    );
    assert(first_workspace.topology.atom_indices
        == second_workspace.topology.atom_indices);
    assert(first_workspace.topology.active_bond_keys
        == second_workspace.topology.active_bond_keys);
    assert(first.actionable_findings.size()
        == second.actionable_findings.size());
    assert(first.actionable_findings.size() == 1);
    for (
        std::size_t index = 0;
        index < first.actionable_findings.size();
        ++index
    ) {
        assert(first.actionable_findings[index].ring_atom_indices
            == second.actionable_findings[index].ring_atom_indices);
        assert(first.actionable_findings[index].bond_key
            == second.actionable_findings[index].bond_key);
        assert(first.actionable_findings[index].relation.state
            == second.actionable_findings[index].relation.state);
    }
}


void test_active_mask_changes_checkpoint_candidates() {
    const auto input = pierced_square_input();
    detail::RingWorkspaceOptions options;
    const auto inactive_workspace = detail::prepare_ring_workspace(
        input,
        input.coordinates,
        {1, 1, 1, 1, 0},
        {0, 0},
        options
    );
    const auto active_workspace = detail::prepare_ring_workspace(
        input,
        input.coordinates,
        {1, 1, 1, 1, 1},
        {0, 0},
        options
    );
    const auto inactive = detail::scan_bond_ring_checkpoint(
        inactive_workspace,
        options
    );
    const auto active = detail::scan_bond_ring_checkpoint(
        active_workspace,
        options
    );
    assert(inactive.active_bond_count == 4);
    assert(inactive.candidate_pair_count == 0);
    assert(active.active_bond_count == 5);
    assert(active.candidate_pair_count == 1);
    assert(active.piercing_pair_count == 1);
}


void test_memberships_include_cycles_excluded_by_size() {
    const auto input = mixed_ring_size_input();
    const auto topology = detail::prepare_ring_topology(
        input,
        std::vector<std::uint8_t>(input.ligand_bond_count(), 1),
        {},
        detail::RingWorkspaceOptions{}
    );
    assert(topology.relevant_cycle_count == 2);
    assert(topology.excluded_large_cycle_count == 1);
    assert(topology.atom_indices
        == std::vector<std::vector<std::size_t>>({{0, 1, 2}}));
    assert(topology.edge_memberships.at(ff::BondIndex{0, 1}) == 2);
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
    test_checkpoint_retains_actionable_evidence_and_counts();
    test_checkpoint_is_stable_under_input_edge_permutation();
    test_active_mask_changes_checkpoint_candidates();
    test_memberships_include_cycles_excluded_by_size();
    test_workspace_cache_tracks_topology_and_coordinate_revisions();
}
