#include "topology_workspace.hpp"

#include "internal_helpers.hpp"

#include "../../geometry/_native/types.hpp"
#include "../../graph/_native/relevant_cycles.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>


namespace hotpot::forcefields {
namespace detail {
namespace {


using hotpot::geometry::ArrayView;
using hotpot::geometry::PiercingState;
using hotpot::geometry::Point3;
using hotpot::geometry::Segment3;
using hotpot::forcefields::internal::canonical_bond_key;


bool declared_metal(
    const ComplexSessionInput& input,
    std::size_t atom_index
) {
    return std::find(
        input.metal_indices.begin(),
        input.metal_indices.end(),
        static_cast<std::int32_t>(atom_index)
    ) != input.metal_indices.end();
}


std::vector<std::size_t> ordered_cycle_vertices(
    const std::vector<hotpot::graph::Edge>& edges,
    const hotpot::graph::CycleEdges& cycle_edges
) {
    std::map<std::size_t, std::vector<std::size_t>> adjacency;
    for (const auto edge_index : cycle_edges) {
        const auto& edge = edges[edge_index];
        adjacency[edge[0]].push_back(edge[1]);
        adjacency[edge[1]].push_back(edge[0]);
    }
    for (auto& item : adjacency) {
        std::sort(item.second.begin(), item.second.end());
        if (item.second.size() != 2) {
            throw std::runtime_error(
                "Relevant Cycles returned a non-simple cycle"
            );
        }
    }
    const std::size_t start = adjacency.begin()->first;
    std::vector<std::size_t> ordered{start};
    std::size_t previous = std::numeric_limits<std::size_t>::max();
    std::size_t current = start;
    while (ordered.size() < adjacency.size()) {
        const auto& neighbours = adjacency.at(current);
        const std::size_t next = neighbours[0] == previous
            ? neighbours[1]
            : neighbours[0];
        if (next == start) {
            throw std::runtime_error("Relevant Cycle closed prematurely");
        }
        ordered.push_back(next);
        previous = current;
        current = next;
    }
    const auto& final_neighbours = adjacency.at(current);
    if (final_neighbours[0] != start && final_neighbours[1] != start) {
        throw std::runtime_error("Relevant Cycle is not closed");
    }
    return ordered;
}


bool same_options(
    const RingWorkspaceOptions& first,
    const RingWorkspaceOptions& second
) noexcept {
    return first.scope == second.scope
        && first.maximum_actionable_ring_size
            == second.maximum_actionable_ring_size
        && first.maximum_relevant_cycle_count
            == second.maximum_relevant_cycle_count
        && first.geometry_tolerances.absolute_length
            == second.geometry_tolerances.absolute_length
        && first.geometry_tolerances.relative_length
            == second.geometry_tolerances.relative_length
        && first.geometry_tolerances.parameter
            == second.geometry_tolerances.parameter
        && first.geometry_tolerances.machine_epsilon_factor
            == second.geometry_tolerances.machine_epsilon_factor
        && first.geometry_tolerances.predicate_guard_factor
            == second.geometry_tolerances.predicate_guard_factor
        && first.geometry_tolerances.planarity_factor
            == second.geometry_tolerances.planarity_factor
        && first.geometry_tolerances.winding_residual
            == second.geometry_tolerances.winding_residual
        && first.geometry_tolerances.intersection_merge_factor
            == second.geometry_tolerances.intersection_merge_factor
        && first.geometry_tolerances.aabb_padding_factor
            == second.geometry_tolerances.aabb_padding_factor
        && first.surface_limits.maximum_cycle_vertices
            == second.surface_limits.maximum_cycle_vertices
        && first.surface_limits.maximum_surface_count
            == second.surface_limits.maximum_surface_count
        && first.surface_limits.maximum_segment_triangle_tests
            == second.surface_limits.maximum_segment_triangle_tests
        && first.surface_limits.maximum_triangle_pair_tests
            == second.surface_limits.maximum_triangle_pair_tests;
}


std::vector<BondIndex> active_bond_keys(
    const ComplexSessionInput& input,
    const std::vector<std::uint8_t>& active_ligand_bond_mask,
    const std::vector<std::uint8_t>& active_coordination_mask,
    RingGraphScope scope
) {
    std::vector<BondIndex> active_edges;
    active_edges.reserve(
        input.ligand_bond_count()
        + input.intended_coordination_bond_count()
    );
    for (std::size_t index = 0; index < input.ligand_bond_count(); ++index) {
        if (active_ligand_bond_mask[index] == 0) {
            continue;
        }
        const auto endpoints = canonical_bond_key(
            input.ligand_bond_indices[index]
        );
        if (
            scope == RingGraphScope::LIGAND_SKELETON
            && (
                declared_metal(input, static_cast<std::size_t>(endpoints[0]))
                || declared_metal(
                    input,
                    static_cast<std::size_t>(endpoints[1])
                )
            )
        ) {
            continue;
        }
        active_edges.push_back(endpoints);
    }
    if (scope == RingGraphScope::FULL_GRAPH) {
        for (
            std::size_t index = 0;
            index < input.intended_coordination_bond_count();
            ++index
        ) {
            if (active_coordination_mask[index] != 0) {
                active_edges.push_back(canonical_bond_key(
                    input.intended_coordination_bonds[index]
                ));
            }
        }
    }
    std::sort(active_edges.begin(), active_edges.end());
    return active_edges;
}


}  // namespace


void RingWorkspaceOptions::validate() const {
    if (maximum_actionable_ring_size < 3) {
        throw std::invalid_argument(
            "maximum_actionable_ring_size must be at least three"
        );
    }
    if (maximum_relevant_cycle_count == 0) {
        throw std::invalid_argument(
            "maximum_relevant_cycle_count must be positive"
        );
    }
    geometry_tolerances.validate();
    surface_limits.validate();
}


RingTopologyWorkspace prepare_ring_topology(
    const ComplexSessionInput& input,
    const std::vector<std::uint8_t>& active_ligand_bond_mask,
    const std::vector<std::uint8_t>& active_coordination_mask,
    const RingWorkspaceOptions& options
) {
    options.validate();
    if (active_ligand_bond_mask.size() != input.ligand_bond_count()
        || active_coordination_mask.size()
            != input.intended_coordination_bond_count()) {
        throw std::invalid_argument(
            "active bond masks must match the complex-session topology"
        );
    }

    const auto active_edges = active_bond_keys(
        input,
        active_ligand_bond_mask,
        active_coordination_mask,
        options.scope
    );

    std::vector<hotpot::graph::Edge> graph_edges;
    graph_edges.reserve(active_edges.size());
    for (const auto& endpoints : active_edges) {
        graph_edges.push_back({
            static_cast<hotpot::graph::VertexId>(endpoints[0]),
            static_cast<hotpot::graph::VertexId>(endpoints[1]),
        });
    }

    RingTopologyWorkspace workspace;
    workspace.active_bond_keys = active_edges;
    const auto cycles = hotpot::graph::relevant_cycles(
        graph_edges,
        {std::nullopt, options.maximum_relevant_cycle_count}
    );
    workspace.relevant_cycle_count = cycles.size();
    struct OrderedCycle {
        std::vector<std::size_t> atom_indices;
        std::vector<BondIndex> edge_keys;
    };
    std::vector<OrderedCycle> selected_cycles;
    for (const auto& cycle_edges : cycles) {
        auto atoms = ordered_cycle_vertices(graph_edges, cycle_edges);
        std::vector<BondIndex> edge_keys;
        edge_keys.reserve(cycle_edges.size());
        for (const auto edge_index : cycle_edges) {
            const BondIndex edge_key = active_edges[edge_index];
            edge_keys.push_back(edge_key);
            ++workspace.edge_memberships[edge_key];
        }
        if (atoms.size() > options.maximum_actionable_ring_size) {
            ++workspace.excluded_large_cycle_count;
            continue;
        }
        std::sort(edge_keys.begin(), edge_keys.end());
        selected_cycles.push_back({std::move(atoms), std::move(edge_keys)});
    }
    std::sort(
        selected_cycles.begin(),
        selected_cycles.end(),
        [](const OrderedCycle& first, const OrderedCycle& second) {
            return std::tie(first.atom_indices, first.edge_keys)
                < std::tie(second.atom_indices, second.edge_keys);
        }
    );
    for (auto& cycle : selected_cycles) {
        workspace.atom_indices.push_back(std::move(cycle.atom_indices));
        workspace.edge_keys.push_back(std::move(cycle.edge_keys));
    }
    return workspace;
}


PreparedRingWorkspace prepare_ring_workspace(
    RingTopologyWorkspace topology,
    const std::vector<Coordinate>& coordinates,
    const RingWorkspaceOptions& options
) {
    options.validate();
    std::vector<std::size_t> cycle_indices;
    std::vector<std::size_t> cycle_offsets{0};
    for (const auto& cycle_atoms : topology.atom_indices) {
        for (const auto atom : cycle_atoms) {
            if (atom >= coordinates.size()) {
                throw std::invalid_argument(
                    "ring topology contains an out-of-range atom index"
                );
            }
            cycle_indices.push_back(atom);
        }
        cycle_offsets.push_back(cycle_indices.size());
    }
    return PreparedRingWorkspace{
        std::move(topology),
        hotpot::geometry::prepare_cycles(
            ArrayView<Point3>(coordinates),
            ArrayView<std::size_t>(cycle_indices),
            ArrayView<std::size_t>(cycle_offsets),
            options.geometry_tolerances,
            options.surface_limits
        ),
    };
}


PreparedRingWorkspace prepare_ring_workspace(
    const ComplexSessionInput& input,
    const std::vector<Coordinate>& coordinates,
    const std::vector<std::uint8_t>& active_ligand_bond_mask,
    const std::vector<std::uint8_t>& active_coordination_mask,
    const RingWorkspaceOptions& options
) {
    return prepare_ring_workspace(
        prepare_ring_topology(
            input,
            active_ligand_bond_mask,
            active_coordination_mask,
            options
        ),
        coordinates,
        options
    );
}


SegmentRingScreeningReport screen_segment_against_rings(
    const Segment3& segment,
    const PreparedRingWorkspace& workspace,
    std::optional<BondIndex> segment_bond_key,
    bool stop_after_confirmed
) {
    SegmentRingScreeningReport report;
    report.selected_ring_count = workspace.prepared_cycles.cycle_count();
    report.excluded_ring_count =
        workspace.topology.excluded_large_cycle_count;
    if (segment_bond_key.has_value()) {
        *segment_bond_key = canonical_bond_key(*segment_bond_key);
    }
    std::vector<hotpot::geometry::SegmentCyclePair> candidate_pairs;
    candidate_pairs.reserve(workspace.prepared_cycles.cycle_count());
    for (
        std::size_t index = 0;
        index < workspace.prepared_cycles.cycle_count();
        ++index
    ) {
        if (
            segment_bond_key.has_value()
            && std::binary_search(
                workspace.topology.edge_keys[index].begin(),
                workspace.topology.edge_keys[index].end(),
                *segment_bond_key
            )
        ) {
            continue;
        }
        candidate_pairs.push_back({0, index});
    }
    const std::array<Segment3, 1> segments = {{segment}};
    const auto batch = hotpot::geometry::screen_segments(
        workspace.prepared_cycles,
        ArrayView<Segment3>(segments.data(), segments.size()),
        ArrayView<hotpot::geometry::SegmentCyclePair>(candidate_pairs),
        hotpot::geometry::DetailLevel::STATE_ONLY,
        stop_after_confirmed
    );
    report.candidate_pair_count = batch.evaluated_pair_count();
    report.aabb_separated_pair_count = batch.aabb_separated_pair_count();
    report.exact_pair_count = batch.exact_pair_count();
    report.piercing_pair_count = batch.piercing_pair_count();
    report.does_not_pierce_pair_count = batch.does_not_pierce_pair_count();
    report.undetermined_pair_count = batch.undetermined_pair_count();
    report.scan_complete = batch.scan_complete()
        && std::all_of(
            batch.surface_complete().begin(),
            batch.surface_complete().end(),
            [](std::uint8_t complete) { return complete != 0; }
        );
    if (report.piercing_pair_count != 0) {
        report.state = PiercingState::PIERCES;
    } else if (report.undetermined_pair_count != 0) {
        report.state = PiercingState::UNDETERMINED;
    }
    return report;
}


BondRingCheckpoint scan_bond_ring_checkpoint(
    const PreparedRingWorkspace& workspace,
    const RingWorkspaceOptions& options
) {
    options.validate();
    BondRingCheckpoint checkpoint;
    checkpoint.scope = options.scope;
    checkpoint.maximum_actionable_ring_size =
        options.maximum_actionable_ring_size;
    checkpoint.maximum_relevant_cycle_count =
        options.maximum_relevant_cycle_count;
    checkpoint.relevant_cycle_count =
        workspace.topology.relevant_cycle_count;
    checkpoint.selected_ring_count = workspace.prepared_cycles.cycle_count();
    checkpoint.excluded_ring_count =
        workspace.topology.excluded_large_cycle_count;
    checkpoint.active_bond_count =
        workspace.topology.active_bond_keys.size();

    std::vector<Segment3> segments;
    segments.reserve(workspace.topology.active_bond_keys.size());
    for (const BondIndex& bond_key : workspace.topology.active_bond_keys) {
        segments.push_back({
            workspace.prepared_cycles.coordinates()[
                static_cast<std::size_t>(bond_key[0])
            ],
            workspace.prepared_cycles.coordinates()[
                static_cast<std::size_t>(bond_key[1])
            ],
        });
    }

    std::vector<hotpot::geometry::SegmentCyclePair> candidate_pairs;
    for (
        std::size_t ring_index = 0;
        ring_index < workspace.prepared_cycles.cycle_count();
        ++ring_index
    ) {
        const auto& ring_edges = workspace.topology.edge_keys[ring_index];
        for (
            std::size_t bond_index = 0;
            bond_index < workspace.topology.active_bond_keys.size();
            ++bond_index
        ) {
            if (std::binary_search(
                ring_edges.begin(),
                ring_edges.end(),
                workspace.topology.active_bond_keys[bond_index]
            )) {
                continue;
            }
            candidate_pairs.push_back({bond_index, ring_index});
        }
    }

    const auto batch = hotpot::geometry::screen_segments(
        workspace.prepared_cycles,
        ArrayView<Segment3>(segments),
        ArrayView<hotpot::geometry::SegmentCyclePair>(candidate_pairs),
        hotpot::geometry::DetailLevel::ACTIONABLE,
        false
    );
    checkpoint.candidate_pair_count = batch.requested_pair_count();
    checkpoint.aabb_separated_pair_count =
        batch.aabb_separated_pair_count();
    checkpoint.exact_pair_count = batch.exact_pair_count();
    checkpoint.piercing_pair_count = batch.piercing_pair_count();
    checkpoint.does_not_pierce_pair_count =
        batch.does_not_pierce_pair_count();
    checkpoint.undetermined_pair_count = batch.undetermined_pair_count();
    checkpoint.scan_complete = batch.scan_complete()
        && std::all_of(
            batch.surface_complete().begin(),
            batch.surface_complete().end(),
            [](std::uint8_t complete) { return complete != 0; }
        );
    if (checkpoint.piercing_pair_count != 0) {
        checkpoint.state = PiercingState::PIERCES;
    } else if (checkpoint.undetermined_pair_count != 0) {
        checkpoint.state = PiercingState::UNDETERMINED;
    }

    checkpoint.actionable_findings.reserve(batch.relations().size());
    for (
        std::size_t relation_index = 0;
        relation_index < batch.relations().size();
        ++relation_index
    ) {
        const std::size_t pair_position =
            batch.relation_positions()[relation_index];
        const auto& pair = candidate_pairs[pair_position];
        checkpoint.actionable_findings.push_back({
            pair.cycle_index,
            workspace.topology.atom_indices[pair.cycle_index],
            workspace.topology.active_bond_keys[pair.segment_index],
            batch.relations()[relation_index],
            batch.aabb_separated()[pair_position] != 0,
            batch.surface_complete()[pair_position] != 0,
        });
    }
    return checkpoint;
}


const PreparedRingWorkspace& RingWorkspaceCache::prepare(
    const ComplexSessionInput& input,
    const StructureSnapshot& snapshot,
    const RingWorkspaceOptions& options
) {
    const bool options_changed = !has_options_
        || !same_options(options_, options);
    if (
        options_changed
        || !has_topology_revision_
        || topology_revision_ != snapshot.topology_revision
    ) {
        topology_ = prepare_ring_topology(
            input,
            snapshot.active_ligand_bond_mask,
            snapshot.active_coordination_mask,
            options
        );
        ++topology_preparation_count_;
        topology_revision_ = snapshot.topology_revision;
        has_topology_revision_ = true;
        prepared_.reset();
        has_coordinate_revision_ = false;
    }
    if (
        !prepared_.has_value()
        || !has_coordinate_revision_
        || coordinate_revision_ != snapshot.coordinate_revision
    ) {
        prepared_.emplace(prepare_ring_workspace(
            *topology_,
            snapshot.coordinates,
            options
        ));
        ++geometry_preparation_count_;
        coordinate_revision_ = snapshot.coordinate_revision;
        has_coordinate_revision_ = true;
    }
    options_ = options;
    has_options_ = true;
    return *prepared_;
}


}  // namespace detail
}  // namespace hotpot::forcefields
