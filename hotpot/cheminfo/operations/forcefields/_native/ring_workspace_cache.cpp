#include "ring_workspace_cache.hpp"


namespace hotpot::forcefields {
namespace detail {
namespace {


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


}  // namespace


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
