#pragma once

#include "defaults.hpp"
#include "placement_evidence.hpp"
#include "target_selection.hpp"
#include "topology_workspace.hpp"

#include "../../geometry/_native/nonplanar_surface.hpp"
#include "../../geometry/_native/prepared_cycle.hpp"
#include "../../geometry/_native/tolerances.hpp"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>


namespace hotpot::forcefields {


struct MetalPlacementOptions {
    std::size_t maximum_candidate_count =
        detail::default_maximum_placement_candidate_count;
    std::size_t fibonacci_direction_count =
        detail::default_fibonacci_direction_count;
    std::size_t sphere_intersection_count =
        detail::default_sphere_intersection_count;
    std::size_t least_squares_iteration_count =
        detail::default_least_squares_iteration_count;
    std::size_t maximum_actionable_ring_size =
        detail::default_maximum_actionable_ring_size;
    double coordination_distance_scale =
        detail::default_coordination_distance_scale;
    double coordination_distance_ratio_minimum =
        detail::default_coordination_distance_ratio_minimum;
    double coordination_distance_ratio_maximum =
        detail::default_coordination_distance_ratio_maximum;
    double absolute_center_clearance_angstrom =
        detail::default_absolute_center_clearance_angstrom;
    double center_covalent_radius_scale =
        detail::default_center_covalent_radius_scale;
    double minimum_path_atom_clearance =
        detail::default_minimum_path_atom_clearance;
    double minimum_path_bond_clearance =
        detail::default_minimum_path_bond_clearance;
    double broad_phase_skin_angstrom =
        detail::default_broad_phase_skin_angstrom;
    double duplicate_tolerance_angstrom =
        detail::default_duplicate_tolerance_angstrom;
    bool retain_candidate_evidence = false;
    hotpot::geometry::NumericTolerances geometry_tolerances =
        hotpot::geometry::default_numeric_tolerances();
    hotpot::geometry::SurfaceEnumerationLimits surface_limits =
        hotpot::geometry::default_surface_enumeration_limits();

    void validate() const {
        if (maximum_candidate_count == 0
            || fibonacci_direction_count == 0
            || sphere_intersection_count == 0
            || least_squares_iteration_count == 0
            || maximum_actionable_ring_size < 3) {
            throw std::invalid_argument(
                "placement counts and actionable ring size must be positive"
            );
        }
        const std::array<double, 9> values = {{
            coordination_distance_scale,
            coordination_distance_ratio_minimum,
            coordination_distance_ratio_maximum,
            absolute_center_clearance_angstrom,
            center_covalent_radius_scale,
            minimum_path_atom_clearance,
            minimum_path_bond_clearance,
            broad_phase_skin_angstrom,
            duplicate_tolerance_angstrom,
        }};
        if (std::any_of(values.begin(), values.end(), [](double value) {
                return !std::isfinite(value) || value < 0.0;
            })) {
            throw std::invalid_argument(
                "placement scalar options must be finite and nonnegative"
            );
        }
        if (coordination_distance_scale <= 0.0
            || coordination_distance_ratio_minimum <= 0.0
            || coordination_distance_ratio_maximum
                < coordination_distance_ratio_minimum
            || duplicate_tolerance_angstrom <= 0.0) {
            throw std::invalid_argument("placement option ranges are invalid");
        }
        geometry_tolerances.validate();
        surface_limits.validate();
    }
};


namespace detail {


struct PlacementEvaluationWorkspace {
    const ComplexSessionInput* input;
    const std::vector<Coordinate>* coordinates;
    const std::vector<std::uint8_t>* active_ligand_bond_mask;
    MetalPlacementTarget target;
    std::vector<double> radii;
    std::vector<std::uint8_t> donor_mask;
    std::vector<BondIndex> active_ligand_bonds;
    std::vector<hotpot::geometry::Aabb> atom_bounds;
    std::vector<hotpot::geometry::Aabb> active_ligand_bond_bounds;
    std::optional<PreparedRingWorkspace> cycles;
};


PlacementEvaluationWorkspace prepare_placement_evaluation(
    const ComplexSessionInput& input,
    const std::vector<Coordinate>& coordinates,
    const std::vector<std::uint8_t>& active_ligand_bond_mask,
    const std::vector<std::int32_t>& component_ids,
    std::int32_t metal_index,
    const MetalPlacementOptions& options
);


PlacementCandidateEvidence evaluate_metal_position(
    const PlacementEvaluationWorkspace& workspace,
    const Coordinate& candidate,
    PlacementProposalKind proposal_kind,
    std::size_t proposal_ordinal,
    const MetalPlacementOptions& options
);


std::vector<PlacementCandidateEvidence> evaluate_metal_positions(
    const PlacementEvaluationWorkspace& workspace,
    const std::vector<PlacementProposal>& proposals,
    const MetalPlacementOptions& options
);


bool prefer_placement_candidate(
    const PlacementCandidateEvidence& candidate,
    const PlacementCandidateEvidence& incumbent
) noexcept;


}  // namespace detail


}  // namespace hotpot::forcefields
