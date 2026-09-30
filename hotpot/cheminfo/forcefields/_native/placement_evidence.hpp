#pragma once

#include "contracts.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>


namespace hotpot::forcefields {


enum class PlacementStatus : std::uint8_t {
    FULLY_FEASIBLE = 0,
    PARTIAL = 1,
    INFEASIBLE = 2,
};


enum class DonorPathStatus : std::uint8_t {
    SAFE = 0,
    UNDETERMINED = 1,
    OUT_OF_RANGE = 2,
    ATOM_OBSTRUCTION = 3,
    BOND_OBSTRUCTION = 4,
    RING_PIERCING = 5,
};


enum class PlacementProposalKind : std::uint8_t {
    CURRENT = 0,
    TARGET_SPHERE = 1,
    SPHERE_INTERSECTION = 2,
    LEAST_SQUARES = 3,
    FIBONACCI_FALLBACK = 4,
};


namespace detail {


struct PlacementProposal {
    Coordinate coordinates;
    PlacementProposalKind kind;
    std::size_t ordinal;
};


}  // namespace detail


struct DonorApproachEvidence {
    std::int32_t neighbour_index;
    std::optional<double> metal_donor_neighbour_angle_degrees;
};


struct DonorPathEvidence {
    std::int32_t donor_index;
    std::int32_t group_index;
    DonorPathStatus status;
    bool distance_reachable;
    bool atom_obstructed;
    bool bond_obstructed;
    double target_distance_angstrom;
    double distance_angstrom;
    double distance_ratio;
    double normalized_atom_clearance;
    double normalized_bond_clearance;
    std::size_t definite_piercing_count;
    std::size_t undetermined_relation_count;
    std::size_t atom_pair_count;
    std::size_t atom_aabb_rejected_pair_count;
    std::size_t bond_pair_count;
    std::size_t bond_aabb_rejected_pair_count;
    std::size_t cycle_pair_count;
    std::size_t cycle_aabb_rejected_pair_count;
    std::vector<DonorApproachEvidence> approach_angles;
};


struct DonorPairEvidence {
    std::int32_t first_donor_index;
    std::int32_t second_donor_index;
    double donor_separation_angstrom;
    double target_distance_sum_angstrom;
    double target_distance_difference_angstrom;
    bool target_shells_intersect;
    std::optional<double> donor_metal_donor_angle_degrees;
};


struct PlacementCandidateEvidence {
    Coordinate coordinates;
    PlacementProposalKind proposal_kind;
    PlacementStatus status;
    std::size_t excluded_large_cycle_count;
    std::size_t covered_group_count;
    std::size_t safe_donor_count;
    std::size_t out_of_range_donor_count;
    std::size_t atom_obstruction_count;
    std::size_t bond_obstruction_count;
    std::size_t definite_piercing_count;
    std::size_t hard_obstruction_count;
    double minimum_normalized_clearance;
    double worst_distance_deviation;
    double rms_distance_deviation;
    std::size_t undetermined_relation_count;
    std::size_t atom_pair_count;
    std::size_t atom_aabb_rejected_pair_count;
    std::size_t bond_pair_count;
    std::size_t bond_aabb_rejected_pair_count;
    std::size_t cycle_pair_count;
    std::size_t cycle_aabb_rejected_pair_count;
    double displacement_angstrom;
    std::size_t proposal_ordinal;
    std::vector<DonorPathEvidence> donor_paths;
    std::vector<DonorPairEvidence> donor_pairs;
};


struct MetalPlacementResult {
    std::int32_t metal_index;
    PlacementStatus status;
    Coordinate original_coordinates;
    Coordinate selected_coordinates;
    bool moved;
    std::size_t candidates_evaluated;
    PlacementCandidateEvidence selected_evidence;
    std::vector<PlacementCandidateEvidence> retained_candidates;
    std::size_t excluded_large_cycle_count;
    std::vector<std::string> warning_codes;
};


struct MetalPlacementReport {
    std::vector<MetalPlacementResult> metals;
    std::vector<Coordinate> selected_coordinates;
    std::vector<std::string> warning_codes;
};


}  // namespace hotpot::forcefields
