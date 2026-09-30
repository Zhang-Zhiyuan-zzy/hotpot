#include "placement_engine.hpp"
#include "placement_policy.hpp"
#include "radii.hpp"
#include "structure_session.hpp"
#include "construction.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>


namespace ff = hotpot::forcefields;


namespace {


ff::ComplexSessionInput make_input(
    std::vector<std::int32_t> atomic_numbers,
    std::vector<ff::Coordinate> coordinates,
    std::vector<ff::BondIndex> ligand_bonds,
    std::vector<ff::BondIndex> coordination_bonds
) {
    const std::size_t atom_count = coordinates.size();
    return ff::ComplexSessionInput{
        1,
        std::move(atomic_numbers),
        std::vector<std::int32_t>(atom_count, 0),
        std::vector<double>(atom_count, 0.0),
        std::move(coordinates),
        std::vector<std::uint8_t>(atom_count, 0),
        ligand_bonds,
        std::vector<double>(ligand_bonds.size(), 1.0),
        std::vector<ff::BondKind>(ligand_bonds.size(), ff::BondKind::SINGLE),
        std::vector<std::uint8_t>(ligand_bonds.size(), 0),
        {0},
        coordination_bonds,
        std::vector<double>(coordination_bonds.size(), 1.0),
        std::vector<ff::BondKind>(
            coordination_bonds.size(),
            ff::BondKind::DATIVE
        ),
        std::nullopt,
    };
}


void assert_optional_equal(
    const std::optional<double>& first,
    const std::optional<double>& second
) {
    assert(first.has_value() == second.has_value());
    if (first.has_value()) {
        assert(*first == *second);
    }
}


void assert_donor_path_equal(
    const ff::DonorPathEvidence& first,
    const ff::DonorPathEvidence& second
) {
    assert(first.donor_index == second.donor_index);
    assert(first.group_index == second.group_index);
    assert(first.status == second.status);
    assert(first.distance_reachable == second.distance_reachable);
    assert(first.atom_obstructed == second.atom_obstructed);
    assert(first.bond_obstructed == second.bond_obstructed);
    assert(first.target_distance_angstrom == second.target_distance_angstrom);
    assert(first.distance_angstrom == second.distance_angstrom);
    assert(first.distance_ratio == second.distance_ratio);
    assert(first.normalized_atom_clearance
        == second.normalized_atom_clearance);
    assert(first.normalized_bond_clearance
        == second.normalized_bond_clearance);
    assert(first.definite_piercing_count == second.definite_piercing_count);
    assert(first.undetermined_relation_count
        == second.undetermined_relation_count);
    assert(first.atom_pair_count == second.atom_pair_count);
    assert(first.atom_aabb_rejected_pair_count
        == second.atom_aabb_rejected_pair_count);
    assert(first.bond_pair_count == second.bond_pair_count);
    assert(first.bond_aabb_rejected_pair_count
        == second.bond_aabb_rejected_pair_count);
    assert(first.cycle_pair_count == second.cycle_pair_count);
    assert(first.cycle_aabb_rejected_pair_count
        == second.cycle_aabb_rejected_pair_count);
    assert(first.approach_angles.size() == second.approach_angles.size());
    for (std::size_t index = 0; index < first.approach_angles.size(); ++index) {
        assert(first.approach_angles[index].neighbour_index
            == second.approach_angles[index].neighbour_index);
        assert_optional_equal(
            first.approach_angles[index].metal_donor_neighbour_angle_degrees,
            second.approach_angles[index].metal_donor_neighbour_angle_degrees
        );
    }
}


void assert_donor_pair_equal(
    const ff::DonorPairEvidence& first,
    const ff::DonorPairEvidence& second
) {
    assert(first.first_donor_index == second.first_donor_index);
    assert(first.second_donor_index == second.second_donor_index);
    assert(first.donor_separation_angstrom
        == second.donor_separation_angstrom);
    assert(first.target_distance_sum_angstrom
        == second.target_distance_sum_angstrom);
    assert(first.target_distance_difference_angstrom
        == second.target_distance_difference_angstrom);
    assert(first.target_shells_intersect == second.target_shells_intersect);
    assert_optional_equal(
        first.donor_metal_donor_angle_degrees,
        second.donor_metal_donor_angle_degrees
    );
}


void assert_candidate_evidence_equal(
    const ff::PlacementCandidateEvidence& first,
    const ff::PlacementCandidateEvidence& second
) {
    assert(first.coordinates == second.coordinates);
    assert(first.proposal_kind == second.proposal_kind);
    assert(first.status == second.status);
    assert(first.excluded_large_cycle_count
        == second.excluded_large_cycle_count);
    assert(first.covered_group_count == second.covered_group_count);
    assert(first.safe_donor_count == second.safe_donor_count);
    assert(first.out_of_range_donor_count
        == second.out_of_range_donor_count);
    assert(first.atom_obstruction_count == second.atom_obstruction_count);
    assert(first.bond_obstruction_count == second.bond_obstruction_count);
    assert(first.definite_piercing_count == second.definite_piercing_count);
    assert(first.hard_obstruction_count == second.hard_obstruction_count);
    assert(first.minimum_normalized_clearance
        == second.minimum_normalized_clearance);
    assert(first.worst_distance_deviation
        == second.worst_distance_deviation);
    assert(first.rms_distance_deviation == second.rms_distance_deviation);
    assert(first.undetermined_relation_count
        == second.undetermined_relation_count);
    assert(first.atom_pair_count == second.atom_pair_count);
    assert(first.atom_aabb_rejected_pair_count
        == second.atom_aabb_rejected_pair_count);
    assert(first.bond_pair_count == second.bond_pair_count);
    assert(first.bond_aabb_rejected_pair_count
        == second.bond_aabb_rejected_pair_count);
    assert(first.cycle_pair_count == second.cycle_pair_count);
    assert(first.cycle_aabb_rejected_pair_count
        == second.cycle_aabb_rejected_pair_count);
    assert(first.displacement_angstrom == second.displacement_angstrom);
    assert(first.proposal_ordinal == second.proposal_ordinal);
    assert(first.donor_paths.size() == second.donor_paths.size());
    for (std::size_t index = 0; index < first.donor_paths.size(); ++index) {
        assert_donor_path_equal(
            first.donor_paths[index],
            second.donor_paths[index]
        );
    }
    assert(first.donor_pairs.size() == second.donor_pairs.size());
    for (std::size_t index = 0; index < first.donor_pairs.size(); ++index) {
        assert_donor_pair_equal(
            first.donor_pairs[index],
            second.donor_pairs[index]
        );
    }
}


ff::PlacementCandidateEvidence ranked_evidence() {
    ff::PlacementCandidateEvidence evidence{};
    evidence.status = ff::PlacementStatus::PARTIAL;
    evidence.covered_group_count = 1;
    evidence.safe_donor_count = 1;
    evidence.definite_piercing_count = 2;
    evidence.hard_obstruction_count = 2;
    evidence.minimum_normalized_clearance = 0.5;
    evidence.worst_distance_deviation = 0.3;
    evidence.rms_distance_deviation = 0.2;
    evidence.undetermined_relation_count = 2;
    evidence.displacement_angstrom = 1.0;
    evidence.proposal_ordinal = 5;
    return evidence;
}


void assert_preferred(
    const ff::PlacementCandidateEvidence& candidate,
    const ff::PlacementCandidateEvidence& incumbent
) {
    assert(ff::detail::prefer_placement_candidate(candidate, incumbent));
    assert(!ff::detail::prefer_placement_candidate(incumbent, candidate));
}


void test_current_feasible_position_is_unchanged() {
    const double target = ff::covalent_radius(63).angstrom
        + ff::covalent_radius(7).angstrom;
    auto session = ff::create_coordination_session(make_input(
        {63, 7, 6, 1},
        {{-target, 0.0, 0.0}, {0.0, 0.0, 0.0},
         {1.4, 0.0, 0.0}, {20.0, 20.0, 20.0}},
        {{1, 2}},
        {{0, 1}}
    ));
    const auto before = ff::snapshot_structure(*session);
    const auto result = ff::place_metal(*session, 0);
    const auto after = ff::snapshot_structure(*session);
    assert(result.status == ff::PlacementStatus::FULLY_FEASIBLE);
    assert(!result.moved);
    assert(result.candidates_evaluated == 1);
    assert(result.selected_evidence.proposal_kind
        == ff::PlacementProposalKind::CURRENT);
    assert(before.coordinates == after.coordinates);
    assert(result.selected_evidence.atom_aabb_rejected_pair_count > 0);
}


void test_center_collision_moves_only_metal() {
    ff::MetalPlacementOptions options;
    options.retain_candidate_evidence = true;
    auto session = ff::create_coordination_session(make_input(
        {63, 7, 6, 1},
        {{0.66, 0.0, 0.0}, {3.35, 0.0, 0.0},
         {0.0, 0.0, 0.0}, {30.0, 0.0, 0.0}},
        {{1, 2}},
        {{0, 1}}
    ));
    const auto before = ff::snapshot_structure(*session);
    const auto initial = ff::assess_metal_position(
        *session,
        0,
        before.coordinates[0],
        options
    );
    assert(initial.status == ff::PlacementStatus::INFEASIBLE);
    assert(initial.hard_obstruction_count > 0);

    const auto result = ff::place_metal(*session, 0, options);
    const auto after = ff::snapshot_structure(*session);
    assert(result.status == ff::PlacementStatus::FULLY_FEASIBLE);
    assert(result.moved);
    assert(result.candidates_evaluated > 1);
    assert(result.retained_candidates.size() == result.candidates_evaluated);
    assert(after.coordinates[0] == result.selected_coordinates);
    for (std::size_t atom = 1; atom < after.coordinates.size(); ++atom) {
        assert(after.coordinates[atom] == before.coordinates[atom]);
    }
    assert(after.active_ligand_bond_mask == before.active_ligand_bond_mask);
    assert(after.active_coordination_mask == before.active_coordination_mask);
}


void test_threshold_and_shared_endpoint_semantics() {
    const double target = ff::covalent_radius(63).angstrom
        + ff::covalent_radius(7).angstrom;
    const double center_cutoff = 0.55 * (
        ff::covalent_radius(63).angstrom
        + ff::covalent_radius(6).angstrom
    );
    auto session = ff::create_coordination_session(make_input(
        {63, 7, 6},
        {{0.0, 0.0, 0.0}, {target, 0.0, 0.0},
         {0.0, center_cutoff + 1.0e-5, 0.0}},
        {},
        {{0, 1}}
    ));
    assert(ff::assess_metal_position(
        *session,
        0,
        {0.0, 0.0, 0.0}
    ).status == ff::PlacementStatus::FULLY_FEASIBLE);
    auto coordinates = ff::snapshot_structure(*session).coordinates;
    coordinates[2][1] = center_cutoff - 1.0e-5;
    ff::update_structure_coordinates(*session, coordinates);
    assert(ff::assess_metal_position(
        *session,
        0,
        {0.0, 0.0, 0.0}
    ).status == ff::PlacementStatus::INFEASIBLE);

    auto shared_endpoint = ff::create_coordination_session(make_input(
        {63, 7, 6},
        {{-target, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.4, 0.0, 0.0}},
        {{1, 2}},
        {{0, 1}}
    ));
    const auto shared = ff::assess_metal_position(
        *shared_endpoint,
        0,
        {-target, 0.0, 0.0}
    );
    assert(shared.donor_paths[0].status == ff::DonorPathStatus::SAFE);

    auto interior_overlap = ff::create_coordination_session(make_input(
        {63, 7, 6},
        {{target, 0.0, 0.0}, {0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}},
        {{1, 2}},
        {{0, 1}}
    ));
    const auto overlap = ff::assess_metal_position(
        *interior_overlap,
        0,
        {target, 0.0, 0.0}
    );
    assert(overlap.donor_paths[0].status != ff::DonorPathStatus::SAFE);
    assert(overlap.hard_obstruction_count > 0);

    auto interior_crossing = ff::create_coordination_session(make_input(
        {63, 7, 6, 6},
        {{target, 0.0, 0.0}, {0.0, 0.0, 0.0},
         {1.3, -2.0, 0.0}, {1.3, 2.0, 0.0}},
        {{2, 3}},
        {{0, 1}}
    ));
    const auto crossing = ff::assess_metal_position(
        *interior_crossing,
        0,
        {target, 0.0, 0.0}
    );
    assert(crossing.donor_paths[0].status
        == ff::DonorPathStatus::BOND_OBSTRUCTION);
}


void test_ring_piercing_and_generator_families() {
    const double half_target = 0.5 * (
        ff::covalent_radius(63).angstrom
        + ff::covalent_radius(7).angstrom
    );
    auto ring_session = ff::create_coordination_session(make_input(
        {63, 7, 6, 6, 6, 6},
        {{0.0, 0.0, half_target}, {0.0, 0.0, -half_target},
         {-3.0, -3.0, 0.0}, {3.0, -3.0, 0.0},
         {3.0, 3.0, 0.0}, {-3.0, 3.0, 0.0}},
        {{2, 3}, {3, 4}, {4, 5}, {5, 2}},
        {{0, 1}}
    ));
    const auto ring = ff::assess_metal_position(
        *ring_session,
        0,
        {0.0, 0.0, half_target}
    );
    assert(ring.status == ff::PlacementStatus::INFEASIBLE);
    assert(ring.donor_paths[0].status == ff::DonorPathStatus::RING_PIERCING);
    assert(ring.definite_piercing_count == 1);

    auto two_donor = ff::create_coordination_session(make_input(
        {63, 7, 8},
        {{0.0, 0.0, 0.0}, {-1.5, 0.0, 0.0}, {1.5, 0.0, 0.0}},
        {},
        {{0, 1}, {0, 2}}
    ));
    ff::MetalPlacementOptions options;
    options.retain_candidate_evidence = true;
    const auto two_result = ff::place_metal(*two_donor, 0, options);
    assert(std::any_of(
        two_result.retained_candidates.begin(),
        two_result.retained_candidates.end(),
        [](const ff::PlacementCandidateEvidence& evidence) {
            return evidence.proposal_kind
                == ff::PlacementProposalKind::SPHERE_INTERSECTION;
        }
    ));

    auto three_donor = ff::create_coordination_session(make_input(
        {63, 7, 8, 16},
        {{0.0, 0.0, 0.0}, {-1.5, 0.0, 0.0},
         {1.5, 0.0, 0.0}, {0.0, 1.5, 0.0}},
        {},
        {{0, 1}, {0, 2}, {0, 3}}
    ));
    const auto three_result = ff::place_metal(*three_donor, 0, options);
    assert(std::any_of(
        three_result.retained_candidates.begin(),
        three_result.retained_candidates.end(),
        [](const ff::PlacementCandidateEvidence& evidence) {
            return evidence.proposal_kind
                == ff::PlacementProposalKind::LEAST_SQUARES;
        }
    ));
}


void test_geometry_construction_kernels_are_translation_invariant() {
    namespace geometry = hotpot::geometry;
    const auto tolerances = geometry::default_numeric_tolerances();
    const std::vector<geometry::Point3> centers = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 2.0},
    };
    const std::vector<double> radii(centers.size(), std::sqrt(3.0));
    const geometry::Point3 initial = {0.7, 0.8, 0.9};
    const auto fitted = geometry::detail::fit_point_to_spheres(
        initial,
        geometry::ArrayView<geometry::Point3>(centers),
        geometry::ArrayView<double>(radii),
        32,
        tolerances
    );
    const geometry::Point3 shift = {1.0e5, -2.0e5, 3.0e5};
    std::vector<geometry::Point3> shifted_centers = centers;
    for (auto& point : shifted_centers) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            point[axis] += shift[axis];
        }
    }
    geometry::Point3 shifted_initial = initial;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        shifted_initial[axis] += shift[axis];
    }
    const auto shifted_fit = geometry::detail::fit_point_to_spheres(
        shifted_initial,
        geometry::ArrayView<geometry::Point3>(shifted_centers),
        geometry::ArrayView<double>(radii),
        32,
        tolerances
    );
    for (std::size_t axis = 0; axis < 3; ++axis) {
        assert(std::abs(
            (shifted_fit[axis] - shift[axis]) - fitted[axis]
        ) < 1.0e-8);
    }

    const auto angle = geometry::detail::angle_at_vertex(
        {1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        tolerances
    );
    assert(angle.has_value());
    assert(std::abs(*angle - 90.0) < 1.0e-12);

    const auto circle = geometry::detail::sphere_intersection_circle(
        {0.0, 0.0, 0.0},
        std::sqrt(2.0),
        {2.0, 0.0, 0.0},
        std::sqrt(2.0),
        tolerances
    );
    assert(circle.has_value());
    const auto point = geometry::detail::point_on_circle(*circle, 0.0);
    const auto distance_to = [&point](const geometry::Point3& target) {
        return std::hypot(
            point[0] - target[0],
            point[1] - target[1],
            point[2] - target[2]
        );
    };
    assert(std::abs(distance_to({0.0, 0.0, 0.0}) - std::sqrt(2.0))
        < 1.0e-12);
    assert(std::abs(distance_to({2.0, 0.0, 0.0}) - std::sqrt(2.0))
        < 1.0e-12);
}


void test_scalar_and_batch_candidate_evaluation_are_identical() {
    auto input = make_input(
        {63, 7, 8, 6, 6, 6, 6, 6},
        {
            {0.0, 0.0, 3.0},
            {0.0, 0.0, -3.0},
            {4.0, 0.0, 0.0},
            {-2.0, -2.0, 0.0},
            {2.0, -2.0, 0.0},
            {2.0, 2.0, 0.0},
            {-2.0, 2.0, 0.0},
            {5.4, 0.0, 0.0},
        },
        {
            {3, 4}, {4, 5}, {5, 6}, {6, 3},
            {1, 3}, {2, 5}, {2, 7},
        },
        {{0, 1}, {0, 2}}
    );
    auto session = ff::create_coordination_session(input);
    const auto snapshot = ff::snapshot_structure(*session);
    const ff::MetalPlacementOptions options;
    const auto workspace = ff::detail::prepare_placement_evaluation(
        input,
        snapshot.coordinates,
        snapshot.active_ligand_bond_mask,
        snapshot.component_ids,
        0,
        options
    );
    const std::vector<ff::detail::PlacementProposal> proposals = {
        {{0.0, 0.0, 3.0}, ff::PlacementProposalKind::CURRENT, 0},
        {{5.0, 0.0, 3.0}, ff::PlacementProposalKind::TARGET_SPHERE, 1},
        {{-4.0, 0.0, 2.0}, ff::PlacementProposalKind::LEAST_SQUARES, 2},
        {{0.0, -4.0, -2.0}, ff::PlacementProposalKind::FIBONACCI_FALLBACK, 3},
    };
    const auto batch = ff::detail::evaluate_metal_positions(
        workspace,
        proposals,
        options
    );
    assert(batch.size() == proposals.size());
    for (std::size_t index = 0; index < proposals.size(); ++index) {
        const auto scalar = ff::detail::evaluate_metal_position(
            workspace,
            proposals[index].coordinates,
            proposals[index].kind,
            proposals[index].ordinal,
            options
        );
        assert_candidate_evidence_equal(batch[index], scalar);
        assert(batch[index].donor_paths.size() == 2);
        assert(batch[index].donor_pairs.size() == 1);
        assert(batch[index].atom_pair_count > 0);
        assert(batch[index].bond_pair_count > 0);
        assert(batch[index].cycle_pair_count > 0);
    }
}


void test_candidate_ranking_is_lexicographic() {
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.status = ff::PlacementStatus::FULLY_FEASIBLE;
        candidate.covered_group_count = 0;
        assert_preferred(candidate, incumbent);
    }
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.covered_group_count = 2;
        candidate.safe_donor_count = 0;
        assert_preferred(candidate, incumbent);
    }
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.safe_donor_count = 2;
        candidate.definite_piercing_count = 100;
        assert_preferred(candidate, incumbent);
    }
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.definite_piercing_count = 1;
        candidate.hard_obstruction_count = 100;
        assert_preferred(candidate, incumbent);
    }
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.hard_obstruction_count = 1;
        candidate.minimum_normalized_clearance = 0.0;
        assert_preferred(candidate, incumbent);
    }
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.minimum_normalized_clearance = 0.6;
        candidate.worst_distance_deviation = 100.0;
        assert_preferred(candidate, incumbent);
    }
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.worst_distance_deviation = 0.2;
        candidate.rms_distance_deviation = 100.0;
        assert_preferred(candidate, incumbent);
    }
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.rms_distance_deviation = 0.1;
        candidate.undetermined_relation_count = 100;
        assert_preferred(candidate, incumbent);
    }
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.undetermined_relation_count = 1;
        candidate.displacement_angstrom = 100.0;
        assert_preferred(candidate, incumbent);
    }
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.displacement_angstrom = 0.5;
        candidate.proposal_ordinal = 100;
        assert_preferred(candidate, incumbent);
    }
    {
        auto candidate = ranked_evidence();
        auto incumbent = ranked_evidence();
        candidate.proposal_ordinal = 4;
        assert_preferred(candidate, incumbent);
    }
}


}  // namespace


int main() {
    test_current_feasible_position_is_unchanged();
    test_center_collision_moves_only_metal();
    test_threshold_and_shared_endpoint_semantics();
    test_ring_piercing_and_generator_families();
    test_geometry_construction_kernels_are_translation_invariant();
    test_scalar_and_batch_candidate_evaluation_are_identical();
    test_candidate_ranking_is_lexicographic();
}
