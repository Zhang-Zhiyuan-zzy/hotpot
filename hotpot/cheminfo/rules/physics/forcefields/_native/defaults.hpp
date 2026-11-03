#pragma once

#include <cstddef>


namespace hotpot::forcefields::detail {


inline constexpr std::size_t default_maximum_actionable_ring_size = 16;
inline constexpr std::size_t default_maximum_relevant_cycle_count = 10000;

inline constexpr std::size_t default_maximum_placement_candidate_count = 96;
inline constexpr std::size_t default_fibonacci_direction_count = 32;
inline constexpr std::size_t default_sphere_intersection_count = 24;
inline constexpr std::size_t default_least_squares_iteration_count = 16;
inline constexpr double default_coordination_distance_scale = 1.0;
inline constexpr double default_coordination_distance_ratio_minimum = 0.70;
inline constexpr double default_coordination_distance_ratio_maximum = 1.50;
inline constexpr double default_absolute_center_clearance_angstrom = 0.50;
inline constexpr double default_center_covalent_radius_scale = 0.55;
inline constexpr double default_minimum_path_atom_clearance = 0.35;
inline constexpr double default_minimum_path_bond_clearance = 0.50;
inline constexpr double default_broad_phase_skin_angstrom = 0.25;
inline constexpr double default_duplicate_tolerance_angstrom = 1.0e-8;


}  // namespace hotpot::forcefields::detail
