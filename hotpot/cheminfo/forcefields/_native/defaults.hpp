#pragma once

#include <cstddef>


namespace hotpot::forcefields::detail {


inline constexpr char default_forcefield[] = "UFF";

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

inline constexpr std::size_t default_stopping_window = 5;
inline constexpr double default_maximum_energy_change_kj_mol = 1.0e-4;
inline constexpr double default_maximum_atom_displacement_angstrom = 1.0e-4;
inline constexpr double default_maximum_rms_gradient_kj_mol_angstrom = 1.0;
inline constexpr double default_maximum_gradient_kj_mol_angstrom = 5.0;

inline constexpr std::size_t default_coordination_attempt_limit = 20;
inline constexpr std::size_t default_coordination_relaxation_steps = 100;
inline constexpr std::size_t default_optimization_epochs = 100;
inline constexpr std::size_t default_steps_per_epoch = 100;
inline constexpr std::size_t default_untangling_attempt_limit = 30;
inline constexpr std::size_t default_untangling_short_optimization_steps = 100;
inline constexpr double default_perturb_sigma = 0.5;
inline constexpr double default_vdw_cutoff_start = 0.0;
inline constexpr double default_vdw_cutoff_end = 12.5;
inline constexpr double default_energy_tolerance = 1.0e-6;


}  // namespace hotpot::forcefields::detail
