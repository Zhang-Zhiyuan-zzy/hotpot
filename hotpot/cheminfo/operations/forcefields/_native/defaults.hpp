#pragma once

#include "../../../rules/physics/forcefields/_native/defaults.hpp"

#include <cstddef>


namespace hotpot::forcefields::detail {


inline constexpr char default_forcefield[] = "UFF";

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
