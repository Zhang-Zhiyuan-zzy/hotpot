#pragma once

#include <cstddef>


namespace hotpot::geometry::detail {


inline constexpr double default_absolute_length = 1.0e-8;
inline constexpr double default_relative_length = 1.0e-10;
inline constexpr double default_parameter_tolerance = 1.0e-10;
inline constexpr double default_machine_epsilon_factor = 64.0;
inline constexpr double default_predicate_guard_factor = 4.0;
inline constexpr double default_planarity_factor = 1.0;
inline constexpr double default_winding_residual = 1.0e-10;
inline constexpr double default_intersection_merge_factor = 4.0;
inline constexpr double default_aabb_padding_factor = 4.0;

inline constexpr std::size_t default_maximum_surface_cycle_vertices = 8;
inline constexpr std::size_t default_maximum_surface_count = 132;
inline constexpr std::size_t default_maximum_segment_triangle_tests = 792;
inline constexpr std::size_t default_maximum_triangle_pair_tests = 1980;


}  // namespace hotpot::geometry::detail
