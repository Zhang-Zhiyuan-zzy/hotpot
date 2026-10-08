#pragma once

#include "contracts.hpp"
#include "defaults.hpp"
#include "placement_policy.hpp"
#include "trajectory.hpp"

#include "../../geometry/_native/nonplanar_surface.hpp"
#include "../../geometry/_native/segment_cycle.hpp"
#include "../../geometry/_native/tolerances.hpp"
#include "../../obWrappers/_native/defaults.hpp"
#include "../../obWrappers/_native/optimization_checks.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>


namespace hotpot::forcefields {


namespace detail {


struct BondRingCheckpoint;
struct RingWorkspaceOptions;


}  // namespace detail


enum class NativeStageStatus : std::uint8_t {
    COMPLETED = 0,
    PARTIAL = 1,
    FAILED = 2,
};


enum class NativeRingGraphScope : std::uint8_t {
    LIGAND_SKELETON = 0,
    FULL_GRAPH = 1,
};


struct RingScreeningOptions {
    std::size_t maximum_actionable_ring_size =
        detail::default_maximum_actionable_ring_size;
    std::size_t maximum_relevant_cycle_count =
        detail::default_maximum_relevant_cycle_count;
    hotpot::geometry::NumericTolerances geometry_tolerances =
        hotpot::geometry::default_numeric_tolerances();
    hotpot::geometry::SurfaceEnumerationLimits surface_limits =
        hotpot::geometry::default_surface_enumeration_limits();

    void validate() const;
};


struct OptimizationStoppingOptions {
    std::size_t window = detail::default_stopping_window;
    double maximum_energy_change_kj_mol =
        detail::default_maximum_energy_change_kj_mol;
    double maximum_atom_displacement_angstrom =
        detail::default_maximum_atom_displacement_angstrom;
    double maximum_rms_gradient_kj_mol_angstrom =
        detail::default_maximum_rms_gradient_kj_mol_angstrom;
    double maximum_gradient_kj_mol_angstrom =
        detail::default_maximum_gradient_kj_mol_angstrom;

    void validate() const;
};


struct CoordinationStageOptions {
    std::string forcefield = detail::default_forcefield;
    std::size_t attempt_limit = detail::default_coordination_attempt_limit;
    std::size_t relaxation_steps =
        detail::default_coordination_relaxation_steps;
    double perturb_sigma = detail::default_perturb_sigma;
    NativeTrajectoryStart trajectory_start =
        NativeTrajectoryStart::COORDINATION_RESTORATION;
    FrameDetail frame_detail = FrameDetail::NONE;
    double torsion_singularity_threshold =
        hotpot::obwrappers::detail::default_torsion_singularity_threshold;
    double torsion_repair_angle_radians =
        hotpot::obwrappers::detail::default_torsion_repair_angle_radians;
    MetalPlacementOptions placement;

    void validate() const;
};


struct ComplexOptimizationOptions {
    std::string forcefield = detail::default_forcefield;
    std::string algorithm = "conjugate";
    std::size_t epochs = detail::default_optimization_epochs;
    std::size_t steps_per_epoch = detail::default_steps_per_epoch;
    std::size_t untangling_attempt_limit =
        detail::default_untangling_attempt_limit;
    std::optional<std::size_t> perturb_interval;
    double perturb_sigma = detail::default_perturb_sigma;
    NativeTrajectoryStart trajectory_start =
        NativeTrajectoryStart::COMPLEX_UNTANGLING;
    FrameDetail frame_detail = FrameDetail::NONE;
    bool retain_epoch_history = false;
    bool increasing_vdw = false;
    double vdw_cutoff_start = detail::default_vdw_cutoff_start;
    double vdw_cutoff_end = detail::default_vdw_cutoff_end;
    double energy_tolerance = detail::default_energy_tolerance;
    std::optional<OptimizationStoppingOptions> stopping;
    double torsion_singularity_threshold =
        hotpot::obwrappers::detail::default_torsion_singularity_threshold;
    double torsion_repair_angle_radians =
        hotpot::obwrappers::detail::default_torsion_repair_angle_radians;
    RingScreeningOptions ring_screening;
    hotpot::obwrappers::ConvergenceLevel convergence_level =
        hotpot::obwrappers::ConvergenceLevel::FAST;

    void validate() const;
};


struct NativeBondRingFinding {
    std::size_t ring_index = 0;
    std::vector<std::size_t> ring_atom_indices;
    BondIndex bond_key = {0, 0};
    hotpot::geometry::PiercingState state =
        hotpot::geometry::PiercingState::DOES_NOT_PIERCE;
    std::vector<hotpot::geometry::SegmentCycleIndeterminacy>
        indeterminacy_causes;
    bool aabb_separated = false;
    bool surface_complete = true;

    void validate() const;
};


struct NativeRingCheckpointReport {
    hotpot::geometry::PiercingState state =
        hotpot::geometry::PiercingState::DOES_NOT_PIERCE;
    NativeRingGraphScope scope = NativeRingGraphScope::FULL_GRAPH;
    std::size_t maximum_actionable_ring_size =
        detail::default_maximum_actionable_ring_size;
    std::size_t maximum_relevant_cycle_count =
        detail::default_maximum_relevant_cycle_count;
    std::size_t relevant_cycle_count = 0;
    std::size_t selected_ring_count = 0;
    std::size_t excluded_ring_count = 0;
    std::size_t active_bond_count = 0;
    std::size_t candidate_pair_count = 0;
    std::size_t aabb_separated_pair_count = 0;
    std::size_t exact_pair_count = 0;
    std::size_t piercing_pair_count = 0;
    std::size_t does_not_pierce_pair_count = 0;
    std::size_t undetermined_pair_count = 0;
    bool scan_complete = true;
    std::vector<NativeBondRingFinding> actionable_findings;

    void validate() const;
};


detail::RingWorkspaceOptions full_graph_ring_workspace_options(
    const RingScreeningOptions& options
);


NativeRingCheckpointReport native_ring_checkpoint_report(
    const detail::BondRingCheckpoint& checkpoint
);


struct CoordinationStageResult {
    NativeStageStatus status = NativeStageStatus::COMPLETED;
    std::vector<Coordinate> selected_coordinates;
    std::vector<Coordinate> terminal_coordinates;
    std::vector<std::uint8_t> final_active_coordination_mask;
    std::size_t attempt_limit = 0;
    std::size_t attempts_completed = 0;
    std::size_t metal_relocation_attempt_count = 0;
    std::vector<std::int32_t> relocated_metal_indices;
    std::vector<std::int32_t> infeasible_metal_indices;
    std::vector<BondIndex> forced_bond_keys;
    std::size_t rejected_piercing_trial_count = 0;
    std::size_t undetermined_trial_count = 0;
    std::size_t excluded_ring_observation_count = 0;
    std::vector<std::string> warning_codes;
    NativeTrajectoryBatch trajectory;
    std::size_t bond_count = 0;
    MetalPlacementReport placement_report;
    double elapsed_seconds = 0.0;

    std::size_t atom_count() const noexcept;
    void validate() const;
};


struct ComplexOptimizationResult {
    NativeStageStatus status = NativeStageStatus::COMPLETED;
    std::vector<Coordinate> selected_coordinates;
    std::vector<Coordinate> terminal_coordinates;
    std::vector<std::uint8_t> final_active_coordination_mask;
    std::size_t untangling_attempt_limit = 0;
    std::size_t untangling_attempts_completed = 0;
    std::size_t initial_piercing_count = 0;
    std::size_t final_piercing_count = 0;
    std::size_t minimum_piercing_count = 0;
    bool untangling_resolved = true;
    std::int64_t selected_frame_index = -1;
    std::int64_t best_epoch = -1;
    double final_energy_kj_mol = 0.0;
    double best_energy_kj_mol = 0.0;
    double rms_gradient_kj_mol_angstrom = 0.0;
    double max_gradient_kj_mol_angstrom = 0.0;
    std::vector<double> energy_changes;
    std::vector<double> max_displacements;
    std::vector<double> epoch_energies;
    bool exploded = false;
    bool converged = false;
    bool terminal_converged = false;
    std::size_t epochs_completed = 0;
    std::size_t steps_submitted = 0;
    std::size_t initialization_steps = 0;
    std::size_t selected_segment_epochs_completed = 0;
    std::string backend_energy_unit;
    std::string termination_reason;
    std::vector<std::string> warning_codes;
    NativeTrajectoryBatch trajectory;
    NativeRingCheckpointReport final_checkpoint;
    double elapsed_seconds = 0.0;

    std::size_t atom_count() const noexcept;
    void validate() const;
};


struct ComplexWorkflowResult {
    CoordinationStageResult coordination;
    ComplexOptimizationResult optimization;
    std::vector<Coordinate> selected_coordinates;
    std::vector<Coordinate> terminal_coordinates;
    std::vector<std::uint8_t> final_active_coordination_mask;
    std::vector<std::string> warning_codes;
    NativeTrajectoryBatch trajectory;

    std::size_t atom_count() const noexcept;
    void validate() const;
};


}  // namespace hotpot::forcefields
