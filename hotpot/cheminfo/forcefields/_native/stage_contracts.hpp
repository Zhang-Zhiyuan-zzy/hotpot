#pragma once

#include "contracts.hpp"
#include "trajectory.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>


namespace hotpot::forcefields {


enum class NativeStageStatus : std::uint8_t {
    COMPLETED = 0,
    PARTIAL = 1,
    FAILED = 2,
};


struct OptimizationStoppingOptions {
    std::size_t window = 5;
    double maximum_energy_change_kj_mol = 1.0e-4;
    double maximum_atom_displacement_angstrom = 1.0e-4;
    double maximum_rms_gradient_kj_mol_angstrom = 1.0;
    double maximum_gradient_kj_mol_angstrom = 5.0;

    void validate() const;
};


struct CoordinationStageOptions {
    std::string forcefield = "UFF";
    std::size_t attempt_limit = 20;
    std::size_t relaxation_steps = 100;
    double perturb_sigma = 0.5;
    NativeTrajectoryStart trajectory_start =
        NativeTrajectoryStart::COORDINATION_RESTORATION;
    FrameDetail frame_detail = FrameDetail::NONE;

    void validate() const;
};


struct ComplexOptimizationOptions {
    std::string forcefield = "UFF";
    std::string algorithm = "conjugate";
    std::size_t epochs = 100;
    std::size_t steps_per_epoch = 100;
    std::size_t untangling_attempt_limit = 30;
    std::optional<std::size_t> perturb_interval;
    double perturb_sigma = 0.5;
    NativeTrajectoryStart trajectory_start =
        NativeTrajectoryStart::COMPLEX_UNTANGLING;
    FrameDetail frame_detail = FrameDetail::NONE;
    bool retain_epoch_history = false;
    bool increasing_vdw = false;
    double vdw_cutoff_start = 0.0;
    double vdw_cutoff_end = 12.5;
    double energy_tolerance = 1.0e-6;
    std::optional<OptimizationStoppingOptions> stopping;

    void validate() const;
};


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
