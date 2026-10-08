#pragma once

#include "molecule_data.hpp"
#include "optimization_checks.hpp"
#include "rules.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


namespace OpenBabel {


class OBForceField;
class OBMol;


}  // namespace OpenBabel


namespace hotpot::obwrappers {


class OptimizationOperation;


class OptimizationFrameFailure : public std::runtime_error {
public:
    explicit OptimizationFrameFailure(std::string forcefield);

    const std::string& forcefield() const noexcept;

private:
    std::string forcefield_;
};


struct OptimizationOptions {
    std::string forcefield;
    std::string algorithm;
    std::size_t epochs;
    std::size_t steps_per_epoch;
    std::optional<std::size_t> perturb_interval;
    bool retain_frames;
    bool retain_epoch_history;
    bool increasing_vdw;
    double vdw_cutoff_start;
    double vdw_cutoff_end;
    double energy_tolerance;
    std::optional<StoppingCriteria> stopping_criteria;
    ConvergenceLevel convergence_level = ConvergenceLevel::STRICT;
};


struct OptimizationFrame {
    std::vector<Coordinate> coordinates;
    double energy;
    double rms_gradient;
    double max_gradient;
    bool exploded;
    bool converged;
    std::size_t epoch_index;
    std::size_t segment_epochs_completed;
    std::size_t segment_index;
    std::optional<double> energy_change;
    std::optional<double> max_displacement;
    std::size_t history_length;
};


struct OptimizationResult {
    std::vector<Coordinate> coordinates;
    std::vector<Coordinate> terminal_coordinates;
    std::vector<OptimizationFrame> frames;
    long selected_frame_index;
    long best_epoch;
    double final_energy;
    double best_energy;
    double rms_gradient;
    double max_gradient;
    bool exploded;
    bool converged;
    std::size_t epochs_completed;
    std::size_t steps_submitted;
    std::size_t initialization_steps;
    std::size_t selected_segment_epochs_completed;
    std::string backend_energy_unit;
    std::string termination_reason;
    bool terminal_converged;
    std::vector<double> energy_changes;
    std::vector<double> max_displacements;
    std::vector<double> epoch_energies;
    RulePlan rules;
    ConvergenceLevel convergence_level = ConvergenceLevel::STRICT;
};


namespace detail {


void validate_optimization_options(
    std::size_t atom_count,
    const OptimizationOptions& options,
    const std::vector<std::vector<Coordinate>>& perturbation_offsets
);


OptimizationResult run_optimization_controller(
    OpenBabel::OBMol& molecule,
    OptimizationOperation& operation,
    const OptimizationOptions& options,
    const std::vector<std::vector<Coordinate>>& perturbation_offsets,
    double energy_unit_to_kj,
    std::string backend_energy_unit,
    const std::vector<Coordinate>& initial_coordinates,
    double singularity_threshold,
    double repair_angle_radians
);


}  // namespace detail


}  // namespace hotpot::obwrappers
