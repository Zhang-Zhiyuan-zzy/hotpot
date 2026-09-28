#pragma once

#include "molecule_data.hpp"
#include "rules.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


namespace hotpot::obwrappers {


class ForceFieldSetupFailure : public std::runtime_error {
public:
    ForceFieldSetupFailure(
        std::string forcefield,
        std::string stage,
        std::string message
    );

    const std::string& forcefield() const noexcept;
    const std::string& stage() const noexcept;

private:
    std::string forcefield_;
    std::string stage_;
};


struct BuildResult {
    bool succeeded;
    std::vector<Coordinate> coordinates;
    RulePlan rules;
};


struct SingleOptimizationResult {
    std::vector<Coordinate> coordinates;
    double energy_kj_mol;
    std::string backend_energy_unit;
    bool exploded;
    RulePlan rules;
};


struct StoppingCriteria {
    std::size_t window;
    double maximum_energy_change_kj_mol;
    double maximum_atom_displacement_angstrom;
    double maximum_rms_gradient_kj_mol_angstrom;
    double maximum_gradient_kj_mol_angstrom;
};


struct OptimizationOptions {
    std::string forcefield;
    std::string algorithm;
    std::size_t epochs;
    std::size_t steps_per_epoch;
    std::optional<std::size_t> perturb_interval;
    bool increasing_vdw;
    double vdw_cutoff_start;
    double vdw_cutoff_end;
    double energy_tolerance;
    std::optional<StoppingCriteria> stopping_criteria;
};


struct OptimizationFrame {
    std::vector<Coordinate> coordinates;
    double energy;
    double rms_gradient;
    double max_gradient;
    bool exploded;
    bool converged;
    std::size_t segment_epochs_completed;
    std::size_t segment_index;
    std::optional<double> energy_change;
    std::optional<double> max_displacement;
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
    RulePlan rules;
};


struct RuntimeInfo {
    std::string compiled_openbabel_version;
    std::string runtime_openbabel_version;
    int cxx11_abi;
    std::string babel_libdir;
    std::string babel_datadir;
};


RuntimeInfo runtime_info();

void seed_random(std::uint32_t seed);


BuildResult build(
    const MoleculeData& molecule,
    std::optional<bool> stereo_warnings
);


SingleOptimizationResult single_optimize(
    const MoleculeData& molecule,
    const std::string& forcefield,
    std::size_t steps,
    double singularity_threshold,
    double repair_angle_radians
);


OptimizationResult optimize(
    const MoleculeData& molecule,
    const OptimizationOptions& options,
    const std::vector<std::vector<Coordinate>>& perturbation_offsets,
    double singularity_threshold,
    double repair_angle_radians
);


}  // namespace hotpot::obwrappers
