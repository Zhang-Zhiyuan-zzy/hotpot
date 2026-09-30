#include "stage_contracts.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace hotpot::forcefields {


namespace {


bool nonnegative_finite(double value) {
    return std::isfinite(value) && value >= 0.0;
}


void validate_active_mask(const std::vector<std::uint8_t>& mask) {
    if (std::any_of(mask.begin(), mask.end(), [](std::uint8_t value) {
            return value > 1;
        })) {
        throw std::invalid_argument(
            "final active coordination masks must be binary"
        );
    }
}


void validate_result_shape(
    const std::vector<Coordinate>& selected,
    const std::vector<Coordinate>& terminal,
    const std::vector<std::uint8_t>& active_mask,
    const NativeTrajectoryBatch& trajectory
) {
    if (selected.size() != terminal.size()) {
        throw std::invalid_argument(
            "selected and terminal coordinate counts must match"
        );
    }
    validate_active_mask(active_mask);
    if (trajectory.atom_count != selected.size()) {
        throw std::invalid_argument(
            "trajectory atom_count must match stage-result coordinates"
        );
    }
    if (trajectory.intended_coordination_bond_count != active_mask.size()) {
        throw std::invalid_argument(
            "trajectory coordination-bond count must match the final mask"
        );
    }
    trajectory.validate();
    if (trajectory.selected_frame_index >= 0
        && trajectory.frames[static_cast<std::size_t>(
            trajectory.selected_frame_index
        )].coordinates != selected) {
        throw std::invalid_argument(
            "selected coordinates must match the selected trajectory frame"
        );
    }
    if (trajectory.terminal_frame_index >= 0
        && trajectory.frames[static_cast<std::size_t>(
            trajectory.terminal_frame_index
        )].coordinates != terminal) {
        throw std::invalid_argument(
            "terminal coordinates must match the terminal trajectory frame"
        );
    }
}


}  // namespace


void OptimizationStoppingOptions::validate() const {
    if (window == 0) {
        throw std::invalid_argument("stopping window must be positive");
    }
    if (!nonnegative_finite(maximum_energy_change_kj_mol)
        || !nonnegative_finite(maximum_atom_displacement_angstrom)
        || !nonnegative_finite(maximum_rms_gradient_kj_mol_angstrom)
        || !nonnegative_finite(maximum_gradient_kj_mol_angstrom)) {
        throw std::invalid_argument(
            "optimization stopping thresholds must be finite and "
            "nonnegative"
        );
    }
}


void CoordinationStageOptions::validate() const {
    if (forcefield.empty()) {
        throw std::invalid_argument("forcefield must not be empty");
    }
    if (attempt_limit == 0) {
        throw std::invalid_argument("attempt_limit must be positive");
    }
    if (relaxation_steps == 0) {
        throw std::invalid_argument("relaxation_steps must be positive");
    }
    if (!nonnegative_finite(perturb_sigma)) {
        throw std::invalid_argument(
            "perturb_sigma must be finite and nonnegative"
        );
    }
    placement.validate();
}


void ComplexOptimizationOptions::validate() const {
    if (forcefield.empty()) {
        throw std::invalid_argument("forcefield must not be empty");
    }
    if (algorithm != "steepest" && algorithm != "conjugate") {
        throw std::invalid_argument(
            "algorithm must be 'steepest' or 'conjugate'"
        );
    }
    if (epochs == 0) {
        throw std::invalid_argument("epochs must be positive");
    }
    if (steps_per_epoch == 0) {
        throw std::invalid_argument("steps_per_epoch must be positive");
    }
    if (untangling_attempt_limit == 0) {
        throw std::invalid_argument(
            "untangling_attempt_limit must be positive"
        );
    }
    if (perturb_interval.has_value() && *perturb_interval == 0) {
        throw std::invalid_argument(
            "perturb_interval must be positive when provided"
        );
    }
    if (!nonnegative_finite(perturb_sigma)) {
        throw std::invalid_argument(
            "perturb_sigma must be finite and nonnegative"
        );
    }
    if (!std::isfinite(vdw_cutoff_start)
        || !std::isfinite(vdw_cutoff_end)
        || vdw_cutoff_start < 0.0
        || vdw_cutoff_end < 0.0
        || (increasing_vdw && vdw_cutoff_end < vdw_cutoff_start)) {
        throw std::invalid_argument("vdw cutoffs are invalid");
    }
    if (!nonnegative_finite(energy_tolerance)) {
        throw std::invalid_argument(
            "energy_tolerance must be finite and nonnegative"
        );
    }
    if (stopping.has_value()) {
        stopping->validate();
    }
}


std::size_t CoordinationStageResult::atom_count() const noexcept {
    return selected_coordinates.size();
}


void CoordinationStageResult::validate() const {
    validate_result_shape(
        selected_coordinates,
        terminal_coordinates,
        final_active_coordination_mask,
        trajectory
    );
    if (attempts_completed > attempt_limit) {
        throw std::invalid_argument(
            "attempts_completed must not exceed attempt_limit"
        );
    }
}


std::size_t ComplexOptimizationResult::atom_count() const noexcept {
    return selected_coordinates.size();
}


void ComplexOptimizationResult::validate() const {
    validate_result_shape(
        selected_coordinates,
        terminal_coordinates,
        final_active_coordination_mask,
        trajectory
    );
    if (untangling_attempts_completed > untangling_attempt_limit) {
        throw std::invalid_argument(
            "untangling attempts must not exceed the attempt limit"
        );
    }
    if (minimum_piercing_count > initial_piercing_count
        || minimum_piercing_count > final_piercing_count) {
        throw std::invalid_argument(
            "minimum_piercing_count is inconsistent with stage evidence"
        );
    }
    if (energy_changes.size() != max_displacements.size()) {
        throw std::invalid_argument(
            "energy-change and displacement histories must have equal lengths"
        );
    }
    if (epoch_energies.size() != epochs_completed) {
        throw std::invalid_argument(
            "epoch energy history must match epochs_completed"
        );
    }
}


std::size_t ComplexWorkflowResult::atom_count() const noexcept {
    return selected_coordinates.size();
}


void ComplexWorkflowResult::validate() const {
    coordination.validate();
    optimization.validate();
    validate_result_shape(
        selected_coordinates,
        terminal_coordinates,
        final_active_coordination_mask,
        trajectory
    );
    if (coordination.atom_count() != atom_count()
        || optimization.atom_count() != atom_count()) {
        throw std::invalid_argument(
            "composed stage results must use one stable atom order"
        );
    }
}


}  // namespace hotpot::forcefields
