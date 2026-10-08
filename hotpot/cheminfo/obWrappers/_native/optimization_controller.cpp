#include "optimization_controller.hpp"

#include "openbabel_adapter.hpp"
#include "optimization_operation.hpp"

#include <openbabel/forcefield.h>
#include <openbabel/mol.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>


namespace hotpot::obwrappers {


OptimizationFrameFailure::OptimizationFrameFailure(std::string forcefield) :
    std::runtime_error(
        "Open Babel force field " + forcefield
        + " completed without producing an optimization frame"
    ),
    forcefield_(std::move(forcefield)) {}


const std::string& OptimizationFrameFailure::forcefield() const noexcept {
    return forcefield_;
}


namespace detail {


namespace {


void append_rule_plan(RulePlan& destination, const RulePlan& source) {
    destination.applications.insert(
        destination.applications.end(),
        source.applications.begin(),
        source.applications.end()
    );
}


}  // namespace


void validate_optimization_options(
    std::size_t atom_count,
    const OptimizationOptions& options,
    const std::vector<std::vector<Coordinate>>& perturbation_offsets
) {
    if (options.forcefield.empty()) {
        throw std::invalid_argument("forcefield must not be empty");
    }
    if (options.epochs < 1 || options.steps_per_epoch < 1) {
        throw std::invalid_argument(
            "epochs and steps_per_epoch must be positive"
        );
    }
    if (options.algorithm != "conjugate"
        && options.algorithm != "steepest") {
        throw std::invalid_argument(
            "algorithm must be 'conjugate' or 'steepest'"
        );
    }
    const auto maximum_backend_steps = static_cast<std::size_t>(
        std::numeric_limits<int>::max()
    );
    if (options.epochs > maximum_backend_steps / options.steps_per_epoch) {
        throw std::invalid_argument(
            "epochs * steps_per_epoch exceeds the Open Babel step limit"
        );
    }
    if (options.perturb_interval.has_value()
        && *options.perturb_interval < 1) {
        throw std::invalid_argument("perturb_interval must be positive");
    }
    if (!options.perturb_interval.has_value()
        && !perturbation_offsets.empty()) {
        throw std::invalid_argument(
            "perturbation offsets require perturb_interval"
        );
    }
    std::size_t expected_offsets = 0;
    if (options.perturb_interval.has_value()) {
        expected_offsets = (options.epochs - 1) / *options.perturb_interval;
    }
    if (perturbation_offsets.size() != expected_offsets) {
        throw std::invalid_argument(
            "perturbation offset count does not match the schedule"
        );
    }
    for (const auto& offsets : perturbation_offsets) {
        if (offsets.size() != atom_count
            || !coordinates_are_finite(offsets)) {
            throw std::invalid_argument(
                "each perturbation offset must be finite and shaped (N, 3)"
            );
        }
    }
    if (!std::isfinite(options.vdw_cutoff_start)
        || !std::isfinite(options.vdw_cutoff_end)
        || options.vdw_cutoff_start < 0.0
        || options.vdw_cutoff_end < 0.0) {
        throw std::invalid_argument(
            "vdw cutoffs must be finite and nonnegative"
        );
    }
    if (options.increasing_vdw
        && options.vdw_cutoff_end < options.vdw_cutoff_start) {
        throw std::invalid_argument(
            "vdw_cutoff_end must not be smaller than vdw_cutoff_start"
        );
    }
    if (!std::isfinite(options.energy_tolerance)
        || options.energy_tolerance < 0.0) {
        throw std::invalid_argument(
            "energy_tolerance must be finite and nonnegative"
        );
    }
    if (!options.stopping_criteria.has_value()) {
        return;
    }
    const auto& criteria = *options.stopping_criteria;
    if (criteria.window < 1) {
        throw std::invalid_argument("stopping window must be positive");
    }
    const std::array<double, 4> limits = {
        criteria.maximum_energy_change_kj_mol,
        criteria.maximum_atom_displacement_angstrom,
        criteria.maximum_rms_gradient_kj_mol_angstrom,
        criteria.maximum_gradient_kj_mol_angstrom,
    };
    if (std::any_of(
            limits.begin(),
            limits.end(),
            [](double value) {
                return !std::isfinite(value) || value < 0.0;
            }
        )) {
        throw std::invalid_argument(
            "stopping thresholds must be finite and nonnegative"
        );
    }
}


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
) {
    auto& forcefield = operation.forcefield();
    RulePlan all_rules{RuleStage::PRE_FORCEFIELD_SETUP, {}};

    if (options.increasing_vdw) {
        operation.set_vdw_cutoff(options.vdw_cutoff_end);
    } else {
        operation.disable_cutoff();
    }
    auto setup_plan = operation.setup_and_validate(
        molecule,
        options.increasing_vdw,
        singularity_threshold,
        repair_angle_radians
    );
    append_rule_plan(all_rules, setup_plan);

    const auto total_steps = options.epochs * options.steps_per_epoch;
    if (options.increasing_vdw) {
        const double first_cutoff = options.vdw_cutoff_start
            + (options.vdw_cutoff_end - options.vdw_cutoff_start)
                / options.epochs;
        operation.set_vdw_cutoff(first_cutoff);
        setup_plan = operation.setup_and_validate(
            molecule,
            true,
            singularity_threshold,
            repair_angle_radians
        );
        append_rule_plan(all_rules, setup_plan);
    }

    auto initialization_for_epoch = operation.initialize(total_steps);
    std::vector<OptimizationFrame> frames;
    if (options.retain_frames) {
        frames.reserve(options.epochs);
    }
    std::vector<double> epoch_energies;
    if (options.retain_epoch_history) {
        epoch_energies.reserve(options.epochs);
    }
    std::vector<std::vector<double>> energy_change_segments(1);
    std::vector<std::vector<double>> displacement_segments(1);
    std::vector<std::vector<double>> rms_gradient_segments(1);
    std::vector<std::vector<double>> max_gradient_segments(1);
    std::optional<std::vector<Coordinate>> previous_coordinates;
    std::optional<double> previous_energy;
    std::size_t segment_index = 0;
    std::size_t segment_epochs_completed = 0;
    std::size_t epochs_completed = 0;
    std::size_t steps_submitted = 0;
    std::size_t initialization_steps = 0;
    std::size_t perturbation_index = 0;
    bool segment_active = true;
    bool terminal_converged = false;
    std::string termination_reason = "budget_exhausted";
    const double not_a_number = std::numeric_limits<double>::quiet_NaN();
    OptimizationFrame latest_returnable_frame{
        initial_coordinates,
        not_a_number,
        not_a_number,
        not_a_number,
        false,
        false,
        0,
        0,
        0,
        std::nullopt,
        std::nullopt,
        0,
    };
    long latest_returnable_epoch = -1;
    std::optional<OptimizationFrame> best_frame;
    long best_epoch = -1;
    std::optional<OptimizationFrame> last_frame;

    for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
        const bool reset_history = options.perturb_interval.has_value()
            && epoch > 0
            && epoch % *options.perturb_interval == 0;
        if (reset_history) {
            auto coordinates = extract_coordinates(molecule);
            const auto& offsets = perturbation_offsets[perturbation_index++];
            for (std::size_t index = 0; index < coordinates.size(); ++index) {
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    coordinates[index][axis] += offsets[index][axis];
                }
            }
            set_coordinates(molecule, coordinates);
        }

        if (options.increasing_vdw && epoch > 0) {
            const double cutoff = options.vdw_cutoff_start
                + (static_cast<double>(epoch + 1) / options.epochs)
                    * (options.vdw_cutoff_end - options.vdw_cutoff_start);
            operation.set_vdw_cutoff(cutoff);
        }

        const bool restart_segment = reset_history
            || (options.increasing_vdw && epoch > 0);
        if (restart_segment) {
            setup_plan = operation.setup_and_validate(
                molecule,
                options.increasing_vdw,
                singularity_threshold,
                repair_angle_radians
            );
            append_rule_plan(all_rules, setup_plan);
            energy_change_segments.emplace_back();
            displacement_segments.emplace_back();
            rms_gradient_segments.emplace_back();
            max_gradient_segments.emplace_back();
            ++segment_index;
            previous_coordinates.reset();
            previous_energy.reset();
            segment_epochs_completed = 0;
            const auto remaining_steps =
                (options.epochs - epoch) * options.steps_per_epoch;
            initialization_for_epoch = operation.initialize(remaining_steps);
            segment_active = true;
        }

        if (!segment_active) {
            continue;
        }

        const auto steps_to_take =
            options.steps_per_epoch - initialization_for_epoch;
        initialization_steps += initialization_for_epoch;
        const bool backend_continues = steps_to_take == 0
            || operation.take_steps(steps_to_take);
        steps_submitted += steps_to_take;
        initialization_for_epoch = 0;
        ++epochs_completed;
        ++segment_epochs_completed;
        const bool backend_stopped = !backend_continues;
        segment_active = backend_continues;
        operation.synchronize_coordinates(molecule);

        if (options.increasing_vdw && epoch < options.epochs - 1) {
            operation.set_vdw_cutoff(options.vdw_cutoff_end);
            setup_plan = operation.setup_and_validate(
                molecule,
                true,
                singularity_threshold,
                repair_angle_radians
            );
            append_rule_plan(all_rules, setup_plan);
        }

        auto coordinates = extract_coordinates(molecule);
        const auto measurements = measure_optimization_state(
            forcefield,
            molecule,
            coordinates,
            previous_coordinates.has_value()
                ? &*previous_coordinates
                : nullptr,
            previous_energy,
            energy_unit_to_kj
        );
        const bool backend_converged = convergence_reached(
            backend_stopped,
            measurements,
            energy_unit_to_kj,
            options.convergence_level
        );
        const bool reported_converged = backend_converged
            && (!options.increasing_vdw || epoch == options.epochs - 1);
        terminal_converged = reported_converged;
        termination_reason = reported_converged
            ? "converged"
            : "budget_exhausted";
        if (measurements.energy_change_kj_mol.has_value()) {
            energy_change_segments[segment_index].push_back(
                *measurements.energy_change_kj_mol
            );
        }
        if (measurements.maximum_displacement_angstrom.has_value()) {
            displacement_segments[segment_index].push_back(
                *measurements.maximum_displacement_angstrom
            );
        }
        OptimizationFrame frame{
            std::move(coordinates),
            measurements.energy_kj_mol,
            measurements.gradients.rms_kj_mol_angstrom,
            measurements.gradients.maximum_kj_mol_angstrom,
            measurements.exploded,
            reported_converged,
            epoch,
            segment_epochs_completed,
            segment_index,
            measurements.energy_change_kj_mol,
            measurements.maximum_displacement_angstrom,
            energy_change_segments[segment_index].size(),
        };
        rms_gradient_segments[segment_index].push_back(frame.rms_gradient);
        max_gradient_segments[segment_index].push_back(frame.max_gradient);
        previous_coordinates = frame.coordinates;
        previous_energy = frame.energy;

        const bool stable = !reported_converged
            && !options.increasing_vdw
            && options.stopping_criteria.has_value()
            && stability_reached(
                measurements,
                energy_change_segments[segment_index],
                displacement_segments[segment_index],
                rms_gradient_segments[segment_index],
                max_gradient_segments[segment_index],
                *options.stopping_criteria
            );
        const long observed_epoch = static_cast<long>(epochs_completed - 1);
        if (coordinates_are_finite(frame.coordinates)) {
            latest_returnable_frame = frame;
            latest_returnable_epoch = observed_epoch;
        }
        if (optimization_state_is_usable(measurements)
            && (!best_frame.has_value()
                || frame.energy < best_frame->energy)) {
            best_frame = frame;
            best_epoch = observed_epoch;
        }
        if (options.retain_epoch_history) {
            epoch_energies.push_back(frame.energy);
        }
        last_frame = frame;
        if (options.retain_frames) {
            frames.push_back(std::move(frame));
        }
        const auto failure = optimization_failure(measurements);
        if (failure != OptimizationFailure::NONE) {
            segment_active = false;
            terminal_converged = false;
            termination_reason = optimization_failure_reason(failure);
            break;
        }
        if (stable) {
            segment_active = false;
            termination_reason = "stability_reached";
        }
        const bool next_epoch_restarts_segment = epoch + 1 < options.epochs
            && (options.increasing_vdw
                || (options.perturb_interval.has_value()
                    && (epoch + 1) % *options.perturb_interval == 0));
        if (backend_stopped
            && !reported_converged
            && !stable
            && epoch + 1 < options.epochs
            && !next_epoch_restarts_segment) {
            // TakeNSteps() returns false for either convergence or exhaustion
            // of its current step budget. A false stop with a large global
            // gradient is neither, so restart from the current coordinates
            // within the caller's remaining epoch/step budget.
            energy_change_segments.emplace_back();
            displacement_segments.emplace_back();
            rms_gradient_segments.emplace_back();
            max_gradient_segments.emplace_back();
            ++segment_index;
            previous_coordinates.reset();
            previous_energy.reset();
            segment_epochs_completed = 0;
            const auto remaining_steps =
                (options.epochs - epoch - 1) * options.steps_per_epoch;
            initialization_for_epoch = operation.initialize(remaining_steps);
            segment_active = true;
        }
        if ((reported_converged || stable)
            && !options.increasing_vdw
            && !options.perturb_interval.has_value()) {
            break;
        }
    }

    if (!last_frame.has_value()) {
        throw OptimizationFrameFailure(options.forcefield);
    }
    if (!best_frame.has_value()) {
        best_frame = latest_returnable_frame;
        best_epoch = latest_returnable_epoch;
    }
    const auto selected_energy_changes = std::vector<double>(
        energy_change_segments[best_frame->segment_index].begin(),
        energy_change_segments[best_frame->segment_index].begin()
            + static_cast<std::ptrdiff_t>(best_frame->history_length)
    );
    const auto selected_displacements = std::vector<double>(
        displacement_segments[best_frame->segment_index].begin(),
        displacement_segments[best_frame->segment_index].begin()
            + static_cast<std::ptrdiff_t>(best_frame->history_length)
    );
    OptimizationResult result{
        best_frame->coordinates,
        last_frame->coordinates,
        std::move(frames),
        best_epoch,
        best_epoch,
        last_frame->energy,
        best_frame->energy,
        best_frame->rms_gradient,
        best_frame->max_gradient,
        best_frame->exploded,
        best_frame->converged,
        epochs_completed,
        steps_submitted,
        initialization_steps,
        best_frame->segment_epochs_completed,
        std::move(backend_energy_unit),
        termination_reason,
        terminal_converged,
        selected_energy_changes,
        selected_displacements,
        std::move(epoch_energies),
        std::move(all_rules),
        options.convergence_level,
    };
    set_coordinates(molecule, result.coordinates);
    return result;
}


}  // namespace detail


}  // namespace hotpot::obwrappers
