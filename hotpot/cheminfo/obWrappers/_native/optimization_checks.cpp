#include "optimization_checks.hpp"

#include <openbabel/forcefield.h>
#include <openbabel/math/vector3.h>
#include <openbabel/mol.h>

#include <algorithm>
#include <cmath>


namespace hotpot::obwrappers {


double optimization_energy_kj(
    OpenBabel::OBForceField& forcefield,
    double energy_unit_to_kj,
    bool calculate_gradients
) {
    return forcefield.Energy(calculate_gradients) * energy_unit_to_kj;
}


GradientMetrics optimization_gradient_metrics(
    OpenBabel::OBForceField& forcefield,
    OpenBabel::OBMol& molecule,
    double energy_unit_to_kj
) {
    double squared_norm_sum = 0.0;
    double maximum_norm = 0.0;
    for (unsigned int index = 1; index <= molecule.NumAtoms(); ++index) {
        const auto gradient = forcefield.GetGradient(
            molecule.GetAtom(static_cast<int>(index))
        );
        const double x = gradient.GetX() * energy_unit_to_kj;
        const double y = gradient.GetY() * energy_unit_to_kj;
        const double z = gradient.GetZ() * energy_unit_to_kj;
        const double squared_norm = x * x + y * y + z * z;
        squared_norm_sum += squared_norm;
        maximum_norm = std::max(maximum_norm, std::sqrt(squared_norm));
    }
    return {
        std::sqrt(squared_norm_sum / molecule.NumAtoms()),
        maximum_norm,
    };
}


bool coordinates_are_finite(
    const Coordinate* coordinates,
    std::size_t count
) noexcept {
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(coordinates[index][0])
            || !std::isfinite(coordinates[index][1])
            || !std::isfinite(coordinates[index][2])) {
            return false;
        }
    }
    return true;
}


double maximum_displacement(
    const Coordinate* current,
    const Coordinate* previous,
    std::size_t count
) noexcept {
    double maximum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double x = current[index][0] - previous[index][0];
        const double y = current[index][1] - previous[index][1];
        const double z = current[index][2] - previous[index][2];
        maximum = std::max(maximum, std::sqrt(x * x + y * y + z * z));
    }
    return maximum;
}


bool forcefield_detects_explosion(
    OpenBabel::OBForceField& forcefield
) noexcept {
    return forcefield.DetectExplosion();
}


OptimizationMeasurements measure_optimization_state(
    OpenBabel::OBForceField& forcefield,
    OpenBabel::OBMol& molecule,
    const std::vector<Coordinate>& coordinates,
    const std::vector<Coordinate>* previous_coordinates,
    std::optional<double> previous_energy_kj_mol,
    double energy_unit_to_kj
) {
    const double energy = optimization_energy_kj(
        forcefield, energy_unit_to_kj, true
    );
    const auto gradients = optimization_gradient_metrics(
        forcefield, molecule, energy_unit_to_kj
    );
    return {
        energy,
        gradients,
        previous_energy_kj_mol.has_value()
            ? std::optional<double>(std::abs(
                energy - *previous_energy_kj_mol
            ))
            : std::nullopt,
        previous_coordinates != nullptr
            ? std::optional<double>(maximum_displacement(
                coordinates, *previous_coordinates
            ))
            : std::nullopt,
        coordinates_are_finite(coordinates),
        forcefield_detects_explosion(forcefield),
    };
}


OptimizationFailure optimization_failure(
    const OptimizationMeasurements& measurements
) noexcept {
    if (!measurements.finite_coordinates) {
        return OptimizationFailure::NONFINITE_COORDINATES;
    }
    if (!std::isfinite(measurements.energy_kj_mol)) {
        return OptimizationFailure::NONFINITE_ENERGY;
    }
    if (!std::isfinite(measurements.gradients.rms_kj_mol_angstrom)
        || !std::isfinite(
            measurements.gradients.maximum_kj_mol_angstrom
        )) {
        return OptimizationFailure::NONFINITE_GRADIENTS;
    }
    if (measurements.exploded) {
        return OptimizationFailure::EXPLOSION_DETECTED;
    }
    return OptimizationFailure::NONE;
}


const char* optimization_failure_reason(
    OptimizationFailure failure
) noexcept {
    switch (failure) {
        case OptimizationFailure::NONE:
            return nullptr;
        case OptimizationFailure::NONFINITE_COORDINATES:
            return "nonfinite_coordinates";
        case OptimizationFailure::NONFINITE_ENERGY:
            return "nonfinite_energy";
        case OptimizationFailure::NONFINITE_GRADIENTS:
            return "nonfinite_gradients";
        case OptimizationFailure::EXPLOSION_DETECTED:
            return "explosion_detected";
    }
    return nullptr;
}


bool optimization_state_is_usable(
    const OptimizationMeasurements& measurements
) noexcept {
    return optimization_failure(measurements) == OptimizationFailure::NONE;
}


bool backend_stop_is_converged(
    bool backend_stopped,
    double maximum_gradient_kj_mol_angstrom,
    double energy_unit_to_kj,
    double backend_maximum_gradient
) noexcept {
    return backend_stopped
        && std::isfinite(maximum_gradient_kj_mol_angstrom)
        && maximum_gradient_kj_mol_angstrom
            <= backend_maximum_gradient * energy_unit_to_kj;
}


bool convergence_reached(
    bool backend_stopped,
    const OptimizationMeasurements& measurements,
    double energy_unit_to_kj,
    ConvergenceLevel level
) noexcept {
    if (!backend_stopped) {
        return false;
    }
    if (level == ConvergenceLevel::STRICT) {
        return backend_stop_is_converged(
            backend_stopped,
            measurements.gradients.maximum_kj_mol_angstrom,
            energy_unit_to_kj
        );
    }
    if (!optimization_state_is_usable(measurements)) {
        return false;
    }
    switch (level) {
        case ConvergenceLevel::OPENBABEL:
            return true;
        case ConvergenceLevel::FAST:
            return measurements.gradients.rms_kj_mol_angstrom <= 3.0
                && measurements.gradients.maximum_kj_mol_angstrom <= 10.0;
        case ConvergenceLevel::BALANCED:
            return measurements.gradients.rms_kj_mol_angstrom <= 1.0
                && measurements.gradients.maximum_kj_mol_angstrom <= 5.0;
        case ConvergenceLevel::STRICT:
            return false;
    }
    return false;
}


bool recent_values_at_most(
    ScalarHistoryView history,
    std::size_t window,
    double maximum
) noexcept {
    if (history.size < window) {
        return false;
    }
    const std::size_t begin = history.size - window;
    for (std::size_t index = begin; index < history.size; ++index) {
        if (!std::isfinite(history.values[index])
            || history.values[index] > maximum) {
            return false;
        }
    }
    return true;
}


bool stability_reached(
    const OptimizationMeasurements& measurements,
    ScalarHistoryView energy_changes,
    ScalarHistoryView displacements,
    ScalarHistoryView rms_gradients,
    ScalarHistoryView maximum_gradients,
    const StoppingCriteria& criteria
) noexcept {
    return optimization_state_is_usable(measurements)
        && recent_values_at_most(
            energy_changes,
            criteria.window,
            criteria.maximum_energy_change_kj_mol
        )
        && recent_values_at_most(
            displacements,
            criteria.window,
            criteria.maximum_atom_displacement_angstrom
        )
        && recent_values_at_most(
            rms_gradients,
            criteria.window,
            criteria.maximum_rms_gradient_kj_mol_angstrom
        )
        && recent_values_at_most(
            maximum_gradients,
            criteria.window,
            criteria.maximum_gradient_kj_mol_angstrom
        );
}


}  // namespace hotpot::obwrappers
