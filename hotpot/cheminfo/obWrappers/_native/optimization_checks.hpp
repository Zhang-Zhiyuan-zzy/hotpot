#pragma once

#include "rules.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>


namespace OpenBabel {


class OBForceField;
class OBMol;


}  // namespace OpenBabel


namespace hotpot::obwrappers {


// Optimization checks report numerical facts.  They neither advance the
// optimizer nor decide how an optimization workflow responds to those facts.


struct GradientMetrics {
    double rms_kj_mol_angstrom;
    double maximum_kj_mol_angstrom;
};


struct OptimizationMeasurements {
    double energy_kj_mol;
    GradientMetrics gradients;
    std::optional<double> energy_change_kj_mol;
    std::optional<double> maximum_displacement_angstrom;
    bool finite_coordinates;
    bool exploded;
};


enum class OptimizationFailure {
    NONE,
    NONFINITE_COORDINATES,
    NONFINITE_ENERGY,
    NONFINITE_GRADIENTS,
    EXPLOSION_DETECTED,
};


enum class ConvergenceLevel : std::uint8_t {
    OPENBABEL = 0,
    FAST = 1,
    BALANCED = 2,
    STRICT = 3,
};


struct StoppingCriteria {
    std::size_t window;
    double maximum_energy_change_kj_mol;
    double maximum_atom_displacement_angstrom;
    double maximum_rms_gradient_kj_mol_angstrom;
    double maximum_gradient_kj_mol_angstrom;
};


struct ScalarHistoryView {
    const double* values;
    std::size_t size;
};


inline ScalarHistoryView history_view(
    const std::vector<double>& values
) noexcept {
    return {values.data(), values.size()};
}


double optimization_energy_kj(
    OpenBabel::OBForceField& forcefield,
    double energy_unit_to_kj,
    bool calculate_gradients = true
);


GradientMetrics optimization_gradient_metrics(
    OpenBabel::OBForceField& forcefield,
    OpenBabel::OBMol& molecule,
    double energy_unit_to_kj
);


bool coordinates_are_finite(
    const Coordinate* coordinates,
    std::size_t count
) noexcept;


inline bool coordinates_are_finite(
    const std::vector<Coordinate>& coordinates
) noexcept {
    return coordinates_are_finite(coordinates.data(), coordinates.size());
}


double maximum_displacement(
    const Coordinate* current,
    const Coordinate* previous,
    std::size_t count
) noexcept;


inline double maximum_displacement(
    const std::vector<Coordinate>& current,
    const std::vector<Coordinate>& previous
) noexcept {
    return maximum_displacement(
        current.data(), previous.data(), current.size()
    );
}


bool forcefield_detects_explosion(
    OpenBabel::OBForceField& forcefield
) noexcept;


OptimizationMeasurements measure_optimization_state(
    OpenBabel::OBForceField& forcefield,
    OpenBabel::OBMol& molecule,
    const std::vector<Coordinate>& coordinates,
    const std::vector<Coordinate>* previous_coordinates,
    std::optional<double> previous_energy_kj_mol,
    double energy_unit_to_kj
);


OptimizationFailure optimization_failure(
    const OptimizationMeasurements& measurements
) noexcept;


const char* optimization_failure_reason(
    OptimizationFailure failure
) noexcept;


bool optimization_state_is_usable(
    const OptimizationMeasurements& measurements
) noexcept;


bool backend_stop_is_converged(
    bool backend_stopped,
    double maximum_gradient_kj_mol_angstrom,
    double energy_unit_to_kj,
    double backend_maximum_gradient = 0.1
) noexcept;


bool convergence_reached(
    bool backend_stopped,
    const OptimizationMeasurements& measurements,
    double energy_unit_to_kj,
    ConvergenceLevel level
) noexcept;


bool recent_values_at_most(
    ScalarHistoryView history,
    std::size_t window,
    double maximum
) noexcept;


bool stability_reached(
    const OptimizationMeasurements& measurements,
    ScalarHistoryView energy_changes,
    ScalarHistoryView displacements,
    ScalarHistoryView rms_gradients,
    ScalarHistoryView maximum_gradients,
    const StoppingCriteria& criteria
) noexcept;


inline bool stability_reached(
    const OptimizationMeasurements& measurements,
    const std::vector<double>& energy_changes,
    const std::vector<double>& displacements,
    const std::vector<double>& rms_gradients,
    const std::vector<double>& maximum_gradients,
    const StoppingCriteria& criteria
) noexcept {
    return stability_reached(
        measurements,
        history_view(energy_changes),
        history_view(displacements),
        history_view(rms_gradients),
        history_view(maximum_gradients),
        criteria
    );
}


}  // namespace hotpot::obwrappers
