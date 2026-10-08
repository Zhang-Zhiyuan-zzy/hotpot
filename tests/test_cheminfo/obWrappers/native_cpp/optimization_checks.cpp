#include "optimization_checks.hpp"

#include <openbabel/forcefield.h>
#include <openbabel/mol.h>
#include <openbabel/plugin.h>

#include <cassert>
#include <cmath>
#include <limits>
#include <vector>


namespace obw = hotpot::obwrappers;


namespace {


obw::OptimizationMeasurements usable_measurements() {
    return {
        -12.0,
        {0.2, 0.4},
        0.01,
        0.001,
        true,
        false,
    };
}


void test_coordinate_facts() {
    const std::vector<obw::Coordinate> previous{
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
    };
    auto current = previous;
    current[1] = {2.0, 3.0, 3.0};
    assert(obw::coordinates_are_finite(current));
    assert(std::abs(obw::maximum_displacement(current, previous) - 3.0)
        < 1.0e-12);

    current[0][1] = std::numeric_limits<double>::infinity();
    assert(!obw::coordinates_are_finite(current));
}


void test_failure_classification() {
    auto measurements = usable_measurements();
    assert(obw::optimization_state_is_usable(measurements));
    assert(obw::optimization_failure(measurements)
        == obw::OptimizationFailure::NONE);

    measurements.finite_coordinates = false;
    assert(obw::optimization_failure(measurements)
        == obw::OptimizationFailure::NONFINITE_COORDINATES);
    measurements = usable_measurements();
    measurements.energy_kj_mol =
        std::numeric_limits<double>::quiet_NaN();
    assert(obw::optimization_failure(measurements)
        == obw::OptimizationFailure::NONFINITE_ENERGY);
    measurements = usable_measurements();
    measurements.gradients.maximum_kj_mol_angstrom =
        std::numeric_limits<double>::infinity();
    assert(obw::optimization_failure(measurements)
        == obw::OptimizationFailure::NONFINITE_GRADIENTS);
    measurements = usable_measurements();
    measurements.exploded = true;
    assert(obw::optimization_failure(measurements)
        == obw::OptimizationFailure::EXPLOSION_DETECTED);
}


void test_backend_stop_semantics() {
    assert(!obw::backend_stop_is_converged(false, 0.0, 1.0));
    assert(obw::backend_stop_is_converged(true, 0.1, 1.0));
    assert(!obw::backend_stop_is_converged(true, 0.1001, 1.0));
    assert(obw::backend_stop_is_converged(true, 0.4184, 4.184));
    assert(!obw::backend_stop_is_converged(
        true,
        std::numeric_limits<double>::quiet_NaN(),
        1.0
    ));
}


void test_stability_window() {
    const std::vector<double> energy_changes{0.5, 0.02, 0.01};
    const std::vector<double> displacements{0.1, 0.002, 0.001};
    const std::vector<double> rms_gradients{4.0, 0.4, 0.2};
    const std::vector<double> maximum_gradients{8.0, 0.8, 0.4};
    const obw::StoppingCriteria criteria{
        2,
        0.02,
        0.002,
        0.4,
        0.8,
    };
    const auto measurements = usable_measurements();

    assert(obw::stability_reached(
        measurements,
        energy_changes,
        displacements,
        rms_gradients,
        maximum_gradients,
        criteria
    ));
    assert(!obw::recent_values_at_most(
        obw::history_view(energy_changes), 3, 0.02
    ));

    auto exploded = measurements;
    exploded.exploded = true;
    assert(!obw::stability_reached(
        exploded,
        energy_changes,
        displacements,
        rms_gradients,
        maximum_gradients,
        criteria
    ));
}


void test_forcefield_measurements() {
    OpenBabel::OBPlugin::LoadAllPlugins();
    OpenBabel::OBMol molecule;
    auto* first = molecule.NewAtom();
    first->SetAtomicNum(6);
    first->SetVector(0.0, 0.0, 0.0);
    auto* second = molecule.NewAtom();
    second->SetAtomicNum(6);
    second->SetVector(1.8, 0.0, 0.0);
    molecule.AddBond(1, 2, 1);

    auto* forcefield = OpenBabel::OBForceField::FindForceField("UFF");
    assert(forcefield != nullptr);
    assert(forcefield->Setup(molecule));
    const double energy = obw::optimization_energy_kj(
        *forcefield, 4.184, true
    );
    const auto gradients = obw::optimization_gradient_metrics(
        *forcefield, molecule, 4.184
    );
    assert(std::isfinite(energy));
    assert(std::isfinite(gradients.rms_kj_mol_angstrom));
    assert(std::isfinite(gradients.maximum_kj_mol_angstrom));
    assert(!obw::forcefield_detects_explosion(*forcefield));

    const std::vector<obw::Coordinate> coordinates{
        {0.0, 0.0, 0.0},
        {1.8, 0.0, 0.0},
    };
    const auto measurements = obw::measure_optimization_state(
        *forcefield,
        molecule,
        coordinates,
        nullptr,
        std::nullopt,
        4.184
    );
    assert(obw::optimization_state_is_usable(measurements));
    assert(!measurements.energy_change_kj_mol.has_value());
    assert(!measurements.maximum_displacement_angstrom.has_value());
}


}  // namespace


int main() {
    test_coordinate_facts();
    test_failure_classification();
    test_backend_stop_semantics();
    test_stability_window();
    test_forcefield_measurements();
}
