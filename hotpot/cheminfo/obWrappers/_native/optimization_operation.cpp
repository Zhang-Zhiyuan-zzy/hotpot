#include "optimization_operation.hpp"

#include "openbabel_adapter.hpp"
#include "optimization_checks.hpp"
#include "registry.hpp"

#include <openbabel/forcefield.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <utility>


namespace hotpot::obwrappers {


namespace {


std::string uppercase(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::toupper(character));
        }
    );
    return value;
}


void apply_coordinate_changes(
    OpenBabel::OBMol& molecule,
    const RulePlan& plan
) {
    for (const auto& application : plan.applications) {
        for (const auto& change : application.coordinate_changes) {
            molecule.GetAtom(
                static_cast<int>(change.atom_index + 1)
            )->SetVector(change.after[0], change.after[1], change.after[2]);
        }
    }
}


RulePlan prepare_optimization(
    OpenBabel::OBMol& molecule,
    const std::string& forcefield_name,
    double singularity_threshold,
    double repair_angle_radians
) {
    RulePlan plan{RuleStage::PRE_FORCEFIELD_SETUP, {}};
    if (uppercase(forcefield_name) != "UFF") {
        return plan;
    }
    auto snapshot = snapshot_obmol(molecule, true);
    plan = plan_optimization(
        std::move(snapshot.atoms),
        std::move(snapshot.bonds),
        std::move(snapshot.coordinates),
        singularity_threshold,
        repair_angle_radians
    );
    apply_coordinate_changes(molecule, plan);
    return plan;
}


}  // namespace


ForceFieldSetupFailure::ForceFieldSetupFailure(
    std::string forcefield,
    std::string stage,
    std::string message
) :
    std::runtime_error(std::move(message)),
    forcefield_(std::move(forcefield)),
    stage_(std::move(stage)) {}


const std::string& ForceFieldSetupFailure::forcefield() const noexcept {
    return forcefield_;
}


const std::string& ForceFieldSetupFailure::stage() const noexcept {
    return stage_;
}


OptimizationAlgorithm optimization_algorithm(const std::string& name) {
    if (name == "conjugate") {
        return OptimizationAlgorithm::CONJUGATE;
    }
    if (name == "steepest") {
        return OptimizationAlgorithm::STEEPEST;
    }
    throw std::invalid_argument(
        "algorithm must be 'conjugate' or 'steepest'"
    );
}


OptimizationOperation::OptimizationOperation(
    OpenBabel::OBForceField& forcefield,
    std::string forcefield_name,
    OptimizationAlgorithm algorithm,
    double energy_tolerance
) :
    forcefield_(forcefield),
    forcefield_name_(std::move(forcefield_name)),
    algorithm_(algorithm),
    energy_tolerance_(energy_tolerance) {}


RulePlan OptimizationOperation::setup(
    OpenBabel::OBMol& molecule,
    bool update_pairs,
    double singularity_threshold,
    double repair_angle_radians
) {
    auto plan = prepare_optimization(
        molecule,
        forcefield_name_,
        singularity_threshold,
        repair_angle_radians
    );
    OpenBabel::OBFFConstraints constraints;
    if (!forcefield_.Setup(molecule, constraints)) {
        throw ForceFieldSetupFailure(
            forcefield_name_,
            "setup",
            "Open Babel could not initialize force field " + forcefield_name_
        );
    }
    if (update_pairs) {
        forcefield_.UpdatePairsSimple();
    }
    return plan;
}


RulePlan OptimizationOperation::setup_and_validate(
    OpenBabel::OBMol& molecule,
    bool update_pairs,
    double singularity_threshold,
    double repair_angle_radians
) {
    auto plan = setup(
        molecule,
        update_pairs,
        singularity_threshold,
        repair_angle_radians
    );
    if (!plan.applications.empty()) {
        const auto energy = optimization_energy_kj(forcefield_, 1.0, true);
        const auto gradients = optimization_gradient_metrics(
            forcefield_, molecule, 1.0
        );
        if (!std::isfinite(energy)
            || !std::isfinite(gradients.rms_kj_mol_angstrom)
            || !std::isfinite(gradients.maximum_kj_mol_angstrom)) {
            throw ForceFieldSetupFailure(
                forcefield_name_,
                "preflight-validation",
                "Open Babel retained a non-finite force-field state after "
                "registered coordinate preparation"
            );
        }
    }
    return plan;
}


void OptimizationOperation::disable_cutoff() {
    forcefield_.EnableCutOff(false);
}


void OptimizationOperation::set_vdw_cutoff(double cutoff) {
    forcefield_.EnableCutOff(true);
    forcefield_.SetVDWCutOff(cutoff);
    forcefield_.SetElectrostaticCutOff(1.0e6);
}


std::size_t OptimizationOperation::initialize(std::size_t steps) {
    if (algorithm_ == OptimizationAlgorithm::CONJUGATE) {
        forcefield_.ConjugateGradientsInitialize(
            static_cast<int>(steps), energy_tolerance_
        );
        return 1;
    }
    forcefield_.SteepestDescentInitialize(
        static_cast<int>(steps), energy_tolerance_
    );
    return 0;
}


bool OptimizationOperation::take_steps(std::size_t steps) {
    return algorithm_ == OptimizationAlgorithm::CONJUGATE
        ? forcefield_.ConjugateGradientsTakeNSteps(static_cast<int>(steps))
        : forcefield_.SteepestDescentTakeNSteps(static_cast<int>(steps));
}


void OptimizationOperation::run_steepest_descent(std::size_t steps) {
    forcefield_.SteepestDescent(static_cast<int>(steps));
}


void OptimizationOperation::synchronize_coordinates(
    OpenBabel::OBMol& molecule
) {
    forcefield_.GetCoordinates(molecule);
}


OpenBabel::OBForceField& OptimizationOperation::forcefield() noexcept {
    return forcefield_;
}


const std::string& OptimizationOperation::forcefield_name() const noexcept {
    return forcefield_name_;
}


OptimizationAlgorithm OptimizationOperation::algorithm() const noexcept {
    return algorithm_;
}


double OptimizationOperation::energy_tolerance() const noexcept {
    return energy_tolerance_;
}


}  // namespace hotpot::obwrappers
