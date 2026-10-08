#pragma once

#include "rules.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>


namespace OpenBabel {


class OBForceField;
class OBMol;


}  // namespace OpenBabel


namespace hotpot::obwrappers {


enum class OptimizationAlgorithm {
    CONJUGATE,
    STEEPEST,
};


OptimizationAlgorithm optimization_algorithm(const std::string& name);


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


class OptimizationOperation final {
public:
    OptimizationOperation(
        OpenBabel::OBForceField& forcefield,
        std::string forcefield_name,
        OptimizationAlgorithm algorithm,
        double energy_tolerance
    );

    RulePlan setup(
        OpenBabel::OBMol& molecule,
        bool update_pairs,
        double singularity_threshold,
        double repair_angle_radians
    );

    RulePlan setup_and_validate(
        OpenBabel::OBMol& molecule,
        bool update_pairs,
        double singularity_threshold,
        double repair_angle_radians
    );

    void disable_cutoff();
    void set_vdw_cutoff(double cutoff);

    std::size_t initialize(std::size_t steps);
    bool take_steps(std::size_t steps);
    void run_steepest_descent(std::size_t steps);
    void synchronize_coordinates(OpenBabel::OBMol& molecule);

    OpenBabel::OBForceField& forcefield() noexcept;
    const std::string& forcefield_name() const noexcept;
    OptimizationAlgorithm algorithm() const noexcept;
    double energy_tolerance() const noexcept;

private:
    OpenBabel::OBForceField& forcefield_;
    std::string forcefield_name_;
    OptimizationAlgorithm algorithm_;
    double energy_tolerance_;
};


}  // namespace hotpot::obwrappers
