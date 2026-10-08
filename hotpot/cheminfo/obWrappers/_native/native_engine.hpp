#pragma once

#include "molecule_data.hpp"
#include "optimization_checks.hpp"
#include "optimization_controller.hpp"
#include "optimization_operation.hpp"
#include "rules.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


namespace OpenBabel {


class OBMol;


}  // namespace OpenBabel


namespace hotpot::obwrappers {


namespace detail {


inline bool backend_stop_is_converged(
    bool backend_stopped,
    double maximum_gradient_kj_mol_angstrom,
    double energy_unit_to_kj
) noexcept {
    return hotpot::obwrappers::backend_stop_is_converged(
        backend_stopped,
        maximum_gradient_kj_mol_angstrom,
        energy_unit_to_kj
    );
}


}  // namespace detail


std::recursive_mutex& openbabel_runtime_mutex();


class ForceFieldEnergyUnitFailure : public std::runtime_error {
public:
    ForceFieldEnergyUnitFailure(
        std::string forcefield,
        std::string unit
    );

    const std::string& forcefield() const noexcept;
    const std::string& unit() const noexcept;

private:
    std::string forcefield_;
    std::string unit_;
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


struct OptimizationCheckResult {
    std::vector<Coordinate> evaluated_coordinates;
    OptimizationMeasurements measurements;
    OptimizationFailure failure;
    std::string backend_energy_unit;
    RulePlan rules;
};


struct RuntimeInfo {
    std::string compiled_openbabel_version;
    std::string runtime_openbabel_version;
    int cxx11_abi;
    std::string openbabel_library_path;
    std::string babel_libdir;
    std::string babel_datadir;
};


RuntimeInfo runtime_info();

void seed_random(std::uint32_t seed);


RulePlan inspect_rules(
    const MoleculeData& molecule,
    RuleStage stage,
    double singularity_threshold,
    double repair_angle_radians
);


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


SingleOptimizationResult single_optimize_in_place(
    OpenBabel::OBMol& molecule,
    const std::string& forcefield,
    std::size_t steps,
    double singularity_threshold,
    double repair_angle_radians
);


OptimizationCheckResult check_optimization_state(
    const MoleculeData& molecule,
    const std::string& forcefield,
    const std::optional<std::vector<Coordinate>>& previous_coordinates,
    std::optional<double> previous_energy_kj_mol,
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


OptimizationResult optimize_in_place(
    OpenBabel::OBMol& molecule,
    const OptimizationOptions& options,
    const std::vector<std::vector<Coordinate>>& perturbation_offsets,
    double singularity_threshold,
    double repair_angle_radians
);


}  // namespace hotpot::obwrappers
