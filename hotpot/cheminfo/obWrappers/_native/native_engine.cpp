#include "native_engine.hpp"

#include "openbabel_adapter.hpp"
#include "optimization_controller.hpp"
#include "optimization_operation.hpp"
#include "registry.hpp"

#include <openbabel/builder.h>
#include <openbabel/babelconfig.h>
#include <openbabel/base.h>
#include <openbabel/forcefield.h>
#include <openbabel/math/vector3.h>
#include <openbabel/plugin.h>

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace hotpot::obwrappers {


std::recursive_mutex& openbabel_runtime_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}


ForceFieldEnergyUnitFailure::ForceFieldEnergyUnitFailure(
    std::string forcefield,
    std::string unit
) :
    std::runtime_error(
        "unsupported energy unit '" + unit
        + "' reported by Open Babel force field " + forcefield
    ),
    forcefield_(std::move(forcefield)),
    unit_(std::move(unit)) {}


const std::string& ForceFieldEnergyUnitFailure::forcefield() const noexcept {
    return forcefield_;
}


const std::string& ForceFieldEnergyUnitFailure::unit() const noexcept {
    return unit_;
}


namespace {


std::once_flag openbabel_runtime_once;
std::string openbabel_library_path;


void set_environment(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}


void ensure_openbabel_runtime() {
    std::call_once(openbabel_runtime_once, []() {
        Dl_info library_info{};
        if (dladdr(
                reinterpret_cast<void*>(OpenBabel::OBReleaseVersion),
                &library_info
            ) != 0
            && library_info.dli_fname != nullptr) {
            const auto library = std::filesystem::canonical(
                library_info.dli_fname
            );
            openbabel_library_path = library.string();
            const auto library_dir = library.parent_path();
            const auto version = OpenBabel::OBReleaseVersion();
            const auto plugin_dir = library_dir / "openbabel" / version;
            const auto data_dir = library_dir.parent_path()
                / "share" / "openbabel" / version;
            if (std::filesystem::is_directory(plugin_dir)) {
                set_environment("BABEL_LIBDIR", plugin_dir.string());
            }
            if (std::filesystem::is_directory(data_dir)) {
                set_environment("BABEL_DATADIR", data_dir.string());
            }
        }
        OpenBabel::OBPlugin::LoadAllPlugins();
    });
}


class HybridizationGuard {
public:
    HybridizationGuard(OpenBabel::OBMol& molecule, const RulePlan& plan) :
        molecule_(molecule),
        plan_(plan) {
        for (const auto& application : plan_.applications) {
            for (const auto& change : application.hybridization_changes) {
                molecule_.GetAtom(
                    static_cast<int>(change.atom_index + 1)
                )->SetHyb(change.after);
            }
        }
    }

    ~HybridizationGuard() {
        for (auto application = plan_.applications.rbegin();
             application != plan_.applications.rend();
             ++application) {
            for (auto change = application->hybridization_changes.rbegin();
                 change != application->hybridization_changes.rend();
                 ++change) {
                molecule_.GetAtom(
                    static_cast<int>(change->atom_index + 1)
                )->SetHyb(change->before);
            }
        }
    }

    HybridizationGuard(const HybridizationGuard&) = delete;
    HybridizationGuard& operator=(const HybridizationGuard&) = delete;

private:
    OpenBabel::OBMol& molecule_;
    const RulePlan& plan_;
};


class CoordinateRollbackGuard {
public:
    explicit CoordinateRollbackGuard(OpenBabel::OBMol& molecule) :
        molecule_(molecule),
        coordinates_(extract_coordinates(molecule)) {}

    ~CoordinateRollbackGuard() noexcept {
        if (!committed_) {
            set_coordinates(molecule_, coordinates_);
        }
    }

    CoordinateRollbackGuard(const CoordinateRollbackGuard&) = delete;
    CoordinateRollbackGuard& operator=(const CoordinateRollbackGuard&) =
        delete;

    const std::vector<Coordinate>& coordinates() const noexcept {
        return coordinates_;
    }

    void commit() noexcept {
        committed_ = true;
    }

private:
    OpenBabel::OBMol& molecule_;
    std::vector<Coordinate> coordinates_;
    bool committed_ = false;
};


OpenBabel::OBForceField& find_forcefield(const std::string& name) {
    ensure_openbabel_runtime();
    auto* forcefield = static_cast<OpenBabel::OBForceField*>(
        OpenBabel::OBPlugin::GetPlugin("forcefields", name.c_str())
    );
    if (forcefield == nullptr) {
        throw ForceFieldSetupFailure(
            name,
            "lookup",
            "unknown Open Babel force field: " + name
        );
    }
    return *forcefield;
}


double energy_factor_to_kj(
    const std::string& forcefield,
    const std::string& unit
) {
    std::string normalized;
    normalized.reserve(unit.size());
    for (const auto character : unit) {
        if (!std::isspace(static_cast<unsigned char>(character))) {
            normalized.push_back(
                static_cast<char>(
                    std::tolower(static_cast<unsigned char>(character))
                )
            );
        }
    }
    if (normalized == "kj/mol"
        || normalized == "kjmol-1"
        || normalized == "kjmol^-1") {
        return 1.0;
    }
    if (normalized == "kcal/mol"
        || normalized == "kcalmol-1"
        || normalized == "kcalmol^-1") {
        return 4.184;
    }
    throw ForceFieldEnergyUnitFailure(forcefield, unit);
}


}  // namespace


RuntimeInfo runtime_info() {
    ensure_openbabel_runtime();
    const auto environment_value = [](const char* name) {
        const auto* value = std::getenv(name);
        return value == nullptr ? std::string{} : std::string(value);
    };
    return RuntimeInfo{
        BABEL_VERSION,
        OpenBabel::OBReleaseVersion(),
        _GLIBCXX_USE_CXX11_ABI,
        openbabel_library_path,
        environment_value("BABEL_LIBDIR"),
        environment_value("BABEL_DATADIR"),
    };
}


void seed_random(std::uint32_t seed) {
    std::lock_guard<std::recursive_mutex> lock(openbabel_runtime_mutex());
    const auto text = std::to_string(seed);
    set_environment("OB_RANDOM_SEED", text);
#if OB_VERSION < OB_VERSION_CHECK(3, 2, 0)
    OpenBabel::vector3 probe;
    probe.randomUnitVector();
    std::srand(seed);
#endif
}


RulePlan inspect_rules(
    const MoleculeData& molecule,
    RuleStage stage,
    double singularity_threshold,
    double repair_angle_radians
) {
    std::lock_guard<std::recursive_mutex> lock(openbabel_runtime_mutex());
    molecule.validate();
    validate_rule_parameters(
        singularity_threshold,
        repair_angle_radians
    );
    auto obmol = make_obmol(molecule);
    auto snapshot = snapshot_obmol(
        obmol,
        stage == RuleStage::PRE_FORCEFIELD_SETUP
    );
    if (stage == RuleStage::PRE_BUILD) {
        return plan_build(
            std::move(snapshot.atoms),
            std::move(snapshot.bonds)
        );
    }
    return plan_optimization(
        std::move(snapshot.atoms),
        std::move(snapshot.bonds),
        std::move(snapshot.coordinates),
        singularity_threshold,
        repair_angle_radians
    );
}


BuildResult build(
    const MoleculeData& molecule,
    std::optional<bool> stereo_warnings
) {
    std::lock_guard<std::recursive_mutex> lock(openbabel_runtime_mutex());
    molecule.validate();
    ensure_openbabel_runtime();
    auto obmol = make_obmol(molecule);
    auto plan = rule_registry().execute(
        RuleStage::PRE_BUILD,
        snapshot_obmol(obmol, false),
        RuleParameters{}
    );
    HybridizationGuard hybridization_guard(obmol, plan);
    OpenBabel::OBBuilder builder;
    const bool succeeded = stereo_warnings.has_value()
        ? builder.Build(obmol, *stereo_warnings)
        : builder.Build(obmol);
    return BuildResult{succeeded, extract_coordinates(obmol), std::move(plan)};
}


SingleOptimizationResult single_optimize(
    const MoleculeData& molecule,
    const std::string& forcefield_name,
    std::size_t steps,
    double singularity_threshold,
    double repair_angle_radians
) {
    std::lock_guard<std::recursive_mutex> lock(openbabel_runtime_mutex());
    molecule.validate();
    auto obmol = make_obmol(molecule);
    return single_optimize_in_place(
        obmol,
        forcefield_name,
        steps,
        singularity_threshold,
        repair_angle_radians
    );
}


SingleOptimizationResult single_optimize_in_place(
    OpenBabel::OBMol& molecule,
    const std::string& forcefield_name,
    std::size_t steps,
    double singularity_threshold,
    double repair_angle_radians
) {
    std::lock_guard<std::recursive_mutex> lock(openbabel_runtime_mutex());
    validate_rule_parameters(
        singularity_threshold,
        repair_angle_radians
    );
    if (forcefield_name.empty()) {
        throw std::invalid_argument("forcefield must not be empty");
    }
    if (steps < 1
        || steps > static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )) {
        throw std::invalid_argument(
            "steps must be positive and fit the Open Babel step limit"
        );
    }
    CoordinateRollbackGuard rollback(molecule);
    auto& forcefield = find_forcefield(forcefield_name);
    OptimizationOperation operation(
        forcefield,
        forcefield_name,
        OptimizationAlgorithm::STEEPEST,
        0.0
    );
    operation.disable_cutoff();
    auto plan = operation.setup_and_validate(
        molecule,
        false,
        singularity_threshold,
        repair_angle_radians
    );
    operation.run_steepest_descent(steps);
    operation.synchronize_coordinates(molecule);
    SingleOptimizationResult result{
        extract_coordinates(molecule),
        optimization_energy_kj(
            forcefield,
            energy_factor_to_kj(forcefield_name, forcefield.GetUnit())
        ),
        forcefield.GetUnit(),
        forcefield.DetectExplosion(),
        std::move(plan),
    };
    rollback.commit();
    return result;
}


OptimizationCheckResult check_optimization_state(
    const MoleculeData& molecule,
    const std::string& forcefield_name,
    const std::optional<std::vector<Coordinate>>& previous_coordinates,
    std::optional<double> previous_energy_kj_mol,
    double singularity_threshold,
    double repair_angle_radians
) {
    std::lock_guard<std::recursive_mutex> lock(openbabel_runtime_mutex());
    molecule.validate();
    validate_rule_parameters(
        singularity_threshold,
        repair_angle_radians
    );
    if (previous_coordinates.has_value()
        && previous_coordinates->size() != molecule.atom_count()) {
        throw std::invalid_argument(
            "previous_coordinates must contain one coordinate per atom"
        );
    }
    auto obmol = make_obmol(molecule);
    auto& forcefield = find_forcefield(forcefield_name);
    OptimizationOperation operation(
        forcefield,
        forcefield_name,
        OptimizationAlgorithm::STEEPEST,
        0.0
    );
    auto plan = operation.setup_and_validate(
        obmol,
        false,
        singularity_threshold,
        repair_angle_radians
    );
    const auto coordinates = extract_coordinates(obmol);
    const auto backend_unit = forcefield.GetUnit();
    const auto measurements = measure_optimization_state(
        forcefield,
        obmol,
        coordinates,
        previous_coordinates.has_value() ? &*previous_coordinates : nullptr,
        previous_energy_kj_mol,
        energy_factor_to_kj(forcefield_name, backend_unit)
    );
    const auto failure = optimization_failure(measurements);
    return {
        coordinates,
        measurements,
        failure,
        backend_unit,
        std::move(plan),
    };
}


OptimizationResult optimize(
    const MoleculeData& molecule,
    const OptimizationOptions& options,
    const std::vector<std::vector<Coordinate>>& perturbation_offsets,
    double singularity_threshold,
    double repair_angle_radians
) {
    molecule.validate();
    std::lock_guard<std::recursive_mutex> lock(openbabel_runtime_mutex());
    auto obmol = make_obmol(molecule);
    return optimize_in_place(
        obmol,
        options,
        perturbation_offsets,
        singularity_threshold,
        repair_angle_radians
    );
}


OptimizationResult optimize_in_place(
    OpenBabel::OBMol& molecule,
    const OptimizationOptions& options,
    const std::vector<std::vector<Coordinate>>& perturbation_offsets,
    double singularity_threshold,
    double repair_angle_radians
) {
    std::lock_guard<std::recursive_mutex> lock(openbabel_runtime_mutex());
    validate_rule_parameters(
        singularity_threshold,
        repair_angle_radians
    );
    detail::validate_optimization_options(
        molecule.NumAtoms(), options, perturbation_offsets
    );
    CoordinateRollbackGuard rollback(molecule);
    auto& forcefield = find_forcefield(options.forcefield);
    OptimizationOperation operation(
        forcefield,
        options.forcefield,
        optimization_algorithm(options.algorithm),
        options.energy_tolerance
    );
    const auto backend_unit = forcefield.GetUnit();
    const double factor = energy_factor_to_kj(
        options.forcefield,
        backend_unit
    );
    auto result = detail::run_optimization_controller(
        molecule,
        operation,
        options,
        perturbation_offsets,
        factor,
        backend_unit,
        rollback.coordinates(),
        singularity_threshold,
        repair_angle_radians
    );
    rollback.commit();
    return result;
}


}  // namespace hotpot::obwrappers
