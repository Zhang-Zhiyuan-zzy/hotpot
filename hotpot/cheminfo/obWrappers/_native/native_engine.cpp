#include "native_engine.hpp"

#include "openbabel_adapter.hpp"
#include "optimization_operation.hpp"
#include "registry.hpp"

#include <openbabel/builder.h>
#include <openbabel/babelconfig.h>
#include <openbabel/base.h>
#include <openbabel/forcefield.h>
#include <openbabel/math/vector3.h>
#include <openbabel/plugin.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace hotpot::obwrappers {


namespace detail {


bool backend_stop_is_converged(
    bool backend_stopped,
    double maximum_gradient_kj_mol_angstrom,
    double energy_unit_to_kj
) noexcept {
    // Open Babel uses 0.1 in backend energy units per angstrom as its
    // gradient target.  Its 3.1/3.2 optimizers accumulate the minimum atom
    // gradient internally, so verify the intended maximum independently.
    constexpr double openbabel_gradient_threshold = 0.1;
    return backend_stopped
        && std::isfinite(maximum_gradient_kj_mol_angstrom)
        && maximum_gradient_kj_mol_angstrom
            <= openbabel_gradient_threshold * energy_unit_to_kj;
}


}  // namespace detail


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


OptimizationFrameFailure::OptimizationFrameFailure(std::string forcefield) :
    std::runtime_error(
        "Open Babel force field " + forcefield
        + " completed without producing an optimization frame"
    ),
    forcefield_(std::move(forcefield)) {}


const std::string& OptimizationFrameFailure::forcefield() const noexcept {
    return forcefield_;
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


void append_rule_plan(RulePlan& destination, const RulePlan& source) {
    destination.applications.insert(
        destination.applications.end(),
        source.applications.begin(),
        source.applications.end()
    );
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


double forcefield_energy_kj(
    OpenBabel::OBForceField& forcefield,
    const std::string& forcefield_name,
    bool calculate_gradients = true
) {
    return forcefield.Energy(calculate_gradients)
        * energy_factor_to_kj(forcefield_name, forcefield.GetUnit());
}


std::pair<double, double> gradient_metrics(
    OpenBabel::OBForceField& forcefield,
    OpenBabel::OBMol& molecule,
    double factor
) {
    double squared_norm_sum = 0.0;
    double maximum_norm = 0.0;
    for (unsigned int index = 1; index <= molecule.NumAtoms(); ++index) {
        const auto gradient = forcefield.GetGradient(
            molecule.GetAtom(static_cast<int>(index))
        );
        const double x = gradient.GetX() * factor;
        const double y = gradient.GetY() * factor;
        const double z = gradient.GetZ() * factor;
        const double squared_norm = x * x + y * y + z * z;
        squared_norm_sum += squared_norm;
        maximum_norm = std::max(maximum_norm, std::sqrt(squared_norm));
    }
    return {
        std::sqrt(squared_norm_sum / molecule.NumAtoms()),
        maximum_norm,
    };
}


bool finite_coordinates(const std::vector<Coordinate>& coordinates) {
    return std::all_of(
        coordinates.begin(),
        coordinates.end(),
        [](const Coordinate& coordinate) {
            return std::all_of(
                coordinate.begin(),
                coordinate.end(),
                [](double value) { return std::isfinite(value); }
            );
        }
    );
}


double maximum_displacement(
    const std::vector<Coordinate>& current,
    const std::vector<Coordinate>& previous
) {
    double maximum = 0.0;
    for (std::size_t index = 0; index < current.size(); ++index) {
        const double x = current[index][0] - previous[index][0];
        const double y = current[index][1] - previous[index][1];
        const double z = current[index][2] - previous[index][2];
        maximum = std::max(maximum, std::sqrt(x * x + y * y + z * z));
    }
    return maximum;
}


bool frame_is_usable(const OptimizationFrame& frame) {
    return finite_coordinates(frame.coordinates)
        && std::isfinite(frame.energy)
        && std::isfinite(frame.rms_gradient)
        && std::isfinite(frame.max_gradient)
        && !frame.exploded;
}


std::optional<std::string> frame_failure_reason(
    const OptimizationFrame& frame
) {
    if (!finite_coordinates(frame.coordinates)) {
        return "nonfinite_coordinates";
    }
    if (!std::isfinite(frame.energy)) {
        return "nonfinite_energy";
    }
    if (!std::isfinite(frame.rms_gradient)
        || !std::isfinite(frame.max_gradient)) {
        return "nonfinite_gradients";
    }
    if (frame.exploded) {
        return "explosion_detected";
    }
    return std::nullopt;
}


bool recent_values_below(
    const std::vector<double>& values,
    std::size_t window,
    double maximum
) {
    if (values.size() < window) {
        return false;
    }
    return std::all_of(
        values.end() - static_cast<std::ptrdiff_t>(window),
        values.end(),
        [maximum](double value) {
            return std::isfinite(value) && value <= maximum;
        }
    );
}


bool stability_reached(
    const OptimizationFrame& frame,
    const std::vector<double>& energy_changes,
    const std::vector<double>& displacements,
    const std::vector<double>& rms_gradients,
    const std::vector<double>& max_gradients,
    const StoppingCriteria& criteria
) {
    return frame_is_usable(frame)
        && recent_values_below(
            energy_changes,
            criteria.window,
            criteria.maximum_energy_change_kj_mol
        )
        && recent_values_below(
            displacements,
            criteria.window,
            criteria.maximum_atom_displacement_angstrom
        )
        && recent_values_below(
            rms_gradients,
            criteria.window,
            criteria.maximum_rms_gradient_kj_mol_angstrom
        )
        && recent_values_below(
            max_gradients,
            criteria.window,
            criteria.maximum_gradient_kj_mol_angstrom
        );
}


RulePlan setup_forcefield(
    OptimizationOperation& operation,
    OpenBabel::OBMol& molecule,
    bool update_pairs,
    double singularity_threshold,
    double repair_angle_radians
) {
    auto plan = operation.setup(
        molecule,
        update_pairs,
        singularity_threshold,
        repair_angle_radians
    );
    auto& forcefield = operation.forcefield();
    if (!plan.applications.empty()) {
        const auto energy = forcefield.Energy(true);
        bool finite_gradients = true;
        for (unsigned int index = 1; index <= molecule.NumAtoms(); ++index) {
            const auto gradient = forcefield.GetGradient(
                molecule.GetAtom(static_cast<int>(index))
            );
            finite_gradients = finite_gradients
                && std::isfinite(gradient.GetX())
                && std::isfinite(gradient.GetY())
                && std::isfinite(gradient.GetZ());
        }
        if (!std::isfinite(energy) || !finite_gradients) {
            throw ForceFieldSetupFailure(
                operation.forcefield_name(),
                "preflight-validation",
                "Open Babel retained a non-finite force-field state after "
                "registered coordinate preparation"
            );
        }
    }
    return plan;
}


void validate_options(
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
            || !finite_coordinates(offsets)) {
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
    auto plan = setup_forcefield(
        operation,
        molecule,
        false,
        singularity_threshold,
        repair_angle_radians
    );
    operation.run_steepest_descent(steps);
    operation.synchronize_coordinates(molecule);
    SingleOptimizationResult result{
        extract_coordinates(molecule),
        forcefield_energy_kj(forcefield, forcefield_name),
        forcefield.GetUnit(),
        forcefield.DetectExplosion(),
        std::move(plan),
    };
    rollback.commit();
    return result;
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
    validate_options(
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
    RulePlan all_rules{RuleStage::PRE_FORCEFIELD_SETUP, {}};

    if (options.increasing_vdw) {
        operation.set_vdw_cutoff(options.vdw_cutoff_end);
    } else {
        operation.disable_cutoff();
    }
    auto setup_plan = setup_forcefield(
        operation,
        molecule,
        options.increasing_vdw,
        singularity_threshold,
        repair_angle_radians
    );
    append_rule_plan(all_rules, setup_plan);

    const auto total_steps = options.epochs * options.steps_per_epoch;
    const auto backend_unit = forcefield.GetUnit();
    const double factor = energy_factor_to_kj(
        options.forcefield,
        backend_unit
    );
    if (options.increasing_vdw) {
        const double first_cutoff = options.vdw_cutoff_start
            + (options.vdw_cutoff_end - options.vdw_cutoff_start)
                / options.epochs;
        operation.set_vdw_cutoff(first_cutoff);
        setup_plan = setup_forcefield(
            operation,
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
        rollback.coordinates(),
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
            setup_plan = setup_forcefield(
                operation,
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
            setup_plan = setup_forcefield(
                operation,
                molecule,
                true,
                singularity_threshold,
                repair_angle_radians
            );
            append_rule_plan(all_rules, setup_plan);
        }

        auto coordinates = extract_coordinates(molecule);
        const double energy = forcefield_energy_kj(
            forcefield,
            options.forcefield
        );
        const auto gradients = gradient_metrics(forcefield, molecule, factor);
        const bool backend_converged = detail::backend_stop_is_converged(
            backend_stopped,
            gradients.second,
            factor
        );
        const bool reported_converged = backend_converged
            && (!options.increasing_vdw || epoch == options.epochs - 1);
        terminal_converged = reported_converged;
        termination_reason = reported_converged
            ? "converged"
            : "budget_exhausted";
        std::optional<double> energy_change;
        std::optional<double> displacement;
        if (previous_energy.has_value()) {
            energy_change = std::abs(energy - *previous_energy);
            energy_change_segments[segment_index].push_back(*energy_change);
        }
        if (previous_coordinates.has_value()) {
            displacement = maximum_displacement(
                coordinates, *previous_coordinates
            );
            displacement_segments[segment_index].push_back(*displacement);
        }
        OptimizationFrame frame{
            std::move(coordinates),
            energy,
            gradients.first,
            gradients.second,
            forcefield.DetectExplosion(),
            reported_converged,
            epoch,
            segment_epochs_completed,
            segment_index,
            energy_change,
            displacement,
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
                frame,
                energy_change_segments[segment_index],
                displacement_segments[segment_index],
                rms_gradient_segments[segment_index],
                max_gradient_segments[segment_index],
                *options.stopping_criteria
            );
        const long observed_epoch = static_cast<long>(epochs_completed - 1);
        if (finite_coordinates(frame.coordinates)) {
            latest_returnable_frame = frame;
            latest_returnable_epoch = observed_epoch;
        }
        if (frame_is_usable(frame)
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
        const auto numerical_failure = frame_failure_reason(*last_frame);
        if (numerical_failure.has_value()) {
            segment_active = false;
            terminal_converged = false;
            termination_reason = *numerical_failure;
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
            // of its current step budget.  A false stop with a large global
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
        backend_unit,
        termination_reason,
        terminal_converged,
        selected_energy_changes,
        selected_displacements,
        std::move(epoch_energies),
        std::move(all_rules),
    };
    set_coordinates(molecule, result.coordinates);
    rollback.commit();
    return result;
}


}  // namespace hotpot::obwrappers
