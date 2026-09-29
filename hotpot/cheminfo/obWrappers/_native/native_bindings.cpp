#include "molecule_data.hpp"
#include "native_engine.hpp"
#include "registry.hpp"
#include "../../forcefields/_native/bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


namespace py = pybind11;
using namespace hotpot::obwrappers;


namespace {


template <typename Value>
const Value* require_array(
    const py::array& array,
    const char* name,
    int dimensions
) {
    if (!array.dtype().is(py::dtype::of<Value>())) {
        throw py::type_error(
            std::string(name) + " has an unexpected dtype"
        );
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(
            std::string(name) + " must be C-contiguous"
        );
    }
    if (array.ndim() != dimensions) {
        throw py::value_error(
            std::string(name) + " has an unexpected number of dimensions"
        );
    }
    return static_cast<const Value*>(array.data());
}


template <typename Value>
std::vector<Value> read_vector(
    const py::array& array,
    const char* name
) {
    const auto* data = require_array<Value>(array, name, 1);
    return std::vector<Value>(data, data + array.shape(0));
}


std::vector<Coordinate> read_coordinates(
    const py::array& array,
    const char* name
) {
    const auto* data = require_array<double>(array, name, 2);
    if (array.shape(1) != 3) {
        throw py::value_error(std::string(name) + " must have shape (N, 3)");
    }
    std::vector<Coordinate> coordinates;
    coordinates.reserve(array.shape(0));
    for (py::ssize_t row = 0; row < array.shape(0); ++row) {
        coordinates.push_back(Coordinate{
            data[row * 3], data[row * 3 + 1], data[row * 3 + 2]
        });
    }
    return coordinates;
}


std::vector<std::array<std::int32_t, 2>> read_bond_indices(
    const py::array& array
) {
    const auto* data = require_array<std::int32_t>(
        array, "bond_indices", 2
    );
    if (array.shape(1) != 2) {
        throw py::value_error("bond_indices must have shape (M, 2)");
    }
    std::vector<std::array<std::int32_t, 2>> indices;
    indices.reserve(array.shape(0));
    for (py::ssize_t row = 0; row < array.shape(0); ++row) {
        indices.push_back({data[row * 2], data[row * 2 + 1]});
    }
    return indices;
}


std::vector<BondKind> read_bond_kinds(const py::array& array) {
    const auto values = read_vector<std::uint8_t>(array, "bond_kinds");
    std::vector<BondKind> kinds;
    kinds.reserve(values.size());
    for (const auto value : values) {
        if (value < static_cast<std::uint8_t>(BondKind::SINGLE)
            || value > static_cast<std::uint8_t>(BondKind::UNKNOWN)) {
            throw py::value_error("bond_kinds contains an unknown value");
        }
        kinds.push_back(static_cast<BondKind>(value));
    }
    return kinds;
}


std::optional<std::array<double, 6>> read_unit_cell(
    const py::object& value
) {
    if (value.is_none()) {
        return std::nullopt;
    }
    const auto array = py::cast<py::array>(value);
    const auto* data = require_array<double>(array, "unit_cell", 1);
    if (array.shape(0) != 6) {
        throw py::value_error("unit_cell must have shape (6,)");
    }
    return std::array<double, 6>{
        data[0], data[1], data[2], data[3], data[4], data[5]
    };
}


std::vector<std::vector<Coordinate>> read_perturbation_offsets(
    const py::object& value,
    std::size_t atom_count
) {
    if (value.is_none()) {
        return {};
    }
    const auto array = py::cast<py::array>(value);
    const auto* data = require_array<double>(
        array, "perturbation_offsets", 3
    );
    if (array.shape(1) != static_cast<py::ssize_t>(atom_count)
        || array.shape(2) != 3) {
        throw py::value_error(
            "perturbation_offsets must have shape (K, N, 3)"
        );
    }
    std::vector<std::vector<Coordinate>> offsets;
    offsets.reserve(array.shape(0));
    for (py::ssize_t frame = 0; frame < array.shape(0); ++frame) {
        std::vector<Coordinate> frame_offsets;
        frame_offsets.reserve(atom_count);
        for (std::size_t atom = 0; atom < atom_count; ++atom) {
            const auto base = (
                static_cast<std::size_t>(frame) * atom_count + atom
            ) * 3;
            frame_offsets.push_back(Coordinate{
                data[base], data[base + 1], data[base + 2]
            });
        }
        offsets.push_back(std::move(frame_offsets));
    }
    return offsets;
}


py::array_t<double> coordinate_array(
    const std::vector<Coordinate>& coordinates
) {
    py::array_t<double> array({coordinates.size(), std::size_t{3}});
    auto* destination = array.mutable_data();
    for (std::size_t row = 0; row < coordinates.size(); ++row) {
        std::memcpy(
            destination + row * 3,
            coordinates[row].data(),
            3 * sizeof(double)
        );
    }
    return array;
}


[[noreturn]] void raise_setup_error(
    const py::exception<ForceFieldSetupFailure>& exception_type,
    const ForceFieldSetupFailure& error
) {
    py::object instance = py::reinterpret_borrow<py::object>(
        exception_type.ptr()
    )(error.what());
    instance.attr("forcefield") = error.forcefield();
    instance.attr("stage") = error.stage();
    PyErr_SetObject(exception_type.ptr(), instance.ptr());
    throw py::error_already_set();
}


[[noreturn]] void raise_energy_unit_error(
    const py::exception<ForceFieldEnergyUnitFailure>& exception_type,
    const ForceFieldEnergyUnitFailure& error
) {
    py::object instance = py::reinterpret_borrow<py::object>(
        exception_type.ptr()
    )(error.what());
    instance.attr("forcefield") = error.forcefield();
    instance.attr("unit") = error.unit();
    PyErr_SetObject(exception_type.ptr(), instance.ptr());
    throw py::error_already_set();
}


[[noreturn]] void raise_frame_error(
    const py::exception<OptimizationFrameFailure>& exception_type,
    const OptimizationFrameFailure& error
) {
    py::object instance = py::reinterpret_borrow<py::object>(
        exception_type.ptr()
    )(error.what());
    instance.attr("forcefield") = error.forcefield();
    PyErr_SetObject(exception_type.ptr(), instance.ptr());
    throw py::error_already_set();
}


void bind_rule_contracts(py::module_& module) {
    py::enum_<RuleStage>(module, "RuleStage", py::module_local())
        .value("PRE_BUILD", RuleStage::PRE_BUILD)
        .value("PRE_FORCEFIELD_SETUP", RuleStage::PRE_FORCEFIELD_SETUP)
        .export_values();
    py::class_<RuleDescriptor>(module, "RuleDescriptor", py::module_local())
        .def_readonly("rule_id", &RuleDescriptor::rule_id)
        .def_readonly("version", &RuleDescriptor::version)
        .def_readonly("stage", &RuleDescriptor::stage)
        .def_readonly("priority", &RuleDescriptor::priority);
    py::class_<HybridizationChange>(
        module, "HybridizationChange", py::module_local()
    )
        .def_readonly("atom_index", &HybridizationChange::atom_index)
        .def_readonly("before", &HybridizationChange::before)
        .def_readonly("after", &HybridizationChange::after);
    py::class_<CoordinateChange>(
        module, "CoordinateChange", py::module_local()
    )
        .def_readonly("atom_index", &CoordinateChange::atom_index)
        .def_readonly("before", &CoordinateChange::before)
        .def_readonly("after", &CoordinateChange::after);
    py::class_<RuleApplication>(
        module, "RuleApplication", py::module_local()
    )
        .def_readonly("rule_id", &RuleApplication::rule_id)
        .def_readonly("version", &RuleApplication::version)
        .def_readonly("stage", &RuleApplication::stage)
        .def_readonly("priority", &RuleApplication::priority)
        .def_readonly("atom_indices", &RuleApplication::atom_indices)
        .def_readonly("metric_before", &RuleApplication::metric_before)
        .def_readonly(
            "hybridization_changes",
            &RuleApplication::hybridization_changes
        )
        .def_readonly("coordinate_changes", &RuleApplication::coordinate_changes);
    py::class_<RulePlan>(module, "RulePlan", py::module_local())
        .def_readonly("stage", &RulePlan::stage)
        .def_readonly("applications", &RulePlan::applications);
}


}  // namespace


PYBIND11_MODULE(_ob_native, module) {
    module.doc() = "Direct Open Babel C++ force-field backend for Hotpot";
    bind_rule_contracts(module);
    hotpot::forcefields::bind_native_forcefield_contracts(module);
    py::exception<ForceFieldSetupFailure> setup_error(
        module, "ForceFieldSetupError", PyExc_RuntimeError
    );
    py::exception<ForceFieldEnergyUnitFailure> energy_unit_error(
        module, "ForceFieldEnergyUnitError", PyExc_RuntimeError
    );
    py::exception<OptimizationFrameFailure> frame_error(
        module, "OptimizationFrameError", PyExc_RuntimeError
    );

    py::enum_<BondKind>(module, "BondKind")
        .value("SINGLE", BondKind::SINGLE)
        .value("DOUBLE", BondKind::DOUBLE)
        .value("TRIPLE", BondKind::TRIPLE)
        .value("AROMATIC", BondKind::AROMATIC)
        .value("ZERO", BondKind::ZERO)
        .value("DATIVE", BondKind::DATIVE)
        .value("UNKNOWN", BondKind::UNKNOWN)
        .export_values();

    py::class_<MoleculeData>(module, "MoleculeData")
        .def(
            py::init([](
                std::int32_t schema_version,
                const py::array& atomic_numbers,
                const py::array& formal_charges,
                const py::array& partial_charges,
                const py::array& coordinates,
                const py::array& atom_aromatic,
                const py::array& bond_indices,
                const py::array& bond_orders,
                const py::array& bond_kinds,
                const py::array& bond_aromatic,
                const py::object& unit_cell
            ) {
                MoleculeData molecule{
                    schema_version,
                    read_vector<std::int32_t>(
                        atomic_numbers, "atomic_numbers"
                    ),
                    read_vector<std::int32_t>(
                        formal_charges, "formal_charges"
                    ),
                    read_vector<double>(partial_charges, "partial_charges"),
                    read_coordinates(coordinates, "coordinates"),
                    read_vector<std::uint8_t>(
                        atom_aromatic, "atom_aromatic"
                    ),
                    read_bond_indices(bond_indices),
                    read_vector<double>(bond_orders, "bond_orders"),
                    read_bond_kinds(bond_kinds),
                    read_vector<std::uint8_t>(
                        bond_aromatic, "bond_aromatic"
                    ),
                    read_unit_cell(unit_cell),
                };
                molecule.validate();
                return molecule;
            }),
            py::arg("schema_version"),
            py::arg("atomic_numbers"),
            py::arg("formal_charges"),
            py::arg("partial_charges"),
            py::arg("coordinates"),
            py::arg("atom_aromatic"),
            py::arg("bond_indices"),
            py::arg("bond_orders"),
            py::arg("bond_kinds"),
            py::arg("bond_aromatic"),
            py::arg("unit_cell") = py::none()
        )
        .def_property_readonly("atom_count", &MoleculeData::atom_count)
        .def_property_readonly("bond_count", &MoleculeData::bond_count);

    py::class_<BuildResult>(module, "BuildResult")
        .def_readonly("succeeded", &BuildResult::succeeded)
        .def_property_readonly(
            "coordinates",
            [](const BuildResult& result) {
                return coordinate_array(result.coordinates);
            }
        )
        .def_readonly("rules", &BuildResult::rules);

    py::class_<SingleOptimizationResult>(module, "SingleOptimizationResult")
        .def_property_readonly(
            "coordinates",
            [](const SingleOptimizationResult& result) {
                return coordinate_array(result.coordinates);
            }
        )
        .def_readonly("energy_kj_mol", &SingleOptimizationResult::energy_kj_mol)
        .def_readonly(
            "backend_energy_unit",
            &SingleOptimizationResult::backend_energy_unit
        )
        .def_readonly("exploded", &SingleOptimizationResult::exploded)
        .def_readonly("rules", &SingleOptimizationResult::rules);

    py::class_<OptimizationFrame>(module, "OptimizationFrame")
        .def_property_readonly(
            "coordinates",
            [](const OptimizationFrame& frame) {
                return coordinate_array(frame.coordinates);
            }
        )
        .def_readonly("energy", &OptimizationFrame::energy)
        .def_readonly("rms_gradient", &OptimizationFrame::rms_gradient)
        .def_readonly("max_gradient", &OptimizationFrame::max_gradient)
        .def_readonly("exploded", &OptimizationFrame::exploded)
        .def_readonly("converged", &OptimizationFrame::converged)
        .def_readonly("epoch_index", &OptimizationFrame::epoch_index)
        .def_readonly(
            "segment_epochs_completed",
            &OptimizationFrame::segment_epochs_completed
        )
        .def_readonly("segment_index", &OptimizationFrame::segment_index)
        .def_readonly("energy_change", &OptimizationFrame::energy_change)
        .def_readonly("max_displacement", &OptimizationFrame::max_displacement);

    py::class_<OptimizationResult>(module, "OptimizationResult")
        .def_property_readonly(
            "coordinates",
            [](const OptimizationResult& result) {
                return coordinate_array(result.coordinates);
            }
        )
        .def_property_readonly(
            "terminal_coordinates",
            [](const OptimizationResult& result) {
                return coordinate_array(result.terminal_coordinates);
            }
        )
        .def_readonly("frames", &OptimizationResult::frames)
        .def_readonly(
            "selected_frame_index",
            &OptimizationResult::selected_frame_index
        )
        .def_readonly("best_epoch", &OptimizationResult::best_epoch)
        .def_readonly("final_energy", &OptimizationResult::final_energy)
        .def_readonly("best_energy", &OptimizationResult::best_energy)
        .def_readonly("rms_gradient", &OptimizationResult::rms_gradient)
        .def_readonly("max_gradient", &OptimizationResult::max_gradient)
        .def_readonly("exploded", &OptimizationResult::exploded)
        .def_readonly("converged", &OptimizationResult::converged)
        .def_readonly("epochs_completed", &OptimizationResult::epochs_completed)
        .def_readonly("steps_submitted", &OptimizationResult::steps_submitted)
        .def_readonly(
            "initialization_steps", &OptimizationResult::initialization_steps
        )
        .def_readonly(
            "selected_segment_epochs_completed",
            &OptimizationResult::selected_segment_epochs_completed
        )
        .def_readonly(
            "backend_energy_unit", &OptimizationResult::backend_energy_unit
        )
        .def_readonly(
            "termination_reason", &OptimizationResult::termination_reason
        )
        .def_readonly(
            "terminal_converged", &OptimizationResult::terminal_converged
        )
        .def_readonly("energy_changes", &OptimizationResult::energy_changes)
        .def_readonly(
            "max_displacements", &OptimizationResult::max_displacements
        )
        .def_readonly("epoch_energies", &OptimizationResult::epoch_energies)
        .def_readonly("rules", &OptimizationResult::rules);

    py::class_<RuntimeInfo>(module, "RuntimeInfo")
        .def_readonly(
            "compiled_openbabel_version",
            &RuntimeInfo::compiled_openbabel_version
        )
        .def_readonly(
            "runtime_openbabel_version",
            &RuntimeInfo::runtime_openbabel_version
        )
        .def_readonly("cxx11_abi", &RuntimeInfo::cxx11_abi)
        .def_readonly(
            "openbabel_library_path", &RuntimeInfo::openbabel_library_path
        )
        .def_readonly("babel_libdir", &RuntimeInfo::babel_libdir)
        .def_readonly("babel_datadir", &RuntimeInfo::babel_datadir);

    module.def("runtime_info", &runtime_info);
    module.def("seed_random", &seed_random, py::arg("seed"));
    module.def(
        "inspect_rules",
        &inspect_rules,
        py::arg("molecule"),
        py::arg("stage"),
        py::arg("singularity_threshold") = 1.0e-6,
        py::arg("repair_angle_radians") = 1.0e-3,
        py::call_guard<py::gil_scoped_release>()
    );

    module.def(
        "available_rules",
        [](std::optional<RuleStage> stage) {
            return rule_registry().descriptors(stage);
        },
        py::arg("stage") = std::nullopt,
        py::call_guard<py::gil_scoped_release>()
    );
    module.def(
        "build",
        &build,
        py::arg("molecule"),
        py::arg("stereo_warnings") = std::nullopt,
        py::call_guard<py::gil_scoped_release>()
    );
    module.def(
        "single_optimize",
        [setup_error, energy_unit_error](
            const MoleculeData& molecule,
            const std::string& forcefield,
            std::size_t steps,
            double singularity_threshold,
            double repair_angle_radians
        ) {
            try {
                py::gil_scoped_release release;
                return single_optimize(
                    molecule,
                    forcefield,
                    steps,
                    singularity_threshold,
                    repair_angle_radians
                );
            } catch (const ForceFieldSetupFailure& error) {
                raise_setup_error(setup_error, error);
            } catch (const ForceFieldEnergyUnitFailure& error) {
                raise_energy_unit_error(energy_unit_error, error);
            }
        },
        py::arg("molecule"),
        py::arg("forcefield"),
        py::arg("steps"),
        py::arg("singularity_threshold") = 1.0e-6,
        py::arg("repair_angle_radians") = 1.0e-3
    );
    module.def(
        "optimize",
        [setup_error, energy_unit_error, frame_error](
            const MoleculeData& molecule,
            const std::string& forcefield,
            const std::string& algorithm,
            std::size_t epochs,
            std::size_t steps_per_epoch,
            std::optional<std::size_t> perturb_interval,
            const py::object& perturbation_offsets,
            bool retain_frames,
            bool retain_epoch_history,
            bool increasing_vdw,
            double vdw_cutoff_start,
            double vdw_cutoff_end,
            double energy_tolerance,
            std::optional<std::size_t> stopping_window,
            double maximum_energy_change_kj_mol,
            double maximum_atom_displacement_angstrom,
            double maximum_rms_gradient_kj_mol_angstrom,
            double maximum_gradient_kj_mol_angstrom,
            double singularity_threshold,
            double repair_angle_radians
        ) {
            const auto offsets = read_perturbation_offsets(
                perturbation_offsets, molecule.atom_count()
            );
            std::optional<StoppingCriteria> stopping;
            if (stopping_window.has_value()) {
                stopping = StoppingCriteria{
                    *stopping_window,
                    maximum_energy_change_kj_mol,
                    maximum_atom_displacement_angstrom,
                    maximum_rms_gradient_kj_mol_angstrom,
                    maximum_gradient_kj_mol_angstrom,
                };
            }
            OptimizationOptions options{
                forcefield,
                algorithm,
                epochs,
                steps_per_epoch,
                perturb_interval,
                retain_frames,
                retain_epoch_history,
                increasing_vdw,
                vdw_cutoff_start,
                vdw_cutoff_end,
                energy_tolerance,
                stopping,
            };
            try {
                py::gil_scoped_release release;
                return optimize(
                    molecule,
                    options,
                    offsets,
                    singularity_threshold,
                    repair_angle_radians
                );
            } catch (const ForceFieldSetupFailure& error) {
                raise_setup_error(setup_error, error);
            } catch (const ForceFieldEnergyUnitFailure& error) {
                raise_energy_unit_error(energy_unit_error, error);
            } catch (const OptimizationFrameFailure& error) {
                raise_frame_error(frame_error, error);
            }
        },
        py::arg("molecule"),
        py::arg("forcefield"),
        py::arg("algorithm"),
        py::arg("epochs"),
        py::arg("steps_per_epoch"),
        py::arg("perturb_interval") = std::nullopt,
        py::arg("perturbation_offsets") = py::none(),
        py::arg("retain_frames") = false,
        py::arg("retain_epoch_history") = false,
        py::arg("increasing_vdw") = false,
        py::arg("vdw_cutoff_start") = 1.0,
        py::arg("vdw_cutoff_end") = 10.0,
        py::arg("energy_tolerance") = 1.0e-6,
        py::arg("stopping_window") = std::nullopt,
        py::arg("maximum_energy_change_kj_mol") = 1.0e-4,
        py::arg("maximum_atom_displacement_angstrom") = 1.0e-4,
        py::arg("maximum_rms_gradient_kj_mol_angstrom") = 1.0,
        py::arg("maximum_gradient_kj_mol_angstrom") = 5.0,
        py::arg("singularity_threshold") = 1.0e-6,
        py::arg("repair_angle_radians") = 1.0e-3
    );
}
