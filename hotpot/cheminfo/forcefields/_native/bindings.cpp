#include "bindings.hpp"

#include "contracts.hpp"
#include "stage_contracts.hpp"
#include "structure_session.hpp"
#include "trajectory.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace py = pybind11;


namespace hotpot::forcefields {


namespace {


template <typename Value>
const Value* require_array(
    const py::array& array,
    const char* name,
    int dimensions
) {
    if (!array.dtype().is(py::dtype::of<Value>())) {
        throw py::type_error(std::string(name) + " has an unexpected dtype");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
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


PerturbationOffsetBatch read_perturbation_offsets(const py::array& array) {
    const auto* data = require_array<double>(
        array, "perturbation_offsets", 3
    );
    if (array.shape(2) != 3) {
        throw py::value_error(
            "perturbation_offsets must have shape (K, N, 3)"
        );
    }
    PerturbationOffsetBatch offsets{
        static_cast<std::size_t>(array.shape(1)),
        {},
    };
    offsets.frames.reserve(array.shape(0));
    for (py::ssize_t frame = 0; frame < array.shape(0); ++frame) {
        std::vector<Coordinate> frame_offsets;
        frame_offsets.reserve(offsets.atom_count);
        for (std::size_t atom = 0; atom < offsets.atom_count; ++atom) {
            const auto base = (
                static_cast<std::size_t>(frame) * offsets.atom_count + atom
            ) * 3;
            frame_offsets.push_back(Coordinate{
                data[base], data[base + 1], data[base + 2]
            });
        }
        offsets.frames.push_back(std::move(frame_offsets));
    }
    offsets.validate();
    return offsets;
}


std::vector<BondIndex> read_bond_indices(
    const py::array& array,
    const char* name
) {
    const auto* data = require_array<std::int32_t>(array, name, 2);
    if (array.shape(1) != 2) {
        throw py::value_error(std::string(name) + " must have shape (M, 2)");
    }
    std::vector<BondIndex> indices;
    indices.reserve(array.shape(0));
    for (py::ssize_t row = 0; row < array.shape(0); ++row) {
        indices.push_back({data[row * 2], data[row * 2 + 1]});
    }
    return indices;
}


std::vector<BondKind> read_bond_kinds(
    const py::array& array,
    const char* name
) {
    const auto values = read_vector<std::uint8_t>(array, name);
    std::vector<BondKind> kinds;
    kinds.reserve(values.size());
    for (const auto value : values) {
        if (value < static_cast<std::uint8_t>(BondKind::SINGLE)
            || value > static_cast<std::uint8_t>(BondKind::UNKNOWN)) {
            throw py::value_error(std::string(name) + " contains an unknown value");
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


template <typename Value>
py::array_t<Value> vector_array(const std::vector<Value>& values) {
    py::array_t<Value> array(values.size());
    if (!values.empty()) {
        std::memcpy(
            array.mutable_data(),
            values.data(),
            values.size() * sizeof(Value)
        );
    }
    return array;
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


py::array_t<double> perturbation_array(
    const PerturbationOffsetBatch& offsets
) {
    py::array_t<double> array({
        offsets.frame_count(), offsets.atom_count, std::size_t{3}
    });
    auto* destination = array.mutable_data();
    for (std::size_t frame = 0; frame < offsets.frame_count(); ++frame) {
        for (std::size_t atom = 0; atom < offsets.atom_count; ++atom) {
            const auto base = (frame * offsets.atom_count + atom) * 3;
            std::memcpy(
                destination + base,
                offsets.frames[frame][atom].data(),
                3 * sizeof(double)
            );
        }
    }
    return array;
}


py::array_t<std::int32_t> bond_index_array(
    const std::vector<BondIndex>& indices
) {
    py::array_t<std::int32_t> array({indices.size(), std::size_t{2}});
    auto* destination = array.mutable_data();
    for (std::size_t row = 0; row < indices.size(); ++row) {
        destination[row * 2] = indices[row][0];
        destination[row * 2 + 1] = indices[row][1];
    }
    return array;
}


void bind_session_contracts(py::module_& module) {
    py::class_<ComplexSessionInput>(module, "ComplexSessionInput")
        .def(
            py::init([](
                std::int32_t schema_version,
                const py::array& atomic_numbers,
                const py::array& formal_charges,
                const py::array& partial_charges,
                const py::array& coordinates,
                const py::array& atom_aromatic,
                const py::array& ligand_bond_indices,
                const py::array& ligand_bond_orders,
                const py::array& ligand_bond_kinds,
                const py::array& ligand_bond_aromatic,
                const py::array& metal_indices,
                const py::array& intended_coordination_bonds,
                const py::array& intended_coordination_orders,
                const py::array& intended_coordination_kinds,
                const py::object& unit_cell
            ) {
                ComplexSessionInput input{
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
                    read_bond_indices(
                        ligand_bond_indices, "ligand_bond_indices"
                    ),
                    read_vector<double>(
                        ligand_bond_orders, "ligand_bond_orders"
                    ),
                    read_bond_kinds(
                        ligand_bond_kinds, "ligand_bond_kinds"
                    ),
                    read_vector<std::uint8_t>(
                        ligand_bond_aromatic, "ligand_bond_aromatic"
                    ),
                    read_vector<std::int32_t>(
                        metal_indices, "metal_indices"
                    ),
                    read_bond_indices(
                        intended_coordination_bonds,
                        "intended_coordination_bonds"
                    ),
                    read_vector<double>(
                        intended_coordination_orders,
                        "intended_coordination_orders"
                    ),
                    read_bond_kinds(
                        intended_coordination_kinds,
                        "intended_coordination_kinds"
                    ),
                    read_unit_cell(unit_cell),
                };
                input.validate();
                return input;
            }),
            py::arg("schema_version"),
            py::arg("atomic_numbers"),
            py::arg("formal_charges"),
            py::arg("partial_charges"),
            py::arg("coordinates"),
            py::arg("atom_aromatic"),
            py::arg("ligand_bond_indices"),
            py::arg("ligand_bond_orders"),
            py::arg("ligand_bond_kinds"),
            py::arg("ligand_bond_aromatic"),
            py::arg("metal_indices"),
            py::arg("intended_coordination_bonds"),
            py::arg("intended_coordination_orders"),
            py::arg("intended_coordination_kinds"),
            py::arg("unit_cell") = py::none()
        )
        .def_property_readonly("atom_count", &ComplexSessionInput::atom_count)
        .def_property_readonly(
            "ligand_bond_count", &ComplexSessionInput::ligand_bond_count
        )
        .def_property_readonly(
            "intended_coordination_bond_count",
            &ComplexSessionInput::intended_coordination_bond_count
        );

    py::class_<PerturbationOffsetBatch>(module, "PerturbationOffsetBatch")
        .def(py::init([](const py::array& offsets) {
            return read_perturbation_offsets(offsets);
        }), py::arg("offsets"))
        .def_readonly("atom_count", &PerturbationOffsetBatch::atom_count)
        .def_property_readonly(
            "frame_count", &PerturbationOffsetBatch::frame_count
        )
        .def_property_readonly("offsets", &perturbation_array);

    py::class_<StructureSnapshot>(module, "StructureSnapshot")
        .def_property_readonly(
            "coordinates",
            [](const StructureSnapshot& snapshot) {
                return coordinate_array(snapshot.coordinates);
            }
        )
        .def_property_readonly(
            "active_ligand_bond_mask",
            [](const StructureSnapshot& snapshot) {
                return vector_array(snapshot.active_ligand_bond_mask);
            }
        )
        .def_property_readonly(
            "active_coordination_mask",
            [](const StructureSnapshot& snapshot) {
                return vector_array(snapshot.active_coordination_mask);
            }
        )
        .def_property_readonly(
            "component_ids",
            [](const StructureSnapshot& snapshot) {
                return vector_array(snapshot.component_ids);
            }
        )
        .def_readonly("ligand_bond_count", &StructureSnapshot::ligand_bond_count)
        .def_readonly("active_bond_count", &StructureSnapshot::active_bond_count)
        .def_readonly(
            "coordinate_revision", &StructureSnapshot::coordinate_revision
        )
        .def_readonly("topology_revision", &StructureSnapshot::topology_revision);

    py::class_<StructureSession>(module, "StructureSession")
        .def_property_readonly("atom_count", &StructureSession::atom_count)
        .def_property_readonly(
            "ligand_bond_count", &StructureSession::ligand_bond_count
        )
        .def_property_readonly(
            "intended_coordination_bond_count",
            &StructureSession::intended_coordination_bond_count
        );

    module.def(
        "create_coordination_session",
        &create_coordination_session,
        py::arg("session_input"),
        py::call_guard<py::gil_scoped_release>()
    );
    module.def(
        "create_optimization_session",
        &create_optimization_session,
        py::arg("session_input"),
        py::call_guard<py::gil_scoped_release>()
    );
    module.def(
        "snapshot_structure",
        &snapshot_structure,
        py::arg("session"),
        py::call_guard<py::gil_scoped_release>()
    );
    module.def(
        "update_structure_coordinates",
        [](StructureSession& session, const py::array& coordinates) {
            const auto copied = read_coordinates(coordinates, "coordinates");
            py::gil_scoped_release release;
            update_structure_coordinates(session, copied);
        },
        py::arg("session"),
        py::arg("coordinates")
    );
    module.def(
        "set_ligand_bond_active_mask",
        [](StructureSession& session, const py::array& active_mask) {
            const auto copied = read_vector<std::uint8_t>(
                active_mask, "active_mask"
            );
            py::gil_scoped_release release;
            set_ligand_bond_active_mask(session, copied);
        },
        py::arg("session"),
        py::arg("active_mask")
    );
    module.def(
        "set_coordination_active_mask",
        [](StructureSession& session, const py::array& active_mask) {
            const auto copied = read_vector<std::uint8_t>(
                active_mask, "active_mask"
            );
            py::gil_scoped_release release;
            set_coordination_active_mask(session, copied);
        },
        py::arg("session"),
        py::arg("active_mask")
    );
}


void bind_stage_options(py::module_& module) {
    py::enum_<FrameDetail>(module, "FrameDetail")
        .value("NONE", FrameDetail::NONE)
        .value("OPTIMIZATION", FrameDetail::OPTIMIZATION)
        .value("ALL_ATTEMPTS", FrameDetail::ALL_ATTEMPTS);
    py::enum_<NativeStageStatus>(module, "NativeStageStatus")
        .value("COMPLETED", NativeStageStatus::COMPLETED)
        .value("PARTIAL", NativeStageStatus::PARTIAL)
        .value("FAILED", NativeStageStatus::FAILED);
    py::enum_<NativeTrajectoryStart>(module, "NativeTrajectoryStart")
        .value("LIGAND_BUILD", NativeTrajectoryStart::LIGAND_BUILD)
        .value(
            "COORDINATION_RESTORATION",
            NativeTrajectoryStart::COORDINATION_RESTORATION
        )
        .value("COMPLEX_UNTANGLING", NativeTrajectoryStart::COMPLEX_UNTANGLING)
        .value("FINAL_OPTIMIZATION", NativeTrajectoryStart::FINAL_OPTIMIZATION);

    py::class_<OptimizationStoppingOptions>(
        module, "OptimizationStoppingOptions"
    )
        .def(py::init([](
            std::size_t window,
            double maximum_energy_change_kj_mol,
            double maximum_atom_displacement_angstrom,
            double maximum_rms_gradient_kj_mol_angstrom,
            double maximum_gradient_kj_mol_angstrom
        ) {
            OptimizationStoppingOptions options{
                window,
                maximum_energy_change_kj_mol,
                maximum_atom_displacement_angstrom,
                maximum_rms_gradient_kj_mol_angstrom,
                maximum_gradient_kj_mol_angstrom,
            };
            options.validate();
            return options;
        }),
        py::arg("window"),
        py::arg("maximum_energy_change_kj_mol"),
        py::arg("maximum_atom_displacement_angstrom"),
        py::arg("maximum_rms_gradient_kj_mol_angstrom"),
        py::arg("maximum_gradient_kj_mol_angstrom"))
        .def_readonly("window", &OptimizationStoppingOptions::window)
        .def_readonly(
            "maximum_energy_change_kj_mol",
            &OptimizationStoppingOptions::maximum_energy_change_kj_mol
        )
        .def_readonly(
            "maximum_atom_displacement_angstrom",
            &OptimizationStoppingOptions::maximum_atom_displacement_angstrom
        )
        .def_readonly(
            "maximum_rms_gradient_kj_mol_angstrom",
            &OptimizationStoppingOptions::maximum_rms_gradient_kj_mol_angstrom
        )
        .def_readonly(
            "maximum_gradient_kj_mol_angstrom",
            &OptimizationStoppingOptions::maximum_gradient_kj_mol_angstrom
        );

    py::class_<CoordinationStageOptions>(module, "CoordinationStageOptions")
        .def(py::init([](
            std::string forcefield,
            std::size_t attempt_limit,
            std::size_t relaxation_steps,
            double perturb_sigma,
            NativeTrajectoryStart trajectory_start,
            FrameDetail frame_detail
        ) {
            CoordinationStageOptions options{
                std::move(forcefield),
                attempt_limit,
                relaxation_steps,
                perturb_sigma,
                trajectory_start,
                frame_detail,
            };
            options.validate();
            return options;
        }),
        py::arg("forcefield"),
        py::arg("attempt_limit"),
        py::arg("relaxation_steps"),
        py::arg("perturb_sigma"),
        py::arg("trajectory_start"),
        py::arg("frame_detail"))
        .def_readonly("forcefield", &CoordinationStageOptions::forcefield)
        .def_readonly("attempt_limit", &CoordinationStageOptions::attempt_limit)
        .def_readonly(
            "relaxation_steps", &CoordinationStageOptions::relaxation_steps
        )
        .def_readonly("perturb_sigma", &CoordinationStageOptions::perturb_sigma)
        .def_readonly(
            "trajectory_start", &CoordinationStageOptions::trajectory_start
        )
        .def_readonly("frame_detail", &CoordinationStageOptions::frame_detail);

    py::class_<ComplexOptimizationOptions>(
        module, "ComplexOptimizationOptions"
    )
        .def(py::init([](
            std::string forcefield,
            std::string algorithm,
            std::size_t epochs,
            std::size_t steps_per_epoch,
            std::size_t untangling_attempt_limit,
            std::optional<std::size_t> perturb_interval,
            double perturb_sigma,
            NativeTrajectoryStart trajectory_start,
            FrameDetail frame_detail,
            bool retain_epoch_history,
            bool increasing_vdw,
            double vdw_cutoff_start,
            double vdw_cutoff_end,
            double energy_tolerance,
            std::optional<OptimizationStoppingOptions> stopping
        ) {
            ComplexOptimizationOptions options{
                std::move(forcefield),
                std::move(algorithm),
                epochs,
                steps_per_epoch,
                untangling_attempt_limit,
                perturb_interval,
                perturb_sigma,
                trajectory_start,
                frame_detail,
                retain_epoch_history,
                increasing_vdw,
                vdw_cutoff_start,
                vdw_cutoff_end,
                energy_tolerance,
                std::move(stopping),
            };
            options.validate();
            return options;
        }),
        py::arg("forcefield"),
        py::arg("algorithm"),
        py::arg("epochs"),
        py::arg("steps_per_epoch"),
        py::arg("untangling_attempt_limit"),
        py::arg("perturb_interval"),
        py::arg("perturb_sigma"),
        py::arg("trajectory_start"),
        py::arg("frame_detail"),
        py::arg("retain_epoch_history"),
        py::arg("increasing_vdw"),
        py::arg("vdw_cutoff_start"),
        py::arg("vdw_cutoff_end"),
        py::arg("energy_tolerance"),
        py::arg("stopping"))
        .def_readonly("forcefield", &ComplexOptimizationOptions::forcefield)
        .def_readonly("algorithm", &ComplexOptimizationOptions::algorithm)
        .def_readonly("epochs", &ComplexOptimizationOptions::epochs)
        .def_readonly(
            "steps_per_epoch", &ComplexOptimizationOptions::steps_per_epoch
        )
        .def_readonly(
            "untangling_attempt_limit",
            &ComplexOptimizationOptions::untangling_attempt_limit
        )
        .def_readonly(
            "perturb_interval", &ComplexOptimizationOptions::perturb_interval
        )
        .def_readonly("perturb_sigma", &ComplexOptimizationOptions::perturb_sigma)
        .def_readonly(
            "trajectory_start", &ComplexOptimizationOptions::trajectory_start
        )
        .def_readonly("frame_detail", &ComplexOptimizationOptions::frame_detail)
        .def_readonly(
            "retain_epoch_history",
            &ComplexOptimizationOptions::retain_epoch_history
        )
        .def_readonly(
            "increasing_vdw", &ComplexOptimizationOptions::increasing_vdw
        )
        .def_readonly(
            "vdw_cutoff_start", &ComplexOptimizationOptions::vdw_cutoff_start
        )
        .def_readonly(
            "vdw_cutoff_end", &ComplexOptimizationOptions::vdw_cutoff_end
        )
        .def_readonly(
            "energy_tolerance", &ComplexOptimizationOptions::energy_tolerance
        )
        .def_readonly("stopping", &ComplexOptimizationOptions::stopping);
}


void bind_trajectory_contracts(py::module_& module) {
    py::enum_<NativeTrajectoryStage>(module, "NativeTrajectoryStage")
        .value("LIGAND_BUILD", NativeTrajectoryStage::LIGAND_BUILD)
        .value(
            "COORDINATION_RESTORATION",
            NativeTrajectoryStage::COORDINATION_RESTORATION
        )
        .value("COMPLEX_UNTANGLING", NativeTrajectoryStage::COMPLEX_UNTANGLING)
        .value("FINAL_OPTIMIZATION", NativeTrajectoryStage::FINAL_OPTIMIZATION);
    py::enum_<NativeTrajectoryEvent>(module, "NativeTrajectoryEvent")
        .value("INITIAL", NativeTrajectoryEvent::INITIAL)
        .value("BUILD_COMPLETE", NativeTrajectoryEvent::BUILD_COMPLETE)
        .value("WARMUP_COMPLETE", NativeTrajectoryEvent::WARMUP_COMPLETE)
        .value("COORDINATION_READY", NativeTrajectoryEvent::COORDINATION_READY)
        .value("BOND_TRIAL", NativeTrajectoryEvent::BOND_TRIAL)
        .value("BOND_ACCEPTED", NativeTrajectoryEvent::BOND_ACCEPTED)
        .value("BOND_REJECTED", NativeTrajectoryEvent::BOND_REJECTED)
        .value("BOND_ROLLBACK", NativeTrajectoryEvent::BOND_ROLLBACK)
        .value("BOND_FORCED", NativeTrajectoryEvent::BOND_FORCED)
        .value(
            "METAL_RELOCATION_TRIAL",
            NativeTrajectoryEvent::METAL_RELOCATION_TRIAL
        )
        .value("METAL_RELOCATED", NativeTrajectoryEvent::METAL_RELOCATED)
        .value(
            "METAL_RELOCATION_FAILED",
            NativeTrajectoryEvent::METAL_RELOCATION_FAILED
        )
        .value("TOPOLOGY_CHECKPOINT", NativeTrajectoryEvent::TOPOLOGY_CHECKPOINT)
        .value("RING_OPENED", NativeTrajectoryEvent::RING_OPENED)
        .value("PERTURBED", NativeTrajectoryEvent::PERTURBED)
        .value("OPTIMIZED", NativeTrajectoryEvent::OPTIMIZED)
        .value("RING_CLOSED", NativeTrajectoryEvent::RING_CLOSED)
        .value("SETTLED", NativeTrajectoryEvent::SETTLED)
        .value("ROLLED_BACK", NativeTrajectoryEvent::ROLLED_BACK)
        .value("EPOCH_COMPLETE", NativeTrajectoryEvent::EPOCH_COMPLETE)
        .value("TERMINAL", NativeTrajectoryEvent::TERMINAL);

    py::class_<NativeRingFrameEvidence>(module, "NativeRingFrameEvidence")
        .def(py::init([](
            std::size_t confirmed_piercing_count,
            std::optional<std::size_t> uncertain_relation_count,
            std::optional<std::string> ring_scope,
            std::optional<std::size_t> max_ring_size,
            std::optional<std::size_t> selected_ring_count,
            std::optional<std::size_t> excluded_ring_count,
            std::optional<std::size_t> candidate_pair_count,
            std::optional<std::size_t> aabb_separated_pair_count,
            std::optional<std::size_t> exact_pair_count,
            std::optional<std::size_t> does_not_pierce_pair_count,
            std::optional<bool> scan_complete
        ) {
            return NativeRingFrameEvidence{
                confirmed_piercing_count,
                uncertain_relation_count,
                std::move(ring_scope),
                max_ring_size,
                selected_ring_count,
                excluded_ring_count,
                candidate_pair_count,
                aabb_separated_pair_count,
                exact_pair_count,
                does_not_pierce_pair_count,
                scan_complete,
            };
        }),
        py::arg("confirmed_piercing_count"),
        py::arg("uncertain_relation_count"),
        py::arg("ring_scope"),
        py::arg("max_ring_size"),
        py::arg("selected_ring_count"),
        py::arg("excluded_ring_count"),
        py::arg("candidate_pair_count"),
        py::arg("aabb_separated_pair_count"),
        py::arg("exact_pair_count"),
        py::arg("does_not_pierce_pair_count"),
        py::arg("scan_complete"))
        .def_readonly(
            "confirmed_piercing_count",
            &NativeRingFrameEvidence::confirmed_piercing_count
        )
        .def_readonly(
            "uncertain_relation_count",
            &NativeRingFrameEvidence::uncertain_relation_count
        )
        .def_readonly("ring_scope", &NativeRingFrameEvidence::ring_scope)
        .def_readonly("max_ring_size", &NativeRingFrameEvidence::max_ring_size)
        .def_readonly(
            "selected_ring_count", &NativeRingFrameEvidence::selected_ring_count
        )
        .def_readonly(
            "excluded_ring_count", &NativeRingFrameEvidence::excluded_ring_count
        )
        .def_readonly(
            "candidate_pair_count", &NativeRingFrameEvidence::candidate_pair_count
        )
        .def_readonly(
            "aabb_separated_pair_count",
            &NativeRingFrameEvidence::aabb_separated_pair_count
        )
        .def_readonly(
            "exact_pair_count", &NativeRingFrameEvidence::exact_pair_count
        )
        .def_readonly(
            "does_not_pierce_pair_count",
            &NativeRingFrameEvidence::does_not_pierce_pair_count
        )
        .def_readonly("scan_complete", &NativeRingFrameEvidence::scan_complete);

    py::class_<NativeCoordinationFrameEvidence>(
        module, "NativeCoordinationFrameEvidence"
    )
        .def(py::init([](
            std::optional<BondIndex> bond_atom_indices,
            std::optional<bool> accepted,
            std::size_t pending_bond_count,
            bool forced,
            std::size_t piercing_relation_count,
            std::size_t undetermined_relation_count,
            std::size_t excluded_ring_count,
            std::optional<std::int32_t> metal_atom_index,
            std::optional<std::string> relocation_status,
            std::size_t relocation_candidates_evaluated,
            std::vector<std::int32_t> safe_donor_atom_indices,
            std::optional<double> minimum_normalized_clearance,
            std::optional<double> coordination_distance_deviation
        ) {
            return NativeCoordinationFrameEvidence{
                bond_atom_indices,
                accepted,
                pending_bond_count,
                forced,
                piercing_relation_count,
                undetermined_relation_count,
                excluded_ring_count,
                metal_atom_index,
                std::move(relocation_status),
                relocation_candidates_evaluated,
                std::move(safe_donor_atom_indices),
                minimum_normalized_clearance,
                coordination_distance_deviation,
            };
        }),
        py::arg("bond_atom_indices"),
        py::arg("accepted"),
        py::arg("pending_bond_count"),
        py::arg("forced"),
        py::arg("piercing_relation_count"),
        py::arg("undetermined_relation_count"),
        py::arg("excluded_ring_count"),
        py::arg("metal_atom_index"),
        py::arg("relocation_status"),
        py::arg("relocation_candidates_evaluated"),
        py::arg("safe_donor_atom_indices"),
        py::arg("minimum_normalized_clearance"),
        py::arg("coordination_distance_deviation"))
        .def_readonly(
            "bond_atom_indices",
            &NativeCoordinationFrameEvidence::bond_atom_indices
        )
        .def_readonly("accepted", &NativeCoordinationFrameEvidence::accepted)
        .def_readonly(
            "pending_bond_count",
            &NativeCoordinationFrameEvidence::pending_bond_count
        )
        .def_readonly("forced", &NativeCoordinationFrameEvidence::forced)
        .def_readonly(
            "piercing_relation_count",
            &NativeCoordinationFrameEvidence::piercing_relation_count
        )
        .def_readonly(
            "undetermined_relation_count",
            &NativeCoordinationFrameEvidence::undetermined_relation_count
        )
        .def_readonly(
            "excluded_ring_count",
            &NativeCoordinationFrameEvidence::excluded_ring_count
        )
        .def_readonly(
            "metal_atom_index",
            &NativeCoordinationFrameEvidence::metal_atom_index
        )
        .def_readonly(
            "relocation_status",
            &NativeCoordinationFrameEvidence::relocation_status
        )
        .def_readonly(
            "relocation_candidates_evaluated",
            &NativeCoordinationFrameEvidence::relocation_candidates_evaluated
        )
        .def_readonly(
            "safe_donor_atom_indices",
            &NativeCoordinationFrameEvidence::safe_donor_atom_indices
        )
        .def_readonly(
            "minimum_normalized_clearance",
            &NativeCoordinationFrameEvidence::minimum_normalized_clearance
        )
        .def_readonly(
            "coordination_distance_deviation",
            &NativeCoordinationFrameEvidence::coordination_distance_deviation
        );

    py::class_<NativeOptimizationFrameEvidence>(
        module, "NativeOptimizationFrameEvidence"
    )
        .def(py::init([](
            bool converged,
            bool exploded,
            bool finite_coordinates,
            bool finite_energy,
            bool finite_gradients,
            std::optional<double> rms_gradient_kj_mol_angstrom,
            std::optional<double> max_gradient_kj_mol_angstrom,
            std::optional<double> energy_change_kj_mol,
            std::optional<double> max_displacement_angstrom
        ) {
            return NativeOptimizationFrameEvidence{
                converged,
                exploded,
                finite_coordinates,
                finite_energy,
                finite_gradients,
                rms_gradient_kj_mol_angstrom,
                max_gradient_kj_mol_angstrom,
                energy_change_kj_mol,
                max_displacement_angstrom,
            };
        }),
        py::arg("converged"),
        py::arg("exploded"),
        py::arg("finite_coordinates"),
        py::arg("finite_energy"),
        py::arg("finite_gradients"),
        py::arg("rms_gradient_kj_mol_angstrom"),
        py::arg("max_gradient_kj_mol_angstrom"),
        py::arg("energy_change_kj_mol"),
        py::arg("max_displacement_angstrom"))
        .def_readonly("converged", &NativeOptimizationFrameEvidence::converged)
        .def_readonly("exploded", &NativeOptimizationFrameEvidence::exploded)
        .def_readonly(
            "finite_coordinates",
            &NativeOptimizationFrameEvidence::finite_coordinates
        )
        .def_readonly(
            "finite_energy", &NativeOptimizationFrameEvidence::finite_energy
        )
        .def_readonly(
            "finite_gradients", &NativeOptimizationFrameEvidence::finite_gradients
        )
        .def_readonly(
            "rms_gradient_kj_mol_angstrom",
            &NativeOptimizationFrameEvidence::rms_gradient_kj_mol_angstrom
        )
        .def_readonly(
            "max_gradient_kj_mol_angstrom",
            &NativeOptimizationFrameEvidence::max_gradient_kj_mol_angstrom
        )
        .def_readonly(
            "energy_change_kj_mol",
            &NativeOptimizationFrameEvidence::energy_change_kj_mol
        )
        .def_readonly(
            "max_displacement_angstrom",
            &NativeOptimizationFrameEvidence::max_displacement_angstrom
        );

    py::class_<NativeTopologyRevision>(module, "NativeTopologyRevision")
        .def(py::init([](
            const py::array& active_ligand_bond_mask,
            const py::array& active_coordination_bond_mask
        ) {
            return NativeTopologyRevision{
                read_vector<std::uint8_t>(
                    active_ligand_bond_mask, "active_ligand_bond_mask"
                ),
                read_vector<std::uint8_t>(
                    active_coordination_bond_mask,
                    "active_coordination_bond_mask"
                ),
            };
        }),
        py::arg("active_ligand_bond_mask"),
        py::arg("active_coordination_bond_mask"))
        .def_property_readonly(
            "active_ligand_bond_mask",
            [](const NativeTopologyRevision& revision) {
                return vector_array(revision.active_ligand_bond_mask);
            }
        )
        .def_property_readonly(
            "active_coordination_bond_mask",
            [](const NativeTopologyRevision& revision) {
                return vector_array(revision.active_coordination_bond_mask);
            }
        );

    py::class_<NativeTrajectoryFrame>(module, "NativeTrajectoryFrame")
        .def(py::init([](
            const py::array& coordinates,
            NativeTrajectoryStage stage,
            NativeTrajectoryEvent event,
            std::optional<std::int32_t> component_index,
            std::optional<std::int32_t> attempt,
            std::optional<std::int64_t> step,
            std::optional<double> energy_kj_mol,
            const py::object& evidence,
            std::int32_t topology_revision
        ) {
            NativeFrameEvidence native_evidence = std::monostate{};
            if (!evidence.is_none()) {
                if (py::isinstance<NativeRingFrameEvidence>(evidence)) {
                    native_evidence = evidence.cast<NativeRingFrameEvidence>();
                } else if (py::isinstance<NativeCoordinationFrameEvidence>(
                        evidence
                    )) {
                    native_evidence = evidence.cast<
                        NativeCoordinationFrameEvidence
                    >();
                } else if (py::isinstance<NativeOptimizationFrameEvidence>(
                        evidence
                    )) {
                    native_evidence = evidence.cast<
                        NativeOptimizationFrameEvidence
                    >();
                } else {
                    throw py::type_error("evidence has an unexpected type");
                }
            }
            return NativeTrajectoryFrame{
                read_coordinates(coordinates, "coordinates"),
                stage,
                event,
                component_index,
                attempt,
                step,
                energy_kj_mol,
                std::move(native_evidence),
                topology_revision,
            };
        }),
        py::arg("coordinates"),
        py::arg("stage"),
        py::arg("event"),
        py::arg("component_index"),
        py::arg("attempt"),
        py::arg("step"),
        py::arg("energy_kj_mol"),
        py::arg("evidence"),
        py::arg("topology_revision"))
        .def_property_readonly(
            "coordinates",
            [](const NativeTrajectoryFrame& frame) {
                return coordinate_array(frame.coordinates);
            }
        )
        .def_readonly("stage", &NativeTrajectoryFrame::stage)
        .def_readonly("event", &NativeTrajectoryFrame::event)
        .def_readonly("component_index", &NativeTrajectoryFrame::component_index)
        .def_readonly("attempt", &NativeTrajectoryFrame::attempt)
        .def_readonly("step", &NativeTrajectoryFrame::step)
        .def_readonly("energy_kj_mol", &NativeTrajectoryFrame::energy_kj_mol)
        .def_property_readonly("evidence", [](
            const NativeTrajectoryFrame& frame
        ) -> py::object {
            if (std::holds_alternative<NativeRingFrameEvidence>(frame.evidence)) {
                return py::cast(std::get<NativeRingFrameEvidence>(frame.evidence));
            }
            if (std::holds_alternative<NativeCoordinationFrameEvidence>(
                    frame.evidence
                )) {
                return py::cast(
                    std::get<NativeCoordinationFrameEvidence>(frame.evidence)
                );
            }
            if (std::holds_alternative<NativeOptimizationFrameEvidence>(
                    frame.evidence
                )) {
                return py::cast(
                    std::get<NativeOptimizationFrameEvidence>(frame.evidence)
                );
            }
            return py::none();
        })
        .def_readonly(
            "topology_revision", &NativeTrajectoryFrame::topology_revision
        );

    py::class_<NativeTrajectoryBatch>(module, "NativeTrajectoryBatch")
        .def(py::init([](
            std::size_t atom_count,
            std::size_t ligand_bond_count,
            std::size_t intended_coordination_bond_count,
            NativeTrajectoryStart start,
            std::vector<NativeTopologyRevision> topology_revisions,
            std::vector<NativeTrajectoryFrame> frames,
            std::optional<std::int64_t> selected_frame_index,
            std::optional<std::int64_t> terminal_frame_index
        ) {
            NativeTrajectoryBatch batch{
                atom_count,
                ligand_bond_count,
                intended_coordination_bond_count,
                start,
                std::move(topology_revisions),
                std::move(frames),
                selected_frame_index.value_or(-1),
                terminal_frame_index.value_or(-1),
            };
            batch.validate();
            return batch;
        }),
        py::arg("atom_count"),
        py::arg("ligand_bond_count"),
        py::arg("intended_coordination_bond_count"),
        py::arg("start"),
        py::arg("topology_revisions"),
        py::arg("frames"),
        py::arg("selected_frame_index"),
        py::arg("terminal_frame_index"))
        .def_property_readonly("frame_count", &NativeTrajectoryBatch::frame_count)
        .def_readonly("atom_count", &NativeTrajectoryBatch::atom_count)
        .def_readonly("ligand_bond_count", &NativeTrajectoryBatch::ligand_bond_count)
        .def_readonly(
            "intended_coordination_bond_count",
            &NativeTrajectoryBatch::intended_coordination_bond_count
        )
        .def_readonly("start", &NativeTrajectoryBatch::start)
        .def_readonly(
            "topology_revisions", &NativeTrajectoryBatch::topology_revisions
        )
        .def_property_readonly(
            "selected_frame_index",
            [](const NativeTrajectoryBatch& batch) -> py::object {
                return batch.selected_frame_index < 0
                    ? py::none()
                    : py::cast(batch.selected_frame_index);
            }
        )
        .def_property_readonly(
            "terminal_frame_index",
            [](const NativeTrajectoryBatch& batch) -> py::object {
                return batch.terminal_frame_index < 0
                    ? py::none()
                    : py::cast(batch.terminal_frame_index);
            }
        )
        .def_readonly("frames", &NativeTrajectoryBatch::frames);
}


void bind_stage_results(py::module_& module) {
    py::class_<CoordinationStageResult>(module, "CoordinationStageResult")
        .def(py::init([](
            NativeStageStatus status,
            const py::array& selected_coordinates,
            const py::array& terminal_coordinates,
            const py::array& final_active_coordination_mask,
            std::size_t attempt_limit,
            std::size_t attempts_completed,
            std::size_t metal_relocation_attempt_count,
            std::vector<std::int32_t> relocated_metal_indices,
            std::vector<std::int32_t> infeasible_metal_indices,
            std::vector<BondIndex> forced_bond_keys,
            std::size_t rejected_piercing_trial_count,
            std::size_t undetermined_trial_count,
            std::size_t excluded_ring_observation_count,
            std::vector<std::string> warning_codes,
            NativeTrajectoryBatch trajectory
        ) {
            CoordinationStageResult result{
                status,
                read_coordinates(selected_coordinates, "selected_coordinates"),
                read_coordinates(terminal_coordinates, "terminal_coordinates"),
                read_vector<std::uint8_t>(
                    final_active_coordination_mask,
                    "final_active_coordination_mask"
                ),
                attempt_limit,
                attempts_completed,
                metal_relocation_attempt_count,
                std::move(relocated_metal_indices),
                std::move(infeasible_metal_indices),
                std::move(forced_bond_keys),
                rejected_piercing_trial_count,
                undetermined_trial_count,
                excluded_ring_observation_count,
                std::move(warning_codes),
                std::move(trajectory),
            };
            result.validate();
            return result;
        }),
        py::arg("status"),
        py::arg("selected_coordinates"),
        py::arg("terminal_coordinates"),
        py::arg("final_active_coordination_mask"),
        py::arg("attempt_limit"),
        py::arg("attempts_completed"),
        py::arg("metal_relocation_attempt_count"),
        py::arg("relocated_metal_indices"),
        py::arg("infeasible_metal_indices"),
        py::arg("forced_bond_keys"),
        py::arg("rejected_piercing_trial_count"),
        py::arg("undetermined_trial_count"),
        py::arg("excluded_ring_observation_count"),
        py::arg("warning_codes"),
        py::arg("trajectory"))
        .def_readonly("status", &CoordinationStageResult::status)
        .def_property_readonly(
            "selected_coordinates",
            [](const CoordinationStageResult& result) {
                return coordinate_array(result.selected_coordinates);
            }
        )
        .def_property_readonly(
            "terminal_coordinates",
            [](const CoordinationStageResult& result) {
                return coordinate_array(result.terminal_coordinates);
            }
        )
        .def_property_readonly(
            "final_active_coordination_mask",
            [](const CoordinationStageResult& result) {
                return vector_array(result.final_active_coordination_mask);
            }
        )
        .def_readonly("attempt_limit", &CoordinationStageResult::attempt_limit)
        .def_readonly(
            "attempts_completed", &CoordinationStageResult::attempts_completed
        )
        .def_readonly(
            "metal_relocation_attempt_count",
            &CoordinationStageResult::metal_relocation_attempt_count
        )
        .def_readonly(
            "relocated_metal_indices",
            &CoordinationStageResult::relocated_metal_indices
        )
        .def_readonly(
            "infeasible_metal_indices",
            &CoordinationStageResult::infeasible_metal_indices
        )
        .def_property_readonly(
            "forced_bond_keys",
            [](const CoordinationStageResult& result) {
                return bond_index_array(result.forced_bond_keys);
            }
        )
        .def_readonly(
            "rejected_piercing_trial_count",
            &CoordinationStageResult::rejected_piercing_trial_count
        )
        .def_readonly(
            "undetermined_trial_count",
            &CoordinationStageResult::undetermined_trial_count
        )
        .def_readonly(
            "excluded_ring_observation_count",
            &CoordinationStageResult::excluded_ring_observation_count
        )
        .def_readonly("warning_codes", &CoordinationStageResult::warning_codes)
        .def_readonly("trajectory", &CoordinationStageResult::trajectory);

    py::class_<ComplexOptimizationResult>(
        module, "ComplexOptimizationResult"
    )
        .def(py::init([](
            NativeStageStatus status,
            const py::array& selected_coordinates,
            const py::array& terminal_coordinates,
            const py::array& final_active_coordination_mask,
            std::size_t untangling_attempt_limit,
            std::size_t untangling_attempts_completed,
            std::size_t initial_piercing_count,
            std::size_t final_piercing_count,
            std::size_t minimum_piercing_count,
            bool untangling_resolved,
            std::int64_t selected_frame_index,
            std::int64_t best_epoch,
            double final_energy_kj_mol,
            double best_energy_kj_mol,
            double rms_gradient_kj_mol_angstrom,
            double max_gradient_kj_mol_angstrom,
            std::vector<double> energy_changes,
            std::vector<double> max_displacements,
            std::vector<double> epoch_energies,
            bool exploded,
            bool converged,
            bool terminal_converged,
            std::size_t epochs_completed,
            std::size_t steps_submitted,
            std::size_t initialization_steps,
            std::size_t selected_segment_epochs_completed,
            std::string backend_energy_unit,
            std::string termination_reason,
            std::vector<std::string> warning_codes,
            NativeTrajectoryBatch trajectory
        ) {
            ComplexOptimizationResult result{
                status,
                read_coordinates(selected_coordinates, "selected_coordinates"),
                read_coordinates(terminal_coordinates, "terminal_coordinates"),
                read_vector<std::uint8_t>(
                    final_active_coordination_mask,
                    "final_active_coordination_mask"
                ),
                untangling_attempt_limit,
                untangling_attempts_completed,
                initial_piercing_count,
                final_piercing_count,
                minimum_piercing_count,
                untangling_resolved,
                selected_frame_index,
                best_epoch,
                final_energy_kj_mol,
                best_energy_kj_mol,
                rms_gradient_kj_mol_angstrom,
                max_gradient_kj_mol_angstrom,
                std::move(energy_changes),
                std::move(max_displacements),
                std::move(epoch_energies),
                exploded,
                converged,
                terminal_converged,
                epochs_completed,
                steps_submitted,
                initialization_steps,
                selected_segment_epochs_completed,
                std::move(backend_energy_unit),
                std::move(termination_reason),
                std::move(warning_codes),
                std::move(trajectory),
            };
            result.validate();
            return result;
        }),
        py::arg("status"),
        py::arg("selected_coordinates"),
        py::arg("terminal_coordinates"),
        py::arg("final_active_coordination_mask"),
        py::arg("untangling_attempt_limit"),
        py::arg("untangling_attempts_completed"),
        py::arg("initial_piercing_count"),
        py::arg("final_piercing_count"),
        py::arg("minimum_piercing_count"),
        py::arg("untangling_resolved"),
        py::arg("selected_frame_index"),
        py::arg("best_epoch"),
        py::arg("final_energy_kj_mol"),
        py::arg("best_energy_kj_mol"),
        py::arg("rms_gradient_kj_mol_angstrom"),
        py::arg("max_gradient_kj_mol_angstrom"),
        py::arg("energy_changes"),
        py::arg("max_displacements"),
        py::arg("epoch_energies"),
        py::arg("exploded"),
        py::arg("converged"),
        py::arg("terminal_converged"),
        py::arg("epochs_completed"),
        py::arg("steps_submitted"),
        py::arg("initialization_steps"),
        py::arg("selected_segment_epochs_completed"),
        py::arg("backend_energy_unit"),
        py::arg("termination_reason"),
        py::arg("warning_codes"),
        py::arg("trajectory"))
        .def_readonly("status", &ComplexOptimizationResult::status)
        .def_property_readonly(
            "selected_coordinates",
            [](const ComplexOptimizationResult& result) {
                return coordinate_array(result.selected_coordinates);
            }
        )
        .def_property_readonly(
            "terminal_coordinates",
            [](const ComplexOptimizationResult& result) {
                return coordinate_array(result.terminal_coordinates);
            }
        )
        .def_property_readonly(
            "final_active_coordination_mask",
            [](const ComplexOptimizationResult& result) {
                return vector_array(result.final_active_coordination_mask);
            }
        )
        .def_readonly(
            "untangling_attempt_limit",
            &ComplexOptimizationResult::untangling_attempt_limit
        )
        .def_readonly(
            "untangling_attempts_completed",
            &ComplexOptimizationResult::untangling_attempts_completed
        )
        .def_readonly(
            "initial_piercing_count",
            &ComplexOptimizationResult::initial_piercing_count
        )
        .def_readonly(
            "final_piercing_count",
            &ComplexOptimizationResult::final_piercing_count
        )
        .def_readonly(
            "minimum_piercing_count",
            &ComplexOptimizationResult::minimum_piercing_count
        )
        .def_readonly(
            "untangling_resolved",
            &ComplexOptimizationResult::untangling_resolved
        )
        .def_readonly(
            "selected_frame_index",
            &ComplexOptimizationResult::selected_frame_index
        )
        .def_readonly("best_epoch", &ComplexOptimizationResult::best_epoch)
        .def_readonly(
            "final_energy_kj_mol",
            &ComplexOptimizationResult::final_energy_kj_mol
        )
        .def_readonly(
            "best_energy_kj_mol",
            &ComplexOptimizationResult::best_energy_kj_mol
        )
        .def_readonly(
            "rms_gradient_kj_mol_angstrom",
            &ComplexOptimizationResult::rms_gradient_kj_mol_angstrom
        )
        .def_readonly(
            "max_gradient_kj_mol_angstrom",
            &ComplexOptimizationResult::max_gradient_kj_mol_angstrom
        )
        .def_readonly(
            "energy_changes", &ComplexOptimizationResult::energy_changes
        )
        .def_readonly(
            "max_displacements", &ComplexOptimizationResult::max_displacements
        )
        .def_readonly(
            "epoch_energies", &ComplexOptimizationResult::epoch_energies
        )
        .def_readonly("exploded", &ComplexOptimizationResult::exploded)
        .def_readonly("converged", &ComplexOptimizationResult::converged)
        .def_readonly(
            "terminal_converged",
            &ComplexOptimizationResult::terminal_converged
        )
        .def_readonly(
            "epochs_completed", &ComplexOptimizationResult::epochs_completed
        )
        .def_readonly(
            "steps_submitted", &ComplexOptimizationResult::steps_submitted
        )
        .def_readonly(
            "initialization_steps",
            &ComplexOptimizationResult::initialization_steps
        )
        .def_readonly(
            "selected_segment_epochs_completed",
            &ComplexOptimizationResult::selected_segment_epochs_completed
        )
        .def_readonly(
            "backend_energy_unit",
            &ComplexOptimizationResult::backend_energy_unit
        )
        .def_readonly(
            "termination_reason",
            &ComplexOptimizationResult::termination_reason
        )
        .def_readonly("warning_codes", &ComplexOptimizationResult::warning_codes)
        .def_readonly("trajectory", &ComplexOptimizationResult::trajectory);

    py::class_<ComplexWorkflowResult>(module, "ComplexWorkflowResult")
        .def(py::init([](
            CoordinationStageResult coordination,
            ComplexOptimizationResult optimization,
            const py::array& selected_coordinates,
            const py::array& terminal_coordinates,
            const py::array& final_active_coordination_mask,
            std::vector<std::string> warning_codes,
            NativeTrajectoryBatch trajectory
        ) {
            ComplexWorkflowResult result{
                std::move(coordination),
                std::move(optimization),
                read_coordinates(selected_coordinates, "selected_coordinates"),
                read_coordinates(terminal_coordinates, "terminal_coordinates"),
                read_vector<std::uint8_t>(
                    final_active_coordination_mask,
                    "final_active_coordination_mask"
                ),
                std::move(warning_codes),
                std::move(trajectory),
            };
            result.validate();
            return result;
        }),
        py::arg("coordination"),
        py::arg("optimization"),
        py::arg("selected_coordinates"),
        py::arg("terminal_coordinates"),
        py::arg("final_active_coordination_mask"),
        py::arg("warning_codes"),
        py::arg("trajectory"))
        .def_readonly("coordination", &ComplexWorkflowResult::coordination)
        .def_readonly("optimization", &ComplexWorkflowResult::optimization)
        .def_property_readonly(
            "selected_coordinates",
            [](const ComplexWorkflowResult& result) {
                return coordinate_array(result.selected_coordinates);
            }
        )
        .def_property_readonly(
            "terminal_coordinates",
            [](const ComplexWorkflowResult& result) {
                return coordinate_array(result.terminal_coordinates);
            }
        )
        .def_property_readonly(
            "final_active_coordination_mask",
            [](const ComplexWorkflowResult& result) {
                return vector_array(result.final_active_coordination_mask);
            }
        )
        .def_readonly("warning_codes", &ComplexWorkflowResult::warning_codes)
        .def_readonly("trajectory", &ComplexWorkflowResult::trajectory);
}


}  // namespace


void bind_native_forcefield_contracts(py::module_& module) {
    bind_session_contracts(module);
    bind_stage_options(module);
    bind_trajectory_contracts(module);
    bind_stage_results(module);
}


}  // namespace hotpot::forcefields
