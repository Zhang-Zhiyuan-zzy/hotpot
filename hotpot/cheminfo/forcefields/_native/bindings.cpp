#include "bindings.hpp"

#include "contracts.hpp"
#include "coordination_stage.hpp"
#include "optimization_stage.hpp"
#include "placement_engine.hpp"
#include "placement_policy.hpp"
#include "radii.hpp"
#include "stage_contracts.hpp"
#include "structure_session.hpp"
#include "trajectory.hpp"
#include "workflow_stage.hpp"

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
    py::enum_<RadiusSource>(module, "RadiusSource")
        .value("OPENBABEL_COVALENT", RadiusSource::OPENBABEL_COVALENT)
        .value("DEFAULT_COVALENT", RadiusSource::DEFAULT_COVALENT);
    py::class_<AtomicRadius>(module, "AtomicRadius")
        .def_readonly("angstrom", &AtomicRadius::angstrom)
        .def_readonly("source", &AtomicRadius::source);
    module.def("covalent_radius", &covalent_radius, py::arg("atomic_number"));

    py::enum_<PlacementStatus>(module, "PlacementStatus")
        .value("FULLY_FEASIBLE", PlacementStatus::FULLY_FEASIBLE)
        .value("PARTIAL", PlacementStatus::PARTIAL)
        .value("INFEASIBLE", PlacementStatus::INFEASIBLE);
    py::enum_<DonorPathStatus>(module, "DonorPathStatus")
        .value("SAFE", DonorPathStatus::SAFE)
        .value("UNDETERMINED", DonorPathStatus::UNDETERMINED)
        .value("OUT_OF_RANGE", DonorPathStatus::OUT_OF_RANGE)
        .value("ATOM_OBSTRUCTION", DonorPathStatus::ATOM_OBSTRUCTION)
        .value("BOND_OBSTRUCTION", DonorPathStatus::BOND_OBSTRUCTION)
        .value("RING_PIERCING", DonorPathStatus::RING_PIERCING);
    py::enum_<PlacementProposalKind>(module, "PlacementProposalKind")
        .value("CURRENT", PlacementProposalKind::CURRENT)
        .value("TARGET_SPHERE", PlacementProposalKind::TARGET_SPHERE)
        .value(
            "SPHERE_INTERSECTION",
            PlacementProposalKind::SPHERE_INTERSECTION
        )
        .value("LEAST_SQUARES", PlacementProposalKind::LEAST_SQUARES)
        .value(
            "FIBONACCI_FALLBACK",
            PlacementProposalKind::FIBONACCI_FALLBACK
        );

    py::class_<MetalPlacementOptions>(module, "MetalPlacementOptions")
        .def(py::init<>())
        .def(py::init([](
            std::size_t maximum_candidate_count,
            std::size_t fibonacci_direction_count,
            std::size_t sphere_intersection_count,
            std::size_t least_squares_iteration_count,
            std::size_t maximum_actionable_ring_size,
            double coordination_distance_scale,
            double coordination_distance_ratio_minimum,
            double coordination_distance_ratio_maximum,
            double absolute_center_clearance_angstrom,
            double center_covalent_radius_scale,
            double minimum_path_atom_clearance,
            double minimum_path_bond_clearance,
            double broad_phase_skin_angstrom,
            double duplicate_tolerance_angstrom,
            bool retain_candidate_evidence,
            double geometry_absolute_length,
            double geometry_relative_length,
            double geometry_parameter,
            double geometry_machine_epsilon_factor,
            double geometry_predicate_guard_factor,
            double geometry_planarity_factor,
            double geometry_winding_residual,
            double geometry_intersection_merge_factor,
            double geometry_aabb_padding_factor,
            std::size_t surface_maximum_cycle_vertices,
            std::size_t surface_maximum_surface_count,
            std::size_t surface_maximum_segment_triangle_tests,
            std::size_t surface_maximum_triangle_pair_tests
        ) {
            MetalPlacementOptions options;
            options.maximum_candidate_count = maximum_candidate_count;
            options.fibonacci_direction_count = fibonacci_direction_count;
            options.sphere_intersection_count = sphere_intersection_count;
            options.least_squares_iteration_count =
                least_squares_iteration_count;
            options.maximum_actionable_ring_size =
                maximum_actionable_ring_size;
            options.coordination_distance_scale = coordination_distance_scale;
            options.coordination_distance_ratio_minimum =
                coordination_distance_ratio_minimum;
            options.coordination_distance_ratio_maximum =
                coordination_distance_ratio_maximum;
            options.absolute_center_clearance_angstrom =
                absolute_center_clearance_angstrom;
            options.center_covalent_radius_scale = center_covalent_radius_scale;
            options.minimum_path_atom_clearance = minimum_path_atom_clearance;
            options.minimum_path_bond_clearance = minimum_path_bond_clearance;
            options.broad_phase_skin_angstrom = broad_phase_skin_angstrom;
            options.duplicate_tolerance_angstrom =
                duplicate_tolerance_angstrom;
            options.retain_candidate_evidence = retain_candidate_evidence;
            options.geometry_tolerances = {
                geometry_absolute_length,
                geometry_relative_length,
                geometry_parameter,
                geometry_machine_epsilon_factor,
                geometry_predicate_guard_factor,
                geometry_planarity_factor,
                geometry_winding_residual,
                geometry_intersection_merge_factor,
                geometry_aabb_padding_factor,
            };
            options.surface_limits = {
                surface_maximum_cycle_vertices,
                surface_maximum_surface_count,
                surface_maximum_segment_triangle_tests,
                surface_maximum_triangle_pair_tests,
            };
            options.validate();
            return options;
        }),
        py::arg("maximum_candidate_count"),
        py::arg("fibonacci_direction_count"),
        py::arg("sphere_intersection_count"),
        py::arg("least_squares_iteration_count"),
        py::arg("maximum_actionable_ring_size"),
        py::arg("coordination_distance_scale"),
        py::arg("coordination_distance_ratio_minimum"),
        py::arg("coordination_distance_ratio_maximum"),
        py::arg("absolute_center_clearance_angstrom"),
        py::arg("center_covalent_radius_scale"),
        py::arg("minimum_path_atom_clearance"),
        py::arg("minimum_path_bond_clearance"),
        py::arg("broad_phase_skin_angstrom"),
        py::arg("duplicate_tolerance_angstrom"),
        py::arg("retain_candidate_evidence"),
        py::arg("geometry_absolute_length"),
        py::arg("geometry_relative_length"),
        py::arg("geometry_parameter"),
        py::arg("geometry_machine_epsilon_factor"),
        py::arg("geometry_predicate_guard_factor"),
        py::arg("geometry_planarity_factor"),
        py::arg("geometry_winding_residual"),
        py::arg("geometry_intersection_merge_factor"),
        py::arg("geometry_aabb_padding_factor"),
        py::arg("surface_maximum_cycle_vertices"),
        py::arg("surface_maximum_surface_count"),
        py::arg("surface_maximum_segment_triangle_tests"),
        py::arg("surface_maximum_triangle_pair_tests"))
        .def_readonly(
            "maximum_candidate_count",
            &MetalPlacementOptions::maximum_candidate_count
        )
        .def_readonly(
            "fibonacci_direction_count",
            &MetalPlacementOptions::fibonacci_direction_count
        )
        .def_readonly(
            "sphere_intersection_count",
            &MetalPlacementOptions::sphere_intersection_count
        )
        .def_readonly(
            "least_squares_iteration_count",
            &MetalPlacementOptions::least_squares_iteration_count
        )
        .def_readonly(
            "maximum_actionable_ring_size",
            &MetalPlacementOptions::maximum_actionable_ring_size
        )
        .def_readonly(
            "coordination_distance_scale",
            &MetalPlacementOptions::coordination_distance_scale
        )
        .def_readonly(
            "coordination_distance_ratio_minimum",
            &MetalPlacementOptions::coordination_distance_ratio_minimum
        )
        .def_readonly(
            "coordination_distance_ratio_maximum",
            &MetalPlacementOptions::coordination_distance_ratio_maximum
        )
        .def_readonly(
            "absolute_center_clearance_angstrom",
            &MetalPlacementOptions::absolute_center_clearance_angstrom
        )
        .def_readonly(
            "center_covalent_radius_scale",
            &MetalPlacementOptions::center_covalent_radius_scale
        )
        .def_readonly(
            "minimum_path_atom_clearance",
            &MetalPlacementOptions::minimum_path_atom_clearance
        )
        .def_readonly(
            "minimum_path_bond_clearance",
            &MetalPlacementOptions::minimum_path_bond_clearance
        )
        .def_readonly(
            "broad_phase_skin_angstrom",
            &MetalPlacementOptions::broad_phase_skin_angstrom
        )
        .def_readonly(
            "duplicate_tolerance_angstrom",
            &MetalPlacementOptions::duplicate_tolerance_angstrom
        )
        .def_readonly(
            "retain_candidate_evidence",
            &MetalPlacementOptions::retain_candidate_evidence
        );

    py::class_<DonorApproachEvidence>(module, "DonorApproachEvidence")
        .def_readonly(
            "neighbour_index",
            &DonorApproachEvidence::neighbour_index
        )
        .def_readonly(
            "metal_donor_neighbour_angle_degrees",
            &DonorApproachEvidence::metal_donor_neighbour_angle_degrees
        );
    py::class_<DonorPathEvidence>(module, "DonorPathEvidence")
        .def_readonly("donor_index", &DonorPathEvidence::donor_index)
        .def_readonly("group_index", &DonorPathEvidence::group_index)
        .def_readonly("status", &DonorPathEvidence::status)
        .def_readonly(
            "distance_reachable",
            &DonorPathEvidence::distance_reachable
        )
        .def_readonly("atom_obstructed", &DonorPathEvidence::atom_obstructed)
        .def_readonly("bond_obstructed", &DonorPathEvidence::bond_obstructed)
        .def_readonly(
            "target_distance_angstrom",
            &DonorPathEvidence::target_distance_angstrom
        )
        .def_readonly("distance_angstrom", &DonorPathEvidence::distance_angstrom)
        .def_readonly("distance_ratio", &DonorPathEvidence::distance_ratio)
        .def_readonly(
            "normalized_atom_clearance",
            &DonorPathEvidence::normalized_atom_clearance
        )
        .def_readonly(
            "normalized_bond_clearance",
            &DonorPathEvidence::normalized_bond_clearance
        )
        .def_readonly(
            "definite_piercing_count",
            &DonorPathEvidence::definite_piercing_count
        )
        .def_readonly(
            "undetermined_relation_count",
            &DonorPathEvidence::undetermined_relation_count
        )
        .def_readonly("atom_pair_count", &DonorPathEvidence::atom_pair_count)
        .def_readonly(
            "atom_aabb_rejected_pair_count",
            &DonorPathEvidence::atom_aabb_rejected_pair_count
        )
        .def_readonly("bond_pair_count", &DonorPathEvidence::bond_pair_count)
        .def_readonly(
            "bond_aabb_rejected_pair_count",
            &DonorPathEvidence::bond_aabb_rejected_pair_count
        )
        .def_readonly("cycle_pair_count", &DonorPathEvidence::cycle_pair_count)
        .def_readonly(
            "cycle_aabb_rejected_pair_count",
            &DonorPathEvidence::cycle_aabb_rejected_pair_count
        )
        .def_readonly("approach_angles", &DonorPathEvidence::approach_angles);
    py::class_<DonorPairEvidence>(module, "DonorPairEvidence")
        .def_readonly(
            "first_donor_index",
            &DonorPairEvidence::first_donor_index
        )
        .def_readonly(
            "second_donor_index",
            &DonorPairEvidence::second_donor_index
        )
        .def_readonly(
            "donor_separation_angstrom",
            &DonorPairEvidence::donor_separation_angstrom
        )
        .def_readonly(
            "target_distance_sum_angstrom",
            &DonorPairEvidence::target_distance_sum_angstrom
        )
        .def_readonly(
            "target_distance_difference_angstrom",
            &DonorPairEvidence::target_distance_difference_angstrom
        )
        .def_readonly(
            "target_shells_intersect",
            &DonorPairEvidence::target_shells_intersect
        )
        .def_readonly(
            "donor_metal_donor_angle_degrees",
            &DonorPairEvidence::donor_metal_donor_angle_degrees
        );
    py::class_<PlacementCandidateEvidence>(
        module,
        "PlacementCandidateEvidence"
    )
        .def_readonly("coordinates", &PlacementCandidateEvidence::coordinates)
        .def_readonly("proposal_kind", &PlacementCandidateEvidence::proposal_kind)
        .def_readonly("status", &PlacementCandidateEvidence::status)
        .def_readonly(
            "excluded_large_cycle_count",
            &PlacementCandidateEvidence::excluded_large_cycle_count
        )
        .def_readonly(
            "covered_group_count",
            &PlacementCandidateEvidence::covered_group_count
        )
        .def_readonly("safe_donor_count", &PlacementCandidateEvidence::safe_donor_count)
        .def_readonly(
            "out_of_range_donor_count",
            &PlacementCandidateEvidence::out_of_range_donor_count
        )
        .def_readonly(
            "atom_obstruction_count",
            &PlacementCandidateEvidence::atom_obstruction_count
        )
        .def_readonly(
            "bond_obstruction_count",
            &PlacementCandidateEvidence::bond_obstruction_count
        )
        .def_readonly(
            "definite_piercing_count",
            &PlacementCandidateEvidence::definite_piercing_count
        )
        .def_readonly(
            "hard_obstruction_count",
            &PlacementCandidateEvidence::hard_obstruction_count
        )
        .def_readonly(
            "minimum_normalized_clearance",
            &PlacementCandidateEvidence::minimum_normalized_clearance
        )
        .def_readonly(
            "worst_distance_deviation",
            &PlacementCandidateEvidence::worst_distance_deviation
        )
        .def_readonly(
            "rms_distance_deviation",
            &PlacementCandidateEvidence::rms_distance_deviation
        )
        .def_readonly(
            "undetermined_relation_count",
            &PlacementCandidateEvidence::undetermined_relation_count
        )
        .def_readonly(
            "atom_pair_count",
            &PlacementCandidateEvidence::atom_pair_count
        )
        .def_readonly(
            "atom_aabb_rejected_pair_count",
            &PlacementCandidateEvidence::atom_aabb_rejected_pair_count
        )
        .def_readonly(
            "bond_pair_count",
            &PlacementCandidateEvidence::bond_pair_count
        )
        .def_readonly(
            "bond_aabb_rejected_pair_count",
            &PlacementCandidateEvidence::bond_aabb_rejected_pair_count
        )
        .def_readonly(
            "cycle_pair_count",
            &PlacementCandidateEvidence::cycle_pair_count
        )
        .def_readonly(
            "cycle_aabb_rejected_pair_count",
            &PlacementCandidateEvidence::cycle_aabb_rejected_pair_count
        )
        .def_readonly(
            "displacement_angstrom",
            &PlacementCandidateEvidence::displacement_angstrom
        )
        .def_readonly("proposal_ordinal", &PlacementCandidateEvidence::proposal_ordinal)
        .def_readonly("donor_paths", &PlacementCandidateEvidence::donor_paths)
        .def_readonly("donor_pairs", &PlacementCandidateEvidence::donor_pairs);
    py::class_<MetalPlacementResult>(module, "MetalPlacementResult")
        .def_readonly("metal_index", &MetalPlacementResult::metal_index)
        .def_readonly("status", &MetalPlacementResult::status)
        .def_readonly(
            "original_coordinates",
            &MetalPlacementResult::original_coordinates
        )
        .def_readonly(
            "selected_coordinates",
            &MetalPlacementResult::selected_coordinates
        )
        .def_readonly("moved", &MetalPlacementResult::moved)
        .def_readonly(
            "candidates_evaluated",
            &MetalPlacementResult::candidates_evaluated
        )
        .def_readonly(
            "selected_evidence",
            &MetalPlacementResult::selected_evidence
        )
        .def_readonly(
            "retained_candidates",
            &MetalPlacementResult::retained_candidates
        )
        .def_readonly(
            "excluded_large_cycle_count",
            &MetalPlacementResult::excluded_large_cycle_count
        )
        .def_readonly("warning_codes", &MetalPlacementResult::warning_codes);
    py::class_<MetalPlacementReport>(module, "MetalPlacementReport")
        .def_readonly("metals", &MetalPlacementReport::metals)
        .def_property_readonly(
            "selected_coordinates",
            [](const MetalPlacementReport& report) {
                return coordinate_array(report.selected_coordinates);
            }
        )
        .def_readonly("warning_codes", &MetalPlacementReport::warning_codes);

    module.def(
        "assess_metal_position",
        [](const StructureSession& session,
           std::int32_t metal_index,
           const Coordinate& candidate,
           const MetalPlacementOptions& options) {
            py::gil_scoped_release release;
            return assess_metal_position(
                session,
                metal_index,
                candidate,
                options
            );
        },
        py::arg("session"),
        py::arg("metal_index"),
        py::arg("candidate"),
        py::arg("options") = MetalPlacementOptions{}
    );
    module.def(
        "place_metal",
        [](StructureSession& session,
           std::int32_t metal_index,
           const MetalPlacementOptions& options) {
            py::gil_scoped_release release;
            return place_metal(session, metal_index, options);
        },
        py::arg("session"),
        py::arg("metal_index"),
        py::arg("options") = MetalPlacementOptions{}
    );
    module.def(
        "place_metals",
        [](StructureSession& session, const MetalPlacementOptions& options) {
            py::gil_scoped_release release;
            return place_metals(session, options);
        },
        py::arg("session"),
        py::arg("options") = MetalPlacementOptions{}
    );

    py::enum_<FrameDetail>(module, "FrameDetail")
        .value("NONE", FrameDetail::NONE)
        .value("OPTIMIZATION", FrameDetail::OPTIMIZATION)
        .value("ALL_ATTEMPTS", FrameDetail::ALL_ATTEMPTS);
    py::enum_<NativeStageStatus>(module, "NativeStageStatus")
        .value("COMPLETED", NativeStageStatus::COMPLETED)
        .value("PARTIAL", NativeStageStatus::PARTIAL)
        .value("FAILED", NativeStageStatus::FAILED);
    py::enum_<NativeRingGraphScope>(module, "NativeRingGraphScope")
        .value(
            "LIGAND_SKELETON",
            NativeRingGraphScope::LIGAND_SKELETON
        )
        .value("FULL_GRAPH", NativeRingGraphScope::FULL_GRAPH);
    py::enum_<hotpot::geometry::PiercingState>(
        module,
        "NativePiercingState",
        py::module_local()
    )
        .value("PIERCES", hotpot::geometry::PiercingState::PIERCES)
        .value(
            "DOES_NOT_PIERCE",
            hotpot::geometry::PiercingState::DOES_NOT_PIERCE
        )
        .value(
            "UNDETERMINED",
            hotpot::geometry::PiercingState::UNDETERMINED
        );
    py::enum_<hotpot::geometry::SegmentCycleIndeterminacy>(
        module,
        "NativeSegmentCycleIndeterminacy",
        py::module_local()
    )
        .value(
            "NONFINITE_INPUT",
            hotpot::geometry::SegmentCycleIndeterminacy::NONFINITE_INPUT
        )
        .value(
            "NUMERIC_BAND",
            hotpot::geometry::SegmentCycleIndeterminacy::NUMERIC_BAND
        )
        .value(
            "TOLERANCE_DOMAIN",
            hotpot::geometry::SegmentCycleIndeterminacy::TOLERANCE_DOMAIN
        )
        .value(
            "DEGENERATE_CYCLE",
            hotpot::geometry::SegmentCycleIndeterminacy::DEGENERATE_CYCLE
        )
        .value(
            "DEGENERATE_SEGMENT",
            hotpot::geometry::SegmentCycleIndeterminacy::DEGENERATE_SEGMENT
        )
        .value(
            "DEGENERATE_TRIANGLE",
            hotpot::geometry::SegmentCycleIndeterminacy::DEGENERATE_TRIANGLE
        )
        .value(
            "SELF_INTERSECTION",
            hotpot::geometry::SegmentCycleIndeterminacy::SELF_INTERSECTION
        )
        .value(
            "SURFACE_DISAGREEMENT",
            hotpot::geometry::SegmentCycleIndeterminacy::SURFACE_DISAGREEMENT
        )
        .value(
            "INCOMPLETE_SURFACE_FAMILY",
            hotpot::geometry::SegmentCycleIndeterminacy::
                INCOMPLETE_SURFACE_FAMILY
        )
        .value(
            "SURFACE_CONSTRUCTION",
            hotpot::geometry::SegmentCycleIndeterminacy::SURFACE_CONSTRUCTION
        );
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

    py::class_<RingScreeningOptions>(module, "RingScreeningOptions")
        .def(py::init([](
            std::size_t maximum_actionable_ring_size,
            std::size_t maximum_relevant_cycle_count,
            double geometry_absolute_length,
            double geometry_relative_length,
            double geometry_parameter,
            double geometry_machine_epsilon_factor,
            double geometry_predicate_guard_factor,
            double geometry_planarity_factor,
            double geometry_winding_residual,
            double geometry_intersection_merge_factor,
            double geometry_aabb_padding_factor,
            std::size_t surface_maximum_cycle_vertices,
            std::size_t surface_maximum_surface_count,
            std::size_t surface_maximum_segment_triangle_tests,
            std::size_t surface_maximum_triangle_pair_tests
        ) {
            RingScreeningOptions options{
                maximum_actionable_ring_size,
                maximum_relevant_cycle_count,
                {
                    geometry_absolute_length,
                    geometry_relative_length,
                    geometry_parameter,
                    geometry_machine_epsilon_factor,
                    geometry_predicate_guard_factor,
                    geometry_planarity_factor,
                    geometry_winding_residual,
                    geometry_intersection_merge_factor,
                    geometry_aabb_padding_factor,
                },
                {
                    surface_maximum_cycle_vertices,
                    surface_maximum_surface_count,
                    surface_maximum_segment_triangle_tests,
                    surface_maximum_triangle_pair_tests,
                },
            };
            options.validate();
            return options;
        }),
        py::arg("maximum_actionable_ring_size"),
        py::arg("maximum_relevant_cycle_count"),
        py::arg("geometry_absolute_length"),
        py::arg("geometry_relative_length"),
        py::arg("geometry_parameter"),
        py::arg("geometry_machine_epsilon_factor"),
        py::arg("geometry_predicate_guard_factor"),
        py::arg("geometry_planarity_factor"),
        py::arg("geometry_winding_residual"),
        py::arg("geometry_intersection_merge_factor"),
        py::arg("geometry_aabb_padding_factor"),
        py::arg("surface_maximum_cycle_vertices"),
        py::arg("surface_maximum_surface_count"),
        py::arg("surface_maximum_segment_triangle_tests"),
        py::arg("surface_maximum_triangle_pair_tests"))
        .def_readonly(
            "maximum_actionable_ring_size",
            &RingScreeningOptions::maximum_actionable_ring_size
        )
        .def_readonly(
            "maximum_relevant_cycle_count",
            &RingScreeningOptions::maximum_relevant_cycle_count
        )
        .def_property_readonly(
            "geometry_absolute_length",
            [](const RingScreeningOptions& options) {
                return options.geometry_tolerances.absolute_length;
            }
        )
        .def_property_readonly(
            "geometry_relative_length",
            [](const RingScreeningOptions& options) {
                return options.geometry_tolerances.relative_length;
            }
        )
        .def_property_readonly(
            "geometry_parameter",
            [](const RingScreeningOptions& options) {
                return options.geometry_tolerances.parameter;
            }
        )
        .def_property_readonly(
            "geometry_machine_epsilon_factor",
            [](const RingScreeningOptions& options) {
                return options.geometry_tolerances.machine_epsilon_factor;
            }
        )
        .def_property_readonly(
            "geometry_predicate_guard_factor",
            [](const RingScreeningOptions& options) {
                return options.geometry_tolerances.predicate_guard_factor;
            }
        )
        .def_property_readonly(
            "geometry_planarity_factor",
            [](const RingScreeningOptions& options) {
                return options.geometry_tolerances.planarity_factor;
            }
        )
        .def_property_readonly(
            "geometry_winding_residual",
            [](const RingScreeningOptions& options) {
                return options.geometry_tolerances.winding_residual;
            }
        )
        .def_property_readonly(
            "geometry_intersection_merge_factor",
            [](const RingScreeningOptions& options) {
                return options.geometry_tolerances.intersection_merge_factor;
            }
        )
        .def_property_readonly(
            "geometry_aabb_padding_factor",
            [](const RingScreeningOptions& options) {
                return options.geometry_tolerances.aabb_padding_factor;
            }
        )
        .def_property_readonly(
            "surface_maximum_cycle_vertices",
            [](const RingScreeningOptions& options) {
                return options.surface_limits.maximum_cycle_vertices;
            }
        )
        .def_property_readonly(
            "surface_maximum_surface_count",
            [](const RingScreeningOptions& options) {
                return options.surface_limits.maximum_surface_count;
            }
        )
        .def_property_readonly(
            "surface_maximum_segment_triangle_tests",
            [](const RingScreeningOptions& options) {
                return options.surface_limits.maximum_segment_triangle_tests;
            }
        )
        .def_property_readonly(
            "surface_maximum_triangle_pair_tests",
            [](const RingScreeningOptions& options) {
                return options.surface_limits.maximum_triangle_pair_tests;
            }
        );

    py::class_<CoordinationStageOptions>(module, "CoordinationStageOptions")
        .def(py::init([](
            std::string forcefield,
            std::size_t attempt_limit,
            std::size_t relaxation_steps,
            double perturb_sigma,
            NativeTrajectoryStart trajectory_start,
            FrameDetail frame_detail,
            double torsion_singularity_threshold,
            double torsion_repair_angle_radians,
            MetalPlacementOptions placement
        ) {
            CoordinationStageOptions options{
                std::move(forcefield),
                attempt_limit,
                relaxation_steps,
                perturb_sigma,
                trajectory_start,
                frame_detail,
                torsion_singularity_threshold,
                torsion_repair_angle_radians,
                std::move(placement),
            };
            options.validate();
            return options;
        }),
        py::arg("forcefield"),
        py::arg("attempt_limit"),
        py::arg("relaxation_steps"),
        py::arg("perturb_sigma"),
        py::arg("trajectory_start"),
        py::arg("frame_detail"),
        py::arg("torsion_singularity_threshold"),
        py::arg("torsion_repair_angle_radians"),
        py::arg("placement") = MetalPlacementOptions{})
        .def_readonly("forcefield", &CoordinationStageOptions::forcefield)
        .def_readonly("attempt_limit", &CoordinationStageOptions::attempt_limit)
        .def_readonly(
            "relaxation_steps", &CoordinationStageOptions::relaxation_steps
        )
        .def_readonly("perturb_sigma", &CoordinationStageOptions::perturb_sigma)
        .def_readonly(
            "trajectory_start", &CoordinationStageOptions::trajectory_start
        )
        .def_readonly("frame_detail", &CoordinationStageOptions::frame_detail)
        .def_readonly(
            "torsion_singularity_threshold",
            &CoordinationStageOptions::torsion_singularity_threshold
        )
        .def_readonly(
            "torsion_repair_angle_radians",
            &CoordinationStageOptions::torsion_repair_angle_radians
        )
        .def_readonly("placement", &CoordinationStageOptions::placement);

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
            std::optional<OptimizationStoppingOptions> stopping,
            double torsion_singularity_threshold,
            double torsion_repair_angle_radians,
            RingScreeningOptions ring_screening
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
                torsion_singularity_threshold,
                torsion_repair_angle_radians,
                std::move(ring_screening),
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
        py::arg("stopping"),
        py::arg("torsion_singularity_threshold"),
        py::arg("torsion_repair_angle_radians"),
        py::arg("ring_screening"))
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
        .def_readonly("stopping", &ComplexOptimizationOptions::stopping)
        .def_readonly(
            "torsion_singularity_threshold",
            &ComplexOptimizationOptions::torsion_singularity_threshold
        )
        .def_readonly(
            "torsion_repair_angle_radians",
            &ComplexOptimizationOptions::torsion_repair_angle_radians
        )
        .def_readonly(
            "ring_screening", &ComplexOptimizationOptions::ring_screening
        );
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
    py::class_<NativeBondRingFinding>(module, "NativeBondRingFinding")
        .def(py::init([](
            std::size_t ring_index,
            std::vector<std::size_t> ring_atom_indices,
            BondIndex bond_key,
            hotpot::geometry::PiercingState state,
            std::vector<hotpot::geometry::SegmentCycleIndeterminacy>
                indeterminacy_causes,
            bool aabb_separated,
            bool surface_complete
        ) {
            NativeBondRingFinding finding{
                ring_index,
                std::move(ring_atom_indices),
                bond_key,
                state,
                std::move(indeterminacy_causes),
                aabb_separated,
                surface_complete,
            };
            finding.validate();
            return finding;
        }),
        py::arg("ring_index"),
        py::arg("ring_atom_indices"),
        py::arg("bond_key"),
        py::arg("state"),
        py::arg("indeterminacy_causes"),
        py::arg("aabb_separated"),
        py::arg("surface_complete"))
        .def_readonly("ring_index", &NativeBondRingFinding::ring_index)
        .def_readonly(
            "ring_atom_indices", &NativeBondRingFinding::ring_atom_indices
        )
        .def_property_readonly(
            "bond_key",
            [](const NativeBondRingFinding& finding) {
                return py::make_tuple(
                    finding.bond_key[0], finding.bond_key[1]
                );
            }
        )
        .def_readonly("state", &NativeBondRingFinding::state)
        .def_readonly(
            "indeterminacy_causes",
            &NativeBondRingFinding::indeterminacy_causes
        )
        .def_readonly(
            "aabb_separated", &NativeBondRingFinding::aabb_separated
        )
        .def_readonly(
            "surface_complete", &NativeBondRingFinding::surface_complete
        );

    py::class_<NativeRingCheckpointReport>(
        module, "NativeRingCheckpointReport"
    )
        .def(py::init([](
            hotpot::geometry::PiercingState state,
            NativeRingGraphScope scope,
            std::size_t maximum_actionable_ring_size,
            std::size_t maximum_relevant_cycle_count,
            std::size_t relevant_cycle_count,
            std::size_t selected_ring_count,
            std::size_t excluded_ring_count,
            std::size_t active_bond_count,
            std::size_t candidate_pair_count,
            std::size_t aabb_separated_pair_count,
            std::size_t exact_pair_count,
            std::size_t piercing_pair_count,
            std::size_t does_not_pierce_pair_count,
            std::size_t undetermined_pair_count,
            bool scan_complete,
            std::vector<NativeBondRingFinding> actionable_findings
        ) {
            NativeRingCheckpointReport report{
                state,
                scope,
                maximum_actionable_ring_size,
                maximum_relevant_cycle_count,
                relevant_cycle_count,
                selected_ring_count,
                excluded_ring_count,
                active_bond_count,
                candidate_pair_count,
                aabb_separated_pair_count,
                exact_pair_count,
                piercing_pair_count,
                does_not_pierce_pair_count,
                undetermined_pair_count,
                scan_complete,
                std::move(actionable_findings),
            };
            report.validate();
            return report;
        }),
        py::arg("state"),
        py::arg("scope"),
        py::arg("maximum_actionable_ring_size"),
        py::arg("maximum_relevant_cycle_count"),
        py::arg("relevant_cycle_count"),
        py::arg("selected_ring_count"),
        py::arg("excluded_ring_count"),
        py::arg("active_bond_count"),
        py::arg("candidate_pair_count"),
        py::arg("aabb_separated_pair_count"),
        py::arg("exact_pair_count"),
        py::arg("piercing_pair_count"),
        py::arg("does_not_pierce_pair_count"),
        py::arg("undetermined_pair_count"),
        py::arg("scan_complete"),
        py::arg("actionable_findings"))
        .def_readonly("state", &NativeRingCheckpointReport::state)
        .def_readonly("scope", &NativeRingCheckpointReport::scope)
        .def_readonly(
            "maximum_actionable_ring_size",
            &NativeRingCheckpointReport::maximum_actionable_ring_size
        )
        .def_readonly(
            "maximum_relevant_cycle_count",
            &NativeRingCheckpointReport::maximum_relevant_cycle_count
        )
        .def_readonly(
            "relevant_cycle_count",
            &NativeRingCheckpointReport::relevant_cycle_count
        )
        .def_readonly(
            "selected_ring_count",
            &NativeRingCheckpointReport::selected_ring_count
        )
        .def_readonly(
            "excluded_ring_count",
            &NativeRingCheckpointReport::excluded_ring_count
        )
        .def_readonly(
            "active_bond_count", &NativeRingCheckpointReport::active_bond_count
        )
        .def_readonly(
            "candidate_pair_count",
            &NativeRingCheckpointReport::candidate_pair_count
        )
        .def_readonly(
            "aabb_separated_pair_count",
            &NativeRingCheckpointReport::aabb_separated_pair_count
        )
        .def_readonly(
            "exact_pair_count", &NativeRingCheckpointReport::exact_pair_count
        )
        .def_readonly(
            "piercing_pair_count",
            &NativeRingCheckpointReport::piercing_pair_count
        )
        .def_readonly(
            "does_not_pierce_pair_count",
            &NativeRingCheckpointReport::does_not_pierce_pair_count
        )
        .def_readonly(
            "undetermined_pair_count",
            &NativeRingCheckpointReport::undetermined_pair_count
        )
        .def_readonly("scan_complete", &NativeRingCheckpointReport::scan_complete)
        .def_readonly(
            "actionable_findings",
            &NativeRingCheckpointReport::actionable_findings
        );

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
            NativeTrajectoryBatch trajectory,
            double elapsed_seconds
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
            result.bond_count = result.final_active_coordination_mask.size();
            result.elapsed_seconds = elapsed_seconds;
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
        py::arg("trajectory"),
        py::arg("elapsed_seconds") = 0.0)
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
        .def_readonly("trajectory", &CoordinationStageResult::trajectory)
        .def_readonly("bond_count", &CoordinationStageResult::bond_count)
        .def_readonly(
            "placement_report", &CoordinationStageResult::placement_report
        )
        .def_readonly(
            "elapsed_seconds", &CoordinationStageResult::elapsed_seconds
        );

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
            NativeTrajectoryBatch trajectory,
            NativeRingCheckpointReport final_checkpoint,
            double elapsed_seconds
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
                std::move(final_checkpoint),
                elapsed_seconds,
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
        py::arg("trajectory"),
        py::arg("final_checkpoint"),
        py::arg("elapsed_seconds") = 0.0)
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
        .def_readonly("trajectory", &ComplexOptimizationResult::trajectory)
        .def_readonly(
            "final_checkpoint", &ComplexOptimizationResult::final_checkpoint
        )
        .def_readonly(
            "elapsed_seconds", &ComplexOptimizationResult::elapsed_seconds
        );

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
    module.def(
        "restore_coordination",
        [](StructureSession& session,
           const CoordinationStageOptions& options,
           const PerturbationOffsetBatch& perturbation_offsets) {
            py::gil_scoped_release release;
            return restore_coordination(
                session,
                options,
                perturbation_offsets
            );
        },
        py::arg("session"),
        py::arg("options"),
        py::arg("perturbation_offsets")
    );
    module.def(
        "optimize_complex",
        [](StructureSession& session,
           const ComplexOptimizationOptions& options,
           const PerturbationOffsetBatch& untangling_offsets,
           const PerturbationOffsetBatch& optimization_offsets) {
            py::gil_scoped_release release;
            return optimize_complex(
                session,
                options,
                untangling_offsets,
                optimization_offsets
            );
        },
        py::arg("session"),
        py::arg("options"),
        py::arg("untangling_offsets"),
        py::arg("optimization_offsets")
    );
    module.def(
        "run_complex_workflow",
        [](StructureSession& session,
           const CoordinationStageOptions& coordination_options,
           const ComplexOptimizationOptions& optimization_options,
           const PerturbationOffsetBatch& coordination_offsets,
           const PerturbationOffsetBatch& untangling_offsets,
           const PerturbationOffsetBatch& optimization_offsets) {
            py::gil_scoped_release release;
            return run_complex_workflow(
                session,
                coordination_options,
                optimization_options,
                coordination_offsets,
                untangling_offsets,
                optimization_offsets
            );
        },
        py::arg("session"),
        py::arg("coordination_options"),
        py::arg("optimization_options"),
        py::arg("coordination_offsets"),
        py::arg("untangling_offsets"),
        py::arg("optimization_offsets")
    );
}


}  // namespace hotpot::forcefields
