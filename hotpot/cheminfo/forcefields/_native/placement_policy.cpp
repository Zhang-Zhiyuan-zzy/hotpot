#include "placement_policy.hpp"

#include "radii.hpp"

#include "../../geometry/_native/construction.hpp"
#include "../../geometry/_native/primitives.hpp"
#include "../../geometry/_native/spatial.hpp"
#include "../../geometry/_native/vector_math.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>


namespace hotpot::forcefields {
namespace {


using hotpot::geometry::ArrayView;
using hotpot::geometry::PiercingState;
using hotpot::geometry::Point3;
using hotpot::geometry::Segment3;
using hotpot::geometry::detail::finite;
using hotpot::geometry::detail::point_distance;
using detail::MetalPlacementTarget;
using detail::PreparedRingWorkspace;
using detail::PlacementDonorTarget;
using detail::PlacementEvaluationWorkspace;
using detail::PlacementProposal;
using detail::RingGraphScope;
using detail::RingWorkspaceOptions;
using detail::default_maximum_relevant_cycle_count;
using detail::prepare_ring_workspace;
using detail::select_metal_placement_target;


bool declared_metal(
    const ComplexSessionInput& input,
    std::size_t atom_index
) {
    return std::find(
        input.metal_indices.begin(),
        input.metal_indices.end(),
        static_cast<std::int32_t>(atom_index)
    ) != input.metal_indices.end();
}


bool ligand_skeleton_bond(
    const ComplexSessionInput& input,
    const BondIndex& bond
) {
    return !declared_metal(input, static_cast<std::size_t>(bond[0]))
        && !declared_metal(input, static_cast<std::size_t>(bond[1]));
}


PreparedRingWorkspace prepare_ligand_cycles(
    const ComplexSessionInput& input,
    const std::vector<Coordinate>& coordinates,
    const std::vector<std::uint8_t>& active_ligand_bond_mask,
    const MetalPlacementOptions& options
) {
    return prepare_ring_workspace(
        input,
        coordinates,
        active_ligand_bond_mask,
        std::vector<std::uint8_t>(
            input.intended_coordination_bond_count(),
            0
        ),
        RingWorkspaceOptions{
            RingGraphScope::LIGAND_SKELETON,
            options.maximum_actionable_ring_size,
            default_maximum_relevant_cycle_count,
            options.geometry_tolerances,
            options.surface_limits,
        }
    );
}


struct DonorBroadPhaseMasks {
    std::vector<std::uint8_t> atoms;
    std::vector<std::uint8_t> bonds;
    std::vector<std::uint8_t> cycles;
};


struct CandidateBroadPhaseMasks {
    std::vector<std::uint8_t> center_atoms;
    std::vector<std::uint8_t> center_bonds;
    std::vector<DonorBroadPhaseMasks> donor_paths;
};


struct BroadPhaseLocation {
    std::size_t candidate;
    std::size_t donor;
    std::size_t obstacle;
};


void assign_separation_mask(
    std::vector<hotpot::geometry::Aabb>& first,
    std::vector<hotpot::geometry::Aabb>& second,
    std::vector<double>& paddings,
    const std::vector<BroadPhaseLocation>& locations,
    std::vector<CandidateBroadPhaseMasks>& candidates,
    std::vector<std::uint8_t> DonorBroadPhaseMasks::*member
) {
    const auto separated = hotpot::geometry::aabb_separation_mask(
        ArrayView<hotpot::geometry::Aabb>(first),
        ArrayView<hotpot::geometry::Aabb>(second),
        ArrayView<double>(paddings)
    );
    for (std::size_t index = 0; index < locations.size(); ++index) {
        const auto& location = locations[index];
        (candidates[location.candidate].donor_paths[location.donor].*member)[
            location.obstacle
        ] = separated[index];
    }
}


std::vector<CandidateBroadPhaseMasks> build_broad_phase_batch(
    const PlacementEvaluationWorkspace& workspace,
    const std::vector<PlacementProposal>& proposals,
    const MetalPlacementOptions& options
) {
    const std::size_t atom_count = workspace.input->atom_count();
    const std::size_t bond_count = workspace.active_ligand_bonds.size();
    const std::size_t donor_count = workspace.target.donors.size();
    const std::size_t cycle_count =
        workspace.cycles->prepared_cycles.cycle_count();
    const auto metal = static_cast<std::size_t>(workspace.target.metal_index);
    std::vector<CandidateBroadPhaseMasks> result;
    result.reserve(proposals.size());
    for (std::size_t candidate = 0; candidate < proposals.size(); ++candidate) {
        CandidateBroadPhaseMasks masks{
            std::vector<std::uint8_t>(atom_count, 0),
            std::vector<std::uint8_t>(bond_count, 0),
            {},
        };
        masks.donor_paths.reserve(donor_count);
        for (std::size_t donor = 0; donor < donor_count; ++donor) {
            masks.donor_paths.push_back({
                std::vector<std::uint8_t>(atom_count, 0),
                std::vector<std::uint8_t>(bond_count, 0),
                std::vector<std::uint8_t>(cycle_count, 0),
            });
        }
        result.push_back(std::move(masks));
    }

    std::vector<hotpot::geometry::Aabb> first;
    std::vector<hotpot::geometry::Aabb> second;
    std::vector<double> paddings;
    std::vector<BroadPhaseLocation> locations;
    for (std::size_t candidate = 0; candidate < proposals.size(); ++candidate) {
        if (!finite(proposals[candidate].coordinates)) {
            continue;
        }
        const std::array<Point3, 1> point = {{proposals[candidate].coordinates}};
        const auto point_bounds = hotpot::geometry::aabb_bounds(
            ArrayView<Point3>(point.data(), point.size())
        );
        for (std::size_t atom = 0; atom < atom_count; ++atom) {
            if (atom == metal || workspace.donor_mask[atom] != 0) {
                continue;
            }
            const double radius_sum = workspace.radii[metal]
                + workspace.radii[atom];
            first.push_back(point_bounds);
            second.push_back(workspace.atom_bounds[atom]);
            paddings.push_back(std::max(
                options.absolute_center_clearance_angstrom,
                options.center_covalent_radius_scale * radius_sum
            ) + options.broad_phase_skin_angstrom);
            locations.push_back({candidate, 0, atom});
        }
    }
    const auto center_atoms = hotpot::geometry::aabb_separation_mask(
        ArrayView<hotpot::geometry::Aabb>(first),
        ArrayView<hotpot::geometry::Aabb>(second),
        ArrayView<double>(paddings)
    );
    for (std::size_t index = 0; index < locations.size(); ++index) {
        result[locations[index].candidate].center_atoms[
            locations[index].obstacle
        ] = center_atoms[index];
    }

    first.clear();
    second.clear();
    paddings.clear();
    locations.clear();
    for (std::size_t candidate = 0; candidate < proposals.size(); ++candidate) {
        if (!finite(proposals[candidate].coordinates)) {
            continue;
        }
        const std::array<Point3, 1> point = {{proposals[candidate].coordinates}};
        const auto point_bounds = hotpot::geometry::aabb_bounds(
            ArrayView<Point3>(point.data(), point.size())
        );
        for (std::size_t bond = 0; bond < bond_count; ++bond) {
            const auto& endpoints = workspace.active_ligand_bonds[bond];
            const auto first_atom = static_cast<std::size_t>(endpoints[0]);
            const auto second_atom = static_cast<std::size_t>(endpoints[1]);
            const double scale = 0.5 * (
                workspace.radii[first_atom]
                + workspace.radii[second_atom]
            );
            first.push_back(point_bounds);
            second.push_back(workspace.active_ligand_bond_bounds[bond]);
            paddings.push_back(
                options.minimum_path_bond_clearance * scale
                    + options.broad_phase_skin_angstrom
            );
            locations.push_back({candidate, 0, bond});
        }
    }
    const auto center_bonds = hotpot::geometry::aabb_separation_mask(
        ArrayView<hotpot::geometry::Aabb>(first),
        ArrayView<hotpot::geometry::Aabb>(second),
        ArrayView<double>(paddings)
    );
    for (std::size_t index = 0; index < locations.size(); ++index) {
        result[locations[index].candidate].center_bonds[
            locations[index].obstacle
        ] = center_bonds[index];
    }

    first.clear();
    second.clear();
    paddings.clear();
    locations.clear();
    for (std::size_t candidate = 0; candidate < proposals.size(); ++candidate) {
        if (!finite(proposals[candidate].coordinates)) {
            continue;
        }
        for (std::size_t donor = 0; donor < donor_count; ++donor) {
            const auto donor_atom = static_cast<std::size_t>(
                workspace.target.donors[donor].atom_index
            );
            const std::array<Point3, 2> endpoints = {{
                proposals[candidate].coordinates,
                (*workspace.coordinates)[donor_atom],
            }};
            const auto path_bounds = hotpot::geometry::aabb_bounds(
                ArrayView<Point3>(endpoints.data(), endpoints.size())
            );
            for (std::size_t atom = 0; atom < atom_count; ++atom) {
                if (atom == metal || atom == donor_atom) {
                    continue;
                }
                const double scale = workspace.radii[metal]
                    + workspace.radii[atom];
                first.push_back(path_bounds);
                second.push_back(workspace.atom_bounds[atom]);
                paddings.push_back(
                    options.minimum_path_atom_clearance * scale
                        + options.broad_phase_skin_angstrom
                );
                locations.push_back({candidate, donor, atom});
            }
        }
    }
    assign_separation_mask(
        first,
        second,
        paddings,
        locations,
        result,
        &DonorBroadPhaseMasks::atoms
    );

    first.clear();
    second.clear();
    paddings.clear();
    locations.clear();
    for (std::size_t candidate = 0; candidate < proposals.size(); ++candidate) {
        if (!finite(proposals[candidate].coordinates)) {
            continue;
        }
        for (std::size_t donor = 0; donor < donor_count; ++donor) {
            const auto donor_atom = static_cast<std::size_t>(
                workspace.target.donors[donor].atom_index
            );
            const std::array<Point3, 2> endpoints = {{
                proposals[candidate].coordinates,
                (*workspace.coordinates)[donor_atom],
            }};
            const auto path_bounds = hotpot::geometry::aabb_bounds(
                ArrayView<Point3>(endpoints.data(), endpoints.size())
            );
            for (std::size_t bond = 0; bond < bond_count; ++bond) {
                const auto& bond_atoms = workspace.active_ligand_bonds[bond];
                const auto first_atom = static_cast<std::size_t>(bond_atoms[0]);
                const auto second_atom = static_cast<std::size_t>(bond_atoms[1]);
                const double scale = 0.5 * (
                    workspace.radii[first_atom]
                    + workspace.radii[second_atom]
                );
                first.push_back(path_bounds);
                second.push_back(workspace.active_ligand_bond_bounds[bond]);
                paddings.push_back(
                    options.minimum_path_bond_clearance * scale
                        + options.broad_phase_skin_angstrom
                );
                locations.push_back({candidate, donor, bond});
            }
        }
    }
    assign_separation_mask(
        first,
        second,
        paddings,
        locations,
        result,
        &DonorBroadPhaseMasks::bonds
    );

    first.clear();
    second.clear();
    paddings.clear();
    locations.clear();
    for (std::size_t candidate = 0; candidate < proposals.size(); ++candidate) {
        if (!finite(proposals[candidate].coordinates)) {
            continue;
        }
        for (std::size_t donor = 0; donor < donor_count; ++donor) {
            const auto donor_atom = static_cast<std::size_t>(
                workspace.target.donors[donor].atom_index
            );
            const std::array<Point3, 2> endpoints = {{
                proposals[candidate].coordinates,
                (*workspace.coordinates)[donor_atom],
            }};
            const auto path_bounds = hotpot::geometry::aabb_bounds(
                ArrayView<Point3>(endpoints.data(), endpoints.size())
            );
            for (std::size_t cycle = 0; cycle < cycle_count; ++cycle) {
                first.push_back(path_bounds);
                second.push_back(
                    workspace.cycles->prepared_cycles.cycle(cycle).bounds()
                );
                paddings.push_back(options.broad_phase_skin_angstrom);
                locations.push_back({candidate, donor, cycle});
            }
        }
    }
    assign_separation_mask(
        first,
        second,
        paddings,
        locations,
        result,
        &DonorBroadPhaseMasks::cycles
    );
    return result;
}


double normalized_center_bond_clearance(
    const Point3& candidate,
    const BondIndex& bond,
    const PlacementEvaluationWorkspace& workspace,
    const MetalPlacementOptions& options,
    bool& obstruction
) {
    const auto first = static_cast<std::size_t>(bond[0]);
    const auto second = static_cast<std::size_t>(bond[1]);
    const auto measurement = hotpot::geometry::point_segment_measurement(
        candidate,
        Segment3{
            (*workspace.coordinates)[first],
            (*workspace.coordinates)[second],
        },
        options.geometry_tolerances
    );
    const double scale = 0.5 * (
        workspace.radii[first] + workspace.radii[second]
    );
    const double normalized = measurement.distance / scale;
    obstruction = measurement.parameter > options.geometry_tolerances.parameter
        && measurement.parameter
            < 1.0 - options.geometry_tolerances.parameter
        && normalized < options.minimum_path_bond_clearance;
    return normalized;
}


DonorPathEvidence evaluate_donor_path(
    const PlacementEvaluationWorkspace& workspace,
    const PlacementDonorTarget& donor,
    const Coordinate& candidate,
    const DonorBroadPhaseMasks& broad_phase,
    const MetalPlacementOptions& options
) {
    const auto donor_position = static_cast<std::size_t>(donor.atom_index);
    const Segment3 path{candidate, (*workspace.coordinates)[donor_position]};
    const double distance = point_distance(candidate, path.end);
    const double ratio = distance / donor.target_distance_angstrom;
    double minimum_atom_clearance = std::numeric_limits<double>::infinity();
    double minimum_bond_clearance = std::numeric_limits<double>::infinity();
    bool atom_obstruction = false;
    bool bond_obstruction = false;
    std::size_t atom_pair_count = 0;
    std::size_t atom_aabb_rejected_pair_count = 0;
    std::size_t bond_pair_count = 0;
    std::size_t bond_aabb_rejected_pair_count = 0;
    std::size_t cycle_pair_count = 0;
    std::size_t cycle_aabb_rejected_pair_count = 0;
    for (std::size_t atom = 0; atom < workspace.input->atom_count(); ++atom) {
        if (atom == donor_position
            || atom == static_cast<std::size_t>(workspace.target.metal_index)) {
            continue;
        }
        ++atom_pair_count;
        const double atom_scale = workspace.radii[static_cast<std::size_t>(
            workspace.target.metal_index
        )] + workspace.radii[atom];
        if (broad_phase.atoms[atom] != 0) {
            ++atom_aabb_rejected_pair_count;
        }
        const auto measurement = hotpot::geometry::point_segment_measurement(
            (*workspace.coordinates)[atom],
            path,
            options.geometry_tolerances
        );
        const double normalized = measurement.distance / atom_scale;
        minimum_atom_clearance = std::min(
            minimum_atom_clearance,
            normalized
        );
        if (broad_phase.atoms[atom] == 0
            && measurement.parameter > options.geometry_tolerances.parameter
            && measurement.parameter
                < 1.0 - options.geometry_tolerances.parameter
            && normalized < options.minimum_path_atom_clearance) {
            atom_obstruction = true;
        }
    }
    for (
        std::size_t bond_position = 0;
        bond_position < workspace.active_ligand_bonds.size();
        ++bond_position
    ) {
        const BondIndex& bond = workspace.active_ligand_bonds[bond_position];
        ++bond_pair_count;
        const auto first = static_cast<std::size_t>(bond[0]);
        const auto second = static_cast<std::size_t>(bond[1]);
        const double bond_scale = 0.5 * (
            workspace.radii[first] + workspace.radii[second]
        );
        if (broad_phase.bonds[bond_position] != 0) {
            ++bond_aabb_rejected_pair_count;
        }
        const Segment3 ligand_bond{
            (*workspace.coordinates)[first],
            (*workspace.coordinates)[second],
        };
        const auto measurement = hotpot::geometry::segment_segment_measurement(
            path,
            ligand_bond,
            options.geometry_tolerances
        );
        const bool donor_incident = first == donor_position
            || second == donor_position;
        const double normalized = measurement.distance / bond_scale;
        if (!donor_incident) {
            minimum_bond_clearance = std::min(
                minimum_bond_clearance,
                normalized
            );
        }
        if (broad_phase.bonds[bond_position] != 0) {
            continue;
        }
        const Point3 path_midpoint = hotpot::geometry::detail::add_scaled(
            path.start,
            hotpot::geometry::detail::subtract(path.end, path.start),
            0.5
        );
        const Point3 bond_midpoint = hotpot::geometry::detail::add_scaled(
            ligand_bond.start,
            hotpot::geometry::detail::subtract(
                ligand_bond.end,
                ligand_bond.start
            ),
            0.5
        );
        const auto path_midpoint_measurement =
            hotpot::geometry::point_segment_measurement(
                path_midpoint,
                ligand_bond,
                options.geometry_tolerances
            );
        const auto bond_midpoint_measurement =
            hotpot::geometry::point_segment_measurement(
                bond_midpoint,
                path,
                options.geometry_tolerances
            );
        const bool path_interior = measurement.first_parameter
                > options.geometry_tolerances.parameter
            && measurement.first_parameter
                < 1.0 - options.geometry_tolerances.parameter;
        const bool bond_interior = measurement.second_parameter
                > options.geometry_tolerances.parameter
            && measurement.second_parameter
                < 1.0 - options.geometry_tolerances.parameter;
        const bool midpoint_overlap = (
            path_midpoint_measurement.parameter
                > options.geometry_tolerances.parameter
            && path_midpoint_measurement.parameter
                < 1.0 - options.geometry_tolerances.parameter
            && path_midpoint_measurement.distance
                / bond_scale < options.minimum_path_bond_clearance
        ) || (
            bond_midpoint_measurement.parameter
                > options.geometry_tolerances.parameter
            && bond_midpoint_measurement.parameter
                < 1.0 - options.geometry_tolerances.parameter
            && bond_midpoint_measurement.distance
                / bond_scale < options.minimum_path_bond_clearance
        );
        // Donor-incident bonds are still checked. The common donor endpoint
        // alone is allowed; a proper interior crossing or collinear interior
        // overlap is obstructive.
        const bool obstructed = (path_interior && bond_interior
                && measurement.distance / bond_scale
                    < options.minimum_path_bond_clearance)
            || midpoint_overlap;
        if (obstructed) {
            bond_obstruction = true;
            if (donor_incident) {
                minimum_bond_clearance = std::min(
                    minimum_bond_clearance,
                    std::min({
                        measurement.distance,
                        path_midpoint_measurement.distance,
                        bond_midpoint_measurement.distance,
                    }) / bond_scale
                );
            }
        }
    }

    std::size_t piercing_count = 0;
    std::size_t undetermined_count = 0;
    for (
        std::size_t cycle_position = 0;
        cycle_position < workspace.cycles->prepared_cycles.cycle_count();
        ++cycle_position
    ) {
        const auto& cycle = workspace.cycles->prepared_cycles.cycle(
            cycle_position
        );
        ++cycle_pair_count;
        if (broad_phase.cycles[cycle_position] != 0) {
            ++cycle_aabb_rejected_pair_count;
            continue;
        }
        const auto screening = hotpot::geometry::screen_segment_cycle(
            path,
            cycle,
            false
        );
        if (screening.state == PiercingState::PIERCES) {
            ++piercing_count;
        } else if (screening.state == PiercingState::UNDETERMINED) {
            ++undetermined_count;
        }
    }

    const bool distance_reachable = ratio
            >= options.coordination_distance_ratio_minimum
        && ratio <= options.coordination_distance_ratio_maximum;
    DonorPathStatus status = DonorPathStatus::SAFE;
    if (!distance_reachable) {
        status = DonorPathStatus::OUT_OF_RANGE;
    } else if (atom_obstruction) {
        status = DonorPathStatus::ATOM_OBSTRUCTION;
    } else if (bond_obstruction) {
        status = DonorPathStatus::BOND_OBSTRUCTION;
    } else if (piercing_count != 0) {
        status = DonorPathStatus::RING_PIERCING;
    } else if (undetermined_count != 0) {
        status = DonorPathStatus::UNDETERMINED;
    }
    std::vector<DonorApproachEvidence> approach_angles;
    approach_angles.reserve(donor.neighbour_indices.size());
    for (const auto neighbour : donor.neighbour_indices) {
        approach_angles.push_back({
            neighbour,
            hotpot::geometry::detail::angle_at_vertex(
                candidate,
                path.end,
                (*workspace.coordinates)[static_cast<std::size_t>(neighbour)],
                options.geometry_tolerances
            ),
        });
    }
    return {
        donor.atom_index,
        donor.group_index,
        status,
        distance_reachable,
        atom_obstruction,
        bond_obstruction,
        donor.target_distance_angstrom,
        distance,
        ratio,
        minimum_atom_clearance,
        minimum_bond_clearance,
        piercing_count,
        undetermined_count,
        atom_pair_count,
        atom_aabb_rejected_pair_count,
        bond_pair_count,
        bond_aabb_rejected_pair_count,
        cycle_pair_count,
        cycle_aabb_rejected_pair_count,
        std::move(approach_angles),
    };
}


int status_rank(PlacementStatus status) noexcept {
    switch (status) {
        case PlacementStatus::FULLY_FEASIBLE:
            return 2;
        case PlacementStatus::PARTIAL:
            return 1;
        case PlacementStatus::INFEASIBLE:
            return 0;
    }
    return 0;
}


}  // namespace


namespace detail {


PlacementEvaluationWorkspace prepare_placement_evaluation(
    const ComplexSessionInput& input,
    const std::vector<Coordinate>& coordinates,
    const std::vector<std::uint8_t>& active_ligand_bond_mask,
    const std::vector<std::int32_t>& component_ids,
    std::int32_t metal_index,
    const MetalPlacementOptions& options
) {
    options.validate();
    if (coordinates.size() != input.atom_count()) {
        throw std::invalid_argument("coordinates must match atom count");
    }
    if (active_ligand_bond_mask.size() != input.ligand_bond_count()) {
        throw std::invalid_argument(
            "active ligand-bond mask must match ligand bonds"
        );
    }
    PlacementEvaluationWorkspace workspace{
        &input,
        &coordinates,
        &active_ligand_bond_mask,
        select_metal_placement_target(
            input,
            component_ids,
            metal_index,
            options.coordination_distance_scale
        ),
        {},
        std::vector<std::uint8_t>(input.atom_count(), 0),
        {},
        {},
        {},
        {},
    };
    workspace.radii.reserve(input.atom_count());
    for (const auto atomic_number : input.atomic_numbers) {
        workspace.radii.push_back(covalent_radius(atomic_number).angstrom);
    }
    for (const auto& donor : workspace.target.donors) {
        workspace.donor_mask[static_cast<std::size_t>(donor.atom_index)] = 1;
    }
    workspace.atom_bounds.reserve(input.atom_count());
    for (const auto& coordinate : coordinates) {
        const std::array<Point3, 1> point = {{coordinate}};
        workspace.atom_bounds.push_back(hotpot::geometry::aabb_bounds(
            ArrayView<Point3>(point.data(), point.size())
        ));
    }
    for (std::size_t index = 0; index < input.ligand_bond_count(); ++index) {
        if (active_ligand_bond_mask[index] != 0) {
            const BondIndex bond = input.ligand_bond_indices[index];
            if (!ligand_skeleton_bond(input, bond)) {
                continue;
            }
            workspace.active_ligand_bonds.push_back(bond);
            const std::array<Point3, 2> endpoints = {{
                coordinates[static_cast<std::size_t>(bond[0])],
                coordinates[static_cast<std::size_t>(bond[1])],
            }};
            workspace.active_ligand_bond_bounds.push_back(
                hotpot::geometry::aabb_bounds(ArrayView<Point3>(
                    endpoints.data(), endpoints.size()
                ))
            );
        }
    }
    workspace.cycles.emplace(prepare_ligand_cycles(
        input,
        coordinates,
        active_ligand_bond_mask,
        options
    ));
    return workspace;
}


namespace {


std::vector<DonorPairEvidence> evaluate_donor_pairs(
    const PlacementEvaluationWorkspace& workspace,
    const Coordinate& candidate,
    const MetalPlacementOptions& options
) {
    std::vector<DonorPairEvidence> evidence;
    const std::size_t donor_count = workspace.target.donors.size();
    evidence.reserve(donor_count * (donor_count - 1) / 2);
    for (std::size_t first = 0; first < donor_count; ++first) {
        const auto& first_target = workspace.target.donors[first];
        const auto& first_point = (*workspace.coordinates)[
            static_cast<std::size_t>(first_target.atom_index)
        ];
        for (std::size_t second = first + 1; second < donor_count; ++second) {
            const auto& second_target = workspace.target.donors[second];
            const auto& second_point = (*workspace.coordinates)[
                static_cast<std::size_t>(second_target.atom_index)
            ];
            const auto relation = hotpot::geometry::detail::measure_sphere_pair(
                first_point,
                first_target.target_distance_angstrom,
                second_point,
                second_target.target_distance_angstrom,
                options.geometry_tolerances
            );
            evidence.push_back({
                first_target.atom_index,
                second_target.atom_index,
                relation.center_distance,
                relation.radius_sum,
                relation.radius_difference,
                relation.intersects,
                hotpot::geometry::detail::angle_at_vertex(
                    first_point,
                    candidate,
                    second_point,
                    options.geometry_tolerances
                ),
            });
        }
    }
    return evidence;
}


PlacementCandidateEvidence evaluate_metal_position_with_masks(
    const PlacementEvaluationWorkspace& workspace,
    const Coordinate& candidate,
    PlacementProposalKind proposal_kind,
    std::size_t proposal_ordinal,
    const CandidateBroadPhaseMasks& broad_phase,
    const MetalPlacementOptions& options
) {
    const auto metal_position = static_cast<std::size_t>(
        workspace.target.metal_index
    );
    PlacementCandidateEvidence evidence{
        candidate,
        proposal_kind,
        PlacementStatus::INFEASIBLE,
        workspace.cycles->topology.excluded_large_cycle_count,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        std::numeric_limits<double>::infinity(),
        0.0,
        0.0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        point_distance(candidate, (*workspace.coordinates)[metal_position]),
        proposal_ordinal,
        {},
        {},
    };
    if (!finite(candidate)) {
        evidence.hard_obstruction_count = 1;
        return evidence;
    }

    bool center_collision = false;
    for (std::size_t atom = 0; atom < workspace.input->atom_count(); ++atom) {
        if (atom == metal_position || workspace.donor_mask[atom] != 0) {
            continue;
        }
        ++evidence.atom_pair_count;
        const double radius_sum = workspace.radii[metal_position]
            + workspace.radii[atom];
        const double collision_cutoff = std::max(
            options.absolute_center_clearance_angstrom,
            options.center_covalent_radius_scale * radius_sum
        );
        if (broad_phase.center_atoms[atom] != 0) {
            ++evidence.atom_aabb_rejected_pair_count;
        }
        const double distance = point_distance(
            candidate,
            (*workspace.coordinates)[atom]
        );
        evidence.minimum_normalized_clearance = std::min(
            evidence.minimum_normalized_clearance,
            distance / radius_sum
        );
        if (broad_phase.center_atoms[atom] == 0
            && distance < collision_cutoff) {
            center_collision = true;
            ++evidence.hard_obstruction_count;
        }
    }
    for (
        std::size_t position = 0;
        position < workspace.active_ligand_bonds.size();
        ++position
    ) {
        const BondIndex& bond = workspace.active_ligand_bonds[position];
        ++evidence.bond_pair_count;
        if (broad_phase.center_bonds[position] != 0) {
            ++evidence.bond_aabb_rejected_pair_count;
        }
        bool obstruction = false;
        const double normalized = normalized_center_bond_clearance(
            candidate,
            bond,
            workspace,
            options,
            obstruction
        );
        evidence.minimum_normalized_clearance = std::min(
            evidence.minimum_normalized_clearance,
            normalized
        );
        if (broad_phase.center_bonds[position] == 0 && obstruction) {
            center_collision = true;
            ++evidence.hard_obstruction_count;
        }
    }

    std::set<std::int32_t> covered_groups;
    double squared_deviation_sum = 0.0;
    evidence.donor_paths.reserve(workspace.target.donors.size());
    std::size_t provisional_donor_count = 0;
    for (
        std::size_t donor_position = 0;
        donor_position < workspace.target.donors.size();
        ++donor_position
    ) {
        const auto& donor = workspace.target.donors[donor_position];
        auto donor_evidence = evaluate_donor_path(
            workspace,
            donor,
            candidate,
            broad_phase.donor_paths[donor_position],
            options
        );
        const double deviation = std::abs(donor_evidence.distance_ratio - 1.0);
        evidence.worst_distance_deviation = std::max(
            evidence.worst_distance_deviation,
            deviation
        );
        squared_deviation_sum += deviation * deviation;
        evidence.minimum_normalized_clearance = std::min({
            evidence.minimum_normalized_clearance,
            donor_evidence.normalized_atom_clearance,
            donor_evidence.normalized_bond_clearance,
        });
        evidence.definite_piercing_count +=
            donor_evidence.definite_piercing_count;
        evidence.undetermined_relation_count +=
            donor_evidence.undetermined_relation_count;
        evidence.atom_pair_count += donor_evidence.atom_pair_count;
        evidence.atom_aabb_rejected_pair_count +=
            donor_evidence.atom_aabb_rejected_pair_count;
        evidence.bond_pair_count += donor_evidence.bond_pair_count;
        evidence.bond_aabb_rejected_pair_count +=
            donor_evidence.bond_aabb_rejected_pair_count;
        evidence.cycle_pair_count += donor_evidence.cycle_pair_count;
        evidence.cycle_aabb_rejected_pair_count +=
            donor_evidence.cycle_aabb_rejected_pair_count;
        if (!donor_evidence.distance_reachable) {
            ++evidence.out_of_range_donor_count;
        }
        if (donor_evidence.atom_obstructed) {
            ++evidence.atom_obstruction_count;
            ++evidence.hard_obstruction_count;
        }
        if (donor_evidence.bond_obstructed) {
            ++evidence.bond_obstruction_count;
            ++evidence.hard_obstruction_count;
        }
        if (donor_evidence.status == DonorPathStatus::SAFE) {
            ++evidence.safe_donor_count;
            covered_groups.insert(donor_evidence.group_index);
        } else if (donor_evidence.distance_reachable
            && !donor_evidence.atom_obstructed
            && !donor_evidence.bond_obstructed
            && donor_evidence.definite_piercing_count == 0
            && donor_evidence.undetermined_relation_count != 0) {
            ++provisional_donor_count;
        }
        evidence.donor_paths.push_back(std::move(donor_evidence));
    }
    evidence.covered_group_count = covered_groups.size();
    evidence.donor_pairs = evaluate_donor_pairs(
        workspace,
        candidate,
        options
    );
    evidence.rms_distance_deviation = std::sqrt(
        squared_deviation_sum
        / static_cast<double>(workspace.target.donors.size())
    );
    if (center_collision
        || (evidence.safe_donor_count == 0 && provisional_donor_count == 0)) {
        evidence.status = PlacementStatus::INFEASIBLE;
    } else if (evidence.safe_donor_count == workspace.target.donors.size()) {
        evidence.status = PlacementStatus::FULLY_FEASIBLE;
    } else {
        evidence.status = PlacementStatus::PARTIAL;
    }
    return evidence;
}


}  // namespace


PlacementCandidateEvidence evaluate_metal_position(
    const PlacementEvaluationWorkspace& workspace,
    const Coordinate& candidate,
    PlacementProposalKind proposal_kind,
    std::size_t proposal_ordinal,
    const MetalPlacementOptions& options
) {
    const std::vector<PlacementProposal> proposals = {{
        candidate,
        proposal_kind,
        proposal_ordinal,
    }};
    return evaluate_metal_positions(workspace, proposals, options).front();
}


std::vector<PlacementCandidateEvidence> evaluate_metal_positions(
    const PlacementEvaluationWorkspace& workspace,
    const std::vector<PlacementProposal>& proposals,
    const MetalPlacementOptions& options
) {
    const auto broad_phase = build_broad_phase_batch(
        workspace,
        proposals,
        options
    );
    std::vector<PlacementCandidateEvidence> evidence;
    evidence.reserve(proposals.size());
    for (std::size_t index = 0; index < proposals.size(); ++index) {
        evidence.push_back(evaluate_metal_position_with_masks(
            workspace,
            proposals[index].coordinates,
            proposals[index].kind,
            proposals[index].ordinal,
            broad_phase[index],
            options
        ));
    }
    return evidence;
}


bool prefer_placement_candidate(
    const PlacementCandidateEvidence& candidate,
    const PlacementCandidateEvidence& incumbent
) noexcept {
    const auto candidate_rank = std::make_tuple(
        status_rank(candidate.status),
        candidate.covered_group_count,
        candidate.safe_donor_count,
        -static_cast<std::int64_t>(candidate.definite_piercing_count),
        -static_cast<std::int64_t>(candidate.hard_obstruction_count)
    );
    const auto incumbent_rank = std::make_tuple(
        status_rank(incumbent.status),
        incumbent.covered_group_count,
        incumbent.safe_donor_count,
        -static_cast<std::int64_t>(incumbent.definite_piercing_count),
        -static_cast<std::int64_t>(incumbent.hard_obstruction_count)
    );
    if (candidate_rank != incumbent_rank) {
        return candidate_rank > incumbent_rank;
    }
    if (candidate.minimum_normalized_clearance
        != incumbent.minimum_normalized_clearance) {
        return candidate.minimum_normalized_clearance
            > incumbent.minimum_normalized_clearance;
    }
    if (candidate.worst_distance_deviation
        != incumbent.worst_distance_deviation) {
        return candidate.worst_distance_deviation
            < incumbent.worst_distance_deviation;
    }
    if (candidate.rms_distance_deviation
        != incumbent.rms_distance_deviation) {
        return candidate.rms_distance_deviation
            < incumbent.rms_distance_deviation;
    }
    if (candidate.undetermined_relation_count
        != incumbent.undetermined_relation_count) {
        return candidate.undetermined_relation_count
            < incumbent.undetermined_relation_count;
    }
    if (candidate.displacement_angstrom != incumbent.displacement_angstrom) {
        return candidate.displacement_angstrom < incumbent.displacement_angstrom;
    }
    return candidate.proposal_ordinal < incumbent.proposal_ordinal;
}


}  // namespace detail
}  // namespace hotpot::forcefields
