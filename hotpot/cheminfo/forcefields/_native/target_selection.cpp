#include "target_selection.hpp"

#include "radii.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>


namespace hotpot::forcefields {
namespace detail {
namespace {


std::vector<std::vector<std::int32_t>> ligand_adjacency(
    const ComplexSessionInput& input
) {
    std::vector<std::vector<std::int32_t>> adjacency(input.atom_count());
    for (const BondIndex& bond : input.ligand_bond_indices) {
        adjacency[static_cast<std::size_t>(bond[0])].push_back(bond[1]);
        adjacency[static_cast<std::size_t>(bond[1])].push_back(bond[0]);
    }
    for (auto& neighbours : adjacency) {
        std::sort(neighbours.begin(), neighbours.end());
    }
    return adjacency;
}


}  // namespace


std::vector<MetalPlacementTarget> select_metal_placement_targets(
    const ComplexSessionInput& input,
    const std::vector<std::int32_t>& component_ids,
    double coordination_distance_scale
) {
    input.validate();
    if (component_ids.size() != input.atom_count()) {
        throw std::invalid_argument("component_ids must match atom count");
    }
    if (!(coordination_distance_scale > 0.0)) {
        throw std::invalid_argument(
            "coordination distance scale must be greater than zero"
        );
    }

    const auto adjacency = ligand_adjacency(input);
    std::map<std::int32_t, std::vector<std::int32_t>> donors_by_metal;
    for (const auto metal : input.metal_indices) {
        donors_by_metal.emplace(metal, std::vector<std::int32_t>{});
    }
    for (const BondIndex& bond : input.intended_coordination_bonds) {
        donors_by_metal.at(bond[0]).push_back(bond[1]);
    }

    std::vector<MetalPlacementTarget> targets;
    targets.reserve(input.metal_indices.size());
    for (const auto metal : input.metal_indices) {
        auto donors = donors_by_metal.at(metal);
        std::sort(donors.begin(), donors.end());
        donors.erase(std::unique(donors.begin(), donors.end()), donors.end());
        MetalPlacementTarget target{metal, {}};
        target.donors.reserve(donors.size());
        const double metal_radius = covalent_radius(
            input.atomic_numbers[static_cast<std::size_t>(metal)]
        ).angstrom;
        for (const auto donor : donors) {
            const auto donor_position = static_cast<std::size_t>(donor);
            target.donors.push_back(PlacementDonorTarget{
                donor,
                component_ids[donor_position],
                coordination_distance_scale * (
                    metal_radius
                    + covalent_radius(input.atomic_numbers[donor_position]).angstrom
                ),
                adjacency[donor_position],
            });
        }
        targets.push_back(std::move(target));
    }
    return targets;
}


MetalPlacementTarget select_metal_placement_target(
    const ComplexSessionInput& input,
    const std::vector<std::int32_t>& component_ids,
    std::int32_t metal_index,
    double coordination_distance_scale
) {
    const auto targets = select_metal_placement_targets(
        input,
        component_ids,
        coordination_distance_scale
    );
    const auto found = std::find_if(
        targets.begin(),
        targets.end(),
        [metal_index](const MetalPlacementTarget& target) {
            return target.metal_index == metal_index;
        }
    );
    if (found == targets.end()) {
        throw std::invalid_argument("metal_index is not a declared metal");
    }
    if (found->donors.empty()) {
        throw std::invalid_argument(
            "metal_index has no intended coordination donor"
        );
    }
    return *found;
}


}  // namespace detail
}  // namespace hotpot::forcefields
