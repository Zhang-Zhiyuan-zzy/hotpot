#include "registry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>


namespace hotpot::obwrappers {


namespace {


constexpr double MINIMUM_VECTOR_LENGTH = 1.0e-12;


struct DegenerateAngle {
    std::size_t first;
    std::size_t center;
    std::size_t second;
    std::size_t branch_root;
    std::vector<std::size_t> branch_atoms;
    double sine;
};


std::size_t other_atom(const BondSnapshot& bond, std::size_t atom_index) {
    return bond.begin == atom_index ? bond.end : bond.begin;
}


Coordinate subtract(const Coordinate& left, const Coordinate& right) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}


Coordinate add(const Coordinate& left, const Coordinate& right) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}


Coordinate multiply(const Coordinate& vector, double scalar) {
    return {
        vector[0] * scalar,
        vector[1] * scalar,
        vector[2] * scalar,
    };
}


double dot(const Coordinate& left, const Coordinate& right) {
    return left[0] * right[0]
        + left[1] * right[1]
        + left[2] * right[2];
}


Coordinate cross(const Coordinate& left, const Coordinate& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}


double norm(const Coordinate& vector) {
    return std::sqrt(dot(vector, vector));
}


Coordinate normalized(const Coordinate& vector) {
    return multiply(vector, 1.0 / norm(vector));
}


std::vector<std::size_t> neighbors(
    const MoleculeSnapshot& snapshot,
    std::size_t atom_index
) {
    std::vector<std::size_t> result;
    result.reserve(snapshot.adjacency[atom_index].size());
    for (const auto bond_index : snapshot.adjacency[atom_index]) {
        result.push_back(other_atom(snapshot.bonds[bond_index], atom_index));
    }
    std::sort(result.begin(), result.end());
    return result;
}


bool edge_matches(
    const BondSnapshot& bond,
    std::size_t first,
    std::size_t second
) {
    return (bond.begin == first && bond.end == second)
        || (bond.begin == second && bond.end == first);
}


std::optional<std::vector<std::size_t>> separable_branch(
    const MoleculeSnapshot& snapshot,
    std::size_t center,
    std::size_t root
) {
    std::vector<bool> visited(snapshot.atoms.size(), false);
    std::queue<std::size_t> pending;
    visited[root] = true;
    pending.push(root);

    while (!pending.empty()) {
        const auto current = pending.front();
        pending.pop();
        for (const auto bond_index : snapshot.adjacency[current]) {
            const auto& bond = snapshot.bonds[bond_index];
            if (edge_matches(bond, center, root)) {
                continue;
            }
            const auto next = other_atom(bond, current);
            if (next == center) {
                return std::nullopt;
            }
            if (!visited[next]) {
                visited[next] = true;
                pending.push(next);
            }
        }
    }

    std::vector<std::size_t> branch;
    for (std::size_t atom_index = 0;
         atom_index < visited.size();
         ++atom_index) {
        if (visited[atom_index]) {
            branch.push_back(atom_index);
        }
    }
    return branch;
}


bool can_participate_in_proper_torsion(
    const MoleculeSnapshot& snapshot,
    std::size_t first,
    std::size_t center,
    std::size_t second
) {
    const auto has_external_neighbor = [&snapshot, center](
        std::size_t atom_index
    ) {
        return std::any_of(
            snapshot.adjacency[atom_index].begin(),
            snapshot.adjacency[atom_index].end(),
            [&snapshot, atom_index, center](std::size_t bond_index) {
                return other_atom(snapshot.bonds[bond_index], atom_index)
                    != center;
            }
        );
    };
    return has_external_neighbor(first) || has_external_neighbor(second);
}


std::optional<DegenerateAngle> candidate_for_center(
    const MoleculeSnapshot& snapshot,
    std::size_t center,
    double singularity_threshold
) {
    const auto& center_atom = snapshot.atoms[center];
    const auto center_degree = snapshot.adjacency[center].size();
    if (center_atom.is_metal
        || center_atom.hybridization < 2
        || (center_degree != 3 && center_degree != 4)) {
        return std::nullopt;
    }

    const auto adjacent_atoms = neighbors(snapshot, center);
    std::optional<DegenerateAngle> best;
    for (std::size_t first_position = 0;
         first_position < adjacent_atoms.size();
         ++first_position) {
        for (std::size_t second_position = first_position + 1;
             second_position < adjacent_atoms.size();
             ++second_position) {
            const auto first = adjacent_atoms[first_position];
            const auto second = adjacent_atoms[second_position];
            if (!can_participate_in_proper_torsion(
                    snapshot,
                    first,
                    center,
                    second
                )) {
                continue;
            }

            const auto first_vector = subtract(
                snapshot.coordinates[first],
                snapshot.coordinates[center]
            );
            const auto second_vector = subtract(
                snapshot.coordinates[second],
                snapshot.coordinates[center]
            );
            const double first_length = norm(first_vector);
            const double second_length = norm(second_vector);
            if (first_length <= MINIMUM_VECTOR_LENGTH
                || second_length <= MINIMUM_VECTOR_LENGTH) {
                continue;
            }
            const double sine = norm(cross(first_vector, second_vector))
                / (first_length * second_length);
            if (sine > singularity_threshold) {
                continue;
            }

            const auto first_branch = separable_branch(snapshot, center, first);
            const auto second_branch = separable_branch(snapshot, center, second);
            if (!first_branch.has_value() && !second_branch.has_value()) {
                continue;
            }

            std::size_t branch_root;
            std::vector<std::size_t> branch_atoms;
            if (!second_branch.has_value()
                || (first_branch.has_value()
                    && std::make_tuple(first_branch->size(), first)
                        <= std::make_tuple(second_branch->size(), second))) {
                branch_root = first;
                branch_atoms = *first_branch;
            } else {
                branch_root = second;
                branch_atoms = *second_branch;
            }

            DegenerateAngle candidate{
                first,
                center,
                second,
                branch_root,
                std::move(branch_atoms),
                sine,
            };
            if (!best.has_value()
                || std::tie(
                       candidate.sine,
                       candidate.first,
                       candidate.second,
                       candidate.branch_root
                   )
                    < std::tie(
                       best->sine,
                       best->first,
                       best->second,
                       best->branch_root
                   )) {
                best = std::move(candidate);
            }
        }
    }
    return best;
}


Coordinate rotation_axis(const Coordinate& direction) {
    const auto unit_direction = normalized(direction);
    const std::array<Coordinate, 3> basis = {
        Coordinate{1.0, 0.0, 0.0},
        Coordinate{0.0, 1.0, 0.0},
        Coordinate{0.0, 0.0, 1.0},
    };
    const auto reference = *std::min_element(
        basis.begin(),
        basis.end(),
        [&unit_direction](const Coordinate& left, const Coordinate& right) {
            return std::abs(dot(unit_direction, left))
                < std::abs(dot(unit_direction, right));
        }
    );
    return normalized(cross(unit_direction, reference));
}


Coordinate rotate_about_axis(
    const Coordinate& vector,
    const Coordinate& axis,
    double angle
) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    return add(
        add(
            multiply(vector, cosine),
            multiply(cross(axis, vector), sine)
        ),
        multiply(axis, dot(axis, vector) * (1.0 - cosine))
    );
}


bool degenerate_torsion_condition(
    const MoleculeSnapshot& snapshot,
    const RuleParameters& parameters
) {
    for (std::size_t center = 0; center < snapshot.atoms.size(); ++center) {
        if (candidate_for_center(
                snapshot,
                center,
                parameters.singularity_threshold
            ).has_value()) {
            return true;
        }
    }
    return false;
}


void degenerate_torsion_action(
    MoleculeSnapshot& snapshot,
    const RuleParameters& parameters,
    const RuleDescriptor& descriptor,
    RulePlan& plan
) {
    for (std::size_t center = 0; center < snapshot.atoms.size(); ++center) {
        while (true) {
            auto candidate = candidate_for_center(
                snapshot,
                center,
                parameters.singularity_threshold
            );
            if (!candidate.has_value()) {
                break;
            }

            const auto center_coordinate = snapshot.coordinates[center];
            const auto direction = subtract(
                snapshot.coordinates[candidate->branch_root],
                center_coordinate
            );
            const auto axis = rotation_axis(direction);
            RuleApplication application{
                descriptor.rule_id,
                descriptor.version,
                descriptor.stage,
                descriptor.priority,
                {candidate->first, candidate->center, candidate->second},
                candidate->sine,
                {},
                {},
            };
            for (const auto atom_index : candidate->branch_atoms) {
                const auto before = snapshot.coordinates[atom_index];
                const auto after = add(
                    center_coordinate,
                    rotate_about_axis(
                        subtract(before, center_coordinate),
                        axis,
                        parameters.repair_angle_radians
                    )
                );
                application.coordinate_changes.push_back(
                    CoordinateChange{atom_index, before, after}
                );
            }
            append_application(plan, std::move(application));
            for (const auto& change :
                 plan.applications.back().coordinate_changes) {
                snapshot.coordinates[change.atom_index] = change.after;
            }
        }
    }
}


const RuleRegistrar degenerate_torsion_registrar(
    RuleDefinition{
        RuleDescriptor{
            "degenerate_nonlinear_torsion",
            "1.0.0",
            RuleStage::PRE_FORCEFIELD_SETUP,
            100,
        },
        degenerate_torsion_condition,
        degenerate_torsion_action,
    }
);


}  // namespace


}  // namespace hotpot::obwrappers
