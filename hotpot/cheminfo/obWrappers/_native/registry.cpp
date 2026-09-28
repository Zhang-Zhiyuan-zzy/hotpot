#include "registry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>


namespace hotpot::obwrappers {


namespace {


bool descriptor_less(
    const RuleDescriptor& left,
    const RuleDescriptor& right
) {
    return std::tie(left.stage, left.priority, left.rule_id, left.version)
        < std::tie(right.stage, right.priority, right.rule_id, right.version);
}


bool definition_less(
    const RuleDefinition& left,
    const RuleDefinition& right
) {
    return descriptor_less(left.descriptor, right.descriptor);
}


void validate_atoms(const std::vector<AtomSnapshot>& atoms) {
    for (const auto& atom : atoms) {
        if (atom.atomic_number < 1) {
            throw std::invalid_argument("atomic numbers must be positive");
        }
        if (atom.hybridization < 0) {
            throw std::invalid_argument("hybridizations must be nonnegative");
        }
    }
}


void validate_bonds(
    const std::vector<BondSnapshot>& bonds,
    std::size_t atom_count
) {
    for (const auto& bond : bonds) {
        if (bond.begin >= atom_count || bond.end >= atom_count) {
            throw std::invalid_argument("bond atom index is out of range");
        }
        if (bond.begin == bond.end) {
            throw std::invalid_argument("self-bonds are not supported");
        }
        if (bond.order < 0) {
            throw std::invalid_argument("bond orders must be nonnegative");
        }
    }
}


void validate_coordinates(
    const std::vector<Coordinate>& coordinates,
    std::size_t atom_count,
    bool required
) {
    if (!required && coordinates.empty()) {
        return;
    }
    if (coordinates.size() != atom_count) {
        throw std::invalid_argument(
            "coordinate count must equal the atom count"
        );
    }
    for (const auto& coordinate : coordinates) {
        if (!std::all_of(
                coordinate.begin(),
                coordinate.end(),
                [](double value) { return std::isfinite(value); }
            )) {
            throw std::invalid_argument("coordinates must be finite");
        }
    }
}


}  // namespace


void append_application(RulePlan& plan, RuleApplication application) {
    if (plan.applications.size() >= MAX_RULE_APPLICATIONS) {
        throw RuleApplicationLimitExceeded(
            "native Open Babel wrapper rule application limit exceeded"
        );
    }
    plan.applications.push_back(std::move(application));
}


void RuleRegistry::add(RuleDefinition definition) {
    const auto duplicate = std::find_if(
        rules_.begin(),
        rules_.end(),
        [&definition](const RuleDefinition& registered) {
            return registered.descriptor.stage == definition.descriptor.stage
                && registered.descriptor.rule_id
                    == definition.descriptor.rule_id;
        }
    );
    if (duplicate != rules_.end()) {
        throw std::logic_error(
            "duplicate native Open Babel wrapper rule registration"
        );
    }
    rules_.push_back(std::move(definition));
    std::sort(rules_.begin(), rules_.end(), definition_less);
}


std::vector<RuleDescriptor> RuleRegistry::descriptors(
    std::optional<RuleStage> stage
) const {
    std::vector<RuleDescriptor> selected;
    for (const auto& rule : rules_) {
        if (!stage.has_value() || rule.descriptor.stage == *stage) {
            selected.push_back(rule.descriptor);
        }
    }
    return selected;
}


RulePlan RuleRegistry::execute(
    RuleStage stage,
    MoleculeSnapshot snapshot,
    const RuleParameters& parameters
) const {
    RulePlan plan{stage, {}};
    for (const auto& rule : rules_) {
        if (rule.descriptor.stage != stage) {
            continue;
        }
        if (rule.condition(snapshot, parameters)) {
            rule.action(snapshot, parameters, rule.descriptor, plan);
        }
    }
    return plan;
}


RuleRegistry& rule_registry() {
    static RuleRegistry registry;
    return registry;
}


RuleRegistrar::RuleRegistrar(RuleDefinition definition) {
    rule_registry().add(std::move(definition));
}


MoleculeSnapshot make_snapshot(
    std::vector<AtomSnapshot> atoms,
    std::vector<BondSnapshot> bonds,
    std::vector<Coordinate> coordinates,
    bool require_coordinates
) {
    validate_atoms(atoms);
    validate_bonds(bonds, atoms.size());
    validate_coordinates(coordinates, atoms.size(), require_coordinates);

    std::vector<std::vector<std::size_t>> adjacency(atoms.size());
    for (std::size_t bond_index = 0; bond_index < bonds.size(); ++bond_index) {
        const auto& bond = bonds[bond_index];
        adjacency[bond.begin].push_back(bond_index);
        adjacency[bond.end].push_back(bond_index);
    }
    for (auto& incident_bonds : adjacency) {
        std::sort(incident_bonds.begin(), incident_bonds.end());
    }
    return MoleculeSnapshot{
        std::move(atoms),
        std::move(bonds),
        std::move(coordinates),
        std::move(adjacency),
    };
}


RulePlan plan_build(
    std::vector<AtomSnapshot> atoms,
    std::vector<BondSnapshot> bonds
) {
    return rule_registry().execute(
        RuleStage::PRE_BUILD,
        make_snapshot(
            std::move(atoms),
            std::move(bonds),
            {},
            false
        ),
        RuleParameters{}
    );
}


RulePlan plan_optimization(
    std::vector<AtomSnapshot> atoms,
    std::vector<BondSnapshot> bonds,
    std::vector<Coordinate> coordinates,
    double singularity_threshold,
    double repair_angle_radians
) {
    if (!std::isfinite(singularity_threshold)
        || singularity_threshold < 0.0
        || singularity_threshold >= 1.0) {
        throw std::invalid_argument(
            "singularity_threshold must be finite and in [0, 1)"
        );
    }
    if (!std::isfinite(repair_angle_radians)
        || repair_angle_radians <= 0.0
        || repair_angle_radians >= 3.14159265358979323846) {
        throw std::invalid_argument(
            "repair_angle_radians must be finite and in (0, pi)"
        );
    }
    if (std::abs(std::sin(repair_angle_radians))
        <= singularity_threshold) {
        throw std::invalid_argument(
            "repair_angle_radians must move the normalized angle sine "
            "above singularity_threshold"
        );
    }
    return rule_registry().execute(
        RuleStage::PRE_FORCEFIELD_SETUP,
        make_snapshot(
            std::move(atoms),
            std::move(bonds),
            std::move(coordinates),
            true
        ),
        RuleParameters{singularity_threshold, repair_angle_radians}
    );
}


}  // namespace hotpot::obwrappers
