#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>


namespace hotpot::obwrappers {


enum class RuleStage {
    PRE_BUILD,
    PRE_FORCEFIELD_SETUP,
};


struct AtomSnapshot {
    int atomic_number;
    int formal_charge;
    int hybridization;
    bool is_metal;
};


struct BondSnapshot {
    std::size_t begin;
    std::size_t end;
    int order;
    bool aromatic;
};


using Coordinate = std::array<double, 3>;


struct RuleDescriptor {
    std::string rule_id;
    std::string version;
    RuleStage stage;
    int priority;
};


struct HybridizationChange {
    std::size_t atom_index;
    int before;
    int after;
};


struct CoordinateChange {
    std::size_t atom_index;
    Coordinate before;
    Coordinate after;
};


struct RuleApplication {
    std::string rule_id;
    std::string version;
    RuleStage stage;
    int priority;
    std::vector<std::size_t> atom_indices;
    std::optional<double> metric_before;
    std::vector<HybridizationChange> hybridization_changes;
    std::vector<CoordinateChange> coordinate_changes;
};


struct RulePlan {
    RuleStage stage;
    std::vector<RuleApplication> applications;
};


struct RuleParameters {
    double singularity_threshold = 0.0;
    double repair_angle_radians = 0.0;
};


struct MoleculeSnapshot {
    std::vector<AtomSnapshot> atoms;
    std::vector<BondSnapshot> bonds;
    std::vector<Coordinate> coordinates;
    std::vector<std::vector<std::size_t>> adjacency;
};


using RuleCondition = bool (*)(
    const MoleculeSnapshot&,
    const RuleParameters&
);
using RuleAction = void (*)(
    MoleculeSnapshot&,
    const RuleParameters&,
    const RuleDescriptor&,
    RulePlan&
);


struct RuleDefinition {
    RuleDescriptor descriptor;
    RuleCondition condition;
    RuleAction action;
};


class RuleApplicationLimitExceeded : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};


constexpr std::size_t MAX_RULE_APPLICATIONS = 256;


void append_application(RulePlan& plan, RuleApplication application);


}  // namespace hotpot::obwrappers
