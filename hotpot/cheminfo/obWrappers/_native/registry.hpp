#pragma once

#include "rules.hpp"

#include <optional>
#include <vector>


namespace hotpot::obwrappers {


class RuleRegistry {
public:
    void add(RuleDefinition definition);

    std::vector<RuleDescriptor> descriptors(
        std::optional<RuleStage> stage = std::nullopt
    ) const;

    RulePlan execute(
        RuleStage stage,
        MoleculeSnapshot snapshot,
        const RuleParameters& parameters
    ) const;

private:
    std::vector<RuleDefinition> rules_;
};


RuleRegistry& rule_registry();


class RuleRegistrar {
public:
    explicit RuleRegistrar(RuleDefinition definition);
};


MoleculeSnapshot make_snapshot(
    std::vector<AtomSnapshot> atoms,
    std::vector<BondSnapshot> bonds,
    std::vector<Coordinate> coordinates,
    bool require_coordinates
);


RulePlan plan_build(
    std::vector<AtomSnapshot> atoms,
    std::vector<BondSnapshot> bonds
);


RulePlan plan_optimization(
    std::vector<AtomSnapshot> atoms,
    std::vector<BondSnapshot> bonds,
    std::vector<Coordinate> coordinates,
    double singularity_threshold,
    double repair_angle_radians
);


}  // namespace hotpot::obwrappers
