#include "registry.hpp"

#include <cstddef>
#include <utility>
#include <vector>


namespace hotpot::obwrappers {


namespace {


constexpr int PHOSPHORUS_ATOMIC_NUMBER = 15;
constexpr int OXYGEN_ATOMIC_NUMBER = 8;
constexpr int SULFUR_ATOMIC_NUMBER = 16;
constexpr int ORIGINAL_HYBRIDIZATION = 5;
constexpr int BUILDER_HYBRIDIZATION = 3;


std::size_t other_atom(const BondSnapshot& bond, std::size_t atom_index) {
    return bond.begin == atom_index ? bond.end : bond.begin;
}


bool is_target_phosphorus(
    const MoleculeSnapshot& snapshot,
    std::size_t atom_index
) {
    const auto& atom = snapshot.atoms[atom_index];
    const auto& incident_bonds = snapshot.adjacency[atom_index];
    if (atom.atomic_number != PHOSPHORUS_ATOMIC_NUMBER
        || atom.formal_charge != 0
        || atom.hybridization != ORIGINAL_HYBRIDIZATION
        || incident_bonds.size() != 4) {
        return false;
    }

    int explicit_valence = 0;
    int heteroatom_double_bonds = 0;
    int nonaromatic_single_bonds = 0;
    for (const auto bond_index : incident_bonds) {
        const auto& bond = snapshot.bonds[bond_index];
        if (bond.aromatic) {
            return false;
        }
        explicit_valence += bond.order;
        if (bond.order == 2) {
            const int neighbor_atomic_number = snapshot.atoms[
                other_atom(bond, atom_index)
            ].atomic_number;
            if (neighbor_atomic_number != OXYGEN_ATOMIC_NUMBER
                && neighbor_atomic_number != SULFUR_ATOMIC_NUMBER) {
                return false;
            }
            ++heteroatom_double_bonds;
        } else if (bond.order == 1) {
            ++nonaromatic_single_bonds;
        } else {
            return false;
        }
    }
    return explicit_valence == 5
        && heteroatom_double_bonds == 1
        && nonaromatic_single_bonds == 3;
}


std::vector<std::size_t> target_phosphorus_atoms(
    const MoleculeSnapshot& snapshot
) {
    std::vector<std::size_t> targets;
    for (std::size_t atom_index = 0;
         atom_index < snapshot.atoms.size();
         ++atom_index) {
        if (is_target_phosphorus(snapshot, atom_index)) {
            targets.push_back(atom_index);
        }
    }
    return targets;
}


bool phosphorus_builder_condition(
    const MoleculeSnapshot& snapshot,
    const RuleParameters&
) {
    return !target_phosphorus_atoms(snapshot).empty();
}


void phosphorus_builder_action(
    MoleculeSnapshot& snapshot,
    const RuleParameters&,
    const RuleDescriptor& descriptor,
    RulePlan& plan
) {
    for (const auto atom_index : target_phosphorus_atoms(snapshot)) {
        const int before = snapshot.atoms[atom_index].hybridization;
        RuleApplication application{
            descriptor.rule_id,
            descriptor.version,
            descriptor.stage,
            descriptor.priority,
            {atom_index},
            std::nullopt,
            {{atom_index, before, BUILDER_HYBRIDIZATION}},
            {},
        };
        append_application(plan, std::move(application));
        snapshot.atoms[atom_index].hybridization = BUILDER_HYBRIDIZATION;
    }
}


const RuleRegistrar phosphorus_builder_registrar(
    RuleDefinition{
        RuleDescriptor{
            "tetracoordinate_pentavalent_phosphorus_build",
            "1.0.0",
            RuleStage::PRE_BUILD,
            100,
        },
        phosphorus_builder_condition,
        phosphorus_builder_action,
    }
);


}  // namespace


}  // namespace hotpot::obwrappers
