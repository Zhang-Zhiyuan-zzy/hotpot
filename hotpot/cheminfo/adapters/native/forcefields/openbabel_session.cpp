#include "openbabel_session.hpp"

#include "../../../obWrappers/_native/openbabel_adapter.hpp"

#include <openbabel/bond.h>

#include <algorithm>
#include <memory>
#include <stdexcept>


namespace hotpot::forcefields::openbabel_session_adapter {
namespace {


int openbabel_ligand_bond_order(double order) {
    return static_cast<int>(order);
}


int openbabel_coordination_bond_order(double order) {
    return std::max(1, static_cast<int>(order));
}


}  // namespace


std::unique_ptr<OpenBabel::OBMol> make_session_obmol(const ComplexSessionInput& input) {
    const auto ligand = ligand_molecule_view(input);
    hotpot::obwrappers::validate_openbabel_compatibility(ligand);
    input.validate();
    return std::make_unique<OpenBabel::OBMol>(
        hotpot::obwrappers::make_obmol(ligand)
    );
}


void apply_ligand_bond_active_mask(
    OpenBabel::OBMol& obmol,
    const ComplexSessionInput& input,
    const std::vector<std::uint8_t>& old_active_mask,
    const std::vector<std::uint8_t>& active_mask
) {
    obmol.BeginModify();
    for (std::size_t index = 0; index < active_mask.size(); ++index) {
        if (active_mask[index] == old_active_mask[index]) {
            continue;
        }
        const auto endpoints = input.ligand_bond_indices[index];
        if (active_mask[index] != 0) {
            if (!obmol.AddBond(
                    static_cast<unsigned long>(endpoints[0] + 1),
                    static_cast<unsigned long>(endpoints[1] + 1),
                    openbabel_ligand_bond_order(
                        input.ligand_bond_orders[index]
                    )
                )) {
                throw std::runtime_error(
                    "Open Babel rejected a ligand bond"
                );
            }
        } else {
            auto* bond = obmol.GetBond(
                static_cast<int>(endpoints[0] + 1),
                static_cast<int>(endpoints[1] + 1)
            );
            if (bond == nullptr || !obmol.DeleteBond(bond)) {
                throw std::runtime_error(
                    "Open Babel could not deactivate a ligand bond"
                );
            }
        }
    }
    obmol.EndModify();
    for (std::size_t index = 0; index < active_mask.size(); ++index) {
        if (active_mask[index] == 0) {
            continue;
        }
        const auto endpoints = input.ligand_bond_indices[index];
        auto* bond = obmol.GetBond(
            static_cast<int>(endpoints[0] + 1),
            static_cast<int>(endpoints[1] + 1)
        );
        bond->SetAromatic(
            input.ligand_bond_kinds[index] == BondKind::AROMATIC
            || input.ligand_bond_aromatic[index] != 0
        );
    }
}


void apply_coordination_active_mask(
    OpenBabel::OBMol& obmol,
    const ComplexSessionInput& input,
    const std::vector<std::uint8_t>& old_active_mask,
    const std::vector<std::uint8_t>& active_mask
) {
    obmol.BeginModify();
    for (std::size_t index = 0; index < active_mask.size(); ++index) {
        if (active_mask[index] == old_active_mask[index]) {
            continue;
        }
        const auto endpoints = input.intended_coordination_bonds[index];
        if (active_mask[index] != 0) {
            if (!obmol.AddBond(
                    static_cast<unsigned long>(endpoints[0] + 1),
                    static_cast<unsigned long>(endpoints[1] + 1),
                    openbabel_coordination_bond_order(
                        input.intended_coordination_orders[index]
                    )
                )) {
                throw std::runtime_error(
                    "Open Babel rejected an intended coordination bond"
                );
            }
        } else {
            auto* bond = obmol.GetBond(
                static_cast<int>(endpoints[0] + 1),
                static_cast<int>(endpoints[1] + 1)
            );
            if (bond == nullptr || !obmol.DeleteBond(bond)) {
                throw std::runtime_error(
                    "Open Babel could not deactivate a coordination bond"
                );
            }
        }
    }
    obmol.EndModify();
}


std::recursive_mutex& openbabel_runtime_mutex() {
    return hotpot::obwrappers::openbabel_runtime_mutex();
}


std::vector<Coordinate> extract_coordinates(const OpenBabel::OBMol& molecule) {
    return hotpot::obwrappers::extract_coordinates(molecule);
}


void set_coordinates(OpenBabel::OBMol& molecule, const std::vector<Coordinate>& coordinates) {
    hotpot::obwrappers::set_coordinates(molecule, coordinates);
}


std::size_t active_bond_count(const OpenBabel::OBMol& molecule) {
    return molecule.NumBonds();
}


hotpot::obwrappers::SingleOptimizationResult single_optimize_in_place(
    OpenBabel::OBMol& molecule,
    const std::string& forcefield,
    std::size_t steps,
    double singularity_threshold,
    double repair_angle_radians
) {
    return hotpot::obwrappers::single_optimize_in_place(
        molecule, forcefield, steps, singularity_threshold, repair_angle_radians
    );
}


hotpot::obwrappers::OptimizationResult optimize_in_place(
    OpenBabel::OBMol& molecule,
    const hotpot::obwrappers::OptimizationOptions& options,
    const std::vector<std::vector<Coordinate>>& perturbation_offsets,
    double singularity_threshold,
    double repair_angle_radians
) {
    return hotpot::obwrappers::optimize_in_place(
        molecule, options, perturbation_offsets, singularity_threshold, repair_angle_radians
    );
}


}  // namespace hotpot::forcefields::openbabel_session_adapter
