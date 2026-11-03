#pragma once

#include "../../../structure/_native/forcefield_input.hpp"
#include "../../../obWrappers/_native/native_engine.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>


namespace hotpot::forcefields::openbabel_session_adapter {


std::unique_ptr<OpenBabel::OBMol> make_session_obmol(const ComplexSessionInput& input);

void apply_ligand_bond_active_mask(
    OpenBabel::OBMol& obmol,
    const ComplexSessionInput& input,
    const std::vector<std::uint8_t>& old_active_mask,
    const std::vector<std::uint8_t>& active_mask
);

void apply_coordination_active_mask(
    OpenBabel::OBMol& obmol,
    const ComplexSessionInput& input,
    const std::vector<std::uint8_t>& old_active_mask,
    const std::vector<std::uint8_t>& active_mask
);

std::recursive_mutex& openbabel_runtime_mutex();
std::vector<Coordinate> extract_coordinates(const OpenBabel::OBMol& molecule);
void set_coordinates(OpenBabel::OBMol& molecule, const std::vector<Coordinate>& coordinates);
std::size_t active_bond_count(const OpenBabel::OBMol& molecule);

hotpot::obwrappers::SingleOptimizationResult single_optimize_in_place(
    OpenBabel::OBMol& molecule,
    const std::string& forcefield,
    std::size_t steps,
    double singularity_threshold,
    double repair_angle_radians
);

hotpot::obwrappers::OptimizationResult optimize_in_place(
    OpenBabel::OBMol& molecule,
    const hotpot::obwrappers::OptimizationOptions& options,
    const std::vector<std::vector<Coordinate>>& perturbation_offsets,
    double singularity_threshold,
    double repair_angle_radians
);


}  // namespace hotpot::forcefields::openbabel_session_adapter
