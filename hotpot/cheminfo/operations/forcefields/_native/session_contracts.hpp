#pragma once

#include "../../../structure/_native/forcefield_input.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>


namespace hotpot::forcefields {


struct PerturbationOffsetBatch {
    std::size_t atom_count = 0;
    std::vector<std::vector<Coordinate>> frames;

    std::size_t frame_count() const noexcept;
    void validate() const;
};


struct StructureSnapshot {
    std::vector<Coordinate> coordinates;
    std::vector<std::uint8_t> active_ligand_bond_mask;
    std::vector<std::uint8_t> active_coordination_mask;
    std::vector<std::int32_t> component_ids;
    std::size_t ligand_bond_count;
    std::size_t active_bond_count;
    std::uint64_t coordinate_revision;
    std::uint64_t topology_revision;
};


}  // namespace hotpot::forcefields
