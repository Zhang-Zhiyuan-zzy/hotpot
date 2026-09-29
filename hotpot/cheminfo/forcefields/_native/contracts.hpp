#pragma once

#include "../../obWrappers/_native/molecule_data.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>


namespace hotpot::forcefields {


using BondIndex = std::array<std::int32_t, 2>;
using BondKind = hotpot::obwrappers::BondKind;
using Coordinate = hotpot::obwrappers::Coordinate;


struct ComplexSessionInput {
    std::int32_t schema_version;
    std::vector<std::int32_t> atomic_numbers;
    std::vector<std::int32_t> formal_charges;
    std::vector<double> partial_charges;
    std::vector<Coordinate> coordinates;
    std::vector<std::uint8_t> atom_aromatic;
    std::vector<BondIndex> ligand_bond_indices;
    std::vector<double> ligand_bond_orders;
    std::vector<BondKind> ligand_bond_kinds;
    std::vector<std::uint8_t> ligand_bond_aromatic;
    std::vector<std::int32_t> metal_indices;
    std::vector<BondIndex> intended_coordination_bonds;
    std::vector<double> intended_coordination_orders;
    std::vector<BondKind> intended_coordination_kinds;
    std::optional<std::array<double, 6>> unit_cell;

    std::size_t atom_count() const noexcept;
    std::size_t ligand_bond_count() const noexcept;
    std::size_t intended_coordination_bond_count() const noexcept;
    void validate() const;
};


struct PerturbationOffsetBatch {
    std::size_t atom_count = 0;
    std::vector<std::vector<Coordinate>> frames;

    std::size_t frame_count() const noexcept;
    void validate() const;
};


hotpot::obwrappers::MoleculeData ligand_molecule_data(
    const ComplexSessionInput& input
);


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
