#pragma once

#include "../../_native/include/hotpot/native/molecule_buffer_view.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>


namespace hotpot::forcefields {


using BondIndex = hotpot::native::BondIndex;
using BondKind = hotpot::native::BondKind;
using Coordinate = hotpot::native::Coordinate;


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


hotpot::native::MoleculeBufferView ligand_molecule_view(
    const ComplexSessionInput& input
) noexcept;


}  // namespace hotpot::forcefields
