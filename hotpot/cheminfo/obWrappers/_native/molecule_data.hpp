#pragma once

#include "../../_native/include/hotpot/native/molecule_buffer_view.hpp"
#include "rules.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>


namespace hotpot::obwrappers {


using BondKind = hotpot::native::BondKind;


struct MoleculeData : hotpot::native::OwnedMoleculeData {
    MoleculeData(
        std::int32_t schema_version,
        std::vector<std::int32_t> atomic_numbers,
        std::vector<std::int32_t> formal_charges,
        std::vector<double> partial_charges,
        std::vector<Coordinate> coordinates,
        std::vector<std::uint8_t> atom_aromatic,
        std::vector<std::array<std::int32_t, 2>> bond_indices,
        std::vector<double> bond_orders,
        std::vector<BondKind> bond_kinds,
        std::vector<std::uint8_t> bond_aromatic,
        std::optional<std::array<double, 6>> unit_cell
    );

    std::size_t atom_count() const noexcept;
    std::size_t bond_count() const noexcept;
    hotpot::native::MoleculeBufferView view() const noexcept;
    void validate() const;
};


}  // namespace hotpot::obwrappers
