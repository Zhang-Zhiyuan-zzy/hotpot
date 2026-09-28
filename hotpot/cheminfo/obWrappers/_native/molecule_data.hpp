#pragma once

#include "rules.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>


namespace hotpot::obwrappers {


enum class BondKind : std::uint8_t {
    SINGLE = 1,
    DOUBLE = 2,
    TRIPLE = 3,
    AROMATIC = 4,
    ZERO = 5,
    DATIVE = 6,
    UNKNOWN = 7,
};


struct MoleculeData {
    std::int32_t schema_version;
    std::vector<std::int32_t> atomic_numbers;
    std::vector<std::int32_t> formal_charges;
    std::vector<double> partial_charges;
    std::vector<Coordinate> coordinates;
    std::vector<std::uint8_t> atom_aromatic;
    std::vector<std::array<std::int32_t, 2>> bond_indices;
    std::vector<double> bond_orders;
    std::vector<BondKind> bond_kinds;
    std::vector<std::uint8_t> bond_aromatic;
    std::optional<std::array<double, 6>> unit_cell;

    std::size_t atom_count() const noexcept;
    std::size_t bond_count() const noexcept;
    void validate() const;
};


}  // namespace hotpot::obwrappers
