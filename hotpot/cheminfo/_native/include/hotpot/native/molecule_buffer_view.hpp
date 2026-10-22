#pragma once

#include "array_view.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>


namespace hotpot::native {


using BondIndex = std::array<std::int32_t, 2>;
using Coordinate = std::array<double, 3>;
using UnitCell = std::array<double, 6>;


enum class BondKind : std::uint8_t {
    SINGLE = 1,
    DOUBLE = 2,
    TRIPLE = 3,
    AROMATIC = 4,
    ZERO = 5,
    DATIVE = 6,
    UNKNOWN = 7,
};


enum class MoleculeField : std::uint64_t {
    ATOMIC_NUMBERS = std::uint64_t{1} << 0,
    FORMAL_CHARGES = std::uint64_t{1} << 1,
    PARTIAL_CHARGES = std::uint64_t{1} << 2,
    COORDINATES = std::uint64_t{1} << 3,
    ATOM_AROMATIC = std::uint64_t{1} << 4,
    BOND_INDICES = std::uint64_t{1} << 5,
    BOND_ORDERS = std::uint64_t{1} << 6,
    BOND_KINDS = std::uint64_t{1} << 7,
    BOND_AROMATIC = std::uint64_t{1} << 8,
    UNIT_CELL = std::uint64_t{1} << 9,
};


constexpr std::uint64_t molecule_field_mask(MoleculeField field) noexcept {
    return static_cast<std::uint64_t>(field);
}


struct MoleculeBufferView {
    std::int32_t schema_version = 1;
    std::size_t atom_count = 0;
    std::size_t bond_count = 0;
    std::uint64_t fields = 0;
    ArrayView<std::int32_t> atomic_numbers;
    ArrayView<std::int32_t> formal_charges;
    ArrayView<double> partial_charges;
    ArrayView<Coordinate> coordinates;
    ArrayView<std::uint8_t> atom_aromatic;
    ArrayView<BondIndex> bond_indices;
    ArrayView<double> bond_orders;
    ArrayView<BondKind> bond_kinds;
    ArrayView<std::uint8_t> bond_aromatic;
    ArrayView<double> unit_cell;

    constexpr bool has(MoleculeField field) const noexcept {
        return (fields & molecule_field_mask(field)) != 0;
    }
};


namespace detail {


inline void validate_field_size(
    const MoleculeBufferView& molecule,
    MoleculeField field,
    std::size_t actual,
    std::size_t expected
) {
    if (molecule.has(field)) {
        if (actual != expected) {
            throw std::invalid_argument(
                "present molecule buffer has an invalid length"
            );
        }
    } else if (actual != 0) {
        throw std::invalid_argument(
            "absent molecule buffer must not carry values"
        );
    }
}


inline bool finite_coordinate(const Coordinate& coordinate) noexcept {
    return std::all_of(
        coordinate.begin(),
        coordinate.end(),
        [](double value) { return std::isfinite(value); }
    );
}


}  // namespace detail


inline void validate_molecule_view(const MoleculeBufferView& molecule) {
    if (molecule.schema_version != 1) {
        throw std::invalid_argument("unsupported molecule buffer schema");
    }

    detail::validate_field_size(
        molecule,
        MoleculeField::ATOMIC_NUMBERS,
        molecule.atomic_numbers.size(),
        molecule.atom_count
    );
    detail::validate_field_size(
        molecule,
        MoleculeField::FORMAL_CHARGES,
        molecule.formal_charges.size(),
        molecule.atom_count
    );
    detail::validate_field_size(
        molecule,
        MoleculeField::PARTIAL_CHARGES,
        molecule.partial_charges.size(),
        molecule.atom_count
    );
    detail::validate_field_size(
        molecule,
        MoleculeField::COORDINATES,
        molecule.coordinates.size(),
        molecule.atom_count
    );
    detail::validate_field_size(
        molecule,
        MoleculeField::ATOM_AROMATIC,
        molecule.atom_aromatic.size(),
        molecule.atom_count
    );
    detail::validate_field_size(
        molecule,
        MoleculeField::BOND_INDICES,
        molecule.bond_indices.size(),
        molecule.bond_count
    );
    detail::validate_field_size(
        molecule,
        MoleculeField::BOND_ORDERS,
        molecule.bond_orders.size(),
        molecule.bond_count
    );
    detail::validate_field_size(
        molecule,
        MoleculeField::BOND_KINDS,
        molecule.bond_kinds.size(),
        molecule.bond_count
    );
    detail::validate_field_size(
        molecule,
        MoleculeField::BOND_AROMATIC,
        molecule.bond_aromatic.size(),
        molecule.bond_count
    );
    detail::validate_field_size(
        molecule,
        MoleculeField::UNIT_CELL,
        molecule.unit_cell.size(),
        6
    );

    if (molecule.has(MoleculeField::ATOMIC_NUMBERS)
        && std::any_of(
            molecule.atomic_numbers.begin(),
            molecule.atomic_numbers.end(),
            [](std::int32_t number) { return number < 1; }
        )) {
        throw std::invalid_argument("atomic numbers must be positive");
    }
    if (molecule.has(MoleculeField::PARTIAL_CHARGES)
        && std::any_of(
            molecule.partial_charges.begin(),
            molecule.partial_charges.end(),
            [](double charge) { return !std::isfinite(charge); }
        )) {
        throw std::invalid_argument("partial charges must be finite");
    }
    if (molecule.has(MoleculeField::COORDINATES)
        && std::any_of(
            molecule.coordinates.begin(),
            molecule.coordinates.end(),
            [](const Coordinate& coordinate) {
                return !detail::finite_coordinate(coordinate);
            }
        )) {
        throw std::invalid_argument("coordinates must be finite");
    }
    if (molecule.has(MoleculeField::BOND_INDICES)) {
        for (const auto& endpoints : molecule.bond_indices) {
            if (endpoints[0] < 0 || endpoints[1] < 0
                || static_cast<std::size_t>(endpoints[0])
                    >= molecule.atom_count
                || static_cast<std::size_t>(endpoints[1])
                    >= molecule.atom_count) {
                throw std::invalid_argument("bond atom index is out of range");
            }
            if (endpoints[0] == endpoints[1]) {
                throw std::invalid_argument("self-bonds are not supported");
            }
        }
    }
    if (molecule.has(MoleculeField::BOND_ORDERS)
        && std::any_of(
            molecule.bond_orders.begin(),
            molecule.bond_orders.end(),
            [](double order) {
                return !std::isfinite(order) || order < 0.0;
            }
        )) {
        throw std::invalid_argument(
            "bond orders must be finite and nonnegative"
        );
    }
    if (molecule.has(MoleculeField::BOND_KINDS)
        && std::any_of(
            molecule.bond_kinds.begin(),
            molecule.bond_kinds.end(),
            [](BondKind kind) {
                const auto value = static_cast<std::uint8_t>(kind);
                return value < static_cast<std::uint8_t>(BondKind::SINGLE)
                    || value > static_cast<std::uint8_t>(BondKind::UNKNOWN);
            }
        )) {
        throw std::invalid_argument("bond kind code is out of range");
    }
    if (molecule.has(MoleculeField::UNIT_CELL)
        && std::any_of(
            molecule.unit_cell.begin(),
            molecule.unit_cell.end(),
            [](double value) { return !std::isfinite(value); }
        )) {
        throw std::invalid_argument("unit-cell parameters must be finite");
    }
}


struct OwnedMoleculeData {
    std::int32_t schema_version = 1;
    std::size_t atom_count = 0;
    std::size_t bond_count = 0;
    std::uint64_t fields = 0;
    std::vector<std::int32_t> atomic_numbers;
    std::vector<std::int32_t> formal_charges;
    std::vector<double> partial_charges;
    std::vector<Coordinate> coordinates;
    std::vector<std::uint8_t> atom_aromatic;
    std::vector<BondIndex> bond_indices;
    std::vector<double> bond_orders;
    std::vector<BondKind> bond_kinds;
    std::vector<std::uint8_t> bond_aromatic;
    std::optional<UnitCell> unit_cell;

    MoleculeBufferView view() const noexcept {
        return MoleculeBufferView{
            schema_version,
            atom_count,
            bond_count,
            fields,
            ArrayView<std::int32_t>(atomic_numbers),
            ArrayView<std::int32_t>(formal_charges),
            ArrayView<double>(partial_charges),
            ArrayView<Coordinate>(coordinates),
            ArrayView<std::uint8_t>(atom_aromatic),
            ArrayView<BondIndex>(bond_indices),
            ArrayView<double>(bond_orders),
            ArrayView<BondKind>(bond_kinds),
            ArrayView<std::uint8_t>(bond_aromatic),
            unit_cell.has_value()
                ? ArrayView<double>(unit_cell->data(), unit_cell->size())
                : ArrayView<double>{},
        };
    }

    void validate() const { validate_molecule_view(view()); }
};


struct ConformerBatchView {
    std::size_t conformer_count = 0;
    std::size_t atom_count = 0;
    ArrayView<Coordinate> coordinates;

    const Coordinate& at(
        std::size_t conformer_index,
        std::size_t atom_index
    ) const noexcept {
        return coordinates[conformer_index * atom_count + atom_index];
    }

    void validate() const {
        if (coordinates.size() != conformer_count * atom_count) {
            throw std::invalid_argument(
                "conformer coordinates must have shape K x N x 3"
            );
        }
        if (std::any_of(
                coordinates.begin(),
                coordinates.end(),
                [](const Coordinate& coordinate) {
                    return !detail::finite_coordinate(coordinate);
                }
            )) {
            throw std::invalid_argument(
                "conformer coordinates must be finite"
            );
        }
    }
};


}  // namespace hotpot::native
