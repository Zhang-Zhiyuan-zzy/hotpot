#include "molecule_data.hpp"


namespace hotpot::obwrappers {


namespace {


constexpr std::uint64_t REQUIRED_FIELDS =
    hotpot::native::molecule_field_mask(
        hotpot::native::MoleculeField::ATOMIC_NUMBERS
    )
    | hotpot::native::molecule_field_mask(
        hotpot::native::MoleculeField::FORMAL_CHARGES
    )
    | hotpot::native::molecule_field_mask(
        hotpot::native::MoleculeField::PARTIAL_CHARGES
    )
    | hotpot::native::molecule_field_mask(
        hotpot::native::MoleculeField::COORDINATES
    )
    | hotpot::native::molecule_field_mask(
        hotpot::native::MoleculeField::ATOM_AROMATIC
    )
    | hotpot::native::molecule_field_mask(
        hotpot::native::MoleculeField::BOND_INDICES
    )
    | hotpot::native::molecule_field_mask(
        hotpot::native::MoleculeField::BOND_ORDERS
    )
    | hotpot::native::molecule_field_mask(
        hotpot::native::MoleculeField::BOND_KINDS
    )
    | hotpot::native::molecule_field_mask(
        hotpot::native::MoleculeField::BOND_AROMATIC
    );


}  // namespace


std::size_t MoleculeData::atom_count() const noexcept {
    return atomic_numbers.size();
}


std::size_t MoleculeData::bond_count() const noexcept {
    return bond_indices.size();
}


hotpot::native::MoleculeBufferView MoleculeData::view() const noexcept {
    return hotpot::native::MoleculeBufferView{
        schema_version,
        atom_count(),
        bond_count(),
        REQUIRED_FIELDS
            | (unit_cell.has_value()
                ? hotpot::native::molecule_field_mask(
                    hotpot::native::MoleculeField::UNIT_CELL
                )
                : 0),
        hotpot::native::ArrayView<std::int32_t>(atomic_numbers),
        hotpot::native::ArrayView<std::int32_t>(formal_charges),
        hotpot::native::ArrayView<double>(partial_charges),
        hotpot::native::ArrayView<hotpot::native::Coordinate>(coordinates),
        hotpot::native::ArrayView<std::uint8_t>(atom_aromatic),
        hotpot::native::ArrayView<hotpot::native::BondIndex>(bond_indices),
        hotpot::native::ArrayView<double>(bond_orders),
        hotpot::native::ArrayView<hotpot::native::BondKind>(bond_kinds),
        hotpot::native::ArrayView<std::uint8_t>(bond_aromatic),
        unit_cell.has_value()
            ? hotpot::native::ArrayView<double>(
                unit_cell->data(), unit_cell->size()
            )
            : hotpot::native::ArrayView<double>{},
    };
}


void MoleculeData::validate() const {
    hotpot::native::validate_molecule_view(view());
}


}  // namespace hotpot::obwrappers
