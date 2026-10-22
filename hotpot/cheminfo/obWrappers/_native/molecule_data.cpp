#include "molecule_data.hpp"

#include <utility>


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


MoleculeData::MoleculeData(
    std::int32_t schema_version_value,
    std::vector<std::int32_t> atomic_numbers_value,
    std::vector<std::int32_t> formal_charges_value,
    std::vector<double> partial_charges_value,
    std::vector<Coordinate> coordinates_value,
    std::vector<std::uint8_t> atom_aromatic_value,
    std::vector<std::array<std::int32_t, 2>> bond_indices_value,
    std::vector<double> bond_orders_value,
    std::vector<BondKind> bond_kinds_value,
    std::vector<std::uint8_t> bond_aromatic_value,
    std::optional<std::array<double, 6>> unit_cell_value
) : hotpot::native::OwnedMoleculeData{} {
    const auto atoms = atomic_numbers_value.size();
    const auto bonds = bond_indices_value.size();
    const auto field_mask = REQUIRED_FIELDS
        | (unit_cell_value.has_value()
            ? hotpot::native::molecule_field_mask(
                hotpot::native::MoleculeField::UNIT_CELL
            )
            : 0);

    hotpot::native::OwnedMoleculeData::schema_version = schema_version_value;
    hotpot::native::OwnedMoleculeData::atom_count = atoms;
    hotpot::native::OwnedMoleculeData::bond_count = bonds;
    hotpot::native::OwnedMoleculeData::fields = field_mask;
    atomic_numbers = std::move(atomic_numbers_value);
    formal_charges = std::move(formal_charges_value);
    partial_charges = std::move(partial_charges_value);
    coordinates = std::move(coordinates_value);
    atom_aromatic = std::move(atom_aromatic_value);
    bond_indices = std::move(bond_indices_value);
    bond_orders = std::move(bond_orders_value);
    bond_kinds = std::move(bond_kinds_value);
    bond_aromatic = std::move(bond_aromatic_value);
    unit_cell = std::move(unit_cell_value);
}


std::size_t MoleculeData::atom_count() const noexcept {
    return atomic_numbers.size();
}


std::size_t MoleculeData::bond_count() const noexcept {
    return bond_indices.size();
}


hotpot::native::MoleculeBufferView MoleculeData::view() const noexcept {
    auto molecule = hotpot::native::OwnedMoleculeData::view();
    molecule.atom_count = atom_count();
    molecule.bond_count = bond_count();
    molecule.fields = REQUIRED_FIELDS
        | (unit_cell.has_value()
            ? hotpot::native::molecule_field_mask(
                hotpot::native::MoleculeField::UNIT_CELL
            )
            : 0);
    return molecule;
}


void MoleculeData::validate() const {
    hotpot::native::validate_molecule_view(view());
}


}  // namespace hotpot::obwrappers
