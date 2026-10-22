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
using TransportBondIndex = std::array<std::uint64_t, 2>;
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
    STEREO_RECORDS = std::uint64_t{1} << 10,
    ATOM_TRANSPORT = std::uint64_t{1} << 11,
    BOND_TRANSPORT = std::uint64_t{1} << 12,
};


constexpr std::uint64_t molecule_field_mask(MoleculeField field) noexcept {
    return static_cast<std::uint64_t>(field);
}


enum class StereoAnchorKind : std::uint32_t {
    ATOM = 1,
    BOND = 2,
};


enum class StereoCarrierKind : std::uint32_t {
    ATOM = 1,
    IMPLICIT_H = 2,
    LONE_PAIR = 3,
    VACANCY = 4,
};


enum class StereoGroupKind : std::uint32_t {
    NONE = 0,
    ABS = 1,
    AND = 2,
    OR = 3,
};


enum class AtomTransportField : std::uint32_t {
    ISOTOPE = std::uint32_t{1} << 0,
    RADICAL_ELECTRONS = std::uint32_t{1} << 1,
};


enum class HydrogenTransportField : std::uint32_t {
    IMPLICIT = std::uint32_t{1} << 0,
    BRACKET_EXPLICIT = std::uint32_t{1} << 1,
    NO_IMPLICIT = std::uint32_t{1} << 2,
};


struct Utf8TableView {
    ArrayView<std::uint8_t> data;
    ArrayView<std::uint64_t> offsets;
    ArrayView<std::uint64_t> lengths;
    ArrayView<std::uint8_t> present;

    constexpr std::size_t size() const noexcept { return offsets.size(); }
    constexpr bool empty() const noexcept { return offsets.empty(); }
};


struct StereoRecordBuffersView {
    std::int32_t schema_version = 1;
    ArrayView<std::uint32_t> anchor_kinds;
    ArrayView<std::uint64_t> anchor_offsets;
    ArrayView<std::uint64_t> anchor_counts;
    ArrayView<std::uint64_t> anchor_indices;
    ArrayView<std::uint64_t> carrier_offsets;
    ArrayView<std::uint64_t> carrier_counts;
    ArrayView<std::uint32_t> carrier_kinds;
    ArrayView<std::uint32_t> carrier_has_atoms;
    ArrayView<std::uint64_t> carrier_atoms;
    ArrayView<std::uint32_t> group_kinds;
    ArrayView<std::int64_t> group_ids;
    ArrayView<std::uint32_t> group_present;
    Utf8TableView group_scopes;
    Utf8TableView descriptors;
    Utf8TableView source_namespaces;
    Utf8TableView source_codes;

    constexpr std::size_t size() const noexcept {
        return anchor_kinds.size();
    }
};


struct AtomTransportBuffersView {
    std::int32_t schema_version = 1;
    ArrayView<std::uint64_t> atoms;
    ArrayView<std::uint32_t> presence_masks;
    ArrayView<std::int32_t> isotopes;
    ArrayView<std::int32_t> radical_electrons;
    ArrayView<std::uint32_t> has_current_hydrogens;
    ArrayView<std::uint32_t> current_hydrogen_presence_masks;
    ArrayView<std::int32_t> current_implicit_hydrogens;
    ArrayView<std::int32_t> current_bracket_explicit_hydrogens;
    ArrayView<std::uint32_t> current_no_implicit;
    ArrayView<std::uint32_t> has_source_hydrogens;
    ArrayView<std::uint32_t> source_hydrogen_presence_masks;
    ArrayView<std::int32_t> source_implicit_hydrogens;
    ArrayView<std::int32_t> source_bracket_explicit_hydrogens;
    ArrayView<std::uint32_t> source_no_implicit;
    Utf8TableView source_namespaces;

    constexpr std::size_t size() const noexcept { return atoms.size(); }
};


struct BondTransportBuffersView {
    std::int32_t schema_version = 1;
    ArrayView<TransportBondIndex> atom_indices;
    Utf8TableView display_directions;

    constexpr std::size_t size() const noexcept {
        return atom_indices.size();
    }
};


struct OwnedUtf8TableData {
    std::vector<std::uint8_t> data;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> lengths;
    std::vector<std::uint8_t> present;

    Utf8TableView view() const noexcept {
        return Utf8TableView{
            ArrayView<std::uint8_t>(data),
            ArrayView<std::uint64_t>(offsets),
            ArrayView<std::uint64_t>(lengths),
            ArrayView<std::uint8_t>(present),
        };
    }
};


struct OwnedStereoRecordData {
    std::int32_t schema_version = 1;
    std::vector<std::uint32_t> anchor_kinds;
    std::vector<std::uint64_t> anchor_offsets;
    std::vector<std::uint64_t> anchor_counts;
    std::vector<std::uint64_t> anchor_indices;
    std::vector<std::uint64_t> carrier_offsets;
    std::vector<std::uint64_t> carrier_counts;
    std::vector<std::uint32_t> carrier_kinds;
    std::vector<std::uint32_t> carrier_has_atoms;
    std::vector<std::uint64_t> carrier_atoms;
    std::vector<std::uint32_t> group_kinds;
    std::vector<std::int64_t> group_ids;
    std::vector<std::uint32_t> group_present;
    OwnedUtf8TableData group_scopes;
    OwnedUtf8TableData descriptors;
    OwnedUtf8TableData source_namespaces;
    OwnedUtf8TableData source_codes;

    StereoRecordBuffersView view() const noexcept {
        return StereoRecordBuffersView{
            schema_version,
            ArrayView<std::uint32_t>(anchor_kinds),
            ArrayView<std::uint64_t>(anchor_offsets),
            ArrayView<std::uint64_t>(anchor_counts),
            ArrayView<std::uint64_t>(anchor_indices),
            ArrayView<std::uint64_t>(carrier_offsets),
            ArrayView<std::uint64_t>(carrier_counts),
            ArrayView<std::uint32_t>(carrier_kinds),
            ArrayView<std::uint32_t>(carrier_has_atoms),
            ArrayView<std::uint64_t>(carrier_atoms),
            ArrayView<std::uint32_t>(group_kinds),
            ArrayView<std::int64_t>(group_ids),
            ArrayView<std::uint32_t>(group_present),
            group_scopes.view(),
            descriptors.view(),
            source_namespaces.view(),
            source_codes.view(),
        };
    }
};


struct OwnedAtomTransportData {
    std::int32_t schema_version = 1;
    std::vector<std::uint64_t> atoms;
    std::vector<std::uint32_t> presence_masks;
    std::vector<std::int32_t> isotopes;
    std::vector<std::int32_t> radical_electrons;
    std::vector<std::uint32_t> has_current_hydrogens;
    std::vector<std::uint32_t> current_hydrogen_presence_masks;
    std::vector<std::int32_t> current_implicit_hydrogens;
    std::vector<std::int32_t> current_bracket_explicit_hydrogens;
    std::vector<std::uint32_t> current_no_implicit;
    std::vector<std::uint32_t> has_source_hydrogens;
    std::vector<std::uint32_t> source_hydrogen_presence_masks;
    std::vector<std::int32_t> source_implicit_hydrogens;
    std::vector<std::int32_t> source_bracket_explicit_hydrogens;
    std::vector<std::uint32_t> source_no_implicit;
    OwnedUtf8TableData source_namespaces;

    AtomTransportBuffersView view() const noexcept {
        return AtomTransportBuffersView{
            schema_version,
            ArrayView<std::uint64_t>(atoms),
            ArrayView<std::uint32_t>(presence_masks),
            ArrayView<std::int32_t>(isotopes),
            ArrayView<std::int32_t>(radical_electrons),
            ArrayView<std::uint32_t>(has_current_hydrogens),
            ArrayView<std::uint32_t>(current_hydrogen_presence_masks),
            ArrayView<std::int32_t>(current_implicit_hydrogens),
            ArrayView<std::int32_t>(current_bracket_explicit_hydrogens),
            ArrayView<std::uint32_t>(current_no_implicit),
            ArrayView<std::uint32_t>(has_source_hydrogens),
            ArrayView<std::uint32_t>(source_hydrogen_presence_masks),
            ArrayView<std::int32_t>(source_implicit_hydrogens),
            ArrayView<std::int32_t>(source_bracket_explicit_hydrogens),
            ArrayView<std::uint32_t>(source_no_implicit),
            source_namespaces.view(),
        };
    }
};


struct OwnedBondTransportData {
    std::int32_t schema_version = 1;
    std::vector<TransportBondIndex> atom_indices;
    OwnedUtf8TableData display_directions;

    BondTransportBuffersView view() const noexcept {
        return BondTransportBuffersView{
            schema_version,
            ArrayView<TransportBondIndex>(atom_indices),
            display_directions.view(),
        };
    }
};


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
    StereoRecordBuffersView stereo_records;
    AtomTransportBuffersView atom_transport;
    BondTransportBuffersView bond_transport;

    constexpr MoleculeBufferView() noexcept = default;

    constexpr MoleculeBufferView(
        std::int32_t schema_version_value,
        std::size_t atom_count_value,
        std::size_t bond_count_value,
        std::uint64_t fields_value,
        ArrayView<std::int32_t> atomic_numbers_value,
        ArrayView<std::int32_t> formal_charges_value,
        ArrayView<double> partial_charges_value,
        ArrayView<Coordinate> coordinates_value,
        ArrayView<std::uint8_t> atom_aromatic_value,
        ArrayView<BondIndex> bond_indices_value,
        ArrayView<double> bond_orders_value,
        ArrayView<BondKind> bond_kinds_value,
        ArrayView<std::uint8_t> bond_aromatic_value,
        ArrayView<double> unit_cell_value,
        StereoRecordBuffersView stereo_records_value = {},
        AtomTransportBuffersView atom_transport_value = {},
        BondTransportBuffersView bond_transport_value = {}
    ) noexcept
        : schema_version(schema_version_value),
          atom_count(atom_count_value),
          bond_count(bond_count_value),
          fields(fields_value),
          atomic_numbers(atomic_numbers_value),
          formal_charges(formal_charges_value),
          partial_charges(partial_charges_value),
          coordinates(coordinates_value),
          atom_aromatic(atom_aromatic_value),
          bond_indices(bond_indices_value),
          bond_orders(bond_orders_value),
          bond_kinds(bond_kinds_value),
          bond_aromatic(bond_aromatic_value),
          unit_cell(unit_cell_value),
          stereo_records(stereo_records_value),
          atom_transport(atom_transport_value),
          bond_transport(bond_transport_value) {}

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


inline void validate_column_size(std::size_t actual, std::size_t expected) {
    if (actual != expected) {
        throw std::invalid_argument(
            "native transport buffer has an invalid column length"
        );
    }
}


inline void validate_utf8_table(
    const Utf8TableView& table,
    std::size_t expected
) {
    validate_column_size(table.offsets.size(), expected);
    validate_column_size(table.lengths.size(), expected);
    validate_column_size(table.present.size(), expected);
    for (std::size_t index = 0; index < expected; ++index) {
        const auto offset = table.offsets[index];
        const auto length = table.lengths[index];
        const auto present = table.present[index];
        if (present > 1) {
            throw std::invalid_argument("UTF-8 presence code is out of range");
        }
        if (present == 0) {
            if (offset != 0 || length != 0) {
                throw std::invalid_argument(
                    "absent UTF-8 value must have a zero slice"
                );
            }
            continue;
        }
        if (offset > table.data.size()
            || length > table.data.size() - offset) {
            throw std::invalid_argument("UTF-8 slice is out of range");
        }
    }
}


inline std::size_t utf8_storage_size(const Utf8TableView& table) noexcept {
    return table.data.size() + table.offsets.size() + table.lengths.size()
        + table.present.size();
}


inline std::size_t stereo_storage_size(
    const StereoRecordBuffersView& stereo
) noexcept {
    return stereo.anchor_kinds.size() + stereo.anchor_offsets.size()
        + stereo.anchor_counts.size() + stereo.anchor_indices.size()
        + stereo.carrier_offsets.size() + stereo.carrier_counts.size()
        + stereo.carrier_kinds.size() + stereo.carrier_has_atoms.size()
        + stereo.carrier_atoms.size() + stereo.group_kinds.size()
        + stereo.group_ids.size() + stereo.group_present.size()
        + utf8_storage_size(stereo.group_scopes)
        + utf8_storage_size(stereo.descriptors)
        + utf8_storage_size(stereo.source_namespaces)
        + utf8_storage_size(stereo.source_codes);
}


inline std::size_t atom_transport_storage_size(
    const AtomTransportBuffersView& atoms
) noexcept {
    return atoms.atoms.size() + atoms.presence_masks.size()
        + atoms.isotopes.size() + atoms.radical_electrons.size()
        + atoms.has_current_hydrogens.size()
        + atoms.current_hydrogen_presence_masks.size()
        + atoms.current_implicit_hydrogens.size()
        + atoms.current_bracket_explicit_hydrogens.size()
        + atoms.current_no_implicit.size()
        + atoms.has_source_hydrogens.size()
        + atoms.source_hydrogen_presence_masks.size()
        + atoms.source_implicit_hydrogens.size()
        + atoms.source_bracket_explicit_hydrogens.size()
        + atoms.source_no_implicit.size()
        + utf8_storage_size(atoms.source_namespaces);
}


inline std::size_t bond_transport_storage_size(
    const BondTransportBuffersView& bonds
) noexcept {
    return bonds.atom_indices.size()
        + utf8_storage_size(bonds.display_directions);
}


inline bool range_fits(
    std::uint64_t offset,
    std::uint64_t count,
    std::size_t size
) noexcept {
    return offset <= size && count <= size - static_cast<std::size_t>(offset);
}


inline void validate_stereo_buffers(
    const StereoRecordBuffersView& stereo,
    std::size_t atom_count
) {
    if (stereo.schema_version != 1) {
        throw std::invalid_argument("unsupported stereo transport schema");
    }
    const auto records = stereo.size();
    validate_column_size(stereo.anchor_offsets.size(), records);
    validate_column_size(stereo.anchor_counts.size(), records);
    validate_column_size(stereo.carrier_offsets.size(), records);
    validate_column_size(stereo.carrier_counts.size(), records);
    validate_column_size(stereo.group_kinds.size(), records);
    validate_column_size(stereo.group_ids.size(), records);
    validate_column_size(stereo.group_present.size(), records);
    validate_column_size(
        stereo.carrier_has_atoms.size(), stereo.carrier_kinds.size()
    );
    validate_column_size(
        stereo.carrier_atoms.size(), stereo.carrier_kinds.size()
    );
    validate_utf8_table(stereo.group_scopes, records);
    validate_utf8_table(stereo.descriptors, records);
    validate_utf8_table(stereo.source_namespaces, records);
    validate_utf8_table(stereo.source_codes, records);

    for (const auto atom : stereo.anchor_indices) {
        if (atom >= atom_count) {
            throw std::invalid_argument("stereo anchor atom is out of range");
        }
    }
    for (std::size_t index = 0; index < records; ++index) {
        const auto anchor_kind = stereo.anchor_kinds[index];
        const auto anchor_count = stereo.anchor_counts[index];
        const auto group_kind = stereo.group_kinds[index];
        const auto group_present = stereo.group_present[index];
        if ((anchor_kind != static_cast<std::uint32_t>(StereoAnchorKind::ATOM)
             && anchor_kind
                != static_cast<std::uint32_t>(StereoAnchorKind::BOND))
            || (anchor_kind
                    == static_cast<std::uint32_t>(StereoAnchorKind::ATOM)
                && anchor_count != 1)
            || (anchor_kind
                    == static_cast<std::uint32_t>(StereoAnchorKind::BOND)
                && anchor_count != 2)
            || !range_fits(
                stereo.anchor_offsets[index],
                anchor_count,
                stereo.anchor_indices.size()
            )) {
            throw std::invalid_argument("invalid stereo anchor record");
        }
        if (!range_fits(
                stereo.carrier_offsets[index],
                stereo.carrier_counts[index],
                stereo.carrier_kinds.size()
            )) {
            throw std::invalid_argument("invalid stereo carrier range");
        }
        if (group_present > 1
            || group_kind > static_cast<std::uint32_t>(StereoGroupKind::OR)
            || (group_present == 0
                && (group_kind
                        != static_cast<std::uint32_t>(StereoGroupKind::NONE)
                    || stereo.group_ids[index] != 0
                    || stereo.group_scopes.present[index] != 0))
            || (group_present == 1
                && (group_kind
                        == static_cast<std::uint32_t>(StereoGroupKind::NONE)
                    || stereo.group_ids[index] < 0
                    || stereo.group_scopes.present[index] != 1
                    || stereo.group_scopes.lengths[index] == 0))) {
            throw std::invalid_argument("invalid stereo group identity");
        }
    }
    for (std::size_t index = 0; index < stereo.carrier_kinds.size(); ++index) {
        const auto kind = stereo.carrier_kinds[index];
        const auto has_atom = stereo.carrier_has_atoms[index];
        if (kind < static_cast<std::uint32_t>(StereoCarrierKind::ATOM)
            || kind > static_cast<std::uint32_t>(StereoCarrierKind::VACANCY)
            || has_atom > 1
            || ((kind == static_cast<std::uint32_t>(StereoCarrierKind::ATOM))
                != (has_atom != 0))
            || (has_atom != 0 && stereo.carrier_atoms[index] >= atom_count)) {
            throw std::invalid_argument("invalid stereo carrier record");
        }
    }
}


inline void validate_atom_transport_buffers(
    const AtomTransportBuffersView& atoms,
    std::size_t atom_count
) {
    if (atoms.schema_version != 1) {
        throw std::invalid_argument("unsupported atom transport schema");
    }
    const auto records = atoms.size();
    validate_column_size(atoms.presence_masks.size(), records);
    validate_column_size(atoms.isotopes.size(), records);
    validate_column_size(atoms.radical_electrons.size(), records);
    validate_column_size(atoms.has_current_hydrogens.size(), records);
    validate_column_size(
        atoms.current_hydrogen_presence_masks.size(), records
    );
    validate_column_size(atoms.current_implicit_hydrogens.size(), records);
    validate_column_size(
        atoms.current_bracket_explicit_hydrogens.size(), records
    );
    validate_column_size(atoms.current_no_implicit.size(), records);
    validate_column_size(atoms.has_source_hydrogens.size(), records);
    validate_column_size(
        atoms.source_hydrogen_presence_masks.size(), records
    );
    validate_column_size(atoms.source_implicit_hydrogens.size(), records);
    validate_column_size(
        atoms.source_bracket_explicit_hydrogens.size(), records
    );
    validate_column_size(atoms.source_no_implicit.size(), records);
    validate_utf8_table(atoms.source_namespaces, records);
    if (std::any_of(
            atoms.atoms.begin(),
            atoms.atoms.end(),
            [atom_count](std::uint64_t atom) { return atom >= atom_count; }
        )) {
        throw std::invalid_argument("atom transport index is out of range");
    }
}


inline void validate_bond_transport_buffers(
    const BondTransportBuffersView& bonds,
    std::size_t atom_count
) {
    if (bonds.schema_version != 1) {
        throw std::invalid_argument("unsupported bond transport schema");
    }
    validate_utf8_table(bonds.display_directions, bonds.size());
    for (const auto& endpoints : bonds.atom_indices) {
        if (endpoints[0] >= atom_count || endpoints[1] >= atom_count
            || endpoints[0] == endpoints[1]) {
            throw std::invalid_argument(
                "bond transport endpoint is out of range"
            );
        }
    }
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
    if (molecule.has(MoleculeField::STEREO_RECORDS)) {
        detail::validate_stereo_buffers(
            molecule.stereo_records, molecule.atom_count
        );
    } else if (detail::stereo_storage_size(molecule.stereo_records) != 0) {
        throw std::invalid_argument(
            "absent stereo transport field must not carry values"
        );
    }
    if (molecule.has(MoleculeField::ATOM_TRANSPORT)) {
        detail::validate_atom_transport_buffers(
            molecule.atom_transport, molecule.atom_count
        );
    } else if (
        detail::atom_transport_storage_size(molecule.atom_transport) != 0
    ) {
        throw std::invalid_argument(
            "absent atom transport field must not carry values"
        );
    }
    if (molecule.has(MoleculeField::BOND_TRANSPORT)) {
        detail::validate_bond_transport_buffers(
            molecule.bond_transport, molecule.atom_count
        );
    } else if (
        detail::bond_transport_storage_size(molecule.bond_transport) != 0
    ) {
        throw std::invalid_argument(
            "absent bond transport field must not carry values"
        );
    }

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
    OwnedStereoRecordData stereo_records;
    OwnedAtomTransportData atom_transport;
    OwnedBondTransportData bond_transport;

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
            stereo_records.view(),
            atom_transport.view(),
            bond_transport.view(),
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
