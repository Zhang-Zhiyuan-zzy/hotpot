#include "contracts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>


namespace hotpot::forcefields {


namespace {


constexpr std::uint64_t LIGAND_FIELDS =
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


bool known_bond_kind(BondKind kind) {
    const auto value = static_cast<std::uint8_t>(kind);
    return value >= static_cast<std::uint8_t>(BondKind::SINGLE)
        && value <= static_cast<std::uint8_t>(BondKind::UNKNOWN);
}


std::pair<std::int32_t, std::int32_t> undirected_key(
    const BondIndex& endpoints
) {
    return std::minmax(endpoints[0], endpoints[1]);
}


}  // namespace


std::size_t ComplexSessionInput::atom_count() const noexcept {
    return atomic_numbers.size();
}


std::size_t ComplexSessionInput::ligand_bond_count() const noexcept {
    return ligand_bond_indices.size();
}


std::size_t ComplexSessionInput::intended_coordination_bond_count()
    const noexcept {
    return intended_coordination_bonds.size();
}


std::size_t PerturbationOffsetBatch::frame_count() const noexcept {
    return frames.size();
}


void PerturbationOffsetBatch::validate() const {
    for (const auto& frame : frames) {
        if (frame.size() != atom_count) {
            throw std::invalid_argument(
                "each perturbation frame must match atom_count"
            );
        }
        for (const auto& offset : frame) {
            if (!std::all_of(
                    offset.begin(),
                    offset.end(),
                    [](double value) { return std::isfinite(value); }
                )) {
                throw std::invalid_argument(
                    "perturbation offsets must be finite"
                );
            }
        }
    }
}


hotpot::native::MoleculeBufferView ligand_molecule_view(
    const ComplexSessionInput& input
) noexcept {
    return hotpot::native::MoleculeBufferView{
        input.schema_version,
        input.atom_count(),
        input.ligand_bond_count(),
        LIGAND_FIELDS
            | (input.unit_cell.has_value()
                ? hotpot::native::molecule_field_mask(
                    hotpot::native::MoleculeField::UNIT_CELL
                )
                : 0),
        hotpot::native::ArrayView<std::int32_t>(input.atomic_numbers),
        hotpot::native::ArrayView<std::int32_t>(input.formal_charges),
        hotpot::native::ArrayView<double>(input.partial_charges),
        hotpot::native::ArrayView<hotpot::native::Coordinate>(
            input.coordinates
        ),
        hotpot::native::ArrayView<std::uint8_t>(input.atom_aromatic),
        hotpot::native::ArrayView<hotpot::native::BondIndex>(
            input.ligand_bond_indices
        ),
        hotpot::native::ArrayView<double>(input.ligand_bond_orders),
        hotpot::native::ArrayView<hotpot::native::BondKind>(
            input.ligand_bond_kinds
        ),
        hotpot::native::ArrayView<std::uint8_t>(
            input.ligand_bond_aromatic
        ),
        input.unit_cell.has_value()
            ? hotpot::native::ArrayView<double>(
                input.unit_cell->data(), input.unit_cell->size()
            )
            : hotpot::native::ArrayView<double>{},
    };
}


void ComplexSessionInput::validate() const {
    hotpot::native::validate_molecule_view(ligand_molecule_view(*this));

    if (metal_indices.empty()) {
        throw std::invalid_argument(
            "a complex session requires at least one metal index"
        );
    }
    std::set<std::int32_t> declared_metals;
    for (const auto index : metal_indices) {
        if (index < 0 || static_cast<std::size_t>(index) >= atom_count()) {
            throw std::invalid_argument("metal atom index is out of range");
        }
        if (!declared_metals.insert(index).second) {
            throw std::invalid_argument("metal atom indices must be unique");
        }
    }

    const auto intended_count = intended_coordination_bond_count();
    if (intended_coordination_orders.size() != intended_count
        || intended_coordination_kinds.size() != intended_count) {
        throw std::invalid_argument(
            "all intended coordination-bond buffers must have the same "
            "leading dimension"
        );
    }

    std::set<std::pair<std::int32_t, std::int32_t>> occupied_bonds;
    for (const auto& endpoints : ligand_bond_indices) {
        occupied_bonds.insert(undirected_key(endpoints));
    }
    for (std::size_t index = 0; index < intended_count; ++index) {
        const auto endpoints = intended_coordination_bonds[index];
        if (endpoints[0] < 0 || endpoints[1] < 0
            || static_cast<std::size_t>(endpoints[0]) >= atom_count()
            || static_cast<std::size_t>(endpoints[1]) >= atom_count()) {
            throw std::invalid_argument(
                "intended coordination-bond atom index is out of range"
            );
        }
        if (endpoints[0] == endpoints[1]) {
            throw std::invalid_argument(
                "self coordination bonds are not supported"
            );
        }
        if (declared_metals.count(endpoints[0]) == 0) {
            throw std::invalid_argument(
                "the first intended coordination-bond endpoint must be a "
                "declared metal"
            );
        }
        const auto order = intended_coordination_orders[index];
        if (!std::isfinite(order) || order < 0.0) {
            throw std::invalid_argument(
                "intended coordination-bond orders must be finite and "
                "nonnegative"
            );
        }
        if (!known_bond_kind(intended_coordination_kinds[index])) {
            throw std::invalid_argument(
                "intended_coordination_kinds contains an unknown value"
            );
        }
        if (!occupied_bonds.insert(undirected_key(endpoints)).second) {
            throw std::invalid_argument(
                "ligand and intended coordination bonds must be distinct"
            );
        }
    }
}


}  // namespace hotpot::forcefields
