#include "molecule_data.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace hotpot::obwrappers {


std::size_t MoleculeData::atom_count() const noexcept {
    return atomic_numbers.size();
}


std::size_t MoleculeData::bond_count() const noexcept {
    return bond_indices.size();
}


void MoleculeData::validate() const {
    if (schema_version != 1) {
        throw std::invalid_argument("unsupported molecule buffer schema");
    }
    const auto atoms = atom_count();
    if (formal_charges.size() != atoms
        || partial_charges.size() != atoms
        || coordinates.size() != atoms
        || atom_aromatic.size() != atoms) {
        throw std::invalid_argument(
            "all atom buffers must have the same leading dimension"
        );
    }
    if (bond_orders.size() != bond_count()
        || bond_kinds.size() != bond_count()
        || bond_aromatic.size() != bond_count()) {
        throw std::invalid_argument(
            "all bond buffers must have the same leading dimension"
        );
    }
    if (std::any_of(
            atomic_numbers.begin(),
            atomic_numbers.end(),
            [](std::int32_t number) { return number < 1; }
        )) {
        throw std::invalid_argument("atomic numbers must be positive");
    }
    if (std::any_of(
            partial_charges.begin(),
            partial_charges.end(),
            [](double charge) { return !std::isfinite(charge); }
        )) {
        throw std::invalid_argument("partial charges must be finite");
    }
    for (const auto& coordinate : coordinates) {
        if (!std::all_of(
                coordinate.begin(),
                coordinate.end(),
                [](double value) { return std::isfinite(value); }
            )) {
            throw std::invalid_argument("coordinates must be finite");
        }
    }
    for (std::size_t index = 0; index < bond_count(); ++index) {
        const auto endpoints = bond_indices[index];
        if (endpoints[0] < 0 || endpoints[1] < 0
            || static_cast<std::size_t>(endpoints[0]) >= atoms
            || static_cast<std::size_t>(endpoints[1]) >= atoms) {
            throw std::invalid_argument("bond atom index is out of range");
        }
        if (endpoints[0] == endpoints[1]) {
            throw std::invalid_argument("self-bonds are not supported");
        }
        if (!std::isfinite(bond_orders[index]) || bond_orders[index] < 0.0) {
            throw std::invalid_argument(
                "bond orders must be finite and nonnegative"
            );
        }
        if (bond_kinds[index] == BondKind::DATIVE) {
            throw std::invalid_argument(
                "Open Babel cannot represent dative bonds losslessly"
            );
        }
    }
    if (unit_cell.has_value()
        && !std::all_of(
            unit_cell->begin(),
            unit_cell->end(),
            [](double value) { return std::isfinite(value); }
        )) {
        throw std::invalid_argument("unit-cell parameters must be finite");
    }
}


}  // namespace hotpot::obwrappers
