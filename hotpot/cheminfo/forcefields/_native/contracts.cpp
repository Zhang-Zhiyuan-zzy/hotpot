#include "contracts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>


namespace hotpot::forcefields {


namespace {


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


hotpot::obwrappers::MoleculeData ligand_molecule_data(
    const ComplexSessionInput& input
) {
    return hotpot::obwrappers::MoleculeData{
        input.schema_version,
        input.atomic_numbers,
        input.formal_charges,
        input.partial_charges,
        input.coordinates,
        input.atom_aromatic,
        input.ligand_bond_indices,
        input.ligand_bond_orders,
        input.ligand_bond_kinds,
        input.ligand_bond_aromatic,
        input.unit_cell,
    };
}


void ComplexSessionInput::validate() const {
    const auto ligand = ligand_molecule_data(*this);
    ligand.validate();

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
