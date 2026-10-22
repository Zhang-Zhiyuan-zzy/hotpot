#include "openbabel_adapter.hpp"

#include "registry.hpp"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/generic.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>


namespace hotpot::obwrappers {


namespace {


bool has_required_atom_fields(
    const hotpot::native::MoleculeBufferView& molecule
) noexcept {
    return molecule.has(hotpot::native::MoleculeField::ATOMIC_NUMBERS)
        && molecule.has(hotpot::native::MoleculeField::FORMAL_CHARGES)
        && molecule.has(hotpot::native::MoleculeField::PARTIAL_CHARGES)
        && molecule.has(hotpot::native::MoleculeField::COORDINATES)
        && molecule.has(hotpot::native::MoleculeField::ATOM_AROMATIC);
}


bool has_required_bond_fields(
    const hotpot::native::MoleculeBufferView& molecule
) noexcept {
    return molecule.has(hotpot::native::MoleculeField::BOND_INDICES)
        && molecule.has(hotpot::native::MoleculeField::BOND_ORDERS)
        && molecule.has(hotpot::native::MoleculeField::BOND_KINDS)
        && molecule.has(hotpot::native::MoleculeField::BOND_AROMATIC);
}


}  // namespace


void validate_openbabel_compatibility(
    const hotpot::native::MoleculeBufferView& molecule
) {
    if (molecule.schema_version != 1) {
        throw std::invalid_argument("unsupported molecule buffer schema");
    }
    if (!has_required_atom_fields(molecule)
        || molecule.atomic_numbers.size() != molecule.atom_count
        || molecule.formal_charges.size() != molecule.atom_count
        || molecule.partial_charges.size() != molecule.atom_count
        || molecule.coordinates.size() != molecule.atom_count
        || molecule.atom_aromatic.size() != molecule.atom_count) {
        throw std::invalid_argument(
            "all atom buffers must have the same leading dimension"
        );
    }
    if (!has_required_bond_fields(molecule)
        || molecule.bond_indices.size() != molecule.bond_count
        || molecule.bond_orders.size() != molecule.bond_count
        || molecule.bond_kinds.size() != molecule.bond_count
        || molecule.bond_aromatic.size() != molecule.bond_count) {
        throw std::invalid_argument(
            "all bond buffers must have the same leading dimension"
        );
    }
    if (std::any_of(
            molecule.atomic_numbers.begin(),
            molecule.atomic_numbers.end(),
            [](std::int32_t number) { return number < 1; }
        )) {
        throw std::invalid_argument("atomic numbers must be positive");
    }
    if (std::any_of(
            molecule.partial_charges.begin(),
            molecule.partial_charges.end(),
            [](double charge) { return !std::isfinite(charge); }
        )) {
        throw std::invalid_argument("partial charges must be finite");
    }
    for (const auto& coordinate : molecule.coordinates) {
        if (!std::all_of(
                coordinate.begin(),
                coordinate.end(),
                [](double value) { return std::isfinite(value); }
            )) {
            throw std::invalid_argument("coordinates must be finite");
        }
    }
    for (std::size_t index = 0; index < molecule.bond_count; ++index) {
        const auto endpoints = molecule.bond_indices[index];
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
        if (!std::isfinite(molecule.bond_orders[index])
            || molecule.bond_orders[index] < 0.0) {
            throw std::invalid_argument(
                "bond orders must be finite and nonnegative"
            );
        }
        if (molecule.bond_kinds[index] == BondKind::DATIVE) {
            throw std::invalid_argument(
                "Open Babel cannot represent dative bonds losslessly"
            );
        }
    }
    if (molecule.has(hotpot::native::MoleculeField::UNIT_CELL)) {
        if (molecule.unit_cell.size() != 6
            || !std::all_of(
                molecule.unit_cell.begin(),
                molecule.unit_cell.end(),
                [](double value) { return std::isfinite(value); }
            )) {
            throw std::invalid_argument("unit-cell parameters must be finite");
        }
    } else if (!molecule.unit_cell.empty()) {
        throw std::invalid_argument("unit-cell parameters must be finite");
    }
    hotpot::native::validate_molecule_view(molecule);
}


void validate_openbabel_compatibility(const MoleculeData& molecule) {
    validate_openbabel_compatibility(molecule.view());
}


OpenBabel::OBMol make_obmol(
    const hotpot::native::MoleculeBufferView& molecule
) {
    validate_openbabel_compatibility(molecule);
    OpenBabel::OBMol obmol;
    obmol.BeginModify();
    for (std::size_t index = 0; index < molecule.atom_count; ++index) {
        auto* atom = obmol.NewAtom();
        atom->SetAtomicNum(molecule.atomic_numbers[index]);
        atom->SetFormalCharge(molecule.formal_charges[index]);
        atom->SetPartialCharge(molecule.partial_charges[index]);
        const auto& coordinate = molecule.coordinates[index];
        atom->SetVector(coordinate[0], coordinate[1], coordinate[2]);
    }
    for (std::size_t index = 0; index < molecule.bond_count; ++index) {
        const auto endpoints = molecule.bond_indices[index];
        if (!obmol.AddBond(
                static_cast<unsigned long>(endpoints[0] + 1),
                static_cast<unsigned long>(endpoints[1] + 1),
                static_cast<int>(molecule.bond_orders[index])
            )) {
            throw std::runtime_error("Open Babel rejected a molecular bond");
        }
    }
    obmol.EndModify();

    for (std::size_t index = 0; index < molecule.atom_count; ++index) {
        obmol.GetAtom(static_cast<int>(index + 1))->SetAromatic(
            molecule.atom_aromatic[index] != 0
        );
    }
    for (std::size_t index = 0; index < molecule.bond_count; ++index) {
        const auto endpoints = molecule.bond_indices[index];
        auto* bond = obmol.GetBond(
            static_cast<int>(endpoints[0] + 1),
            static_cast<int>(endpoints[1] + 1)
        );
        bond->SetAromatic(
            molecule.bond_kinds[index] == BondKind::AROMATIC
            || molecule.bond_aromatic[index] != 0
        );
    }
    obmol.SetAromaticPerceived(true);

    if (molecule.has(hotpot::native::MoleculeField::UNIT_CELL)) {
        const auto& values = molecule.unit_cell;
        auto cell = std::make_unique<OpenBabel::OBUnitCell>();
        cell->SetData(
            values[0], values[1], values[2],
            values[3], values[4], values[5]
        );
        obmol.SetData(cell.release());
    }
    return obmol;
}


OpenBabel::OBMol make_obmol(const MoleculeData& molecule) {
    return make_obmol(molecule.view());
}


MoleculeSnapshot snapshot_obmol(
    OpenBabel::OBMol& molecule,
    bool include_coordinates
) {
    std::vector<AtomSnapshot> atoms;
    atoms.reserve(molecule.NumAtoms());
    std::vector<Coordinate> coordinates;
    if (include_coordinates) {
        coordinates.reserve(molecule.NumAtoms());
    }
    for (unsigned int index = 1; index <= molecule.NumAtoms(); ++index) {
        auto* atom = molecule.GetAtom(static_cast<int>(index));
        atoms.push_back(AtomSnapshot{
            static_cast<int>(atom->GetAtomicNum()),
            atom->GetFormalCharge(),
            static_cast<int>(atom->GetHyb()),
            atom->IsMetal(),
        });
        if (include_coordinates) {
            coordinates.push_back(
                Coordinate{atom->GetX(), atom->GetY(), atom->GetZ()}
            );
        }
    }

    std::vector<BondSnapshot> bonds;
    bonds.reserve(molecule.NumBonds());
    for (unsigned int index = 0; index < molecule.NumBonds(); ++index) {
        const auto* bond = molecule.GetBond(static_cast<int>(index));
        bonds.push_back(BondSnapshot{
            static_cast<std::size_t>(bond->GetBeginAtomIdx() - 1),
            static_cast<std::size_t>(bond->GetEndAtomIdx() - 1),
            static_cast<int>(bond->GetBondOrder()),
            bond->IsAromatic(),
        });
    }
    return make_snapshot(
        std::move(atoms),
        std::move(bonds),
        std::move(coordinates),
        include_coordinates
    );
}


std::vector<Coordinate> extract_coordinates(
    const OpenBabel::OBMol& molecule
) {
    std::vector<Coordinate> coordinates;
    coordinates.reserve(molecule.NumAtoms());
    for (unsigned int index = 1; index <= molecule.NumAtoms(); ++index) {
        const auto* atom = molecule.GetAtom(static_cast<int>(index));
        coordinates.push_back(
            Coordinate{atom->GetX(), atom->GetY(), atom->GetZ()}
        );
    }
    return coordinates;
}


void set_coordinates(
    OpenBabel::OBMol& molecule,
    const std::vector<Coordinate>& coordinates
) {
    if (coordinates.size() != molecule.NumAtoms()) {
        throw std::invalid_argument(
            "coordinate count must equal the atom count"
        );
    }
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        const auto& coordinate = coordinates[index];
        molecule.GetAtom(static_cast<int>(index + 1))->SetVector(
            coordinate[0], coordinate[1], coordinate[2]
        );
    }
}


}  // namespace hotpot::obwrappers
