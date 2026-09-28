#include "openbabel_adapter.hpp"

#include "registry.hpp"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/generic.h>

#include <memory>
#include <stdexcept>
#include <utility>


namespace hotpot::obwrappers {


OpenBabel::OBMol make_obmol(const MoleculeData& molecule) {
    molecule.validate();
    OpenBabel::OBMol obmol;
    obmol.BeginModify();
    for (std::size_t index = 0; index < molecule.atom_count(); ++index) {
        auto* atom = obmol.NewAtom();
        atom->SetAtomicNum(molecule.atomic_numbers[index]);
        atom->SetFormalCharge(molecule.formal_charges[index]);
        atom->SetPartialCharge(molecule.partial_charges[index]);
        const auto& coordinate = molecule.coordinates[index];
        atom->SetVector(coordinate[0], coordinate[1], coordinate[2]);
    }
    for (std::size_t index = 0; index < molecule.bond_count(); ++index) {
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

    for (std::size_t index = 0; index < molecule.atom_count(); ++index) {
        obmol.GetAtom(static_cast<int>(index + 1))->SetAromatic(
            molecule.atom_aromatic[index] != 0
        );
    }
    for (std::size_t index = 0; index < molecule.bond_count(); ++index) {
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

    if (molecule.unit_cell.has_value()) {
        const auto& values = *molecule.unit_cell;
        auto cell = std::make_unique<OpenBabel::OBUnitCell>();
        cell->SetData(
            values[0], values[1], values[2],
            values[3], values[4], values[5]
        );
        obmol.SetData(cell.release());
    }
    return obmol;
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
