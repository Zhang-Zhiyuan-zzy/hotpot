#pragma once

#include "molecule_data.hpp"

#include <openbabel/mol.h>

#include <vector>


namespace hotpot::obwrappers {


OpenBabel::OBMol make_obmol(const MoleculeData& molecule);

MoleculeSnapshot snapshot_obmol(
    OpenBabel::OBMol& molecule,
    bool include_coordinates
);

std::vector<Coordinate> extract_coordinates(
    const OpenBabel::OBMol& molecule
);

void set_coordinates(
    OpenBabel::OBMol& molecule,
    const std::vector<Coordinate>& coordinates
);


}  // namespace hotpot::obwrappers
