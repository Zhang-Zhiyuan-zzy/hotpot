#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/RWMol.h>
#include <RDGeneral/versions.h>

#include <cmath>
#include <iostream>
#include <string_view>

int main() {
  RDKit::RWMol molecule;
  RDKit::Atom carbon(6);
  const auto carbon_index = molecule.addAtom(&carbon);
  for (unsigned int i = 0; i < 4; ++i) {
    RDKit::Atom hydrogen(1);
    const auto hydrogen_index = molecule.addAtom(&hydrogen);
    molecule.addBond(carbon_index, hydrogen_index, RDKit::Bond::SINGLE);
  }
  RDKit::MolOps::sanitizeMol(molecule);

  auto parameters = RDKit::DGeomHelpers::ETKDGv3;
  parameters.randomSeed = 0x5EED;
  parameters.numThreads = 1;
  const int conformer_id =
      RDKit::DGeomHelpers::EmbedMolecule(molecule, parameters);
  if (conformer_id < 0 || molecule.getNumConformers() != 1) {
    return 1;
  }

  const auto &conformer = molecule.getConformer(conformer_id);
  if (conformer.getNumAtoms() != molecule.getNumAtoms()) {
    return 2;
  }
  for (unsigned int atom_index = 0; atom_index < molecule.getNumAtoms();
       ++atom_index) {
    const auto &position = conformer.getAtomPos(atom_index);
    if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
        !std::isfinite(position.z)) {
      return 3;
    }
  }

  if (std::string_view(RDKit::rdkitVersion) != "2026.09.1") {
    return 4;
  }
  std::cout << "rdkit=" << RDKit::rdkitVersion
            << " boost=" << RDKit::boostVersion
            << " conformers=" << molecule.getNumConformers() << '\n';
  return 0;
}

