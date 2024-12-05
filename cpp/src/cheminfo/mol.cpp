#include <string>
#include <vector>

# include "../../include/cheminfo.h"

using namespace std;


// Implementation of Molecule.
Molecule::Molecule() {};

// Construct from OpenBabel OBMol
Molecule::Molecule(OpenBabel::OBMol obMol)
{
    // this -> charges = obMol.
}

Molecule::readFromOBMol(OpenBabel::OBMol obMol) {};

int main() {
    return 0;
}
